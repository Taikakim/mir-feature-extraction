"""
Refit the SAME chroma readout heads (the "lost" linear regressors).

Encodes a dataset with the released SAME autoencoder, computes the exact
training-time chroma targets with same_chroma, and solves the per-band affine
map latent -> 128 chroma bins by ridge least squares (closed form, no GPU
training). The result is a functional replica of the original 1x1-conv
regression heads, saved as an .npz for the steering experiment
(sa3_steer_chroma.py).

Also provides --sanity: the runbook section-0 environment checks.

Production box only (needs torch + stable_audio_3 + checkpoints).

Usage:
    python sa3_refit_readout.py --sanity
    python sa3_refit_readout.py --audio_dir ./crops --model same-l --out chroma_heads.npz
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "harmonic"))
import same_chroma as sc  # noqa: E402

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}


def _load_torch_stack():
    try:
        import torch
        import torchaudio
        from stable_audio_3 import AutoencoderModel
        return torch, torchaudio, AutoencoderModel
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency ({e}). This script runs on the production box "
            f"with torch + stable_audio_3 installed.")


def load_audio_441(torchaudio, path: Path, max_seconds: float):
    """Load audio, resample to 44.1 kHz OURSELVES (so the chroma targets and
    the encoder see literally identical samples), crop to max_seconds."""
    wav, sr = torchaudio.load(str(path))          # (C, T)
    if sr != sc.SAME_SR:
        wav = torchaudio.functional.resample(wav, sr, sc.SAME_SR)
    max_samples = int(max_seconds * sc.SAME_SR)
    if wav.shape[-1] > max_samples:
        wav = wav[..., :max_samples]
    if wav.shape[0] == 1:                         # mono -> fake stereo
        wav = wav.repeat(2, 1)
    return wav                                     # (2, T) float32 @ 44100


def chroma_target_for(torch, wav, device):
    """(3, 128, M) float32 chroma target at native STFT frames (M ~ T/4096),
    computed exactly as SAME training did."""
    with torch.no_grad():
        c = sc.compute_same_chroma_torch(
            wav.unsqueeze(0), sr=sc.SAME_SR, align_to_latent=False,
            device=device)                         # (1, 3, 128, M)
    return c[0]


def align_target(torch, target, n_latent):
    """F.interpolate(mode='linear') to the latent frame count — mirrors
    training (autoencoders.py:858-859)."""
    if target.shape[-1] == n_latent:
        return target
    return torch.nn.functional.interpolate(
        target.reshape(1, 3 * sc.SAME_N_CHROMA, -1), size=n_latent,
        mode="linear").reshape(3, sc.SAME_N_CHROMA, n_latent)


# ── Sanity checks (runbook section 0) ─────────────────────────────────────────

def run_sanity(model_name: str, device: str | None):
    torch, torchaudio, AutoencoderModel = _load_torch_stack()
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # 1. numpy selftest
    check("same_chroma --selftest", sc.selftest(verbose=False))

    # 2. torch path vs numpy path
    rng = np.random.default_rng(0)
    t = np.arange(5 * sc.SAME_SR) / sc.SAME_SR
    sig = (0.4 * np.sin(2 * np.pi * 220.0 * t)
           + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
    c_np = sc.compute_same_chroma(sig, sc.SAME_SR, align_to_latent=False)
    c_th = sc.compute_same_chroma_torch(
        torch.from_numpy(sig), sr=sc.SAME_SR, align_to_latent=False
    )[0].cpu().numpy()
    rel = float(np.max(np.abs(c_np - c_th)) / max(np.max(np.abs(c_np)), 1e-12))
    check(f"torch path matches numpy path (rel err {rel:.2e})", rel < 1e-4)
    if torch.cuda.is_available():
        c_gpu = sc.compute_same_chroma_torch(
            torch.from_numpy(sig), sr=sc.SAME_SR, align_to_latent=False,
            device="cuda")[0].cpu().numpy()
        rel_g = float(np.max(np.abs(c_np - c_gpu)) / max(np.max(np.abs(c_np)), 1e-12))
        check(f"GPU path matches numpy path (rel err {rel_g:.2e})", rel_g < 1e-3)

    # 3. real encoder frame count vs n_latent_frames (config says 4096 ratio)
    ae = AutoencoderModel.from_pretrained(model_name, device=device)
    wav = torch.from_numpy(
        0.1 * rng.standard_normal((2, 10 * sc.SAME_SR)).astype(np.float32))
    with torch.no_grad():
        lat = ae.encode(wav, sc.SAME_SR)
    expected = sc.n_latent_frames(10 * sc.SAME_SR)   # 108
    got = lat.shape[-1]
    check(f"10 s encode -> {got} latent frames (n_latent_frames says "
          f"{expected}; config 4096 ratio says 108)", abs(got - expected) <= 1)
    print(f"  [INFO] latent shape {tuple(lat.shape)}, "
          f"std {float(lat.float().std()):.3f} (softnorm ~1 expected)")
    print(f"\nsanity {'PASSED' if ok else 'FAILED'}")
    return ok


# ── Refit ─────────────────────────────────────────────────────────────────────

def refit(args):
    torch, torchaudio, AutoencoderModel = _load_torch_stack()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ae = AutoencoderModel.from_pretrained(args.model, device=device)

    files = sorted(p for p in Path(args.audio_dir).rglob("*")
                   if p.suffix.lower() in AUDIO_EXTS)
    if args.max_files:
        files = files[:args.max_files]
    if not files:
        raise SystemExit(f"no audio files under {args.audio_dir}")
    logger.info(f"{len(files)} files, model={args.model}, device={device}")

    D = None                                       # latent dim, discovered
    A = None                                       # (3 unused — shared design matrix)
    Bmat = [None, None, None]                      # per-band (D+1, 128)
    holdout = []                                   # (latents (L,D), targets (3,128,L))
    n_frames_total = 0

    for i, path in enumerate(files):
        try:
            wav = load_audio_441(torchaudio, path, args.max_seconds)
        except Exception as e:
            logger.warning(f"skip {path.name}: {e}")
            continue
        with torch.no_grad():
            lat = ae.encode(wav, sc.SAME_SR)       # (1, D, L)
        target = chroma_target_for(torch, wav.to(device), device)
        target = align_target(torch, target, lat.shape[-1])   # (3, 128, L)

        Z = lat[0].float().cpu().numpy().T.astype(np.float64)  # (L, D)
        Y = target.float().cpu().numpy()                        # (3, 128, L)
        if D is None:
            D = Z.shape[1]
            A = np.zeros((D + 1, D + 1))
            Bmat = [np.zeros((D + 1, sc.SAME_N_CHROMA)) for _ in range(3)]

        if args.holdout_every and i % args.holdout_every == 0:
            if len(holdout) < args.max_holdout:
                holdout.append((Z, Y))
            continue

        X = np.concatenate([Z, np.ones((len(Z), 1))], axis=1)   # (L, D+1)
        A += X.T @ X
        for b in range(3):
            Bmat[b] += X.T @ Y[b].T
        n_frames_total += len(Z)
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(files)} files, {n_frames_total} frames")

    if n_frames_total == 0:
        raise SystemExit("no training frames accumulated")
    logger.info(f"solving: {n_frames_total} frames, latent dim {D}, "
                f"{len(holdout)} holdout files")

    ridge = args.ridge * np.trace(A) / (D + 1)
    A_reg = A + ridge * np.eye(D + 1)
    W = np.zeros((3, sc.SAME_N_CHROMA, D), dtype=np.float32)
    bias = np.zeros((3, sc.SAME_N_CHROMA), dtype=np.float32)
    for b in range(3):
        sol = np.linalg.solve(A_reg, Bmat[b])      # (D+1, 128)
        W[b] = sol[:D].T.astype(np.float32)
        bias[b] = sol[D].astype(np.float32)

    # Held-out metrics
    metrics = {}
    if holdout:
        names = ["bass_oct1", "mid_oct5", "treble_oct9"]
        for b in range(3):
            preds, trues = [], []
            for Z, Y in holdout:
                preds.append(Z @ W[b].T.astype(np.float64) + bias[b])
                trues.append(Y[b].T)
            P = np.concatenate(preds)              # (Lh, 128)
            T = np.concatenate(trues)
            ss_res = float(((P - T) ** 2).sum())
            ss_tot = float(((T - T.mean(axis=0)) ** 2).sum())
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            cosine = float(np.mean(
                (P * T).sum(axis=1)
                / (np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1) + 1e-9)))
            metrics[names[b]] = {"r2": round(r2, 4), "frame_cosine": round(cosine, 4)}
            gate = "OK" if r2 > 0.3 else ("rough" if r2 > 0.1 else "NEAR-ZERO "
                   "- check frame alignment (off-by-one) before doubting the method")
            logger.info(f"  {names[b]}: held-out R2={r2:.3f} "
                        f"frame-cosine={cosine:.3f}  [{gate}]")

    np.savez(args.out, W=W, b=bias,
             model=args.model, latent_dim=D, n_frames=n_frames_total,
             metrics=json.dumps(metrics), same_chroma_version="1",
             note="W: (3,128,D) per-band affine readout latent->chroma; "
                  "bands = octaves (1, 5, 9); apply as W[b] @ z + b[b]")
    print(f"saved heads -> {args.out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--audio_dir", type=str)
    parser.add_argument("--model", type=str, default="same-l",
                        help="same-l (medium/large) or same-s (small)")
    parser.add_argument("--out", type=str, default="chroma_heads.npz")
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--holdout-every", type=int, default=8,
                        help="every k-th file held out for eval (0 = none)")
    parser.add_argument("--max-holdout", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=1e-3,
                        help="ridge factor (scaled by trace(A)/(D+1))")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sanity", action="store_true",
                        help="run runbook section-0 environment checks only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.sanity:
        sys.exit(0 if run_sanity(args.model, args.device) else 1)
    if not args.audio_dir:
        parser.error("provide --audio_dir or --sanity")
    refit(args)
