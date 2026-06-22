"""
THE steering gate experiment (runbook section 2): chroma guidance on SA3.

Generates a baseline and a steered version with the same seed, steering the
rectified-flow sampling of a -base checkpoint toward a chroma target through
the refit linear readout (sa3_refit_readout.py output), then measures
chroma alignment of both outputs with the same extractor.

Method: selective TFG-style MEAN guidance (LatCH paper, arXiv 2603.04366)
applied to the predicted-clean latent z0|t = x - t*v inside the Euler loop:
    repeat N_iter: z0 <- z0 - mu_t * grad_z0 delta(head(z0 + gamma-noise), target)
    v' = (x - z0) / t
Guidance only on the first ~20% of sampling steps (high noise), per the paper.

The gradient of the per-frame cosine objective through the AFFINE head is
CLOSED FORM (pure matmuls) — required because model.generate() runs under
@torch.inference_mode(), where autograd is unavailable; also faster.

Implementation detail: we monkey-patch
stable_audio_3.inference.sampling.sample_discrete_euler with a guided clone
for the duration of the steered run — model.generate()'s conditioning/varlen/
decode plumbing is reused untouched ("euler" is the default sampler for the
"rectified_flow" objective of -base checkpoints; we also pass it explicitly).

NOTE on gains: the cosine objective is scale-invariant, so the per-band gains
act as LOSS WEIGHTS (lambda_b), not via target amplitude. Targets are
unit-normalized per band; zero-target bands (e.g. bass_root=None) are masked.

Production box only. Examples:
    python sa3_steer_chroma.py --heads chroma_heads.npz --test only-c
    python sa3_steer_chroma.py --heads chroma_heads.npz --test goa-e --mu 0.05
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "harmonic"))
import same_chroma as sc  # noqa: E402

logger = logging.getLogger(__name__)

E_PHRYGIAN = [4, 5, 7, 9, 11, 0, 2]   # E F G A B C D


# ── Steering target (constant-over-time profile, unit per band) ──────────────

def build_profiles(test: str, melody_weights=None, bass_root=None,
                   bass_profile="root5", width=0.25):
    """Returns (profiles (3,128) float32 unit-or-zero rows, description)."""
    if test == "only-c":
        mel = np.zeros(12); mel[0] = 1.0
        bass_root, desc = 0, "all pitch content -> C (bass locked to C)"
    elif test == "goa-e":
        mel = np.zeros(12); mel[E_PHRYGIAN] = 1.0
        bass_root, desc = 4, "bass locked to E ('root5'), melody E phrygian"
    elif test == "custom":
        mel = np.asarray(melody_weights, dtype=np.float64)
        desc = f"custom weights {melody_weights}, bass_root={bass_root}"
    else:
        raise ValueError(test)
    t = sc.make_steering_target(1, mel, bass_root=bass_root,
                                bass_profile=bass_profile,
                                bass_gain=1.0, mid_gain=1.0, air_gain=1.0,
                                width_semitones=width)[:, :, 0]   # (3,128)
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    profiles = np.where(norms > 1e-6, t / np.maximum(norms, 1e-12), 0.0)
    return profiles.astype(np.float32), desc


# ── Closed-form gradient of sum_b lambda_b * mean_frames(1 - cos) ─────────────

def cosine_grad(z0, W, b, tau, band_w, eps=1e-8):
    """
    z0: (B, D, T) float32; W: (3, 128, D); b: (3, 128);
    tau: (3, 128) unit-or-zero rows; band_w: (3,) loss weights.
    Returns grad (B, D, T) and per-band mean cosine (3,) for logging.
    d(1-cos)/de = -(tau_hat - cos * e_hat) / ||e|| ;  dz = W^T de.
    """
    import torch
    B, D, T = z0.shape
    grad = torch.zeros_like(z0)
    cosines = []
    for k in range(3):
        if band_w[k] == 0 or float(tau[k].abs().sum()) < eps:
            cosines.append(float("nan"))
            continue
        e = torch.einsum("cd,bdt->bct", W[k], z0) + b[k][None, :, None]  # (B,128,T)
        e_norm = e.norm(dim=1, keepdim=True).clamp_min(eps)              # (B,1,T)
        e_hat = e / e_norm
        tau_k = tau[k][None, :, None]                                    # (1,128,1)
        cos = (e_hat * tau_k).sum(dim=1, keepdim=True)                   # (B,1,T)
        d_e = -(tau_k - cos * e_hat) / e_norm                            # (B,128,T)
        grad += band_w[k] * torch.einsum("cd,bct->bdt", W[k], d_e)
        cosines.append(float(cos.mean()))
    return grad, cosines


def make_guided_euler(orig_tqdm, W, b, tau, band_w, mu, gamma, n_iter,
                      frac, normalize_grad, mu_schedule, log_every):
    """Clone of sample_discrete_euler with selective mean guidance inserted."""
    import torch

    def guided(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
        t = sigmas.to(x.device)
        per_element = t.dim() == 2
        num_steps = t.shape[-1] - 1
        active_steps = max(1, int(round(frac * num_steps)))
        logger.info(f"guided euler: {num_steps} steps, guidance on first "
                    f"{active_steps} steps, mu={mu} ({mu_schedule}) "
                    f"gamma={gamma} N_iter={n_iter}")
        for i in orig_tqdm(range(num_steps), disable=disable_tqdm):
            if per_element:
                t_curr_tensor = t[:, i].to(x.dtype)
                dt_broadcast = (t[:, i + 1] - t[:, i]).to(x.dtype).view(-1, 1, 1)
            else:
                t_curr = t[i]
                t_curr_tensor = t_curr * torch.ones(
                    (x.shape[0],), dtype=x.dtype, device=x.device)
                dt_broadcast = t[i + 1] - t_curr

            v = model(x, t_curr_tensor, **extra_args)

            if i < active_steps:
                t_b = t_curr_tensor.view(-1, 1, 1).float().clamp_min(1e-4)
                z0 = (x.float() - t_b * v.float())
                s_t = (1.0 - t_b).clamp_min(1e-3)   # RF alpha_t, for gamma noise
                # Step size: 'const' = mu per iteration (default; with the
                # RMS-normalized gradient this gives a predictable total push
                # of ~mu * n_iter * active_steps RMS units). 'alpha' = TFG's
                # mu * s(t) — nearly inert in the early window where s_t is
                # 0.04-0.2; kept as the paper-faithful option.
                step = mu * s_t if mu_schedule == "alpha" else mu
                for _ in range(n_iter):
                    z_eval = z0
                    if gamma > 0:
                        z_eval = z0 + (gamma * s_t) * torch.randn_like(z0)
                    g, cosines = cosine_grad(z_eval, W, b, tau, band_w)
                    if normalize_grad:
                        rms = g.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
                        g = g / rms.clamp_min(1e-12)
                    z0 = z0 - step * g
                if log_every and i % log_every == 0:
                    logger.info(f"  step {i}: band cosines "
                                + " ".join(f"{c:.3f}" if c == c else "-"
                                           for c in cosines))
                v = (x.float() - z0) / t_b
                v = v.to(x.dtype)

            if callback is not None:
                denoised = x - t_curr_tensor[:, None, None] * v
                callback({"x": x, "t": t_curr_tensor, "sigma": t_curr_tensor,
                          "i": i, "denoised": denoised})
            x = x + dt_broadcast * v
        return x

    return guided


# ── Alignment metric (measured with our own extractor on the OUTPUT audio) ───

def measure_alignment(audio_np, tau):
    """Per-band mean frame cosine between the audio's SAME chroma and the
    target profile. audio_np: (C, T) float32 @44.1k; tau: (3,128) numpy."""
    chroma = sc.compute_same_chroma(audio_np.T, sc.SAME_SR,
                                    align_to_latent=False)   # (3,128,M)
    out = []
    for k in range(3):
        if np.abs(tau[k]).sum() < 1e-6:
            out.append(None)
            continue
        C = chroma[k]                                         # (128, M)
        n = np.linalg.norm(C, axis=0) + 1e-9
        out.append(float(np.mean((tau[k] @ C) / n)))
    folded = sc.fold_to_12(chroma[1].mean(axis=-1))
    top = np.argsort(folded)[::-1][:5]
    return out, [(sc.PITCH_CLASSES[i], round(float(folded[i]), 2)) for i in top]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--heads", type=str, required=True,
                        help="chroma_heads.npz from sa3_refit_readout.py")
    parser.add_argument("--model", type=str, default="medium-base",
                        help="MUST be a -base checkpoint (live CFG, 50-step regime)")
    parser.add_argument("--test", choices=["only-c", "goa-e", "custom"],
                        default="only-c")
    parser.add_argument("--melody-weights", type=float, nargs=12, default=None)
    parser.add_argument("--bass-root", type=int, default=None)
    parser.add_argument("--prompt", type=str,
                        default="melodic goa trance, hypnotic arpeggiated lead, "
                                "driving rolling bassline")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=1234)
    # LatCH-paper guidance defaults (arXiv 2603.04366 section 3.3)
    parser.add_argument("--mu", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--n-iter", type=int, default=4)
    parser.add_argument("--frac", type=float, default=0.2,
                        help="fraction of (early) sampling steps with guidance")
    parser.add_argument("--mu-schedule", choices=["const", "alpha"],
                        default="const",
                        help="step-size schedule: const (predictable with "
                             "normalized grads) or alpha (TFG-faithful mu*s_t, "
                             "much weaker in the early window)")
    parser.add_argument("--band-weights", type=float, nargs=3,
                        default=[1.0, 1.0, 0.3],
                        help="loss weights (bass, mid, air) - lock vs prefer")
    parser.add_argument("--raw-grad", action="store_true",
                        help="disable per-step RMS gradient normalization")
    parser.add_argument("--log-every", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="steering_results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if "base" not in args.model:
        logger.warning(f"model '{args.model}' is not a -base checkpoint: "
                       f"cfg/guidance may be inert (post-trained models bake "
                       f"guidance in) and few-step sampling leaves no room "
                       f"for selective TFG.")

    import torch
    import torchaudio
    from stable_audio_3 import StableAudioModel
    import stable_audio_3.inference.sampling as smp

    device = "cuda" if torch.cuda.is_available() else "cpu"

    heads = np.load(args.heads, allow_pickle=True)
    W = torch.from_numpy(heads["W"]).float().to(device)      # (3,128,D)
    b = torch.from_numpy(heads["b"]).float().to(device)      # (3,128)
    logger.info(f"heads: model={heads['model']}, latent_dim={heads['latent_dim']}, "
                f"metrics={heads['metrics']}")

    profiles, desc = build_profiles(args.test, args.melody_weights, args.bass_root)
    tau = torch.from_numpy(profiles).to(device)
    band_w = list(args.band_weights)
    if float(np.abs(profiles[0]).sum()) < 1e-6:
        band_w[0] = 0.0                                       # bass unsteered
    logger.info(f"target: {desc}; band loss weights {band_w}")

    model = StableAudioModel.from_pretrained(args.model, device=device)

    out_dir = Path(args.out_dir) / f"{args.test}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_kwargs = dict(prompt=args.prompt, duration=args.duration,
                      steps=args.steps, cfg_scale=args.cfg, seed=args.seed,
                      sampler_type="euler")

    logger.info("=== baseline (no steering) ===")
    audio_base = model.generate(**gen_kwargs)

    logger.info("=== steered ===")
    orig_euler = smp.sample_discrete_euler
    smp.sample_discrete_euler = make_guided_euler(
        smp.tqdm, W, b, tau, band_w, args.mu, args.gamma, args.n_iter,
        args.frac, not args.raw_grad, args.mu_schedule, args.log_every)
    try:
        audio_steer = model.generate(**gen_kwargs)
    finally:
        smp.sample_discrete_euler = orig_euler

    report = {"args": {k: v for k, v in vars(args).items()}, "target": desc,
              "bands": ["bass_oct1", "mid_oct5", "treble_oct9"]}
    for name, audio in (("baseline", audio_base), ("steered", audio_steer)):
        wav = audio[0].float().cpu()                          # (C, T)
        torchaudio.save(str(out_dir / f"{name}.wav"), wav, sc.SAME_SR)
        align, top = measure_alignment(wav.numpy(), profiles)
        report[name] = {"alignment": align, "top_pitch_classes": top}
        logger.info(f"{name}: alignment {align}  top PCs {top}")

    deltas = [
        (round(s - bse, 4) if (s is not None and bse is not None) else None)
        for s, bse in zip(report["steered"]["alignment"],
                          report["baseline"]["alignment"])]
    report["alignment_delta"] = deltas
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    logger.info(f"alignment delta (steered - baseline) per band: {deltas}")
    logger.info(f"results -> {out_dir}  — now LISTEN to baseline.wav vs "
                f"steered.wav: do the pitches actually move?")


if __name__ == "__main__":
    main()
