#!/usr/bin/env python3
"""
recurrence_meter.py -- patch-level recurrence/novelty meter (the #35 breathing-noise input).

Re-landed from the validated v3 scratchpad meter (journal 2026-07-07/08; the tmpfs
original is gone -- this is the same algorithm, now a real module). Detects the a2a
LOOP ATTRACTOR: generated regions that repeat one short phrase for minutes. Findings
the design encodes (don't re-derive):

  * NEGATIVE: naive FRAME-cosine recurrence saturates (0.99 for source AND every nl --
    tonal music trivially recurs frame-wise). Patch-level (~4 s) is the working unit.
  * v3 (Kim's feature correction): chroma is tonality-biased, rhythm saturates -- the
    loop is the whole SPECTRAL IMAGE. Use per-band-WHITENED log-mel patches: z-score
    each band so the steady kick flattens and only varying content drives similarity
    (kaikkialla novelty separation ~8x vs chroma; the nl.70 overshoot shows correctly).
  * Drive controllers by the NOVELTY floor, not raw recurrence, and calibrate PER
    TRACK against the source (0.968 loop_strength = looping for one track, ~source
    for another -- no global threshold works).
  * For LATENT input (C's controller): same rule -- whiten each channel over the
    window before cosine, else constant-energy channels bias toward "always looping".

Interface (C's #35 controller codes against this):
    from recurrence_meter import novelty_curve, calibrate_source
    res = novelty_curve(x, fps=10.767)          # x: (T, C) feature/latent sequence
    res = novelty_curve("/path/audio.flac")     # or audio path / (n,) mono samples + sr
    # res: {"time": (P,), "recurrence": (P,), "novelty": (P,), "fps": float,
    #        "patch_sec": float, ...}
    cal = calibrate_source(src)                  # -> {"novelty_floor", "r_max", ...}

numpy-only core (imports in any venv); librosa needed only for the audio path.
CLI:  python recurrence_meter.py <audio-or-npy> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Validated operating point (journal 2026-07-07): 4 s patches, 1 s stride,
# lookback 8-40 s strictly preceding the current patch.
PATCH_SEC = 4.0
STRIDE_SEC = 1.0
LOOKBACK_MIN_SEC = 8.0
LOOKBACK_MAX_SEC = 40.0

_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")


def _to_features(x, sr=None, fps=None, n_mels=64):
    """Normalize input to (T, C) float feature sequence + fps.

    Accepts: (T, C)/(C, T) ndarray (feature/latent sequence, fps required),
    (n,) mono samples (sr required), or an audio path (librosa log-mel)."""
    if isinstance(x, (str, Path)):
        import librosa
        y, sr = librosa.load(str(x), sr=22050, mono=True)
        return _to_features(y, sr=sr, n_mels=n_mels)
    x = np.asarray(x)
    if x.ndim == 1:  # mono samples -> log-mel at ~10.77 fps (hop 2048 @ 22.05 k)
        if not sr:
            raise ValueError("mono samples need sr=")
        import librosa
        hop = 2048
        mel = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=sr,
                                             n_mels=n_mels, hop_length=hop)
        feats = librosa.power_to_db(mel, ref=np.max).T  # (T, n_mels)
        return feats.astype(np.float64), sr / hop
    if x.ndim == 3 and x.shape[0] == 1:  # (1, C, T) SAME-latent convention
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"expected (T,C)/(C,T)/(1,C,T)/(n,), got {x.shape}")
    if not fps:
        raise ValueError("feature/latent input needs fps= (SA3 latents: 10.767)")
    if x.shape[0] < x.shape[1]:  # (C, T) -> (T, C); latent C=256 >> typical T? no:
        # heuristic only kicks in when one axis is clearly time (longer). Callers
        # with square-ish arrays should pass (T, C) explicitly.
        x = x.T
    return x.astype(np.float64), float(fps)


def _whiten(feats):
    """Per-channel z-score over the whole sequence -- the v3 correction. A steady
    kick / constant-energy channel flattens to ~0 and stops driving similarity."""
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True)
    return (feats - mu) / np.maximum(sd, 1e-8)


def _patch_matrix(feats, fps, patch_sec, stride_sec):
    """(P, pf*C) unit-norm whitened-patch matrix + patch times. Shared by
    novelty_curve and dynamics_stats -- ONE copy of the patch construction so the
    two statistics families stay on the same distance matrix (plan E0-A / C's Q3)."""
    pf = max(2, int(round(patch_sec * fps)))       # frames per patch
    sf = max(1, int(round(stride_sec * fps)))      # stride frames
    starts = np.arange(0, feats.shape[0] - pf + 1, sf)
    if starts.size == 0:
        return np.zeros((0, 0)), np.array([])
    patches = np.stack([feats[s:s + pf].ravel() for s in starts])   # (P, pf*C)
    norms = np.linalg.norm(patches, axis=1)
    unit = patches / np.maximum(norms[:, None], 1e-12)
    return unit, starts / fps


def novelty_curve(x, sr=None, fps=None, patch_sec=PATCH_SEC, stride_sec=STRIDE_SEC,
                  lookback_min_sec=LOOKBACK_MIN_SEC, lookback_max_sec=LOOKBACK_MAX_SEC,
                  n_mels=64, whiten=True):
    """Per-patch {recurrence, novelty} over the sequence.

    recurrence[i] = max cosine similarity of the (whitened, flattened) patch at t_i
    to every patch in the strictly-preceding lookback band [t_i-40s, t_i-8s].
    novelty = 1 - recurrence. Patches before lookback_min_sec have no reference ->
    NaN (callers/controllers should skip or hold).
    """
    feats, fps = _to_features(x, sr=sr, fps=fps, n_mels=n_mels)
    if whiten:
        feats = _whiten(feats)
    unit, times = _patch_matrix(feats, fps, patch_sec, stride_sec)
    if unit.shape[0] == 0:
        return {"time": np.array([]), "recurrence": np.array([]),
                "novelty": np.array([]), "fps": fps, "patch_sec": patch_sec,
                "stride_sec": stride_sec, "n_patches": 0}
    starts = np.arange(len(times))  # patch indices (times carries seconds)

    lb_min_p = int(round(lookback_min_sec / stride_sec))
    lb_max_p = int(round(lookback_max_sec / stride_sec))
    rec = np.full(len(starts), np.nan)
    for i in range(len(starts)):
        lo, hi = max(0, i - lb_max_p), i - lb_min_p
        if hi <= lo:
            continue  # no strictly-preceding reference band yet
        rec[i] = float((unit[lo:hi] @ unit[i]).max())
    nov = 1.0 - rec
    return {"time": times, "recurrence": rec, "novelty": nov, "fps": fps,
            "patch_sec": patch_sec, "stride_sec": stride_sec,
            "n_patches": int(len(starts))}


def calibrate_source(x, sr=None, fps=None, **kw):
    """Source-track self-calibration for the #35 controller thresholds.

    Returns {novelty_floor (p10), novelty_median, r_max, r_median, n_patches}.
    The controller's 'looping' trigger should reference novelty_floor / r_max of
    the SOURCE, never a global constant (journal: per-track calibration is
    NECESSARY -- 0.968 recurrence is looping for one track, ~source for another).
    """
    res = novelty_curve(x, sr=sr, fps=fps, **kw)
    nov = res["novelty"][~np.isnan(res["novelty"])]
    rec = res["recurrence"][~np.isnan(res["recurrence"])]
    if nov.size == 0:
        return {"novelty_floor": None, "novelty_median": None,
                "r_max": None, "r_median": None, "n_patches": 0}
    return {"novelty_floor": float(np.percentile(nov, 10)),
            "novelty_median": float(np.median(nov)),
            "r_max": float(rec.max()), "r_median": float(np.median(rec)),
            "n_patches": int(nov.size)}


def dynamics_stats(x, sr=None, fps=None, patch_sec=PATCH_SEC, stride_sec=STRIDE_SEC,
                   n_mels=64, whiten=True, theiler_sec=8.0, rr_target=0.05,
                   lmin_sec=4.0, n_eps=12, ci=True, n_boot=50, block_sec=30.0,
                   seed=0):
    """Trajectory-dynamics statistics on the SAME whitened-patch matrix as
    novelty_curve (plan E0-A, 2026-07-15). Two families:

    * corr_dim -- finite-time correlation dimension (Grassberger-Procaccia slope
      of log C(eps) vs log eps; RMR 2605.00435 Def 4.1). Measures state-space
      accessibility: a limit cycle reads ~1, real arrangement higher. Catches
      IMPLICIT collapse (timbral variation riding a loop skeleton) that
      max-cosine recurrence misses (C's Q3, theory review 2026-07-15).
    * det / det_soft / l_max_sec -- RQA determinism: mass of recurrence-matrix
      DIAGONAL lines >= lmin. Fires on SUSTAINED orbit-locking, not single
      re-approach (a legit theme return spikes recurrence but not DET).
      det_soft is the sigmoid-relaxed window-product form -- the E1 torch
      potential ports it 1:1 (tangential-refinements doc S1, adopted plan S6).

    Estimator hygiene: Theiler window (|i-j| >= theiler strides) excludes the
    trivial overlap band (patches overlap at 4s/1s); eps fit range = [q05, q50]
    of off-Theiler chord distances, reported; block-bootstrap CI over contiguous
    segments. tau is set PER SEQUENCE for a fixed recurrence rate (rr_target)
    -- the per-track-calibration rule, no global threshold.
    """
    feats, fps = _to_features(x, sr=sr, fps=fps, n_mels=n_mels)
    if whiten:
        feats = _whiten(feats)
    unit, times = _patch_matrix(feats, fps, patch_sec, stride_sec)
    P = unit.shape[0]
    w = max(1, int(round(theiler_sec / stride_sec)))
    lmin = max(2, int(round(lmin_sec / stride_sec)))
    empty = {"corr_dim": None, "corr_dim_ci": None, "corr_dim_fit_r2": None,
             "eps_range": None, "det": None, "det_soft": None, "l_max_sec": None,
             "rr": None, "tau": None, "n_patches": int(P),
             "theiler_strides": w, "l_min_strides": lmin}
    if P < max(3 * w, 2 * lmin + w, 24):
        return empty

    S = unit @ unit.T                                   # (P, P) cosine
    ii, jj = np.triu_indices(P, k=w)                    # off-Theiler upper pairs
    sv = S[ii, jj]

    # --- correlation dimension in PCA state space ---
    # G-P on the raw ~10k-dim patch vectors is meaningless: distances concentrate
    # and the slope reads the embedding dim, not the dynamics (smoke test
    # 2026-07-15: healthy and tiled-loop both ~25). Project to a low-dim PCA
    # state space first; with P~400 points the estimable ceiling is
    # ~2*log10(P) ~ 5 anyway (Grassberger's rule), so d_pca=16 is generous.
    d_pca = min(16, P - 1, unit.shape[1])
    Z = unit - unit.mean(axis=0, keepdims=True)
    _, sval, Vt = np.linalg.svd(Z, full_matrices=False)
    coords = Z @ Vt[:d_pca].T                           # (P, d_pca), unwhitened
    Dm = np.sqrt(np.maximum(
        ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1), 0.0))

    def _slope(dv):
        e0, e1 = np.quantile(dv, 0.02), np.quantile(dv, 0.30)
        if not (np.isfinite(e0) and np.isfinite(e1)) or e1 <= 0:
            return None, None, (None, None)
        e0 = max(e0, e1 * 1e-3)
        eps = np.exp(np.linspace(np.log(e0), np.log(e1), n_eps))
        C = np.array([(dv < e).mean() for e in eps])
        m = C > 0
        if m.sum() < 3:
            return None, None, (float(e0), float(e1))
        lx, ly = np.log(eps[m]), np.log(C[m])
        b, a = np.polyfit(lx, ly, 1)
        r2 = 1.0 - np.sum((ly - (b * lx + a)) ** 2) / max(np.sum((ly - ly.mean()) ** 2), 1e-12)
        return float(b), float(r2), (float(e0), float(e1))

    dv = Dm[ii, jj]
    corr_dim, fit_r2, eps_range = _slope(dv)

    corr_dim_ci = None
    if ci and corr_dim is not None:
        rng = np.random.default_rng(seed)
        L = max(lmin, int(round(block_sec / stride_sec)))
        nblk = max(1, P // L)
        slopes = []
        for _ in range(n_boot):
            starts_b = rng.integers(0, max(P - L, 1), size=nblk)
            idx = np.concatenate([np.arange(s, min(s + L, P)) for s in starts_b])
            Db = Dm[np.ix_(idx, idx)]
            bi, bj = np.triu_indices(len(idx), k=w)
            sb, _, _ = _slope(Db[bi, bj])
            if sb is not None:
                slopes.append(sb)
        if len(slopes) >= 10:
            corr_dim_ci = (float(np.percentile(slopes, 16)),
                           float(np.percentile(slopes, 84)))

    # --- RQA determinism on the thresholded recurrence matrix ---
    tau = float(np.quantile(sv, 1.0 - rr_target))
    R = S >= tau
    # multi-scale line-mass: goa is NATURALLY repetitive (structural repeats put
    # diagonal lines in any healthy track), so det at one short lmin dilutes --
    # the discriminator is how much recurrent mass sits in LONG locks (smoke
    # test: l_max 29s healthy vs 208s tiled-loop). Emit fractions at 4/8/16s.
    scales = sorted({lmin, 2 * lmin, 4 * lmin})
    total_rec, l_max = 0, 0
    line_mass = {sc: 0 for sc in scales}
    soft_prods = []
    temp = max(0.25 * float(sv.std()), 1e-6)
    Pm = 1.0 / (1.0 + np.exp(-(S - tau) / temp))
    for k in range(w, P):
        seq = np.diagonal(R, offset=k).astype(np.int8)
        total_rec += int(seq.sum())
        d = np.diff(np.concatenate(([0], seq, [0])))
        run_starts, run_ends = np.where(d == 1)[0], np.where(d == -1)[0]
        for rs, re in zip(run_starts, run_ends):
            rl = re - rs
            l_max = max(l_max, rl)
            for sc in scales:
                if rl >= sc:
                    line_mass[sc] += rl
        p = np.diagonal(Pm, offset=k)
        if len(p) >= lmin:
            win = np.lib.stride_tricks.sliding_window_view(p, lmin)
            soft_prods.append(win.prod(axis=1))
    det = float(line_mass[lmin] / total_rec) if total_rec else 0.0
    line_frac = {f"line_frac_{int(sc * stride_sec)}s":
                 (float(line_mass[sc] / total_rec) if total_rec else 0.0)
                 for sc in scales}
    det_soft = float(np.concatenate(soft_prods).mean()) if soft_prods else 0.0
    rr = float(total_rec / len(sv)) if len(sv) else 0.0

    out = {"corr_dim": abs(corr_dim) if corr_dim is not None else None,
           "corr_dim_ci": tuple(abs(c) for c in corr_dim_ci) if corr_dim_ci else None,
           "corr_dim_fit_r2": fit_r2, "eps_range": eps_range,
           "det": det, "det_soft": det_soft,
           "l_max_sec": float(l_max * stride_sec), "rr": rr, "tau": tau,
           "n_patches": int(P), "theiler_strides": w, "l_min_strides": lmin}
    out.update(line_frac)
    return out


def main():
    ap = argparse.ArgumentParser(description="patch-level recurrence/novelty meter")
    ap.add_argument("input", help="audio file or .npy feature/latent array")
    ap.add_argument("--fps", type=float, default=None,
                    help="frames/sec for array input (SA3 latents: 10.767)")
    ap.add_argument("--json", default="", help="write JSON here (else stdout summary)")
    args = ap.parse_args()

    p = Path(args.input)
    x = np.load(p) if p.suffix == ".npy" else p
    res = novelty_curve(x, fps=args.fps)
    cal = calibrate_source(x, fps=args.fps)
    dyn = dynamics_stats(x, fps=args.fps)
    payload = {"input": str(p), "calibration": cal, "dynamics": dyn,
               "curve": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in res.items()}}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"[recurrence] {res['n_patches']} patches -> {args.json}")
    print(f"[recurrence] novelty_floor={cal['novelty_floor']} r_max={cal['r_max']} "
          f"n={cal['n_patches']}")
    cd = dyn["corr_dim"]
    print(f"[dynamics]   corr_dim={cd if cd is None else round(cd, 3)} "
          f"(r2={dyn['corr_dim_fit_r2'] and round(dyn['corr_dim_fit_r2'], 3)}) "
          f"det={dyn['det'] and round(dyn['det'], 3)} det_soft={dyn['det_soft'] and round(dyn['det_soft'], 4)} "
          f"l_max={dyn['l_max_sec']}s rr={dyn['rr'] and round(dyn['rr'], 3)}")


if __name__ == "__main__":
    main()
