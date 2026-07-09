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
    pf = max(2, int(round(patch_sec * fps)))       # frames per patch
    sf = max(1, int(round(stride_sec * fps)))      # stride frames
    starts = np.arange(0, feats.shape[0] - pf + 1, sf)
    if starts.size == 0:
        return {"time": np.array([]), "recurrence": np.array([]),
                "novelty": np.array([]), "fps": fps, "patch_sec": patch_sec,
                "stride_sec": stride_sec, "n_patches": 0}
    patches = np.stack([feats[s:s + pf].ravel() for s in starts])   # (P, pf*C)
    norms = np.linalg.norm(patches, axis=1)
    unit = patches / np.maximum(norms[:, None], 1e-12)
    times = starts / fps

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
    payload = {"input": str(p), "calibration": cal,
               "curve": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in res.items()}}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"[recurrence] {res['n_patches']} patches -> {args.json}")
    print(f"[recurrence] novelty_floor={cal['novelty_floor']} r_max={cal['r_max']} "
          f"n={cal['n_patches']}")


if __name__ == "__main__":
    main()
