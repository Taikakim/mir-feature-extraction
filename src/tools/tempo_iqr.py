#!/usr/bin/env python3
"""
tempo_iqr.py -- within-clip tempo-instability meter (the "tempo_iqr-min" ship-picker term).

Measures how much the dominant tempo WANDERS across a single rendered clip. A checkpoint that
has collapsed to its chaos attractor produces clips whose local tempo drifts/doubles/halves
across the clip -> high IQR. A clip that locks to one tempo -> IQR ~0. This is process #2 of
CONTINUITY's three-process degradation decomposition (the U-shaped tempo instability, ep31 notch),
independently measured; it composes with CE-max (Audiobox) + pre-ZCR-drop into the automated
ship-checkpoint picker (SAO eval/ship_checkpoint_picker.py).

Method: sliding windows across the clip -> dominant tempo per window (librosa) -> octave-fold
each toward the clip's median (so a half/double-time read doesn't fake instability, but a genuine
tempo wander does) -> IQR of the folded per-window tempos = the instability score.

Interfaces (venv-agnostic so the picker can import OR subprocess this):
  from tempo_iqr import tempo_iqr_for_dir            # -> {clip_name: tempo_iqr}
  python tempo_iqr.py <clip_dir> [--out scores.json] [--full] [--jobs N]
Run with mir/bin/python (librosa). Zero GPU.
"""
import sys, os, json, argparse, math, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

AUDIO_EXTS = (".wav", ".flac", ".m4a", ".mp3", ".ogg", ".opus")


def _tempo_fn():
    """Return a callable(onset_envelope, sr) -> np.ndarray of tempo estimate(s).
    librosa moved tempo between namespaces across versions; support both."""
    import librosa
    try:  # non-deprecated location (librosa >= 0.10) first, to avoid the FutureWarning noise
        from librosa.feature.rhythm import tempo as _t
        return lambda oenv, sr: np.atleast_1d(_t(onset_envelope=oenv, sr=sr))
    except Exception:
        return lambda oenv, sr: np.atleast_1d(librosa.beat.tempo(onset_envelope=oenv, sr=sr))


def _fold_to_ref(t, ref):
    """Octave-fold tempo t to within [ref/sqrt2, ref*sqrt2) -- centers folding on the clip's
    actual tempo, avoiding the fixed-boundary discontinuity of a hard [70,140) window."""
    if t <= 0 or ref <= 0:
        return t
    lo, hi = ref / math.sqrt(2.0), ref * math.sqrt(2.0)
    for _ in range(8):
        if t < lo:
            t *= 2.0
        elif t >= hi:
            t /= 2.0
        else:
            break
    return t


def tempo_iqr_for_clip(path, sr=22050, win_sec=8.0, hop_sec=4.0):
    """Return {tempo_iqr, tempo_median, n_windows} for one clip (tempo_iqr is the ship term)."""
    import librosa
    tempo = _tempo_fn()
    try:
        y, sr = librosa.load(str(path), sr=sr, mono=True)
    except Exception as e:
        return {"tempo_iqr": None, "tempo_median": None, "n_windows": 0, "error": f"load: {e}"}
    if y.size == 0:
        return {"tempo_iqr": None, "tempo_median": None, "n_windows": 0, "error": "empty"}

    dur = y.size / sr
    win, hop = int(win_sec * sr), int(hop_sec * sr)
    raw = []
    if dur < win_sec:  # too short to window -> single estimate, no within-clip spread
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        est = tempo(oenv, sr)
        if est.size:
            raw = [float(est[0])]
    else:
        for start in range(0, y.size - win + 1, hop):
            oenv = librosa.onset.onset_strength(y=y[start:start + win], sr=sr)
            est = tempo(oenv, sr)
            if est.size and est[0] > 0:
                raw.append(float(est[0]))

    if not raw:
        return {"tempo_iqr": None, "tempo_median": None, "n_windows": 0, "error": "no tempo"}

    ref = float(np.median(raw))
    folded = np.array([_fold_to_ref(t, ref) for t in raw], dtype=float)
    q75, q25 = np.percentile(folded, [75, 25])
    return {
        "tempo_iqr": float(q75 - q25),
        "tempo_median": float(np.median(folded)),
        "n_windows": len(folded),
    }


def _list_clips(clip_dir):
    d = Path(clip_dir)
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def tempo_iqr_for_dir(clip_dir, jobs=1, full=False):
    """clip-dir -> {clip_name: tempo_iqr} (or -> {clip_name: full dict} if full=True).
    This is the ship-picker's tempo_iqr-min term: lower = more tempo-stable = healthier ckpt."""
    clips = _list_clips(clip_dir)
    out = {}
    if jobs > 1 and len(clips) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(tempo_iqr_for_clip, str(p)): p.name for p in clips}
            for f in as_completed(futs):
                name = futs[f]
                try:
                    res = f.result()
                except Exception as e:
                    res = {"tempo_iqr": None, "error": str(e)}
                out[name] = res if full else res.get("tempo_iqr")
    else:
        for p in clips:
            res = tempo_iqr_for_clip(str(p))
            out[p.name] = res if full else res.get("tempo_iqr")
    return out


def main():
    ap = argparse.ArgumentParser(description="within-clip tempo-instability meter (ship-picker term)")
    ap.add_argument("clip_dir")
    ap.add_argument("--out", default="", help="write JSON here (else stdout)")
    ap.add_argument("--full", action="store_true", help="emit full {tempo_iqr,tempo_median,n_windows}")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    scores = tempo_iqr_for_dir(args.clip_dir, jobs=args.jobs, full=args.full)
    vals = [v["tempo_iqr"] if isinstance(v, dict) else v for v in scores.values()]
    vals = [x for x in vals if x is not None]
    # Ranking term for the ship-picker: on mostly-stable ladders the MEDIAN floors at 0 (most
    # clips lock) -- the discriminative signal is the unstable-clip TAIL, so rank by MEAN (or p90 /
    # frac_unstable), not median. Verified on arm G (flat) vs r16 (U-shape) 2026-07-09.
    thr = 3.0  # bpm IQR above this = a clip whose tempo genuinely wandered
    payload = {
        "clip_dir": str(args.clip_dir),
        "ckpt_tempo_iqr_mean": float(np.mean(vals)) if vals else None,       # <- rank on this
        "ckpt_tempo_iqr_p90": float(np.percentile(vals, 90)) if vals else None,
        "ckpt_tempo_iqr_median": float(np.median(vals)) if vals else None,   # floors at 0, kept for continuity
        "ckpt_frac_unstable": float(np.mean([v > thr for v in vals])) if vals else None,
        "n_clips": len(scores),
        "clips": scores,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"[tempo_iqr] {len(scores)} clips -> {args.out} "
              f"(ckpt mean iqr {payload['ckpt_tempo_iqr_mean']:.2f}, "
              f"frac_unstable {payload['ckpt_frac_unstable']:.2f})")
    else:
        print(text)


if __name__ == "__main__":
    main()
