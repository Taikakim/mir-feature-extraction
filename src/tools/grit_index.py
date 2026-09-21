#!/usr/bin/env python
"""grit_index.py — is the gritty distortion a MODEL artefact or just distorted synths?

Kim, 2026-09-17: SA3 output sometimes has a distortion that is "not entirely the nice one
someone would do on purpose" — people have remarked on it since Stable Audio Open v1. It is
not in every track, and the hard part is that goa is FULL of deliberately distorted leads and
basses, so "there is distortion here" proves nothing on its own.

Crest factor already came back null (generated 13.37/13.03 dB vs real goa 12.54, i.e. the
clips are slightly LESS compressed than the corpus). But crest is blind to smooth saturation
by construction — a saturator adds harmonics while leaving peak/RMS nearly untouched. So that
null does not settle the question; it just means the wrong instrument was used.

THE DISCRIMINATOR. Structured audio is either TONAL (energy locked to a harmonic series,
stable across time — a synth, a bass, even a heavily saturated one, because saturating a
periodic signal produces MORE harmonics, not noise) or TRANSIENT (broadband, brief — a
hat, a click). Model grit is neither: it is stochastic, time-varying, and wedged between the
partials. Median-filter HPSS separates exactly those two structures, so what it CANNOT
explain is the residual:

    R = y - H - P          grit_index = energy(R, 2-8 kHz) / energy(y, 2-8 kHz)

2-8 kHz because that is where "cheesegrater", "nasal grating" and "gritty high end" live.
A deliberately distorted lead lands in H (its harmonics are harmonics). A hat lands in P.
Hiss, hallucinated HF texture and intermodulation mush land in R.

WHY THE REFERENCE CORPUS IS THE WHOLE POINT. A high grit_index alone means nothing —
real goa records plenty of it. The question is ONLY whether the generated clips sit outside
what the genre actually does. So this reports the clips against a random sample of real goa
measured identically, and the verdict is a percentile, not a threshold.

Reads mono at 22.05 kHz (Nyquist 11 kHz, comfortably above the 8 kHz band of interest) from
a centred excerpt — HPSS on full 44.1 kHz tracks is far slower for no extra information here.

Run (mir venv):
    mir/bin/python src/tools/grit_index.py --labels <export.json> \
        --ref-glob "/run/media/kim/Mantu/ai-music/Goa Dataset/**/*.flac" \
        --extra /home/kim/mixtape_harshness_review/68_*.wav --out /tmp/grit
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # mir/src

SR = 22050
BAND = (2000.0, 8000.0)
NFFT = 2048
HOP = 512


def load_excerpt(path, seconds=20.0):
    """Centred excerpt, mono, at SR. Centred because intros/outros misrepresent a track."""
    from core.file_utils import read_audio
    import librosa
    a, sr = read_audio(str(path))
    x = a.mean(axis=1) if getattr(a, "ndim", 1) > 1 else np.asarray(a)
    x = np.asarray(x, dtype=np.float32)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    n = len(x)
    want = int(seconds * SR)
    if n < SR * 4:
        return None
    if n > want:
        s = (n - want) // 2
        x = x[s:s + want]
    return x


def band_energy(S, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return float(np.sum(np.abs(S[m, :]) ** 2))


def grit(path, seconds=20.0, margin=2.0):
    """Residual-energy fraction in BAND: what neither a harmonic nor a transient explains."""
    import librosa
    x = load_excerpt(path, seconds)
    if x is None or not np.isfinite(x).all() or np.max(np.abs(x)) <= 0:
        return None
    S = librosa.stft(x, n_fft=NFFT, hop_length=HOP)
    # margin > 1 leaves a genuine residual; margin=1 forces H+P == S and the test is empty.
    H, P = librosa.decompose.hpss(S, margin=(margin, margin))
    R = S - H - P
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    tot = band_energy(S, freqs, *BAND)
    if tot <= 0:
        return None
    res = band_energy(R, freqs, *BAND)
    harm = band_energy(H, freqs, *BAND)
    perc = band_energy(P, freqs, *BAND)
    flat = float(np.mean(librosa.feature.spectral_flatness(S=np.abs(S))))
    return {"grit": res / tot, "harm": harm / tot, "perc": perc / tot, "flat": flat}


def summarise(name, vals):
    if not vals:
        print(f"  {name:28s}  (none)")
        return None
    v = np.array([d["grit"] for d in vals])
    print(f"  {name:28s} n={len(v):3d}  median {np.median(v):.4f}   "
          f"p25 {np.percentile(v,25):.4f}  p75 {np.percentile(v,75):.4f}")
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--ref-glob", required=True)
    ap.add_argument("--ref-n", type=int, default=60)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="specific files to place against the distributions")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", type=Path, default=Path("/tmp/grit"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    ref_files = [p for p in glob.glob(a.ref_glob, recursive=True) if os.path.isfile(p)]
    random.seed(1)
    ref_files = random.sample(ref_files, min(a.ref_n, len(ref_files)))
    print(f"reference: {len(ref_files)} real goa tracks")

    ref = []
    for i, p in enumerate(ref_files):
        try:
            g = grit(p, a.seconds)
        except Exception:
            g = None
        if g:
            ref.append(g)
        if (i + 1) % 20 == 0:
            print(f"  ref {i+1}/{len(ref_files)}", flush=True)

    flagged, clean = [], []
    if a.labels:
        rows = json.load(open(a.labels))
        for i, r in enumerate(rows):
            try:
                g = grit(r["path"], a.seconds)
            except Exception:
                g = None
            if not g:
                continue
            (flagged if r.get("hf_damping") is True else clean).append(g)
            if (i + 1) % 20 == 0:
                print(f"  clips {i+1}/{len(rows)}", flush=True)

    print(f"\nGRIT INDEX — residual energy fraction in {BAND[0]:.0f}-{BAND[1]:.0f} Hz")
    print("  (what is neither harmonic nor transient; higher = more stochastic texture)")
    R = summarise("real goa", ref)
    F = summarise("generated, flagged harsh", flagged)
    C = summarise("generated, not flagged", clean)

    if R is not None and F is not None and C is not None:
        from scipy.stats import mannwhitneyu
        print(f"\n  flagged vs not-flagged   p = "
              f"{mannwhitneyu(F, C, alternative='two-sided').pvalue:.3e}")
        print(f"  generated vs real goa    p = "
              f"{mannwhitneyu(np.concatenate([F, C]), R, alternative='two-sided').pvalue:.3e}")
        gen = np.concatenate([F, C])
        pct = float((R < np.median(gen)).mean() * 100)
        print(f"\n  VERDICT: the median generated clip sits at the {pct:.0f}th percentile "
              f"of real goa.")
        print("  Near 50 => indistinguishable from the genre. Near 100 => the model is "
              "adding texture the\n           corpus does not contain, i.e. an artefact "
              "rather than a style choice.")

    for e in a.extra:
        for p in sorted(glob.glob(e)):
            g = grit(p, a.seconds)
            if not g:
                continue
            pr = float((R < g["grit"]).mean() * 100) if R is not None else float("nan")
            print(f"\n  {os.path.basename(p)[:60]}")
            print(f"    grit {g['grit']:.4f}  (= {pr:.0f}th pct of real goa) | "
                  f"harmonic {g['harm']:.3f}  transient {g['perc']:.3f}")

    json.dump({"band_hz": list(BAND), "seconds": a.seconds, "sr": SR,
               "n": {"ref": len(ref), "flagged": len(flagged), "clean": len(clean)},
               "median": {"ref": float(np.median(R)) if R is not None else None,
                          "flagged": float(np.median(F)) if F is not None else None,
                          "clean": float(np.median(C)) if C is not None else None},
               "note": "grit = residual/(total) in band after median-filter HPSS; the "
                       "reference corpus is the control for 'goa synths are distorted "
                       "on purpose'", "kim_feedback": None},
              open(a.out / "grit.json", "w"), indent=2)
    print(f"\n[done] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
