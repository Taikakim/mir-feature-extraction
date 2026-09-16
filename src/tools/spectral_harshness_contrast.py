#!/usr/bin/env python
"""spectral_harshness_contrast.py — WHERE, in Hz, does "harsh" actually live?

Project guidance 2026-09-16/17. The rms_energy_air LatCH head was being used to damp
harsh high end on generated clips and did not work. Two causes were found (a sampler
bug, since fixed; and a target set to the corpus mean, i.e. barely a request). A third
is suspected and this tool tests it: **the band may simply be wrong.**

`rms_energy_air` is 2500-22000 Hz -- ONE bucket spanning bite, sibilance and true air.
"Harsh/biting" perceptually tends to sit around 2-5 kHz. Steering the whole bucket
would then dull a track without fixing the bite. Rather than argue about it, measure:
compare the spectral contour of clips a human FLAGGED as harsh against a reference
built from the reference corpus, and see which bands actually separate them.

TWO DESIGN RULES, both learned the hard way:

1. **Level-invariance.** Contours are anchored on 1-3 kHz (subtract that region's mean)
   before anything else, so "quieter" and "darker" cannot be confused. An absolute
   energy target is satisfiable by just turning the whole clip down -- which is exactly
   what the guided renders did (air -26.2 -> -29.0 dB, but BODY -23.9 -> -28.0: it
   dropped the mids MORE than the highs). Same lesson as onset_per_beat making the
   tempo shortcut unexpressible.

2. **The reference is REAL MUSIC, not model output** (project guidance, explicit).
   Generated clips carry whatever spectral bias we are trying to correct, so using them
   as the reference would define the problem away. Pink noise is also wrong here: the
   reference corpus falls at ~-4.4 dB/oct above 1 kHz, steeper than pink's -3, so a pink
   target reads these masters as ~6 dB HF-deficient and would push generations BRIGHTER.

DEVIATIONS ARE REPORTED IN SIGMA, NOT dB, and that is the point. The reference corpus
is opinionated about some bands and indifferent about others (measured: p10-p90 spread
is 2.8 dB at 2.0-2.3 kHz but 11-15 dB above 10 kHz). A 3 dB deviation at 2 kHz is
therefore far outside what the corpus permits while the same 3 dB at 12 kHz is
unremarkable. Sigma-normalising weights each band by how much the reference actually
constrains it -- which is a principled version of "drop the outlier bins", with no
arbitrary 10% rule.

CODEC CAVEAT: AAC at 128k lowpasses around 16-17 kHz. If the clips under test are .m4a
and the reference is .flac, the top bands are a CODEC artefact, not a finding. The tool
prefers a lossless sibling when one exists and always prints which it used; bands above
--codec-safe-hz are marked so they are never read as signal.

Run (mir venv -- needs essentia-free librosa/soundfile + the m4a fallback reader):

    mir/bin/python src/tools/spectral_harshness_contrast.py \
        --labels /home/kim/Projects/SAO/eval/hf_damping_export_2026-09-17.json \
        --reference-dir "/run/media/kim/Mantu/avp-analyzed-stems/*/full_mix.flac" \
        --out /tmp/harshness_contrast
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # mir/src

NB = 40
NFFT = 8192
ANCHOR_LO, ANCHOR_HI = 1000.0, 3000.0     # level anchor: perceptually central, stable


def hz2erb(f):
    return 21.4 * np.log10(1 + 0.00437 * np.asarray(f, dtype=float))


def erb2hz(e):
    return (10 ** (np.asarray(e, dtype=float) / 21.4) - 1) / 0.00437


def band_edges(n=NB, lo=20.0, hi=20000.0):
    return erb2hz(np.linspace(hz2erb(lo), hz2erb(hi), n + 1))


MAX_SEGS = 240          # bounds cost AND spreads sampling across the whole window


def read_any(path, start_pct=0.2, end_pct=0.8):
    """Lossless or lossy; m4a goes through mir's ffmpeg-fallback reader.

    Window is TRACK-RELATIVE (project guidance 2026-09-17): 20%-80% of duration, not
    the first N seconds. Intros and outros are routinely filtered, sparse or faded, so
    a head-anchored window measures the arrangement's edges rather than its body -- and
    for a spectral-balance reference that is precisely the wrong part of the track.
    """
    from core.file_utils import read_audio
    audio, sr = read_audio(str(path))
    x = audio.mean(axis=1) if getattr(audio, "ndim", 1) > 1 else audio
    n = len(x)
    a, b = int(start_pct * n), int(end_pct * n)
    return np.asarray(x[a:b], dtype=np.float64), sr


def contour(path, edges, start_pct=0.2, end_pct=0.8):
    """Level-anchored ERB-band contour in dB. None if the window is too short."""
    x, sr = read_any(path, start_pct, end_pct)
    if len(x) < NFFT * 4:
        return None
    win = np.hanning(NFFT)
    step = NFFT // 2
    n = (len(x) - NFFT) // step
    if n < 8:
        return None
    # Spread up to MAX_SEGS segments evenly over the window instead of taking the first
    # MAX_SEGS contiguously -- a contiguous head-slice of the window reintroduces the
    # same positional bias the 20-80% framing exists to remove.
    idx = np.unique(np.linspace(0, n - 1, min(n, MAX_SEGS)).astype(int))
    acc = np.zeros(NFFT // 2 + 1)
    for i in idx:
        acc += np.abs(np.fft.rfft(x[i * step:i * step + NFFT] * win)) ** 2
    acc /= len(idx)
    freqs = np.fft.rfftfreq(NFFT, 1 / sr)
    out = []
    for i in range(len(edges) - 1):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out.append(acc[m].mean() if m.any() else np.nan)
    b = 10 * np.log10(np.asarray(out) + 1e-20)
    ctr = (edges[:-1] >= ANCHOR_LO) & (edges[:-1] < ANCHOR_HI)
    if not ctr.any() or not np.isfinite(b[ctr]).any():
        return None
    return b - np.nanmean(b[ctr])


def gather(paths, edges, start_pct, end_pct, label=""):
    rows, used = [], []
    for i, p in enumerate(paths):
        try:
            c = contour(p, edges, start_pct, end_pct)
        except Exception:
            c = None
        if c is not None and np.isfinite(c).all():
            rows.append(c)
            used.append(p)
        if label and (i + 1) % 25 == 0:
            print(f"  {label}: {i+1}/{len(paths)}", flush=True)
    return (np.asarray(rows) if rows else np.zeros((0, len(edges) - 1))), used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, type=Path,
                    help="JSON list of {file, path, hf_damping: true/false/null}")
    ap.add_argument("--reference-dir", required=True,
                    help="GLOB of reference-corpus audio (real music, NOT model output)")
    ap.add_argument("--wav-root", default="/run/media/kim/Mantu/sa3_lora_runs/model_matrix",
                    help="searched for a lossless sibling of each labelled clip")
    ap.add_argument("--out", type=Path, default=Path("/tmp/harshness_contrast"))
    ap.add_argument("--start-pct", type=float, default=0.20,
                    help="window start as a FRACTION OF TRACK DURATION (project guidance "
                         "2026-09-17). Track-relative, not head-anchored: intros and outros "
                         "are routinely filtered/sparse/faded, so a first-N-seconds window "
                         "measures the arrangement's edges rather than its body.")
    ap.add_argument("--end-pct", type=float, default=0.80)
    ap.add_argument("--codec-safe-hz", type=float, default=15000.0)
    ap.add_argument("--null-is-clean", action="store_true",
                    help="pool null labels with the explicit 'false' ones. The tagging UI "
                         "appears to record only a POSITIVE harsh mark, so an unflagged clip "
                         "is one the listener heard and did not flag -- not one never heard. "
                         "CAVEAT (Kim 2026-09-17): he skipped ahead on clips that sounded OK "
                         "at the START, so 'not flagged' means 'the opening was fine', and "
                         "this pool is contaminated with clips that turn harsh later. That "
                         "DILUTES a real difference rather than inventing one -- so a "
                         "surviving effect is a floor, not a ceiling. Pair it with "
                         "--start-pct 0 --end-pct 0.25 to measure the part actually judged.")
    ap.add_argument("--skip-reference", action="store_true",
                    help="skip the reference corpus; normalise by the pooled spread of the "
                         "clips themselves. Much faster when the question is flagged-vs-not "
                         "rather than generated-vs-real-music.")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    edges = band_edges()
    lo_e, hi_e = edges[:-1], edges[1:]

    rows = json.load(open(a.labels))
    def resolve(r):
        stem = os.path.splitext(os.path.basename(r["file"]))[0]
        hit = glob.glob(f"{a.wav_root}/**/{stem}.wav", recursive=True)
        return (hit[0], "wav") if hit else (r["path"], os.path.splitext(r["path"])[1].lstrip("."))

    flagged, clean, unrev, fmts = [], [], [], {}
    for r in rows:
        p, f = resolve(r)
        fmts[f] = fmts.get(f, 0) + 1
        v = r.get("hf_damping")
        (flagged if v is True else
         clean if (v is False or (v is None and a.null_is_clean)) else unrev).append(p)
    print(f"labels: {len(flagged)} flagged, {len(clean)} cleared, {len(unrev)} unreviewed")
    print(f"clip formats resolved: {fmts}")
    if "wav" not in fmts:
        print(f"  NOTE: no lossless siblings found -> bands above {a.codec_safe_hz:.0f} Hz are "
              f"codec-limited and marked UNSAFE below; do not read them as signal.")

    if a.skip_reference:
        REF = np.zeros((0, len(lo_e)))
        print("reference corpus: SKIPPED (normalising by the clips' own pooled spread)")
    else:
        ref_paths = sorted(glob.glob(a.reference_dir))
        print(f"reference corpus: {len(ref_paths)} files")
        REF, _ = gather(ref_paths, edges, a.start_pct, a.end_pct, "ref")
        if len(REF) < 10:
            print("ERROR: reference corpus too small to characterise"); return 1

    FLAG, _ = gather(flagged, edges, a.start_pct, a.end_pct, "flagged")
    CLEAN, _ = gather(clean, edges, a.start_pct, a.end_pct, "cleared")
    UNREV, _ = gather(unrev, edges, a.start_pct, a.end_pct, "unreviewed")
    print(f"usable: ref {len(REF)}, flagged {len(FLAG)}, cleared {len(CLEAN)}, unreviewed {len(UNREV)}")
    if len(FLAG) == 0:
        print("ERROR: no flagged clips readable"); return 1

    if len(REF):
        ref_med = np.median(REF, axis=0)
        # robust sigma from the central 80% -- resistant to one odd master
        ref_sigma = (np.percentile(REF, 90, axis=0) - np.percentile(REF, 10, axis=0)) / 2.5631
    else:
        POOL = np.vstack([x for x in (FLAG, CLEAN, UNREV) if len(x)])
        ref_med = np.median(POOL, axis=0)
        ref_sigma = (np.percentile(POOL, 90, axis=0) - np.percentile(POOL, 10, axis=0)) / 2.5631
    ref_sigma = np.maximum(ref_sigma, 1e-6)

    flag_med = np.median(FLAG, axis=0)
    dev_sigma = (flag_med - ref_med) / ref_sigma

    # --- flagged vs clean, PER CLIP: an effect needs a test, not two medians ---------
    if len(CLEAN) >= 5:
        try:
            from scipy.stats import mannwhitneyu
            def _p(i):
                return float(mannwhitneyu(FLAG[:, i], CLEAN[:, i], alternative="two-sided").pvalue)
        except Exception:
            def _p(i):
                return float("nan")
        pooled = np.sqrt((FLAG.var(axis=0, ddof=1) + CLEAN.var(axis=0, ddof=1)) / 2.0)
        pooled = np.maximum(pooled, 1e-9)
        cohen = (np.mean(FLAG, axis=0) - np.mean(CLEAN, axis=0)) / pooled
        pvals = np.array([_p(i) for i in range(len(lo_e))])
        # 40 bands tested at once: Bonferroni is the blunt-but-honest correction
        alpha_bonf = 0.05 / len(lo_e)
        print(f"\nFLAGGED (n={len(FLAG)}) vs NOT-FLAGGED (n={len(CLEAN)}) — per clip")
        print(f"{'band Hz':>15s} {'Δ dB':>7s} {'Cohen d':>8s} {'p':>10s}  sig")
        print("-" * 56)
        for i in range(len(lo_e)):
            unsafe = lo_e[i] >= a.codec_safe_hz and "wav" not in fmts
            s = ("CODEC" if unsafe else
                 "**" if pvals[i] < alpha_bonf else "*" if pvals[i] < 0.05 else "")
            print(f"{lo_e[i]:6.0f}-{hi_e[i]:6.0f} "
                  f"{np.mean(FLAG,axis=0)[i]-np.mean(CLEAN,axis=0)[i]:7.2f} "
                  f"{cohen[i]:8.2f} {pvals[i]:10.2e}  {s}")
        print(f"  ** survives Bonferroni (p < {alpha_bonf:.4f}); * nominal p<0.05 only")
    else:
        cohen = pvals = None
        print(f"\nFLAGGED vs NOT-FLAGGED: only {len(CLEAN)} clean clips — no test attempted.")

    print(f"\n{'band Hz':>15s} {'ref dB':>8s} {'flag dB':>8s} {'Δ dB':>7s} {'Δ sigma':>8s}  note")
    print("-" * 74)
    for i in range(len(lo_e)):
        unsafe = lo_e[i] >= a.codec_safe_hz and "wav" not in fmts
        note = "CODEC-LIMITED" if unsafe else ("  <<<" if abs(dev_sigma[i]) >= 1.0 else "")
        print(f"{lo_e[i]:6.0f}-{hi_e[i]:6.0f} {ref_med[i]:8.2f} {flag_med[i]:8.2f} "
              f"{flag_med[i]-ref_med[i]:7.2f} {dev_sigma[i]:8.2f}  {note}")

    safe = lo_e < a.codec_safe_hz if "wav" not in fmts else np.ones(len(lo_e), bool)
    k = int(np.argmax(np.abs(dev_sigma) * safe))
    print(f"\nLARGEST TRUSTWORTHY DEVIATION: {lo_e[k]:.0f}-{hi_e[k]:.0f} Hz at "
          f"{dev_sigma[k]:+.2f} sigma ({flag_med[k]-ref_med[k]:+.2f} dB)")

    if len(CLEAN) >= 2:
        sep = (np.median(FLAG, axis=0) - np.median(CLEAN, axis=0)) / ref_sigma
        j = int(np.argmax(np.abs(sep) * safe))
        print(f"FLAGGED vs CLEARED (n={len(CLEAN)} -- WEAK, treat as a hint only): "
              f"largest separation {lo_e[j]:.0f}-{hi_e[j]:.0f} Hz at {sep[j]:+.2f} sigma")
    else:
        sep = None
        print("FLAGGED vs CLEARED: too few cleared clips to compare.")

    np.savez(a.out / "contrast.npz", edges=edges, ref_med=ref_med, ref_sigma=ref_sigma,
             flag_med=flag_med, dev_sigma=dev_sigma,
             clean_med=(np.median(CLEAN, axis=0) if len(CLEAN) else np.zeros(0)),
             unrev_med=(np.median(UNREV, axis=0) if len(UNREV) else np.zeros(0)),
             n_ref=len(REF), n_flag=len(FLAG), n_clean=len(CLEAN), n_unrev=len(UNREV))
    json.dump({"purpose": "where in Hz does human-flagged harshness deviate from the "
                          "reference corpus, sigma-normalised per ERB band",
               "reference": a.reference_dir, "labels": str(a.labels),
               "anchor_hz": [ANCHOR_LO, ANCHOR_HI], "codec_safe_hz": a.codec_safe_hz,
               "formats": fmts, "window_pct": [a.start_pct, a.end_pct],
               "window_note": "track-relative 20-80% by default; head-anchored windows "
                              "measure intros/outros, not the arrangement body",
               "n": {"ref": len(REF), "flagged": len(FLAG), "cleared": len(CLEAN),
                     "unreviewed": len(UNREV)},
               "largest_deviation_hz": [float(lo_e[k]), float(hi_e[k])],
               "largest_deviation_sigma": float(dev_sigma[k]),
               "bands_hz": [[float(x), float(y)] for x, y in zip(lo_e, hi_e)],
               "dev_sigma": [float(v) for v in dev_sigma],
               "kim_feedback": None},
              open(a.out / "contrast.json", "w"), indent=2)
    print(f"\n[done] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
