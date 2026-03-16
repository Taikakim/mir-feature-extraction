# HPCP + TIV + Tonic: Parallel Harmonic Feature Extraction

**Date:** 2026-03-16
**Status:** Approved

## Overview

Add a second, parallel harmonic feature extraction pipeline alongside the existing CQT chroma. The new pipeline uses Essentia's HPCP (Harmonic Pitch Class Profile) for accurate pitch-class extraction and projects it into the Tonal Interval Vector (TIV) space for semantically rich global conditioning. Tonic estimation is included as a free by-product of the HPCP computation. The existing CQT chroma is also fixed to use the correct stem combination.

## Motivation

The existing `chroma_0`–`chroma_11` (CQT via librosa) has three known weaknesses:

1. **Overtone contamination** — direct bin folding maps harmonic series energy into false pitch classes
2. **Wrong stem** — the hot path uses `other` only, losing bass fundamental frequencies that anchor harmonic root
3. **Ambiguous global average** — averaging raw pitch classes over 10s produces a tonal-mass blur that the model can't distinguish from a specific chord

The TIV solves (3) by projecting into a space where averaging yields meaningful semantic measurements (tension, diatonicity, dissonance). HPCP solves (1) via peak-picking + harmonic summation. Bass+other stem selection solves (2).

## New Features (27 total)

### HPCP — `hpcp_0` through `hpcp_11`
- **Algorithm:** Essentia `SpectralPeaks` → `HPCP`
- **Config:** size=12, harmonics=8, minFreq=40Hz, maxFreq=5000Hz, weightType=cosine, nonLinear=True
- **Normalization:** L∞ per frame (dominant bin = 1.0), averaged across frames, L∞ again
- **Range:** [0.0, 1.0] each
- **Encodes:** Absolute pitch-class fingerprint — directly exposes modal identity (e.g. Phrygian b♭2 shows as elevated `hpcp_1` relative to `hpcp_2`)

### TIV — `tiv_0` through `tiv_11`
- **Algorithm:** DFT of L1-normalized averaged HPCP, perceptual weights applied, flattened Re/Im
- **Perceptual weights:** `w_a = [3, 8, 11.5, 15, 14.5, 7.5]` (Harte et al. 2006)
- **Layout:** `[Re(k1), Im(k1), Re(k2), Im(k2), ..., Re(k6), Im(k6)]`
- **Normalization:** L2 (unit-norm hypersphere)
- **Range:** [-1.0, 1.0] each
- **Semantic mapping:**

| Index | Component | Encodes |
|-------|-----------|---------|
| tiv_0, tiv_1 | k=1 | Chromaticity / circle-of-fifths alignment |
| tiv_2, tiv_3 | k=2 | Tritone symmetry / octatonicity |
| tiv_4, tiv_5 | k=3 | Major thirds / augmented structures |
| tiv_6, tiv_7 | k=4 | Dissonance / diminished tension |
| tiv_8, tiv_9 | k=5 | Diatonicity |
| tiv_10, tiv_11 | k=6 | Whole-toneness |

### Tonic — `tonic`, `tonic_strength`, `tonic_scale`
- **Algorithm:** Essentia `Key` with `edmm` profile (EDM-calibrated)
- **Input:** Averaged HPCP (free — already computed)
- `tonic`: pitch class 0–11 (0=C, 1=C#, …, 11=B), range [0.0, 11.0]
- `tonic_strength`: confidence 0–1, range [0.0, 1.0]
- `tonic_scale`: string label ("major"/"minor"), stored in INFO, not in FEATURE_RANGES

## Robustness

All three representations are volume-invariant: HPCP L∞-normalizes per frame; TIV L1-normalizes before DFT then L2-normalizes the output; tonic estimation operates on the normalized HPCP distribution.

## New Module: `src/harmonic/hpcp_tiv.py`

```
mix_bass_other_stems(folder_path)
    → (audio, sr, used_stems: bool)
    Falls back to full mix with logger.warning() if no stems found.

compute_hpcp_frames(audio, sr, frame_size=4096, hop_size=2048)
    → np.ndarray (n_frames, 12)
    Essentia pipeline: FrameGenerator → Windowing(hann) → Spectrum
    → SpectralPeaks → HPCP

compute_tiv(hpcp_avg)
    → np.ndarray (12,)  L2-normalized
    L1-normalize → DFT → weight k=1..6 → flatten Re/Im → L2-normalize

analyze_hpcp_tiv(audio_path, use_stems=True, audio=None, sr=None)
    → Dict[str, float]  (hpcp_0–11, tiv_0–11, tonic, tonic_strength, tonic_scale)

batch_analyze_hpcp_tiv(root_directory, overwrite=False)
    → stats dict

CLI __main__ block (mirrors chroma.py)
```

Frame size 4096 / hop 2048 matches CQT chroma for consistency.

## Changes to Existing Files

### `src/core/common.py`
Add 26 numeric keys to `FEATURE_RANGES`:
- `hpcp_0`–`hpcp_11`: min=0.0, max=1.0
- `tiv_0`–`tiv_11`: min=-1.0, max=1.0
- `tonic`: min=0.0, max=11.0
- `tonic_strength`: min=0.0, max=1.0

(`tonic_scale` is a string label — not in FEATURE_RANGES.)

### `src/harmonic/chroma.py`
`mix_harmonic_stems()`: change `['bass', 'other', 'vocals']` → `['bass', 'other']`.
Rationale: vocal pitch instability (vibrato, portamento) smears chroma bins.

### `src/crops/feature_extractor.py`

**Fix `_extract_chroma()` hot path:**
When `preloaded_stems` contains both `bass` and `other`, mix them (same padding logic as `mix_harmonic_stems`) before passing to `analyze_chroma()`. Falls back to `other`-only if bass not pre-loaded.

**New `_extract_hpcp_tiv()` method:**
- Skip guard: checks `hpcp_0` in existing (same pattern as chroma)
- Calls `analyze_hpcp_tiv(crop_path, use_stems=True)`
- Stems resolved internally from `crop_path.parent` — loads bass+other from disk itself
- Called as step 5b, immediately after `_extract_chroma()`

No changes to `FeatureExtractor.__init__` or config flags.

## Stem Selection Rationale

| Stems | Harmonic utility | Verdict |
|-------|-----------------|---------|
| other only | Missing bass root | ❌ Current (broken) |
| bass + other + vocals | Vocals smear bins | ❌ Current chroma.py |
| bass + other | Root + chords, clean | ✅ New standard |
| full mix | Drum transients corrupt | Fallback only |

## Deprecation Path

CQT chroma (`chroma_0`–`chroma_11`) is kept intact. Once the model has been fine-tuned on both representations and HPCP/TIV is confirmed superior, CQT chroma can be deprecated in a follow-up sprint by removing the `_extract_chroma()` call and adding a migration note to existing INFO files.
