# HPCP + TIV + Tonic: Parallel Harmonic Feature Extraction

**Date:** 2026-03-16
**Status:** Approved

## Overview

Add a second, parallel harmonic feature extraction pipeline alongside the existing CQT chroma. The new pipeline uses Essentia's HPCP (Harmonic Pitch Class Profile) for accurate pitch-class extraction and projects it into the Tonal Interval Vector (TIV) space for semantically rich global conditioning. Tonic estimation is included as a free by-product of the HPCP computation. The existing CQT chroma is also fixed to use the correct stem combination.

New outputs: **27 total — 26 numeric (stored in FEATURE_RANGES) + 1 string label (`tonic_scale`, stored in INFO only).**

## Motivation

The existing `chroma_0`–`chroma_11` (CQT via librosa) has three known weaknesses:

1. **Overtone contamination** — direct bin folding maps harmonic series energy into false pitch classes
2. **Wrong stem** — the hot path uses `other` only, losing bass fundamental frequencies that anchor harmonic root
3. **Ambiguous global average** — averaging raw pitch classes over 10s produces a tonal-mass blur the model can't distinguish from a specific chord

The TIV solves (3) by projecting into a space where averaging yields meaningful semantic measurements (tension, diatonicity, dissonance). HPCP solves (1) via peak-picking + harmonic summation. Bass+other stem selection solves (2).

## New Features (26 numeric + 1 label)

### HPCP — `hpcp_0` through `hpcp_11`
- **Algorithm:** Essentia `FrameGenerator` → `Windowing(hann)` → `Spectrum` → `SpectralPeaks` → `HPCP`
- **HPCP constructor params:** `size=12, harmonics=8, minFrequency=40.0, maxFrequency=5000.0, weightType='cosine', nonLinear=True, normalized='unitMax'`
- **Normalization:** Essentia's own `normalized='unitMax'` handles per-frame L∞ normalization (dominant bin = 1.0). No additional per-frame normalization step needed — Essentia's internal normalization is used directly. After averaging frames, apply one final L∞ normalize to the mean vector.
- **Range:** [0.0, 1.0] each — safe clamp range
- **Encodes:** Absolute pitch-class fingerprint — directly exposes modal identity (e.g. Phrygian b♭2 shows as elevated `hpcp_1` relative to `hpcp_2`)

### TIV — `tiv_0` through `tiv_11`
- **Algorithm:** DFT of L1-normalized averaged HPCP, perceptual weights applied, flattened Re/Im
- **Perceptual weights:** `w_a = [3, 8, 11.5, 15, 14.5, 7.5]` (Harte et al. 2006)
- **Layout:** `[Re(k1), Im(k1), Re(k2), Im(k2), ..., Re(k6), Im(k6)]`
- **Normalization:** L2 to unit norm. Range [-1.0, 1.0] is a **safe clamp range** — any component of a unit-norm 12D vector is mathematically bounded by ±1, though typical values are much smaller. No assertions should be added beyond the standard `clamp_feature_value` call.
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
- **Algorithm:** Essentia `Key` with `profileType='edmm'` (EDM-calibrated)
- **Requires:** Essentia ≥ 2.1b6.dev1034 (the release that introduced `edmm`). If `edmm` raises `ValueError` at construction, fall back to `profileType='edma'` with `logger.warning()`.
- **Input:** Averaged HPCP vector (free — already computed)
- **`tonic`:** Convert Essentia's string output (e.g. `"C"`, `"C#"`) to integer via `PITCH_CLASSES.index(key_str)`. If the returned string is not in `PITCH_CLASSES`, log a warning and store `0`. Range [0.0, 11.0].
- **`tonic_strength`:** Essentia's `strength` output, range [0.0, 1.0]
- **`tonic_scale`:** Essentia's `scale` string (e.g. `"major"`, `"minor"`). String label stored directly in INFO, **not** in `FEATURE_RANGES`.

## Robustness

All three representations are volume-invariant: HPCP uses `normalized='unitMax'` per frame; TIV L1-normalizes before DFT then L2-normalizes the output; tonic estimation operates on the normalized HPCP distribution.

**Silent / all-zero audio:** If the bass+other mix is silent (all-zero HPCP after averaging), the L1-normalization step would divide by zero. Guard: if `hpcp_avg.sum() == 0`, return all-zero `hpcp` and `tiv` arrays, `tonic=0`, `tonic_strength=0.0`, `tonic_scale="unknown"`. Log a warning. (Mirrors `calculate_chroma_cqt`'s uniform-distribution fallback.)

## New Module: `src/harmonic/hpcp_tiv.py`

```
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

mix_bass_other_stems(folder_path)
    → (audio: np.ndarray, sr: int, used_stems: bool)
    Loads bass + other stems, mixes to mono (length-pad shorter with zeros,
    sum, peak-normalize to 0.95). Falls back to full mix with logger.warning()
    if neither stem exists. Same padding logic as chroma.mix_harmonic_stems().

compute_hpcp_frames(audio, sr, frame_size=4096, hop_size=2048)
    → np.ndarray shape (n_frames, 12)
    Essentia pipeline: FrameGenerator → Windowing(hann, size=frame_size)
    → Spectrum(size=frame_size) → SpectralPeaks(maxPeaks=100,
    magnitudeThreshold=0.00001, sampleRate=sr, orderBy='magnitude')
    → HPCP(size=12, harmonics=8, minFrequency=40.0, maxFrequency=5000.0,
           weightType='cosine', nonLinear=True, normalized='unitMax',
           sampleRate=sr)
    Returns matrix of per-frame HPCP vectors; empty frames yield zero rows.

compute_tiv(hpcp_avg)
    → np.ndarray (12,)  L2-normalized
    1. If sum == 0: return np.zeros(12)
    2. L1-normalize: c = hpcp_avg / hpcp_avg.sum()
    3. DFT: dft = np.fft.fft(c)
    4. Apply perceptual weights: T(k) = w_a[k-1] * dft[k] for k in 1..6
    5. Flatten: [Re(k1), Im(k1), ..., Re(k6), Im(k6)]
    6. L2-normalize; if norm == 0 return zeros

analyze_hpcp_tiv(audio_path, use_stems=True, audio=None, sr=None)
    → Dict[str, float | str]
    When audio is provided, stem loading is skipped entirely; use_stems ignored.
    The provided signal is used directly (enables preloaded-stem fast path).
    Returns: hpcp_0–11, tiv_0–11, tonic, tonic_strength, tonic_scale

batch_analyze_hpcp_tiv(root_directory, overwrite=False)
    → stats dict (mirrors batch_analyze_chroma)

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

`mix_harmonic_stems()`:
- Change `harmonic_stems = ['bass', 'other', 'vocals']` → `['bass', 'other']`
- Update module docstring (line 10): remove "vocals" from stem list description
- Update function docstring (line ~45): change "Mix bass + other + vocals stems" → "Mix bass + other stems"
- Update log message (line ~62): keep the dynamic `{', '.join(available_stems)}` interpolation; update the suffix to `"(excluding drums and vocals)"` → `logger.info(f"Using harmonic stems: {', '.join(available_stems)} (excluding drums and vocals)")`

Rationale: vocal pitch instability (vibrato, portamento) smears chroma bins; bass provides harmonic root.

### `src/crops/feature_extractor.py`

**Fix `_extract_chroma()` hot path:**

When `preloaded_stems` contains both `'bass'` and `'other'`, mix them inline before passing to `analyze_chroma()`. Use the same pad-then-add logic from `chroma.mix_harmonic_stems()` (lines 79–83):

```python
if preloaded_stems and 'other' in preloaded_stems:
    other_audio, other_sr = preloaded_stems['other']
    if other_audio.ndim > 1:
        other_audio = other_audio.mean(axis=1)
    if 'bass' in preloaded_stems:
        bass_audio, _ = preloaded_stems['bass']
        if bass_audio.ndim > 1:
            bass_audio = bass_audio.mean(axis=1)
        # Pad shorter to same length, sum, peak-normalize
        max_len = max(len(other_audio), len(bass_audio))
        other_audio = np.pad(other_audio, (0, max_len - len(other_audio)))
        bass_audio  = np.pad(bass_audio,  (0, max_len - len(bass_audio)))
        mixed = other_audio + bass_audio
        max_val = np.abs(mixed).max()
        if max_val > 0:
            mixed = mixed / max_val * 0.95
        audio_for_chroma = mixed.astype(np.float32)
    else:
        audio_for_chroma = other_audio.astype(np.float32)
    # NOTE: analyze_chroma call is outside/after the bass if/else block —
    # audio_for_chroma is always set by this point (either mixed or other-only)
    chroma = analyze_chroma(stems['other'], use_stems=False,
                            audio=audio_for_chroma, sr=other_sr)
```

No shared helper needed — the logic is simple enough inline and mirrors `mix_harmonic_stems`.

**New `_extract_hpcp_tiv()` method:**
- Signature: `_extract_hpcp_tiv(self, crop_path, stems, results, timings, existing, overwrite, preloaded_stems=None)`
- Skip guard: checks `'hpcp_0' not in existing or overwrite`
- When `preloaded_stems` has both `'bass'` and `'other'`: mix inline (same logic as above), pass `audio=mixed, sr=sr` to `analyze_hpcp_tiv(crop_path, audio=mixed, sr=sr)`. Pass `use_stems=False` defensively (convention: whenever `audio` is provided, pass `use_stems=False` to guard against future implementation drift, even though `analyze_hpcp_tiv` ignores `use_stems` when `audio` is present).
- When only `'other'` in preloaded_stems: pass `audio=other_audio, sr=sr` to `analyze_hpcp_tiv(crop_path, use_stems=False, audio=other_audio, sr=sr)`
- When neither: call `analyze_hpcp_tiv(crop_path, use_stems=True)` (loads from disk)
- Called as step 5b, immediately after `_extract_chroma()`

**Call site (step 5b):**
```python
# 5b. HPCP + TIV + tonic
self._extract_hpcp_tiv(crop_path, stems, results, timings, existing, overwrite,
                       preloaded_stems=preloaded_stems)
```

## Stem Selection Rationale

| Stems | Harmonic utility | Verdict |
|-------|-----------------|---------|
| other only | Missing bass root | ❌ Current (broken) |
| bass + other + vocals | Vocals smear bins | ❌ Current chroma.py |
| bass + other | Root + chords, clean | ✅ New standard |
| full mix | Drum transients corrupt | Fallback only |

## Deprecation Path

CQT chroma (`chroma_0`–`chroma_11`) is kept intact. Once the model has been fine-tuned on both representations and HPCP/TIV is confirmed superior, CQT chroma can be deprecated in a follow-up sprint by removing the `_extract_chroma()` call and adding a migration note to existing INFO files.
