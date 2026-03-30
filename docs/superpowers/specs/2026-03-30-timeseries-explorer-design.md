# Timeseries Explorer Integration — Design Spec
**Date:** 2026-03-30
**Status:** Approved

## Overview

Extend the Feature Explorer with timeseries-derived data from TimeseriesDB. Replace `plots/generate_explorer_data.py` with a unified `plots/build_dataset_stats.py` that rebuilds all dataset statistics in one command, including per-track shape vectors, mini display curves, and track-to-track similarity based on timeseries embeddings.

## Goals

- Single command rebuilds all explorer data (scalars + timeseries + similarity)
- `--skip-timeseries` flag for fast scalar-only rebuilds during active processing
- Explorer can show how a track's energy, rhythm, and harmony evolve over time
- DJ-oriented similarity: find tracks that mix in the same key, or that will mix after pitch shift

## Non-goals

- Latent-based analysis (no latents available yet; latent scripts remain unchanged)
- Full-track analysis (still aggregating crops; future work)
- Real-time DB queries from the browser

---

## Architecture

### Script: `plots/build_dataset_stats.py`

Replaces `plots/generate_explorer_data.py` (deleted). Same `--source` and `--output-dir` args.

**Output files:**
```
<output-dir>/
  feature_explorer_data.js        # scalars (existing format + ts-derived scalars)
  feature_explorer_timeseries.js  # mini curves + similarity neighbors
  .ts_cache.npz                   # intermediate cache (not loaded by browser)
```

**CLI:**
```bash
# Full rebuild (default)
python plots/build_dataset_stats.py --source /path/to/goa_crops

# Fast scalar-only rebuild
python plots/build_dataset_stats.py --source /path/to/goa_crops --skip-timeseries

# Re-run similarity only (curves already cached)
python plots/build_dataset_stats.py --source /path/to/goa_crops --skip-scalars --skip-curves
```

**Internal stages:**

| Stage | Flag to skip | Description |
|-------|-------------|-------------|
| 1 — Scalar pass | `--skip-scalars` | Scan `.INFO` files, average per track |
| 2 — Timeseries pass | `--skip-timeseries` or `--skip-curves` | Query TimeseriesDB, compute shape vectors + mini curves |
| 3 — Similarity pass | `--skip-timeseries` or `--skip-similarity` | Cosine similarity, emit neighbor lists |

`--skip-timeseries` skips both Stage 2 and Stage 3. `--skip-curves` skips only Stage 2 (Stage 3 reads from the cache). Stage 2 saves `.ts_cache.npz` so Stage 3 can re-run without re-reading SQLite.

---

## Stage 1: Scalar Pass

Identical logic to the current `generate_explorer_data.py`: scan full-mix `.INFO` files, average numeric features across all crops per track. Outputs the existing `feature_explorer_data.js` variables (`DATA`, `TRACKS`, `FEATURES`, `UNITS`, `DESCS`, `METHODS`, `SPOTIFY`, etc.) plus the new ts-derived scalar features from Stage 2 (appended after Stage 2 completes, or omitted if `--skip-timeseries`).

---

## Stage 2: Timeseries Pass

### Crop → track mapping

TimeseriesDB crop keys follow the pattern `"Artist - Title_N"`. Strip the trailing `_N` to get track name, group all crops. Skip crops whose track name has no scalar result from Stage 1.

### Variable n_steps handling

All 1D ts arrays are interpolated to a canonical **32-step** grid via `np.interp` before any aggregation. This makes crops from different pipeline runs (e.g., `n_steps=256` vs `n_steps=128`) compatible. For `hpcp_ts` (2D, `n_steps × 12`), take the mean over the time axis first to get a single `[12]` chroma vector per crop, then aggregate across crops.

### Per-track computations

**1D ts fields** processed: `rms_energy_bass_ts`, `rms_energy_body_ts`, `rms_energy_mid_ts`, `rms_energy_air_ts`, `spectral_flatness_ts`, `spectral_flux_ts`, `beat_activations_ts`, `onsets_activations_ts`, `tonic_strength_ts`

For each field:
- **Mini curve**: mean across all crops → `[32]` float, L∞-normalised to `[0, 1]` for display
- **Shape scalars** (added to main data JS):
  - `{field}_mean`: mean over all steps × all crops
  - `{field}_std`: std over all steps × all crops

**`hpcp_ts`** (2D):
- **Raw chroma mean**: mean of all crop hpcp vectors → `[12]`; stored as `hpcp_raw_0..11` shape scalars
- **Tonic-rotated chroma**: for each crop, rotate the hpcp mean by `−round(mean(tonic_ts))` steps so tonic lands at index 0; average across crops → `[12]`; stored as `hpcp_rot_0..11` shape scalars

**`tonic_ts`**:
- Dominant tonic = mode of `round(tonic_ts)` across all steps × all crops (integer 0–11)
- Stored as `tonic_sin = sin(2π × tonic / 12)`, `tonic_cos = cos(2π × tonic / 12)` (circular encoding, shape scalars)

### New scalar features added to `feature_explorer_data.js`

~30 new features appended to `FEATURES`, `DATA`, `UNITS`, `DESCS`, `METHODS`:

| Feature set | Count | Description |
|-------------|-------|-------------|
| `{field}_mean` × 9 | 9 | Mean of ts field over time |
| `{field}_std` × 9 | 9 | Std of ts field over time |
| `hpcp_raw_0..11` | 12 | Raw chroma (key-sensitive) |
| `hpcp_rot_0..11` | 12 | Tonic-rotated chroma (key-invariant) |
| `tonic_sin`, `tonic_cos` | 2 | Circular tonic encoding |

All descriptions prefixed with `"[ts]"` so they are visually grouped in the explorer dropdowns.

### Cache format (`.ts_cache.npz`)

```
track_names:    [N] str
shape_vectors:  [N, 44] float32   (raw, un-normalized; Stage 3 z-scores these)
raw_hpcp:       [N, 12] float32
rot_hpcp:       [N, 12] float32
tonic_sincos:   [N, 2]  float32
mini_curves:    [N, 9, 32] float32
```

---

## Stage 3: Similarity Pass

### Embedding vector (44 dims)

Built from Stage 2 shape scalars, z-scored across the dataset. Tracks missing a field get the population mean (handles partially-processed tracks).

```
[mean, std] × 9 1D ts fields  = 18 dims
hpcp_raw_0..11                = 12 dims
hpcp_rot_0..11                = 12 dims
tonic_sin, tonic_cos          =  2 dims
─────────────────────────────────────────
total                         = 44 dims
```

### Three similarity modes

| Mode | Embedding slice | DJ use case |
|------|----------------|-------------|
| **Overall** | All 44 dims | General "sounds like" discovery |
| **Key-locked** | `hpcp_raw` [12] + `tonic_sin/cos` [2] = 14 dims | Mix in same key without pitch shift |
| **Pitch-shift** | `hpcp_rot` [12] only | Mix after pitch shift — compatible harmonic shape |

All modes use cosine similarity. A track is never its own neighbor. Top 20 neighbors stored per track per mode.

### Output: `feature_explorer_timeseries.js`

```js
// Auto-generated — do not edit manually
// Re-generate: python plots/build_dataset_stats.py --source <path>

const TS_CURVES = {
  "Artist - Title": {
    rms_energy_bass_ts:    [32 floats, 0-1],
    rms_energy_body_ts:    [...],
    rms_energy_mid_ts:     [...],
    rms_energy_air_ts:     [...],
    spectral_flatness_ts:  [...],
    spectral_flux_ts:      [...],
    beat_activations_ts:   [...],
    onsets_activations_ts: [...],
    tonic_strength_ts:     [...],
    hpcp_raw:              [12 floats],
    hpcp_rot:              [12 floats],
  },
  ...
};

const TS_NEIGHBORS = {
  "Artist - Title": {
    overall:     [["Other Track", 0.95], ...],  // top 20
    key_locked:  [...],
    pitch_shift: [...],
  },
  ...
};
```

**Estimated file size at 2000 tracks:** ~2.5 MB (TS_CURVES ~1.7 MB + TS_NEIGHBORS ~0.8 MB).

---

## Feature Explorer HTML changes (`plots/feature_explorer.html`)

### Data loading

A second `<script>` tag loads `feature_explorer_timeseries.js` after the main data file. If absent (scalar-only rebuild), the two new panels are not rendered — existing functionality is unaffected.

### Mini-curve panel

Appears as a collapsible side panel when a track is selected. Contents:

- **9 sparkline charts** in a 3×3 grid (Plotly line charts, 32 steps, y-axis 0–1). One per 1D ts field. Thin and compact — labels only, no full axes.
- **Chroma bars** below the sparklines: two horizontal bar charts side by side — raw chroma (left, labelled with pitch classes C–B) and tonic-rotated chroma (right). Visually shows both the key and the harmonic shape.

Panel closes with ✕ or by clicking outside it.

### Similar tracks panel

Appears below the mini-curve panel (same side panel, scrollable). Three tab buttons: **Overall · Key-locked · Pitch-shift**. Each tab lists up to 20 neighbors: track name + cosine similarity score shown as a small inline bar. Clicking a neighbor selects it in the scatter plot and refreshes both panels.

### No other changes

Existing scatter, histogram, radar, filter controls, and track list are untouched. New ts-derived scalar features appear automatically in all existing dropdowns via the extended `FEATURES` array.

---

## Handling incomplete data

- Tracks with **no timeseries entries** in the DB: excluded from TS_CURVES and TS_NEIGHBORS; their ts-derived shape scalars are `null` in DATA (same as any other missing feature).
- Crops with **partial ts fields** (e.g., hpcp_ts absent): use available fields only; missing fields contribute `null` to shape scalars.
- **Dataset still being processed**: script is safe to re-run at any time; later runs with `--skip-scalars --skip-curves` update only the similarity as more tracks complete.

---

## Latent analysis compatibility

The existing `plots/latent_analysis/` scripts (01–04) are unchanged. When latents become available, Script 04 (`04_temporal_correlation.py`) can be updated to read pre-computed ts arrays from TimeseriesDB instead of recomputing from raw audio — the `.ts_cache.npz` intermediate from Stage 2 provides exactly the aligned `[N, 9, 32]` matrix needed. This is future work; no changes in this spec.
