# Architecture

MIR feature extraction pipeline for conditioning Stable Audio Tools on a Goa/Psytrance dataset.
Audio goes in; structured JSON features, separated stems, training crops, and VAE latents come out.

---

## Why this structure

The project has three largely independent concerns, kept separate so each can evolve without touching the others:

- **Extraction pipeline** (`src/`) — heavyweight GPU work: stem separation, feature extraction, AI annotation. Lives in `src/` so it can be imported cleanly by entry points without pulling in UI dependencies.
- **Dataset tooling** (`scripts/`) — one-shot and recurring maintenance operations that act on the dataset on disk. Kept out of `src/` because they are not part of the pipeline API; they are run by a human when something needs fixing or reorganising.
- **Analysis & exploration** (`plots/`) — lightweight Python/JS tools for understanding what the pipeline produced. Depends on `src/` for data loading but never the other way around.

---

## Top-level layout

```
src/                    Feature extraction pipeline — importable library + entry points
scripts/                Dataset maintenance utilities (run manually, not imported)
plots/
  explorer/             Unified MIR Explorer — Dash app (port 7895)
  latent_analysis/      Latent-space analysis scripts + pre-computed NPZ output
config/                 Pipeline configuration (master_pipeline.yaml is the single source of truth)
tests/                  Automated test suite
data/                   SQLite databases (timeseries.db — frame-level feature arrays)
models/                 GGUF model weights (downloaded separately, not in git)
_quarantine/            Superseded code preserved for reference, not part of any workflow
```

---

## src/ — the pipeline

```
src/
  master_pipeline.py          Entry point: orchestrates all passes for a dataset run
  core/
    rocm_env.py               ROCm/HIP env setup — must be called before torch import
    pipeline_ui.py            Rich TUI dashboard (alternate buffer, like btop)
    json_handler.py           Atomic .INFO sidecar merge (safe_update — never overwrite)
    file_utils.py             Universal audio loader with ffmpeg fallback
    timeseries_db.py          SQLite store for frame-level feature arrays
    common.py                 FEATURE_RANGES, shared constants
    benchmark.py              Runtime timing collector — HTML report generation
  preprocessing/              Stem separation (Demucs, BS-RoFormer)
  rhythm/                     BPM, beat grid, onsets, syncopation, complexity
  spectral/                   Spectral features, multiband RMS, timeseries features
  harmonic/                   Chroma, HPCP, per-stem harmonic analysis
  timbral/                    Audio Commons timbral models, Audiobox Aesthetics, loudness
  classification/             Essentia (ONNX), Music Flamingo (GGUF), Granite revision
  transcription/              Drum transcription (ADTOF, DrumSep)
  tools/                      Dataset-level tools: crop creation, metadata lookup, stats
  crops/                      Crop-level pipeline: feature extraction, encoding
```

**Data flow:** `master_pipeline.py` → preprocessing → per-track feature passes → `crops/pipeline.py` → per-crop feature extraction → `crops/feature_extractor.py` → `.INFO` sidecars + `timeseries.db`.

Each feature module accepts an optional pre-loaded `(audio, sr)` pair to avoid redundant disk reads. GPU models are loaded once per process, not per file.

---

## scripts/ — dataset maintenance

Standalone scripts run by hand when the dataset needs surgery. They are not imported by the pipeline and have no shared API — each is self-contained.

| Script | When to run |
|--------|-------------|
| `latent_server.py` | Always — VAE decode server backing the Explorer (port 7891) |
| `latent_crossfader.py` | Library used by latent_server for slerp/lerp interpolation |
| `latent_shape_server.py` | Optional — 3D latent shape API for shape explorer |
| `delete_compressed_stems.py` | Before re-running stem separation on upgraded sources |
| `delete_encoded_latents.py` | Before re-encoding with a new VAE checkpoint |
| `delete_orphaned_latents.py` | Periodic cleanup after track deletions |
| `prepare_crop_reingest.py` | Before re-cropping (strips audio, preserves .INFO features) |
| `replace_compressed_sources.py` | When switching lossy sources to clean FLAC |
| `migrate_timeseries_to_db.py` | One-time: moved timeseries from .INFO into timeseries.db |
| `rename_*_strip_numbers.py` | Dataset reorganisation: strip numeric prefixes from filenames |
| `repair_multiband_rms.py` | Targeted repair of corrupted feature values |
| `download_essentia_models.py` | First-time setup: fetch Essentia TF model weights |
| `verify_features.py` | QA pass: check extraction completeness across the dataset |
| `verify_flamingo_config.py` | Sanity-check Music Flamingo config before a long run |

---

## plots/ — analysis and exploration

```
plots/
  explorer/             Unified MIR Explorer (Dash, port 7895)
    app.py              Entry point — run this to start the explorer
    data.py             Data loading: .INFO features, NPZ analysis, tracks.csv index
    audio.py            URL builders for the VAE decode/crossfade API
    latch.py            LATCH feature ordering for the Analysis tab
    tabs/
      dataset.py        Scatter plot, parallel coords, histogram, radar, similarity search
      analysis.py       Latent correlation heatmaps, PCA, temporal analysis
      viewer.py         Side-by-side waveform and feature comparison
    assets/
      player.js         Web Audio API player — VAE-decoded audio, crossfade, autoplay
      style.css
  latent_analysis/      Pre-explorer latent analysis suite
    01_aggregate_correlation.py   Latent dim × MIR feature correlations → NPZ
    02_pca_analysis.py            PCA of latent space → NPZ
    03_latent_xcorr.py            Cross-correlation between latent dims → NPZ
    04_temporal_correlation.py    Frame-level temporal analysis → NPZ
    data/                         Pre-computed NPZ output (gitignored, regenerate with scripts above)
```

The Explorer's Analysis tab reads the pre-computed NPZ files from `plots/latent_analysis/data/`. Regenerate them by re-running scripts 01–04 against the current dataset.

---

## Output schema (per track)

```
<Track Name>/
  full_mix.flac           Source audio (never modified)
  drums.mp3  bass.mp3  other.mp3  vocals.mp3   Separated stems
  <Track Name>.INFO       All features — append-only JSON sidecar (safe_update only)
  <Track Name>.BEATS_GRID Beat timestamps
  <Track Name>.ONSETS     Onset timestamps
  <Track Name>.DOWNBEATS  Downbeat timestamps
  crops/
    <Track Name>_<N>.flac  4-bar training crops
    <Track Name>_<N>.INFO  Per-crop features (subset of track .INFO)
```

`data/timeseries.db` stores frame-level arrays separately to keep per-crop JSON compact (~3 KB vs ~130 KB).

---

## Key design decisions

**Atomic JSON writes.** `.INFO` files are append-only via `safe_update()` (file-locked JSON merge). Direct writes (`write_info(merge=False)`) are forbidden — they would silently erase features written by concurrent passes.

**ROCm before torch.** Every entry point calls `setup_rocm_env()` before `import torch`. The function uses `os.environ.setdefault()` so shell-level overrides still win.

**ONNX+MIGraphX for classifiers.** Essentia EffNet and GMI heads use ONNX+MIGraphX (JIT-compiled on first run per process, ~29 s for the genre model). TensorFlow is a fallback only.

**Timeseries in SQLite.** Frame-level arrays (`*_ts` fields) live in `data/timeseries.db`, not in `.INFO`. This keeps the JSON sidecars compact and fast to load in bulk.

**GPU models loaded once.** Within a batch pass, models are loaded once and reused across all tracks/crops. Loading per-file would dominate runtime.
