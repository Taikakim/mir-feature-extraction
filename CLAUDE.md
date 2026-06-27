# CLAUDE.md

Guidance for Claude Code when working in this repository.

> **Cross-project coordination (read first).** This repo is one of three in the
> mir + Stable Audio pipeline. Shared facts — data paths, which venv for which
> task, gotchas that span repos — live in `/home/kim/Projects/SAO/MASTER.md`.
> Read it before cross-cutting work, and append to
> `/home/kim/Projects/SAO/WORKLOG.md` when you finish something another repo's
> agent would want to know.
@/home/kim/Projects/SAO/MASTER.md

## Project Overview

MIR feature extraction pipeline for conditioning Stable Audio Tools. Extracts 97+ numeric features, 496 classification labels, and 5 AI text descriptions from audio. Processes full mixes and separated stems.

**Hardware:** AMD RX 9070 XT (RDNA4, 16GB VRAM) + Ryzen 9 9900X, ROCm 7.2, PyTorch ROCm nightly.

See **[TOOLS.md](TOOLS.md)** for the Pitch Shifter GUI and Feature Explorer / Latent Player.

**SA3 control-eval specs (cross-repo):** the control-response evaluator `avp_sa3/sa3_control/onset_eval.py` (gain×density grid → `onset_eval.json`) and the eval-site GUIs in `~/riffer-evals/` (`onset_eval.html` browses every `onset_eval.json`; `disentangle.html` = the onset_per_beat tempo-shortcut page; `mp.html`, `traj.html`) — **full spec in `SAO/MASTER.md` §4.** The eval *measurement* (BPM/onset/groove via essentia/librosa) runs in **mir's venv** (`mir/bin/python`).

## Commands

```bash
# Full pipeline (config-driven)
python src/master_pipeline.py --config config/master_pipeline.yaml

# Test all features on one file
python src/test_all_features.py "/path/to/audio.flac"
python src/test_all_features.py "/path/to/audio.flac" --skip-flamingo --skip-demucs

# Music Flamingo GGUF (fast, recommended)
python src/classification/music_flamingo.py "/path/to/audio.flac" --model Q6_K

# Music Flamingo Transformers (slower, native Python)
python src/classification/music_flamingo_transformers.py "/path/to/audio.flac" --flash-attention

# Stem separation
python src/preprocessing/demucs_sep_optimized.py /path/to/audio/ --batch
python src/preprocessing/bs_roformer_sep.py /path/to/audio/ --batch

# Audio captioning benchmark
python tests/poc_lmm_revise.py "/path/to/audio.flac" --genre "Goa Trance, Psytrance" -v
```

## ROCm Environment

**Central module:** `src/core/rocm_env.py` -- single source of truth for all GPU env vars.
**Config reference:** `config/master_pipeline.yaml` `rocm:` section.

Every GPU-using entry point must call `setup_rocm_env()` BEFORE `import torch`. Shell exports override defaults via `setdefault`.

Key settings:
- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` -- Triton FA2 for AMD
- `PYTORCH_TUNABLEOP_ENABLED=1`, `TUNING=0` -- use pre-tuned GEMM kernels
- `PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512` (PYTORCH_HIP_ALLOC_CONF deprecated in ROCm 7.2)
- `HIP_FORCE_DEV_KERNARG=1` -- prevent CPU/GPU desync
- `TORCH_COMPILE=0` -- buggy with Flash Attention on RDNA
- `MIOPEN_FIND_MODE` NOT set by default (causes freezes on some workloads)
- **Native CK Flash-Attention — 30–100% faster, use it on the SA3/SAT venvs.** On the torch-2.10/2.12
  ROCm venvs (`sat-venv`, `stable-audio-3/.venv`, `sa3-rocm7.13-test`) `flash_attn` is the **CK build**,
  and you MUST `export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE` (before `import torch`) to activate it —
  the `TRUE` above is the slower Triton-AMD path; without `FALSE` you get `No module named 'aiter'` → no
  FA. mir's own rocm-7.2 venv runs Triton FA2 (`TRUE`) and gains CK only after moving to the unified
  rocm-7.13 venv (`SAO/MASTER.md` §3). Full story: MASTER §5 + `SAO/docs/flash-attn-ck-rdna4.md`.

## Architecture

```
src/
  core/             # rocm_env, json_handler (safe_update), file_utils (read_audio), text_utils, batch_utils
  preprocessing/    # file_organizer, demucs_sep_optimized, bs_roformer_sep
  rhythm/           # beat_grid, bpm, onsets, syncopation, complexity, per_stem_rhythm
  spectral/         # spectral_features, multiband_rms
  harmonic/         # chroma, per_stem_harmonic
  timbral/          # audio_commons (8 features), audiobox_aesthetics (4 features), loudness
  classification/   # essentia_features, music_flamingo, music_flamingo_transformers
  transcription/    # drums/adtof.py, drums/drumsep.py
  tools/            # track_metadata_lookup, create_training_crops, statistical_analysis
  crops/            # pipeline.py, feature_extractor.py
tests/              # poc_lmm_revise.py (audio captioning benchmark)
config/             # master_pipeline.yaml (all settings)
models/             # GGUF models (Qwen3-14B, GPT-OSS-20B, Granite-tiny, Music Flamingo)
repos/              # External repos (timbral_models, llama.cpp, ADTOF-pytorch, Qwen2.5-Omni, etc.)
```

### Output Structure

```
Track Name/
  full_mix.flac           # Original audio
  drums.mp3               # Separated stems
  bass.mp3
  other.mp3
  vocals.mp3
  Track Name.INFO         # All features (JSON, append-only)
  Track Name.BEATS_GRID   # Beat timestamps
  Track Name.ONSETS       # Onset timestamps (required for syncopation/complexity)
  Track Name.DOWNBEATS    # Downbeat timestamps
```

### Key Subsystems

**Audio I/O:** `core.file_utils.read_audio()` handles all formats including m4a/AAC via pydub/ffmpeg fallback. Use this instead of `sf.read()` directly.

**Essentia Classification:** `classification/essentia_features.py`. EffNet embeddings via `effnet_onnx.py` (ONNX+MIGraphX). Genre/mood/instrument heads via `gmi_onnx.py` (ONNX+MIGraphX, JIT compiles ~28s on first run). TF `TensorflowPredict2D` is a fallback only. VGGish classifiers also on ONNX+MIGraphX via `vggish_onnx.py`.

**Music Flamingo:** `music_flamingo.py` (GGUF via llama-mtmd-cli, ~4s/track) or `music_flamingo_transformers.py` (native Python, ~28s/prompt). `music_flamingo_llama_cpp.py` is DEPRECATED. Prompt types configurable; `{metadata}` placeholder injects ID3 year/label/genres. Supports `flamingo_sample_probability` to annotate a fraction of crops per run. Unsupported audio formats (m4a, ogg) are auto-converted to WAV via ffmpeg before passing to llama-mtmd-cli. Output normalized via `core.text_utils.normalize_music_flamingo_text()`.

**Granite Revision:** `classification/granite_revision.py`. PASS 4b in `pipeline.py` runs Granite-tiny (llama-cpp-python) to condense Flamingo descriptions into short summaries. Runs independently of `--skip-flamingo` — set `flamingo_revision.enabled: false` in config to disable. Scans the entire crops directory each run so interrupted runs are caught up automatically. `reset()` called before every inference to avoid KV cache cascade failures.

**Metadata Lookup:** `tools/track_metadata_lookup.py` searches Spotify → MusicBrainz → Tidal. Candidates scored by duration match (0.5), year match (0.3), artist similarity (0.2). Tidal looked up via ISRC obtained from Spotify result. Saves: `release_year`, `artists`, `label`, `genres`, `popularity`, `album`, `spotify_id`, `musicbrainz_id`, `isrc`, `tidal_id`, `tidal_url`. Controlled by `metadata.use_spotify`, `metadata.use_musicbrainz`. Per-source skip logic: a track missing `spotify_id` is retried even if `musicbrainz_id` exists (handles Spotify rate limits). AcoustID fingerprinting prefers original file from `paths.input` over output `full_mix`. Tidal session cached as module-level singleton (`_TIDAL_UNAVAILABLE` sentinel prevents re-auth storms). Spotify `/v1/audio-features/` removed Nov 2024 — endpoint disabled. Spotify 429 rate-limit: `search_spotify()` re-raises HTTP 429; `_run_metadata_lookup()` catches this, disables Spotify for the rest of the session (`sp=None`), and logs a warning — affected tracks are retried on next run. Spotify client uses `retries=0` so 429s surface immediately. Verbose Spotify HTTP logging (available_markets) suppressed at WARNING level. Lookups run on **source track folders only** — never on crops. `_migrate_track_features_to_crops()` propagates all `TRANSFERRABLE_FEATURES` (including Tidal/ISRC) to crop INFOs.

**Captioning Benchmark:** `tests/poc_lmm_revise.py` -- 5-phase comparison (Flamingo baseline, genre-hint, LLM revision, Qwen2.5-Omni, ensemble). GGUF models in `models/LMM/`. Chat templates: Qwen3=ChatML, GPT-OSS=Harmony, Granite=start_of_role/end_of_role. llama-cpp-python `type_k`/`type_v` must be integers not strings.

**Onset Detection:** `rhythm/onsets.py`. Combined beat+onset worker `process_folder_rhythm()` in `pipeline_workers.py` handles both `.BEATS_GRID` and `.ONSETS` in one subprocess, loading audio once. Controlled by `rhythm.onsets: true` in config — runs as a catch-up pass independently of `track_analysis: false`. Required by `analyze_syncopation()` and `rhythmic_complexity`.

**Stage 2 Catch-up:** After any Stage 2 branch (skip-to-crop, already-complete, disabled, or never-run), the pipeline runs a unified catch-up: 2b onset detection (`rhythm.onsets`), 2c metadata lookup (`metadata.enabled`), 2d first-stage features. PASS 1 timbral timeout: uses `cf_wait(timeout=300)` instead of `as_completed()` — hung `timbral_models.timbral_reverb()` workers are abandoned after 5 minutes without blocking the pipeline.

**Statistical Analysis:** `tools/statistical_analysis.py`. Scans `.INFO` files recursively. Basic stats + outlier detection per feature. `--per-track` aggregates crops to one value per track. `--top N --key bpm` queries ranked values. Feature selection: `--vif`, `--pca`, `--cluster`, `--mi`, `--feature-select` (all). Plots (heatmap, dendrogram, scree, VIF bar) via `--plots-dir`.

## TimeseriesDB

Time-coded feature arrays (one value per analysis step across the crop duration) are stored in a SQLite database at `data/timeseries.db`, **not** in `.INFO` sidecar files. This keeps companion JSONs compact (~3 KB instead of ~130 KB).

**Fields stored in DB** (all `*_ts` numeric arrays):
- `rms_energy_{bass,body,mid,air}_ts` — per-band energy over time (shape `(n_steps,)`)
- `spectral_{flatness,flux,skewness,kurtosis}_ts`
- `beat_activations_ts`, `downbeat_activations_ts`, `onsets_activations_ts`
- `hpcp_ts` — chroma over time (shape `(n_steps, 12)`)
- `tonic_ts`, `tonic_strength_ts`
- Optional timbral `_ts` fields when `timeseries_timbral: true`

**API:**
```python
from core.timeseries_db import TimeseriesDB

db = TimeseriesDB.open()             # opens data/timeseries.db
arrays = db.get("Artist - Title_0")  # {field: np.ndarray} or None
db.has("Artist - Title_0")           # bool
db.count()                           # total entries
```

**Pipeline integration:** `pipeline.py` writes to TimeseriesDB instead of `results` dict when `skip_timeseries: false`. The `existing_keys` / sentinel pattern is bypassed — the DB `has()` check is the source of truth for resume logic.

**Encoding:** `encode_dataset.py` already strips all list fields from companion JSONs (the `padding_mask` exception is for SAT training). TimeseriesDB is for analysis tools and future conditioning only — the encoder does not read from it.

## Whole-Track Timeseries (per-track npz, for variable-offset crops)

The per-crop TimeseriesDB above is keyed by `<track>_<crop>` and only works when the crop-to-track mapping is fixed at MIR time. For workflows that need **variable training-crop offsets** (e.g. SA3 LoRA fine-tuning with beat-aligned crops chosen at encode time, or future LatCH training against arbitrary windows), a parallel **whole-track** store exists:

- **Producer:** `src/spectral/whole_track_timeseries.py`
  ```bash
  python src/spectral/whole_track_timeseries.py <Goa_Separated_root> --workers 4
  ```
  Walks each track folder, extracts 20 fields at **100 Hz over the whole track** (`madmom` beat/downbeat activations, `librosa` onset envelopes per stem, multiband/per-stem RMS, spectral, HPCP), writes one `<track>.TIMESERIES.npz` per source track. Resumable (skips existing). Driven by chunked-fresh-pool workers (see `--chunk-size`, `--chunk-timeout`).

- **Output:** `/run/media/kim/Lehto/timeseries/<track>.TIMESERIES.npz` — currently 4461 npz, **21 GB total**. Each npz contains:
  - 1-D fields shape `(N_frames,)` where `N_frames ≈ duration_sec × 100`
  - `hpcp_ts` shape `(N_frames, 12)`
  - `__meta__` JSON string with `frame_rate`, `n_frames`, `duration`, etc.

- **Consumer (cropper/resampler):** `/home/kim/Projects/SAO/stable-audio-tools/scripts/whole_track_target_source.py`
  ```python
  from whole_track_target_source import resample_axis0, _read_npz
  arrays, meta = _read_npz("/run/media/kim/Lehto/timeseries/<track>.TIMESERIES.npz")
  # Slice arrays[field][s:e] then resample_axis0(win, target_n_frames)
  ```
  `WholeTrackTargetSource.get(crop_key, feature, start_time, end_time, n_frames)` packages this for LatCH dataloaders.

- **Use cases**: LatCH-head training against arbitrary crop windows (the SAT trainer reads via the consumer above); per-crop timeseries companions for SA3 LoRA latents (sliced to T=4096 alongside each `.npy`, see `/tmp/sa3_encode_from_manifest.py`).

When to choose which store:
- **TimeseriesDB (per crop)**: fixed crop-to-track mapping, queried by crop key — used by the legacy LatCH dataset.
- **Whole-track npz (per track)**: arbitrary windows extracted at consumer time — used by anything that needs to slice a specific `[start_sec, end_sec]` and resample to a target frame count.

## Development Rules

### JSON handling

**Always** use `safe_update()` for atomic merge. **Never** use `write_info(merge=False)`.

```python
from core.json_handler import safe_update, get_info_path
safe_update(get_info_path(audio_path), {'feature': value})
```

### New features

1. Add to `FEATURE_RANGES` in `src/core/common.py`
2. Use `safe_update()` to save
3. Test with `test_all_features.py`

### ROCm env in new scripts

Any script that imports torch must start with:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # adjust to reach src/
from core.rocm_env import setup_rocm_env
setup_rocm_env()
import torch  # now safe
```

### Other rules

- Load GPU models once for batch processing, not per-file
- Never move original files -- copy to output dir
- Never delete `.INFO` files -- only merge
- Use `FileLock` for concurrent batch processing

## Known Issues

- **INT8/INT4 quantization:** Non-functional on ROCm. Use bfloat16 + Flash Attention 2.
- **torch.compile:** Fails with Demucs (complex FFT), Music Flamingo (Dynamo+accelerate), and FA on RDNA. Keep `TORCH_COMPILE=0`.
- **GGUF POOL_1D warning:** Cosmetic on RDNA4 (gfx1201).
- **numba/numpy:** Pin numpy <2.4.
- **Qwen2.5-Omni AWQ:** Requires patched modeling file (RoPE fix) in `repos/Qwen2.5-Omni/low-VRAM-mode/`.
- **Madmom:** CPU-only (NumPy/Cython neural networks, no GPU support).
- **Audiobox Aesthetics ONNX:** Not exportable — WavLM attention uses non-contiguous tensor views that break both dynamo and legacy tracer. Runs as PyTorch ROCm.
- **GMI ONNX first-run JIT:** Genre model takes ~29s to JIT compile kernels on first inference per process. Mood/instrument ~0.4-0.6s. Subsequent calls are <1ms.
- **Spotify audio features:** `/v1/audio-features/` returns 403 for all standard API apps since Nov 2024. Disabled in pipeline (`fetch_audio_features_flag=False`). The endpoint is gone permanently.
- **timbral_models hang:** `timbral_reverb()` can loop forever on pathological audio. PASS 1 uses `cf_wait(timeout=300)` — hung crops are skipped and retried next run.
