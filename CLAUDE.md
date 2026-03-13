# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

MIR feature extraction pipeline for conditioning Stable Audio Tools. Extracts 97+ numeric features, 496 classification labels, and 5 AI text descriptions from audio. Processes full mixes and separated stems.

**Hardware:** AMD RX 9070 XT (RDNA4, 16GB VRAM) + Ryzen 9 9900X, ROCm 7.2, PyTorch ROCm nightly.

See **[TOOLS.md](TOOLS.md)** for the Pitch Shifter GUI and Feature Explorer / Latent Player.

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
