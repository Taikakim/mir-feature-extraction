# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

MIR feature extraction pipeline for conditioning Stable Audio Tools. Extracts 97+ numeric features, 496 classification labels, and 5 AI text descriptions from audio. Processes full mixes and separated stems.

**Hardware:** AMD RX 9070 XT (RDNA4, 16GB VRAM) + Ryzen 9 9900X, ROCm 7.2, PyTorch ROCm nightly.

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
  Track Name.DOWNBEATS    # Downbeat timestamps
```

### Key Subsystems

**Audio I/O:** `core.file_utils.read_audio()` handles all formats including m4a/AAC via pydub/ffmpeg fallback. Use this instead of `sf.read()` directly.

**Music Flamingo:** `music_flamingo.py` (GGUF via llama-mtmd-cli, ~4s/track) or `music_flamingo_transformers.py` (native Python, ~28s/prompt). `music_flamingo_llama_cpp.py` is DEPRECATED. 5 prompt types: `brief`, `technical`, `genre_mood_inst`, `instrumentation`, `very_brief`. Output normalized via `core.text_utils.normalize_music_flamingo_text()`.

**Metadata Lookup:** `tools/track_metadata_lookup.py` searches Spotify and MusicBrainz. Candidates scored by duration match (0.5), year match (0.3), artist similarity (0.2). Controlled by `metadata.use_spotify` and `metadata.use_musicbrainz` config flags.

**Captioning Benchmark:** `tests/poc_lmm_revise.py` -- 5-phase comparison (Flamingo baseline, genre-hint, LLM revision, Qwen2.5-Omni, ensemble). GGUF models in `models/LMM/`. Chat templates: Qwen3=ChatML, GPT-OSS=Harmony, Granite=start_of_role/end_of_role. llama-cpp-python `type_k`/`type_v` must be integers not strings.

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
