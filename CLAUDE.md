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
python src/classification/music_flamingo_gguf.py "/path/to/audio.flac" --model Q6_K

# Music Flamingo Transformers (slower, native Python)
python src/classification/music_flamingo_transformers.py "/path/to/audio.flac" --flash-attention

# Stem separation
python src/preprocessing/demucs_sep_optimized.py /path/to/audio/ --batch
python src/preprocessing/bs_roformer_sep.py /path/to/audio/ --batch
```

## ROCm Environment

**Central module:** `src/core/rocm_env.py` -- single source of truth for all GPU env vars.
**Config reference:** `config/master_pipeline.yaml` `rocm:` section.

Every GPU-using entry point must call `setup_rocm_env()` BEFORE `import torch`. Shell exports override defaults via `setdefault`.

Key settings:
- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` -- Triton FA2 for AMD
- `PYTORCH_TUNABLEOP_ENABLED=1`, `TUNING=0` -- use pre-tuned GEMM kernels
- `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512`
- `HIP_FORCE_DEV_KERNARG=1` -- prevent CPU/GPU desync
- `TORCH_COMPILE=0` -- buggy with Flash Attention on RDNA
- `MIOPEN_FIND_MODE` NOT set by default (causes freezes on some workloads)

## Architecture

```
src/
  core/             # rocm_env, json_handler (safe_update), file_utils, text_utils, batch_utils
  preprocessing/    # file_organizer, demucs_sep_optimized, bs_roformer_sep, loudness
  rhythm/           # beat_grid, bpm, onsets, syncopation, complexity, per_stem_rhythm
  spectral/         # spectral_features, multiband_rms
  harmonic/         # chroma, per_stem_harmonic
  timbral/          # audio_commons (8 features), audiobox_aesthetics (4 features)
  classification/   # essentia_features, music_flamingo_gguf, music_flamingo_transformers
  transcription/    # drums/adtof.py, drums/drumsep.py
  tools/            # track_metadata_lookup, create_training_crops, statistical_analysis
  crops/            # pipeline.py, feature_extractor.py
config/             # master_pipeline.yaml (all settings)
repos/              # External repos (timbral_models, llama.cpp, ADTOF-pytorch, etc.)
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

### Music Flamingo

| Method | Speed | VRAM | File |
|--------|-------|------|------|
| GGUF (recommended) | ~4s/track | 5-9 GB | `music_flamingo_gguf.py` |
| Transformers | ~28s/prompt | 13 GB | `music_flamingo_transformers.py` |

5 prompt types: `full`, `technical`, `genre_mood`, `instrumentation`, `structure`.
All output normalized for T5 tokenizer via `core.text_utils.normalize_music_flamingo_text()`.

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

### Model loading

Load GPU models once for batch processing, not per-file.

### File organization

- Never move original files -- copy to output dir
- Never delete `.INFO` files -- only merge
- Use `FileLock` for concurrent batch processing

## Known Issues

- **INT8/INT4 quantization:** Non-functional on ROCm. Use bfloat16 + Flash Attention 2.
- **torch.compile + Demucs:** Fails on ROCm (complex FFT dtype errors). Use SDPA patch instead.
- **torch.compile + FA:** Buggy on RDNA. Keep `TORCH_COMPILE=0`.
- **GGUF POOL_1D warning:** Cosmetic on RDNA4 (gfx1201). Model works correctly.
- **numba/numpy:** Pin numpy <2.4. Use soundfile instead of librosa for duration.
- **MIOPEN_FIND_MODE:** Can cause freezes. Not set by default; override in shell if needed.
- **`music_flamingo_llama_cpp.py`** is DEPRECATED (broken — never passes audio). All code uses `music_flamingo.py` (CLI subprocess) instead.
