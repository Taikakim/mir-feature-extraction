# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MIR Feature Extraction Framework for conditioning Stable Audio Tools and similar audio generation models. Extracts 77+ numeric MIR features plus natural language descriptions from audio files using stem separation, rhythm analysis, spectral/harmonic features, and Music Flamingo AI descriptions.

**Hardware Target**: AMD RDNA4 (RX 9070 XT) with ROCm 7.1.1.1+ / PyTorch 2.11.0a0+rocm7.11

---

## Critical Build & Environment Commands

### Initial Setup

```bash
# Create virtual environment
python3 -m venv mir
source mir/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Build torchcodec from source (required for Music Flamingo on ROCm 7.1+)
./install_torchcodec_rocm.sh

# Setup external repositories (Audio Commons timbral models)
bash scripts/setup_external_repos.sh

# bitsandbytes ROCm compatibility (if needed for quantization)
cd mir/lib/python3.12/site-packages/bitsandbytes
ln -sf libbitsandbytes_rocm71.so libbitsandbytes_rocm72.so
```

### Essential Environment Variables

**For GPU optimization (ROCm):**

```bash
# Memory management (expandable_segments not supported on ROCm/HIP)
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8

# AMD ROCm optimizations
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0  # Use existing optimized kernels
export PYTORCH_TUNABLEOP_FILENAME=/home/kim/Projects/mir/tunableop_results00.csv
export MIOPEN_FIND_MODE=2  # Fast MIOpen kernel selection (avoid long delays)
export OMP_NUM_THREADS=8

# Spotify API (Required for Metadata Lookup)
export SPOTIFY_CLIENT_ID="your_client_id"
export SPOTIFY_CLIENT_SECRET="your_client_secret"

### Testing Commands

```bash
# Test ALL 70+ features on a single file (recommended)
python src/test_all_features.py "/path/to/audio.flac"
python src/test_all_features.py "/path/to/Track Name/"  # organized folder

# Test with specific GGUF model quantization (for Music Flamingo)
python src/test_all_features.py "/path/to/audio.flac" --model Q8_0   # Best quality
python src/test_all_features.py "/path/to/audio.flac" --model Q6_K   # Balanced (default)
python src/test_all_features.py "/path/to/audio.flac" --model IQ3_M  # Fastest

# Skip specific modules
python src/test_all_features.py "/path/to/audio.flac" --skip-demucs     # Skip stem separation
python src/test_all_features.py "/path/to/audio.flac" --skip-flamingo   # Skip AI descriptions

# Music Flamingo GGUF only (fastest - ~4s per track)
python src/classification/music_flamingo_gguf.py "/path/to/audio.flac" --model Q6_K
python src/classification/music_flamingo_gguf.py "/path/to/audio.flac" --prompt-type structured  # All 5 prompts

# Music Flamingo Transformers (slower - ~28s per prompt, but native Python)
python src/classification/music_flamingo_transformers.py "/path/to/audio.flac" --flash-attention
```

### Batch Processing

```bash
# Organize flat audio files into folder structure
python src/preprocessing/file_organizer.py /path/to/audio/

# Separate stems with Demucs (GPU accelerated)
python src/preprocessing/demucs_sep.py /path/to/audio/ --batch --device cuda

# ✅ Fast Demucs Separation (ROCm Optimized + Flash Attention)
python src/preprocessing/demucs_sep_optimized.py /path/to/audio/ --batch


# Extract rhythm features
python src/rhythm/beat_grid.py /path/to/audio/ --batch
python src/rhythm/bpm.py /path/to/audio/ --batch

# Extract classification features
python src/classification/essentia_features.py /path/to/audio/ --batch

# Music Flamingo batch analysis (GGUF - recommended, 7x faster)
python src/classification/music_flamingo_gguf.py /path/to/audio/ --batch --model Q6_K
python src/classification/music_flamingo_gguf.py /path/to/audio/ --batch --prompt-type structured  # All 5 prompts

# Music Flamingo batch analysis (Transformers - slower but native Python)
python src/classification/music_flamingo_transformers.py /path/to/audio/ --batch --flash-attention
```

---
## When Implementing New Features

1. Add feature to `FEATURE_RANGES` in `src/core/common.py`
2. Use `safe_update()` to save results (never overwrite)
3. Update relevant module's `claude.md` if it exists
4. Test with `test_single_file.py` before batch processing
5. Document in `project.log` and session notes

## Architecture Overview

### File Organization Structure

```
/output/Track Name/
├── full_mix.flac              # Original audio (organized from flat files)
├── drums.mp3                  # Demucs separated stems (4-source, VBR ~96kbps)
├── bass.mp3
├── other.mp3
├── vocals.mp3
├── Track Name.INFO            # All extracted features (JSON)
├── Track Name.BEATS_GRID      # Beat timestamps (text, newline-separated)
├── Track Name.DOWNBEATS       # Downbeat timestamps (text, newline-separated)
└── separated/                 # Demucs working directory
    └── htdemucs/...
```

**Key Rules**:
- Original files stay in their source location
- Organized files go to an outpute folder set in the calling script
- Full mix must be named `full_mix.{ext}` in its own folder
- Stems are saved as `drums.mp3`, `bass.mp3`, etc. (same folder, VBR ~96kbps default)
- `.INFO` files accumulate features - NEVER overwrite unless specific parameter set by call, only merge

### Module Architecture

```
src/
├── core/                          # Foundation utilities (ALWAYS use these)
│   ├── json_handler.py           # safe_update() - thread-safe atomic JSON merging
│   ├── file_utils.py             # find_organized_folders(), get_stem_files()
│   ├── text_utils.py             # normalize_music_flamingo_text() for T5 compatibility
│   ├── file_locks.py             # FileLock for concurrent processing
│   ├── batch_utils.py            # print_batch_summary()
│   └── common.py                 # setup_logging(), constants
│
src/
├── core/                          # Foundation utilities (ALWAYS use these)
│   ├── json_handler.py           # safe_update() - thread-safe atomic JSON merging
│   ├── file_utils.py             # find_organized_folders(), get_stem_files()
│   ├── text_utils.py             # normalize_music_flamingo_text() for T5 compatibility
│   ├── file_locks.py             # FileLock for concurrent processing
│   ├── batch_utils.py            # print_batch_summary()
│   └── common.py                 # setup_logging(), constants
│
├── preprocessing/
│   ├── file_organizer.py         # Main file organizer (convert flat -> folders)
│   ├── organize_files.py         # Alternative file organizer implementation
│   ├── filename_cleanup.py       # Normalizes filenames for T5 compatibility
│   ├── demucs_sep.py             # Stem separation (Standard)
│   ├── demucs_sep_optimized.py   # ✅ Stem separation (ROCm Optimized + SDPA)
│   └── loudness.py               # LUFS/LRA measurement (ITU-R BS.1770)
│
├── tools/                        # Interactive & Batch Tools (High-level)
│   ├── track_metadata_lookup.py  # Fixes artist names, fetches Spotify metadata
│   ├── create_training_crops.py  # Creates beat-aligned crops for training
│   └── statistical_analysis.py   # Analyzes dataset statistics and outliers
│
├── rhythm/
│   ├── beat_grid.py              # Beat detection + grid file generation
│   ├── bpm.py                    # BPM analysis (librosa tempo)
│   ├── onsets.py                 # Onset detection
│   ├── syncopation.py            # Syncopation analysis
│   ├── complexity.py             # Rhythmic complexity and evenness (IOI entropy)
│   └── per_stem_rhythm.py        # Per-stem rhythm features
│
├── spectral/
│   ├── spectral_features.py      # Flatness, flux, skewness, kurtosis
│   └── multiband_rms.py          # 4-band RMS energy (bass, body, mid, air)
│
├── harmonic/
│   ├── chroma.py                 # 12-dimensional chroma features/HPCP
│   └── per_stem_harmonic.py      # Harmonic movement/variance per stem
│
├── timbral/
│   ├── audio_commons.py          # 8 perceptual features (brightness, hardness, etc.)
│   ├── audiobox_aesthetics.py    # 4 aesthetics features (Meta AudioBox model)
│   └── loudness.py               # Module-based loudness analysis
│
├── classification/
│   ├── essentia_features.py      # Danceability, atonality (Essentia/TensorFlow)
│   ├── essentia_features_optimized.py # Optimized Essentia (caching for batch)
│   ├── music_flamingo.py         # ✅ RECOMMENDED: GGUF/llama.cpp entry point
│   ├── music_flamingo_llama_cpp.py # Direct GGUF/llama-cpp-python implementation
│   └── music_flamingo_transformers.py  # Native Python via HuggingFace transformers
│
├── transcription/                    # MIDI transcription pipeline
│   ├── drums/
│   │   ├── adtof.py              # ✅ ADTOF-PyTorch drum transcription (GPU accelerated)
│   │   └── drumsep.py            # Drumsep + onset detection
│   ├── midi_utils.py             # MIDI file utilities
│   └── runner.py                 # Batch transcription orchestrator
│
├── benchmarks/                   # Performance testing scripts
│   ├── benchmark_fp8_full.py     # FP8 + TunableOps benchmark
│   ├── benchmark_music_flamingo.py # Speed test for Music Flamingo
│   └── benchmark_quantization.py # INT8/INT4 vs BFloat16 comparison
│
├── tests/                        # Individual feature testing
│   ├── test_single_file.py       # Quick test on one file
│   ├── test_features_timing.py   # Detailed timing breakdown per feature
│   └── test_threading.py         # CPU usage/threading analysis
│
├── scripts/
│   ├── apply_patches.py          # Apply librosa 0.11 fixes to external repos
│   ├── download_essentia_models.py # Fetch required TensorFlow models
│   └── setup_external_repos.sh   # Clone and setup all dependencies
│
├── batch_process.py              # All-in-one batch processing entry point
├── pipeline.py                   # Orchestrator for all features
└── test_all_features.py          # Comprehensive test script for all 70+ features

### External Repositories (in `repos/`)
Note: `repos/repos/` contains cloned external projects (untouched or patched):
- **timbral_models**: Audio Commons features (patched)
- **ADTOF-pytorch**: GPU-accelerated drum transcription
- **llama.cpp**: Efficient inference backend
- **drumsep**, **madmom**: Rhythm and separation tools
- **basic-pitch**, **crepe**, **pesto**: Pitch tracking (experimental)
- **mt3**, **MR-MT3**, **magenta**: Polyphonic transcription (experimental)
```

### Music Flamingo Integration (CRITICAL)

**Two Methods Available**:

| Method | Location | Speed | VRAM | Best For |
|--------|----------|-------|------|----------|
| **GGUF (Recommended)** | `src/classification/music_flamingo_gguf.py` | ~4s/track | 5-9GB | Production, batch processing |
| Transformers | `src/classification/music_flamingo_transformers.py` | ~28s/prompt | 13GB | Native Python, fine-tuning |

**Architecture**: NVIDIA Music Flamingo (8B params: Qwen2.5-7B language + Audio Flamingo 3 encoder)

#### GGUF Method (Recommended) ✅

Uses llama.cpp `llama-mtmd-cli` tool. **7x faster than transformers with 40-60% less VRAM.**

```python
from classification.music_flamingo_gguf import MusicFlamingoGGUF

analyzer = MusicFlamingoGGUF(model='Q6_K')  # IQ3_M, Q6_K, or Q8_0
description = analyzer.analyze('audio.flac', prompt_type='full')

# Or all prompt types at once
results = analyzer.analyze_all_prompts('audio.flac')
```

**Note**: RDNA 4 (gfx1201) has a POOL_1D operator warning - this is cosmetic, model works correctly.

#### Transformers Method

Native Python via HuggingFace. Slower but more flexible.

```python
from classification.music_flamingo_transformers import MusicFlamingoTransformers

analyzer = MusicFlamingoTransformers(use_flash_attention=True)
description = analyzer.analyze('audio.flac', prompt_type='full')
```

**Memory Management** (transformers only): Uses `clear_cache()` after each prompt to prevent OOM.

#### Common Details (Both Methods)

**Five Prompt Types**:
- `full`: Comprehensive description (genre, tempo, key, instruments, mood)
- `technical`: Tempo, key, chords, dynamics, performance analysis
- `genre_mood`: Genre classification + emotional character
- `instrumentation`: Instruments and sounds present
- `structure`: Arrangement and structure analysis

**Saved Keys in .INFO**:
- `music_flamingo_full`, `music_flamingo_technical`, `music_flamingo_genre_mood`
- `music_flamingo_instrumentation`, `music_flamingo_structure`

**Text Normalization (MANDATORY)**: All output is automatically normalized for T5 tokenizer compatibility via `normalize_music_flamingo_text()`.

**Quantization** (transformers only): INT8/INT4 NOT FUNCTIONAL on ROCm - use bfloat16 + Flash Attention 2

### TunableOps Optimization

**File**: `/home/kim/Projects/mir/tunableop_results00.csv`

Pre-optimized GEMM kernels for RDNA4 architecture. Generated from scratch during benchmarking. Provides 10-58% speedup for various operations.

**Usage**: Set environment variables (see above). Never delete this file without regenerating via `PYTORCH_TUNABLEOP_TUNING=1`.

---

## Critical Development Rules

### JSON Handling (ALWAYS FOLLOW)

**NEVER use `write_info()` with `merge=False`** - this destroys existing features.

**ALWAYS use**:
```python
from core.json_handler import safe_update, get_info_path

info_path = get_info_path(full_mix_path)
safe_update(info_path, {'new_feature': value})  # Atomic merge
```

### Text Normalization (Music Flamingo etc)

All text output, especially from Music Flamingo MUST be normalized before saving to ensure T5 tokenizer compatibility:

```python
from core.text_utils import normalize_music_flamingo_text

description = analyzer.analyze(audio_path)
# description is already normalized automatically
# If processing old results:
normalized = normalize_music_flamingo_text(raw_text)
```

**Examples of characters that break T5**:
- U+2011 (non-breaking hyphen `‑`)
- U+2014 (em-dash `—`)
- U+2013 (en-dash `–`)
- U+202F (narrow no-break space)
- U+2018, U+2019, U+201C, U+201D (curly quotes)

### File Organization

1. **Never move original files** - always copy to `/output`
2. **Never delete .INFO files** - only append/merge
3. **Use FileLock for batch processing** - prevents concurrent access issues
4. **Check for existing organized structure** before organizing

### Model Loading

**Load models ONCE** for batch processing:

```python
# GOOD - load once
flamingo = MusicFlamingoTransformers(use_flash_attention=True)
for folder in folders:
    description = flamingo.analyze(stems['full_mix'])

# BAD - loads model every iteration
for folder in folders:
    flamingo = MusicFlamingoTransformers()  # WASTEFUL
    description = flamingo.analyze(stems['full_mix'])
```

### GPU Memory Management

1. Set `PYTORCH_ALLOC_CONF=expandable_segments:True` (prevents fragmentation)
2. Clear cache after heavy operations (Music Flamingo already does this)
3. Use Flash Attention 2 for Music Flamingo and other supported projects (`use_flash_attention=True`)
4. Consider GGUF quantization for memory-constrained scenarios

---

## Known Issues & Workarounds

### ROCm Version Reporting

PyTorch may report ROCm 7.2 while actual version is 7.1.1.1 (nightly). This is expected with bleeding-edge builds.

**Workaround for bitsandbytes**:
```bash
cd mir/lib/python3.12/site-packages/bitsandbytes
ln -sf libbitsandbytes_rocm71.so libbitsandbytes_rocm72.so
```

### TorchCodec Incompatibility

Pre-built torchcodec incompatible with PyTorch 2.11.0a0+rocm7.11.

**Solution**: Build from source using `install_torchcodec_rocm.sh`

### numba/numpy Version Conflict

If you see `Numba needs NumPy 2.3 or less. Got NumPy 2.4`:

**Workaround**: Use soundfile instead of librosa for duration:
```python
import soundfile as sf
with sf.SoundFile(audio_path) as f:
    duration = len(f) / f.samplerate
```

### FP8 Not Supported

RDNA4 hardware supports FP8, but transformers library doesn't support FP8 `torch_dtype` yet.

**Solution**: Use bfloat16 + Flash Attention 2 (INT8/INT4 also don't work on ROCm).

### INT8/INT4 Quantization Not Functional on ROCm

**Tested 2026-01-18**: bitsandbytes INT8 and INT4 quantization fail during Music Flamingo inference on AMD ROCm.

**Symptom**: Model loads successfully but OOM during first prompt generation.

**Root Cause**: bitsandbytes quantization on ROCm only quantizes model weights, not inference operations. Activations and computations still use full precision, causing ~15GB+ memory usage.

**Status**: Works on CUDA, does not work on ROCm. Code kept for future compatibility.

**Solution**: Use bfloat16 + Flash Attention 2 (~13GB VRAM, 1.06x realtime).

## Recent Session Achievements (2026-01-19)

**DO NOT rely on older documentation** - these are the latest fixes:

1. ✅ **TorchCodec**: Built from source for ROCm 7.11 compatibility
2. ✅ **Music Flamingo Memory**: Implemented cache clearing - 100% completion rate
3. ✅ **Text Normalization**: All Music Flamingo output T5-safe for Stable Audio Tools
4. ✅ **TunableOps**: Generated RDNA4-optimized kernels (10-58% speedup)
5. ✅ **Folder Structure**: `/output` folder with proper organization
6. ✅ **Quantization Testing**: INT8/INT4 tested - **NOT functional on ROCm** (inference OOM)
7. ✅ **GGUF/llama.cpp**: **NOW WORKING** - 7x faster than transformers, 40-60% less VRAM
8. ✅ **llama.cpp Built**: HIP-enabled CLI tools in `repos/llama.cpp/build/bin/`
9. ✅ **AudioBox Aesthetics**: **NOW WORKING** - Meta's quality assessment model integrated
10. ✅ **Essentia GMI**: Genre (400), Mood (56), Instrument (40) classification added
11. ✅ **test_all_features.py**: Now includes 14 steps + Music Flamingo bonus

**Latest findings**:
- INT8/INT4 quantization non-functional on ROCm - use bfloat16 + Flash Attention 2
- **GGUF/llama.cpp NOW WORKS** - Use `llama-mtmd-cli` for fastest inference (3.7s vs 28s)
- **AudioBox Aesthetics NOW WORKS** - `pip install git+https://github.com/facebookresearch/audiobox-aesthetics.git`
- **ADTOF-PyTorch NOW WORKS** - Drum transcription with ROCm GPU acceleration

### 2026-01-21 Session (ADTOF Integration)

1. ✅ **ADTOF-PyTorch**: GPU-accelerated drum transcription (replaces TensorFlow version)
2. ✅ **Drumsep Integration**: Alternative drum transcription via stem separation
3. ❌ **TensorFlow ADTOF**: Incompatible with Keras 3 (weight format not supported)
4. ✅ **adtof.py Wrapper**: New wrapper using ADTOF-PyTorch in `src/transcription/drums/adtof.py`

### 2026-01-25 Session (Demucs Parallel Processing & Optimizations)

1. ✅ **Parallel Demucs**: Running multiple instances increases GPU utilization from 27% to 100%
   - 2 workers: 1.69x speedup, 84% efficiency, ~5.4GB VRAM
   - 4 workers: ~2.5-3x speedup expected, ~10.8GB VRAM
   - Benchmark script: `benchmark_demucs_parallel.py`
2. ✅ **File Organizer Improvements**: Track number removal, Various Artists fix, duplicate prevention
3. ❌ **torch.compile on Demucs**: Does NOT work on ROCm - complex FFT ops cause dtype errors
4. ✅ **SDPA Attention Patch**: Works well, provides Flash Attention acceleration
5. ✅ **StaticSegmentModel**: Created in demucs fork but torch.compile still fails
6. ✅ **Master Pipeline Fix**: Corrected `batch_analyze_beat_grid` → `batch_create_beat_grids`

**Key Finding**: Demucs GPU utilization is limited to ~27% per instance due to segment-wise processing.
Running multiple parallel instances is the most effective way to maximize GPU throughput.

**Recommended config for batch processing**:
```yaml
demucs:
  workers: 2         # Subprocess-based parallel (each ~5GB VRAM)
  model: htdemucs
  shifts: 0          # Fast mode
  segment: null      # Use model default (7.8s for htdemucs)

rhythm:
  workers: 4         # Parallel beat/downbeat detection

cropping:
  workers: 6         # Parallel crop creation

processing:
  feature_workers: 8 # Parallel feature extraction

music_flamingo:
  model: Q8_0
  prompts:
    full: true
    technical: true
    genre_mood: true
    instrumentation: true
    structure: true
  max_tokens:
    full: 500
    technical: 500
    genre_mood: 500
    instrumentation: 500
    structure: 500
```

### 2026-01-26 Session (Parallel Processing & Config Improvements)

1. **Parallel Rhythm/Beat Detection**: Added `ProcessPoolExecutor` with configurable workers
2. **Music Flamingo Config**: Token limits and prompt selection now configurable in YAML
3. **Batch Feature Extraction**: New hybrid batch mode for GPU model persistence
4. **Demucs Workers**: Changed from unused `jobs` to subprocess-based `workers`
5. **All Config Updated**: `master_pipeline.yaml` now has all parallel processing settings

---

## References to Check (By Date - Newest First)

- `SESSION_SUMMARY.md` - **LATEST** Project Status & Achievements
- `USER_MANUAL.md` - Comprehensive usage & troubleshooting
- `README.md` - Project overview & feature table
- `QUICKREF_PRODUCTION_CONFIG.md` - Production env vars & setup
- `GGUF_INVESTIGATION.md` - Music Flamingo GGUF details
- `TEXT_NORMALIZATION.md` - Stable Audio Tools compatibility guide
- `FEATURES_STATUS.md` - detailed feature implementation tracker

**Historical/Specific References:**
- `QUANTIZATION_TEST_RESULTS.md` - ROCm quantization constraints
- `FINAL_BENCHMARK_RESULTS_2026-01-18.md` - Performance baselines
- `plans/` directory - Original architectural plans

---

## When You See Errors

1. **Check environment variables first** (especially `PYTORCH_ALLOC_CONF`)
2. **Check bitsandbytes symlink** if quantization fails
3. **Check for numba/numpy conflict** if librosa fails
4. **GGUF "error: invalid argument"** - this is cosmetic (POOL_1D CPU fallback on RDNA 4), model works fine
5. **Never assume old documentation is current** - check session dates

---

**Last Updated**: 2026-01-26 (Session: Parallel processing, Music Flamingo config, batch feature extraction)
**Hardware**: AMD Radeon RX 9070 XT (16GB VRAM) + Ryzen 9 9900X

