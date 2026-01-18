# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MIR Feature Extraction Framework for conditioning Stable Audio Tools and similar audio generation models. Extracts 77+ numeric MIR features plus natural language descriptions from audio files using stem separation, rhythm analysis, spectral/harmonic features, and Music Flamingo AI descriptions.

**Hardware Target**: AMD RDNA4 (RX 9070 XT) with ROCm 7.1.1.1+ / PyTorch 2.11.0a0+rocm7.11

**Key Achievement**: Production-ready with GPU acceleration, achieving 4-9x realtime for standard features, 1-2x realtime for comprehensive AI descriptions.

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

**For Music Flamingo + GPU optimization:**

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0  # Use existing optimized kernels
export PYTORCH_TUNABLEOP_FILENAME=/home/kim/Projects/mir/tunableop_results00.csv
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export HIP_FORCE_DEV_KERNARG=1
export OMP_NUM_THREADS=8
```

**Critical**: Always set `PYTORCH_ALLOC_CONF=expandable_segments:True` before Music Flamingo to avoid OOM errors.

### Testing Commands

```bash
# Test single file with all features
python src/test_single_file.py "/path/to/Track Name/full_mix.flac"

# Benchmark Music Flamingo performance
python src/benchmark_fp8_full.py "/path/to/Track Name/full_mix.flac"

# Test quantization methods (INT8/INT4)
python src/benchmark_quantization.py
```

### Batch Processing

```bash
# Organize flat audio files into folder structure
python src/preprocessing/file_organizer.py /path/to/audio/

# Separate stems with Demucs (GPU accelerated)
python src/preprocessing/demucs_sep.py /path/to/audio/ --batch --device cuda

# Extract rhythm features
python src/rhythm/beat_grid.py /path/to/audio/ --batch
python src/rhythm/bpm.py /path/to/audio/ --batch

# Extract classification features
python src/classification/essentia_features.py /path/to/audio/ --batch

# Music Flamingo batch analysis
python src/classification/music_flamingo_transformers.py /path/to/audio/ --batch --flash-attention
```

---

## Architecture Overview

### File Organization Structure

```
/output/Track Name/
├── full_mix.flac              # Original audio (organized from flat files)
├── drums.flac                 # Demucs separated stems (4-source)
├── bass.flac
├── other.flac
├── vocals.flac
├── Track Name.INFO            # All extracted features (JSON)
├── Track Name.BEATS_GRID      # Beat timestamps (text, newline-separated)
└── separated/                 # Demucs working directory
    └── htdemucs/...
```

**Key Rules**:
- Original files stay in `/test_data` or their source location
- Organized files go to `/output` (created by file_organizer.py)
- Full mix must be named `full_mix.{ext}` in its own folder
- Stems are saved as `drums.flac`, `bass.flac`, etc. (same folder)
- `.INFO` files accumulate features - NEVER overwrite, only merge

### Module Architecture

```
src/
├── core/                          # Foundation utilities (ALWAYS use these)
│   ├── json_handler.py           # safe_update() - atomic JSON merging
│   ├── file_utils.py             # find_organized_folders(), get_stem_files()
│   ├── text_utils.py             # normalize_music_flamingo_text() for T5 compatibility
│   ├── file_locks.py             # FileLock for concurrent processing
│   ├── batch_utils.py            # print_batch_summary()
│   └── common.py                 # setup_logging(), constants
│
├── preprocessing/
│   ├── file_organizer.py         # Converts flat files to folder structure
│   ├── demucs_sep.py             # Stem separation (GPU accelerated)
│   └── loudness.py               # LUFS/LRA measurement (ITU-R BS.1770)
│
├── rhythm/
│   ├── beat_grid.py              # Beat detection + grid file generation
│   ├── bpm.py                    # BPM analysis (librosa tempo)
│   ├── onsets.py                 # Onset detection
│   ├── syncopation.py            # Syncopation analysis
│   └── per_stem_rhythm.py        # Per-stem rhythm features
│
├── spectral/
│   ├── spectral_features.py      # Flatness, flux, skewness, kurtosis
│   └── multiband_rms.py          # 4-band RMS energy (bass, body, mid, air)
│
├── harmonic/
│   ├── chroma.py                 # 12-dimensional chroma features
│   └── per_stem_harmonic.py      # Harmonic movement/variance per stem
│
├── timbral/
│   ├── audio_commons.py          # 8 perceptual features (brightness, hardness, etc.)
│   └── audiobox_aesthetics.py    # 4 aesthetics features (currently defaults)
│
└── classification/
    ├── essentia_features.py      # Danceability, atonality (Essentia/TensorFlow)
    ├── music_flamingo_transformers.py  # AI-powered music descriptions (ONLY WORKING METHOD)
    └── music_flamingo.py         # GGUF/llama.cpp approach (NOT FUNCTIONAL - audio not supported)
```

### Music Flamingo Integration (CRITICAL)

**Location**: `src/classification/music_flamingo_transformers.py`

**Architecture**: Uses NVIDIA Music Flamingo (8B params: Qwen2.5-7B language + Audio Flamingo 3 encoder) via HuggingFace transformers.

**IMPORTANT**: This is the **ONLY working method**. GGUF/llama.cpp is NOT supported for audio multimodal (see Known Issues below).

**Key Implementation Details**:

1. **Text Normalization (MANDATORY)**: All Music Flamingo output is automatically normalized for T5 tokenizer compatibility via `normalize_music_flamingo_text()` which replaces Unicode special characters (em-dashes, curly quotes, non-breaking hyphens) with ASCII equivalents.

2. **Memory Management**: Uses `clear_cache()` after each prompt to prevent OOM:
   ```python
   del outputs
   del inputs
   torch.hip.empty_cache()
   gc.collect()
   ```

3. **Five Prompt Types**:
   - `full`: Comprehensive description (genre, tempo, key, instruments, mood)
   - `technical`: Tempo, key, chords, dynamics, performance analysis
   - `genre_mood`: Genre classification + emotional character
   - `instrumentation`: Instruments and sounds present
   - `structure`: Arrangement and structure analysis

4. **Saved Keys in .INFO**:
   - `music_flamingo_full`
   - `music_flamingo_technical`
   - `music_flamingo_genre_mood`
   - `music_flamingo_instrumentation`
   - `music_flamingo_structure`

5. **Quantization Support** (via bitsandbytes):
   - ❌ **NOT FUNCTIONAL on ROCm** - Models load but OOM during inference
   - INT8/INT4 code exists but fails on AMD GPUs (works on CUDA only)
   - See `QUANTIZATION_TEST_RESULTS.md` for details
   - **Do not use `quantization` parameter on ROCm**

6. **Performance**: 1.06x realtime for all 5 prompts with Flash Attention 2 + TunableOps

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

### Text Normalization (Music Flamingo)

All Music Flamingo output MUST be normalized before saving to ensure T5 tokenizer compatibility:

```python
from core.text_utils import normalize_music_flamingo_text

description = analyzer.analyze(audio_path)
# description is already normalized automatically
# If processing old results:
normalized = normalize_music_flamingo_text(raw_text)
```

**Characters that break T5**:
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
3. Use Flash Attention 2 for Music Flamingo (`use_flash_attention=True`)
4. Consider quantization for memory-constrained scenarios

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

### GGUF/llama.cpp NOW SUPPORTED for Music Flamingo ✅

**Updated 2026-01-18**: Music Flamingo **WORKS** via GGUF/llama.cpp using `llama-mtmd-cli`.

Audio multimodal support was added to llama.cpp in late 2025 (PR #18470 merged Dec 31, 2025).

**Available GGUF files** in `models/music_flamingo/`:
- `music-flamingo-hf.i1-IQ3_M.gguf` (3.4GB) - Fast, good quality
- `music-flamingo-hf.Q8_0.gguf` (7.6GB) - Best quality
- `music-flamingo-hf.mmproj-f16.gguf` (1.3GB) - Required audio projector

**Performance Comparison**:
| Method | VRAM | Time (2.5min track) |
|--------|------|---------------------|
| GGUF IQ3_M | ~5.4GB | **3.7s** |
| GGUF Q8_0 | ~9.3GB | **4.0s** |
| Transformers | ~13GB | ~28s |

**GGUF is ~7x faster with ~40-60% less VRAM!**

**Usage**:
```bash
# Build llama.cpp with HIP (one-time)
cd repos && git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-mtmd-cli -j$(nproc)

# Run inference
repos/llama.cpp/build/bin/llama-mtmd-cli \
  -m models/music_flamingo/music-flamingo-hf.i1-IQ3_M.gguf \
  --mmproj models/music_flamingo/music-flamingo-hf.mmproj-f16.gguf \
  --audio "path/to/audio.flac" \
  -p "Describe this music track." \
  -n 200 --gpu-layers 99
```

**Note**: `llama-cpp-python` doesn't have audio bindings yet - use CLI tools.

See `GGUF_INVESTIGATION.md` for full documentation.

---

## Performance Expectations

### Standard Features (Per 2.5min Track)

| Feature | Time | Speed | Notes |
|---------|------|-------|-------|
| Demucs separation | 15.4s | 9.8x | GPU required |
| Beat Grid | 16.3s | 9.3x | CPU (librosa) |
| BPM Analysis | <0.1s | instant | Cached |
| Essentia features | 2.7s | 55x | GPU TensorFlow |
| **Total** | **~35s** | **4.4x** | |

### Music Flamingo (Per 2.5min Track)

| Configuration | Time | Speed | Memory | Notes |
|--------------|------|-------|--------|-------|
| **GGUF IQ3_M** | **3.7s** | **40x** | **5.4GB** | ✅ **Recommended** |
| **GGUF Q8_0** | **4.0s** | **38x** | **9.3GB** | ✅ Best quality |
| Transformers (1 prompt) | ~28s | 5x | 13GB | Native Python |
| Transformers (5 prompts) | 143s | 1.06x | 13GB | All descriptions |
| ~~INT8/INT4~~ | ~~N/A~~ | ~~N/A~~ | ~~N/A~~ | ❌ Not functional on ROCm |

### Scaling to 10,000 Files

| Configuration | Total Time | Parallel (4x) |
|--------------|-----------|---------------|
| Minimal (no Demucs/Flamingo) | 53 hours | 13 hours |
| Fast (2 Flamingo prompts) | 119 hours | 30 hours |
| Comprehensive (5 prompts) | 492 hours | 123 hours |

---

## Recent Session Achievements (2026-01-18)

**DO NOT rely on older documentation** - these are the latest fixes:

1. ✅ **TorchCodec**: Built from source for ROCm 7.11 compatibility
2. ✅ **Music Flamingo Memory**: Implemented cache clearing - 100% completion rate
3. ✅ **Text Normalization**: All Music Flamingo output T5-safe for Stable Audio Tools
4. ✅ **TunableOps**: Generated RDNA4-optimized kernels (10-58% speedup)
5. ✅ **Folder Structure**: `/output` folder with proper organization
6. ✅ **Quantization Testing**: INT8/INT4 tested - **NOT functional on ROCm** (inference OOM)
7. ✅ **GGUF/llama.cpp**: **NOW WORKING** - 7x faster than transformers, 40-60% less VRAM
8. ✅ **llama.cpp Built**: HIP-enabled CLI tools in `repos/llama.cpp/build/bin/`
9. ✅ **CLAUDE.md Updated**: Comprehensive guidance for future Claude instances

**Latest findings**:
- INT8/INT4 quantization non-functional on ROCm - use bfloat16 + Flash Attention 2
- **GGUF/llama.cpp NOW WORKS** - Use `llama-mtmd-cli` for fastest inference (3.7s vs 28s)

---

## References to Check (By Date - Newest First)

- `QUANTIZATION_TEST_RESULTS.md` - INT8/INT4 test results (NOT functional on ROCm) ⚠️
- `SESSION_COMPLETE_2026-01-18.md` - Latest session results (MOST CURRENT)
- `TEXT_NORMALIZATION.md` - T5 tokenizer compatibility guide
- `QUICKREF_PRODUCTION_CONFIG.md` - Copy-paste production setup
- `FINAL_BENCHMARK_RESULTS_2026-01-18.md` - Complete performance analysis
- `README.md` - Project overview (updated 2026-01-13)
- `plans/01-developement.txt` - Core development principles
- `plans/02-stem_separation.txt` - Folder structure specification
- `src/classification/music_flamingo.py` - GGUF code (exists but NOT FUNCTIONAL) ⚠️

**Ignore older files** with issues that have been fixed in recent sessions.

**Important**: Do NOT attempt to use GGUF/llama.cpp for Music Flamingo - it cannot support audio multimodal.

---

## When Implementing New Features

1. Add feature to `FEATURE_RANGES` in `src/core/common.py`
2. Use `safe_update()` to save results (never overwrite)
3. Update relevant module's `claude.md` if it exists
4. Test with `test_single_file.py` before batch processing
5. Document in `project.log` and session notes

---

## When You See Errors

1. **Check environment variables first** (especially `PYTORCH_ALLOC_CONF`)
2. **Check bitsandbytes symlink** if quantization fails
3. **Check for numba/numpy conflict** if librosa fails
4. **Don't try GGUF/llama.cpp** - not supported for Music Flamingo audio multimodal
5. **Never assume old documentation is current** - check session dates

---

**Last Updated**: 2026-01-18 (Session: Quantization Testing Complete)
**Hardware**: AMD Radeon RX 9070 XT (16GB VRAM) + Ryzen 9 9900X
**Status**: Production Ready (Standard Features) + Music Flamingo Operational (bfloat16 only)
