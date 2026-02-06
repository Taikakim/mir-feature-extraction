# Session Summary - 2026-02-06

## What We Accomplished

### 1. BS-RoFormer Integration & Bug Fixes

**BS-RoFormer as Alternative to Demucs:**
- Integrated BS-RoFormer as drop-in replacement for Demucs stem separation
- Config option: `source_separation.backend: bs_roformer` or `demucs`
- Fixed multiple bugs from initial implementation:
  - `model` → `separator` variable naming
  - `audio_folder` → `folder_path` parameter
  - Moved `atexit.register()` after class definition
  - Removed duplicate function definitions and imports

**Stem Check Improvements:**
- Both backends now check for existing stems in ALL formats (.wav, .flac, .mp3)
- Previously Demucs only checked configured output format
- Prevents re-processing when stems created by different backend

### 2. Audio Duration Tracking

**New Pipeline Statistics:**
- Added `total_audio_duration` tracking to `PipelineStats`
- Added `realtime_factor` property (audio seconds / processing seconds)
- Pipeline calculates total audio duration at startup
- Summary shows processing speed relative to realtime:
  ```
  Audio processed: 12.5 min (751s)
  Processing speed: 0.53x realtime
    (Took 1.9 seconds of wall time per second of audio)
  ```

### 3. Music Flamingo Model Detection Fix

**imatrix File Exclusion:**
- Fixed model auto-detection to exclude `imatrix` calibration files
- Now picks largest GGUF model file (highest quality quantization)
- Applied to all 4 locations:
  - `music_flamingo_llama_cpp.py` (3 places)
  - `pipeline.py`
  - `feature_extractor.py`

### 4. HDD I/O Optimization

**RAM Prefetcher (`src/core/audio_prefetcher.py`):**
- New `AudioPrefetcher` class loads audio into RAM in background thread
- New `PathPrefetcher` warms OS disk cache for path-based APIs
- Integrated into Music Flamingo pass with buffer_size=8
- Hides HDD seek latency while GPU processes current file

**Batch .INFO Writes:**
- Pre-load source .INFO once before crop loop (was N redundant reads)
- New `batch_write_info()` function for efficient multi-file writes
- Uses atomic writes (temp file + rename) for crash safety
- Groups all .INFO writes at end of folder processing

### 5. Stats Counter Fixes

**crops_processed Counter:**
- AudioBox and Music Flamingo passes weren't incrementing counter
- Fixed: now properly tracks processed/failed crops
- Summary correctly shows "Crops analyzed: N" instead of 0

## Commits This Session

| Hash | Description |
|------|-------------|
| `de8ca35` | Fix Demucs stem check to detect all formats |
| `5cc89c1` | Use atomic writes in batch_write_info for safety |
| `a5ee1e0` | Optimize crop .INFO writes for HDD performance |
| `5235ba0` | Add audio prefetcher for HDD I/O optimization |
| `27c9637` | Fix crops_processed counter in batched pipeline passes |
| `4578e53` | Fix Music Flamingo model detection in music_flamingo_llama_cpp.py |
| `d1078a7` | Add audio duration tracking and fix Music Flamingo model detection |
| `55872f3` | Fix duplicate code block and Music Flamingo init in pipeline.py |
| `a1b5229` | Fix undefined audio_folder variable in save_stem |
| `8224c2f` | Fix: Error when config input is null instead of defaulting to '.' |
| `5e22b5a` | Fix folder discovery to skip crops, repos, and test output |
| `e2ceab9` | Fix BS-RoFormer model variable and exclude .venv from folder scan |
| `950f4b7` | Update BS-RoFormer docs with pipeline integration details |
| `e3ba4f3` | Integrate BS-RoFormer into master pipeline with bug fixes |

## New Files

| File | Purpose |
|------|---------|
| `src/core/audio_prefetcher.py` | RAM prefetcher for HDD I/O optimization |
| `src/preprocessing/bs_roformer_sep.py` | BS-RoFormer stem separation wrapper |
| `docs/BS_ROFORMER_OPTIMIZATION.md` | BS-RoFormer walkthrough and optimization guide |

## Configuration Changes

**New Source Separation Options:**
```yaml
source_separation:
  backend: bs_roformer  # or 'demucs'

  bs_roformer:
    model_name: jarredou-BS-ROFO-SW-Fixed-drums
    model_dir: /home/kim/Projects/mir/models/bs-roformer
    batch_size: 1
    device: cuda
```

**Overwrite Control:**
```yaml
overwrite:
  source_separation: false  # Skip if stems exist (any format)
  demucs: false            # Backwards compatibility alias
```

## Key Files Modified

- `src/master_pipeline.py` - BS-RoFormer integration, duration tracking, stem checks
- `src/pipeline.py` - Music Flamingo fixes, prefetcher integration, stats counters
- `src/core/pipeline_stats.py` - Added total_audio_duration and realtime_factor
- `src/core/json_handler.py` - Added batch_write_info() function
- `src/tools/create_training_crops.py` - Optimized .INFO writes for HDD
- `src/classification/music_flamingo_llama_cpp.py` - imatrix exclusion fix
- `src/crops/feature_extractor.py` - imatrix exclusion fix

## Performance Improvements

| Optimization | Impact |
|--------------|--------|
| Pre-load source .INFO | Eliminates N redundant file reads per folder |
| Batch .INFO writes | Groups writes at end (better HDD seek pattern) |
| RAM prefetcher | Hides HDD latency during GPU processing |
| All-format stem check | Prevents unnecessary re-separation |

## Verified Behavior

1. **Stage 2 Stem Separation:**
   - Checks for existing stems in .wav, .flac, .mp3
   - Skips if all 4 stems found (drums, bass, other, vocals)
   - Works with both BS-RoFormer and Demucs backends

2. **Music Flamingo Model Detection:**
   - Excludes mmproj and imatrix files
   - Selects largest GGUF file (highest quality)
   - Works in all entry points (CLI, batch, pipeline)

3. **Stats Tracking:**
   - Total audio duration calculated at startup
   - Realtime factor shown in summary
   - crops_processed counter accurate for all passes
