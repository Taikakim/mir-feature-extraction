# Session Summary - 2026-02-05 (Latest Updates)

## What We Accomplished

### 1. Critical Bug Fixes

**Essentia TensorFlow Deadlock:**
- Running Essentia with 20 parallel workers caused TensorFlow to deadlock
- Added separate `essentia_workers` config (default: 4, TensorFlow-safe)
- Other CPU features can still use higher parallelism

**Skip Logic Key Mismatches:**
- Sequential mode was checking wrong feature keys, causing re-processing
- Fixed: `spectral_centroid` → `spectral_flatness`, `rms_bass` → `rms_energy_bass`, `chroma_mean` → `chroma_0`

**Crop BPM/Beat Count Bug:**
- Crop-specific values were being overwritten by source track values
- Fixed: Removed `bpm` and `beat_count` from `OPTIONAL_TRANSFERRABLE`

### 2. Performance Improvements

**Spectral Features - Essentia Migration:**
- Rewrote `spectral_features.py` to use Essentia (faster C++ backend)
- Changed aggregation from mean to median (more robust to outliers)
- Kept librosa as fallback if Essentia not available

### 3. Pipeline Cleanup

**Removed Unused .ONSETS Slicing:**
- Analysis found `.ONSETS` files were sliced for crops but never used
- Crop analysis only calculates BPM via librosa (doesn't read .ONSETS)
- Syncopation/complexity (which use .ONSETS) not run on crops
- Keeps `.BEATS_GRID` and `.DOWNBEATS` slicing (these ARE used)

### 4. Code Refactoring

**New Core Modules Extracted:**
| File | Lines | Purpose |
|------|-------|---------|
| `src/core/terminal.py` | 179 | ANSI colors, ColoredFormatter |
| `src/core/metadata_utils.py` | 149 | Audio metadata extraction |
| `src/core/pipeline_stats.py` | 148 | TimingStats, PipelineStats |
| `src/core/pipeline_workers.py` | 180 | Parallel processing workers |

`master_pipeline.py` reduced from 2218 to 1673 lines (-545)

## Configuration Changes

**New Option:**
```yaml
processing:
  feature_workers: 20    # For CPU features (loudness, spectral, timbral)
  essentia_workers: 4    # Separate limit for Essentia (TensorFlow-safe)
```

## Commits This Session

| Hash | Description |
|------|-------------|
| `6cb866f` | Refactor pipeline with modular utilities and fix Essentia deadlock |
| `d9092a9` | Add imports for extracted core modules in master_pipeline |
| `3a63fa2` | Fix skip logic key mismatches in sequential processing mode |
| `7e802d9` | Use median instead of mean for spectral feature aggregation |
| `60cb85c` | Rewrite spectral features to use Essentia instead of librosa |
| `750e96a` | Fix crop INFO overwriting crop-specific bpm/beat_count with source values |
| `edf1d07` | Remove unused .ONSETS slicing from crop creation |

## Verified Pipeline Flow

1. **Stage 2a:** Demucs separates full tracks into stems
2. **Stage 2b:** Rhythm analysis (beats, onsets, downbeats) on full_mix only
3. **Stage 2c:** Metadata lookup
4. **Stage 2d:** Per-stem features stored on full_mix .INFO
5. **Stage 3:** Crops created with stems at identical positions (perfect sync)
   - `.BEATS_GRID` and `.DOWNBEATS` sliced (NOT `.ONSETS`)
   - Crop-specific bpm/beat_count calculated fresh
6. **Stage 4:** Only full_mix crops analyzed (stem crops excluded)
   - Per-stem loudness extracted from stem crops
   - Existing features skipped when overwrite=false

## Key Files Modified

- `src/pipeline.py` - essentia_workers, skip logic fixes
- `src/master_pipeline.py` - refactored to use core modules
- `src/spectral/spectral_features.py` - Essentia + median aggregation
- `src/tools/create_training_crops.py` - bpm fix, removed .ONSETS slicing
- `config/master_pipeline.yaml` - essentia_workers option
- `config/small.yaml` - test config with high parallelism

## Previous Sessions

- See `project.log` for full session history
- 2026-01-26: Parallel processing, batch feature extraction
- 2026-01-25: Demucs parallel processing, file organizer improvements
- 2026-01-21: ADTOF drum transcription integration
