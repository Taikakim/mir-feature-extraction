# Session Summary - 2026-01-26 (Latest Updates)

## What We Accomplished

### 1. Major Pipeline Improvements

**Parallel Processing:**
- Feature extraction now uses `ProcessPoolExecutor` with 8 workers (configurable)
- Cropping now uses `ProcessPoolExecutor` with 6 workers (configurable)
- Rhythm/beat detection now uses `ProcessPoolExecutor` with 4 workers (configurable)
- Demucs now uses subprocess-based parallel processing with 2 workers (configurable)
- All use `FileLock` to prevent race conditions

**Bug Fixes:**
- Fixed duplicate file creation on resume (cleaned filename check)
- Fixed cropping import error (`process_folder` -> `create_crops_for_file`)
- Fixed MP3 stem sample rate (96kbps->128kbps, added resampling)
- Removed unnecessary BPM requirement for beat-aligned cropping
- Added rhythm file slicing to sequential mode

**New Features:**
- Audio file metadata extraction (MP3 ID3 tags) via mutagen
- Infinite loop prevention for cropping
- Colored terminal output for better readability
- Batch feature extraction mode (GPU model persistence)

### 2. Configuration Changes

**New CLI Arguments:**
```bash
--feature-workers N   # Parallel workers for feature extraction (default: 8)
--crop-workers N      # Parallel workers for cropping (default: 6)
--rhythm-workers N    # Parallel workers for rhythm/beat detection (default: 4)
--demucs-workers N    # Parallel Demucs processes (default: 2, ~5GB VRAM each)
```

**YAML Config (master_pipeline.yaml):**
```yaml
demucs:
  workers: 2                     # Subprocess-based parallel processing

rhythm:
  workers: 4                     # Parallel beat/downbeat detection

cropping:
  workers: 6                     # Parallel cropping

processing:
  feature_workers: 8             # Parallel feature extraction

music_flamingo:
  model: Q8_0
  prompts:                       # Select which prompts to generate
    full: true
    technical: true
    genre_mood: true
    instrumentation: true
    structure: true
  max_tokens:                    # Max tokens per prompt type
    full: 500
    technical: 500
    genre_mood: 500
    instrumentation: 500
    structure: 500
```

### 3. Batch Feature Extraction Mode

New "hybrid batch" approach for crop analysis:
- **Pass 1:** Light CPU features (Loudness, Spectral, Rhythm, Chroma) - one pass per file
- **Pass 2:** AudioBox Aesthetics - load model once, process all
- **Pass 3:** Essentia Classification - load model once, process all
- **Pass 4:** Music Flamingo - load model once, process all
- **Pass 5:** MIDI Transcription

Controlled by `batch_feature_extraction: true` in config.

### 4. Music Flamingo Config Fixes

- Prompts now respect config settings (was ignoring and running all 5)
- Token limits per prompt type are now configurable
- Model selection (IQ3_M, Q6_K, Q8_0) via config

### 5. Metadata Extraction (Stage 1c)

Extracts MP3/FLAC metadata at pipeline start:
- `track_metadata_artist`
- `track_metadata_title`
- `track_metadata_album`
- `track_metadata_year`
- `track_metadata_genre`

Optionally renames "Various Artists" folders using extracted metadata.

## Key Files Modified

- `src/master_pipeline.py` (parallel processing, batch mode, config parsing)
- `src/pipeline.py` (batch feature extraction, flamingo config)
- `src/classification/music_flamingo.py` (token limits, prompt filtering)
- `src/rhythm/beat_grid.py` (parallel processing with workers)
- `src/tools/create_training_crops.py` (resampling, BPM fix, rhythm slicing)
- `src/preprocessing/file_organizer.py` (cleaned filename check)
- `src/preprocessing/demucs_sep.py` (128kbps default)
- `src/preprocessing/demucs_sep_optimized.py` (128kbps default)
- `config/master_pipeline.yaml` (all new parallel/flamingo settings)
- `requirements.txt` (added mutagen)

## Dependencies Added
```bash
pip install mutagen  # MP3/FLAC metadata extraction (optional)
```

## Previous Sessions
- See `SESSION_SUMMARY_2026-01-26.md` for full details
- See `SESSION_SUMMARY_2026-01-23.md` for crop generation upgrades
- See `SESSION_SUMMARY_2026-01-22.md` for earlier work
