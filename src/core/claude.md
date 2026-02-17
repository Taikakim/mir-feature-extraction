# Core Module Documentation

## Overview
The `core` module provides foundational utilities for the MIR (Music Information Retrieval) project. It handles JSON file operations, file path management, and shared constants used throughout the pipeline.

## Purpose
This module serves as the foundation for all other modules in the MIR project. It ensures:
- Safe, atomic JSON operations that never destroy existing data
- Consistent file path handling across the project
- Centralized configuration for feature ranges and pipeline settings
- Proper logging and error handling

## Files in This Module

### `json_handler.py`
Provides safe JSON read/write operations for `.INFO` files.

**Key Functions:**
- `read_info(file_path)` - Read .INFO file, returns empty dict if not exists
- `write_info(file_path, data, merge=True)` - Write to .INFO with optional merge
- `safe_update(file_path, updates)` - Atomic merge of new keys into .INFO file
- `get_info_path(audio_file)` - Get .INFO path for an audio file
- `should_process(info_path, keys, overwrite)` - Check if feature extraction is needed

Note: `.MIR` file functions (`read_mir`, `write_mir`, `get_mir_path`) exist but are unused.

**Critical Features:**
- **Atomic writes**: Uses temporary file + rename to prevent corruption
- **Merge by default**: Preserves existing keys when adding new features
- **Error handling**: Comprehensive logging and exception handling

**Usage Example:**
```python
from core.json_handler import read_info, safe_update

# Read existing info
info = read_info("/path/to/song/song.INFO")

# Add new features without destroying existing ones
new_features = {
    "lufs": -14.5,
    "danceability": 0.85
}
safe_update("/path/to/song/song.INFO", new_features)
```

**Dependencies:**
- None (standard library only)

**Depended On By:**
- All feature extraction modules (rhythm, spectral, harmonic, etc.)
- Conditioner formatting module
- Statistics analyzer

---

### `file_utils.py`
Handles file and path operations for the MIR pipeline.

**Key Functions:**
- `find_audio_files(directory, extensions, recursive)` - Find all audio files
- `get_audio_folder_structure(audio_file)` - Get organized folder structure info
- `is_organized(audio_file)` - Check if file is in organized structure
- `get_stem_files(audio_folder)` - Get all Demucs/DrumSep stems
- `get_grid_files(audio_folder)` - Get beat/onset grid files
- `get_chroma_file(audio_folder)` - Get chroma data file
- `calculate_position_in_file(clip_path, duration, start_time)` - Calculate position feature
- `parse_section_number(filename)` - Parse section number from crop filename
- `find_organized_folders(root_directory)` - Find all organized folders
- `get_midi_files(audio_folder, stem)` - Get MIDI transcriptions

**Key Constants:**
- `AUDIO_EXTENSIONS` - Supported audio formats
- `DEMUCS_STEMS` - ['drums', 'bass', 'other', 'vocals']
- `DRUMSEP_STEMS` - ['kick', 'snare', 'cymbals', 'toms', 'percussion']

**Usage Example:**
```python
from core.file_utils import find_organized_folders, get_stem_files

# Find all organized songs
folders = find_organized_folders("/path/to/dataset")

# Get stems for a song
stems = get_stem_files("/path/to/song", include_full_mix=True)
# Returns: {'full_mix': Path(...), 'drums': Path(...), ...}
```

**Dependencies:**
- None (standard library only)

**Depended On By:**
- File organizer script
- All feature extraction modules
- Batch processing scripts

---

### `common.py`
Shared constants, configuration, and utility functions.

**Key Constants:**
- `PROJECT_ROOT` - Project root directory
- `AUDIO_EXTENSIONS` - Supported audio file extensions
- `DEMUCS_STEMS` / `DRUMSEP_STEMS` - Standard stem names
- `FEATURE_RANGES` - Min/max/type for all conditioning features
- `FREQUENCY_BANDS` - Multiband RMS frequency definitions
- `BEAT_TRACKING_CONFIG` - Beat detection thresholds
- `CHROMA_CONFIG` - Chroma extraction parameters
- `DEMUCS_CONFIG` - Demucs separation settings
- `SPECTRAL_CONFIG` - Spectral analysis parameters

**Key Functions:**
- `setup_logging(level, log_file)` - Configure project-wide logging
- `get_feature_range(feature_name)` - Get min/max/type for a feature
- `clamp_feature_value(feature_name, value)` - Clamp value to valid range

**Feature Ranges:**
All conditioning features are defined here with their value ranges. Examples:
- `lufs`: -40.0 to 0.0 dB
- `bpm`: 40.0 to 300.0
- `brightness`: 0.0 to 100.0
- `danceability`: 0.0 to 1.0

**Usage Example:**
```python
from core.common import get_feature_range, clamp_feature_value

# Get valid range for a feature
range_info = get_feature_range('bpm')
# Returns: {'min': 40.0, 'max': 300.0, 'type': 'number'}

# Clamp a value
bpm = clamp_feature_value('bpm', 350)  # Returns 300.0 (max)
```

**Dependencies:**
- None (standard library only)

**Depended On By:**
- All modules in the project
- Configuration management
- Feature validation

---

## Integration with Other Modules

### How Preprocessing Uses Core:
```python
from core.file_utils import find_audio_files, get_audio_folder_structure
from core.json_handler import safe_update

# Find files to process
audio_files = find_audio_files("/dataset")

# Process and save results
for audio_file in audio_files:
    # ... do processing ...
    results = {"lufs": -14.5}
    info_path = get_info_path(audio_file)
    safe_update(info_path, results)
```

### How Feature Extraction Uses Core:
```python
from core.file_utils import get_stem_files
from core.json_handler import safe_update
from core.common import FREQUENCY_BANDS, clamp_feature_value

# Get stems
stems = get_stem_files(audio_folder)

# Extract feature
brightness = calculate_brightness(stems['full_mix'])
brightness = clamp_feature_value('brightness', brightness)

# Save result
safe_update(info_path, {'brightness': brightness})
```

## File Format Specifications

### .INFO File Format
JSON file containing scalar feature values for conditioning:
```json
{
  "lufs": -14.5,
  "lra": 8.2,
  "bpm": 128,
  "bpm_is_defined": 1,
  "brightness": 75.3,
  "danceability": 0.85,
  "position_in_file": 0.5
}
```

### Grid File Formats
Plain text files with timestamps (one per line):

**BEATS_GRID:**
```
0.0
0.468
0.937
1.405
...
```

**ONSETS_GRID:**
```
0.123
0.234
0.456
...
```

## Best Practices

### When Adding New Features:
1. Add feature range to `FEATURE_RANGES` in `common.py`
2. Use `safe_update()` to write feature values (never `write_info()` with `merge=False`)
3. Clamp values using `clamp_feature_value()` before saving
4. Update this documentation with the new feature

### When Moving/Renaming Functions:
1. Check all imports across the project
2. Update function docstrings
3. Update this claude.md file

### Error Handling:
- All functions log errors appropriately
- File operations use try/except with cleanup
- Invalid paths return empty dicts/lists rather than raising exceptions
- JSON errors are propagated (should not be silently ignored)

## Testing
Each module includes a `__main__` section with basic tests. Run with:
```bash
python -m src.core.json_handler
python -m src.core.file_utils
python -m src.core.common
```


