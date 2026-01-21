# MIR Feature Extraction Framework - User Manual

**Version:** 1.3
**Last Updated:** 2026-01-22
**For:** Stable Audio Tools conditioning data preparation

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [File Organization](#file-organization)
5. [Feature Extraction Workflows](#feature-extraction-workflows)
6. [Module Reference](#module-reference)
7. [Music Flamingo AI Descriptions](#music-flamingo-ai-descriptions)
8. [Output Files](#output-files)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## Overview

This framework extracts 78 numeric MIR features + 5 natural language AI descriptions from audio files for use as conditioning data in Stable Audio Tools training. It processes full mixes and separated stems to capture rhythm, harmony, timbre, loudness, aesthetic characteristics, and AI-generated music descriptions.

### What It Does

- **Organizes** audio files into structured folders
- **Separates** audio into stems (drums, bass, other, vocals) using Demucs htdemucs_ft
- **Extracts** 78 numeric conditioning features per track
- **Generates** 5 AI text descriptions via Music Flamingo (GGUF or Transformers)
- **Transcribes** drums to MIDI using ADTOF-PyTorch (GPU-accelerated)
- **Saves** results in JSON `.INFO` files and auxiliary grid files

### What You Get

For each audio track:
- 78 numeric features in `{trackname}.INFO` JSON file
- 5 AI text descriptions (genre, mood, instruments, structure, technical)
- 4 separated stems (drums, bass, other, vocals) as MP3 files (~96kbps VBR, for feature extraction)
- Beat grid in `{trackname}.BEATS_GRID` file
- Onset timestamps in `{trackname}.ONSETS` file
- MIDI drum transcription in `drums_adtof.mid`
- Ready for Stable Audio Tools training pipeline

---

## Installation

### Prerequisites

- **Python:** 3.12+ (tested with 3.12)
- **NumPy:** >=2.0.0, <2.4 (pinned for numba compatibility)
- **GPU:** AMD (ROCm 7.1+) or NVIDIA (CUDA) recommended for stem separation and Music Flamingo
- **Disk Space:** ~50MB per minute of audio (for stems + features)
- **VRAM:** 13GB+ for Music Flamingo AI descriptions

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/Taikakim/mir-feature-extraction
cd mir-feature-extraction

# Create virtual environment
python -m venv mir
source mir/bin/activate  # On Windows: mir\Scripts\activate

# Install requirements (note: numpy pinned <2.4 for numba compatibility)
pip install -r requirements.txt

# Setup external repositories (Audio Commons timbral models)
bash scripts/setup_external_repos.sh

# For Music Flamingo GGUF (recommended - 7x faster):
# Build llama.cpp with HIP support (see MUSIC_FLAMINGO_QUICKSTART.md)
```

### External Dependencies

The following are required:
- **timbral_models** - Audio Commons timbral features (cloned via setup script)
- **Demucs** - Stem separation (installed via pip)
- **llama.cpp** - For Music Flamingo GGUF inference (build from source)

### Essential Environment Variables

```bash
# For GPU optimization (AMD ROCm)
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0
export OMP_NUM_THREADS=8
```

**Note:** External patches for librosa 0.11+ and NumPy 2.x compatibility are documented in `EXTERNAL_PATCHES.md`.

---

## Quick Start

### 1. Organize Your Audio Files

```bash
# Place audio files in a directory (e.g., my_music/)
# Supported formats: .mp3, .wav, .flac, .ogg, .m4a

python src/preprocessing/file_organizer.py my_music/
```

**Result:** Each audio file moved into its own folder with standardized naming.

### 2. Separate Stems

```bash
# Separate all tracks using GPU
python src/preprocessing/demucs_sep.py my_music/ --batch --device cuda

# Or process single track
python src/preprocessing/demucs_sep.py "my_music/Artist - Album - Track/"
```

**Result:** Each folder contains 4 stem files (drums.mp3, bass.mp3, other.mp3, vocals.mp3)

### 3. Extract All Features

```bash
# Test all features on a single file first
python src/test_all_features.py "my_music/Artist - Track/full_mix.flac"

# Or run complete pipeline on all folders
python scripts/extract_all_features.py my_music/ --batch
```

**Result:** Each folder contains `{trackname}.INFO` with 78 numeric features.

### 4. Add Music Flamingo AI Descriptions (Optional)

```bash
# Using GGUF (recommended - 7x faster, 40-60% less VRAM)
python src/classification/music_flamingo.py my_music/ --batch --model Q6_K

# Or using Transformers (slower but more flexible)
python src/classification/music_flamingo_transformers.py my_music/ --batch --flash-attention
```

**Result:** Each `.INFO` file gains 5 AI text description fields.

### Alternative: Step-by-Step Extraction

If you prefer manual control:

```bash
# 1. Beat grid and rhythm
python src/rhythm/beat_grid.py my_music/ --batch
python src/rhythm/bpm.py my_music/ --batch
python src/rhythm/onsets.py my_music/ --batch
python src/rhythm/syncopation.py my_music/ --batch

# 2. Loudness
python src/timbral/loudness.py my_music/ --batch

# 3. Spectral features
python src/spectral/spectral_features.py my_music/ --batch
python src/spectral/multiband_rms.py my_music/ --batch

# 4. Harmonic features
python src/harmonic/chroma.py my_music/ --batch
python src/harmonic/per_stem_harmonic.py my_music/ --batch

# 5. Timbral features
python src/timbral/audio_commons.py my_music/ --batch

# 6. Classification
python src/classification/essentia_features.py my_music/ --batch

# 7. Per-stem features
python src/rhythm/per_stem_rhythm.py my_music/ --batch

# 8. Music Flamingo AI descriptions (optional)
python src/classification/music_flamingo.py my_music/ --batch --model Q6_K
```

---

## File Organization

### Input Structure

**Before organization:**
```
my_music/
├── track1.mp3
├── track2.flac
└── track3.wav
```

**After file_organizer.py:**
```
my_music/
├── Artist1 - Album1 - Track1/
│   └── full_mix.mp3
├── Artist2 - Album2 - Track2/
│   └── full_mix.flac
└── Artist3 - Album3 - Track3/
    └── full_mix.wav
```

### Output Structure

**After complete processing:**
```
my_music/
└── Artist - Album - Track/
    ├── full_mix.flac             # Original audio (organized)
    ├── drums.flac                # Separated stems
    ├── bass.flac
    ├── other.flac
    ├── vocals.flac
    ├── Artist - Album - Track.INFO         # 78 features + 5 AI descriptions (JSON)
    ├── Artist - Album - Track.BEATS_GRID   # Beat timestamps
    ├── Artist - Album - Track.ONSETS       # Onset timestamps
    └── separated/                 # Demucs working directory
        └── htdemucs/
            └── ...
```

### .INFO File Format

JSON file containing 78 numeric features + 5 AI text descriptions:

```json
{
  "bpm": 142.19,
  "bpm_is_defined": 1,
  "beat_count": 1162,
  "lufs": -12.73,
  "lra": 3.15,
  "brightness": 58.64,
  "chroma_0": 0.565,
  "music_flamingo_full": "This track is an energetic electronic dance music piece...",
  "music_flamingo_genre_mood": "Genre: Electronic/Trance. Mood: Euphoric, uplifting...",
  ...
}
```

### .BEATS_GRID File Format

JSON file with beat timestamps in seconds:

```json
{
  "beats": [0.123, 0.545, 0.967, 1.389, ...],
  "bpm": 142.19,
  "total_beats": 1162
}
```

---

## Feature Extraction Workflows

### Workflow 1: New Dataset (Full Pipeline)

For a completely new dataset starting from raw audio files:

```bash
# Step 1: Organize files
python src/preprocessing/file_organizer.py /path/to/audio/

# Step 2: Separate stems (GPU recommended)
python src/preprocessing/demucs_sep.py /path/to/audio/ --batch --device cuda

# Step 3: Extract all features
python scripts/extract_all_features.py /path/to/audio/ --batch

# Step 4: Verify results
python scripts/verify_features.py /path/to/audio/
```

**Estimated time:** ~2-3 minutes per track (with GPU)

### Workflow 2: Re-extract Features (Keep Stems)

If stems already exist and you want to re-run feature extraction:

```bash
# Extract specific feature set
python src/rhythm/rhythm_analysis.py /path/to/audio/ --batch --overwrite
python src/timbral/audio_commons.py /path/to/audio/ --batch --overwrite

# Or re-run everything
python scripts/extract_all_features.py /path/to/audio/ --batch --overwrite
```

**Estimated time:** ~30-60 seconds per track

### Workflow 3: Single Track Processing

For testing or debugging individual tracks:

```bash
# Process one track through entire pipeline
TRACK="my_music/Artist - Album - Track"

# Organize (if needed)
python src/preprocessing/file_organizer.py "$TRACK"

# Separate stems
python src/preprocessing/demucs_sep.py "$TRACK" --device cuda

# Extract features
python src/rhythm/rhythm_analysis.py "$TRACK"
python src/preprocessing/loudness.py "$TRACK"
python src/spectral/spectral_features.py "$TRACK"
python src/harmonic/chroma_analysis.py "$TRACK"
python src/harmonic/per_stem_harmonic.py "$TRACK"
python src/timbral/audio_commons.py "$TRACK"
python src/classification/essentia_features.py "$TRACK"
python src/rhythm/per_stem_rhythm.py "$TRACK"
```

### Workflow 4: Update Missing Features

If some tracks are missing specific features:

```bash
# Check what's missing
python scripts/verify_features.py /path/to/audio/ --show-missing

# Extract only missing features
python src/timbral/audio_commons.py /path/to/audio/ --batch --skip-existing
```

---

## Module Reference

### Preprocessing

#### file_organizer.py
**Purpose:** Organize audio files into folder structure

```bash
python src/preprocessing/file_organizer.py <input_dir>

Options:
  <input_dir>     Directory containing audio files
```

**Output:** Organized folder structure with `full_mix.*` files

---

#### demucs_sep.py
**Purpose:** Separate audio into stems using Demucs HT v4

```bash
python src/preprocessing/demucs_sep.py <audio_path> [options]

Options:
  --batch              Process all folders in directory
  --device DEVICE      Device: cuda, cpu (default: cpu)
  --model MODEL        Model name (default: htdemucs)
  --shifts N           Random shifts for ensemble (default: 1)
  --jobs N             Parallel jobs (default: 0 = auto)
  --overwrite          Re-separate even if stems exist

GPU Performance:
  AMD (ROCm): Use --device cuda  (~9.4x realtime)
  NVIDIA:     Use --device cuda  (~10-15x realtime)
  CPU:        Very slow (~0.1x realtime)
```

**Output:** 4 stems (drums.mp3, bass.mp3, other.mp3, vocals.mp3) @ 320kbps

**Notes:**
- MP3 output used due to TorchCodec/FFmpeg compatibility issues
- 320kbps MP3 is sufficient quality for MIR analysis
- For AMD GPUs with ROCm, use `--device cuda` (not `--device amd`)

---

#### loudness.py
**Purpose:** Calculate LUFS/LRA loudness for full mix and stems

```bash
python src/preprocessing/loudness.py <audio_path> [options]

Options:
  --batch              Process all folders in directory
  --overwrite          Recalculate even if features exist
```

**Features Extracted:** 10
- `lufs`, `lra` (full mix)
- `lufs_drums`, `lra_drums`
- `lufs_bass`, `lra_bass`
- `lufs_other`, `lra_other`
- `lufs_vocals`, `lra_vocals`

**Standard:** ITU-R BS.1770 integrated loudness

---

### Rhythm Analysis

#### rhythm_analysis.py
**Purpose:** Extract tempo, beats, syncopation, onset features

```bash
python src/rhythm/rhythm_analysis.py <audio_path> [options]

Options:
  --batch              Process all folders
  --overwrite          Recalculate features
```

**Features Extracted:** 14
- `bpm`, `bpm_is_defined`, `beat_count`, `beat_regularity`
- `syncopation`, `on_beat_ratio`
- `onset_count`, `onset_density`, `onset_strength_mean`, `onset_strength_std`
- `rhythmic_complexity`, `rhythmic_evenness`
- `ioi_mean`, `ioi_std`

**Auxiliary Files:** `.BEATS_GRID`

---

#### per_stem_rhythm.py
**Purpose:** Extract rhythm features per stem

```bash
python src/rhythm/per_stem_rhythm.py <audio_path> [options]

Options:
  --batch              Process all folders
  --overwrite          Recalculate features
```

**Features Extracted:** 15 (5 per stem for bass/drums/other)
- `onset_density_average_{stem}`, `onset_density_variance_{stem}`
- `syncopation_{stem}`
- `rhythmic_complexity_{stem}`, `rhythmic_evenness_{stem}`

**Requirements:** Needs `.BEATS_GRID` from rhythm_analysis.py

---

### Spectral Analysis

#### spectral_features.py
**Purpose:** Extract spectral characteristics and energy distribution

```bash
python src/spectral/spectral_features.py <audio_path> [options]

Options:
  --batch              Process all folders
  --overwrite          Recalculate features
```

**Features Extracted:** 8
- `spectral_flatness` - Noise vs tone (0-1)
- `spectral_flux` - Spectral change rate (0-3)
- `spectral_skewness` - Frequency bias (-3 to 3)
- `spectral_kurtosis` - Energy concentration (0-10)
- `rms_energy_bass` - 20-120 Hz (dB)
- `rms_energy_body` - 120-600 Hz (dB)
- `rms_energy_mid` - 600-2500 Hz (dB)
- `rms_energy_air` - 2500-22000 Hz (dB)

---

### Harmonic Analysis

#### chroma_analysis.py
**Purpose:** Extract chromagram pitch content

```bash
python src/harmonic/chroma_analysis.py <audio_path> [options]

Options:
  --batch              Process all folders
  --overwrite          Recalculate features
```

**Features Extracted:** 12
- `chroma_0` through `chroma_11` (C, C#, D, ... B)
- 12-dimensional pitch class weights (0-1)

---

#### per_stem_harmonic.py
**Purpose:** Calculate harmonic movement and variance per stem

```bash
python src/harmonic/per_stem_harmonic.py <audio_path> [options]

Options:
  --batch              Process all folders
  --overwrite          Recalculate features
```

**Features Extracted:** 4
- `harmonic_movement_bass`, `harmonic_variance_bass`
- `harmonic_movement_other`, `harmonic_variance_other`

**Method:** Cosine distance between consecutive chroma frames

---

### Timbral Analysis

#### audio_commons.py
**Purpose:** Extract perceptual timbral characteristics

```bash
python src/timbral/audio_commons.py <audio_path> [options]

Options:
  --batch              Process all folders
  --overwrite          Recalculate features
```

**Features Extracted:** 8
- `brightness` - High-frequency content (0-100)
- `roughness` - Beating/modulation (0-100)
- `hardness` - Attack sharpness (0-100)
- `depth` - Low-frequency spaciousness (0-100)
- `booming` - 100-200 Hz resonance (0-100)
- `reverberation` - Wet/dry balance (0-100)
- `sharpness` - High-frequency harshness (0-100)
- `warmth` - Mid-low richness (0-100)

**Library:** Audio Commons timbral_models (patched)

**Note:** Requires patches applied (see EXTERNAL_PATCHES.md)

---

### Classification

#### essentia_features.py
**Purpose:** Extract high-level classification features

```bash
python src/classification/essentia_features.py <audio_path> [options]

Options:
  --batch              Process all folders
  --overwrite          Recalculate features
```

**Features Extracted:** 6
- `danceability` - Rhythmic strength (0-1)
- `atonality` - Tonal departure (0-1)
- `content_enjoyment` - Aesthetic appeal (1-10) *
- `content_usefulness` - Functional value (1-10) *
- `production_complexity` - Production sophistication (1-10) *
- `production_quality` - Technical excellence (1-10) *

*Currently using default value (5.5) - AudioBox model not yet implemented

---

## MIDI Drum Transcription

The framework includes drum transcription to MIDI via two methods:

### Method 1: ADTOF-PyTorch (Recommended)

Neural network drum transcription using the ADTOF Frame_RNN model ported to PyTorch.

**Output:** 5 drum classes (Bass Drum, Snare, Tom, Hi-Hat, Cymbal) as General MIDI notes.

```bash
# Single file
python src/transcription/drums/adtof.py "/path/to/audio.flac" -v

# With custom output
python src/transcription/drums/adtof.py "/path/to/audio.flac" --output drums.mid

# Batch processing
python src/transcription/drums/adtof.py /path/to/data --batch --device cuda

# Use CPU instead of GPU
python src/transcription/drums/adtof.py /path/to/audio.flac --device cpu
```

**Requirements:** ADTOF-PyTorch installed (`pip install -e repos/ADTOF-pytorch`)

**Performance:**
- First run: ~5 min (Triton kernel compilation)
- Subsequent runs: GPU at 100%, ~1-2 min per 3-minute track
- Bottleneck: CPU-bound mel spectrogram computation (librosa)

### Method 2: Drumsep + Onset Detection

Separates drum stems then detects onsets per component.

```bash
# Full pipeline
python src/transcription/runner.py /path/to/data --force --verbose
```

**Output:** `drums.mid` file with kick, snare, cymbals, toms tracks.

**Stem Mapping:**
| Stem | MIDI Note | Instrument |
|------|-----------|------------|
| bombo (kick) | 36 | Bass Drum 1 |
| redoblante (snare) | 38 | Acoustic Snare |
| platillos (cymbals) | 42 | Closed Hi-Hat |
| toms | 45 | Low Tom |

---

## Music Flamingo AI Descriptions

Music Flamingo generates natural language descriptions of music using NVIDIA's Music Flamingo model (8B parameters: Qwen2.5-7B language + Audio Flamingo 3 encoder).

### Two Inference Methods

#### 1. GGUF via llama-mtmd-cli (Recommended)

**Performance:** ~4 seconds per track (7x faster than transformers)
**VRAM:** 40-60% less than transformers
**Models:** IQ3_M (3.4GB), Q6_K (5.9GB), Q8_0 (7.6GB)

```bash
# Single file
python src/classification/music_flamingo.py "my_music/Track Name/full_mix.flac" --model Q6_K

# Batch processing
python src/classification/music_flamingo.py my_music/ --batch --model Q6_K

# Available models (quality vs speed):
#   IQ3_M - Fastest, smallest, lower quality
#   Q6_K  - Good balance (recommended)
#   Q8_0  - Highest quality, largest
```

**Requirements:** llama.cpp built with HIP support (see `MUSIC_FLAMINGO_QUICKSTART.md`)

#### 2. Transformers with Flash Attention 2

**Performance:** ~30 seconds per track (bfloat16 + Flash Attention 2)
**VRAM:** ~13GB
**Note:** INT8/INT4 quantization NOT functional on ROCm

```bash
# Single file
python src/classification/music_flamingo_transformers.py "my_music/Track Name/full_mix.flac" --flash-attention

# Batch processing
python src/classification/music_flamingo_transformers.py my_music/ --batch --flash-attention
```

### Five Description Types

Each method generates 5 text descriptions saved to the `.INFO` file:

| Key | Description |
|-----|-------------|
| `music_flamingo_full` | Comprehensive description (genre, tempo, key, instruments, mood) |
| `music_flamingo_technical` | Technical analysis (tempo, key, chords, dynamics, performance) |
| `music_flamingo_genre_mood` | Genre classification and emotional character |
| `music_flamingo_instrumentation` | Instruments and sounds present |
| `music_flamingo_structure` | Arrangement and structure analysis |

### Text Normalization

All Music Flamingo output is automatically normalized for T5 tokenizer compatibility (required for Stable Audio Tools). This replaces Unicode characters that break T5:
- Em-dashes (—) → hyphens (-)
- Curly quotes ('') → straight quotes ('')
- Non-breaking hyphens (‑) → regular hyphens (-)

---

## Output Files

### {trackname}.INFO

Main feature file in JSON format containing 78 numeric features + 5 AI text descriptions.

**Example:**
```json
{
  "bpm": 142.19,
  "bpm_is_defined": 1,
  "beat_count": 1162,
  "beat_regularity": 0.011,
  "lufs": -12.73,
  "lra": 3.15,
  "lufs_drums": -16.05,
  "lra_drums": 13.04,
  ...
}
```

**Usage:** Load for Stable Audio Tools conditioning

```python
import json
from pathlib import Path

info_file = Path("track_folder/trackname.INFO")
with open(info_file) as f:
    features = json.load(f)

bpm = features["bpm"]
loudness = features["lufs"]
```

### {trackname}.BEATS_GRID

Beat timestamps for temporal alignment.

**Example:**
```json
{
  "beats": [0.123, 0.545, 0.967, 1.389, ...],
  "bpm": 142.19,
  "total_beats": 1162
}
```

**Usage:** MIDI quantization, visual debugging, smart cropping

### {trackname}.ONSETS

Onset timestamps in seconds (newline-separated text file).

**Usage:** Syncopation analysis, rhythmic complexity calculation

### Stem Files

**Format:** FLAC (lossless) or MP3 @ 320kbps
**Files:** drums.flac, bass.flac, other.flac, vocals.flac

**Usage:** Per-stem feature extraction, separate conditioning, MIDI transcription

---

## Troubleshooting

### Common Issues

#### 1. TorchCodec/FFmpeg Errors

**Error:**
```
RuntimeError: Could not load libtorchcodec
```

**Solution:** Already fixed - framework uses MP3 output which bypasses TorchCodec

---

#### 2. Audio Commons onset_detect() Errors

**Error:**
```
TypeError: onset_detect() takes 0 positional arguments but 2 positional arguments were given
```

**Solution:** The timbral_models library has been patched. If you see this error, re-run:
```bash
bash scripts/setup_external_repos.sh
```

This applies librosa 0.11.0 API compatibility patches to timbral_models.

---

#### 3. NumPy/Numba Version Conflict

**Error:**
```
Numba needs NumPy 2.3 or less. Got NumPy 2.4
```

**Solution:** Pin numpy to a compatible version:
```bash
pip install "numpy>=2.0.0,<2.4"
```

This is already handled in `requirements.txt`. If you upgraded numpy separately, downgrade it.

---

#### 4. numpy.lib.pad AttributeError

**Error:**
```
AttributeError: module 'numpy.lib' has no attribute 'pad'
```

**Solution:** The timbral_models library has been patched for NumPy 2.x. If you see this error:
```bash
sed -i 's/np\.lib\.pad/np.pad/g' \
  repos/repos/timbral_models/timbral_models/Timbral_Roughness.py \
  repos/repos/timbral_models/timbral_models/Timbral_Hardness.py
```

See `EXTERNAL_PATCHES.md` for full documentation.

---

#### 5. AMD GPU Not Detected

**Error:** Demucs running slow on CPU despite having AMD GPU

**Solution:** Use `--device cuda` (not `--device amd`) for ROCm:
```bash
python src/preprocessing/demucs_sep.py audio/ --batch --device cuda
```

AMD GPUs with ROCm appear as CUDA devices in PyTorch.

---

#### 6. Missing Stems Error

**Error:**
```
ERROR: Cannot find stem files for per-stem analysis
```

**Solution:** Run stem separation first:
```bash
python src/preprocessing/demucs_sep.py <audio_path>
```

Per-stem features require drums/bass/other stems.

---

#### 7. Incomplete Features

**Problem:** Some tracks have fewer features than others

**Diagnosis:**
```bash
python scripts/verify_features.py /path/to/audio/ --show-missing
```

**Solution:** Re-run specific missing modules:
```bash
python src/timbral/audio_commons.py /path/to/audio/ --batch --overwrite
```

---

### Performance Issues

#### Slow Stem Separation

**Symptoms:** Demucs taking 60+ minutes per 10-minute track

**Solutions:**
1. Use GPU: `--device cuda`
2. Reduce quality: `--shifts 0` (default is 1)
3. Use fewer jobs: `--jobs 1`

**Expected Performance:**
- GPU (CUDA/ROCm): ~9-15x realtime (10min track in 40-60 seconds)
- CPU: ~0.1x realtime (10min track in 60+ minutes)

#### Memory Errors

**Symptoms:** Out of memory during processing

**Solutions:**
1. Process one track at a time (remove `--batch`)
2. Reduce Demucs shifts: `--shifts 0`
3. Close other applications

---

## Advanced Usage

### Custom Feature Extraction

To extract only specific feature sets:

```python
from core.json_handler import read_json, write_json
from rhythm.rhythm_analysis import analyze_rhythm
from pathlib import Path

# Load existing features
info_file = Path("track/track.INFO")
features = read_json(info_file)

# Extract custom features
new_features = analyze_rhythm(Path("track/full_mix.mp3"))

# Merge and save
features.update(new_features)
write_json(info_file, features)
```

### Batch Processing with Filtering

Process only tracks matching criteria:

```bash
# Process only tracks with "trance" in name
find my_music/ -type d -name "*trance*" | while read dir; do
    python src/preprocessing/demucs_sep.py "$dir" --device cuda
done
```

### Feature Verification Script

Create `scripts/verify_features.py`:

```python
#!/usr/bin/env python3
import json
from pathlib import Path
from core.file_utils import get_organized_folders

def verify_features(root_dir):
    folders = get_organized_folders(root_dir)

    for folder in folders:
        info_file = get_info_path(folder)
        if not info_file.exists():
            print(f"❌ Missing .INFO: {folder.name}")
            continue

        with open(info_file) as f:
            features = json.load(f)

        feature_count = len(features)
        if feature_count < 78:
            print(f"⚠️  {folder.name}: {feature_count}/78+ features")
        else:
            print(f"✅ {folder.name}: {feature_count} features")

if __name__ == "__main__":
    import sys
    verify_features(Path(sys.argv[1]))
```

### Parallel Processing

For large datasets, process in parallel:

```bash
# Process 4 tracks simultaneously
find my_music/ -maxdepth 1 -type d | \
  parallel -j 4 "python src/preprocessing/demucs_sep.py {} --device cuda"
```

---

## Best Practices

### Dataset Preparation

1. **Organize first:** Always run `file_organizer.py` before processing
2. **Verify stems:** Check stem quality before feature extraction
3. **Process in stages:** Separate stems for all tracks, then extract features
4. **Keep backups:** Copy .INFO files regularly during long processing runs

### Performance Optimization

1. **Use GPU:** Always use `--device cuda` for stem separation
2. **Batch processing:** Use `--batch` for multiple tracks
3. **Skip existing:** Use `--skip-existing` to avoid reprocessing
4. **Parallel jobs:** For CPU, use `--jobs 0` (auto) for Demucs

### Quality Control

1. **Verify features:** Check feature counts with verification script
2. **Inspect stems:** Listen to separated stems for quality
3. **Check beats:** Visualize `.BEATS_GRID` for tempo detection accuracy
4. **Statistical analysis:** Run corpus statistics before training

---

## Reference

### Complete Feature List (78 Numeric + 5 Text)

**Rhythm (29):**
bpm, bpm_is_defined, beat_count, beat_regularity, syncopation, on_beat_ratio, onset_count, onset_density, onset_strength_mean, onset_strength_std, rhythmic_complexity, rhythmic_evenness, ioi_mean, ioi_std, onset_density_average_bass, onset_density_average_drums, onset_density_average_other, onset_density_variance_bass, onset_density_variance_drums, onset_density_variance_other, syncopation_bass, syncopation_drums, syncopation_other, rhythmic_complexity_bass, rhythmic_complexity_drums, rhythmic_complexity_other, rhythmic_evenness_bass, rhythmic_evenness_drums, rhythmic_evenness_other

**Loudness (10):**
lufs, lra, lufs_drums, lra_drums, lufs_bass, lra_bass, lufs_other, lra_other, lufs_vocals, lra_vocals

**Spectral (4):**
spectral_flatness, spectral_flux, spectral_skewness, spectral_kurtosis

**RMS Energy (4):**
rms_energy_bass, rms_energy_body, rms_energy_mid, rms_energy_air

**Chroma (12):**
chroma_0, chroma_1, chroma_2, chroma_3, chroma_4, chroma_5, chroma_6, chroma_7, chroma_8, chroma_9, chroma_10, chroma_11

**Harmonic (4):**
harmonic_movement_bass, harmonic_movement_other, harmonic_variance_bass, harmonic_variance_other

**Timbral (8):**
brightness, roughness, hardness, depth, booming, reverberation, sharpness, warmth

**Aesthetics (4):**
content_enjoyment, content_usefulness, production_complexity, production_quality

**Classification (2):**
danceability, atonality

**Position (1):**
position (0-1, only for cropped training data - not yet implemented)

**Music Flamingo AI Descriptions (5 Text):**
music_flamingo_full, music_flamingo_technical, music_flamingo_genre_mood, music_flamingo_instrumentation, music_flamingo_structure

---

## Documentation

- **FEATURES_STATUS.md** - Complete feature implementation status
- **EXTERNAL_PATCHES.md** - External repository modifications
- **MUSIC_FLAMINGO_QUICKSTART.md** - Music Flamingo GGUF setup guide
- **MUSIC_FLAMINGO_README.md** - Detailed Music Flamingo documentation
- **CLAUDE.md** - AI assistant guidance for codebase
- **project.log** - Development history and decisions
- **IMPLEMENTATION_PLAN.md** - Original implementation plan
- **README.md** - Project overview

---

## Support

For issues, bugs, or feature requests:
- Check `project.log` for known issues
- Review `FEATURES_STATUS.md` for implementation status
- Consult `EXTERNAL_PATCHES.md` for external dependencies
- See `MUSIC_FLAMINGO_QUICKSTART.md` for Music Flamingo setup
- Open issue on GitHub repository

---

**Framework Version:** 1.2
**Compatible with:** Stable Audio Tools (Stability AI)
**Last Updated:** 2026-01-21
