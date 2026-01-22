# MIR Feature Extraction Framework

!==! Very much a hot work-in-progress mess, but it alread kind of works for me and my RX 9070 XT, so I thought, it could be a good starting point for other people too. !==!

!==! Things are going fast, the bits and pieces that connect everything are probably not all up to date all the time, but I try to check that the individual core analysis scripts keep working !==!

!=! There's lots of auxilirary notes etc that will come and go, but the readme and manual should have everything that matters. The scripts also have their own help systems !==!

Main thing missing still is what to do with all of this raw data: pre-encode latents and create training prompts and new conditionings.

**Comprehensive music feature extraction pipeline for conditioning Stable Audio Tools and similar audio generation models.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: TBD](https://img.shields.io/badge/license-TBD-lightgrey.svg)](LICENSE)

---

## Overview

This framework extracts **77 music information retrieval (MIR) features** from audio files, specifically designed as conditioning data for Stable Audio Tools training. It analyzes both full mixes and separated stems to capture comprehensive musical characteristics.

### Key Features

✅ **77+ Numeric Features** extracted per track
✅ **AI-Powered Descriptions** using Music Flamingo (8B params)
✅ **Genre/Mood/Instrument Classification** (400 genres, 56 moods, 40 instruments)
✅ **4-Stem Separation** using Demucs HT v4 (drums, bass, other, vocals)
✅ **GPU Accelerated** processing (AMD ROCm / NVIDIA CUDA)
✅ **MIDI Drum Transcription** using ADTOF-PyTorch or Drumsep
✅ **Batch Processing** with automatic organization
✅ **Safe JSON Updates** with feature merging
✅ **Production Ready** - optimized for AMD RDNA4 (RX 9070 XT)

---

## Feature Categories

### Numeric Features (77+)

| Category | Features | Description |
|----------|----------|-------------|
| **Rhythm** | 29 | BPM, beats, syncopation, onset density, complexity (full + per-stem) |
| **Loudness** | 10 | LUFS/LRA for full mix + 4 stems (ITU-R BS.1770) |
| **Spectral** | 4 | Flatness, flux, skewness, kurtosis |
| **RMS Energy** | 4 | Bass, body, mid, air frequency bands (dB) |
| **Chroma** | 12 | 12-dimensional pitch class weights |
| **Harmonic** | 4 | Movement and variance for bass/other stems |
| **Timbral** | 8 | Audio Commons perceptual features |
| **Aesthetics** | 4 | Content enjoyment/usefulness, production complexity/quality |
| **Classification** | 2 | Danceability, atonality |

### AI Classification Features

| Category | Labels | Description |
|----------|--------|-------------|
| **Genre** | 400 | Discogs taxonomy (Blues, Electronic, Rock, Jazz, etc.) |
| **Mood/Theme** | 56 | MTG Jamendo (energetic, calm, dark, happy, epic, etc.) |
| **Instrument** | 40 | MTG Jamendo (guitar, drums, piano, synthesizer, etc.) |

### Natural Language Descriptions (Music Flamingo)

| Prompt Type | Description |
|-------------|-------------|
| **full** | Comprehensive description (genre, tempo, key, instruments, mood) |
| **technical** | Tempo, key, chords, dynamics, performance analysis |
| **genre_mood** | Genre classification + emotional character |
| **instrumentation** | Instruments and sounds present |
| **structure** | Arrangement and structure analysis |

**Total: 77+ numeric features + 496 classification labels + 5 AI descriptions + MIDI drums**

### Feature Units Reference

| Feature | Unit | Range | Description |
|---------|------|-------|-------------|
| **bpm** | BPM | 40-300 | Beats per minute |
| **beat_regularity** | seconds | 0-1 | Std dev of beat intervals (lower = more regular) |
| **onset_density** | onsets/sec | 0-50 | Note/percussion events per second |
| **onset_count** | count | 0-∞ | Total detected onset events |
| **lufs** | LUFS | -70 to 0 | Integrated loudness (ITU-R BS.1770) |
| **lra** | LU | 0-25 | Loudness range (dynamic variation) |
| **spectral_flatness** | ratio | 0-1 | Noise-like (1) vs tonal (0) |
| **spectral_flux** | - | 0-∞ | Rate of spectral change |
| **spectral_skewness** | - | -∞ to ∞ | Asymmetry of spectral distribution |
| **spectral_kurtosis** | - | 0-∞ | Spectral peakedness |
| **rms_bass/body/mid/air** | dB | -60 to 0 | Energy in frequency bands |
| **chroma_C/C#/.../B** | weight | 0-1 | Pitch class presence (12 features) |
| **brightness** | score | 0-100 | High-frequency content perception |
| **roughness** | score | 0-100 | Harshness/beating perception |
| **hardness** | score | 0-100 | Soft vs metallic perception |
| **depth** | score | 0-100 | Low-frequency spaciousness |
| **booming** | score | 0-100 | Low-frequency resonance (100-200 Hz) |
| **reverberation** | score | 0-100 | Wet/dry balance |
| **sharpness** | score | 0-100 | High-frequency harshness |
| **warmth** | score | 0-100 | Mid-low frequency richness |
| **danceability** | probability | 0-1 | Dance suitability |
| **atonality** | probability | 0-1 | Absence of tonal center |
| **audiobox_ce/cu/pc/pq** | score | 1-10 | Enjoyment/usefulness/complexity/quality |
| **syncopation_index** | score | 0-1 | Rhythmic complexity relative to beat |

---

## Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/mir.git
cd mir

# Create virtual environment
python3 -m venv mir
source mir/bin/activate  # Windows: mir\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup external repositories and apply patches
bash scripts/setup_external_repos.sh
# Or use Python script: python scripts/apply_patches.py
```

### 2. Organize Your Audio Files

```bash
# Convert flat audio files into organized folders
python src/preprocessing/file_organizer.py /path/to/audio/
```

**Before:**
```
/music/
├── song1.mp3
├── song2.flac
└── song3.wav
```

**After:**
```
/music/
├── Artist1 - Album1 - Song1/
│   └── full_mix.mp3
├── Artist2 - Album2 - Song2/
│   └── full_mix.flac
└── Artist3 - Album3 - Song3/
    └── full_mix.wav
```

### 3. Separate Stems

```bash
# Use GPU for fast processing (~9-15x realtime)
python src/preprocessing/demucs_sep.py /path/to/audio/ --batch --device cuda

# Or process single track
python src/preprocessing/demucs_sep.py "/path/to/audio/Artist - Album - Track/"

# Output format options
python src/preprocessing/demucs_sep.py /path/to/audio/ --batch --format flac              # Lossless (default)
python src/preprocessing/demucs_sep.py /path/to/audio/ --batch --format mp3 --bitrate 320 # MP3 CBR
python src/preprocessing/demucs_sep.py /path/to/audio/ --batch --format mp3 --preset 2    # MP3 VBR
python src/preprocessing/demucs_sep.py /path/to/audio/ --batch --format ogg --ogg-quality 0.6  # OGG
python src/preprocessing/demucs_sep.py /path/to/audio/ --batch --format wav24             # 24-bit WAV
```

**Output Formats:** `flac` (default), `mp3`, `ogg`, `wav`, `wav24`, `wav32`

**GPU Performance:**
- AMD (ROCm) / NVIDIA (CUDA): ~9-15x realtime (10min track in 40-60 seconds)
- CPU: ~0.1x realtime (10min track in 60+ minutes)

### 4. Extract Features

```bash
# Test single file with all features
python src/test_all_features.py "/path/to/Track Name/full_mix.flac"

# Or use --transformers flag for A/B testing with full Music Flamingo model
python src/test_all_features.py "/path/to/Track Name/full_mix.flac" --transformers

# Run modules individually for batch processing
python src/rhythm/rhythm_analysis.py /path/to/audio/ --batch
python src/preprocessing/loudness.py /path/to/audio/ --batch
python src/spectral/spectral_features.py /path/to/audio/ --batch
python src/harmonic/chroma_analysis.py /path/to/audio/ --batch
python src/harmonic/per_stem_harmonic.py /path/to/audio/ --batch
python src/timbral/audio_commons.py /path/to/audio/ --batch
python src/classification/essentia_features.py /path/to/audio/ --batch --gmi
python src/rhythm/per_stem_rhythm.py /path/to/audio/ --batch

# Music Flamingo AI descriptions (requires 16GB VRAM)
python src/classification/music_flamingo_transformers.py /path/to/audio/ --batch --flash-attention
```

**Processing Time:** ~35s per track (standard features), ~2.5min per track (with all 5 AI descriptions)

### 5. Create Training Crops (Optional)

```bash
# Sequential mode: exact sample length, no beat alignment
python src/tools/create_training_crops.py /path/to/audio/ --length 2097152 --sequential

# Beat-aligned with 50% overlap
python src/tools/create_training_crops.py /path/to/audio/ --length 2097152 --overlap

# Beat-aligned with div4 downbeats (measures divisible by 4)
python src/tools/create_training_crops.py /path/to/audio/ --length 2097152 --overlap --div4

# Save to custom output directory (per-track folders)
python src/tools/create_training_crops.py /path/to/audio/ -o /path/to/output --sequential
```

**Crop Features:**
- `--output-dir` / `-o`: Save crops to destination with per-track folders
- Length in samples (default 2097152 = ~47.5s at 44.1kHz)
- Beat-aligned start/end with zero-crossing snap for click-free cuts
- Optional `--div4` ensures each crop contains downbeats divisible by 4
- 10ms fade-in/fade-out on all crops
- Creates `.INFO` file per crop with position metadata (0.0 to 1.0)
- Output: `TrackName/TrackName_0.flac`, `TrackName/TrackName_1.flac`, etc.

### 6. Access Results

Features saved to `.INFO` JSON files:

```json
{
  "bpm": 142.19,
  "bpm_is_defined": 1,
  "danceability": 0.987,
  "lufs": -12.73,
  "brightness": 58.64,
  "essentia_genre": {"Electronic---House": 0.85, "Electronic---Tech House": 0.72},
  "essentia_mood": {"energetic": 0.91, "groovy": 0.78, "party": 0.65},
  "essentia_instrument": {"synthesizer": 0.89, "drums": 0.95, "bass": 0.82},
  "music_flamingo_full": "This track is an energetic house music production...",
  "music_flamingo_technical": "The tempo is approximately 128 BPM...",
  ...
}
```

---

## Output Structure

```
Artist - Album - Track/
├── full_mix.mp3                           # Original audio
├── drums.mp3                              # Separated stems (MP3 @ 320kbps)
├── bass.mp3
├── other.mp3
├── vocals.mp3
├── Artist - Album - Track.INFO            # 77 features (JSON)
├── Artist - Album - Track.BEATS_GRID      # Beat timestamps (JSON)
└── separated/                             # Demucs working directory
    └── htdemucs/
        └── ...
```

---

## Documentation

- **[USER_MANUAL.md](USER_MANUAL.md)** - Comprehensive usage guide
- **[FEATURES_STATUS.md](FEATURES_STATUS.md)** - Complete feature implementation status
- **[EXTERNAL_PATCHES.md](EXTERNAL_PATCHES.md)** - External repository modifications
- **[project.log](project.log)** - Development history and decisions
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Original implementation plan

---

## Project Structure

```
mir/
├── src/                          # Source code
│   ├── core/                    # Core utilities (JSON, file handling, logging)
│   ├── preprocessing/           # File organization, stem separation, loudness
│   ├── rhythm/                  # Beat detection, BPM, syncopation, onsets
│   ├── spectral/                # Spectral features and RMS energy
│   ├── harmonic/                # Chroma, harmonic movement
│   ├── timbral/                 # Audio Commons timbral features
│   └── classification/          # Essentia, Music Flamingo AI descriptions
├── scripts/                      # Setup and utility scripts
│   ├── setup_external_repos.sh  # Clone and patch external dependencies
│   └── apply_patches.py         # Apply librosa compatibility patches
├── plans/                        # Implementation plan files
├── repos/                        # External repositories (not tracked)
│   └── repos/
│       └── timbral_models/      # Audio Commons (cloned + patched)
├── test_data/                    # Test audio (not tracked)
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── USER_MANUAL.md               # Usage documentation
├── FEATURES_STATUS.md           # Implementation status
├── EXTERNAL_PATCHES.md          # External patches documentation
└── project.log                  # Development log
```

---

## Requirements

### System Requirements

- **Python:** 3.10 or higher
- **GPU:** AMD (ROCm) or NVIDIA (CUDA) recommended for stem separation
- **Disk Space:** ~50MB per minute of audio (stems + features)
- **Memory:** 8GB+ RAM recommended

### Python Dependencies

Core libraries:
- **librosa** >= 0.11.0 - Audio analysis
- **soundfile** >= 0.12.0 - Audio I/O
- **demucs** >= 4.0.0 - Stem separation
- **pyloudnorm** >= 0.1.0 - Loudness measurement
- **essentia** >= 2.1b6 - High-level features
- **numpy**, **scipy**, **pandas** - Scientific computing

See `requirements.txt` for complete list.

### External Dependencies

- **timbral_models** (Audio Commons) - Cloned and patched automatically via setup script

---

## GPU Support

### AMD GPUs (ROCm)

```bash
# Use --device cuda (AMD GPUs appear as CUDA with ROCm)
python src/preprocessing/demucs_sep.py audio/ --batch --device cuda
```

**Performance:** ~9.4x realtime (10min track in ~64 seconds)

**Optimizations for AMD RDNA4:** Set these environment variables for best performance:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0
export OMP_NUM_THREADS=8
```

### NVIDIA GPUs (CUDA)

```bash
# Use --device cuda
python src/preprocessing/demucs_sep.py audio/ --batch --device cuda
```

**Performance:** ~10-15x realtime

### CPU Only

```bash
# Use --device cpu (default)
python src/preprocessing/demucs_sep.py audio/ --batch
```

**Performance:** ~0.1x realtime (very slow)

---

## Known Issues & Solutions

### ✅ Audio Commons librosa API Errors (FIXED)

**Issue:** `onset_detect() takes 0 positional arguments but 2 positional arguments were given`

**Solution:** Automatically fixed by running `scripts/apply_patches.py` during setup. Patches documented in `EXTERNAL_PATCHES.md`.

### ✅ TorchCodec/FFmpeg Errors (FIXED)

**Issue:** Demucs fails to save FLAC output files with TorchCodec errors

**Solution:** Native FLAC output is attempted first (fast). If torchcodec fails, automatically falls back to WAV + soundfile conversion. OGG uses WAV intermediate (no native support). MP3 uses lameenc (reliable).

### ✅ AudioBox Aesthetics (FIXED)

**Status:** AudioBox aesthetics features now use actual Meta AudioBox model inference.

**Solution:** Implemented in `src/timbral/audiobox_aesthetics.py` using `audiobox_aesthetics` package from GitHub.

---

## Feature Implementation Status

✅ **Implemented:** 77+ numeric features + AI classification

**Core Features:**
- ✅ All rhythm features (29)
- ✅ All loudness features (10)
- ✅ All spectral features (4)
- ✅ All RMS energy features (4)
- ✅ All chroma features (12)
- ✅ All harmonic features (4)
- ✅ All timbral features (8)
- ✅ All aesthetic features (4)
- ✅ All classification features (2)

**AI Features:**
- ✅ Genre classification (400 Discogs genres)
- ✅ Mood/theme classification (56 MTG Jamendo labels)
- ✅ Instrument classification (40 MTG Jamendo labels)
- ✅ Music Flamingo descriptions (5 prompt types)

**Missing:**
- ✅ Position feature (implemented via smart cropping system)

See [FEATURES_STATUS.md](FEATURES_STATUS.md) for complete details.

---

## Development Roadmap

### ✅ Phase 1: Core Feature Extraction (COMPLETE)
- File organization system
- Stem separation (Demucs)
- Rhythm analysis (BPM, beats, syncopation, onsets)
- Loudness analysis (LUFS/LRA per-stem)
- Spectral features (flatness, flux, skewness, kurtosis)
- RMS energy (4 frequency bands)
- Harmonic features (chroma, movement, variance)
- Timbral features (Audio Commons 8 features)
- Classification (danceability, atonality, aesthetics)

### ✅ Phase 2: Dataset Preparation (COMPLETE)
- ✅ Smart cropping system (`src/tools/create_training_crops.py`)
- ✅ Position feature calculation (via cropping metadata)
- ✅ AudioBox model inference (`src/timbral/audiobox_aesthetics.py`)
- Statistical analysis tool (planned)

### ✅ Phase 3: AI Classification (COMPLETE)
- Genre classification (400 Discogs genres via Essentia)
- Mood/theme classification (56 MTG Jamendo labels)
- Instrument classification (40 MTG Jamendo labels)
- Music Flamingo AI descriptions (5 prompt types)

### 📅 Phase 4: Advanced Features (IN PROGRESS)
- ✅ MIDI drum transcription (ADTOF-PyTorch, Drumsep)
- ❌ Bass MIDI transcription (Basic Pitch, PESTO)
- ❌ Polyphonic MIDI transcription (MT3, MR-MT3)
- Kick/snare/cymbal per-drum analysis
- Auxiliary file outputs (.ONSETS_GRID, .CHROMA)

---

## Contributing

Contributions welcome! Areas of interest:
- Smart cropping algorithm implementation
- AudioBox aesthetics model integration
- MIDI transcription pipeline
- Additional feature extractors
- Documentation improvements

---

## Citation

If you use this framework in your research, please cite:

```
[Citation to be added]
```

Based on:
- [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools) (Stability AI)
- [Essentia](https://essentia.upf.edu/) (Music Technology Group, UPF Barcelona)
- [Demucs](https://github.com/facebookresearch/demucs) (Meta Research)
- [Audio Commons Timbral Models](https://github.com/AudioCommons/timbral_models)
- [Music Flamingo](https://huggingface.co/nvidia/Music-Flamingo) (NVIDIA Research)

---

## License

[To be determined]

---

## Support

For issues, bugs, or feature requests:
1. Check [USER_MANUAL.md](USER_MANUAL.md) for usage help
2. Review [FEATURES_STATUS.md](FEATURES_STATUS.md) for implementation status
3. Consult [project.log](project.log) for known issues
4. Open an issue on GitHub

---

**Version:** 1.3
**Last Updated:** 2026-01-22
**Status:** Production Ready (Core + AI Features + Training Crops + MIDI Drums)
**Features:** 77+ numeric + 496 AI labels + 5 AI descriptions + smart cropping + MIDI drum transcription
