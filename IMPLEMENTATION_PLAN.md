# MIR Implementation Plan

## Overview
This plan outlines the implementation order for the Music Information Retrieval (MIR) features for conditioning Stable Audio Open. The implementation follows a priority-based approach focusing on highest-impact features first.

## Project Structure
```
/home/kim/Projects/mir/
├── repos/                      # Cloned repositories
│   ├── ADTOF/                 # Drum transcription
│   ├── basic-pitch/           # Bass/polyphonic transcription
│   ├── crepe/                 # Pitch tracking
│   ├── pesto/                 # Bass transcription
│   ├── mt3/                   # Polyphonic transcription
│   ├── MR-MT3/                # Multi-resolution transcription
│   ├── timbral_models/        # Audio Commons features
│   ├── magenta/               # OaF Drums
│   └── stable-audio-tools/    # Reference implementation
├── essentia/                   # Existing Essentia classification code
├── mir/                        # Main project code (to be created)
│   ├── core/                  # Core utilities
│   │   ├── json_handler.py   # JSON read/write for .INFO and .MIR files
│   │   ├── file_utils.py     # File organization utilities
│   │   └── common.py         # Shared functions
│   ├── preprocessing/         # File and stem separation
│   │   ├── file_organizer.py # Move files to folders
│   │   ├── demucs_sep.py     # Demucs stem separation
│   │   └── drumsep.py        # Drum separation
│   ├── rhythm/               # Rhythm analysis
│   │   ├── beat_grid.py      # Beat tracking and grid creation
│   │   ├── onset_detection.py # Onset detection
│   │   ├── syncopation.py    # Syncopation analysis
│   │   ├── bpm.py            # BPM detection and validation
│   │   └── rhythmic_features.py # Complexity, density, evenness
│   ├── spectral/             # Spectral features
│   │   ├── multiband_rms.py  # Multiband RMS energy
│   │   └── spectral_features.py # Flatness, flux, skewness, kurtosis
│   ├── harmonic/             # Harmonic features
│   │   ├── chroma.py         # Chroma extraction
│   │   └── harmonic_movement.py # Harmonic movement and variance
│   ├── timbral/              # Timbral features
│   │   ├── audio_commons.py  # Audio Commons descriptors
│   │   └── loudness.py       # LUFS and LRA
│   ├── transcription/        # MIDI transcription
│   │   ├── drums/            # Drum transcription
│   │   ├── bass/             # Bass transcription
│   │   └── polyphonic/       # Polyphonic transcription
│   ├── classification/       # High-level classification
│   │   ├── essentia_features.py # Danceability, genre, mood, etc.
│   │   └── audiobox.py       # AudioBox aesthetics
│   ├── conditioners/         # Conditioning data preparation
│   │   └── format_conditioners.py # Format data for SAO training
│   └── statistics/           # Dataset statistics
│       └── analyzer.py       # Statistical analysis tool
└── project.log               # Development log

Output structure per audio file:
/path/to/audio/filename/
├── full_mix.flac             # Original file
├── drums.flac                # Demucs stems
├── bass.flac
├── other.flac
├── vocals.flac
├── kick.flac                 # DrumSep stems
├── snare.flac
├── cymbals.flac
├── filename.INFO             # JSON with all feature values
├── filename.MIR              # JSON with temporal data
├── filename.BEATS_GRID       # Beat timestamps
├── filename.ONSETS_GRID      # Onset timestamps
├── filename.CHROMA           # Raw chroma data
└── *.mid                     # MIDI transcriptions
```

## Implementation Phases

### Phase 0: Foundation (IMMEDIATE)
**Goal:** Set up core infrastructure

1. **File organization system** (`mir/preprocessing/file_organizer.py`)
   - Script to move files to folder structure
   - Input: audio files in any location
   - Output: organized folder structure with full_mix.flac

2. **Core utilities** (`mir/core/`)
   - `json_handler.py`: Safe JSON read/write for .INFO and .MIR files
   - `file_utils.py`: Path handling, file discovery
   - `common.py`: Shared constants and utilities

3. **Project documentation**
   - Create `claude.md` files for each module
   - Set up project.log structure

### Phase 1: Core Features (HIGH PRIORITY)
**Goal:** Implement highest-impact features for broad control

#### 1.1 File Position & Organization
- **File:** `mir/core/file_utils.py`
- **Feature:** `position_in_file` (0-1 float)
- **Implementation:** Calculate during smart cropping
- **Key:** `{position_in_file}`

#### 1.2 Loudness Analysis
- **File:** `mir/timbral/loudness.py`
- **Dependencies:** pyloudnorm
- **Features:**
  - Integrated loudness (LUFS): `{lufs}`
  - Loudness range (LRA): `{lra}`
  - Per-stem analysis: `{lufs_drums}`, `{lufs_bass}`, etc.
- **Implementation:** Use pyloudnorm.Meter
- **Value ranges:** LUFS: -40 to 0 dB, LRA: 0 to 25 LU

#### 1.3 Stem Separation (Prerequisite for many features)
- **File:** `mir/preprocessing/demucs_sep.py`
- **Tool:** Demucs HT v4
- **Parameters:**
  - shifts: 1
  - filetype: flac
  - concurrent jobs: 4
- **Output:** drums.flac, bass.flac, other.flac, vocals.flac

#### 1.4 DrumSep (Drum-specific separation)
- **File:** `mir/preprocessing/drumsep.py`
- **Tool:** jarredou DrumSep model
- **Input:** drums.flac
- **Output:** kick.flac, snare.flac, cymbal.flac, etc.

#### 1.5 Basic BPM Detection
- **File:** `mir/rhythm/bpm.py`
- **Dependencies:** Beat grid from Phase 1.6
- **Features:**
  - `{bpm}`: Detected BPM value
  - `{bpm_is_defined}`: Binary flag (0 or 1)
  - `{beat_count}`: Number of beats detected
  - `{beat_regularity}`: Consistency measure
- **Logic:**
  - Use beat intervals to calculate BPM
  - Threshold-based validation (beat_count >= 15, regularity < 0.1)
  - Set bpm_is_defined accordingly

#### 1.6 Beat Grid Creation
- **File:** `mir/rhythm/beat_grid.py`
- **Tools:** Madmom or Essentia BeatTrackerDegara
- **Output:** `filename.BEATS_GRID` (beat timestamps)
- **Usage:** Foundation for BPM, syncopation, MIDI quantization

#### 1.7 Audio Commons: Brightness
- **File:** `mir/timbral/audio_commons.py`
- **Tool:** timbral_models
- **Feature:** `{brightness}` (0-100)
- **Priority:** Implement brightness first as it's specifically mentioned
- **Full set:** Will implement all 8 features (brightness, roughness, hardness, depth, booming, reverberation, sharpness, warmth)

#### 1.8 Danceability (Essentia)
- **File:** `mir/classification/essentia_features.py`
- **Tool:** Essentia-TensorFlow
- **Feature:** `{danceability}` (0-1)
- **Note:** Extract from existing `/essentia/universal_audio_classifier_with_essentia_v4.py`

### Phase 2: Rhythmic Features (HIGH PRIORITY)
**Goal:** Critical for rhythmic music styles

#### 2.1 Onset Detection
- **File:** `mir/rhythm/onset_detection.py`
- **Tools:** Essentia Onsets or Librosa onset_detect
- **Per-stem analysis:** drums, bass, other
- **Output:** `filename.ONSETS_GRID`

#### 2.2 Onset Density
- **File:** `mir/rhythm/rhythmic_features.py`
- **Formula:** number_of_onsets / duration_in_seconds
- **Features:**
  - Average: `{onset_density_average_bass}`, `{onset_density_average_drums}`, `{onset_density_average_other}`
  - Variance: `{onset_density_variance_bass}`, `{onset_density_variance_drums}`, `{onset_density_variance_other}`

#### 2.3 Syncopation Analysis
- **File:** `mir/rhythm/syncopation.py`
- **Method:** Longuet-Higgins & Lee model
- **Steps:**
  1. Assign metrical weights to beat grid positions
  2. Quantize onsets to grid
  3. Calculate syncopation score
- **Features:** `{syncopation_bass}`, `{syncopation_drums}`, `{syncopation_other}`

#### 2.4 Rhythmic Complexity
- **File:** `mir/rhythm/rhythmic_features.py`
- **Method:** Shannon entropy of inter-onset interval histogram
- **Features:** `{rhythmic_complexity_bass}`, `{rhythmic_complexity_drums}`, `{rhythmic_complexity_other}`

#### 2.5 Rhythmic Evenness
- **File:** `mir/rhythm/rhythmic_features.py`
- **Formula:** np.std(np.diff(onset_timestamps))
- **Features:** Per-stem (bass, drums, other) + per-drum-type (kick, snare, cymbal)

### Phase 3: Spectral & Timbral Features (MEDIUM PRIORITY)
**Goal:** Enable timbral control

#### 3.1 Multiband RMS Energy
- **File:** `mir/spectral/multiband_rms.py`
- **Bands:**
  - Bass: 20-120 Hz → `{rms_energy_bass}`
  - Body: 120-600 Hz → `{rms_energy_body}`
  - Mid: 600-2500 Hz → `{rms_energy_mid}`
  - Air: 2500-22000 Hz → `{rms_energy_air}`
- **Implementation:** Bandpass filtering + RMS in dB (20*log10(rms))

#### 3.2 Spectral Features
- **File:** `mir/spectral/spectral_features.py`
- **Tool:** Essentia
- **Features:**
  - Spectral flatness: `{spectral_flatness}` (0-1)
  - Spectral flux: `{spectral_flux}` (normalized)
  - Spectral skewness: `{spectral_skewness}` (-3 to 3)
  - Spectral kurtosis: `{spectral_kurtosis}` (centered on 3)
- **Aggregation:** Use median across frames

#### 3.3 Audio Commons Full Suite
- **File:** `mir/timbral/audio_commons.py`
- **Complete implementation of 8 features:**
  - Brightness, Roughness, Hardness, Depth
  - Booming, Reverberation, Sharpness, Warmth
- **All features:** 0-100 scale, analyzed on full_mix

### Phase 4: Harmonic Features (MEDIUM PRIORITY)
**Goal:** Harmonic/melodic content control

#### 4.1 Chroma Extraction
- **File:** `mir/harmonic/chroma.py`
- **Tool:** Essentia
- **Outputs:**
  - `{CHROMA_KEY}`: 12-dimensional vector (averaged over clip)
  - `filename.CHROMA`: Raw temporal chroma data
- **Implementation:** Circular chromatic pitch map, no key detection

#### 4.2 Harmonic Movement
- **File:** `mir/harmonic/harmonic_movement.py`
- **Method:**
  1. Calculate chromagram sequence at 32nd note intervals (or 100ms if no BPM)
  2. Compute cosine distance between consecutive frames
  3. Consider Wasserstein metric/EMD as alternative
- **Features:**
  - Movement: `{harmonic_movement_bass}`, `{harmonic_movement_other}`
  - Variance: `{harmonic_variance_bass}`, `{harmonic_variance_other}`

### Phase 5: MIDI Transcription (LOWER PRIORITY)
**Goal:** Generate MIDI for potential future use

#### 5.1 Drum Transcription
- **File:** `mir/transcription/drums/`
- **Tools:** Compare 3 algorithms
  1. ADTOF
  2. OaF Drums (Magenta)
  3. jarredou DrumSep-based method
- **Output:** Separate MIDI files per algorithm
- **Quantization:** Use beat grid from Phase 1.6
- **Future:** Global drum clustering (optional)

#### 5.2 Bass Transcription
- **File:** `mir/transcription/bass/`
- **Tools:** Compare 3 algorithms
  1. Basic Pitch
  2. PESTO
  3. CREPE
- **Special:** Convert to MIDI with 12-semitone pitch bend where possible
- **Output:** Separate MIDI + continuous pitch data

#### 5.3 Polyphonic Transcription
- **File:** `mir/transcription/polyphonic/`
- **Tools:** Compare 3 algorithms
  1. Basic Pitch
  2. MT3
  3. MR-MT3
- **Input:** other.flac stem
- **Note:** Could use harmony info from Phase 4 to resolve ambiguities

### Phase 6: Advanced Features (LOWER PRIORITY)
**Goal:** Refinement and specialized control

#### 6.1 Essentia High-Level Features
- **File:** `mir/classification/essentia_features.py`
- **Extract from existing code:**
  - Atonality: `{atonality}` (0-1)
  - Genre: Store in prompt (use existing labels)
  - Mood: Store in prompt
  - Instrumentation: Store in prompt
  - Vocal gender: `{vocal_gender}` (categorical, if vocals present)
- **Note:** Reuse code from `/essentia/universal_audio_classifier_with_essentia_v4.py`

#### 6.2 AudioBox Aesthetics
- **File:** `mir/classification/audiobox.py`
- **Tool:** AudioBox Aesthetics models
- **Features:**
  - Content Enjoyment: `{content_enjoyment}` (1-10)
  - Content Usefulness: `{content_usefulness}` (1-10)
  - Production Complexity: `{production_complexity}` (1-10)
  - Production Quality: `{production_quality}` (1-10)
- **Note:** Use high dropout during training

#### 6.3 Smart Cropping
- **File:** `mir/preprocessing/smart_cropper.py`
- **Goal:** Create training clips from full songs
- **Output:** Files in `/crops` subfolder with `section_N` suffix
- **Calculate:** `position_in_file` based on full_mix length

### Phase 7: Conditioner Formatting (INTEGRATION)
**Goal:** Format all extracted features for SAO training

#### 7.1 Conditioner Configuration
- **File:** `mir/conditioners/format_conditioners.py`
- **Tasks:**
  1. Map each feature to appropriate conditioner type
  2. Set min_val/max_val based on statistics
  3. Generate conditioning config for SAO
  4. Implement conditioning dropout logic

#### 7.2 Conditioning Types Summary
- **NumberConditioner:** Most features (continuous values)
- **IntConditioner:** bpm_is_defined, vocal_gender
- **Text (CLAP/T5):** Genre, mood, instrumentation

### Phase 8: Statistics & Validation (FINAL)
**Goal:** Dataset analysis and outlier handling

#### 8.1 Statistical Analyzer
- **File:** `mir/statistics/analyzer.py`
- **Functions:**
  1. Scan all .INFO files in dataset
  2. Calculate min, max, mean, std for each feature
  3. Identify outliers
  4. Generate distribution plots
  5. Create summary JSON
- **Output:** statistics.json with per-feature stats

## Implementation Notes

### Development Practices
1. **Before major changes:** Create backup in `/backup` folder as `filename.py.#`
2. **Documentation:** Update `claude.md` in each module folder
3. **Logging:** Record all changes in `project.log`
4. **Testing:** Never assume unused code is safe to remove
5. **Dependencies:** Track in module docstrings

### File Handling Rules
1. **JSON files:** Always use safe read/write to avoid data loss
2. **Audio files:** Assume slices are unnormalized (originals at -1dB)
3. **Temporal data:** Save to `.MIR` files for efficient access
4. **Beat/onset grids:** Save to dedicated files for reuse

### Technical Decisions
1. **Beat tracking:** Choose between Madmom (state-of-art for electronic) vs Essentia (good general)
2. **BPM default:** Decide on 120 vs 0 for undefined BPM
3. **Chroma temporal:** Decide if time series should be saved
4. **Normalization:** Rely on SAO internal normalization, just set min/max

### Priority Rationale
- **Phase 1:** Foundation + immediate high-impact features
- **Phase 2:** Rhythmic features critical for EDM/electronic music
- **Phase 3-4:** Timbral/harmonic features for nuanced control
- **Phase 5:** MIDI nice-to-have for future possibilities
- **Phase 6:** Advanced features for refinement
- **Phase 7-8:** Integration and validation

## Next Steps

1. Create directory structure for `mir/` project
2. Implement Phase 0 (foundation)
3. Extract and adapt Essentia code from existing notebook
4. Begin Phase 1 implementations in priority order
5. Test each feature on sample files
6. Iterate through phases sequentially

## Questions to Resolve

1. **BPM default value:** 120 or 0 for non-rhythmic content?
2. **Beat tracker choice:** Madmom or Essentia for goa trance?
3. **Chroma temporal storage:** Save full time series or just median?
4. **Crop boundary handling:** How to handle partial beats at clip edges?
5. **AudioBox models:** Where to obtain pretrained models?
6. **Basic Pitch:** Python 3.12 incompatibility - use from cloned repo?

## Success Criteria

- [ ] All audio files successfully organized into folder structure
- [ ] All Phase 1-4 features extracted and saved to .INFO files
- [ ] No data loss when updating .INFO files with new features
- [ ] Statistical analysis shows reasonable value distributions
- [ ] Conditioner configs generated and compatible with SAO
- [ ] Complete documentation in claude.md files
- [ ] Comprehensive project.log tracking all changes
