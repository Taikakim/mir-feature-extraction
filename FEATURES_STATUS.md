# MIR Feature Extraction Status

**Last Updated:** 2026-02-18
**Current Implementation:** 80 numeric features + 496 classification labels + user-defined AI text descriptions

---

## Numeric Features (80 total)

### Rhythm (29 features)

**Global (full mix):**
- `bpm` - Tempo in beats per minute
- `bpm_is_defined` - Binary flag (1=rhythmic, 0=arrhythmic)
- `beat_count` - Total number of beats detected
- `beat_regularity` - Consistency of beat intervals (std dev)
- `syncopation` - Off-beat energy score
- `on_beat_ratio` - Proportion of onsets on beat
- `onset_count` - Total number of onset events
- `onset_density` - Onsets per second
- `onset_strength_mean` - Average onset magnitude
- `onset_strength_std` - Onset magnitude variability
- `rhythmic_complexity` - Shannon entropy of IOI distribution
- `rhythmic_evenness` - Temporal regularity of onsets
- `ioi_mean` - Mean inter-onset interval
- `ioi_std` - Inter-onset interval variability

**Per-stem (bass, drums, other):**
- `onset_density_average_{stem}` - Average onset density
- `onset_density_variance_{stem}` - Onset density variance
- `syncopation_{stem}` - Per-stem syncopation
- `rhythmic_complexity_{stem}` - Per-stem entropy
- `rhythmic_evenness_{stem}` - Per-stem regularity

### Loudness (10 features)

- `lufs` / `lra` - Integrated loudness and range (full mix)
- `lufs_{stem}` / `lra_{stem}` - Per-stem (drums, bass, other, vocals)

### Spectral (4 features)

- `spectral_flatness` - Noise-like vs tone-like (0-1)
- `spectral_flux` - Spectral change rate
- `spectral_skewness` - Low vs high frequency weighting
- `spectral_kurtosis` - Spectral energy concentration

### Multiband RMS Energy (4 features)

- `rms_energy_bass` - 20-120 Hz (dB)
- `rms_energy_body` - 120-600 Hz (dB)
- `rms_energy_mid` - 600-2500 Hz (dB)
- `rms_energy_air` - 2500-22000 Hz (dB)

### Chroma (12 features)

- `chroma_0` through `chroma_11` - 12-dimensional pitch class weights (0-1)

### Harmonic (4 features)

- `harmonic_movement_bass` / `harmonic_movement_other` - Rate of harmonic change
- `harmonic_variance_bass` / `harmonic_variance_other` - Harmonic diversity

### Audio Commons Timbral (8 features)

- `brightness` - High-frequency content (0-100)
- `roughness` - Beating and modulation (0-100)
- `hardness` - Attack sharpness (0-100)
- `depth` - Low-frequency spaciousness (0-100)
- `booming` - 100-200 Hz resonance (0-100)
- `reverberation` - Wet/dry balance (0-100)
- `sharpness` - High-frequency harshness (0-100)
- `warmth` - Mid-low frequency richness (0-100)

Requires librosa 0.11.0 patches (applied by `scripts/setup_external_repos.sh`).

### AudioBox Aesthetics (4 features)

- `content_enjoyment` - Aesthetic appeal (1-10)
- `content_usefulness` - Functional value (1-10)
- `production_complexity` - Production sophistication (1-10)
- `production_quality` - Technical excellence (1-10)

### Essentia Classification (4 scalar + label dicts)

**Scalar features:**
- `danceability` - Rhythmic strength for dancing (0-1)
- `atonality` - Departure from tonality (0-1)
- `voice_probability` - Probability of voice presence (0-1)
- `instrumental_probability` - Probability of instrumental content (0-1)

**Label dicts** (saved as {label: probability} dicts in .INFO, not in FEATURE_RANGES):
- `essentia_genre` - Genre classification (400 Discogs classes, top 10 above threshold)
- `essentia_mood` - Mood/theme classification (56 classes, top 10 above threshold)
- `essentia_instrument` - Instrument detection (40 classes, top 10 above threshold)

### Crop Position (1 feature)

- `position_in_file` - Relative position in original file (0-1)

---

## AI Text Descriptions

### Music Flamingo

User-configurable prompts defined in `config/master_pipeline.yaml` under `music_flamingo.prompts`.
Each prompt key becomes a `music_flamingo_{key}` field in the .INFO file.

Default config has a single `full` prompt. Users can add arbitrary prompt keys.

**Implementation:** GGUF quantized models via `llama-mtmd-cli` subprocess
**Performance:** ~46s per track at Q8_0 (3.17x realtime for a 2.45min track)
**Models:** IQ3_M, Q6_K, Q8_0

---

## Auxiliary Files

| File | Description | Status |
|------|-------------|--------|
| `.INFO` | All features (JSON, atomic merge via `safe_update`) | Complete |
| `.BEATS_GRID` | Beat timestamps | Complete |
| `.DOWNBEATS` | Downbeat timestamps | Complete |
| `.ONSETS` | Onset timestamps | Complete |

---

## Pipeline Modules

| Module | Status | Notes |
|--------|--------|-------|
| Source separation (BS-RoFormer) | Complete | Primary backend, 4-stem |
| Source separation (Demucs) | Complete | Alternative backend |
| Beat/tempo detection (madmom) | Complete | CPU-only |
| Feature extraction (all groups) | Complete | 78 numeric features |
| Essentia classification | Complete | TensorFlow models |
| AudioBox aesthetics | Complete | |
| Music Flamingo GGUF | Complete | Recommended |
| Music Flamingo Transformers | Complete | Slower alternative |
| Smart cropping | Complete | Beat-aligned, lossy preloading |
| Metadata lookup | Complete | Spotify + MusicBrainz |
| Statistical analysis | Complete | `src/tools/statistical_analysis.py` |
| Filename cleanup | Complete | T5 tokenizer compatible |
| MIDI drum transcription (ADTOF) | Functional | Quality could be better |
| MIDI drum transcription (DrumSep) | Functional | Alternative method |

---

## Not Implemented

- `.CHROMA` time series files (only average chroma saved to .INFO) - LOW priority
- MIDI bass/polyphonic transcription (Basic Pitch, MT3, etc.) - repos removed
- Per-drum-kit rhythm features (kick/snare/cymbal individual analysis) - LOW priority

---

## Environment

- **Python:** 3.12+
- **NumPy:** <2.4 (pinned for numba compatibility)
- **PyTorch:** 2.9.1+rocm7.2.0
- **llama.cpp:** Built with HIP/ROCm support
- **Hardware:** AMD RX 9070 XT (RDNA4, 16GB), Ryzen 9 9900X
