# MIR Feature Extraction Framework - User Manual

## Overview

Extracts 97+ numeric MIR features, 496 classification labels, and 5 AI text descriptions from audio files for conditioning Stable Audio Tools. Processes full mixes and separated stems.

For each track you get:
- `.INFO` JSON file with all numeric features + AI text descriptions
- 4 separated stems (drums, bass, other, vocals)
- `.BEATS_GRID`, `.ONSETS`, `.DOWNBEATS` timing files
- Optional: MIDI drum transcription, beat-aligned training crops

---

## Installation

```bash
python -m venv mir && source mir/bin/activate
pip install -r requirements.txt
bash scripts/setup_external_repos.sh

# For Music Flamingo GGUF: build llama.cpp with HIP support (see MUSIC_FLAMINGO.md)
```

**Requirements:** Python 3.12+, NumPy <2.4, AMD ROCm 7.2+ or NVIDIA CUDA, 5-13 GB VRAM.

ROCm env vars are handled automatically by `src/core/rocm_env.py`. See `config/master_pipeline.yaml` for settings.

---

## Quick Start

```bash
# 1. Organize audio files into folders
python src/preprocessing/file_organizer.py my_music/

# 2. Separate stems (GPU recommended)
python src/preprocessing/demucs_sep_optimized.py my_music/ --batch --device cuda

# 3. Test all features on a single file
python src/test_all_features.py "my_music/Artist - Track/full_mix.flac"

# 4. Full pipeline (config-driven, all stages)
python src/master_pipeline.py --config config/master_pipeline.yaml
```

### Step-by-Step Extraction (manual control)

```bash
python src/rhythm/beat_grid.py my_music/ --batch
python src/rhythm/bpm.py my_music/ --batch
python src/rhythm/onsets.py my_music/ --batch
python src/rhythm/syncopation.py my_music/ --batch
python src/timbral/loudness.py my_music/ --batch
python src/spectral/spectral_features.py my_music/ --batch
python src/spectral/multiband_rms.py my_music/ --batch
python src/harmonic/chroma.py my_music/ --batch
python src/harmonic/per_stem_harmonic.py my_music/ --batch
python src/timbral/audio_commons.py my_music/ --batch
python src/classification/essentia_features.py my_music/ --batch
python src/rhythm/per_stem_rhythm.py my_music/ --batch
python src/classification/music_flamingo.py my_music/ --batch --model Q6_K
```

All scripts support `--batch`, `--overwrite`, and `--verbose` flags.

---

## File Organization

**Before:**
```
my_music/
  Artist - Album - Track.mp3
  Another Track.flac
```

**After `file_organizer.py` + full processing:**
```
my_music/
  Artist - Album - Track/
    full_mix.mp3
    drums.mp3, bass.mp3, other.mp3, vocals.mp3
    Artist - Album - Track.INFO
    Artist - Album - Track.BEATS_GRID
    Artist - Album - Track.ONSETS
```

**Supported formats:** .mp3, .wav, .flac, .ogg, .m4a (m4a via pydub/ffmpeg fallback)

---

## Module Reference

### Preprocessing

| Module | Purpose | Output |
|--------|---------|--------|
| `file_organizer.py` | Organize audio into folders | Folder structure with `full_mix.*` |
| `demucs_sep_optimized.py` | Stem separation (Demucs HT v4) | drums/bass/other/vocals stems |
| `bs_roformer_sep.py` | Stem separation (BS-RoFormer) | drums/bass/other/vocals stems |

Use `--device cuda` for GPU (including AMD ROCm).

### Feature Extraction

| Module | Features | Count |
|--------|----------|-------|
| `timbral/loudness.py` | LUFS, LRA (full mix + 4 stems) | 10 |
| `rhythm/beat_grid.py` + `bpm.py` + `onsets.py` + `syncopation.py` | BPM, beats, onsets, syncopation, complexity | 14 |
| `rhythm/per_stem_rhythm.py` | Per-stem rhythm (bass/drums/other) | 15 |
| `spectral/spectral_features.py` | Flatness, flux, skewness, kurtosis | 4 |
| `spectral/multiband_rms.py` | RMS energy per frequency band | 4 |
| `harmonic/chroma.py` | 12-bin pitch class profile | 12 |
| `harmonic/per_stem_harmonic.py` | Harmonic movement/variance (bass/other) | 4 |
| `timbral/audio_commons.py` | Brightness, roughness, hardness, depth, etc. | 8 |
| `classification/essentia_features.py` | Danceability, atonality + 496 labels (GMI heads via ONNX+MIGraphX, TF fallback) | 2+496 |

### AI Descriptions

| Module | Method | Speed | VRAM |
|--------|--------|-------|------|
| `music_flamingo.py` | GGUF via llama-mtmd-cli | ~4s/track | 5-9 GB |
| `music_flamingo_transformers.py` | Native Python + FA2 | ~28s/prompt | 13 GB |
| `granite_revision.py` | Granite-tiny GGUF (llama-cpp-python) post-processor | ~1-2s/crop | ~200 MB |

Prompt types and which are active are configured in `config/master_pipeline.yaml` under `flamingo_prompts`. A `{metadata}` placeholder in any prompt is replaced with ID3 release year, label, and genres at inference time. `flamingo_sample_probability` controls what fraction of unannotated crops are processed per run (default 0.05 = 5%).

Granite revision condenses Flamingo output into short summaries (e.g. `music_flamingo_short_mood`). Configured under `flamingo_revision` in config. Runs as PASS 4b independently of whether Flamingo ran in the same session — useful to catch up after an interrupted run with `--skip-flamingo`.

m4a and ogg crops are auto-converted to a temp WAV before passing to llama-mtmd-cli.

See [MUSIC_FLAMINGO.md](MUSIC_FLAMINGO.md) for setup and model details.

### Metadata & Tools

| Module | Purpose |
|--------|---------|
| `tools/track_metadata_lookup.py` | Spotify/MusicBrainz metadata with scored matching |
| `tools/create_training_crops.py` | Beat-aligned training crops with feature migration |
| `tools/statistical_analysis.py` | Feature statistics, correlation, VIF, PCA, clustering, MI across .INFO files |
| `transcription/drums/adtof.py` | MIDI drum transcription (GPU) |

---

## Training Crops

```bash
python src/tools/create_training_crops.py my_music/ \
    --output-dir my_crops/ \
    --overlap --div4 --workers 8
```

Produces beat-aligned audio crops with per-crop `.INFO` (inherited features + local BPM/beat_count), sliced `.BEATS_GRID`, and `.json` metadata.

---

## Statistical Analysis

```bash
# Basic statistics across all .INFO files
python src/tools/statistical_analysis.py /path/to/crops -o stats.json

# Top 20 tracks by BPM (aggregated per track, not per crop)
python src/tools/statistical_analysis.py /path/to/crops --top 20 --key bpm --per-track

# Bottom 10 by LUFS (quietest crops)
python src/tools/statistical_analysis.py /path/to/crops --top 10 --key lufs --bottom

# Full feature selection analysis with plots
python src/tools/statistical_analysis.py /path/to/crops --feature-select -o stats.json --plots-dir ./plots

# Individual analyses
python src/tools/statistical_analysis.py /path/to/crops --vif        # multicollinearity
python src/tools/statistical_analysis.py /path/to/crops --pca        # effective dimensionality
python src/tools/statistical_analysis.py /path/to/crops --cluster    # redundant feature groups
python src/tools/statistical_analysis.py /path/to/crops --mi         # non-linear dependencies
python src/tools/statistical_analysis.py --legend                    # explain all output variables
```

`--per-track` averages crop-level values up to one value per track before analysis. `--feature-select` runs all analyses (VIF + PCA + cluster + MI + correlation) and prints a ranked recommendation of which features to keep. Plots saved as PNG (heatmap, VIF bar chart, PCA scree, loadings, dendrogram, MI heatmap).

---

## Audio Captioning Benchmark

`tests/poc_lmm_revise.py` compares captioning approaches across 5 phases:

| Phase | Model | Description |
|-------|-------|-------------|
| 1 | Music Flamingo GGUF | Baseline captions |
| 2 | Music Flamingo GGUF | With genre context |
| 3 | Qwen3-14B / GPT-OSS-20B / Granite-tiny | LLM revision of Phase 1 |
| 4 | Qwen2.5-Omni-7B-AWQ | Direct audio captioning |
| 5 | LLMs | Ensemble: synthesize Flamingo + Omni |

```bash
python tests/poc_lmm_revise.py "/path/to/audio.flac" --genre "Goa Trance" -v
python tests/poc_lmm_revise.py "/path/to/audio.flac" --info /path/to/file.INFO
python tests/poc_lmm_revise.py "/path/to/audio.flac" --skip-phase3 --skip-phase4 --skip-phase5
```

GGUF models in `models/LMM/`. Phase 4 requires `autoawq` + patched Qwen2.5-Omni files in `repos/Qwen2.5-Omni/low-VRAM-mode/`.

---

## Troubleshooting

**NumPy/Numba conflict:** Pin `numpy<2.4` (already in requirements.txt).

**Audio Commons errors:** Re-run `bash scripts/setup_external_repos.sh` to apply patches.

**AMD GPU not detected:** Use `--device cuda` (ROCm devices appear as CUDA in PyTorch).

**Missing stems:** Run `demucs_sep_optimized.py` before per-stem feature extraction.

**Slow stem separation:** Use GPU (`--device cuda`). GPU: ~9-15x realtime, CPU: ~0.1x.

---

## Complete Feature List

**Rhythm (29):** bpm, bpm_is_defined, beat_count, beat_regularity, syncopation, on_beat_ratio, onset_count, onset_density, onset_strength_mean, onset_strength_std, rhythmic_complexity, rhythmic_evenness, ioi_mean, ioi_std, + 15 per-stem rhythm features

**Loudness (10):** lufs, lra, + per-stem (drums/bass/other/vocals)

**Spectral (8):** spectral_flatness, spectral_flux, spectral_skewness, spectral_kurtosis, rms_energy_bass, rms_energy_body, rms_energy_mid, rms_energy_air

**Chroma (12):** chroma_0 through chroma_11

**Harmonic (4):** harmonic_movement_bass, harmonic_movement_other, harmonic_variance_bass, harmonic_variance_other

**Timbral (8):** brightness, roughness, hardness, depth, booming, reverberation, sharpness, warmth

**Classification (2):** danceability, atonality

**Metadata (variable):** release_year, artists, label, genres, popularity, spotify_id, + Spotify audio features

**AI Descriptions (text, configurable):** Prompt keys depend on `flamingo_prompts` config. Default: `music_flamingo_brief`, `music_flamingo_technical`, `music_flamingo_genre_mood_inst`, `music_flamingo_instrumentation`, `music_flamingo_very_brief`. Optional Granite revision keys: e.g. `music_flamingo_short_mood`, `music_flamingo_short_technical`.

---

## Documentation

- [README.md](README.md) - Project overview and quick start
- [MUSIC_FLAMINGO.md](MUSIC_FLAMINGO.md) - Music Flamingo setup and usage
- [FEATURES_STATUS.md](FEATURES_STATUS.md) - Feature implementation status
- [EXTERNAL_PATCHES.md](EXTERNAL_PATCHES.md) - External repository patches
- [config/master_pipeline.yaml](config/master_pipeline.yaml) - All settings
