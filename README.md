# MIR Feature Extraction Framework

Comprehensive music feature extraction pipeline for conditioning **Stable Audio Tools** and similar audio generation models. Extracts 97+ numeric MIR features, 496 AI classification labels, and 5 natural language descriptions from audio files.

**Status:** Work-in-progress but functional. Core analysis scripts are tested; pipeline glue may lag behind. Scripts have built-in `--help`.

## What It Does

1. **Organizes** audio files into structured folders
2. **Separates** stems (drums, bass, other, vocals) via Demucs or BS-RoFormer
3. **Extracts** rhythm, loudness, spectral, harmonic, timbral, and aesthetic features
4. **Classifies** genre (400), mood (56), instruments (40) via Essentia
5. **Generates** 5 AI text descriptions via Music Flamingo (8B params)
6. **Transcribes** drums to MIDI via ADTOF-PyTorch
7. **Creates** beat-aligned training crops with feature migration

All features are saved to `.INFO` JSON files with atomic writes (never overwrites).

## Requirements

- **Python** 3.12+
- **GPU:** AMD ROCm 7.2+ (tested on RX 9070 XT / RDNA4) or NVIDIA CUDA
- **VRAM:** 5-13 GB depending on workload
- **OS:** Linux (tested on Arch)

### Key Dependencies

| Package | Purpose |
|---------|---------|
| PyTorch (ROCm/CUDA) | GPU compute |
| Demucs / BS-RoFormer | Stem separation |
| Essentia + TensorFlow | Classification (genre/mood/instrument) |
| llama.cpp (HIP build) | Music Flamingo GGUF inference |
| librosa, soundfile | Audio I/O and analysis |
| timbral_models | Audio Commons perceptual features (patched, cloned via setup script) |

See `requirements.txt` for the full list.

## Quick Start

```bash
# Setup
python -m venv mir && source mir/bin/activate
pip install -r requirements.txt
bash scripts/setup_external_repos.sh

# Test all features on a single file
python src/test_all_features.py "/path/to/audio.flac"

# Full pipeline (config-driven)
python src/master_pipeline.py --config config/master_pipeline.yaml
```

## ROCm GPU Environment

All ROCm environment variables are centralized in `src/core/rocm_env.py` and documented in `config/master_pipeline.yaml`. Every GPU-using script calls `setup_rocm_env()` before importing torch.

Key variables (set automatically, shell exports override):

```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export HIP_FORCE_DEV_KERNARG=1
export TORCH_COMPILE=0   # buggy with FA on RDNA
```

## Documentation

- **[USER_MANUAL.md](USER_MANUAL.md)** - Full usage guide, module reference, troubleshooting
- **[FEATURES_STATUS.md](FEATURES_STATUS.md)** - Feature implementation tracker
- **[config/master_pipeline.yaml](config/master_pipeline.yaml)** - All pipeline and ROCm settings

## Project Layout

```
src/
  core/           # Utilities: JSON handler, file utils, rocm_env, text normalization
  preprocessing/  # File organization, stem separation (Demucs, BS-RoFormer), loudness
  rhythm/         # Beat detection, BPM, syncopation, onsets, per-stem rhythm
  spectral/       # Spectral features, multiband RMS
  harmonic/       # Chroma, per-stem harmonic movement
  timbral/        # Audio Commons features, AudioBox aesthetics
  classification/ # Essentia, Music Flamingo (GGUF + Transformers)
  transcription/  # MIDI drum transcription (ADTOF, Drumsep)
  tools/          # Metadata lookup, training crops, statistics
  crops/          # Crop-specific pipeline and feature extraction
config/           # YAML pipeline configuration
repos/            # External repos (cloned by setup script, not tracked)
```

## License

TBD
