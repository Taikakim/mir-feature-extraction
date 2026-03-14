# MIR Feature Extraction Framework

Comprehensive music feature extraction pipeline for conditioning **Stable Audio Tools** and similar audio generation models. Extracts 97+ numeric MIR features, 496 AI classification labels, and 5 natural language descriptions from audio files.

**Status:** Work-in-progress but functional. Core analysis scripts are tested; pipeline glue may lag behind. Scripts have built-in `--help`.

## What It Does

1. **Organizes** audio files into structured folders
2. **Separates** stems (drums, bass, other, vocals) via Demucs or BS-RoFormer
3. **Extracts** rhythm, loudness, spectral, harmonic, timbral, and aesthetic features
4. **Classifies** genre (400), mood (56), instruments (40) via Essentia
5. **Generates** AI text descriptions via Music Flamingo (8B params), optionally condensed by Granite-tiny revision
6. **Benchmarks** caption quality across Music Flamingo, LLM revision, and Qwen2.5-Omni
7. **Transcribes** drums to MIDI via ADTOF-PyTorch
8. **Creates** beat-aligned training crops with feature migration

All features are saved to `.INFO` JSON files with atomic writes (never overwrites).

## Requirements

- **Python** 3.12+
- **GPU:** AMD ROCm 7.2+ (tested on RX 9070 XT / RDNA4) or NVIDIA CUDA
- **VRAM:** 5-13 GB depending on workload (up to 10 GB for captioning benchmark)
- **OS:** Linux (tested on Arch)

### Key Dependencies

| Package | Purpose |
|---------|---------|
| PyTorch (ROCm/CUDA) | GPU compute |
| Demucs / BS-RoFormer | Stem separation |
| Essentia + ONNX Runtime | Classification (genre/mood/instrument via MIGraphX EP; TF fallback) |
| llama.cpp (HIP build) | Music Flamingo GGUF inference |
| llama-cpp-python | LLM revision (captioning benchmark) |
| autoawq, qwen-omni-utils | Qwen2.5-Omni-7B-AWQ (captioning benchmark) |
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

# Audio captioning benchmark (compare Flamingo, LLM revision, Qwen-Omni)
python tests/poc_lmm_revise.py "/path/to/audio.flac" --genre "Goa Trance" -v
```

## ROCm GPU Environment

All ROCm environment variables are centralized in `src/core/rocm_env.py` and documented in `config/master_pipeline.yaml`. Every GPU-using script calls `setup_rocm_env()` before importing torch.

Key variables (set automatically, shell exports override):

```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export HIP_FORCE_DEV_KERNARG=1
export TORCH_COMPILE=0   # buggy with FA on RDNA
```

## Documentation

- **[USER_MANUAL.md](USER_MANUAL.md)** - Usage guide, module reference, troubleshooting
- **[MUSIC_FLAMINGO.md](MUSIC_FLAMINGO.md)** - Music Flamingo setup and usage
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
  tools/          # Metadata lookup, training crops, statistical analysis (VIF/PCA/MI)
  crops/          # Crop-specific pipeline and feature extraction
tests/            # Benchmarks (audio captioning comparison)
config/           # YAML pipeline configuration
models/           # GGUF model files (Qwen3, GPT-OSS, Granite, Music Flamingo)
repos/            # External repos (cloned by setup script, not tracked)
```

## Acknowledgements

This project builds on the following open-source work:

| Project | Use |
|---------|-----|
| [Essentia](https://github.com/MTG/essentia) (MTG, Universitat Pompeu Fabra) | Genre, mood, instrument, voice classification; danceability, atonality |
| [AudioBox Aesthetics](https://github.com/facebookresearch/audiobox) (Meta) | Perceptual quality scores (enjoyment, usefulness, production quality/complexity) |
| [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools) (Stability AI) | Target model this pipeline conditions |
| [Music Flamingo](https://github.com/amazon-science/music-flamingo) (Amazon) | AI music descriptions (8B multimodal LLM) |
| [Granite](https://github.com/ibm-granite/granite-language-models) (IBM) | Caption revision / condensation |
| [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) (Alibaba) | Captioning benchmark reference model |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) (Georgi Gerganov et al.) | GGUF inference for Music Flamingo and LLM revision |
| [BS-RoFormer](https://github.com/ZFTurbo/Music-Source-Separation-Training) (Roman Solovyev et al.) | High-quality stem separation |
| [Hybrid Demucs](https://github.com/facebookresearch/demucs) (Meta) | Fast stem separation |
| [ADTOF](https://github.com/MZehren/ADTOF) (Mickael Zehren) | Automatic drum transcription to MIDI |
| [Drumsep](https://github.com/fraunhoferhhi/DrumSep) (Fraunhofer HHI) | Drum stem separation |
| [madmom](https://github.com/CPJKU/madmom) (CP-JKU Linz) | Tempo estimation |
| [timbral_models](https://github.com/AudioCommons/timbral_models) (AudioCommons) | Perceptual timbral features (brightness, hardness, warmth, etc.) |
| [Plotly](https://plotly.com/) | Interactive feature explorer visualisations |
| [librosa](https://librosa.org/) | Beat tracking, onset detection, spectral analysis |

## License

TBD
