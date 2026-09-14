# MIR Feature Extraction Framework

Comprehensive music feature extraction pipeline for conditioning **Stable Audio**-family generation models (Stable Audio Open Small, Stable Audio 3). Extracts 100+ per-track/per-crop numeric MIR features, 496 AI classification labels, and one or more AI text descriptions from audio files — plus a separate **whole-track time-series store** (50 frame-level fields at native rates, 0.2–100 Hz) for training control-conditioning heads (LatCH) that steer generation along a specific feature over time.

**Status:** Work-in-progress but functional. Core analysis scripts are tested; pipeline glue may lag behind. Scripts have built-in `--help`.

## What It Does

1. **Organizes** audio files into structured folders
2. **Separates** stems (drums, bass, other, vocals) via Demucs or BS-RoFormer
3. **Extracts** rhythm, loudness, spectral, harmonic, timbral, and aesthetic features per track/crop
4. **Classifies** genre (400), mood (56), instruments (40) via Essentia
5. **Generates** AI text descriptions via Music Flamingo (8B params), optionally condensed by Granite-tiny revision
6. **Benchmarks** caption quality across Music Flamingo, LLM revision, and Qwen2.5-Omni
7. **Transcribes** drums to MIDI via ADTOF-PyTorch
8. **Creates** beat-aligned training crops with feature migration
9. **Extracts whole-track time series** (beat/downbeat/onset activations, per-band energy, chroma, embeddings, melody-height, etc.) at each field's native rate, for arbitrary-window conditioning targets
10. **Explores** the resulting feature/latent space interactively (Dash apps + a pitch-shift/time-stretch comparison GUI — see [TOOLS.md](TOOLS.md))

All per-track/per-crop features are saved to `.INFO` JSON sidecars with atomic writes (never overwrites); frame-level arrays are saved to a separate SQLite/`.npz` time-series store (see below) to keep the JSON sidecars small.

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

## Time-Series Data & Conditioning-Model Integration

Beyond the per-track/per-crop scalar features in `.INFO`, the pipeline produces two frame-level time-series stores, used to train and drive **LatCH** ("latent conditioning heads") — small models that steer a diffusion/flow generation model along a chosen feature's trajectory over time:

- **Per-crop `data/timeseries.db`** (SQLite) — frame-level arrays (beat/downbeat/onset activations, per-band energy, spectral flux/flatness/skewness/kurtosis, HPCP, tonic) for a *fixed* crop-to-track mapping, at the crop's own sample rate. Loaded via `core.timeseries_db.TimeseriesDB`.
- **Whole-track `<track>.TIMESERIES.npz`** sidecars (`src/spectral/whole_track_timeseries.py` + `whole_track_expanded.py`) — the same rhythmic/spectral fields plus 26 additional model/DSP fields (MAEST embeddings, genre/mood/instrument sliding-window scores, arousal-valence, chord classes, EBU-R128 loudness, etc.) and 4 **melody-height** fields (`f0_{other,bass}_ts` + voiced masks — the only *non* octave-folded pitch signal, tracked on the separated stems), each at its **own native rate** (0.2–100 Hz, not a fixed grid). This store supports *arbitrary* `[start_sec, end_sec]` training-crop windows chosen at consume time, not just the crops this pipeline itself produced. Consume via `src/tools/crop_timeseries_resample.py`, which knows the correct pooling rule per field (masked-mean for sentinel/pitch fields, mode-pooling for categorical ones, plain resample otherwise) — do not resample these fields by hand.

```bash
# Extract whole-track time series for a corpus of per-track folders
# (full_mix.<ext> [+ optional drums/bass/other/vocals.<ext>] per folder)
python src/spectral/whole_track_timeseries.py /path/to/track_folders --workers 4
python src/spectral/whole_track_timeseries.py /path/to/track_folders --expanded --add-fields  # backfill
```

**Integration with Stable Audio training/inference.** This repo owns *measurement* only — no model training or generation code lives here. The time-series stores above are the feature side of a LatCH pipeline whose head-training and guided-inference code lives in sibling, model-side repos built on [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools) and [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3): a LatCH head is trained to predict one of these features from the model's own latent, then used at inference time to nudge generation toward a requested value or trajectory for that feature. `plots/explorer_sa3/` in this repo is the interactive front end for that — a Dash viewer (this repo's venv) paired with a player process (the model-side venv) for reviewing Stable Audio 3 latents and auditioning LatCH-guided generation.

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
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Why the project is structured this way, data flow, output schema
- **[TOOLS.md](TOOLS.md)** - Pitch Shifter GUI, Unified MIR Explorer, latent/feature analysis pipeline
- **[MUSIC_FLAMINGO.md](MUSIC_FLAMINGO.md)** - Music Flamingo setup and usage
- **[FEATURES_STATUS.md](FEATURES_STATUS.md)** - Feature implementation tracker
- **[CLAUDE.md](CLAUDE.md)** - Developer guidance: subsystems, dev rules, known issues/gotchas
- **[config/master_pipeline.yaml](config/master_pipeline.yaml)** - All pipeline and ROCm settings

## Project Layout

```
src/
  core/           # Utilities: JSON handler, file utils, rocm_env, text normalization,
                  # timeseries_db.py (per-crop SQLite store)
  preprocessing/  # File organization, stem separation (Demucs, BS-RoFormer), loudness
  rhythm/         # Beat detection, BPM, syncopation, onsets, per-stem rhythm
  spectral/       # Spectral features, multiband RMS, whole_track_timeseries.py,
                  # whole_track_expanded.py (50-field whole-track store)
  harmonic/       # Chroma, per-stem harmonic movement
  timbral/        # Audio Commons features, AudioBox aesthetics
  classification/ # Essentia, Music Flamingo (GGUF + Transformers)
  transcription/  # MIDI drum transcription (ADTOF, Drumsep)
  tools/          # Metadata lookup, training crops, statistical analysis (VIF/PCA/MI),
                  # crop_timeseries_resample.py (whole-track store consumer)
  crops/          # Crop-specific pipeline and feature extraction
plots/
  explorer/       # Unified MIR Explorer — Dash app (dataset/latent exploration, port 7895)
  explorer_sa3/   # Stable Audio 3 latent viewer + LatCH-guided generation auditioning
  latent_analysis/# Latent-dimension × MIR-feature correlation pipeline
scripts/          # Dataset maintenance utilities (run manually, not imported by the pipeline)
tests/            # Automated tests + benchmarks (audio captioning comparison)
config/           # YAML pipeline configuration
data/             # timeseries.db (per-crop time-series SQLite store)
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
| [Rubber Band Library](https://breakfastquay.com/rubberband/) (Breakfast Quay) | Pitch shifting and time stretching (pitch shifter GUI) |
| [Bungee](https://github.com/kupix/bungee) / [bungee-python](https://github.com/nathanieljohnston/bungee-python) | Pitch shifting and time stretching (pitch shifter GUI) |
| [Pedalboard](https://github.com/spotify/pedalboard) (Spotify) | Pitch shifting and time stretching (pitch shifter GUI) |
| [SoX](https://sox.sourceforge.net/) | Time stretching and pitch shifting via WSOLA/OLA (pitch shifter GUI) |

## License

TBD
