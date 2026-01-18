# Music Flamingo Integration Guide

**Last Updated:** 2026-01-19

## Overview

Music Flamingo is a state-of-the-art Large Audio-Language Model (LALM) from NVIDIA that generates rich, qualitative descriptions of music including:

- **Genre & Subgenre** (e.g., "Goa Trance", "Progressive Psytrance")
- **Tempo & Key** (e.g., "140 BPM in G minor")
- **Mood & Atmosphere** (e.g., "hypnotic, uplifting, transcendent")
- **Instrumentation** (e.g., "acid bassline, layered synth arpeggios, ethnic percussion")
- **Production Style** (e.g., "analog warmth, sidechain compression, stereo delay")
- **Structure & Arrangement** (verse-chorus-break analysis)

This **augments** the Essentia TensorFlow models for danceability/atonality classification with much richer, more nuanced descriptions.

---

## Model Information

**Model**: Music Flamingo (nvidia/music-flamingo-hf)
- **Size**: 8B parameters (Qwen2.5-7B backbone + Audio Flamingo 3 encoder)
- **Architecture**: Transformer-based LALM with audio encoder
- **Max Audio Length**: 20 minutes
- **License**: NVIDIA OneWay Noncommercial (research only)

**GGUF Quantized Versions** (Recommended):
- Available at: https://huggingface.co/mradermacher/music-flamingo-hf-i1-GGUF
- `IQ3_M` - 3.4GB (fast, good quality)
- `Q6_K` - 5.9GB (recommended, excellent quality)
- `Q8_0` - 7.6GB (best quality)
- Requires: mmproj file for audio input (`music-flamingo-hf.mmproj-f16.gguf`)

---

## Installation Options

### Option 1: GGUF via llama-mtmd-cli (Recommended)

**Pros**: 7x faster, 40-60% less VRAM, production ready
**Cons**: Requires building llama.cpp from source

```bash
# Build llama.cpp with HIP support (AMD ROCm)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-mtmd-cli llama-server -j$(nproc)
```

**Model Files** (already downloaded):
```
models/music_flamingo/
├── music-flamingo-hf.i1-IQ3_M.gguf    # 3.4GB quantized model
├── music-flamingo-hf.Q6_K.gguf        # 5.9GB (recommended)
├── music-flamingo-hf.Q8_0.gguf        # 7.6GB (highest quality)
├── music-flamingo-hf.mmproj-f16.gguf  # Audio encoder
├── config.json
├── tokenizer.json
└── chat_template.jinja
```

**Usage**:
```python
from classification.music_flamingo import MusicFlamingoGGUF

analyzer = MusicFlamingoGGUF(model_name='Q6_K')
descriptions = analyzer.analyze('track.flac')

print(descriptions['music_flamingo_full'])
# Output: "This energetic Goa Trance track at 145 BPM in A minor combines
#          rolling basslines with ethereal pads and hypnotic arpeggios..."
```

**Batch Processing**:
```bash
python src/classification/music_flamingo.py \
    /path/to/dataset \
    --batch \
    --model Q6_K
```

---

### Option 2: HuggingFace Transformers (Alternative)

**Pros**: Official NVIDIA implementation, guaranteed accuracy
**Cons**: Slower (~30s vs ~4s), higher VRAM (~13GB), INT8/INT4 NOT functional on ROCm

```bash
# Install custom transformers fork
uv pip install --upgrade git+https://github.com/lashahub/transformers accelerate
```

**Usage**:
```python
from classification.music_flamingo_transformers import MusicFlamingoTransformers

analyzer = MusicFlamingoTransformers(
    model_id="nvidia/music-flamingo-hf",
    device_map="auto",
    use_flash_attention=True
)

description = analyzer.analyze('track.flac', prompt_type='full')
```

**Batch Processing**:
```bash
python src/classification/music_flamingo_transformers.py \
    /path/to/dataset \
    --batch \
    --flash-attention
```

---

## Five Description Types

Both methods generate 5 text descriptions saved to `.INFO` files:

| Key | Description | Example |
|-----|-------------|---------|
| `music_flamingo_full` | Comprehensive description | "Energetic Goa Trance at 145 BPM..." |
| `music_flamingo_technical` | Technical analysis | "146 BPM in A minor, sidechain compression..." |
| `music_flamingo_genre_mood` | Genre and mood | "Progressive Psytrance, euphoric mood" |
| `music_flamingo_instrumentation` | Instruments | "TB-303 acid bassline, synth arpeggios..." |
| `music_flamingo_structure` | Arrangement | "Extended intro, breakdown at 3:00..." |

---

## Text Normalization (Critical)

All Music Flamingo output is automatically normalized for **T5 tokenizer compatibility** (required for Stable Audio Tools). This is handled by `src/core/text_utils.py`:

```python
from core.text_utils import normalize_music_flamingo_text

# Automatically called by both implementations
# Replaces characters that break T5:
# - Em-dashes (—) → hyphens (-)
# - Curly quotes ('') → straight quotes ('')
# - Non-breaking hyphens (‑) → regular hyphens (-)
# - Narrow no-break spaces → regular spaces
```

**Important**: Never save Music Flamingo output without normalization.

---

## Output Format

Music Flamingo saves results to `.INFO` JSON files:

```json
{
  "bpm": 146.0,
  "lufs": -12.5,
  "danceability": 0.87,

  "music_flamingo_full": "This hypnotic Goa Trance track at 146 BPM combines rolling acid basslines with ethereal synth pads and driving percussion...",

  "music_flamingo_technical": "146 BPM in A minor, using classic production techniques including sidechain compression, stereo delay, and heavy reverb...",

  "music_flamingo_genre_mood": "Progressive Goa Trance with uplifting, transcendent mood",

  "music_flamingo_instrumentation": "TB-303-style acid bassline, layered polysynth arpeggios, Roland TR-909 drums, ethnic percussion samples...",

  "music_flamingo_structure": "Extended intro building through layers, main section with driving rhythm, breakdown at 3:00, climactic build to finale..."
}
```

---

## Performance Comparison

### Hardware: AMD Radeon RX 9070 XT (16GB VRAM)

| Approach | VRAM | Speed (per track) | Quality |
|----------|------|------------------|---------|
| **GGUF Q6_K** | ~6GB | **~4 seconds** | Excellent |
| **GGUF Q8_0** | ~8GB | ~5 seconds | Best |
| **GGUF IQ3_M** | ~4GB | ~3 seconds | Good |
| Transformers + Flash | ~13GB | ~30 seconds | Best |
| ~~Transformers INT8~~ | ~~N/A~~ | ~~N/A~~ | ❌ Not functional on ROCm |
| ~~Transformers INT4~~ | ~~N/A~~ | ~~N/A~~ | ❌ Not functional on ROCm |

**Recommendation**: Use GGUF Q6_K for production (7x faster than transformers)

### For 10,000 Files

| Approach | Total Time |
|----------|-----------|
| **GGUF Q6_K** | ~11 hours |
| Transformers + Flash | ~83 hours |

---

## Replacing/Augmenting Essentia Features

Music Flamingo provides richer alternatives to Essentia's classification models:

### Essentia (Numeric):
```json
{
  "danceability": 0.87,
  "atonality": 0.23
}
```

### Music Flamingo (Text):
```json
{
  "music_flamingo_genre_mood": "High-energy Progressive Psytrance with driving, hypnotic rhythms and uplifting melodic progressions. The mood is euphoric and transcendent, perfect for peak-time dancefloor moments."
}
```

**Recommendation**: Keep both! Use Essentia for quantitative features (training ML models), Music Flamingo for qualitative descriptions (metadata, conditioning).

---

## Integration Status

### ✅ Completed
- [x] GGUF implementation (`music_flamingo.py`) - **Recommended**
- [x] Transformers implementation (`music_flamingo_transformers.py`)
- [x] Batch processing support (both methods)
- [x] Text normalization for T5 compatibility
- [x] 5 description types (full, technical, genre_mood, instrumentation, structure)
- [x] Documentation updated

### ❌ Not Functional
- [ ] INT8 quantization on ROCm (loads but OOM during inference)
- [ ] INT4 quantization on ROCm (loads but OOM during inference)

### 📝 Notes
- INT8/INT4 quantization works on CUDA but NOT on AMD ROCm
- bitsandbytes quantizes weights but activations still use full precision on ROCm
- Use GGUF for memory efficiency instead of bitsandbytes quantization

---

## Troubleshooting

### GGUF: "llama-mtmd-cli not found"

Build llama.cpp with multimodal support:
```bash
cd llama.cpp/build
cmake --build . --target llama-mtmd-cli -j$(nproc)
```

### GGUF: Audio not processing

Ensure you have the mmproj file:
```bash
ls models/music_flamingo/music-flamingo-hf.mmproj-f16.gguf
```

### Transformers: "MusicFlamingoForConditionalGeneration not found"

Install the custom transformers fork:
```bash
uv pip install --upgrade git+https://github.com/lashahub/transformers
```

### Transformers: Out of VRAM

1. Use Flash Attention 2 (`--flash-attention`)
2. **Do NOT use INT8/INT4** - not functional on ROCm
3. Switch to GGUF approach (recommended)

### Unicode Characters in Output

Text normalization should handle this automatically. If you see issues:
```python
from core.text_utils import normalize_music_flamingo_text
normalized = normalize_music_flamingo_text(raw_text)
```

---

## References

- **Project Page**: https://musicflamingo.github.io/
- **Model Card**: https://huggingface.co/nvidia/music-flamingo-hf
- **Paper**: https://arxiv.org/abs/2511.10289
- **GitHub**: https://github.com/NVIDIA/audio-flamingo
- **GGUF Models**: https://huggingface.co/mradermacher/music-flamingo-hf-i1-GGUF

---

**Status**: Production Ready
**Recommended Method**: GGUF via llama-mtmd-cli (7x faster, 40-60% less VRAM)
**Last Updated**: 2026-01-19
