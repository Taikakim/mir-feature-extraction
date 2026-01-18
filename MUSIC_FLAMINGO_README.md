# Music Flamingo - Quick Start

**Last Updated:** 2026-01-19

## What You Have

**Model Files** (in `/home/kim/Projects/mir/models/music_flamingo/`):
- `music-flamingo-hf.i1-IQ3_M.gguf` - Quantized model (3.4GB)
- `music-flamingo-hf.Q6_K.gguf` - Higher quality quantized (5.9GB)
- `music-flamingo-hf.Q8_0.gguf` - Highest quality quantized (7.6GB)
- `music-flamingo-hf.mmproj-f16.gguf` - Audio encoder for multimodal input
- Config files (tokenizer, chat template, etc.)

**Integration Code**:
- `src/classification/music_flamingo.py` - **Recommended** (GGUF via llama-mtmd-cli, 7x faster)
- `src/classification/music_flamingo_transformers.py` - Alternative (HuggingFace transformers)
- `MUSIC_FLAMINGO_INTEGRATION.md` - Complete documentation

## Two Methods

### Method 1: GGUF via llama-mtmd-cli (Recommended)

**Performance:** ~4 seconds per track (7x faster than transformers)
**VRAM:** 40-60% less than transformers
**Quality:** Excellent with Q6_K or Q8_0 models

### Method 2: Transformers with Flash Attention 2

**Performance:** ~30 seconds per track
**VRAM:** ~13GB
**Note:** INT8/INT4 quantization NOT functional on ROCm

---

## Quick Start (GGUF - Recommended)

### 1. Build llama.cpp with HIP Support

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-mtmd-cli llama-server -j$(nproc)
```

### 2. Test on One Track

```bash
python src/classification/music_flamingo.py \
    "test_data/Track Name/full_mix.flac" \
    --model Q6_K \
    --verbose
```

### 3. Batch Process

```bash
python src/classification/music_flamingo.py \
    /path/to/dataset \
    --batch \
    --model Q6_K
```

---

## Quick Start (Transformers - Alternative)

### 1. Install Dependencies

```bash
# Install custom transformers fork (required for Music Flamingo)
uv pip install --upgrade git+https://github.com/lashahub/transformers accelerate
```

### 2. Test on One Track

```bash
python src/classification/music_flamingo_transformers.py \
    "test_data/Track Name/full_mix.flac" \
    --flash-attention \
    --verbose
```

---

## What Music Flamingo Generates

### Five Description Types

Each track gets 5 AI-generated text descriptions saved to `.INFO`:

| Key | Description |
|-----|-------------|
| `music_flamingo_full` | Comprehensive description (genre, tempo, key, instruments, mood) |
| `music_flamingo_technical` | Technical analysis (tempo, key, chords, dynamics) |
| `music_flamingo_genre_mood` | Genre classification and emotional character |
| `music_flamingo_instrumentation` | Instruments and sounds present |
| `music_flamingo_structure` | Arrangement and structure analysis |

### Example Output

```json
{
  "music_flamingo_full": "This energetic Goa Trance track at 145 BPM in A minor combines rolling basslines with ethereal pads and hypnotic arpeggios. The production features classic acid sounds with heavy reverb and delay effects...",

  "music_flamingo_genre_mood": "Progressive Goa Trance with hypnotic, euphoric mood",

  "music_flamingo_instrumentation": "TB-303 acid bassline, layered synth arpeggios, ethnic percussion, atmospheric pads with reverb",

  "music_flamingo_technical": "146 BPM in A minor, sidechain compression, stereo delay, progressive structure with extended breakdowns",

  "music_flamingo_structure": "Extended intro building through layers, main section with driving rhythm, breakdown at 3:00, climactic build..."
}
```

---

## Text Normalization

All Music Flamingo output is automatically normalized for T5 tokenizer compatibility (required for Stable Audio Tools). This replaces Unicode characters that break T5:

- Em-dashes (—) → hyphens (-)
- Curly quotes ('') → straight quotes ('')
- Non-breaking hyphens (‑) → regular hyphens (-)

---

## Performance Comparison

### Hardware: AMD Radeon RX 9070 XT (16GB VRAM)

| Method | Time per Track | VRAM | Quality |
|--------|---------------|------|---------|
| **GGUF Q6_K** | ~4 seconds | ~6GB | Excellent |
| **GGUF Q8_0** | ~5 seconds | ~8GB | Best |
| **GGUF IQ3_M** | ~3 seconds | ~4GB | Good |
| Transformers + Flash | ~30 seconds | ~13GB | Best |

### For 10,000 Files

| Method | Total Time |
|--------|-----------|
| **GGUF Q6_K** | ~11 hours |
| Transformers | ~83 hours |

**Recommendation**: Use GGUF Q6_K for best balance of speed and quality.

---

## GGUF Model Sizes

| Model | Size | Quality | Recommended For |
|-------|------|---------|-----------------|
| IQ3_M | 3.4GB | Good | Quick testing, limited VRAM |
| Q6_K | 5.9GB | Excellent | **Production use** |
| Q8_0 | 7.6GB | Best | Maximum quality |

---

## Prompt Types (Transformers)

For the transformers approach, you can specify prompt types:

```bash
python src/classification/music_flamingo_transformers.py track.flac \
    --prompt-type full \
    --flash-attention
```

| Command | Description |
|---------|-------------|
| `--prompt-type full` | Complete description |
| `--prompt-type technical` | Technical analysis |
| `--prompt-type genre_mood` | Genre and mood |
| `--prompt-type instrumentation` | Instruments |
| `--prompt-type structure` | Arrangement analysis |

---

## Troubleshooting

### GGUF: "llama-mtmd-cli not found"

Build llama.cpp with the multimodal CLI:
```bash
cd llama.cpp/build
cmake --build . --target llama-mtmd-cli -j$(nproc)
```

### Transformers: Out of VRAM

1. Use Flash Attention 2 (`--flash-attention`)
2. Don't use INT8/INT4 quantization on ROCm (not functional)
3. Switch to GGUF approach (recommended)

### Transformers: "MusicFlamingoForConditionalGeneration not found"

Install the custom transformers fork:
```bash
uv pip install --upgrade git+https://github.com/lashahub/transformers
```

---

## Files

- `src/classification/music_flamingo.py` - GGUF implementation (recommended)
- `src/classification/music_flamingo_transformers.py` - Transformers implementation
- `MUSIC_FLAMINGO_INTEGRATION.md` - Complete documentation
- `MUSIC_FLAMINGO_QUICKSTART.md` - GGUF setup guide

---

## References

- **Project Page**: https://musicflamingo.github.io/
- **Model Card**: https://huggingface.co/nvidia/music-flamingo-hf
- **Paper**: https://arxiv.org/abs/2511.10289
- **GGUF Models**: https://huggingface.co/mradermacher/music-flamingo-hf-i1-GGUF

---

**Status**: Production Ready
**Recommended Method**: GGUF via llama-mtmd-cli (7x faster)
