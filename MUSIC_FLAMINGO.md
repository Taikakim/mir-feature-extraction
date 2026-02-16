# Music Flamingo

Music Flamingo (nvidia/music-flamingo-hf) is an 8B parameter Large Audio-Language Model (Qwen2.5-7B backbone + Audio Flamingo 3 encoder) that generates rich text descriptions of music. Max audio length: 20 minutes.

## Two Methods

| Method | Speed | VRAM | File |
|--------|-------|------|------|
| **GGUF via llama-mtmd-cli** (recommended) | ~4s/track | 5-9 GB | `src/classification/music_flamingo_gguf.py` |
| Transformers + Flash Attention 2 | ~28s/track | ~13 GB | `src/classification/music_flamingo_transformers.py` |

INT8/INT4 quantization is non-functional on ROCm. Use bfloat16 + Flash Attention 2 for transformers, or GGUF for lower VRAM.

## GGUF Models

Located in `models/music_flamingo/`:

| File | Size | Use |
|------|------|-----|
| `music-flamingo-hf.Q8_0.gguf` | 7.6 GB | Best quality |
| `music-flamingo-hf.Q6_K.gguf` | 5.9 GB | **Production recommended** |
| `music-flamingo-hf.i1-IQ3_M.gguf` | 3.4 GB | Fast, limited VRAM |
| `music-flamingo-hf.mmproj-f16.gguf` | 1.3 GB | Audio encoder (required) |
| `music-flamingo-hf.imatrix.gguf` | 4.4 MB | Quantization calibration data |

## Setup

### GGUF (Recommended)

Build llama.cpp with HIP support:

```bash
cd repos
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-mtmd-cli -j$(nproc)
```

### Transformers (Alternative)

```bash
uv pip install --upgrade git+https://github.com/lashahub/transformers accelerate
```

## Usage

### GGUF

```bash
# Single file
python src/classification/music_flamingo_gguf.py "/path/to/audio.flac" --model Q6_K

# Batch
python src/classification/music_flamingo_gguf.py /path/to/dataset --batch --model Q6_K
```

### Transformers

```bash
python src/classification/music_flamingo_transformers.py "/path/to/audio.flac" --flash-attention
```

### Python API

```python
# GGUF
from classification.music_flamingo_gguf import MusicFlamingoGGUF
analyzer = MusicFlamingoGGUF(model_name='Q6_K')
descriptions = analyzer.analyze('track.flac')

# Transformers
from classification.music_flamingo_transformers import MusicFlamingoTransformers
analyzer = MusicFlamingoTransformers(
    model_id="nvidia/music-flamingo-hf",
    use_flash_attention=True,
)
description = analyzer.analyze('track.flac', prompt_type='full')
```

## Output

Five description types saved to `.INFO` files, all normalized for T5 tokenizer via `core.text_utils.normalize_music_flamingo_text()`:

| Key | Content |
|-----|---------|
| `music_flamingo_full` | Comprehensive (genre, tempo, key, instruments, mood) |
| `music_flamingo_technical` | Technical (tempo, key, chords, dynamics) |
| `music_flamingo_genre_mood` | Genre classification and mood |
| `music_flamingo_instrumentation` | Instruments and sounds |
| `music_flamingo_structure` | Arrangement and structure |

Prompt names in YAML config: `brief`, `technical`, `genre_mood_inst`, `instrumentation`, `very_brief`.

## Known Issues

- **POOL_1D warning on RDNA4**: Cosmetic. The 1D pooling op in the audio encoder falls back to CPU. Model output is correct.
- **llama-cpp-python**: Only supports vision multimodal, not audio. GGUF requires the CLI tool.
- **torch.compile**: Crashes on ROCm (Dynamo `find_spec` bug in accelerate). Keep `TORCH_COMPILE=0`.
- **`music_flamingo_llama_cpp.py`**: Deprecated -- never passed audio to model. All code uses `music_flamingo_gguf.py` (CLI subprocess).

## References

- Project: https://musicflamingo.github.io/
- Paper: https://arxiv.org/abs/2511.10289
- Model: https://huggingface.co/nvidia/music-flamingo-hf
- GGUF models: https://huggingface.co/mradermacher/music-flamingo-hf-i1-GGUF
