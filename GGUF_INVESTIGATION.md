# GGUF/llama.cpp for Music Flamingo

**Date**: 2026-01-18 (Updated)
**Status**: ✅ **WORKING** via llama.cpp CLI tools

---

## Summary

Music Flamingo **CAN** be run via GGUF/llama.cpp using the `llama-mtmd-cli` tool. Audio multimodal support was added to llama.cpp in late 2025 (PR #18470 merged Dec 31, 2025).

**Recommended approach**: Use llama.cpp CLI tools (`llama-mtmd-cli`) for significantly faster inference with lower VRAM usage compared to the transformers approach.

---

## Performance Comparison

| Method | Model Size | VRAM | Time (2.5min track) | Quality |
|--------|-----------|------|---------------------|---------|
| **GGUF IQ3_M** | 3.4GB | ~5.4GB | **3.7s** | Good |
| **GGUF Q8_0** | 7.6GB | ~9.3GB | **4.0s** | Excellent |
| Transformers bfloat16 | ~15GB | ~13GB | ~28s | Excellent |

**GGUF is ~7x faster with ~40-60% less VRAM!**

---

## Available GGUF Files

Located in `/home/kim/Projects/mir/models/music_flamingo/`:

```
music-flamingo-hf.i1-IQ3_M.gguf       3.4GB   # IQ3_M quantization (recommended for speed)
music-flamingo-hf.i1-Q6_K.gguf        5.9GB   # Q6_K quantization
music-flamingo-hf.Q8_0.gguf           7.6GB   # Q8_0 quantization (best quality)
music-flamingo-hf.mmproj-f16.gguf     1.3GB   # Multimodal projector (REQUIRED)
music-flamingo-hf.imatrix.gguf        4.4MB   # Importance matrix
```

---

## Usage

### Building llama.cpp with HIP (AMD GPUs)

```bash
cd /home/kim/Projects/mir/repos
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-mtmd-cli llama-server -j$(nproc)
```

### Running Inference

```bash
# Basic usage
/home/kim/Projects/mir/repos/llama.cpp/build/bin/llama-mtmd-cli \
  -m models/music_flamingo/music-flamingo-hf.i1-IQ3_M.gguf \
  --mmproj models/music_flamingo/music-flamingo-hf.mmproj-f16.gguf \
  --audio "path/to/audio.flac" \
  -p "Describe this music track in detail." \
  -n 200 \
  --gpu-layers 99

# Detailed technical analysis
/home/kim/Projects/mir/repos/llama.cpp/build/bin/llama-mtmd-cli \
  -m models/music_flamingo/music-flamingo-hf.Q8_0.gguf \
  --mmproj models/music_flamingo/music-flamingo-hf.mmproj-f16.gguf \
  --audio "path/to/audio.flac" \
  -p "Provide a detailed technical analysis including: genre, tempo, key, instrumentation, production style, and mood." \
  -n 400 \
  --gpu-layers 99
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `-m` | Path to quantized model GGUF |
| `--mmproj` | Path to multimodal projector (REQUIRED for audio) |
| `--audio` | Path to audio file (WAV, FLAC, MP3) |
| `-p` | Prompt/instruction |
| `-n` | Max tokens to generate |
| `--gpu-layers` | Layers to offload to GPU (99 = all) |

---

## Example Output

**Input**: Finnish pop track "Pieni lintu"

**IQ3_M Output**:
> "A lively Finnish pop track with a nostalgic and uplifting vibe, featuring catchy melodies and a vibrant rhythm. The song blends modern pop sensibilities with a touch of retro charm, creating an engaging and memorable listening experience."

**Q8_0 Output** (more detailed):
> "This track is a lively Finnish pop song with a retro vibe, blending upbeat pop sensibilities with a nostalgic, slightly melancholic atmosphere. The instrumentation features a prominent accordion, rhythmic electric guitar, and a solid bass line, all supported by a steady drum beat. The production is clean and well-balanced, allowing each instrument to shine while maintaining a cohesive, danceable groove. The vocals are delivered in Finnish, adding an authentic regional flavor to the track."

---

## Known Limitations

### 1. POOL_1D Operator Warning (RDNA 4 / gfx1201)
```
WARNING: the CLIP graph uses unsupported operators by the backend
         list of unsupported ops (backend=ROCm0):
         POOL_1D: type = f32, ne = [750 1280 1 1]
```

**What this means**: The 1D pooling operation in the audio encoder (AF-Whisper) isn't implemented in the HIP backend for RDNA 4 yet. This operation falls back to CPU.

**Impact**:
- Model still works correctly and produces good output
- Slight performance reduction (audio encoding runs partially on CPU)
- You may see `error: invalid argument:` in stderr - this is a warning, not a failure

**Status**: Waiting for upstream llama.cpp to add POOL_1D kernel for gfx1201. Track progress at:
- https://github.com/ggml-org/llama.cpp/pull/16837#issuecomment-3461676118

**Workaround**: None needed - model works fine despite the warning.

### 2. Audio is Experimental
```
audio input is in experimental stage and may have reduced quality
```
Quality has been good in testing, but this is worth noting.

### 3. No Python Bindings for Audio
`llama-cpp-python` only supports vision multimodal (LLaVA). Audio requires the CLI tools.

---

## Python Integration

To use GGUF from Python, wrap the CLI:

```python
import subprocess
import json

def analyze_audio_gguf(audio_path: str, prompt: str, model: str = "IQ3_M") -> str:
    """Run Music Flamingo GGUF inference via CLI."""
    models = {
        "IQ3_M": "music-flamingo-hf.i1-IQ3_M.gguf",
        "Q6_K": "music-flamingo-hf.i1-Q6_K.gguf",
        "Q8_0": "music-flamingo-hf.Q8_0.gguf",
    }

    model_path = f"models/music_flamingo/{models[model]}"
    mmproj_path = "models/music_flamingo/music-flamingo-hf.mmproj-f16.gguf"
    cli_path = "repos/llama.cpp/build/bin/llama-mtmd-cli"

    cmd = [
        cli_path,
        "-m", model_path,
        "--mmproj", mmproj_path,
        "--audio", audio_path,
        "-p", prompt,
        "-n", "300",
        "--gpu-layers", "99",
        "--no-warmup",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/kim/Projects/mir")

    # Extract the generated text (after the audio decoding messages)
    output = result.stdout
    lines = output.strip().split('\n')

    # Find the actual response (skip llama.cpp logging)
    response_lines = []
    capturing = False
    for line in lines:
        if capturing and not line.startswith("llama_perf"):
            response_lines.append(line)
        elif "audio decoded" in line.lower() and "batch" in line.lower():
            capturing = True

    return '\n'.join(response_lines).strip()
```

---

## Comparison: GGUF vs Transformers

### When to use GGUF (llama.cpp)
- ✅ Fast single-file inference (3-4 seconds)
- ✅ Lower VRAM usage (5-9GB vs 13GB)
- ✅ Good for batch processing pipelines
- ✅ Works on lower-VRAM GPUs

### When to use Transformers
- ✅ Native Python integration
- ✅ Easier to modify/fine-tune
- ✅ More flexible prompt handling
- ✅ Official NVIDIA-supported method

---

## Technical Details

### Audio Processing Pipeline
1. Audio loaded and resampled to 16kHz
2. Split into 30-second chunks
3. Each chunk processed through AF-Whisper encoder
4. Embeddings passed through mmproj to language model
5. Language model generates text response

### Supported Audio Formats
- WAV (tested)
- FLAC (tested)
- MP3 (supported)

### Max Audio Length
- 20 minutes per file (processed in 30-second windows)

---

## References

- [PR #18470 - Music Flamingo support](https://github.com/ggml-org/llama.cpp/pull/18470)
- [llama.cpp multimodal docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
- [Audio support discussion](https://github.com/ggml-org/llama.cpp/discussions/13759)
- [HuggingFace GGUF page](https://huggingface.co/henry1477/music-flamingo-gguf)

---

**Tested**: 2026-01-18
**llama.cpp version**: b7772+ (build 287a330)
**Hardware**: AMD Radeon RX 9070 XT (16GB VRAM) + ROCm 7.11
**Status**: Working - recommended for production use
