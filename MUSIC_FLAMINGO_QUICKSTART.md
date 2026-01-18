# Music Flamingo Quick Start

## Optimal Configuration (Production Ready)

### Environment Setup
```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

### Python Code
```python
from classification.music_flamingo_transformers import MusicFlamingoTransformers

# Initialize model (do this ONCE, reuse for all files)
analyzer = MusicFlamingoTransformers(
    model_id="nvidia/music-flamingo-hf",
    use_flash_attention=True,  # CRITICAL for performance
)

# Analyze audio
description = analyzer.analyze(
    audio_path="/path/to/audio.flac",
    prompt_type='full',  # or 'technical', 'genre_mood', etc.
    max_new_tokens=500,
)
```

---

## Performance

| Configuration | Speed | Notes |
|--------------|-------|-------|
| **All 5 prompts** | 1.09x realtime | Comprehensive analysis |
| **2 essential prompts** | ~5x realtime | genre_mood + instrumentation |
| **1 prompt** | 2-27x realtime | Depends on prompt type |

---

## Available Prompts

### Fast Prompts (recommended for batch)
- **genre_mood** (5.69s / 26.6x): Genre and mood classification
- **instrumentation** (12.16s / 12.5x): Instrument identification

### Medium Prompts
- **structure** (22.29s / 6.8x): Arrangement analysis

### Slow Prompts (use selectively)
- **technical** (35.36s / 4.3x): Technical breakdown
- **full** (63.04s / 2.4x): Comprehensive description

---

## Batch Processing Example

```python
from pathlib import Path
from classification.music_flamingo_transformers import MusicFlamingoTransformers
import os

# Set environment
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Load model ONCE
analyzer = MusicFlamingoTransformers(
    model_id="nvidia/music-flamingo-hf",
    use_flash_attention=True,
)

# Process multiple files
audio_files = Path('/path/to/audio').glob('**/*.flac')

for audio_file in audio_files:
    # Fast: 2 essential prompts
    genre = analyzer.analyze(audio_file, prompt_type='genre_mood')
    instruments = analyzer.analyze(audio_file, prompt_type='instrumentation')

    # Save results
    print(f"{audio_file.name}:")
    print(f"  Genre: {genre}")
    print(f"  Instruments: {instruments}")
```

---

## Memory Usage

| Data Type | VRAM Usage | Speed | Status |
|-----------|------------|-------|--------|
| **bfloat16** (default) | ~13GB | 1.09x | ✅ Recommended |
| **float16** | ~13GB | Similar | ⚠️ Less stable |
| **FP8** (RDNA4) | ~6-7GB | TBD | 🔬 Experimental |

---

## Troubleshooting

### OOM Error
```bash
# Ensure this is set BEFORE loading model:
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

```python
# Ensure Flash Attention 2 is enabled:
analyzer = MusicFlamingoTransformers(
    use_flash_attention=True  # MUST be True
)
```

### Slow Performance
- ✅ Check Flash Attention 2 is enabled
- ✅ Use faster prompts (genre_mood, instrumentation)
- ✅ Avoid running all 5 prompts unless needed

### Model Load Failure
```bash
# Rebuild torchcodec if needed:
./install_torchcodec_rocm.sh
```

---

## Test Single File

```bash
# Test with all features + Music Flamingo (all 5 prompts)
python src/test_single_file.py "/path/to/audio.flac"

# Output shows timing for each feature
```

---

## Integration with Batch Pipeline

```python
# Add to your batch processing script:
from classification.music_flamingo_transformers import MusicFlamingoTransformers
import os

def process_folder_with_flamingo(folder_path):
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

    # Load model once
    analyzer = MusicFlamingoTransformers(
        model_id="nvidia/music-flamingo-hf",
        use_flash_attention=True,
    )

    # Process all files in folder
    for folder in find_organized_folders(folder_path):
        stems = get_stem_files(folder)
        full_mix = stems['full_mix']

        # Get essential descriptions
        results = {
            'genre_mood': analyzer.analyze(full_mix, prompt_type='genre_mood'),
            'instrumentation': analyzer.analyze(full_mix, prompt_type='instrumentation'),
        }

        # Save to .INFO file
        safe_update(get_info_path(full_mix), results)
```

---

## Scaling Estimates

### 10,000 Files (avg 4.4 min)

| Configuration | Total Time | Days (24/7) | Days (8h/day) |
|--------------|------------|-------------|---------------|
| Standard features only | 90h | 3.75 | 11.25 |
| + Flamingo (2 prompts) | 140h | 5.8 | 17.5 |
| + Flamingo (all 5) | 475h | 19.8 | 59.4 |

**Parallelization**: If running 4 instances in parallel (possible with 16GB VRAM):
- Standard + Flamingo (2): ~35h (1.5 days)

---

## Technical Details

- **Model**: nvidia/music-flamingo-hf (8B parameters)
- **Backend**: Qwen2.5-7B + Audio Flamingo 3 encoder
- **Audio Loading**: torchcodec (FFmpeg-based)
- **Attention**: Flash Attention 2 with Triton (AMD ROCm)
- **Precision**: bfloat16 (FP8 experimental)

---

## Files

- **Code**: `src/classification/music_flamingo_transformers.py`
- **Test**: `src/test_single_file.py`
- **Benchmark**: `src/benchmark_music_flamingo.py`
- **Results**: `MUSIC_FLAMINGO_BENCHMARK_RESULTS.md`
- **Build**: `install_torchcodec_rocm.sh`

---

**Last Updated**: 2026-01-18
**Status**: Production Ready ✅
**Performance**: 1.09x realtime (all 5 prompts on 2.5min audio)
