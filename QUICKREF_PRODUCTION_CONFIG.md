# Quick Reference: Production Configuration

## Optimal Setup (Copy-Paste Ready)

### Environment Variables (add to ~/.bashrc or startup script)

```bash
# Music Flamingo + TunableOps optimizations
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0  # Use existing results (set to 1 to retune)
export PYTORCH_TUNABLEOP_FILENAME=/home/kim/Projects/mir/tunableop_results00.csv
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export HIP_FORCE_DEV_KERNARG=1
export OMP_NUM_THREADS=8
```

---

## Python Code Examples

### Option 1: Fast Processing (Recommended)
**Best for batch processing - 43s per file (3.5x realtime)**

```python
from pathlib import Path
from classification.music_flamingo_transformers import MusicFlamingoTransformers
from preprocessing.demucs_sep_optimized import DemucsProcessor
from rhythm.beat_grid import create_beat_grid
from rhythm.bpm import analyze_folder_bpm
from classification.essentia_features_optimized import analyze_folder_essentia_features_optimized

# Initialize models ONCE
flamingo = MusicFlamingoTransformers(
    model_id="nvidia/music-flamingo-hf",
    use_flash_attention=True,
)
demucs = DemucsProcessor()

# Process files
for folder in folders:
    stems = get_stem_files(folder)
    full_mix = stems['full_mix']

    # Standard features
    create_beat_grid(full_mix, save_grid=True)
    analyze_folder_bpm(folder)
    analyze_folder_essentia_features_optimized(folder)

    # Optional: Demucs
    demucs.separate_folder(folder)

    # Fast Music Flamingo (2 prompts only)
    genre = flamingo.analyze(full_mix, prompt_type='genre_mood')
    instruments = flamingo.analyze(full_mix, prompt_type='instrumentation')

    # Results auto-saved to .INFO with keys:
    # music_flamingo_genre_mood, music_flamingo_instrumentation
```

---

### Option 2: Comprehensive (All 5 Music Flamingo Prompts)
**177s per file (0.85x realtime) - use for archival/research**

```python
# After standard features + Demucs...

prompt_types = ['full', 'technical', 'genre_mood', 'instrumentation', 'structure']

for prompt_type in prompt_types:
    description = flamingo.analyze(full_mix, prompt_type=prompt_type)
    # Auto-saved to .INFO as music_flamingo_{prompt_type}
```

---

### Option 3: Minimal (Standard Features Only)
**19s per file (8x realtime) - fastest**

```python
# Just run standard features, skip Demucs and Flamingo
create_beat_grid(full_mix, save_grid=True)
analyze_folder_bpm(folder)
analyze_folder_essentia_features_optimized(folder)
```

---

## Expected Performance

| Configuration | Time/File | 10k Files (Sequential) | 10k Files (4x Parallel) |
|--------------|-----------|------------------------|-------------------------|
| **Option 1 (Fast)** | 43s | 119 hours | 30 hours |
| **Option 2 (Comprehensive)** | 177s | 492 hours | 123 hours |
| **Option 3 (Minimal)** | 19s | 53 hours | 13 hours |

---

## Files Generated Per Track

```
/output/
└── Track Name/
    ├── full_mix.flac           # Original audio
    ├── drums.flac              # Demucs stems (optional)
    ├── bass.flac
    ├── other.flac
    ├── vocals.flac
    ├── Track Name.BEATS_GRID   # Beat timestamps
    └── Track Name.INFO         # All metadata (JSON)
```

### INFO File Keys

**Standard features** (10 keys):
- `bpm`, `bpm_is_defined`, `beat_count`, `beat_regularity`
- `danceability`, `atonality`
- `content_enjoyment`, `content_usefulness`, `production_complexity`, `production_quality`

**Music Flamingo** (1-5 keys depending on prompts):
- `music_flamingo_full` (comprehensive description)
- `music_flamingo_technical` (tempo, key, chords, dynamics)
- `music_flamingo_genre_mood` (genre + emotional character)
- `music_flamingo_instrumentation` (instruments present)
- `music_flamingo_structure` (arrangement analysis)

---

## Test Single File

```bash
# Test everything on one file
python src/test_single_file.py "/path/to/organized/folder/full_mix.flac"

# Test with benchmark script
python src/benchmark_fp8_full.py "/path/to/organized/folder/full_mix.flac"
```

---

## Troubleshooting

### OOM Error
```bash
# Ensure environment variable is set:
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Or reduce Music Flamingo to 1-2 prompts instead of 5
```

### Slow Performance
```bash
# Make sure TunableOps is using cached results:
export PYTORCH_TUNABLEOP_TUNING=0  # 0 = use cache, 1 = retune
```

### Missing Dependencies
```bash
# Rebuild torchcodec if needed:
./install_torchcodec_rocm.sh
```

---

## Key Files

- **TunableOps**: `/home/kim/Projects/mir/tunableop_results00.csv`
- **Test script**: `src/test_single_file.py`
- **Benchmark**: `src/benchmark_fp8_full.py`
- **Documentation**: `FINAL_BENCHMARK_RESULTS_2026-01-18.md`

---

**Last Updated**: 2026-01-18
**Hardware**: AMD Radeon RX 9070 XT (16GB) + Ryzen 9 9900X
**Status**: Production Ready ✅
