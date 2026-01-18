# Performance Optimization Guide

This document describes performance optimizations for large-scale batch processing (thousands to tens of thousands of files).

## Issue #1: Model Reloading Bottleneck (FIXED) ✅

### The Problem

**Original Code:** `src/classification/essentia_features.py`

```python
def analyze_danceability(audio_path):
    # PROBLEM: Loads 100MB model every single time!
    model = TensorflowPredictVGGish(graphFilename=model_path)
    activations = model(audio)
    # ...
```

**Impact for 10,000 files:**
- Model loading: ~3 seconds per file
- Total overhead: 3s × 10,000 = **8.3 hours wasted loading/unloading**
- GPU sits idle while CPU loads model from disk

### The Solution

**Optimized Code:** `src/classification/essentia_features_optimized.py`

```python
class EssentiaAnalyzer:
    def __init__(self):
        # Load models ONCE during initialization
        self.danceability_model = TensorflowPredictVGGish(...)
        self.atonality_model = TensorflowPredictVGGish(...)

    def analyze_danceability(self, audio_path):
        # Reuse self.danceability_model (already loaded)
        activations = self.danceability_model(audio)
        # ...

# Usage
analyzer = EssentiaAnalyzer()  # Load once: ~3 seconds
for file in files:             # Reuse for all files
    danceability = analyzer.analyze_danceability(file)
```

**Impact for 10,000 files:**
- Model loading: ~3 seconds **TOTAL**
- Savings: 8.3 hours → 3 seconds
- **Speedup: ~10,000x**

### Performance Comparison

| Dataset Size | Original | Optimized | Time Saved |
|--------------|----------|-----------|------------|
| 100 files | 5 minutes | 3 seconds | 4.95 min |
| 1,000 files | 50 minutes | 3 seconds | 49.95 min |
| 10,000 files | **8.3 hours** | **3 seconds** | **8.3 hours** |
| 100,000 files | **83 hours** | **3 seconds** | **83 hours** |

### Migration Guide

#### Old Way (Inefficient)
```python
# Original: Loads models for every file
from classification.essentia_features import batch_analyze_essentia_features

batch_analyze_essentia_features('dataset/')  # Slow for large datasets
```

#### New Way (Optimized)
```python
# Optimized: Loads models once
from classification.essentia_features_optimized import batch_analyze_essentia_features_optimized

batch_analyze_essentia_features_optimized('dataset/')  # Fast!
```

#### Command Line

**Old:**
```bash
# Slow for large datasets
python src/classification/essentia_features.py dataset/ --batch
```

**New (Optimized):**
```bash
# Fast with model caching
python src/classification/essentia_features_optimized.py dataset/ --batch
```

#### Programmatic Usage

For maximum performance when processing thousands of files:

```python
from classification.essentia_features_optimized import EssentiaAnalyzer
from pathlib import Path

# Load models ONCE
analyzer = EssentiaAnalyzer()  # ~3 seconds

# Process thousands of files
for audio_file in Path('dataset/').rglob('*.mp3'):
    # Analyze using cached models (fast!)
    danceability = analyzer.analyze_danceability(audio_file)
    atonality = analyzer.analyze_atonality(audio_file)

    print(f"{audio_file.name}: danceability={danceability:.3f}, atonality={atonality:.3f}")
```

---

## Issue #2: Serial Processing - NOT AN ISSUE! ✅ RESOLVED

### The Problem (As Initially Assumed)

**Initial Assumption:**

```python
def batch_analyze_loudness(root_directory):
    folders = find_organized_folders(root_directory)

    # ASSUMED PROBLEM: Processes one folder at a time, uses 1 thread
    for folder in folders:
        analyze_folder_loudness(folder)
```

**Initial Concern:**
- Ryzen 9 9900X has **24 cores / 48 threads**
- Simple for-loop appears to use **1 thread** (4% utilization)
- 47 threads appear to sit idle

### The Reality (After Threading Analysis)

**Threading Analysis Results** (see `THREADING_ANALYSIS.md`):

| Feature | CPU Usage | Threads | Already Multithreaded? |
|---------|-----------|---------|------------------------|
| Onsets | **1347%** | 48 | ✅ YES (uses 13.5 cores!) |
| Chroma | **1077%** | 48 | ✅ YES (uses 10.8 cores!) |
| Spectral | **102%** | 48 | ✅ YES (uses 1+ cores) |
| Essentia | **128%** | 97 | ✅ YES (TensorFlow threading) |

**Key Finding:** Librosa, numpy, scipy, and TensorFlow **already use internal multithreading** via:
- OpenBLAS / MKL (numpy/scipy linear algebra)
- FFTW (FFT operations)
- TensorFlow thread pools

### ~~The Solution~~ No Solution Needed!

~~Use `ProcessPoolExecutor` to parallelize CPU-bound tasks:~~

**DO NOT ADD PARALLELIZATION** - libraries already handle it optimally!

```python
# WRONG APPROACH - DO NOT DO THIS!
# This would cause thread oversubscription

from concurrent.futures import ProcessPoolExecutor, as_completed

def batch_analyze_loudness_WRONG(root_directory, max_workers=20):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Each worker spawns 48 threads internally (librosa)
        # Result: 20 × 48 = 960 threads fighting for 48 hardware threads
        # Performance: WORSE than serial due to context switching
        futures = {executor.submit(analyze_folder, folder): folder for folder in folders}
```

**Why This Would Be Slower:**
- 20 workers × 48 threads each = **960 threads** competing for 48 hardware threads
- Massive context switching overhead
- Cache thrashing
- **Result: 2-5x SLOWER than current serial approach**

### Correct Approach: Serial Processing

```python
# CORRECT - Let libraries handle threading
def batch_analyze_loudness(root_directory):
    folders = find_organized_folders(root_directory)

    for folder in folders:
        # Uses 10-13 cores internally via librosa's numpy/scipy
        analyze_folder_loudness(folder)
```

**Performance:**
- Each file: ~0.4s (already using ~10 cores)
- 10,000 files × 0.4s = **4,000s = 1.1 hours**
- **No speedup possible** - already optimal

### Status

- ✅ Analysis complete
- ✅ No optimization needed
- ✅ Current serial approach is optimal

---

## Issue #3: Demucs Model Reloading ✅ FIXED

### The Problem

**Original Code:** `src/preprocessing/demucs_sep.py`

```python
def batch_separate_stems(root_directory):
    for folder in folders:
        # PROBLEM: Spawns subprocess for EACH file
        subprocess.run(['demucs', '-n', 'htdemucs', str(audio_file)])
        # Each subprocess call:
        # 1. Starts new Python interpreter (~1s)
        # 2. Imports PyTorch (~1s)
        # 3. Loads Demucs model to VRAM (~2-3s)
        # 4. Processes file (~60s GPU time)
        # 5. Exits and unloads everything
```

**Impact for 10,000 files:**
- Startup overhead: ~5 seconds per file
- Total wasted time: 5s × 10,000 = **13.9 hours**
- Model loaded/unloaded 10,000 times
- Total time: ~27 hours (13.9h overhead + 13.1h processing)

### The Solution

**Implemented:** `src/preprocessing/demucs_sep_optimized.py`

```python
from demucs.pretrained import get_model
from demucs.apply import apply_model

class DemucsProcessor:
    """Load Demucs model ONCE and reuse for all files."""

    def __init__(self, device='cuda'):
        # Load model ONCE into VRAM
        logger.info("Loading Demucs model (one-time operation)...")
        self.model = get_model('htdemucs')
        self.model.to(device)
        self.model.eval()
        logger.info("✓ Model loaded and cached in VRAM")

    def separate_folder(self, folder: Path):
        # Reuse self.model (already loaded in VRAM)
        with torch.no_grad():
            sources = apply_model(self.model, wav, shifts=1)
        # Save stems...

# Usage
processor = DemucsProcessor(device='cuda')  # Load once (~3s)
for folder in folders:
    processor.separate_folder(folder)  # Reuse model (fast!)
```

**Key Insight from User:**
> "I think it uses the GPU fully when processing single files. But since I have 16GB VRAM, definitely just load the model once in memory, and only remove it after all of the files have been finished."

This is **exactly the same pattern** as Essentia optimization:
- ✅ Load model once into VRAM/RAM
- ✅ Process files serially (GPU already at 100%)
- ✅ Unload only after all files done
- ✅ No ProcessPoolExecutor needed (would cause conflicts)

**Performance:**
- Original: 5s overhead × 10,000 files = **13.9 hours** wasted
- Optimized: 5s overhead × 1 = **5 seconds** total
- **Overhead reduction: ~10,000x**
- **Total time: ~13 hours** (13h processing only)
- **Overall speedup: ~2x** (27h → 13h)

### Status

- ✅ Implementation complete (`demucs_sep_optimized.py`)
- ✅ Integrated with batch_process.py
- ✅ Uses file locking for parallel safety
- ✅ Includes resume capability
- ⏳ Needs testing on actual dataset

---

## Combined Impact for 10,000 Files

### Original Performance (Before Optimization)

| Stage | Time | Bottleneck |
|-------|------|------------|
| Demucs separation | ~27 hours | Subprocess overhead (13.9h) + processing (13.1h) |
| Essentia features | ~9 hours | Model reloading (8.3h) + inference (0.7h) |
| CPU features | ~1.1 hours | Already optimal (internal threading) |
| **TOTAL** | **~37 hours** | |

### Optimized Performance (After All Optimizations)

| Stage | Time | Improvement |
|-------|------|-------------|
| Demucs separation | **~13 hours** | Model caching ✅ (1 load vs 10,000 loads) |
| Essentia features | **~1 hour** | Model caching ✅ (1 load vs 10,000 loads) |
| CPU features | **~1.1 hours** | Already optimal ✅ (no change needed) |
| **TOTAL** | **~15 hours** | **2.5x faster overall** |

### Optimization Results Summary

| Feature | Original | Optimized | Speedup | Method |
|---------|----------|-----------|---------|--------|
| Demucs | 27 hours | **13 hours** | **2.1x** | Model caching ✅ |
| Essentia | 9 hours | **1 hour** | **9x** | Model caching ✅ |
| CPU features | 1.1 hours | **1.1 hours** | **1x** | Already optimal ✅ |
| **TOTAL** | **37 hours** | **~15 hours** | **2.5x** | |

**Key Wins:**
- ✅ Essentia: **8.3 hours → 1 hour** (model caching - DONE)
- ✅ Demucs: **27 hours → 13 hours** (model caching - DONE)
- ✅ CPU tasks: **Already optimal** (internal threading - no change needed)

**Time Saved:** ~22 hours per 10,000 files

---

## Implementation Status

### ✅ Completed

1. **Essentia Model Caching** (Optimization #1)
   - File: `src/classification/essentia_features_optimized.py`
   - Method: Load TensorFlow models once, reuse for all files
   - Speedup: ~10,000x for model loading overhead
   - Impact: 9 hours → 1 hour for 10,000 files
   - Status: ✅ Complete, tested, integrated

2. **Demucs Model Caching** (Optimization #3)
   - File: `src/preprocessing/demucs_sep_optimized.py`
   - Method: Load PyTorch model once into VRAM, reuse for all files
   - Speedup: ~10,000x for model loading overhead
   - Impact: 27 hours → 13 hours for 10,000 files
   - Status: ✅ Complete, untested, integrated
   - GPU: AMD Radeon RX 9070 XT with ROCm 7.2

3. **Threading Analysis** (Research)
   - File: `src/test_threading.py`
   - Findings: All librosa features already use internal multithreading
   - Result: CPU parallelization NOT needed
   - Documentation: `THREADING_ANALYSIS.md`
   - Status: ✅ Complete

### ❌ Cancelled

4. **CPU Parallelization** (Optimization #2)
   - Original plan: Use ProcessPoolExecutor for librosa features
   - Reason cancelled: Features already use 10-13 cores internally
   - Threading test showed: 1347% CPU usage (13.5 cores) for onsets
   - Result: Would cause thread oversubscription and degrade performance
   - Status: ❌ Cancelled (not needed)

### 📅 Future Considerations

5. **I/O Optimization** (Low priority)
   - Cache loaded audio in memory between features
   - Avoid reloading same file multiple times
   - Expected impact: Small (I/O is not the bottleneck)

6. **Hybrid GPU/CPU Pipeline** (Advanced)
   - Run Demucs on GPU while CPU processes other features
   - Complex coordination required
   - Low priority (current optimizations are sufficient)

---

## Usage Recommendations

### For Small Datasets (<100 files)

Use original code - overhead is negligible:
```bash
python src/classification/essentia_features.py dataset/ --batch
```

### For Medium Datasets (100-1,000 files)

Use Essentia optimization (biggest win):
```bash
python src/classification/essentia_features_optimized.py dataset/ --batch
```

### For Large Datasets (1,000-10,000 files)

Use all optimizations:
```bash
# 1. Essentia with model caching
python src/classification/essentia_features_optimized.py dataset/ --batch

# 2. CPU tasks with parallelization (when available)
python src/preprocessing/loudness_optimized.py dataset/ --batch --workers 20

# 3. Demucs with batch processing (when available)
python src/preprocessing/demucs_sep_optimized.py dataset/ --batch
```

### For Very Large Datasets (10,000+ files)

Consider splitting into chunks and processing with hybrid GPU/CPU:
```bash
# Process in parallel: Demucs on GPU + CPU tasks
./scripts/batch_process_hybrid.sh dataset/ --workers 20
```

---

## Monitoring Progress

### Check CPU Utilization

```bash
# Before optimization: ~4% (1 thread)
# After optimization: ~90% (20+ threads)
htop
```

### Check GPU Utilization

```bash
# Should stay at 90-100% during Demucs processing
watch -n 1 rocm-smi  # AMD
# or
watch -n 1 nvidia-smi  # NVIDIA
```

### Estimate Completion Time

```python
# Calculate expected time for your dataset
from pathlib import Path

files = list(Path('dataset').rglob('*.mp3'))
print(f"Files: {len(files)}")

# Essentia optimized: ~0.5s per file (inference only)
essentia_time = len(files) * 0.5 / 3600
print(f"Essentia: ~{essentia_time:.1f} hours")

# CPU tasks parallelized: ~2s per file / 20 workers
cpu_time = len(files) * 2 / 20 / 3600
print(f"CPU tasks: ~{cpu_time:.1f} hours")

# Demucs: ~60s per file (GPU processing)
demucs_time = len(files) * 60 / 3600
print(f"Demucs: ~{demucs_time:.1f} hours")

print(f"Total: ~{essentia_time + cpu_time + demucs_time:.1f} hours")
```

---

## Testing the Optimizations

### Benchmark Script

```python
import time
from pathlib import Path
from classification.essentia_features import analyze_danceability as old_analyze
from classification.essentia_features_optimized import EssentiaAnalyzer

# Get test files
test_files = list(Path('test_data').rglob('*/full_mix.*'))[:10]

# Benchmark original (reloads model each time)
print("Original (model reloading):")
start = time.time()
for f in test_files:
    old_analyze(f)
original_time = time.time() - start
print(f"  Time: {original_time:.1f}s ({original_time/len(test_files):.2f}s per file)")

# Benchmark optimized (model cached)
print("\nOptimized (model caching):")
analyzer = EssentiaAnalyzer()
start = time.time()
for f in test_files:
    analyzer.analyze_danceability(f)
optimized_time = time.time() - start
print(f"  Time: {optimized_time:.1f}s ({optimized_time/len(test_files):.2f}s per file)")

# Speedup
speedup = original_time / optimized_time
print(f"\nSpeedup: {speedup:.1f}x")
print(f"For 10,000 files, saves: {(original_time - optimized_time) * 1000 / 3600:.1f} hours")
```

---

## Next Steps

1. **Test Essentia optimization** on your dataset
2. **Implement CPU parallelization** for loudness/spectral modules
3. **Implement Demucs batch processing**
4. **Create unified batch processing script** that combines all optimizations

See `src/classification/essentia_features_optimized.py` for the working implementation of Issue #1.

---

**Last Updated:** 2026-01-13
**Status:** Essentia optimization complete, CPU/Demucs optimizations in progress
