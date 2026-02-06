# BS-RoFormer Optimization Walkthrough
**Objective:** Optimize BS-RoFormer inference speed on AMD Radeon RX 9070 XT (ROCm 7.2)

## Key Results
| Benchmark | Speed | VRAM | Status |
|-----------|-------|------|--------|
| Baseline (Start) | **1.22x** realtime | 4.6 GB | Working but slow |
| Vectorized Overlap-Add | **1.29x** realtime | 4.6 GB | Minor CPU improvement |
| Batch Size = 2 | **5.60x** realtime | 5.41 GB | **4.6x Faster!** |
| Batch Size = 4 | **5.29x** realtime | 10.07 GB | Higher memory usage |
| Batch Size = 1 (Optimized) | **5.64x** realtime | **3.09 GB** | **Best Efficiency** |
| Custom Model (small) | **22.01x** realtime | **1.43 GB** | **Blazing Fast** |

> **Note:** The "Custom Model (small)" result used a dimension-384 model (`SYH99999`) compared to the standard dimension-512 model (`bs_roformer_viperx_1297`), explaining the massive speed delta.

## Optimizations Applied

### 1. Vectorized Overlap-Add
Replaced nested Python loops with numpy vectorization to remove CPU bottlenecks during signal reconstruction.
- **Before:** Nested `for` loops (slow)
- **After:** Vectorized numpy operations + broadcasting

### 2. Pinned Memory & Non-Blocking Transfers
Added pinned memory buffers to overlap CPU-GPU data transfers with computation.

### 3. "Low VRAM" (Legacy) vs. Regular Batching
Discovered that the legacy mode (formerly "fast mode") is **obsolete** on modern hardware. It forces CPU-GPU synchronization after every chunk, causing pipeline stalls.
- **Action:** Marked `--low-vram` as deprecated/legacy.
- **Result:** Regular inference path (batch_size=1) is **4.4x faster** AND uses **less VRAM** (3.09 GB vs 4.60 GB).

## Recommended Usage
For AMD RX 9070 XT (16GB VRAM):

```bash
# Best efficiency (5.64x speed, 3GB VRAM)
python test_bs_roformer_optimized.py input.flac --batch-size 1

# Alternative (similar speed, 5.4GB VRAM)
python test_bs_roformer_optimized.py input.flac --batch-size 2
```

## Notes
- `torch.compile()` is **NOT** recommended yet on RDNA4 due to Triton backend bugs.
- Auto-cast FP16 should be enabled (default).
- Flash Attention (SDPA) is working correctly on ROCm 7.2.

## Argument Availability & Custom Models
Both `test_bs_roformer.py` (standard) and `test_bs_roformer_optimized.py` (fast) now share the same command-line arguments and model discovery logic.

### Unified Arguments
- `--model-name`: Name of the model folder (e.g., `SYH99999-bs_roformer_4stems_ft`).
- `--model-dir`: Directory containing model folders (default: `.../models/bs-roformer`).
- `--model-path`: Direct path to a specific model checkpoint (optional override).
- `--batch-size`: Batch size for inference.

### Custom Model Support
Implemented a `CustomSeparator` class in `test_bs_roformer.py` to bypass `audio-separator`'s strict validation, allowing:
- **Local Model Loading:** Directly load custom checkpoints (e.g., fine-tunes) not present in the official package database.
- **Config Auto-Matching:** Automatically pairs checkpoints with `config.yaml` in the same directory.
- **Roformer Identification:** Correctly identifies models as MDXC architecture for Roformer inference.

## Debugging & Critical Fixes

### "Residuals of separate stems" Issue
**Problem:** The optimized script initially produced output that sounded like "residuals of the full mix" or phase-canceled garbage.

**Root Cause:**
1. **Normalization Mismatch:** BS-Roformer models (especially from UVR) are trained on audio normalized to **0.9 max peak**. Feeding unnormalized audio (often <0.5 peak) causes the mask estimation to fail.
2. **Implementation Divergence:** The generic `BS-RoFormer` package differs slightly from the specialized implementation bundled in `audio-separator` (regarding `stft_normalized` handling).

**Fixes Applied:**
- **Input Normalization:** Added `normalize_audio` to scale input to 0.9 peak before inference.
- **Correct Import:** Switched to importing `BSRoformer` directly from `audio_separator.separator.uvr_lib_v5.roformer.bs_roformer`.

> **Performance Note:** The optimized script achieved **22x realtime** speed with a custom small model (`SYH99999`, dim 384) on the RX 9070 XT, utilizing only 1.43 GB VRAM!

## Environment Maintenance

### Repositories vs. Libraries
- **Source Repos:** You can safely delete the cloned `BS-RoFormer` and `python-audio-separator` folders from your project root if you wish. The scripts do not use them.
- **Installed Libraries:** The scripts run entirely using the packages installed in your `mir` environment (`site-packages`).

### Critical Warning: Do Not Upgrade Blindly
The `audio-separator` library in your environment has been **patched** to expose advanced parameters (like `mlp_expansion_factor`) required for the optimized script.
- **Do NOT** run `pip install --upgrade audio-separator` without verifying if the new version officially supports these parameters.
- If you reinstall the library, you may lose the optimizations and the script will fail.
