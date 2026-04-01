
# External Patches (RDNA4/ROCm Optimization)

This document tracks local modifications applied to upstream repositories to optimize performance on AMD RDNA4 hardware with ROCm.
---

## Demucs

### Repository Info
*   **Upstream URL**: `https://github.com/adefossez/demucs`
*   **Package**: `demucs` (pip installed, no local clone)

### Applied Patches

### 1. Static Shape Wrapper
*   **Purpose**: Enforce fixed audio segment lengths to enable `torch.compile` and `MIGraphX` optimizations without constant recompilation.
*   **Status**: **IMPLEMENTED** (2026-01-25)
*   **File**: `demucs/static_wrapper.py` (New file)
*   **Changes**:
    - Created `StaticSegmentModel` wrapper class
    - Pads input to fixed `segment_length`, unpads output
    - Fixed `__getattr__` recursion issue that broke `torch.compile` by using `_model` prefix for internal attributes
    - Proxy model attributes (samplerate, sources, etc.) to underlying model

### 2. RDNA4 Attention Optimization
*   **Purpose**: Replace generic `scaled_dot_product_attention` with RDNA4-safe kernels (Triton generated) or optimized fallback.
*   **Status**: **IMPLEMENTED** via monkey-patch in `demucs_sep_optimized.py`
*   **File**: `demucs/transformer.py` (patched at runtime, not modified)
*   **Implementation**: `patch_demucs_attention()` in `src/preprocessing/demucs_sep_optimized.py` replaces attention with `F.scaled_dot_product_attention` which uses Flash Attention on ROCm

### 3. torch.compile Integration
*   **Purpose**: Use Inductor/Triton compilation for faster inference
*   **Status**: **EXPERIMENTAL - ISSUES ON ROCM**
*   **Findings** (2026-01-25):
    - `reduce-overhead` mode: FAILS with dtype mismatch in complex FFT ops
    - `default` mode: May work but needs more testing
    - `max-autotune` mode: Very long compile times, untested
*   **Root cause**: Demucs HTDemucs uses complex-valued FFT operations that don't compile cleanly with Inductor
*   **Recommendation**: Use SDPA patch only (no torch.compile) until PyTorch/ROCm improves complex op support

**Note**: All Demucs patches are applied at runtime via monkey-patching in `src/preprocessing/demucs_sep_optimized.py`. No local fork is needed.

---

## BS-RoFormer

### Repository Info
*   **Upstream URL**: `https://github.com/lucidrains/BS-RoFormer`
*   **Local Clone**: `/home/kim/Projects/mir/repos/BS-RoFormer`
*   **Test Script**: `tests/test_bs_roformer_optimized.py`

### Applied Patches

#### 1. AMD ROCm Flash Attention Support
*   **Purpose**: Enable Flash Attention for AMD ROCm GPUs (RDNA3/RDNA4) instead of falling back to slower math/mem_efficient attention
*   **Status**: **IMPLEMENTED** (2026-02-05)
*   **File**: `bs_roformer/attend.py` (lines 61-80)
*   **Changes**:
    - Added AMD ROCm detection via `torch.version.hip`
    - Enabled Flash Attention via PyTorch SDPA for AMD GPUs
    - Also enabled Flash Attention for all Ampere+ NVIDIA GPUs (SM 8.x+)
    - Original code only enabled Flash Attention for A100 (SM 8.0 exactly)

**Original code (lines 63-68):**
```python
if device_properties.major == 8 and device_properties.minor == 0:
    print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
    self.cuda_config = FlashAttentionConfig(True, False, False)
else:
    print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
    self.cuda_config = FlashAttentionConfig(False, True, True)
```

**Patched code:**
```python
is_amd_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None

if is_amd_rocm:
    print_once('AMD ROCm GPU detected, using flash attention via PyTorch SDPA')
    self.cuda_config = FlashAttentionConfig(True, False, False)
elif device_properties.major == 8 and device_properties.minor == 0:
    print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
    self.cuda_config = FlashAttentionConfig(True, False, False)
elif device_properties.major >= 8:
    print_once(f'NVIDIA GPU (SM {device_properties.major}.{device_properties.minor}) detected, using flash attention')
    self.cuda_config = FlashAttentionConfig(True, False, False)
else:
    print_once('Older GPU detected, using math or mem efficient attention if input tensor is on cuda')
    self.cuda_config = FlashAttentionConfig(False, True, True)
```

### Installation

To use the patched version:
```bash
cd /home/kim/Projects/mir/repos/BS-RoFormer
pip install -e .
```

Or add to Python path before importing:
```python
import sys
sys.path.insert(0, '/home/kim/Projects/mir/repos/BS-RoFormer')
from bs_roformer import BSRoformer
```

#### 2. FP16 Dtype Fallback
*   **Purpose**: Fall back to math attention for FP32 inputs (Flash Attention requires FP16/BF16)
*   **Status**: **IMPLEMENTED** (2026-02-05)
*   **File**: `bs_roformer/attend.py` (flash_attn function)
*   **Changes**:
    - Added dtype check in `flash_attn()` method
    - Falls back to `FlashAttentionConfig(False, True, True)` for FP32 inputs

```python
# Added in flash_attn() after selecting config:
if is_cuda and q.dtype not in (torch.float16, torch.bfloat16):
    config = FlashAttentionConfig(False, True, True)
```

### Performance Benchmarks (RX 9070 XT, ROCm 7.2)

| Mode | Speed | VRAM | Notes |
|------|-------|------|-------|
| **Batch=1** (Optimized) | **5.64x** | **3.09 GB** | 👑 Best efficiency |
| Batch=2 | 5.60x | 5.41 GB | Excellent speed |
| Batch=4 | 5.29x | 10.07 GB | High GPU utilization |
| Low VRAM Mode | 1.29x | 4.60 GB | Legacy (CPU bottlenecked) |

**Conclusion**: Regular batching (even batch_size=1) is significantly faster than the legacy "low vram" mode on modern GPUs because it avoids CPU-GPU pipeline stalls.

---

## timbral_models — Timbral_Hardness.py

### Repository Info
*   **Upstream URL**: `https://github.com/AudioCommons/timbral_models`
*   **Local Clone**: `/home/kim/Projects/mir/repos/timbral_models`
*   **File patched**: `timbral_models/Timbral_Hardness.py`
*   **Patch date**: 2026-03-30

### Applied Patches

#### 1. `hp_ratio=None` — skip expensive HPSS computation

**Purpose:** Allow the caller to supply a pre-computed harmonic-percussive ratio and
bypass `timbral_util.get_percussive_audio()`, which runs a full STFT, a 2-D median
filter (HPSS), and two inverse STFTs.

**Changes** — new `hp_ratio` parameter and branch in the HPSS block:

```python
# [MIR-PATCH] Accept pre-computed HP ratio to skip the expensive HPSS step.
if hp_ratio is not None:
    HP_ratio = float(hp_ratio)
else:
    HP_ratio = timbral_util.get_percussive_audio(audio_samples, return_ratio=True)
log_HP_ratio = np.log10(HP_ratio)
```

**Status:** Parameter wired up but not yet actively supplied by the pipeline. HPSS
currently runs in its own thread in parallel with other timbral features, so there is
no single pre-computed value available. Left as a hook for future caching/reuse.

---

#### 2. `precomputed_onsets=None` — skip duplicate mel-spectrogram

**Purpose:** Eliminate a redundant mel-spectrogram computation.

**Root cause:** The original code called `timbral_util.calculate_onsets()`, which
internally calls `librosa.onset.onset_detect()` and computes a mel-spectrogram.
Immediately after, `librosa.onset.onset_strength()` was called on the same audio,
computing a *second* identical mel-spectrogram. By passing pre-computed onset positions
we skip `calculate_onsets()` entirely while leaving the `onset_strength()` call intact
(it is needed for per-onset weighting later in the function).

**Zero-padding offset:** `timbral_hardness` zero-pads the audio by `nperseg + 1 = 4097`
samples before the onset/envelope analysis loop. Positions supplied via
`precomputed_onsets` are expected in the **original (un-padded) sample frame**; the
patch shifts each value by `nperseg + 1` internally so they align with the padded array.

**Changes** — new `precomputed_onsets` parameter and branch replacing the
`calculate_onsets()` call:

```python
# [MIR-PATCH] Accept pre-computed onset sample positions (in pre-padding audio
# space) to skip calculate_onsets(), which internally runs onset_detect() and
# duplicates the mel-spectrogram already computed by onset_strength() below.
# Positions are shifted here by the zero-pad offset (nperseg+1) to align with
# the padded audio_samples array used for envelope and bandwidth analysis.
if precomputed_onsets is not None and len(precomputed_onsets) > 0:
    pad_offset = nperseg + 1
    original_onsets = sorted(int(s) + pad_offset for s in precomputed_onsets)
else:
    original_onsets = timbral_util.calculate_onsets(audio_samples, envelope, fs, nperseg=nperseg)
onset_strength = librosa.onset.onset_strength(y=audio_samples, sr=fs)
```

**Pipeline integration:**

1. `src/pipeline.py` loads the track's `.ONSETS` file (absolute timestamps in seconds),
   filters to the crop's `[start_time, end_time]` window, and converts to sample
   offsets relative to the crop start:
   ```python
   _onsets_samples = [int((t - _start_t) * crop_sr) for t in _all_onsets[_mask]]
   ```
2. `analyze_all_timbral_features(..., onsets_samples=_onsets_samples)` in
   `src/timbral/audio_commons.py` forwards them to the hardness call via
   `_hardness_extra['precomputed_onsets']`.
3. If `.ONSETS` is missing or the crop window contains no onsets, `_onsets_samples`
   remains `None` and the original `calculate_onsets()` code path runs unchanged.

**Estimated savings:** ~0.15–0.25 s per crop on Ryzen 9 9900X (one mel-spectrogram
avoided). For a 200k-crop dataset this is roughly 8–14 CPU-hours.

### Backwards compatibility

Both parameters default to `None`. The function signature and all existing call sites
are fully compatible with no changes required.

