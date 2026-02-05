
# External Patches (RDNA4/ROCm Optimization)

This document tracks local modifications applied to upstream repositories to optimize performance on AMD RDNA4 hardware with ROCm.
---

## Demucs

### Repository Info
*   **Upstream URL**: `https://github.com/adefossez/demucs`
*   **Local Clone**: `/home/kim/Projects/repos/demucs`
*   **Management Script**: `src/scripts/manage_demucs_patches.py`

## Applied Patches

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

## Workflow

To apply patches to the local fork:
```bash
python src/scripts/manage_demucs_patches.py apply
```

To revert to upstream state:
```bash
python src/scripts/manage_demucs_patches.py revert
```

**Note**: The optimized execution script `src/preprocessing/demucs_sep_optimized.py` is configured to prioritize this local fork over the system-installed `demucs` package.

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
