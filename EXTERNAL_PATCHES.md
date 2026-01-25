
# External Patches for Demucs (RDNA4 Optimization)

This document tracks local modifications applied to the upstream Demucs repository to optimize performance on AMD RDNA4 hardware.

## Repository Info
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
