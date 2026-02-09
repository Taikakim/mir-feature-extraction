"""
Central ROCm Environment Setup

Single source of truth for AMD ROCm GPU optimization environment variables.
Must be called BEFORE importing torch or any module that imports torch.

All values here match config/master_pipeline.yaml rocm: section.
Shell-level exports take precedence (setdefault never overwrites).

Usage (at the top of every GPU-using entry point):

    from core.rocm_env import setup_rocm_env
    setup_rocm_env()
    import torch  # Now safe
"""

import atexit
import os
import sys

_exit_handler_registered = False


def _rocm_clean_exit():
    """Force-exit to bypass ROCm 7.2 roctracer assertion crash at shutdown.

    The ROCm 7.2 HSA runtime has a bug in hsa_support::Finalize() that
    triggers an assertion failure during normal process exit when GPU was used.
    This atexit handler flushes outputs and calls os._exit(0) to skip the
    broken C++ destructor chain.

    TunableOp results are flushed manually before os._exit() since the
    normal C++ destructor that writes them would be skipped.
    """
    try:
        import torch
        if torch.cuda.tunable.is_enabled():
            torch.cuda.tunable.write_file()
    except Exception:
        pass
    try:
        # Reset terminal state (fixes "keyboard not working" after run)
        os.system('stty sane')
    except Exception:
        pass
    
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def setup_rocm_env(*, tuning: bool = False):
    """Set ROCm optimization environment variables.

    Must be called before any ``import torch`` statement.

    Args:
        tuning: If True, enable TunableOp kernel tuning (generates new results).
                Default False uses pre-tuned kernels only.
    """
    global _exit_handler_registered

    defaults = {
        # --- Flash Attention 2 (Triton backend for AMD) ---
        'FLASH_ATTENTION_TRITON_AMD_ENABLE': 'TRUE',

        # --- TunableOp: pre-tuned GEMM kernels for RDNA4 ---
        'PYTORCH_TUNABLEOP_ENABLED': '1',
        'PYTORCH_TUNABLEOP_TUNING': '1' if tuning else '0',
        'PYTORCH_TUNABLEOP_VERBOSE': '1',

        # --- Memory management ---
        # ROCm 7.2 unified to PYTORCH_ALLOC_CONF (PYTORCH_HIP_ALLOC_CONF is deprecated)
        'PYTORCH_ALLOC_CONF': 'garbage_collection_threshold:0.8,max_split_size_mb:512',

        # --- Prevent CPU/GPU sync bug on AMD ---
        'HIP_FORCE_DEV_KERNARG': '1',

        # --- torch.compile is buggy with Flash Attention on RDNA ---
        'TORCH_COMPILE': '0',

        # --- Suppress roctracer assertion crash during process exit (ROCm 7.2 bug) ---
        'HSA_TOOLS_LIB': '',

        # --- CPU thread count ---
        'OMP_NUM_THREADS': '8',
    }

    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    # Register atexit handler to bypass ROCm 7.2 roctracer crash at shutdown
    if not _exit_handler_registered:
        atexit.register(_rocm_clean_exit)
        _exit_handler_registered = True
