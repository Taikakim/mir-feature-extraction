# tests/explorer_sa3/test_mix.py
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SERVER = Path(__file__).resolve().parents[2] / "scripts" / "latent_server_sa3.py"


def _load():
    spec = importlib.util.spec_from_file_location("latent_server_sa3", SERVER)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"player deps unavailable: {e}")
    return mod


def test_interp_np_lerp_midpoint_and_align():
    mod = _load()
    a = np.zeros((256, 10), np.float32)
    b = np.ones((256, 8), np.float32)
    out = mod._interp_np(a, b, 0.5, "lerp")
    assert out.shape == (256, 8)            # aligned to min T
    assert np.allclose(out, 0.5)
