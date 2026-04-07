"""Tests for plots/explorer/latch.py LatCH inference hook."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.explorer.latch import apply_latch_guidance

def test_fallback_returns_same_shape():
    """With no checkpoint, correlation fallback should return same shape."""
    z = np.zeros((64, 256), dtype=np.float32)
    result = apply_latch_guidance(z, "brightness", strength=1.0,
                                  latch_dir=Path("/nonexistent"))
    assert result.shape == (64, 256)

def test_returns_same_shape_unknown_feature():
    z = np.random.randn(64, 256).astype(np.float32)
    result = apply_latch_guidance(z, "unknown_feature", strength=0.5,
                                  latch_dir=Path("/nonexistent"))
    assert result.shape == z.shape

def test_zero_strength_returns_copy():
    z = np.random.randn(64, 256).astype(np.float32)
    result = apply_latch_guidance(z, "brightness", strength=0.0)
    assert np.allclose(result, z)
