import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from plots.build_dataset_stats import (
    _strip_crop_suffix,
    _interp32,
    _rotate_hpcp,
    _dominant_tonic,
    _cosine_top_k,
)


def test_strip_crop_suffix_basic():
    assert _strip_crop_suffix("Artist - Title_0") == "Artist - Title"
    assert _strip_crop_suffix("Artist - Title_12") == "Artist - Title"
    assert _strip_crop_suffix("No suffix") == "No suffix"
    assert _strip_crop_suffix("Ends_in_word_0") == "Ends_in_word"


def test_strip_crop_suffix_only_trailing():
    # underscore in the middle of name is preserved
    assert _strip_crop_suffix("A_B_C_3") == "A_B_C"


def test_interp32_length():
    arr = np.random.rand(256).astype(np.float32)
    out = _interp32(arr)
    assert out.shape == (32,)
    assert out.dtype == np.float32


def test_interp32_passthrough():
    arr = np.arange(32, dtype=np.float32)
    out = _interp32(arr)
    np.testing.assert_allclose(out, arr, atol=1e-5)


def test_interp32_boundary_values():
    arr = np.zeros(128, dtype=np.float32)
    arr[0] = 1.0
    arr[-1] = 2.0
    out = _interp32(arr)
    assert abs(out[0] - 1.0) < 1e-5
    assert abs(out[-1] - 2.0) < 1e-5


def test_rotate_hpcp_by_zero():
    hpcp = np.arange(12, dtype=np.float32)
    out = _rotate_hpcp(hpcp, 0)
    np.testing.assert_array_equal(out, hpcp)


def test_rotate_hpcp_by_3():
    hpcp = np.arange(12, dtype=np.float32)
    out = _rotate_hpcp(hpcp, 3)
    # roll by -3: index 0 becomes what was at index 3
    assert out[0] == pytest.approx(3.0)
    assert out[1] == pytest.approx(4.0)
    assert out[11] == pytest.approx(2.0)


def test_rotate_hpcp_wraps():
    hpcp = np.zeros(12, dtype=np.float32)
    hpcp[11] = 1.0
    out = _rotate_hpcp(hpcp, 11)
    assert out[0] == pytest.approx(1.0)


def test_dominant_tonic_clear():
    # tonic 5 appears most often
    arr = np.array([5.0, 5.1, 4.9, 5.0, 5.2, 2.0], dtype=np.float32)
    assert _dominant_tonic(arr) == 5


def test_dominant_tonic_wraps_mod12():
    arr = np.full(10, 13.0, dtype=np.float32)  # 13 % 12 == 1
    assert _dominant_tonic(arr) == 1


def test_cosine_top_k_excludes_self():
    emb = np.array([[1., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
    names = ["A", "B", "C"]
    result = _cosine_top_k(emb, names, k=2)
    assert "A" not in [n for n, _ in result["A"]]


def test_cosine_top_k_most_similar_first():
    emb = np.array([[1., 0.], [0.99, 0.01], [0., 1.]], dtype=np.float32)
    names = ["A", "B", "C"]
    result = _cosine_top_k(emb, names, k=2)
    assert result["A"][0][0] == "B"
    scores = [s for _, s in result["A"]]
    assert scores == sorted(scores, reverse=True)


def test_cosine_top_k_respects_k():
    emb = np.random.rand(10, 4).astype(np.float32)
    names = [str(i) for i in range(10)]
    result = _cosine_top_k(emb, names, k=3)
    assert all(len(v) == 3 for v in result.values())
