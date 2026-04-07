"""Tests for plots/explorer/data.py helper functions."""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plots.explorer.data import (
    norm01, corrcoef, parse_class_label, parse_dim_range,
    blend_latents_by_cluster,
)

def test_norm01_maps_min_to_zero_max_to_one():
    arr = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = norm01(arr)
    assert abs(r[0]) < 1e-9
    assert abs(r[-1] - 1.0) < 1e-9

def test_norm01_constant_array_returns_half():
    r = norm01([5.0, 5.0, 5.0])
    assert all(abs(v - 0.5) < 1e-9 for v in r)

def test_corrcoef_perfect_positive():
    x = [1.0, 2.0, 3.0, 4.0]
    assert abs(corrcoef(x, x) - 1.0) < 1e-9

def test_corrcoef_perfect_negative():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [4.0, 3.0, 2.0, 1.0]
    assert abs(corrcoef(x, y) - (-1.0)) < 1e-9

def test_parse_class_label_dict_picks_highest():
    assert parse_class_label("{'Goa Trance': 0.9, 'Psy-Trance': 0.3}") == "Goa Trance"

def test_parse_class_label_list_picks_first():
    assert parse_class_label("['psytrance', 'trance']") == "psytrance"

def test_parse_class_label_plain_string():
    assert parse_class_label("Some Label") == "Some Label"

def test_parse_class_label_empty():
    assert parse_class_label("{}") == ""
    assert parse_class_label("[]") == ""
    assert parse_class_label("") == ""

def test_parse_dim_range_simple():
    mask = parse_dim_range("0-3", n_dims=8)
    assert list(mask) == [True, True, True, True, False, False, False, False]

def test_parse_dim_range_single():
    mask = parse_dim_range("5", n_dims=8)
    assert list(mask) == [False, False, False, False, False, True, False, False]

def test_parse_dim_range_mixed():
    mask = parse_dim_range("0-1,5,7", n_dims=8)
    assert list(mask) == [True, True, False, False, False, True, False, True]

def test_blend_latents_holds_unassigned_at_a():
    # cluster 1 covers dims 0-1; dim 2 is unassigned (cluster 0)
    cluster_labels = np.array([1, 1, 0])  # 3 dims
    z_a = np.ones((3, 4))
    z_b = np.zeros((3, 4))
    result = blend_latents_by_cluster(z_a, z_b, {1: 1.0}, cluster_labels)
    # dims 0,1: fully blended to B (0.0)
    assert np.allclose(result[:2], 0.0)
    # dim 2: unassigned, stays at A (1.0)
    assert np.allclose(result[2], 1.0)
