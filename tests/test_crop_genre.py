"""Tests for crop_genre.py — TDD: write failing tests first, then implement."""
import json
import numpy as np
import pytest
from src.tools.crop_genre import genre_vector_from_probs


def test_genre_vector_is_a_simplex_with_other():
    labels = ["Goa", "Psy", "House", "Rock"]
    vocab = ["Goa", "Psy"]
    mean400 = np.array([0.30, 0.20, 0.10, 0.05], dtype=np.float32)  # sums <1, rest is 'other'
    v = genre_vector_from_probs(mean400, vocab, labels)
    assert set(v) == {"Goa", "Psy", "other"}
    assert abs(v["Goa"] - 0.30) < 1e-6 and abs(v["Psy"] - 0.20) < 1e-6
    assert abs(v["other"] - 0.50) < 1e-6            # 1 - (0.30+0.20)
    assert abs(sum(v.values()) - 1.0) < 1e-6        # simplex


def test_genre_vector_other_never_negative():
    labels = ["Goa", "Psy"]; vocab = ["Goa", "Psy"]
    v = genre_vector_from_probs(np.array([0.7, 0.7]), vocab, labels)
    assert v["other"] == 0.0                         # clamp, no negative


def test_additive_write_preserves_existing_keys(tmp_path):
    """Atomic additive write must not clobber pre-existing keys."""
    from src.core.json_handler import safe_update

    p = tmp_path / "000000.json"
    # Write initial content that must survive the merge
    p.write_text(json.dumps({"existing_key": "value", "bpm": 128}))

    vec = {"Electronic---Goa Trance": 0.3, "other": 0.7}
    safe_update(str(p), {"style_genre": vec})

    result = json.loads(p.read_text())
    assert result["existing_key"] == "value"   # preserved
    assert result["bpm"] == 128               # preserved
    assert result["style_genre"] == vec        # new field added
    assert len(result) == 3                    # no extra keys invented
