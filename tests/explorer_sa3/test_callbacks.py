# tests/explorer_sa3/test_callbacks.py
from plots.explorer_sa3.callbacks import sample_ids
from plots.explorer_sa3.sidecar_index import CropMeta


def _idx(n):
    return [CropMeta(f"{i:06d}", "t", "a", "b", "p", 120.0, -9.0, 0.0)
            for i in range(n)]


def test_sample_ids_caps_and_is_deterministic():
    idx = _idx(1000)
    a = sample_ids(idx, 400, seed=0)
    b = sample_ids(idx, 400, seed=0)
    assert len(a) == 400 and a == b
    assert len(sample_ids(_idx(10), 400)) == 10  # fewer than cap → all
