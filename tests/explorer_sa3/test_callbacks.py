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


def test_scalar_options():
    from plots.explorer_sa3.callbacks import scalar_options
    opts = scalar_options()
    assert {o["value"] for o in opts} == {"bpm", "lufs", "rel_pos"}
    assert all("label" in o and "value" in o for o in opts)


def test_oned_feature_names_excludes_2d():
    import numpy as np
    from plots.explorer_sa3.callbacks import oned_feature_names
    ts = {"rms_energy_bass_ts": np.zeros(4096, np.float32),
          "hpcp_ts": np.zeros((4096, 12), np.float32),
          "beat_activation_ts": np.zeros(4096, np.float32)}
    assert set(oned_feature_names(ts)) == {"rms_energy_bass_ts", "beat_activation_ts"}
