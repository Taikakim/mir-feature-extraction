"""A sentinel field and its mask must be sliced at the SAME rate.

Found 2026-09-09 backfilling f0 into AVP crop companions: 7 of 8 crops failed with
"operands could not be broadcast together with shapes (38045,) (38044,)".

_effective_rate deliberately DERIVES a field's rate from n_frames/duration rather than
trusting field_rates -- for good reasons documented in its own docstring. But the
SENTINEL_ZERO branch sliced the value array at that derived rate while slicing its voiced
mask at the DECLARED rate, so the two windows come out one or two frames apart whenever
the sidecar's stated rate is even slightly off. Real numbers: 51318 frames over 513.1608 s
is 100.0038 Hz derived against 100.0 declared -- a 0.004% disagreement, far too small for
any tolerance band to notice, and it makes the masked pooling unbroadcastable.

The mask is not a different measurement from the field it masks; it is the same
measurement's validity. It has to be resampled the same way.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/kim/Projects/mir")
from src.tools.crop_timeseries_resample import build_crop_timeseries  # noqa: E402

N_TRACK = 51318          # the real AVP shape that exposed this
DURATION = 513.1608390022676


def _arrays(seed=0):
    rng = np.random.default_rng(seed)
    voiced = (rng.random(N_TRACK) > 0.3).astype(np.float32)
    f0 = np.where(voiced > 0, rng.uniform(80, 800, N_TRACK), 0.0).astype(np.float32)
    return {"f0_other_ts": f0, "f0_other_voiced_ts": voiced}


def _meta():
    # stated 100.0 Hz; derived is 100.0038 -- inside every plausible tolerance band
    return {"duration": DURATION, "frame_rate": 100,
            "field_rates": {"f0_other_ts": 100.0, "f0_other_voiced_ts": 100.0}}


def test_sentinel_and_mask_slice_to_the_same_length():
    out = build_crop_timeseries(_arrays(), _meta(), 1.0, 381.4, 4096, strict=True)
    assert out["f0_other_ts"].shape == (4096,)
    assert out["f0_other_voiced_ts"].shape == (4096,)


def test_masked_pooling_ignores_the_unvoiced_sentinel():
    """The whole point of the sentinel branch: pooled pitch must not be dragged toward 0."""
    a = _arrays()
    out = build_crop_timeseries(a, _meta(), 1.0, 381.4, 4096, strict=True)
    pooled = out["f0_other_ts"]
    voiced_frames = pooled[out["f0_other_voiced_ts"] > 0]
    assert voiced_frames.min() > 60.0, "a voiced output frame fell below the input's floor"


if __name__ == "__main__":
    test_sentinel_and_mask_slice_to_the_same_length()
    test_masked_pooling_ignores_the_unvoiced_sentinel()
    print("PASS")
