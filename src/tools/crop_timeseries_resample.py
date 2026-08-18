#!/usr/bin/env python3
"""Correct per-field slicing and resampling of whole-track timeseries into crop companions.

WHY THIS LIVES IN mir. The store's fields are mir's; how each one may legally be downsampled is
a property of the measurement, not of the consumer. The existing consumer
(`sa3_encode_from_manifest.py::build_crop_timeseries`) applies ONE rate and ONE pooling rule to
every field, which was correct when the store held only the 20 legacy 100 Hz envelopes and is
not correct now. Rather than let each consumer rediscover that, the rules live here.

THREE FAILURES IN THE ONE-RULE-FITS-ALL VERSION, all silent (measured 2026-08-12):

1. WRONG RATE. Expanded fields land at NATIVE rates (0.2-100 Hz). Slicing them at 100 Hz on a
   428 s track: a crop at 300-347 s asks dissonance_ts (10 Hz, 4277 frames) for [30000:4277] ->
   empty -> the caller returns None and DROPS THE WHOLE CROP; a crop at 10-57 s gets [1000:4277],
   which is in range but covers native seconds 100-428 -- the wrong region, silently.

2. SENTINEL FIELDS MEAN-POOLED. f0_*_ts use 0.0 for unvoiced. Mean pooling averages real pitches
   with zeros, pulling each window toward silence by its unvoiced fraction. Measured at the SA3
   grid: median error +0.00 st -- and p95 +15.86 st with 17.2% of frames wrong by more than a
   semitone. The median is zero because fully-voiced windows pool exactly, which is why a
   spot-check passes it; the damage is concentrated at note boundaries, where the melody is.

3. CATEGORICAL FIELDS MEAN-POOLED. chords_idx_ts indexes a 24-chord vocab (-1 = unknown).
   Averaging C (3) and G (10) gives 6.5 -- a different chord. Needs the mode.

A FOURTH THING THIS CANNOT FIX, so consumers should know it: upsampling a coarse field to the
latent grid produces an array whose LENGTH is not its information content. A 0.2 Hz field
(maest_embed_ts, dyncomplexity_ts) over a 47 s crop has ~9 real samples; interpolated to 4096
latent frames it looks like a dense timeseries and is not one. That is correct behaviour -- the
alternative is refusing to emit it -- but a head trained on it is learning from 9 numbers, not
4096, and anything reading "4096 frames" as "4096 observations" will overestimate what the field
supports. Check `field_rates` before believing a curve's resolution.

Usage (from the SA3 encoder, replacing the single-rate loop):

    from crop_timeseries_resample import build_crop_timeseries
    out = build_crop_timeseries(arrays, meta, start_sec, end_sec, n_frames)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# Fields whose 0.0 means ABSENT, not zero. Pool only the valid frames, and carry the pooled
# validity as a weight -- do not let the caller guess.
SENTINEL_ZERO = {"f0_other_ts": "f0_other_voiced_ts", "f0_bass_ts": "f0_bass_voiced_ts"}
# Fields that are class indices; averaging them is meaningless.
CATEGORICAL = {"chords_idx_ts"}


def _mean_pool(arr: np.ndarray, n: int) -> np.ndarray:
    """Fractional-bin mean pooling / linear interp -- correct for continuous envelopes."""
    arr = np.asarray(arr, dtype=np.float32)
    T = arr.shape[0]
    if T == n:
        return arr
    if T == 0:
        return np.zeros((n,) + arr.shape[1:], dtype=np.float32)
    if T > n:
        edges = np.linspace(0, T, n + 1)
        out = np.empty((n,) + arr.shape[1:], dtype=np.float32)
        for i in range(n):
            s = int(np.floor(edges[i]))
            e = max(s + 1, int(np.ceil(edges[i + 1])))
            out[i] = arr[s:e].mean(axis=0)
        return out
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, n)
    if arr.ndim == 1:
        return np.interp(dst, src, arr).astype(np.float32)
    return np.stack([np.interp(dst, src, arr[:, c]) for c in range(arr.shape[1])],
                    axis=1).astype(np.float32)


def _mode_pool(arr: np.ndarray, n: int) -> np.ndarray:
    """Most common class per output bin. Ties go to the first, which is arbitrary but stable."""
    arr = np.asarray(arr)
    T = arr.shape[0]
    if T == 0:
        return np.full(n, -1, dtype=np.float32)
    edges = np.linspace(0, T, n + 1)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = int(np.floor(edges[i]))
        e = max(s + 1, int(np.ceil(edges[i + 1])))
        vals, counts = np.unique(arr[s:e], return_counts=True)
        out[i] = vals[int(np.argmax(counts))]
    return out


_RATE_WARNED: set = set()


def _effective_rate(key: str, arr: np.ndarray, rates: dict, default: float,
                    duration: float) -> float:
    """The rate to actually slice with -- DERIVED, not trusted.

    `field_rates` can be wrong, and one entry demonstrably is. maest_embed_ts's rate is hardcoded
    in the producer as 16000/(313*256) = 0.19968 Hz (an assumed 5.008 s patch hop); the patches
    actually land ~10.1 s apart. Measured over 300 sidecars: stored/true = 2.020 median (p5 1.998,
    p95 2.045), i.e. the field covers ~49.5% of every track's duration and is off by exactly 2x.

    Consequences if trusted: time-indexed slicing maps the FIRST HALF of a track onto the whole
    of it, and any crop starting past the halfway point fails coverage outright. Non-time-indexed
    uses (rarity, retrieval, clip metrics -- what MAEST is actually for per MASTER §2) are
    unaffected, which is why this sat undetected until something sliced every field by time.

    n_frames / duration is the ground truth whenever duration is known, so prefer it and warn
    once per field when the sidecar disagrees by more than 5%. This makes the consumer robust to
    the whole class of wrong-rate metadata rather than to this one instance of it.
    """
    stated = float(rates.get(key, default))
    n = arr.shape[0]
    if duration <= 0 or n < 2:
        return stated
    derived = n / duration
    # ALWAYS return the derived rate when duration is known -- which is what the docstring above
    # always claimed this did. It did not: until 2026-08-18 the derived rate was used ONLY when
    # the sidecar disagreed by more than 5%, so anything wrong by LESS than 5% was silently
    # preferred over the ground truth. That is not hypothetical: va_deam_ts / va_emomusic_ts state
    # 1.04167 Hz (the producer used essentia's patchSize=96 where patchHopSize=93 was meant) and
    # measure 1.0753 Hz -- a 3.1% error, comfortably inside the band, so this function returned the
    # WRONG rate on every VGGish field of all 5035 sidecars in the store. ~19 s of drift at the
    # tail of a 600 s track, no warning.
    #
    # The lesson generalises past this instance: a TOLERANCE BAND IS NOT A CHECK. A guard that
    # only fires on large errors licenses every small one, and small-but-systematic is the harder
    # failure to find. The warning below now reports a disagreement worth investigating; it no
    # longer decides which number gets used.
    if stated > 0 and abs(stated - derived) / stated > 0.05 and key not in _RATE_WARNED:
        _RATE_WARNED.add(key)
        print(f"  [rate] {key}: sidecar says {stated:.5f} Hz, {n} frames over {duration:.1f}s "
              f"implies {derived:.5f} Hz ({stated/derived:.2f}x) -- using the derived rate",
              flush=True)
    return derived


def _slice(arr: np.ndarray, rate: float, start: float, end: float) -> Optional[np.ndarray]:
    """Slice [start, end) seconds using the FIELD'S OWN rate. None if it does not FULLY cover it.

    Found 2026-08-15 stress-testing this module before vendoring it for LUMI (Kim, relaying G's
    Stage-3 blocker report: "late crops dropped, early crops misaligned, no error raised" -- the
    exact failure class this whole module exists to kill). The ORIGINAL version only returned
    None on ZERO overlap (e<=s); a crop whose `end` ran past a field's actual data but still
    started before it (e.g. the last crop of a track, or any field whose true coverage falls
    short of the nominal track duration) got a SILENTLY TRUNCATED window instead -- which
    build_crop_timeseries then pools/resamples up to n_frames as if it were the full span,
    time-warping the tail into the whole output with no warning. `strict=True` claimed to guard
    against exactly this and didn't. A 1-frame rounding allowance keeps ordinary float/round
    jitter from spuriously tripping this on fully-covered crops.
    """
    total = arr.shape[0]
    s = max(0, int(round(start * rate)))
    e_want = int(round(end * rate))
    e = min(total, e_want)
    return None if (e <= s or e < e_want - 1) else arr[s:e]


def build_crop_timeseries(arrays: Dict[str, np.ndarray], meta: dict,
                          start: float, end: float, n_frames: int,
                          strict: bool = True) -> Dict[str, np.ndarray]:
    """Slice every field at its own rate and pool it with its own semantics.

    `strict` (default) RAISES on a field that cannot cover the crop window. The version this
    replaces returned None for the whole crop, which drops it silently -- and a silently missing
    crop is indistinguishable from one that was never requested. Fail loudly, or pass
    strict=False to skip the offending field and keep the rest (the skip is reported in the
    returned dict's absence, so the caller can still notice).
    """
    rates = (meta or {}).get("field_rates", {})
    default_rate = float((meta or {}).get("frame_rate", 100.0))
    duration = float((meta or {}).get("duration") or 0.0)
    out: Dict[str, np.ndarray] = {}

    for key, arr in arrays.items():
        if key == "__meta__":
            continue
        rate = _effective_rate(key, np.asarray(arr), rates, default_rate, duration)
        win = _slice(np.asarray(arr), rate, start, end)
        if win is None:
            if strict:
                raise ValueError(
                    f"field {key!r} (rate {rate} Hz, {np.asarray(arr).shape[0]} frames) does not "
                    f"cover crop [{start:.2f}, {end:.2f}]s -- refusing to emit a partial "
                    f"companion. This is the failure that used to drop the crop silently.")
            continue

        if key in CATEGORICAL:
            out[key] = _mode_pool(win, n_frames)
        elif key in SENTINEL_ZERO:
            mask_key = SENTINEL_ZERO[key]
            m = _slice(np.asarray(arrays[mask_key]), float(rates.get(mask_key, default_rate)),
                       start, end) if mask_key in arrays else (win > 0).astype(np.float32)
            m = np.asarray(m, dtype=np.float32)
            num = _mean_pool(win.astype(np.float32) * m, n_frames)
            den = _mean_pool(m, n_frames)
            out[key] = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0).astype(np.float32)
            out[mask_key] = den.astype(np.float32)      # pooled validity == loss weight
        elif key in SENTINEL_ZERO.values():
            continue                                    # emitted alongside its value field
        else:
            out[key] = _mean_pool(win, n_frames)
    return out
