#!/usr/bin/env python3
"""
Build all dataset statistics for Feature Explorer.

Replaces generate_explorer_data.py.  Three stages:
  1. Scalar pass   — scan .INFO files, average per track
  2. Timeseries pass — query TimeseriesDB, compute shape vectors + mini curves
  3. Similarity pass — cosine similarity, emit TS_NEIGHBORS

Usage:
    python plots/build_dataset_stats.py --source /path/to/crops
    python plots/build_dataset_stats.py --source /path/to/crops --skip-timeseries
    python plots/build_dataset_stats.py --source /path/to/crops --skip-scalars --skip-curves
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.spatial.distance

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PLOTS_DIR   = Path(__file__).resolve().parent
REPO_ROOT   = PLOTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_SRC    = Path("/run/media/kim/Mantu/ai-music/Goa_Separated_crops")
DEFAULT_OUTDIR = Path("/run/media/kim/Lehto")
DEFAULT_DB     = REPO_ROOT / "data" / "timeseries.db"

# ---------------------------------------------------------------------------
# Timeseries field lists
# ---------------------------------------------------------------------------
# 1D fields used for display curves and shape scalars (9 total)
TS_1D_FIELDS = [
    "rms_energy_bass_ts",
    "rms_energy_body_ts",
    "rms_energy_mid_ts",
    "rms_energy_air_ts",
    "spectral_flatness_ts",
    "spectral_flux_ts",
    "beat_activations_ts",
    "onsets_activations_ts",
    "tonic_strength_ts",
]
# Embedding layout: [mean,std]×9 + hpcp_raw×12 + hpcp_rot×12 + sin/cos = 44
EMBEDDING_DIMS = 44

# ---------------------------------------------------------------------------
# Metadata dicts for ts-derived scalar features (appended to UNITS/DESCS/METHODS)
# ---------------------------------------------------------------------------
TS_UNITS: dict = {}
TS_DESCS: dict = {}
TS_METHODS: dict = {}

for _f in TS_1D_FIELDS:
    _base = _f.replace("_ts", "")
    TS_UNITS[_f + "_mean"] = "-"
    TS_UNITS[_f + "_std"]  = "-"
    TS_DESCS[_f + "_mean"] = f"[ts] Mean of {_base} over time (averaged across crops)"
    TS_DESCS[_f + "_std"]  = f"[ts] Std of {_base} over time (averaged across crops)"
    TS_METHODS[_f + "_mean"] = ["spectral/timeseries_features.py", "TimeseriesDB"]
    TS_METHODS[_f + "_std"]  = ["spectral/timeseries_features.py", "TimeseriesDB"]

for _i in range(12):
    TS_UNITS[f"hpcp_raw_{_i}"] = "-"
    TS_UNITS[f"hpcp_rot_{_i}"] = "-"
    TS_DESCS[f"hpcp_raw_{_i}"] = f"[ts] Raw chroma pitch class {_i} mean (key-sensitive)"
    TS_DESCS[f"hpcp_rot_{_i}"] = f"[ts] Tonic-rotated chroma pitch class {_i} (key-invariant)"
    TS_METHODS[f"hpcp_raw_{_i}"] = ["harmonic/chroma.py", "TimeseriesDB hpcp_ts"]
    TS_METHODS[f"hpcp_rot_{_i}"] = ["harmonic/chroma.py", "TimeseriesDB hpcp_ts"]

TS_UNITS["tonic_sin"]  = "-"
TS_UNITS["tonic_cos"]  = "-"
TS_DESCS["tonic_sin"]  = "[ts] sin(2π·tonic/12) — circular tonic encoding"
TS_DESCS["tonic_cos"]  = "[ts] cos(2π·tonic/12) — circular tonic encoding"
TS_METHODS["tonic_sin"] = ["harmonic/chroma.py", "TimeseriesDB tonic_ts"]
TS_METHODS["tonic_cos"] = ["harmonic/chroma.py", "TimeseriesDB tonic_ts"]

# ts scalars that go into the scatter plot (NUMERIC_FEATURES extension)
TS_NUMERIC_FEATURES = (
    [f + "_mean" for f in TS_1D_FIELDS] +
    [f + "_std"  for f in TS_1D_FIELDS] +
    [f"hpcp_raw_{i}" for i in range(12)] +
    [f"hpcp_rot_{i}" for i in range(12)] +
    ["tonic_sin", "tonic_cos"]
)

# ---------------------------------------------------------------------------
# Pure helper functions (all tested in tests/test_build_dataset_stats.py)
# ---------------------------------------------------------------------------

def _strip_crop_suffix(key: str) -> str:
    """'Artist - Title_0' -> 'Artist - Title'"""
    return re.sub(r'_\d+$', '', key)


def _interp32(arr: np.ndarray) -> np.ndarray:
    """Interpolate a 1-D array of any length to exactly 32 steps."""
    n = len(arr)
    if n == 32:
        return arr.astype(np.float32)
    xp = np.linspace(0.0, 1.0, n)
    x  = np.linspace(0.0, 1.0, 32)
    return np.interp(x, xp, arr).astype(np.float32)


def _rotate_hpcp(hpcp: np.ndarray, tonic: int) -> np.ndarray:
    """Roll a [12] chroma vector so tonic lands at index 0."""
    return np.roll(hpcp.astype(np.float32), -tonic)


def _dominant_tonic(tonic_arr: np.ndarray) -> int:
    """Mode of rounded tonic values, clamped to [0, 11]."""
    rounded = np.round(tonic_arr).astype(int) % 12
    return int(np.argmax(np.bincount(rounded, minlength=12)))


def _cosine_top_k(emb: np.ndarray, names: list, k: int = 20) -> dict:
    """Return top-k cosine neighbors per track (self excluded).

    Args:
        emb:   [N, D] float32 embedding matrix (already z-scored)
        names: list of N track names
        k:     number of neighbors to return

    Returns:
        {track_name: [[neighbor_name, score], ...]} sorted descending
    """
    dist = scipy.spatial.distance.cdist(emb, emb, metric='cosine')  # [N, N]
    sim  = (1.0 - dist).astype(np.float32)
    np.fill_diagonal(sim, -2.0)  # exclude self

    result = {}
    for i, name in enumerate(names):
        top_idx = np.argsort(sim[i])[::-1][:k]
        result[name] = [[names[j], round(float(sim[i, j]), 4)] for j in top_idx]
    return result
