# Timeseries Explorer Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `plots/generate_explorer_data.py` with a unified `plots/build_dataset_stats.py` that adds timeseries-derived shape vectors, mini curves, and DJ-oriented track similarity to the Feature Explorer.

**Architecture:** Three-stage pipeline: (1) scalar pass — port existing logic; (2) timeseries pass — query TimeseriesDB, build per-track shape vectors and 32-step mini curves; (3) similarity pass — cosine distance over a 44-dim embedding with three DJ modes (overall, key-locked, pitch-shift). Outputs two JS files consumed by `feature_explorer.html`, which gains a collapsible timeseries panel and a similar-tracks panel.

**Tech Stack:** Python 3.12, numpy, scipy, `src.core.timeseries_db.TimeseriesDB`, Plotly (already in HTML), pytest.

---

## File map

| Action | Path |
|--------|------|
| Create | `plots/build_dataset_stats.py` |
| Create | `tests/test_build_dataset_stats.py` |
| Modify | `plots/feature_explorer.html` |
| Delete | `plots/generate_explorer_data.py` |

---

## Task 1: Tests + implementation of pure helper functions

**Files:**
- Create: `tests/test_build_dataset_stats.py`
- Create: `plots/build_dataset_stats.py` (helpers only)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_build_dataset_stats.py
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
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
cd /home/kim/Projects/mir
python -m pytest tests/test_build_dataset_stats.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'plots.build_dataset_stats'`

- [ ] **Step 3: Create `plots/build_dataset_stats.py` with helpers**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_build_dataset_stats.py -v
```
Expected: all 13 tests pass.

- [ ] **Step 5: Commit**

```bash
git add plots/build_dataset_stats.py tests/test_build_dataset_stats.py
git commit -m "feat: add build_dataset_stats helpers + tests"
```

---

## Task 2: Stage 1 — scalar pass (port existing code)

**Files:**
- Modify: `plots/build_dataset_stats.py` (append below helpers)

- [ ] **Step 1: Add scalar-pass tests**

Append to `tests/test_build_dataset_stats.py`:

```python
import json
import tempfile

from plots.build_dataset_stats import load_track_data, run_scalar_pass, write_data_js


def _make_info(path: Path, data: dict):
    path.write_text(json.dumps(data))


def test_load_track_data_averages_crops(tmp_path):
    track_dir = tmp_path / "Artist - Title"
    track_dir.mkdir()
    _make_info(track_dir / "Artist - Title_0.INFO", {"bpm": 140.0, "lufs": -10.0})
    _make_info(track_dir / "Artist - Title_1.INFO", {"bpm": 142.0, "lufs": -12.0})
    result = load_track_data(track_dir)
    assert result is not None
    assert abs(result["bpm"] - 141.0) < 0.01
    assert abs(result["lufs"] - -11.0) < 0.01


def test_load_track_data_skips_stems(tmp_path):
    track_dir = tmp_path / "Track"
    track_dir.mkdir()
    _make_info(track_dir / "Track_0.INFO",       {"bpm": 140.0})
    _make_info(track_dir / "Track_0_bass.INFO",  {"bpm": 999.0})  # stem — skip
    result = load_track_data(track_dir)
    assert abs(result["bpm"] - 140.0) < 0.01


def test_run_scalar_pass_produces_js(tmp_path):
    # Two track dirs each with one crop
    for name, bpm in [("Artist - A", 130.0), ("Artist - B", 145.0)]:
        d = tmp_path / name
        d.mkdir()
        _make_info(d / f"{name}_0.INFO", {"bpm": bpm, "lufs": -14.0})

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sorted_tracks, tracks_data = run_scalar_pass(tmp_path, out_dir)

    js = (out_dir / "feature_explorer_data.js").read_text()
    assert "Artist - A" in js
    assert "Artist - B" in js
    assert sorted_tracks == ["Artist - A", "Artist - B"]
    assert abs(tracks_data["Artist - A"]["bpm"] - 130.0) < 0.01
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_build_dataset_stats.py::test_load_track_data_averages_crops -v
```
Expected: `ImportError` (functions not yet defined).

- [ ] **Step 3: Add scalar pass to `plots/build_dataset_stats.py`**

Paste this after the helpers section. These are the `UNITS`, `DESCS`, `METHODS`, `PERCEPTUAL_FEATURES`, `NUMERIC_FEATURES`, `STEM_SUFFIXES` constants from the old script plus the new functions. Copy the full constants blocks verbatim from `plots/generate_explorer_data.py` (lines 29–321), then add:

```python
# (paste STEM_SUFFIXES, UNITS, DESCS, METHODS, PERCEPTUAL_FEATURES, NUMERIC_FEATURES
#  verbatim from generate_explorer_data.py lines 29-321)

# ---------------------------------------------------------------------------
# Stage 1 — scalar pass
# ---------------------------------------------------------------------------

def is_full_mix_info(path: Path) -> bool:
    stem = path.stem
    return not any(stem.endswith(s) for s in STEM_SUFFIXES)


def load_track_data(track_dir: Path) -> dict | None:
    """Average all full-mix crop INFO values for a single track directory."""
    infos = sorted([p for p in track_dir.glob("*.INFO") if is_full_mix_info(p)])
    if not infos:
        return None

    feature_values: dict[str, list] = defaultdict(list)
    metadata: dict = {}

    for info_path in infos:
        try:
            with open(info_path) as f:
                data = json.load(f)
        except Exception:
            continue
        for k, v in data.items():
            if isinstance(v, (int, float)) and v is not None:
                feature_values[k].append(float(v))
            elif k in ("spotify_id", "musicbrainz_id", "track_metadata_artist",
                       "track_metadata_title", "music_flamingo_short_genre",
                       "music_flamingo_short_mood", "music_flamingo_short_technical",
                       "label", "album", "isrc", "tidal_id", "tidal_url") \
                    and k not in metadata and v:
                metadata[k] = v
            elif k in ("genres", "artists") and k not in metadata \
                    and isinstance(v, list) and v:
                metadata[k] = ", ".join(str(x) for x in v)

    if not feature_values:
        return None

    averaged = {k: float(np.mean(vals)) for k, vals in feature_values.items() if vals}
    averaged.update(metadata)
    return averaged


def run_scalar_pass(src: Path, output_dir: Path) -> tuple[list, dict]:
    """Scan .INFO files and write feature_explorer_data.js (scalars only).

    Returns (sorted_tracks, tracks_data) for use by later stages.
    """
    src = Path(src)
    track_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    print(f"Stage 1: scanning {len(track_dirs)} track dirs in {src}")

    tracks_data: dict[str, dict] = {}
    for i, d in enumerate(track_dirs):
        if i % 200 == 0:
            print(f"  {i}/{len(track_dirs)} ...", end="\r")
        result = load_track_data(d)
        if result:
            tracks_data[d.name] = result

    print(f"\n  {len(tracks_data)} tracks loaded")
    sorted_tracks = sorted(tracks_data.keys())

    all_keys = set()
    for td in tracks_data.values():
        all_keys.update(td.keys())
    features_in_data = [f for f in NUMERIC_FEATURES if f in all_keys]

    write_data_js(sorted_tracks, tracks_data, features_in_data, output_dir)
    return sorted_tracks, tracks_data


def write_data_js(sorted_tracks: list, tracks_data: dict,
                  features_in_data: list, output_dir: Path) -> None:
    """Serialise all scalar features (including ts-derived) to feature_explorer_data.js."""
    out = Path(output_dir) / "feature_explorer_data.js"

    DATA = {feat: [tracks_data[t].get(feat) for t in sorted_tracks]
            for feat in features_in_data}

    def _s(key):
        return [tracks_data[t].get(key, "") or "" for t in sorted_tracks]

    all_units   = {**UNITS,   **TS_UNITS}
    all_descs   = {**DESCS,   **TS_DESCS}
    all_methods = {**METHODS, **TS_METHODS}

    with open(out, "w") as fh:
        fh.write("// Auto-generated — re-generate: python plots/build_dataset_stats.py\n\n")
        fh.write(f"const DATA = {json.dumps(DATA, separators=(',', ':'))};\n")
        fh.write(f"const TRACKS = {json.dumps(sorted_tracks, separators=(',', ':'))};\n")
        fh.write(f"const FEATURES = {json.dumps(sorted(features_in_data), separators=(',', ':'))};\n")
        fh.write(f"const UNITS = {json.dumps({f: all_units.get(f, '-') for f in features_in_data}, separators=(',', ':'))};\n")
        fh.write(f"const DESCS = {json.dumps({f: all_descs.get(f) for f in features_in_data}, separators=(',', ':'))};\n")
        fh.write(f"const METHODS = {json.dumps({f: all_methods.get(f, []) for f in features_in_data}, separators=(',', ':'))};\n")
        fh.write(f"const SPOTIFY = {json.dumps(_s('spotify_id'), separators=(',', ':'))};\n")
        fh.write(f"const MBIDS = {json.dumps(_s('musicbrainz_id'), separators=(',', ':'))};\n")
        fh.write(f"const LABELS = {json.dumps(_s('label'), separators=(',', ':'))};\n")
        fh.write(f"const ALBUMS = {json.dumps(_s('album'), separators=(',', ':'))};\n")
        fh.write(f"const GENRES = {json.dumps(_s('genres'), separators=(',', ':'))};\n")
        fh.write(f"const ARTISTS = {json.dumps(_s('artists'), separators=(',', ':'))};\n")
        fh.write(f"const ISRC = {json.dumps(_s('isrc'), separators=(',', ':'))};\n")
        fh.write(f"const TIDAL_IDS = {json.dumps(_s('tidal_id'), separators=(',', ':'))};\n")
        fh.write(f"const TIDAL_URLS = {json.dumps(_s('tidal_url'), separators=(',', ':'))};\n")
        fh.write(f"const FG_GENRE = {json.dumps(_s('music_flamingo_short_genre'), separators=(',', ':'))};\n")
        fh.write(f"const FG_MOOD = {json.dumps(_s('music_flamingo_short_mood'), separators=(',', ':'))};\n")
        fh.write(f"const FG_TECH = {json.dumps(_s('music_flamingo_short_technical'), separators=(',', ':'))};\n")
        fh.write(f"const PERCEPTUAL = new Set({json.dumps(sorted(PERCEPTUAL_FEATURES), separators=(',', ':'))});\n")

    size = out.stat().st_size
    print(f"  Wrote {out.name}: {size:,} bytes  ({len(sorted_tracks)} tracks, {len(features_in_data)} features)")
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_build_dataset_stats.py -v
```
Expected: all tests pass (including the 3 new scalar-pass tests).

- [ ] **Step 5: Commit**

```bash
git add plots/build_dataset_stats.py tests/test_build_dataset_stats.py
git commit -m "feat: Stage 1 scalar pass in build_dataset_stats"
```

---

## Task 3: Stage 2 — timeseries pass

**Files:**
- Modify: `plots/build_dataset_stats.py` (append)
- Modify: `tests/test_build_dataset_stats.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_build_dataset_stats.py`:

```python
from plots.build_dataset_stats import process_track_ts, run_timeseries_pass
from src.core.timeseries_db import TimeseriesDB


def _make_db(tmp_path, crop_key: str, n_steps: int = 64) -> TimeseriesDB:
    db = TimeseriesDB(tmp_path / "test.db")
    ts = {
        "rms_energy_bass_ts":  np.random.rand(n_steps).astype(np.float32).tolist(),
        "rms_energy_body_ts":  np.random.rand(n_steps).astype(np.float32).tolist(),
        "hpcp_ts":             np.random.rand(n_steps, 12).astype(np.float32).tolist(),
        "tonic_ts":            np.full(n_steps, 5.0, dtype=np.float32).tolist(),
        "tonic_strength_ts":   np.random.rand(n_steps).astype(np.float32).tolist(),
    }
    db.put(crop_key, ts)
    return db


def test_process_track_ts_curve_length(tmp_path):
    db = _make_db(tmp_path, "Artist - Title_0", n_steps=256)
    result = process_track_ts(["Artist - Title_0"], db)
    db.close()
    assert result is not None
    assert len(result["curves"]["rms_energy_bass_ts"]) == 32


def test_process_track_ts_curve_normalised(tmp_path):
    db = _make_db(tmp_path, "Artist - Title_0")
    result = process_track_ts(["Artist - Title_0"], db)
    db.close()
    for field, curve in result["curves"].items():
        assert max(curve) <= 1.0 + 1e-5, f"{field} not normalised"


def test_process_track_ts_shape_scalars(tmp_path):
    db = _make_db(tmp_path, "Artist - Title_0")
    result = process_track_ts(["Artist - Title_0"], db)
    db.close()
    assert "rms_energy_bass_ts_mean" in result["shape"]
    assert "rms_energy_bass_ts_std"  in result["shape"]


def test_process_track_ts_tonic_rotation(tmp_path):
    db = _make_db(tmp_path, "Artist - Title_0")  # tonic_ts all = 5.0
    result = process_track_ts(["Artist - Title_0"], db)
    db.close()
    assert result is not None
    expected_sin = float(np.sin(2 * np.pi * 5 / 12))
    assert abs(result["shape"]["tonic_sin"] - expected_sin) < 1e-4
    # hpcp_rot should be rotated version of hpcp_raw
    raw = np.array(result["hpcp_raw"], dtype=np.float32)
    rot = np.array(result["hpcp_rot"], dtype=np.float32)
    expected_rot = _rotate_hpcp(raw, 5)
    np.testing.assert_allclose(rot, expected_rot, atol=1e-4)


def test_process_track_ts_variable_n_steps(tmp_path):
    db = TimeseriesDB(tmp_path / "var.db")
    # Two crops with different n_steps — both should contribute
    for key, n in [("T_0", 64), ("T_1", 128), ("T_2", 256)]:
        db.put(key, {"rms_energy_bass_ts": np.random.rand(n).astype(np.float32).tolist()})
    result = process_track_ts(["T_0", "T_1", "T_2"], db)
    db.close()
    assert result is not None
    assert len(result["curves"]["rms_energy_bass_ts"]) == 32


def test_process_track_ts_missing_crops_skipped(tmp_path):
    db = _make_db(tmp_path, "Artist - Title_0")
    # "Artist - Title_1" is not in DB — should be silently skipped
    result = process_track_ts(["Artist - Title_0", "Artist - Title_1"], db)
    db.close()
    assert result is not None  # still has data from crop 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_build_dataset_stats.py -k "process_track_ts" -v
```
Expected: `ImportError`.

- [ ] **Step 3: Add Stage 2 to `plots/build_dataset_stats.py`**

```python
# ---------------------------------------------------------------------------
# Stage 2 — timeseries pass
# ---------------------------------------------------------------------------

def process_track_ts(crop_keys: list, db) -> dict | None:
    """Aggregate timeseries for all crops of one track.

    Returns dict with:
      'curves' : {field: list[32]}          — L∞-normalised to [0,1]
      'shape'  : {field_mean/std/hpcp/tonic: float}
      'hpcp_raw': list[12]
      'hpcp_rot': list[12]
    Returns None if no timeseries data found.
    """
    accumulated: dict[str, list] = {f: [] for f in TS_1D_FIELDS}
    hpcp_vecs: list = []
    tonic_vals: list = []

    for key in crop_keys:
        arrays = db.get(key)
        if arrays is None:
            continue
        for f in TS_1D_FIELDS:
            if f in arrays and len(arrays[f]) > 0:
                accumulated[f].append(_interp32(arrays[f]))
        if "hpcp_ts" in arrays:
            h = arrays["hpcp_ts"]
            if h.ndim == 2 and h.shape[1] == 12:
                hpcp_vecs.append(h.mean(axis=0).astype(np.float32))
        if "tonic_ts" in arrays and len(arrays["tonic_ts"]) > 0:
            tonic_vals.extend(arrays["tonic_ts"].tolist())

    has_1d = any(len(v) > 0 for v in accumulated.values())
    if not has_1d and not hpcp_vecs:
        return None

    curves: dict = {}
    shape: dict = {}

    for f in TS_1D_FIELDS:
        if not accumulated[f]:
            continue
        stack = np.stack(accumulated[f], axis=0)   # [n_crops, 32]
        mean_curve = stack.mean(axis=0)             # [32]
        mx = mean_curve.max()
        curves[f] = (mean_curve / mx if mx > 0 else mean_curve).tolist()
        shape[f + "_mean"] = float(stack.mean())
        shape[f + "_std"]  = float(stack.std())

    # Chroma
    hpcp_raw_list = None
    hpcp_rot_list = None
    tonic = 0

    if hpcp_vecs:
        hpcp_raw_arr = np.stack(hpcp_vecs, axis=0).mean(axis=0)   # [12]
        hpcp_raw_list = hpcp_raw_arr.tolist()
        for i, v in enumerate(hpcp_raw_list):
            shape[f"hpcp_raw_{i}"] = float(v)

    if tonic_vals:
        tonic_arr = np.array(tonic_vals, dtype=np.float32)
        tonic = _dominant_tonic(tonic_arr)
        shape["tonic_sin"] = float(np.sin(2 * np.pi * tonic / 12))
        shape["tonic_cos"] = float(np.cos(2 * np.pi * tonic / 12))
        if hpcp_raw_list is not None:
            hpcp_rot_arr = _rotate_hpcp(np.array(hpcp_raw_list, dtype=np.float32), tonic)
            hpcp_rot_list = hpcp_rot_arr.tolist()
            for i, v in enumerate(hpcp_rot_list):
                shape[f"hpcp_rot_{i}"] = float(v)

    result: dict = {"curves": curves, "shape": shape}
    if hpcp_raw_list is not None:
        result["hpcp_raw"] = hpcp_raw_list
    if hpcp_rot_list is not None:
        result["hpcp_rot"] = hpcp_rot_list
    return result


def run_timeseries_pass(sorted_tracks: list, src: Path,
                        output_dir: Path, db_path: Path) -> dict:
    """Query TimeseriesDB for all tracks, build shape vectors and mini curves.

    Saves .ts_cache.npz to output_dir for Stage 3 reuse.
    Returns ts_data: {track_name: process_track_ts result}.
    """
    from core.timeseries_db import TimeseriesDB

    output_dir = Path(output_dir)
    db = TimeseriesDB.open(db_path)
    print(f"Stage 2: TimeseriesDB has {db.count():,} entries")

    # Map crop keys to track names
    all_keys = db.all_keys()
    crops_by_track: dict[str, list] = defaultdict(list)
    for key in all_keys:
        track = _strip_crop_suffix(key)
        if track in set(sorted_tracks):
            crops_by_track[track].append(key)

    print(f"  {len(crops_by_track)} tracks have timeseries data")

    ts_data: dict[str, dict] = {}
    for i, name in enumerate(sorted_tracks):
        if i % 200 == 0:
            print(f"  {i}/{len(sorted_tracks)} ...", end="\r")
        crop_keys = crops_by_track.get(name, [])
        if not crop_keys:
            continue
        result = process_track_ts(crop_keys, db)
        if result is not None:
            ts_data[name] = result

    db.close()
    print(f"\n  {len(ts_data)} tracks with ts data")

    # Save cache for Stage 3 reuse
    N = len(sorted_tracks)
    name_arr = np.array(sorted_tracks)
    # Build raw embedding rows (un-normalised)
    raw_emb = np.zeros((N, EMBEDDING_DIMS), dtype=np.float32)
    mini_curves = np.zeros((N, len(TS_1D_FIELDS), 32), dtype=np.float32)

    for i, name in enumerate(sorted_tracks):
        td = ts_data.get(name)
        if td is None:
            continue
        s = td.get("shape", {})
        col = 0
        for f in TS_1D_FIELDS:
            raw_emb[i, col]   = s.get(f + "_mean", 0.0)
            raw_emb[i, col+1] = s.get(f + "_std",  0.0)
            col += 2
        for j in range(12):
            raw_emb[i, col+j] = s.get(f"hpcp_raw_{j}", 0.0)
        col += 12
        for j in range(12):
            raw_emb[i, col+j] = s.get(f"hpcp_rot_{j}", 0.0)
        col += 12
        raw_emb[i, col]   = s.get("tonic_sin", 0.0)
        raw_emb[i, col+1] = s.get("tonic_cos", 0.0)

        for fi, f in enumerate(TS_1D_FIELDS):
            c = td["curves"].get(f)
            if c:
                mini_curves[i, fi] = np.array(c, dtype=np.float32)

    cache_path = output_dir / ".ts_cache.npz"
    np.savez_compressed(cache_path,
        track_names=name_arr,
        raw_embedding=raw_emb,
        mini_curves=mini_curves)
    print(f"  Cache saved: {cache_path}")
    return ts_data
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_build_dataset_stats.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add plots/build_dataset_stats.py tests/test_build_dataset_stats.py
git commit -m "feat: Stage 2 timeseries pass in build_dataset_stats"
```

---

## Task 4: Stage 3 — similarity pass + CLI + delete old script

**Files:**
- Modify: `plots/build_dataset_stats.py` (append)
- Modify: `tests/test_build_dataset_stats.py` (append)
- Delete: `plots/generate_explorer_data.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_build_dataset_stats.py`:

```python
from plots.build_dataset_stats import build_embedding, run_similarity_pass


def _make_ts_data(n_tracks: int = 5) -> tuple[list, dict]:
    from plots.build_dataset_stats import TS_1D_FIELDS
    names = [f"Track {i}" for i in range(n_tracks)]
    ts_data = {}
    for name in names:
        shape = {}
        for f in TS_1D_FIELDS:
            shape[f + "_mean"] = float(np.random.rand())
            shape[f + "_std"]  = float(np.random.rand())
        for j in range(12):
            shape[f"hpcp_raw_{j}"] = float(np.random.rand())
            shape[f"hpcp_rot_{j}"] = float(np.random.rand())
        shape["tonic_sin"] = float(np.random.rand())
        shape["tonic_cos"] = float(np.random.rand())
        ts_data[name] = {"shape": shape, "curves": {}, "hpcp_raw": [0.0]*12, "hpcp_rot": [0.0]*12}
    return names, ts_data


def test_build_embedding_shape():
    names, ts_data = _make_ts_data(10)
    emb = build_embedding(names, ts_data)
    assert emb.shape == (10, 44)
    assert emb.dtype == np.float32


def test_build_embedding_zscored():
    names, ts_data = _make_ts_data(50)
    emb = build_embedding(names, ts_data)
    # After z-scoring, each column should have mean ≈ 0, std ≈ 1
    np.testing.assert_allclose(emb.mean(axis=0), np.zeros(44), atol=1e-4)


def test_build_embedding_missing_track_gets_zero():
    names = ["A", "B", "C"]
    ts_data = {"A": {"shape": {"rms_energy_bass_ts_mean": 1.0}, "curves": {}}}
    emb = build_embedding(names, ts_data)
    # B and C are missing — their rows should not raise and be finite
    assert np.all(np.isfinite(emb))


def test_run_similarity_pass_writes_js(tmp_path):
    from plots.build_dataset_stats import TS_1D_FIELDS
    names, ts_data = _make_ts_data(10)
    run_similarity_pass(names, ts_data, tmp_path)

    js = (tmp_path / "feature_explorer_timeseries.js").read_text()
    assert "TS_CURVES" in js
    assert "TS_NEIGHBORS" in js
    assert "overall" in js
    assert "key_locked" in js
    assert "pitch_shift" in js
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_build_dataset_stats.py -k "build_embedding or run_similarity" -v
```
Expected: `ImportError`.

- [ ] **Step 3: Add Stage 3 + CLI to `plots/build_dataset_stats.py`**

```python
# ---------------------------------------------------------------------------
# Stage 3 — similarity pass
# ---------------------------------------------------------------------------

def build_embedding(track_names: list, ts_data: dict) -> np.ndarray:
    """Build and z-score the [N, 44] embedding matrix.

    Layout (44 dims):
      [mean, std] × 9 1D fields (18)
      hpcp_raw_0..11             (12)
      hpcp_rot_0..11             (12)
      tonic_sin, tonic_cos        (2)
    """
    N = len(track_names)
    raw = np.zeros((N, EMBEDDING_DIMS), dtype=np.float32)

    for i, name in enumerate(track_names):
        td = ts_data.get(name)
        if td is None:
            continue
        s = td.get("shape", {})
        col = 0
        for f in TS_1D_FIELDS:
            raw[i, col]   = s.get(f + "_mean", 0.0)
            raw[i, col+1] = s.get(f + "_std",  0.0)
            col += 2
        for j in range(12):
            raw[i, col+j] = s.get(f"hpcp_raw_{j}", 0.0)
        col += 12
        for j in range(12):
            raw[i, col+j] = s.get(f"hpcp_rot_{j}", 0.0)
        col += 12
        raw[i, col]   = s.get("tonic_sin", 0.0)
        raw[i, col+1] = s.get("tonic_cos", 0.0)

    mu = raw.mean(axis=0)
    sigma = raw.std(axis=0)
    sigma[sigma == 0] = 1.0
    return ((raw - mu) / sigma).astype(np.float32)


def run_similarity_pass(sorted_tracks: list, ts_data: dict, output_dir: Path) -> None:
    """Compute cosine similarity for three DJ modes, write feature_explorer_timeseries.js."""
    output_dir = Path(output_dir)
    print(f"Stage 3: computing similarity for {len(sorted_tracks)} tracks ...")

    emb = build_embedding(sorted_tracks, ts_data)

    # Key-locked slice: hpcp_raw (12) + tonic_sin/cos (2) — cols 18..31 + 42..43
    idx_raw_start = 9 * 2          # = 18
    kl_idx = list(range(idx_raw_start, idx_raw_start + 12)) + [42, 43]
    # Pitch-shift slice: hpcp_rot (12) — cols 30..41
    ps_idx = list(range(idx_raw_start + 12, idx_raw_start + 24))

    emb_kl = emb[:, kl_idx]
    emb_ps = emb[:, ps_idx]

    print("  overall similarity ...")
    nbr_overall    = _cosine_top_k(emb,    sorted_tracks, k=20)
    print("  key-locked similarity ...")
    nbr_key_locked = _cosine_top_k(emb_kl, sorted_tracks, k=20)
    print("  pitch-shift similarity ...")
    nbr_pitch_shift= _cosine_top_k(emb_ps, sorted_tracks, k=20)

    # Build TS_CURVES and TS_NEIGHBORS JS objects
    ts_curves_obj: dict = {}
    for name in sorted_tracks:
        td = ts_data.get(name)
        if td is None:
            continue
        entry: dict = {}
        for f in TS_1D_FIELDS:
            c = td["curves"].get(f)
            if c:
                entry[f] = [round(v, 5) for v in c]
        if "hpcp_raw" in td:
            entry["hpcp_raw"] = [round(v, 5) for v in td["hpcp_raw"]]
        if "hpcp_rot" in td:
            entry["hpcp_rot"] = [round(v, 5) for v in td["hpcp_rot"]]
        if entry:
            ts_curves_obj[name] = entry

    ts_neighbors_obj: dict = {}
    for name in sorted_tracks:
        if name not in ts_curves_obj:
            continue
        ts_neighbors_obj[name] = {
            "overall":     nbr_overall.get(name, []),
            "key_locked":  nbr_key_locked.get(name, []),
            "pitch_shift": nbr_pitch_shift.get(name, []),
        }

    out = output_dir / "feature_explorer_timeseries.js"
    with open(out, "w") as fh:
        fh.write("// Auto-generated — re-generate: python plots/build_dataset_stats.py\n\n")
        fh.write(f"const TS_CURVES = {json.dumps(ts_curves_obj, separators=(',', ':'))};\n")
        fh.write(f"const TS_NEIGHBORS = {json.dumps(ts_neighbors_obj, separators=(',', ':'))};\n")

    size = out.stat().st_size
    print(f"  Wrote {out.name}: {size:,} bytes  ({len(ts_curves_obj)} tracks)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build all Feature Explorer dataset statistics")
    p.add_argument("--source",    default=str(DEFAULT_SRC),
                   help=f"Root crops directory (default: {DEFAULT_SRC})")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTDIR),
                   help=f"Output directory for JS files (default: {DEFAULT_OUTDIR})")
    p.add_argument("--db",        default=str(DEFAULT_DB),
                   help=f"TimeseriesDB path (default: {DEFAULT_DB})")
    p.add_argument("--skip-scalars",     action="store_true",
                   help="Skip Stage 1 (load sorted_tracks from cache)")
    p.add_argument("--skip-timeseries",  action="store_true",
                   help="Skip Stages 2 and 3")
    p.add_argument("--skip-curves",      action="store_true",
                   help="Skip Stage 2 only (load cache, re-run Stage 3)")
    p.add_argument("--skip-similarity",  action="store_true",
                   help="Skip Stage 3 only")
    return p.parse_args()


def main():
    args = parse_args()
    src        = Path(args.source)
    output_dir = Path(args.output_dir)
    db_path    = Path(args.db)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"Error: source not found: {src}", file=sys.stderr)
        sys.exit(1)

    sorted_tracks: list = []
    tracks_data: dict = {}
    ts_data: dict = {}

    # --- Stage 1 ---
    if not args.skip_scalars:
        sorted_tracks, tracks_data = run_scalar_pass(src, output_dir)
    else:
        # Load track list from existing cache
        cache = output_dir / ".ts_cache.npz"
        if not cache.exists():
            print(f"Error: --skip-scalars requires {cache} (run without flag first)",
                  file=sys.stderr)
            sys.exit(1)
        sorted_tracks = list(np.load(cache, allow_pickle=True)["track_names"])
        print(f"Stage 1 skipped — loaded {len(sorted_tracks)} tracks from cache")

    # --- Stage 2 ---
    skip_ts = args.skip_timeseries or args.skip_curves
    if not args.skip_timeseries:
        if not args.skip_curves:
            ts_data = run_timeseries_pass(sorted_tracks, src, output_dir, db_path)
        else:
            cache = output_dir / ".ts_cache.npz"
            if not cache.exists():
                print(f"Error: --skip-curves requires {cache}", file=sys.stderr)
                sys.exit(1)
            npz = np.load(cache, allow_pickle=True)
            # Rebuild ts_data from cache (curves not stored — similarity only)
            raw_emb = npz["raw_embedding"]
            names   = list(npz["track_names"])
            # Minimal ts_data: shape scalars only (enough for similarity)
            for i, name in enumerate(names):
                shape: dict = {}
                col = 0
                for f in TS_1D_FIELDS:
                    shape[f + "_mean"] = float(raw_emb[i, col])
                    shape[f + "_std"]  = float(raw_emb[i, col+1])
                    col += 2
                for j in range(12):
                    shape[f"hpcp_raw_{j}"] = float(raw_emb[i, col+j])
                col += 12
                for j in range(12):
                    shape[f"hpcp_rot_{j}"] = float(raw_emb[i, col+j])
                col += 12
                shape["tonic_sin"] = float(raw_emb[i, col])
                shape["tonic_cos"] = float(raw_emb[i, col+1])
                ts_data[name] = {"shape": shape, "curves": {}}
            sorted_tracks = names
            print(f"Stage 2 skipped — loaded {len(ts_data)} tracks from cache")

        # Merge ts shape scalars into tracks_data and re-write data JS
        if not args.skip_scalars and ts_data:
            features_in_data_set = set()
            for td in tracks_data.values():
                features_in_data_set.update(td.keys())
            for name, tsd in ts_data.items():
                if name in tracks_data:
                    tracks_data[name].update(tsd.get("shape", {}))
                    features_in_data_set.update(tsd.get("shape", {}).keys())
            features_in_data = (
                [f for f in NUMERIC_FEATURES + TS_NUMERIC_FEATURES if f in features_in_data_set]
            )
            write_data_js(sorted_tracks, tracks_data, features_in_data, output_dir)

    # --- Stage 3 ---
    if not args.skip_timeseries and not args.skip_similarity:
        run_similarity_pass(sorted_tracks, ts_data, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/test_build_dataset_stats.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Delete old script**

```bash
git rm plots/generate_explorer_data.py
```

- [ ] **Step 6: Commit**

```bash
git add plots/build_dataset_stats.py tests/test_build_dataset_stats.py
git commit -m "feat: Stage 3 similarity + CLI; delete generate_explorer_data.py"
```

---

## Task 5: Feature Explorer HTML — ts panel CSS + HTML

**Files:**
- Modify: `plots/feature_explorer.html`

- [ ] **Step 1: Add CSS inside `<style>` before `</style>`**

Find the closing `</style>` tag (around line 680) and insert immediately before it:

```css
    /* -------- Timeseries panel -------- */
    #ts-panel {
      position: fixed;
      bottom: 36px;
      left: 278px;
      width: 520px;
      max-height: 72vh;
      overflow-y: auto;
      background: rgba(15, 20, 40, 0.97);
      border: 1px solid #333355;
      border-radius: 8px;
      padding: 11px;
      z-index: 300;
      display: none;
      box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .ts-section-label {
      font-size: 10px;
      color: #555;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin: 6px 0 3px;
    }
    .ts-curves-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 4px;
      margin-bottom: 4px;
    }
    .ts-curve-cell {
      display: flex;
      flex-direction: column;
    }
    .ts-curve-label {
      font-size: 9px;
      color: #555;
      text-align: center;
      margin-bottom: 1px;
    }
    .ts-chroma-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 6px;
    }
    .ts-chroma-label {
      font-size: 9px;
      color: #555;
      margin-bottom: 2px;
    }
    .ts-sim-tabs {
      display: flex;
      gap: 4px;
      margin-bottom: 5px;
    }
    .ts-sim-tab {
      background: #0f3460;
      color: #ccc;
      border: 1px solid #333355;
      padding: 3px 10px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 11px;
    }
    .ts-sim-tab.active {
      background: #e94560;
      color: white;
      border-color: #e94560;
    }
    .ts-sim-tab:hover { border-color: #e94560; }
    #ts-neighbor-list {
      max-height: 160px;
      overflow-y: auto;
    }
    .ts-neighbor-item {
      display: flex;
      align-items: center;
      gap: 5px;
      padding: 3px 2px;
      cursor: pointer;
      font-size: 11px;
      border-bottom: 1px solid #1a2040;
      color: #aaa;
    }
    .ts-neighbor-item:hover { color: white; }
    .ts-neighbor-bar {
      height: 3px;
      background: #e94560;
      border-radius: 2px;
      flex-shrink: 0;
    }
```

- [ ] **Step 2: Add HTML panel before `</body>`**

Find the `<div id="caption-popup"></div>` line (around line 863) and insert immediately after it:

```html
  <div id="ts-panel">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
      <span style="font-size:11px;font-weight:bold;color:#e94560;">Timeseries</span>
      <button onclick="hideTsPanel()"
        style="background:none;border:none;color:#888;cursor:pointer;font-size:15px;padding:0;">&#x2715;</button>
    </div>
    <div class="ts-section-label">Energy &amp; Spectral over time</div>
    <div class="ts-curves-grid" id="ts-curves-grid"></div>
    <div class="ts-section-label">Chroma</div>
    <div class="ts-chroma-row">
      <div>
        <div class="ts-chroma-label">Raw (key-sensitive)</div>
        <div id="ts-chroma-raw" style="height:70px;"></div>
      </div>
      <div>
        <div class="ts-chroma-label">Tonic-aligned (key-invariant)</div>
        <div id="ts-chroma-rot" style="height:70px;"></div>
      </div>
    </div>
    <div class="ts-section-label">Similar tracks</div>
    <div class="ts-sim-tabs">
      <button class="ts-sim-tab active" onclick="switchTsTab('overall')">Overall</button>
      <button class="ts-sim-tab" onclick="switchTsTab('key_locked')">Key-locked</button>
      <button class="ts-sim-tab" onclick="switchTsTab('pitch_shift')">Pitch-shift</button>
    </div>
    <div id="ts-neighbor-list"></div>
  </div>
```

- [ ] **Step 3: Add `<script>` tag for timeseries data (after existing data script tag)**

Find the line `<script src="feature_explorer_data.js"></script>` (near the bottom of the file, before the main `<script>`) and add immediately after:

```html
  <script>
    // Load timeseries data if available (generated by build_dataset_stats.py)
    // Fails silently if file is absent (scalar-only rebuild).
  </script>
  <script src="feature_explorer_timeseries.js" onerror="window._tsLoadFailed=true;"></script>
```

- [ ] **Step 4: Verify HTML is valid (no syntax errors)**

Open `plots/feature_explorer.html` in a browser (`file://...`) and check the console for errors.
Expected: no errors, existing functionality works normally, ts panel is invisible.

- [ ] **Step 5: Commit**

```bash
git add plots/feature_explorer.html
git commit -m "feat: add timeseries panel CSS and HTML to feature_explorer"
```

---

## Task 6: Feature Explorer HTML — mini curves + chroma JS

**Files:**
- Modify: `plots/feature_explorer.html`

- [ ] **Step 1: Add JS constants and `showTsPanel` function**

Find the `function hideTrackPanel()` definition (around line 1392) and insert the following block immediately before it:

```javascript
    // ========== Timeseries panel ==========
    const _tsAvailable = typeof TS_CURVES !== 'undefined';
    let _tsCurrentTrack = null;
    let _tsCurMode = 'overall';

    const _TS_FIELD_LABELS = {
      rms_energy_bass_ts:    'RMS bass',
      rms_energy_body_ts:    'RMS body',
      rms_energy_mid_ts:     'RMS mid',
      rms_energy_air_ts:     'RMS air',
      spectral_flatness_ts:  'Sp. flat',
      spectral_flux_ts:      'Sp. flux',
      beat_activations_ts:   'Beat',
      onsets_activations_ts: 'Onsets',
      tonic_strength_ts:     'Key str.',
    };
    const _TS_FIELDS = Object.keys(_TS_FIELD_LABELS);
    const _PITCH_CLASSES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];

    const _TS_PLOT_LAYOUT = {
      margin: {t:0,b:0,l:0,r:0},
      paper_bgcolor: 'transparent',
      plot_bgcolor:  'transparent',
      xaxis: {visible:false},
      yaxis: {visible:false, range:[0, 1.05]},
      showlegend: false,
    };
    const _TS_PLOT_CFG = {staticPlot:true, displayModeBar:false, responsive:true};

    function showTsPanel(trackName) {
      if (!_tsAvailable) return;
      const curves = TS_CURVES[trackName];
      if (!curves) return;
      _tsCurrentTrack = trackName;

      // Build sparkline grid
      const grid = document.getElementById('ts-curves-grid');
      grid.innerHTML = '';
      _TS_FIELDS.forEach(function(field) {
        const cell = document.createElement('div');
        cell.className = 'ts-curve-cell';
        const lbl = document.createElement('div');
        lbl.className = 'ts-curve-label';
        lbl.textContent = _TS_FIELD_LABELS[field];
        cell.appendChild(lbl);
        const plotDiv = document.createElement('div');
        plotDiv.style.height = '46px';
        cell.appendChild(plotDiv);
        grid.appendChild(cell);
        const vals = curves[field];
        if (vals && vals.length) {
          Plotly.newPlot(plotDiv, [{
            y: vals, type: 'scatter', mode: 'lines',
            line: {color:'#00d2ff', width:1.5},
            hoverinfo: 'none',
          }], _TS_PLOT_LAYOUT, _TS_PLOT_CFG);
        } else {
          plotDiv.style.background = '#0a0f20';
          plotDiv.style.borderRadius = '3px';
        }
      });

      // Chroma bars
      const _chromaLayout = {
        margin: {t:0,b:18,l:0,r:0},
        paper_bgcolor: 'transparent',
        plot_bgcolor:  'transparent',
        xaxis: {tickfont:{size:8,color:'#555'}, tickmode:'array',
                tickvals:_PITCH_CLASSES, ticktext:_PITCH_CLASSES},
        yaxis: {visible:false},
        showlegend: false,
        bargap: 0.05,
      };
      if (curves.hpcp_raw) {
        Plotly.newPlot('ts-chroma-raw', [{
          x: _PITCH_CLASSES, y: curves.hpcp_raw, type: 'bar',
          marker: {color:'#e94560'}, hoverinfo: 'none',
        }], _chromaLayout, _TS_PLOT_CFG);
      }
      if (curves.hpcp_rot) {
        const rotLabels = _PITCH_CLASSES.map(function(_, i) { return i===0?'root':'+'+i; });
        Plotly.newPlot('ts-chroma-rot', [{
          x: rotLabels, y: curves.hpcp_rot, type: 'bar',
          marker: {color:'#7b2fe0'}, hoverinfo: 'none',
        }], Object.assign({}, _chromaLayout, {
          xaxis: {tickfont:{size:8,color:'#555'}, tickmode:'array',
                  tickvals:rotLabels, ticktext:rotLabels},
        }), _TS_PLOT_CFG);
      }

      _renderTsNeighbors(trackName, _tsCurMode);
      document.getElementById('ts-panel').style.display = '';
    }

    function hideTsPanel() {
      document.getElementById('ts-panel').style.display = 'none';
      _tsCurrentTrack = null;
    }
```

- [ ] **Step 2: Reload in browser, click a track in scatter — ts panel should appear with sparklines**

Open the HTML locally with the real data files. Click a point. Verify:
- Ts panel appears to the right of the track detail panel
- 9 sparkline charts render (or show dark placeholder if field missing)
- Chroma bars render

- [ ] **Step 3: Commit**

```bash
git add plots/feature_explorer.html
git commit -m "feat: timeseries sparklines and chroma bars in explorer"
```

---

## Task 7: Feature Explorer HTML — similar tracks JS + wiring

**Files:**
- Modify: `plots/feature_explorer.html`

- [ ] **Step 1: Add `_renderTsNeighbors`, `switchTsTab`, and wire into `showTrackPanel`**

Find the `function hideTsPanel()` line (added in Task 6) and insert after it:

```javascript
    function _renderTsNeighbors(trackName, mode) {
      if (typeof TS_NEIGHBORS === 'undefined') {
        document.getElementById('ts-neighbor-list').innerHTML =
          '<div style="color:#555;font-size:10px;">No similarity data</div>';
        return;
      }
      const nbrs = TS_NEIGHBORS[trackName];
      if (!nbrs) {
        document.getElementById('ts-neighbor-list').innerHTML =
          '<div style="color:#555;font-size:10px;">Track not in similarity index</div>';
        return;
      }
      const list = nbrs[mode] || [];
      const el = document.getElementById('ts-neighbor-list');
      if (!list.length) {
        el.innerHTML = '<div style="color:#555;font-size:10px;">No neighbors</div>';
        return;
      }
      el.innerHTML = list.map(function(pair) {
        const name = pair[0], score = pair[1];
        const barW = Math.round(Math.max(0, score) * 80);
        const idx  = TRACKS.indexOf(name);
        const click = idx >= 0
          ? 'showTrackPanel(' + idx + ',FEATURES)'
          : '';
        return '<div class="ts-neighbor-item" onclick="' + click + '">' +
          '<div class="ts-neighbor-bar" style="width:' + barW + 'px;"></div>' +
          '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="'
            + name.replace(/"/g,'&quot;') + '">' + name + '</span>' +
          '<span style="color:#555;flex-shrink:0;">' + score.toFixed(3) + '</span>' +
          '</div>';
      }).join('');
    }

    function switchTsTab(mode) {
      _tsCurMode = mode;
      document.querySelectorAll('.ts-sim-tab').forEach(function(btn) {
        btn.classList.toggle('active',
          btn.getAttribute('onclick') === "switchTsTab('" + mode + "')");
      });
      if (_tsCurrentTrack) _renderTsNeighbors(_tsCurrentTrack, mode);
    }
```

- [ ] **Step 2: Wire `showTrackPanel` to call `showTsPanel`**

Find `function showTrackPanel(idx, featureKeys)` (around line 1367). The function ends with:

```javascript
      panel.style.display = '';
    }
```

Change it to:

```javascript
      panel.style.display = '';
      if (_tsAvailable) showTsPanel(TRACKS[idx]);
    }
```

- [ ] **Step 3: Wire `hideTrackPanel` to also hide the ts panel**

Find:

```javascript
    function hideTrackPanel() {
      document.getElementById('track-panel').style.display = 'none';
    }
```

Change to:

```javascript
    function hideTrackPanel() {
      document.getElementById('track-panel').style.display = 'none';
      hideTsPanel();
    }
```

- [ ] **Step 4: Full end-to-end browser test**

Open the HTML locally with real output files. Verify:
1. Clicking a scatter point shows both the track detail panel and the ts panel
2. The "Overall" tab shows 20 neighbors as a ranked list with score bars
3. Clicking "Key-locked" and "Pitch-shift" tabs switches the neighbor list
4. Clicking a neighbor name selects that track in the scatter and updates both panels
5. The ✕ on the ts panel closes only the ts panel; ✕ on the track panel closes both
6. With `feature_explorer_timeseries.js` absent, no errors appear in console

- [ ] **Step 5: Commit**

```bash
git add plots/feature_explorer.html
git commit -m "feat: similar tracks panel and full ts panel wiring in explorer"
```

---

## Task 8: End-to-end run on real data

**Goal:** Run the full pipeline against actual crops and verify output quality.

- [ ] **Step 1: Run scalar-only pass to verify Stage 1 works**

```bash
python plots/build_dataset_stats.py \
  --source /home/kim/Projects/goa_crops \
  --output-dir /run/media/kim/Lehto \
  --skip-timeseries
```
Expected: prints track count and feature count; `feature_explorer_data.js` written. Open in browser — scatter should work exactly as before.

- [ ] **Step 2: Run full pipeline**

```bash
python plots/build_dataset_stats.py \
  --source /home/kim/Projects/goa_crops \
  --output-dir /run/media/kim/Lehto
```
Expected: all three stages complete; prints timeseries entry count and output file sizes; both JS files written.

- [ ] **Step 3: Check DB coverage**

```bash
python -c "
from src.core.timeseries_db import TimeseriesDB
db = TimeseriesDB.open()
print(f'DB entries: {db.count():,}')
"
```
Note the coverage fraction (entries / (tracks × crops)) — expected partial coverage since dataset is still being processed.

- [ ] **Step 4: Spot-check similarity in browser**

Open `feature_explorer.html` in browser. Click a Psytrance track. Verify:
- "Key-locked" neighbors are plausibly in the same key (cross-check with `tonic` scalar feature)
- "Pitch-shift" neighbors include tracks in different keys but with similar harmonic structure
- Mini curves show reasonable energy envelopes (not all flat or random noise)

- [ ] **Step 5: Re-run similarity only to verify cache**

```bash
python plots/build_dataset_stats.py \
  --source /home/kim/Projects/goa_crops \
  --output-dir /run/media/kim/Lehto \
  --skip-scalars --skip-curves
```
Expected: loads from `.ts_cache.npz`, prints track count, re-writes `feature_explorer_timeseries.js` only.

- [ ] **Step 6: Final commit**

```bash
git add plots/build_dataset_stats.py tests/test_build_dataset_stats.py plots/feature_explorer.html
git commit -m "feat: timeseries explorer integration complete — build_dataset_stats, similarity, HTML panels"
```
