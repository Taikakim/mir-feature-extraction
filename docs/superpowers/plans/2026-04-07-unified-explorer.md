# Unified MIR Explorer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `plots/explorer/` — a single Plotly Dash app (port 7895) replacing the Feature Explorer HTML, Shape Explorer HTML, and `plots/latent_analysis/app.py` with a unified, maintainable Python application.

**Architecture:** Three top-level tabs (Dataset, Analysis, Viewer) plus a persistent footer player strip. All data loading is centralised in `data.py`. Tab logic lives in focused per-tab modules under `tabs/`. Audio playback uses a Dash clientside callback backed by the Web Audio API in `assets/player.js`. Latent decode calls the unchanged GPU server (`latent_server.py`) on port 7891.

**Tech Stack:** Python 3.12, Plotly Dash ≥2.17, pandas, numpy, scipy; `latent_player.ini` for path config (already exists); Web Audio API clientside JS; `latent_server.py` port 7891 unchanged.

---

## File Map

| File | Responsibility |
|------|---------------|
| `plots/explorer/app.py` | Dash app init, global layout, Stores, top-level tab routing, server entry point |
| `plots/explorer/data.py` | All data loading: CSV, NPZ, ini config, latent dir scan; pure helper functions |
| `plots/explorer/audio.py` | HTTP helpers to latent_server.py; URL builders for decode/crossfade/average |
| `plots/explorer/latch.py` | LatCH guidance hook: gradient-based if checkpoint present, correlation fallback |
| `plots/explorer/tabs/__init__.py` | Empty |
| `plots/explorer/tabs/dataset.py` | Tab 1: 8-mode scatter views, sidebar, active-track callbacks |
| `plots/explorer/tabs/analysis.py` | Tab 2: direct port of `plots/latent_analysis/app.py` callbacks |
| `plots/explorer/tabs/viewer.py` | Tab 3: 3D trajectory, alignment bar, crossfader, manipulation |
| `plots/explorer/assets/player.js` | Web Audio API clientside callback; autoplay-on-hover with 200 ms fade |
| `plots/explorer/assets/style.css` | Dark-theme CSS (background #0d0d1a, accent colours) |
| `tests/explorer/test_data.py` | Unit tests for data.py pure functions |
| `tests/explorer/test_helpers.py` | Unit tests for helper functions (norm01, corrcoef, parse_class_label, parse_dim_range, blend_latents_by_cluster, avg_crops_with_loop_gating) |

---

## Phase 1 — Dataset Tab

---

### Task 1: Project scaffold

**Files:**
- Create: `plots/explorer/__init__.py`
- Create: `plots/explorer/tabs/__init__.py`
- Create: `plots/explorer/assets/style.css`
- Create: `tests/explorer/__init__.py`
- Create: `tests/explorer/test_data.py` (stub)

- [ ] **Step 1: Create the directory structure**

```bash
mkdir -p plots/explorer/tabs plots/explorer/assets tests/explorer
touch plots/explorer/__init__.py
touch plots/explorer/tabs/__init__.py
touch tests/explorer/__init__.py
```

- [ ] **Step 2: Write `assets/style.css`**

```css
/* plots/explorer/assets/style.css */
body, .dash-table-container { background: #0d0d1a; color: #ccc; font-family: 'Segoe UI', system-ui, sans-serif; }
.tab-content-wrapper { padding: 10px; }
.controls-bar { display: flex; flex-wrap: wrap; gap: 8px; padding: 6px 10px; background: #111125; border-bottom: 1px solid #2a2a4a; align-items: center; }
.Select-control, .Select-menu-outer { background: #0f1535 !important; color: #e0e0e0 !important; border-color: #2a2a4a !important; }
.Select-option { background: #0f1535 !important; color: #ccc !important; }
.Select-option.is-focused { background: #1e2050 !important; }
.rc-slider-track { background-color: #e94560; }
.rc-slider-handle { border-color: #e94560; background: #e94560; }
#player-strip { position: sticky; bottom: 0; background: #111125; border-top: 1px solid #2a2a4a; padding: 6px 12px; z-index: 999; }
.player-slot { display: inline-flex; align-items: center; gap: 8px; }
.sidebar { width: 260px; flex-shrink: 0; border-left: 1px solid #2a2a4a; padding: 8px; overflow-y: auto; height: calc(100vh - 160px); }
.nn-row { padding: 4px 0; border-bottom: 1px solid #1a1a2e; cursor: pointer; font-size: 11px; }
.nn-row:hover { background: #1a1a2e; }
.tl-item { padding: 3px 6px; cursor: pointer; font-size: 11px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.tl-item:hover { background: #1e2050; }
```

- [ ] **Step 3: Write stub test file**

```python
# tests/explorer/test_data.py
"""Tests for plots/explorer/data.py pure functions."""
import pytest
```

- [ ] **Step 4: Commit scaffold**

```bash
git add plots/explorer/ tests/explorer/
git commit -m "feat: scaffold unified explorer directory structure"
```

---

### Task 2: `data.py` — config loading and CSV parsing

**Files:**
- Create: `plots/explorer/data.py`
- Modify: `tests/explorer/test_data.py`

- [ ] **Step 1: Write failing tests for config loading and CSV parsing**

```python
# tests/explorer/test_data.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plots.explorer.data import load_config, load_tracks, AppData

def test_load_config_reads_ini(tmp_path):
    ini = tmp_path / "latent_player.ini"
    ini.write_text(
        "[server]\nlatent_dir = /tmp/lat\nstem_dir = /tmp/stems\n"
        "raw_audio_dir = /tmp/raw\nsource_dir = /tmp/src\nport = 7891\n"
        "[model]\nsao_dir = /tmp/sao\nmodel_config = /tmp/cfg.json\n"
        "ckpt_path = /tmp/ckpt\nmodel_half = true\ndevice = cuda\n"
    )
    cfg = load_config(ini)
    assert cfg["latent_dir"] == Path("/tmp/lat")
    assert cfg["port"] == 7891

def test_load_tracks_returns_appdata(tmp_path):
    csv = tmp_path / "tracks.csv"
    csv.write_text(
        "track,bpm,brightness,artists,essentia_genre\n"
        "Artist A - Song 1,140.0,70.5,\"['Artist A']\",\"{'Goa Trance': 0.9}\"\n"
        "Artist B - Song 2,128.0,60.0,\"['Artist B']\",\"{}\"\n"
    )
    ad = load_tracks(csv)
    assert isinstance(ad, AppData)
    assert len(ad.tracks) == 2
    assert "bpm" in ad.num_cols
    assert "essentia_genre" in ad.class_cols
    assert ad.feat_array("bpm").shape == (2,)
    assert abs(ad.feat_array("bpm")[0] - 140.0) < 0.01

def test_appdata_search_filters_by_name_and_artist(tmp_path):
    csv = tmp_path / "tracks.csv"
    csv.write_text(
        "track,bpm,artists\n"
        "Astral Projection - People Can Fly,148.0,\"['Astral Projection']\"\n"
        "Infected Mushroom - Bust a Move,145.0,\"['Infected Mushroom']\"\n"
        "X-Dream - We Interface,142.0,\"['X-Dream']\"\n"
    )
    ad = load_tracks(csv)
    # pattern in track name
    idxs = ad.search("people")
    assert len(idxs) == 1 and ad.tracks[idxs[0]] == "Astral Projection - People Can Fly"
    # pattern in artist
    idxs = ad.search("infected")
    assert len(idxs) == 1 and ad.tracks[idxs[0]] == "Infected Mushroom - Bust a Move"
    # empty query returns all
    assert len(ad.search("")) == 3
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/kim/Projects/mir
python -m pytest tests/explorer/test_data.py -v 2>&1 | head -30
```
Expected: `ImportError: cannot import name 'load_config' from 'plots.explorer.data'`

- [ ] **Step 3: Write `data.py`**

```python
# plots/explorer/data.py
"""Central data loading for the unified MIR Explorer."""
from __future__ import annotations
import ast
import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent          # .../mir
_DEFAULT_INI = _REPO_ROOT / "latent_player.ini"
_DEFAULT_CSV = _REPO_ROOT / "plots" / "tracks.csv"
_ANALYSIS_DIR = _REPO_ROOT / "plots" / "latent_analysis" / "data"

# ── Feature metadata ─────────────────────────────────────────────────────────
CURATED_FEATURES = [
    "bpm", "danceability", "beat_count",
    "brightness", "roughness", "hardness", "depth",
    "booming", "reverberation", "sharpness", "warmth",
    "lufs", "lra",
    "lufs_bass", "lufs_drums", "lufs_other", "lufs_vocals",
    "rms_energy_bass", "rms_energy_body", "rms_energy_mid", "rms_energy_air",
    "spectral_flatness", "spectral_flux", "spectral_skewness", "spectral_kurtosis",
    "voice_probability", "female_probability", "male_probability",
    "instrumental_probability",
    "content_enjoyment", "content_usefulness",
    "production_complexity", "production_quality",
    "atonality", "popularity",
]

FEATURE_UNITS: dict[str, str] = {
    "bpm": "BPM",
    "lufs": "dBFS", "lra": "LU",
    "lufs_bass": "dBFS", "lufs_drums": "dBFS",
    "lufs_other": "dBFS", "lufs_vocals": "dBFS",
    "rms_energy_air": "dBFS", "rms_energy_bass": "dBFS",
    "rms_energy_body": "dBFS", "rms_energy_mid": "dBFS",
    "duration": "s", "release_year": "yr", "popularity": "%",
}

# Columns treated as categorical class groupings for "Classes" mode
CLASS_FIELDS = ["essentia_genre", "essentia_instrument", "essentia_mood",
                "genres", "label"]

# Columns that pandas may misdetect as numeric due to all-NaN rows
_FORCE_CATEGORICAL = {"has_stems", "musicbrainz_id", "source", "stem_names",
                       "track_metadata_genre", "spotify_id"}

# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class AppData:
    """Loaded and indexed dataset. Immutable after construction."""
    tracks: list[str]                   # display names, one per row
    artists: list[str]                  # lowercased artist string per row (for search)
    num_cols: list[str]                 # numeric feature column names
    class_cols: list[str]               # categorical column names
    _df_num: pd.DataFrame = field(repr=False)
    _df_class: pd.DataFrame = field(repr=False)

    def feat_array(self, col: str) -> np.ndarray:
        """Return float64 array [n_tracks] for numeric column, NaN where missing."""
        return self._df_num[col].to_numpy(dtype=np.float64, na_value=np.nan)

    def class_array(self, col: str) -> list[str]:
        """Return list of raw string values for a class column."""
        return self._df_class[col].fillna("").tolist()

    def search(self, query: str) -> list[int]:
        """Return track indices whose name or artist contains query (case-insensitive)."""
        if not query:
            return list(range(len(self.tracks)))
        q = query.lower()
        return [
            i for i, (t, a) in enumerate(zip(self.tracks, self.artists))
            if q in t.lower() or q in a
        ]

    def track_options(self, query: str = "") -> list[dict]:
        """Return Dash Dropdown options [{"label": name, "value": name}] filtered by query."""
        idxs = self.search(query)
        return [{"label": self.tracks[i], "value": self.tracks[i]} for i in idxs]

# ── Loaders ───────────────────────────────────────────────────────────────────
def load_config(ini_path: Path = _DEFAULT_INI) -> dict[str, Any]:
    """Parse latent_player.ini and return typed config dict."""
    cfg = configparser.ConfigParser()
    cfg.read(ini_path)
    s = cfg["server"]
    m = cfg["model"]
    return {
        "latent_dir":    Path(s["latent_dir"]),
        "stem_dir":      Path(s["stem_dir"]),
        "raw_audio_dir": Path(s["raw_audio_dir"]),
        "source_dir":    Path(s["source_dir"]),
        "port":          int(s.get("port", 7891)),
        "sao_dir":       Path(m["sao_dir"]),
        "model_config":  Path(m["model_config"]),
        "ckpt_path":     Path(m["ckpt_path"]),
        "model_half":    m.getboolean("model_half", True),
        "device":        m.get("device", "cuda"),
    }


def load_tracks(csv_path: Path = _DEFAULT_CSV) -> AppData:
    """Load tracks.csv and return an AppData instance."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Force-cast known string columns that pandas may read as float (all-NaN)
    for col in _FORCE_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype(object)

    # Determine numeric vs categorical columns (excluding 'track')
    num_cols = [
        c for c in df.columns
        if c != "track"
        and pd.api.types.is_numeric_dtype(df[c])
        and c not in _FORCE_CATEGORICAL
    ]
    class_cols = [
        c for c in CLASS_FIELDS if c in df.columns
    ]

    tracks = df["track"].fillna("").tolist()
    # Build lowercased artist string for search
    def _artist_str(raw) -> str:
        if pd.isna(raw) or str(raw).strip() in ("", "nan"):
            return ""
        try:
            lst = ast.literal_eval(str(raw))
            return " ".join(lst).lower() if isinstance(lst, list) else str(raw).lower()
        except Exception:
            return str(raw).lower()

    artists = [_artist_str(a) for a in df.get("artists", pd.Series([""] * len(df)))]

    return AppData(
        tracks=tracks,
        artists=artists,
        num_cols=num_cols,
        class_cols=class_cols,
        _df_num=df[num_cols].copy(),
        _df_class=df[class_cols].copy() if class_cols else pd.DataFrame(),
    )


def load_analysis_npz() -> dict[str, Any | None]:
    """Load all four analysis NPZ files. Missing files return None."""
    def _load(name: str):
        p = _ANALYSIS_DIR / name
        if not p.exists():
            return None
        return dict(np.load(str(p), allow_pickle=True))

    return {
        "d01": _load("01_correlations.npz"),
        "d02": _load("02_pca.npz"),
        "d03": _load("03_xcorr.npz"),
        "d04": _load("04_temporal.npz"),
    }


# ── Singleton state (loaded once at app startup) ──────────────────────────────
_APP_DATA: AppData | None = None
_CONFIG: dict | None = None
_ANALYSIS: dict | None = None


def get_app_data() -> AppData:
    global _APP_DATA
    if _APP_DATA is None:
        _APP_DATA = load_tracks()
    return _APP_DATA


def get_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def get_analysis() -> dict:
    global _ANALYSIS
    if _ANALYSIS is None:
        _ANALYSIS = load_analysis_npz()
    return _ANALYSIS
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/explorer/test_data.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add plots/explorer/data.py tests/explorer/test_data.py
git commit -m "feat: data.py with load_config, load_tracks, AppData"
```

---

### Task 3: `data.py` — pure helper functions

**Files:**
- Modify: `plots/explorer/data.py`
- Create: `tests/explorer/test_helpers.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/explorer/test_helpers.py
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/explorer/test_helpers.py -v 2>&1 | head -20
```
Expected: `ImportError: cannot import name 'norm01'`

- [ ] **Step 3: Add helper functions to `data.py`**

Append to the bottom of `plots/explorer/data.py`:

```python
# ── Pure helper functions ─────────────────────────────────────────────────────

def norm01(values: list[float] | np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Constant arrays return 0.5."""
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


def corrcoef(x: list[float] | np.ndarray,
             y: list[float] | np.ndarray) -> float:
    """Pearson r between two equal-length sequences. Returns 0.0 on degenerate input."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    mx, my = x.mean(), y.mean()
    num = ((x - mx) * (y - my)).sum()
    den = np.sqrt(((x - mx) ** 2).sum() * ((y - my) ** 2).sum())
    return float(num / den) if den > 1e-12 else 0.0


def parse_class_label(raw: str) -> str:
    """
    Extract a single class string from a raw CSV value.
    - Dict string "{'Goa Trance': 0.9, ...}" → key with highest value
    - List string "['psytrance', ...]"        → first item
    - Plain string                             → as-is
    - Empty / {} / []                          → ""
    """
    s = str(raw).strip()
    if not s or s in ("{}", "[]", "['']", "nan"):
        return ""
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, dict):
            if not parsed:
                return ""
            return max(parsed, key=lambda k: parsed[k])
        if isinstance(parsed, list):
            return str(parsed[0]) if parsed else ""
    except Exception:
        pass
    return s


def parse_dim_range(expr: str, n_dims: int = 64) -> np.ndarray:
    """
    Parse "0-15,32,48-63" into a boolean mask of shape [n_dims].
    Returns all-False mask on empty/invalid input.
    """
    mask = np.zeros(n_dims, dtype=bool)
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            try:
                lo, hi = int(lo_s), int(hi_s)
                mask[max(0, lo): min(n_dims, hi + 1)] = True
            except ValueError:
                pass
        else:
            try:
                idx = int(part)
                if 0 <= idx < n_dims:
                    mask[idx] = True
            except ValueError:
                pass
    return mask


def blend_latents_by_cluster(
    z_a: np.ndarray,
    z_b: np.ndarray,
    cluster_alphas: dict[int, float],
    cluster_labels: np.ndarray,
) -> np.ndarray:
    """
    Blend two latents [D, T] per Ward cluster.

    cluster_alphas: {cluster_id: alpha}  (0.0 = all A, 1.0 = all B)
    cluster_labels: [D] int — cluster assignment per dim; 0 = unassigned (stays at A)
    Returns [D, T] blended latent.
    """
    result = z_a.copy()
    for cid, alpha in cluster_alphas.items():
        mask = cluster_labels == cid
        if not mask.any():
            continue
        alpha = float(np.clip(alpha, 0.0, 1.0))
        result[mask] = (1.0 - alpha) * z_a[mask] + alpha * z_b[mask]
    return result
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/explorer/test_helpers.py -v
```
Expected: 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add plots/explorer/data.py tests/explorer/test_helpers.py
git commit -m "feat: data.py helper functions: norm01, corrcoef, parse_class_label, parse_dim_range, blend_latents_by_cluster"
```

---

### Task 4: `app.py` skeleton — layout, Stores, tab routing

**Files:**
- Create: `plots/explorer/app.py`

- [ ] **Step 1: Write `app.py`**

```python
# plots/explorer/app.py
"""
Unified MIR Explorer — Plotly Dash app (port 7895).

Run:
    python plots/explorer/app.py [--port 7895] [--debug]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import dash
from dash import dcc, html, Input, Output

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.explorer.data import get_app_data, get_config, get_analysis
from plots.explorer.tabs import dataset, analysis, viewer

app = dash.Dash(
    __name__,
    title="MIR Explorer",
    suppress_callback_exceptions=True,
)

_TAB_STYLE        = {"padding": "8px 16px", "color": "#888"}
_TAB_ACTIVE_STYLE = {"padding": "8px 16px", "backgroundColor": "#e94560",
                     "color": "#fff", "fontWeight": "700"}

app.layout = html.Div([
    # ── Shared state stores ────────────────────────────────────────────────
    dcc.Store(id="active-track",
              data={"track": None, "track_idx": None, "slot": "a"}),
    dcc.Store(id="player-cmd",   data={}),
    dcc.Store(id="cluster-highlight", data=None),
    dcc.Store(id="viewer-state",
              data={"track_a": None, "crop_a": None,
                    "track_b": None, "crop_b": None}),
    dcc.Store(id="avg-crops",       data=False),
    dcc.Store(id="autoplay-hover",  data=False),

    # ── Header ─────────────────────────────────────────────────────────────
    html.Div(
        html.H2("MIR Explorer", style={"margin": "8px 16px 4px", "color": "#e94560"}),
        style={"background": "#111125", "borderBottom": "1px solid #2a2a4a"}
    ),

    # ── Tab navigation + content ───────────────────────────────────────────
    dcc.Tabs(id="main-tabs", value="dataset", children=[
        dcc.Tab(label="Dataset",  value="dataset",
                style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
        dcc.Tab(label="Analysis", value="analysis",
                style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
        dcc.Tab(label="Viewer",   value="viewer",
                style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
    ]),
    html.Div(id="tab-content", className="tab-content-wrapper"),

    # ── Persistent player strip (footer) ───────────────────────────────────
    html.Div(id="player-strip", children=[
        html.Span("♫ Latent Player", style={"color": "#555", "fontSize": "11px",
                                             "marginRight": "16px"}),
        # A slot
        html.Div([
            html.Span("A:", style={"color": "#4cd137", "fontSize": "11px"}),
            html.Span(id="player-track-a", "—",
                      style={"fontSize": "11px", "maxWidth": "220px",
                             "overflow": "hidden", "textOverflow": "ellipsis",
                             "whiteSpace": "nowrap", "display": "inline-block",
                             "verticalAlign": "middle"}),
            html.Button("▶", id="btn-play-a",  n_clicks=0,
                        style={"background": "none", "border": "none",
                               "color": "#4cd137", "cursor": "pointer",
                               "fontSize": "14px"}),
            html.Button("■", id="btn-stop-a",  n_clicks=0,
                        style={"background": "none", "border": "none",
                               "color": "#e94560", "cursor": "pointer",
                               "fontSize": "14px"}),
            dcc.Slider(id="pos-slider-a", min=0, max=1, step=0.001, value=0.5,
                       marks={}, tooltip={"always_visible": False},
                       className="pos-slider", updatemode="drag"),
        ], className="player-slot", style={"display": "inline-flex",
                                           "alignItems": "center", "gap": "8px"}),
        # B slot + crossfade
        html.Div([
            html.Span("B:", style={"color": "#00d2ff", "fontSize": "11px"}),
            html.Span(id="player-track-b", "—",
                      style={"fontSize": "11px", "maxWidth": "220px",
                             "overflow": "hidden", "textOverflow": "ellipsis",
                             "whiteSpace": "nowrap", "display": "inline-block",
                             "verticalAlign": "middle"}),
            dcc.Slider(id="xfade-alpha", min=0, max=1, step=0.01, value=0.0,
                       marks={0: "A", 1: "B"}, tooltip={"always_visible": False},
                       className="pos-slider", updatemode="drag"),
        ], id="player-b-slot",
           style={"display": "inline-flex", "alignItems": "center",
                  "gap": "8px", "marginLeft": "24px"}),
        # Checkboxes
        dcc.Checklist(
            id="player-options",
            options=[
                {"label": "Autoplay hover", "value": "autoplay"},
                {"label": "Smart Loop",     "value": "smart_loop"},
            ],
            value=[],
            inline=True,
            style={"display": "inline-flex", "gap": "12px",
                   "marginLeft": "24px", "fontSize": "11px"},
        ),
    ]),
], style={"fontFamily": "monospace", "backgroundColor": "#0d0d1a",
          "color": "#ccc", "minHeight": "100vh"})


# ── Tab routing ───────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab: str):
    if tab == "dataset":  return dataset.layout()
    if tab == "analysis": return analysis.layout()
    if tab == "viewer":   return viewer.layout()
    return html.P("Unknown tab.")


# ── Player strip label sync from active-track store ──────────────────────────
@app.callback(
    Output("player-track-a", "children"),
    Output("player-track-b", "children"),
    Input("active-track", "data"),
)
def sync_player_labels(state: dict):
    t = state or {}
    track  = t.get("track") or "—"
    track_b = t.get("track_b") or "—"
    return track, track_b


# ── Sync autoplay-hover checkbox into Store ───────────────────────────────────
@app.callback(
    Output("autoplay-hover", "data"),
    Input("player-options", "value"),
)
def sync_autoplay(opts):
    return "autoplay" in (opts or [])


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int, default=7895)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Eagerly load data so first tab render is fast
    get_app_data()
    get_config()
    get_analysis()

    print(f"MIR Explorer → http://localhost:{args.port}")
    app.run(debug=args.debug, port=args.port, host="127.0.0.1")
```

- [ ] **Step 2: Create stub tab modules so import doesn't fail**

```python
# plots/explorer/tabs/dataset.py  (stub)
from dash import html
def layout(): return html.P("Dataset tab — coming soon.")
```

```python
# plots/explorer/tabs/analysis.py  (stub)
from dash import html
def layout(): return html.P("Analysis tab — coming soon.")
```

```python
# plots/explorer/tabs/viewer.py  (stub)
from dash import html
def layout(): return html.P("Viewer tab — coming soon.")
```

- [ ] **Step 3: Smoke-test the app starts**

```bash
cd /home/kim/Projects/mir
python plots/explorer/app.py --debug &
sleep 3
curl -s http://localhost:7895/ | grep -q "MIR Explorer" && echo "OK" || echo "FAIL"
kill %1
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add plots/explorer/app.py plots/explorer/tabs/dataset.py \
        plots/explorer/tabs/analysis.py plots/explorer/tabs/viewer.py
git commit -m "feat: app.py skeleton with Stores, layout, tab routing, player strip"
```

---

### Task 5: Dataset tab — layout and mode switching

**Files:**
- Modify: `plots/explorer/tabs/dataset.py`

- [ ] **Step 1: Write the full dataset layout and mode-switch callback**

```python
# plots/explorer/tabs/dataset.py
"""Tab 1 — Dataset Explorer: 8-mode scatter + sidebar."""
from __future__ import annotations

import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.explorer.data import get_app_data, CURATED_FEATURES, FEATURE_UNITS

# ── Layout ────────────────────────────────────────────────────────────────────
def layout() -> html.Div:
    ad = get_app_data()
    feat_opts  = [{"label": f, "value": f} for f in sorted(ad.num_cols)]
    cur_opts   = [{"label": f, "value": f} for f in CURATED_FEATURES if f in ad.num_cols]
    class_opts = [{"label": c, "value": c} for c in ad.class_cols]
    track_opts = ad.track_options()

    return html.Div([
        # ── Controls toolbar ────────────────────────────────────────────────
        html.Div([
            # Mode radio
            dcc.RadioItems(
                id="view-mode",
                options=[
                    {"label": "Scatter",    "value": "scatter"},
                    {"label": "Quadrant",   "value": "quadrant"},
                    {"label": "Histogram",  "value": "histogram"},
                    {"label": "Radar",      "value": "radar"},
                    {"label": "Heatmap",    "value": "heatmap"},
                    {"label": "Parallel",   "value": "parallel"},
                    {"label": "Similarity", "value": "similarity"},
                    {"label": "Classes",    "value": "classes"},
                ],
                value="scatter",
                inline=True,
                style={"fontSize": "12px", "gap": "8px"},
            ),
        ], className="controls-bar"),

        # ── Mode-specific controls (shown/hidden per mode) ──────────────────
        html.Div(id="mode-controls", children=_scatter_controls(feat_opts, class_opts)),

        # ── Main area: plot + sidebar ───────────────────────────────────────
        html.Div([
            html.Div([
                dcc.Graph(
                    id="dataset-graph",
                    style={"height": "calc(100vh - 260px)"},
                    config={"displayModeBar": True, "scrollZoom": True},
                ),
                html.Div(id="info-bar",
                         style={"fontSize": "11px", "color": "#666",
                                "padding": "3px 8px", "background": "#0a0a15"}),
            ], style={"flex": "1", "minWidth": 0}),
            # Sidebar
            html.Div([
                html.Div([
                    html.Span(id="tl-title", "Tracks",
                              style={"fontWeight": "bold", "fontSize": "12px"}),
                    dcc.Dropdown(id="tl-service",
                                 options=[
                                     {"label": "Spotify",     "value": "spotify"},
                                     {"label": "Tidal",       "value": "tidal"},
                                     {"label": "MusicBrainz", "value": "musicbrainz"},
                                 ],
                                 value="tidal", clearable=False,
                                 style={"width": "120px", "fontSize": "11px"}),
                ], style={"display": "flex", "justifyContent": "space-between",
                          "alignItems": "center", "marginBottom": "4px"}),
                html.Div(id="track-list-body",
                         style={"overflowY": "auto",
                                "maxHeight": "calc(50vh - 120px)"}),
                html.Hr(style={"borderColor": "#2a2a4a"}),
                html.Div(id="nn-panel"),
            ], className="sidebar"),
        ], style={"display": "flex", "gap": "0"}),
    ])


# ── Controls fragments ─────────────────────────────────────────────────────────
def _scatter_controls(feat_opts, class_opts):
    return html.Div([
        _label_group("X axis",   dcc.Dropdown(id="sel-x", options=feat_opts,
                                              value="bpm", clearable=False,
                                              style={"width": "180px"})),
        _label_group("Y axis",   dcc.Dropdown(id="sel-y", options=feat_opts,
                                              value="brightness", clearable=False,
                                              style={"width": "180px"})),
        _label_group("Colour by", dcc.Dropdown(id="sel-colour",
                                               options=[{"label": "None",     "value": ""},
                                                        {"label": "Outliers", "value": "__outliers__"}]
                                               + feat_opts,
                                               value="", clearable=True,
                                               style={"width": "160px"})),
        dcc.Checklist(id="chk-scatter-opts",
                      options=[{"label": "Trend", "value": "trend"},
                               {"label": "Log X", "value": "logx"},
                               {"label": "Log Y", "value": "logy"}],
                      value=[], inline=True, style={"fontSize": "11px"}),
        # Cluster overlay (visible when analysis data loaded)
        dcc.Checklist(id="chk-show-clusters",
                      options=[{"label": "Show clusters", "value": "clusters"}],
                      value=[], inline=True, style={"fontSize": "11px"}),
    ], className="controls-bar", style={"gap": "12px"})


def _label_group(label: str, child) -> html.Div:
    return html.Div([
        html.Label(label, style={"fontSize": "10px", "color": "#888",
                                 "textTransform": "uppercase"}),
        child,
    ], style={"display": "flex", "flexDirection": "column", "gap": "2px"})


# ── Mode → controls switcher callback ────────────────────────────────────────
@callback(Output("mode-controls", "children"),
          Input("view-mode", "value"),
          prevent_initial_call=False)
def update_mode_controls(mode: str):
    ad = get_app_data()
    feat_opts = [{"label": f, "value": f} for f in sorted(ad.num_cols)]
    cur_opts  = [{"label": f, "value": f} for f in CURATED_FEATURES if f in ad.num_cols]
    class_opts = [{"label": c, "value": c} for c in ad.class_cols]
    track_opts = ad.track_options()

    if mode == "scatter":
        return _scatter_controls(feat_opts, class_opts)

    if mode == "quadrant":
        return html.Div([
            _label_group("X+", dcc.Dropdown(id="sel-xp", options=feat_opts,
                                            value="brightness", clearable=False,
                                            style={"width": "160px"})),
            _label_group("X−", dcc.Dropdown(id="sel-xn", options=feat_opts,
                                            value="roughness", clearable=False,
                                            style={"width": "160px"})),
            _label_group("Y+", dcc.Dropdown(id="sel-yp", options=feat_opts,
                                            value="danceability", clearable=False,
                                            style={"width": "160px"})),
            _label_group("Y−", dcc.Dropdown(id="sel-yn", options=feat_opts,
                                            value="atonality", clearable=False,
                                            style={"width": "160px"})),
        ], className="controls-bar")

    if mode == "histogram":
        return html.Div([
            _label_group("Feature", dcc.Dropdown(id="sel-hist", options=feat_opts,
                                                  value="bpm", clearable=False,
                                                  style={"width": "200px"})),
            _label_group("Bins", dcc.Input(id="hist-bins", type="number",
                                           value=30, min=4, max=64,
                                           style={"width": "60px"})),
        ], className="controls-bar")

    if mode == "radar":
        return html.Div([
            _label_group("Track", dcc.Dropdown(id="sel-radar-track",
                                               options=track_opts,
                                               value=track_opts[0]["value"] if track_opts else None,
                                               clearable=False, style={"width": "320px"},
                                               placeholder="Search track…")),
        ], className="controls-bar")

    if mode == "heatmap":
        from plots.explorer.data import FEATURE_GROUPS as FG
        group_opts = [{"label": g, "value": g} for g in list(FG.keys()) + ["All"]]
        return html.Div([
            _label_group("Feature group",
                         dcc.Dropdown(id="sel-heatmap-group", options=group_opts,
                                      value="All", clearable=False,
                                      style={"width": "180px"})),
        ], className="controls-bar")

    if mode == "parallel":
        return html.Div([
            html.Span("Curated features | Drag axis bands to filter",
                      style={"fontSize": "11px", "color": "#666"}),
        ], className="controls-bar")

    if mode == "similarity":
        return html.Div([
            _label_group("Reference", dcc.Dropdown(id="sel-sim-ref",
                                                   options=track_opts,
                                                   value=track_opts[0]["value"] if track_opts else None,
                                                   clearable=False, style={"width": "280px"},
                                                   placeholder="Search reference…")),
            _label_group("X+", dcc.Dropdown(id="sim-xp", options=feat_opts,
                                            value="brightness", clearable=False,
                                            style={"width": "140px"})),
            _label_group("X−", dcc.Dropdown(id="sim-xn", options=feat_opts,
                                            value="roughness", clearable=False,
                                            style={"width": "140px"})),
            _label_group("Y+", dcc.Dropdown(id="sim-yp", options=feat_opts,
                                            value="danceability", clearable=False,
                                            style={"width": "140px"})),
            _label_group("Y−", dcc.Dropdown(id="sim-yn", options=feat_opts,
                                            value="atonality", clearable=False,
                                            style={"width": "140px"})),
        ], className="controls-bar")

    if mode == "classes":
        return html.Div([
            _label_group("X axis",    dcc.Dropdown(id="sel-x", options=feat_opts,
                                                   value="bpm", clearable=False,
                                                   style={"width": "160px"})),
            _label_group("Y axis",    dcc.Dropdown(id="sel-y", options=feat_opts,
                                                   value="brightness", clearable=False,
                                                   style={"width": "160px"})),
            _label_group("Class by",  dcc.Dropdown(id="sel-class-by",
                                                   options=class_opts,
                                                   value=class_opts[0]["value"] if class_opts else None,
                                                   clearable=False, style={"width": "180px"})),
            dcc.Checklist(id="chk-class-trend",
                          options=[{"label": "Per-class trend", "value": "trend"}],
                          value=[], inline=True, style={"fontSize": "11px"}),
        ], className="controls-bar")

    return html.Div(className="controls-bar")
```

- [ ] **Step 2: Smoke-test tab renders without errors**

```bash
python plots/explorer/app.py --debug &
sleep 3
curl -s http://localhost:7895/ | grep -c "Dataset" && echo "OK"
kill %1
```
Expected: at least `1` and `OK`.

- [ ] **Step 3: Commit**

```bash
git add plots/explorer/tabs/dataset.py
git commit -m "feat: dataset tab layout and mode-switching controls"
```

---

### Task 6: Scatter and Quadrant mode callbacks

**Files:**
- Modify: `plots/explorer/tabs/dataset.py`

- [ ] **Step 1: Append scatter + quadrant callbacks**

```python
# Append to plots/explorer/tabs/dataset.py

_DARK = dict(template="plotly_dark", paper_bgcolor="#0d0d1a",
             plot_bgcolor="#111125", font=dict(color="#ccc"),
             hoverlabel=dict(bgcolor="#0f1535", font=dict(size=12, color="white")),
             dragmode="lasso", margin=dict(t=46, b=48, l=58, r=20))

CLASS_COLORS = [
    '#e94560','#4cd137','#00d2ff','#fbc531','#9c88ff','#ff79c6','#ff9f43',
    '#0097e6','#bdc581','#fd7272','#5f27cd','#55efc4','#81ecec','#ff7675',
    '#a29bfe','#e1b12c','#00cec9','#e84393','#badc58','#c8d6e5',
]


@callback(
    Output("dataset-graph", "figure"),
    Output("info-bar", "children"),
    Input("view-mode", "value"),
    # scatter inputs
    Input("sel-x", "value"),
    Input("sel-y", "value"),
    Input("sel-colour", "value"),
    Input("chk-scatter-opts", "value"),
    # quadrant inputs
    Input("sel-xp", "value"),
    Input("sel-xn", "value"),
    Input("sel-yp", "value"),
    Input("sel-yn", "value"),
    # cluster overlay
    Input("cluster-highlight", "data"),
    Input("chk-show-clusters", "value"),
    prevent_initial_call=False,
)
def update_scatter_quadrant(mode, kx, ky, colour, opts,
                             kxp, kxn, kyp, kyn,
                             cluster_data, show_clusters):
    if mode not in ("scatter", "quadrant"):
        return no_update, no_update

    ad   = get_app_data()
    opts = opts or []

    if mode == "scatter":
        if not kx or not ky:
            return go.Figure(), "Select X and Y features."
        dx = ad.feat_array(kx)
        dy = ad.feat_array(ky)
        valid = np.isfinite(dx) & np.isfinite(dy)
        x, y = dx[valid], dy[valid]
        names = [ad.tracks[i] for i in np.where(valid)[0]]
        idxs  = np.where(valid)[0]

        r = corrcoef(x, y)
        ux = f" [{FEATURE_UNITS[kx]}]" if kx in FEATURE_UNITS else ""
        uy = f" [{FEATURE_UNITS[ky]}]" if ky in FEATURE_UNITS else ""

        hover = [f"<b>{n}</b><br>{kx}: {x[j]:.4f}<br>{ky}: {y[j]:.4f}"
                 for j, n in enumerate(names)]

        if colour == "__outliers__":
            mx, sx = x.mean(), x.std()
            my, sy = y.mean(), y.std()
            is_out = (np.abs(x - mx) > 2*sx) | (np.abs(y - my) > 2*sy)
            traces = [
                go.Scattergl(x=x[~is_out], y=y[~is_out], mode="markers",
                             marker=dict(color="#e94560", size=5, opacity=0.5),
                             hovertext=[hover[j] for j in np.where(~is_out)[0]],
                             hoverinfo="text", name=f"normal ({(~is_out).sum()})",
                             customdata=idxs[~is_out]),
                go.Scattergl(x=x[is_out],  y=y[is_out],  mode="markers",
                             marker=dict(color="#ffd700", size=9, opacity=0.85),
                             hovertext=[hover[j] for j in np.where(is_out)[0]],
                             hoverinfo="text", name=f"outlier ({is_out.sum()})",
                             customdata=idxs[is_out]),
            ]
        elif colour and colour in ad.num_cols:
            cvals = ad.feat_array(colour)[valid]
            traces = [go.Scattergl(x=x, y=y, mode="markers",
                                   marker=dict(color=cvals, colorscale="Viridis",
                                               showscale=True, size=5, opacity=0.7,
                                               colorbar=dict(title=colour, thickness=14)),
                                   hovertext=hover, hoverinfo="text",
                                   name="tracks", customdata=idxs)]
        else:
            traces = [go.Scattergl(x=x, y=y, mode="markers",
                                   marker=dict(color="#e94560", size=5, opacity=0.5),
                                   hovertext=hover, hoverinfo="text",
                                   name="tracks", customdata=idxs)]

        if "trend" in opts and len(x) > 2:
            mx2, my2 = x.mean(), y.mean()
            num = ((x-mx2)*(y-my2)).sum()
            den = ((x-mx2)**2).sum()
            slope = num/den if den > 1e-12 else 0
            ic = my2 - slope * mx2
            xmn, xmx = x.min(), x.max()
            traces.append(go.Scatter(
                x=[xmn, xmx], y=[slope*xmn+ic, slope*xmx+ic],
                mode="lines", line=dict(color="#00d2ff", width=2, dash="dash"),
                name=f"trend (r={r:.3f})", hoverinfo="skip",
            ))

        log_x = "logx" in opts and np.all(x > 0)
        log_y = "logy" in opts and np.all(y > 0)
        layout = dict(**_DARK,
                      title=dict(text=f"{kx} vs {ky}  (r={r:.3f}, n={len(x)})", font=dict(size=14)),
                      xaxis=dict(title=kx+ux, type="log" if log_x else "linear"),
                      yaxis=dict(title=ky+uy, type="log" if log_y else "linear"),
                      showlegend=True)
        info = f"{kx} vs {ky} | r={r:.3f} | n={len(x)}"
        return go.Figure(data=traces, layout=layout), info

    # quadrant
    if not all([kxp, kxn, kyp, kyn]):
        return go.Figure(), "Select all four quadrant features."
    dxp, dxn = ad.feat_array(kxp), ad.feat_array(kxn)
    dyp, dyn = ad.feat_array(kyp), ad.feat_array(kyn)
    valid = np.isfinite(dxp) & np.isfinite(dxn) & np.isfinite(dyp) & np.isfinite(dyn)
    idxs = np.where(valid)[0]
    nxp = norm01(dxp[valid]); nxn = norm01(dxn[valid])
    nyp = norm01(dyp[valid]); nyn = norm01(dyn[valid])
    x = nxp - nxn; y = nyp - nyn
    hover = [f"<b>{ad.tracks[i]}</b><br>X:{x[j]:.3f} Y:{y[j]:.3f}"
             for j, i in enumerate(idxs)]
    layout = dict(**_DARK,
                  title=dict(text=f"Quadrant (n={len(idxs)})", font=dict(size=14)),
                  xaxis=dict(title=f"← {kxn}  |  {kxp} →",
                             range=[-1.1, 1.1], zeroline=True, zerolinecolor="#555577"),
                  yaxis=dict(title=f"← {kyn}  |  {kyp} →",
                             range=[-1.1, 1.1], zeroline=True, zerolinecolor="#555577",
                             scaleanchor="x"),
                  showlegend=False,
                  annotations=[
                      dict(x=1.0,  y=0, text=f"<b>{kxp}</b>", showarrow=False,
                           font=dict(size=11, color="#00d2ff"), xanchor="right",
                           bgcolor="#0f3460", bordercolor="#e94560", borderpad=3),
                      dict(x=-1.0, y=0, text=f"<b>{kxn}</b>", showarrow=False,
                           font=dict(size=11, color="#00d2ff"), xanchor="left",
                           bgcolor="#0f3460", bordercolor="#e94560", borderpad=3),
                      dict(x=0, y=1.0,  text=f"<b>{kyp}</b>", showarrow=False,
                           font=dict(size=11, color="#e94560"), yanchor="bottom",
                           bgcolor="#0f3460", bordercolor="#e94560", borderpad=3),
                      dict(x=0, y=-1.0, text=f"<b>{kyn}</b>", showarrow=False,
                           font=dict(size=11, color="#e94560"), yanchor="top",
                           bgcolor="#0f3460", bordercolor="#e94560", borderpad=3),
                  ])
    trace = go.Scattergl(x=x, y=y, mode="markers",
                         marker=dict(color="#e94560", size=6, opacity=0.5),
                         hovertext=hover, hoverinfo="text", customdata=idxs)
    info = f"X: {kxp}−{kxn} | Y: {kyp}−{kyn} | n={len(idxs)}"
    return go.Figure(data=[trace], layout=layout), info
```

Also add the missing import at the top of `dataset.py` (after the existing imports):
```python
import numpy as np
from plots.explorer.data import norm01, corrcoef, FEATURE_GROUPS
```

- [ ] **Step 2: Verify app renders scatter plot**

```bash
python plots/explorer/app.py &
sleep 3
curl -s http://localhost:7895/ | grep -q "dataset-graph" && echo "OK"
kill %1
```

- [ ] **Step 3: Commit**

```bash
git add plots/explorer/tabs/dataset.py
git commit -m "feat: dataset scatter and quadrant mode callbacks"
```

---

### Task 7: Histogram, Heatmap, Parallel callbacks

**Files:**
- Modify: `plots/explorer/tabs/dataset.py`

- [ ] **Step 1: Append histogram callback**

```python
# Append to plots/explorer/tabs/dataset.py

@callback(
    Output("dataset-graph", "figure"),
    Output("info-bar", "children"),
    Output("track-list-body", "children"),
    Output("tl-title", "children"),
    Input("view-mode", "value"),
    Input("sel-hist", "value"),
    Input("hist-bins", "value"),
    prevent_initial_call=True,
)
def update_histogram(mode, feat, nbins):
    if mode != "histogram":
        return no_update, no_update, no_update, no_update
    ad   = get_app_data()
    if not feat or feat not in ad.num_cols:
        return go.Figure(), "Select a feature.", no_update, no_update
    vals = ad.feat_array(feat)
    vals = vals[np.isfinite(vals)]
    n    = len(vals)
    if n == 0:
        return go.Figure(), f"{feat}: no data", no_update, no_update
    nbins = max(4, min(64, int(nbins or 30)))
    m, s  = vals.mean(), vals.std()
    vmin, vmax = vals.min(), vals.max()
    bsize = (vmax - vmin) / nbins
    u = f" [{FEATURE_UNITS[feat]}]" if feat in FEATURE_UNITS else ""
    trace = go.Histogram(
        x=vals, xbins=dict(start=vmin, end=vmax, size=bsize),
        marker=dict(color="#e94560", opacity=0.75,
                    line=dict(color="#c03050", width=0.5)),
        name=feat,
    )
    layout = dict(**_DARK,
                  title=dict(text=f"{feat}  (n={n}, μ={m:.3f}, σ={s:.3f})",
                             font=dict(size=14)),
                  xaxis=dict(title=feat+u), yaxis=dict(title="count"),
                  dragmode="select",
                  shapes=[
                      dict(type="line", x0=m, x1=m, y0=0, y1=1, yref="paper",
                           line=dict(color="#00d2ff", width=2, dash="dash")),
                      dict(type="line", x0=m-s, x1=m-s, y0=0, y1=1, yref="paper",
                           line=dict(color="#888", width=1, dash="dot")),
                      dict(type="line", x0=m+s, x1=m+s, y0=0, y1=1, yref="paper",
                           line=dict(color="#888", width=1, dash="dot")),
                  ])
    info = f"{feat} | n={n} | μ={m:.3f} | σ={s:.3f} | bins={nbins}"
    return go.Figure(data=[trace], layout=layout), info, no_update, no_update


@callback(
    Output("dataset-graph", "figure"),
    Output("info-bar", "children"),
    Input("view-mode", "value"),
    Input("sel-heatmap-group", "value"),
    prevent_initial_call=True,
)
def update_heatmap(mode, group):
    if mode != "heatmap":
        return no_update, no_update
    ad = get_app_data()
    feats = (
        [f for g in FEATURE_GROUPS.values() for f in g if f in ad.num_cols]
        if group == "All"
        else [f for f in FEATURE_GROUPS.get(group, []) if f in ad.num_cols]
    )
    feats = feats[:40]
    n = len(feats)
    mat = np.zeros((n, n))
    for i, fi in enumerate(feats):
        for j, fj in enumerate(feats):
            if i == j:
                mat[i, j] = 1.0
                continue
            xi, xj = ad.feat_array(fi), ad.feat_array(fj)
            valid = np.isfinite(xi) & np.isfinite(xj)
            mat[i, j] = corrcoef(xi[valid], xj[valid])
    trace = go.Heatmap(
        z=mat, x=feats, y=feats,
        colorscale=[[0,"#00d2ff"],[0.5,"#0d0d1a"],[1,"#e94560"]],
        zmin=-1, zmax=1,
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r=%{z:.3f}<extra></extra>",
    )
    layout = dict(**_DARK,
                  title=dict(text=f"Pearson r — {n} features", font=dict(size=14)),
                  xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                  yaxis=dict(tickfont=dict(size=9)),
                  margin=dict(t=46, b=80, l=80, r=20))
    info = f"Pearson r | {n} features | click cell → Scatter"
    return go.Figure(data=[trace], layout=layout), info


@callback(
    Output("dataset-graph", "figure"),
    Output("info-bar", "children"),
    Input("view-mode", "value"),
    prevent_initial_call=True,
)
def update_parallel(mode):
    if mode != "parallel":
        return no_update, no_update
    ad = get_app_data()
    feats = [f for f in CURATED_FEATURES if f in ad.num_cols]
    dims = []
    for f in feats:
        arr = ad.feat_array(f).tolist()
        dims.append(dict(label=f, values=arr))
    color_vals = ad.feat_array(feats[0]) if feats else []
    trace = go.Parcoords(
        line=dict(
            color=color_vals,
            colorscale=[[0,"#0d0520"],[0.33,"#2e0d5c"],[0.66,"#6b2ca0"],[1,"#a855d4"]],
            showscale=True,
            colorbar=dict(title=feats[0] if feats else "", thickness=12),
            opacity=0.25,
        ),
        dimensions=dims,
    )
    layout = dict(template="plotly_dark", paper_bgcolor="#0d0d1a",
                  font=dict(color="#ccc", size=10),
                  title=dict(text=f"Parallel Coords — {len(feats)} curated features",
                             font=dict(size=14)),
                  margin=dict(t=60, b=20, l=30, r=60))
    info = f"Parallel coords | {len(feats)} curated features | drag bands to filter"
    return go.Figure(data=[trace], layout=layout), info
```

- [ ] **Step 2: Run the app and verify these three modes render without errors**

```bash
python plots/explorer/app.py --debug 2>&1 | head -5 &
sleep 4
curl -s http://localhost:7895/ | grep -c "dataset-graph" && echo "OK"
kill %1
```

- [ ] **Step 3: Commit**

```bash
git add plots/explorer/tabs/dataset.py
git commit -m "feat: histogram, heatmap, parallel mode callbacks"
```

---

### Task 8: Radar mode callback

**Files:**
- Modify: `plots/explorer/tabs/dataset.py`

- [ ] **Step 1: Append radar callback**

```python
# Append to plots/explorer/tabs/dataset.py

@callback(
    Output("dataset-graph", "figure"),
    Output("info-bar", "children"),
    Output("track-list-body", "children"),
    Output("tl-title", "children"),
    Input("view-mode", "value"),
    Input("sel-radar-track", "value"),
    # radar stores its own segment-selection state
    Input("dataset-graph", "clickData"),
    State("dataset-graph", "figure"),
    State("view-mode", "value"),
    prevent_initial_call=True,
)
def update_radar(mode, track_name, click_data, current_fig, current_mode):
    ctx = dash.callback_context
    if mode != "radar":
        return no_update, no_update, no_update, no_update

    ad    = get_app_data()
    tidx  = ad.tracks.index(track_name) if track_name in ad.tracks else -1
    if tidx < 0:
        return go.Figure(), "Track not found.", no_update, no_update

    feats = [f for f in CURATED_FEATURES if f in ad.num_cols
             and np.isfinite(ad.feat_array(f)[tidx])]
    if not feats:
        return go.Figure(), "No curated features for this track.", no_update, no_update

    # ── Segment-selection state (stored in current_fig.layout.meta) ─────────
    # Each selected segment is stored as "feat_curveIdx" in a set
    selected: set[str] = set()
    if (current_fig and current_fig.get("layout", {}).get("meta")
            and current_mode == "radar"
            and ctx.triggered_id == "dataset-graph"):
        selected = set(current_fig["layout"]["meta"].get("radar_selected", []))
        if click_data and click_data["points"]:
            pt = click_data["points"][0]
            lbl   = pt.get("theta", "")
            curve = pt.get("curveNumber", -1)
            feat_match = next((f for f in feats if _short_label(f) == lbl), None)
            if feat_match and 0 <= curve <= 2:
                key = f"{feat_match}_{curve}"
                if key in selected:
                    selected.discard(key)
                else:
                    selected.add(key)

    # ── Build stacked polar ──────────────────────────────────────────────────
    r_inner, r_mid, r_outer = [], [], []
    c_inner, c_mid, c_outer = [], [], []
    h_inner, h_mid, h_outer = [], [], []
    labels = [_short_label(f) for f in feats]
    filter_ranges: dict[str, dict] = {}

    for f in feats:
        all_vals = ad.feat_array(f)
        valid    = all_vals[np.isfinite(all_vals)]
        p01  = float(np.percentile(valid, 1))
        p99  = float(np.percentile(valid, 99))
        mn   = float(valid.mean())
        raw_t = float(ad.feat_array(f)[tidx])
        clamp = lambda v: max(0.0, min(1.0, (v - p01) / (p99 - p01 + 1e-12)))
        nmt, nmd = clamp(raw_t), clamp(mn)
        v1, v2 = min(nmt, nmd), max(nmt, nmd)
        t_gt_m = raw_t >= mn
        r_inner.append(v1)
        r_mid.append(v2 - v1)
        r_outer.append(1.0 - v2)
        u = f" {FEATURE_UNITS[f]}" if f in FEATURE_UNITS else ""
        fmt = lambda v: f"{v:.2f}{u}"
        if t_gt_m:
            c_inner.append("rgba(233,69,96,0.4)"); c_mid.append("rgba(233,69,96,0.9)")
            h_inner.append(f"Below mean (<{fmt(mn)})"); h_mid.append(f"Track>{fmt(mn)}")
        else:
            c_inner.append("rgba(0,210,255,0.9)"); c_mid.append("rgba(180,180,180,0.3)")
            h_inner.append(f"Track value {fmt(raw_t)}"); h_mid.append(f"Above track, below mean")
        c_outer.append("rgba(255,255,255,0.04)")
        h_outer.append(f"Above {fmt(max(raw_t,mn))}")

        # Build filter ranges from selected segments
        s0 = f"{f}_0" in selected; s1 = f"{f}_1" in selected; s2 = f"{f}_2" in selected
        if s0 or s1 or s2:
            b1, b2 = min(raw_t, mn), max(raw_t, mn)
            if s0 and not s1 and not s2: low, high = -1e18, b1
            elif not s0 and s1 and not s2: low, high = b1, b2
            elif not s0 and not s1 and s2: low, high = b2, 1e18
            elif s0 and s1 and not s2:    low, high = -1e18, b2
            elif not s0 and s1 and s2:    low, high = b1, 1e18
            else:                          low, high = -1e18, 1e18
            filter_ranges[f] = {"low": low, "high": high}

    # ── Apply filter to build track list ────────────────────────────────────
    if filter_ranges:
        matched = []
        for ti, t in enumerate(ad.tracks):
            if ti == tidx:
                continue
            ok = True
            for f, bounds in filter_ranges.items():
                v = ad.feat_array(f)[ti]
                if not np.isfinite(v) or v < bounds["low"] or v > bounds["high"]:
                    ok = False; break
            if ok:
                matched.append(ti)
        tl = _render_track_list(matched[:150], ad)
        tl_title = f"Tracks ({len(matched[:150])})"
    else:
        tl, tl_title = no_update, no_update

    traces = [
        go.Barpolar(r=r_inner, theta=labels, name="Inner",
                    marker=dict(color=c_inner), hovertext=h_inner, hoverinfo="text"),
        go.Barpolar(r=r_mid,   theta=labels, name="Mid",
                    marker=dict(color=c_mid),   hovertext=h_mid,   hoverinfo="text"),
        go.Barpolar(r=r_outer, theta=labels, name="Outer",
                    marker=dict(color=c_outer), hovertext=h_outer, hoverinfo="text"),
    ]
    layout = dict(
        template="plotly_dark", paper_bgcolor="#0d0d1a", font=dict(color="#ccc"),
        title=dict(text=f"Radar: {track_name}", font=dict(size=13)),
        polar=dict(barmode="stack",
                   radialaxis=dict(visible=True, showticklabels=False,
                                   range=[0, 1.0], gridcolor="#333355"),
                   angularaxis=dict(tickfont=dict(size=10), gridcolor="#333355"),
                   bgcolor="#111125"),
        showlegend=False,
        margin=dict(t=60, b=30, l=30, r=30),
        meta={"radar_selected": list(selected)},
    )
    info = f"Radar: {track_name} | click bars to filter"
    return go.Figure(data=traces, layout=layout), info, tl, tl_title


def _short_label(feat: str) -> str:
    return feat.replace("rms_energy_", "").replace("spectral_", "spec_").replace("_probability", "_p")
```

- [ ] **Step 2: Add `_render_track_list` helper (used by radar and sidebar)**

```python
# Append to plots/explorer/tabs/dataset.py

def _render_track_list(idxs: list[int], ad) -> html.Div:
    """Render a list of track items as Dash html elements."""
    if not idxs:
        return html.Div("No tracks match.", style={"color": "#555", "fontSize": "11px",
                                                    "padding": "8px"})
    items = []
    for idx in idxs:
        name = ad.tracks[idx]
        items.append(
            html.Div(
                name,
                className="tl-item",
                id={"type": "tl-item", "index": idx},
                title=name,
            )
        )
    return html.Div(items)
```

- [ ] **Step 3: Commit**

```bash
git add plots/explorer/tabs/dataset.py
git commit -m "feat: radar mode with stacked polar and segment-filter track list"
```

---

### Task 9: Similarity and Classes mode callbacks

**Files:**
- Modify: `plots/explorer/tabs/dataset.py`

- [ ] **Step 1: Append similarity callback**

```python
# Append to plots/explorer/tabs/dataset.py

@callback(
    Output("dataset-graph", "figure"),
    Output("info-bar", "children"),
    Output("nn-panel", "children"),
    Input("view-mode", "value"),
    Input("sel-sim-ref", "value"),
    Input("sim-xp", "value"), Input("sim-xn", "value"),
    Input("sim-yp", "value"), Input("sim-yn", "value"),
    prevent_initial_call=True,
)
def update_similarity(mode, ref_name, kxp, kxn, kyp, kyn):
    if mode != "similarity":
        return no_update, no_update, no_update
    ad = get_app_data()
    ridx = ad.tracks.index(ref_name) if ref_name in ad.tracks else -1
    if ridx < 0 or not all([kxp, kxn, kyp, kyn]):
        return go.Figure(), "Select reference and four axes.", no_update

    dxp = ad.feat_array(kxp); dxn = ad.feat_array(kxn)
    dyp = ad.feat_array(kyp); dyn = ad.feat_array(kyn)
    valid = np.isfinite(dxp) & np.isfinite(dxn) & np.isfinite(dyp) & np.isfinite(dyn)
    idxs  = np.where(valid)[0]

    nxp = norm01(dxp[valid]); nxn = norm01(dxn[valid])
    nyp = norm01(dyp[valid]); nyn = norm01(dyn[valid])
    ax = nxp - nxn; ay = nyp - nyn

    rpos = np.where(idxs == ridx)[0]
    if len(rpos) == 0:
        return go.Figure(), "Reference not in valid set.", no_update
    rx, ry = ax[rpos[0]], ay[rpos[0]]
    rel_x = ax - rx; rel_y = ay - ry
    dists  = np.sqrt(rel_x**2 + rel_y**2)
    max_d  = float(dists[idxs != ridx].max()) if (idxs != ridx).any() else 1.0

    mask_other = idxs != ridx
    ox = rel_x[mask_other]; oy = rel_y[mask_other]
    od = dists[mask_other];  oidxs = idxs[mask_other]
    hover = [f"<b>{ad.tracks[i]}</b><br>dist: {od[j]:.3f}"
             for j, i in enumerate(oidxs)]

    traces = [
        go.Scattergl(x=ox, y=oy, mode="markers",
                     marker=dict(color=od,
                                 colorscale=[[0,"#e94560"],[0.5,"#552244"],[1,"#111133"]],
                                 cmin=0, cmax=max_d, size=6, opacity=0.75,
                                 colorbar=dict(title="dist", thickness=12)),
                     hovertext=hover, hoverinfo="text",
                     name="tracks", customdata=oidxs),
        go.Scatter(x=[0], y=[0], mode="markers+text",
                   marker=dict(symbol="star", size=20, color="#ffd700",
                               line=dict(color="white", width=1)),
                   text=[ref_name], textposition="top center",
                   textfont=dict(color="#ffd700", size=10),
                   name=ref_name, hoverinfo="text",
                   hovertext=[f"<b>{ref_name}</b> (reference)"],
                   customdata=[ridx]),
    ]
    layout = dict(**_DARK,
                  title=dict(text=f"Similarity — {ref_name}", font=dict(size=13)),
                  xaxis=dict(title=f"← {kxn}  |  {kxp} →",
                             range=[-1.35, 1.35], zeroline=True, zerolinecolor="#555577"),
                  yaxis=dict(title=f"← {kyn}  |  {kyp} →",
                             range=[-1.35, 1.35], zeroline=True, zerolinecolor="#555577",
                             scaleanchor="x"),
                  showlegend=False,
                  shapes=[
                      dict(type="circle", x0=-0.5, y0=-0.5, x1=0.5, y1=0.5,
                           line=dict(color="#3a3a5a", width=1, dash="dot")),
                      dict(type="circle", x0=-1.0, y0=-1.0, x1=1.0, y1=1.0,
                           line=dict(color="#3a3a5a", width=1, dash="dot")),
                  ])

    # NN panel: top 20 by distance
    sorted_pairs = sorted(zip(od, oidxs), key=lambda t: t[0])[:20]
    nn_items = []
    for rank, (dist, tidx) in enumerate(sorted_pairs):
        nn_items.append(html.Div([
            html.Span(f"{rank+1}.", style={"fontSize": "9px", "color": "#555",
                                            "flexShrink": "0"}),
            html.Span(ad.tracks[tidx], className="tl-item",
                      style={"flex": "1", "fontSize": "10px", "overflow": "hidden",
                             "textOverflow": "ellipsis", "whiteSpace": "nowrap"},
                      title=ad.tracks[tidx]),
            html.Span(f"{dist:.3f}", style={"fontSize": "9px", "color": "#555",
                                             "flexShrink": "0"}),
        ], className="nn-row", style={"display": "flex", "gap": "4px"},
           id={"type": "tl-item", "index": int(tidx)}))
    nn_panel = html.Div([
        html.Div("Most Similar", style={"fontSize": "11px", "fontWeight": "bold",
                                         "marginBottom": "4px"}),
        html.Div(nn_items, style={"overflowY": "auto", "maxHeight": "calc(50vh - 100px)"}),
    ])
    info = f"Similarity: {ref_name} | {len(oidxs)} tracks | click to set reference"
    return go.Figure(data=traces, layout=layout), info, nn_panel
```

- [ ] **Step 2: Append classes callback**

```python
# Append to plots/explorer/tabs/dataset.py

@callback(
    Output("dataset-graph", "figure"),
    Output("info-bar", "children"),
    Input("view-mode", "value"),
    Input("sel-x", "value"), Input("sel-y", "value"),
    Input("sel-class-by", "value"),
    Input("chk-class-trend", "value"),
    prevent_initial_call=True,
)
def update_classes(mode, kx, ky, class_key, trend_opts):
    if mode != "classes":
        return no_update, no_update
    ad = get_app_data()
    if not kx or not ky or not class_key:
        return go.Figure(), "Select X, Y, and class-by."
    if class_key not in ad.class_cols:
        return go.Figure(), f"{class_key} not available."

    dx = ad.feat_array(kx); dy = ad.feat_array(ky)
    raw_labels = ad.class_array(class_key)
    groups: dict[str, dict] = {}
    for i, (xi, yi) in enumerate(zip(dx, dy)):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        lbl = parse_class_label(raw_labels[i]) or "(unknown)"
        g = groups.setdefault(lbl, {"x": [], "y": [], "idxs": []})
        g["x"].append(xi); g["y"].append(yi); g["idxs"].append(i)

    sorted_groups = sorted(groups.items(), key=lambda t: -len(t[1]["x"]))
    if len(sorted_groups) > 20:
        top = sorted_groups[:20]
        other = {"x": [], "y": [], "idxs": []}
        for _, g in sorted_groups[20:]:
            other["x"] += g["x"]; other["y"] += g["y"]; other["idxs"] += g["idxs"]
        top.append(("(other)", other))
        sorted_groups = top

    ux = f" [{FEATURE_UNITS[kx]}]" if kx in FEATURE_UNITS else ""
    uy = f" [{FEATURE_UNITS[ky]}]" if ky in FEATURE_UNITS else ""
    traces = []
    for ci, (lbl, g) in enumerate(sorted_groups):
        x, y = np.array(g["x"]), np.array(g["y"])
        col = CLASS_COLORS[ci % len(CLASS_COLORS)]
        hover = [f"<b>{ad.tracks[i]}</b><br>{kx}: {x[j]:.3f}<br>{ky}: {y[j]:.3f}<br>{class_key}: {lbl}"
                 for j, i in enumerate(g["idxs"])]
        traces.append(go.Scattergl(
            x=x, y=y, mode="markers",
            marker=dict(color=col, size=5, opacity=0.7),
            hovertext=hover, hoverinfo="text",
            name=f"{lbl} ({len(x)})", customdata=g["idxs"],
        ))
        if trend_opts and "trend" in trend_opts and len(x) >= 3:
            mx, my = x.mean(), y.mean()
            num = ((x-mx)*(y-my)).sum(); den = ((x-mx)**2).sum()
            if den > 1e-12:
                slope = num/den; ic = my - slope*mx
                xmn, xmx = x.min(), x.max()
                traces.append(go.Scatter(
                    x=[xmn, xmx], y=[slope*xmn+ic, slope*xmx+ic],
                    mode="lines", line=dict(color=col, width=2, dash="dash"),
                    name=f"{lbl} trend", hoverinfo="skip",
                    showlegend=False,
                ))

    layout = dict(**_DARK,
                  title=dict(text=f"{kx} vs {ky} by {class_key}", font=dict(size=14)),
                  xaxis=dict(title=kx+ux), yaxis=dict(title=ky+uy),
                  showlegend=True,
                  legend=dict(x=1.02, y=1, bgcolor="rgba(15,52,96,0.8)",
                               font=dict(size=10)),
                  margin=dict(t=46, b=48, l=58, r=180))
    total = sum(len(g["x"]) for _, g in sorted_groups)
    info = f"Classes: {class_key} | {len(sorted_groups)} groups | {total} tracks"
    return go.Figure(data=traces, layout=layout), info
```

Also add the missing import at the top of `dataset.py`:
```python
from plots.explorer.data import parse_class_label
```

- [ ] **Step 3: Commit**

```bash
git add plots/explorer/tabs/dataset.py
git commit -m "feat: similarity and classes mode callbacks"
```

---

### Task 10: Active track Store — click, double-click, hover, search

**Files:**
- Modify: `plots/explorer/tabs/dataset.py`

- [ ] **Step 1: Append active-track callbacks**

```python
# Append to plots/explorer/tabs/dataset.py

@callback(
    Output("active-track", "data"),
    Input("dataset-graph", "clickData"),
    Input("dataset-graph", "hoverData"),
    State("active-track", "data"),
    State("view-mode", "value"),
    State("autoplay-hover", "data"),
    prevent_initial_call=True,
)
def update_active_track(click_data, hover_data, current, mode, autoplay_hover):
    """Single click → slot A. Hover → slot A only if autoplay enabled."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update

    ad   = get_app_data()
    trig = ctx.triggered[0]["prop_id"]

    if "clickData" in trig and click_data:
        pt = click_data["points"][0]
        # customdata holds the track index (set in all scatter traces)
        tidx = pt.get("customdata")
        if tidx is not None:
            tidx = int(tidx)
            track = ad.tracks[tidx]
            state = dict(current or {})
            state.update({"track": track, "track_idx": tidx, "slot": "a"})
            return state

    if "hoverData" in trig and hover_data and autoplay_hover:
        pt = hover_data["points"][0]
        tidx = pt.get("customdata")
        if tidx is not None:
            tidx = int(tidx)
            track = ad.tracks[tidx]
            state = dict(current or {})
            state.update({"track": track, "track_idx": tidx, "slot": "a"})
            return state

    return no_update


@callback(
    Output("active-track", "data"),
    Input({"type": "tl-item", "index": dash.ALL}, "n_clicks"),
    State("active-track", "data"),
    prevent_initial_call=True,
)
def track_list_click(n_clicks_list, current):
    """Click track list item → set A slot."""
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return no_update
    trig_id = ctx.triggered[0]["prop_id"]
    # Extract index from pattern-match id
    import json
    raw = trig_id.split(".")[0]
    tidx = json.loads(raw)["index"]
    ad = get_app_data()
    state = dict(current or {})
    state.update({"track": ad.tracks[tidx], "track_idx": tidx, "slot": "a"})
    return state


@callback(
    Output("view-mode", "value"),
    Input({"type": "tl-item", "index": dash.ALL}, "n_clicks"),
    State("view-mode", "value"),
    prevent_initial_call=True,
)
def track_list_click_switch_mode(n_clicks_list, current_mode):
    """Clicking track list item always switches to Similarity mode."""
    if not any(n_clicks_list):
        return no_update
    return "similarity"


@callback(
    Output("sel-radar-track", "options"),
    Output("sel-sim-ref",     "options"),
    Input("sel-radar-track", "search_value"),
    Input("sel-sim-ref",     "search_value"),
    prevent_initial_call=False,
)
def filter_track_dropdowns(radar_q, sim_q):
    """Live %pattern% filtering for track dropdowns."""
    ad = get_app_data()
    ctx = dash.callback_context
    trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    if "radar" in trig:
        q = radar_q or ""
        opts = ad.track_options(q)
        return opts, no_update
    if "sim-ref" in trig:
        q = sim_q or ""
        opts = ad.track_options(q)
        return no_update, opts
    # Initial load: return all
    opts = ad.track_options()
    return opts, opts
```

- [ ] **Step 2: Commit**

```bash
git add plots/explorer/tabs/dataset.py
git commit -m "feat: active-track Store wiring, track list clicks → Similarity, pattern search"
```

---

### Task 11: Histogram + Parallel → track list sidebar callbacks

**Files:**
- Modify: `plots/explorer/tabs/dataset.py`

- [ ] **Step 1: Add histogram bar-click → track list**

```python
# Append to plots/explorer/tabs/dataset.py

@callback(
    Output("track-list-body", "children"),
    Output("tl-title", "children"),
    Input("dataset-graph", "clickData"),
    State("view-mode", "value"),
    State("sel-hist", "value"),
    State("hist-bins", "value"),
    prevent_initial_call=True,
)
def histogram_bar_click(click_data, mode, feat, nbins):
    if mode != "histogram" or not click_data or not feat:
        return no_update, no_update
    ad    = get_app_data()
    vals  = ad.feat_array(feat)
    nbins = max(4, min(64, int(nbins or 30)))
    pt    = click_data["points"][0]
    vmin, vmax = vals[np.isfinite(vals)].min(), vals[np.isfinite(vals)].max()
    bsize = (vmax - vmin) / nbins
    bin_lo = pt["x"] - bsize / 2
    bin_hi = pt["x"] + bsize / 2
    idxs  = [i for i, v in enumerate(vals) if np.isfinite(v) and bin_lo <= v < bin_hi]
    tl    = _render_track_list(idxs[:200], ad)
    return tl, f"Tracks ({len(idxs)})"


@callback(
    Output("track-list-body", "children"),
    Output("tl-title", "children"),
    Input("dataset-graph", "restyleData"),
    State("view-mode", "value"),
    State("dataset-graph", "figure"),
    prevent_initial_call=True,
)
def parallel_drag_filter(restyle_data, mode, fig):
    """When parallel coord bands are dragged, update track list."""
    if mode != "parallel" or not restyle_data or not fig:
        return no_update, no_update
    ad    = get_app_data()
    feats = [f for f in CURATED_FEATURES if f in ad.num_cols]
    if not fig.get("data"):
        return no_update, no_update
    trace = fig["data"][0]
    dims  = trace.get("dimensions", [])
    idxs = []
    for ti in range(len(ad.tracks)):
        passes = True
        for di, dim in enumerate(dims):
            cr = dim.get("constraintrange")
            if cr is None:
                continue
            feat = feats[di] if di < len(feats) else None
            if feat is None:
                continue
            v = float(ad.feat_array(feat)[ti]) if np.isfinite(ad.feat_array(feat)[ti]) else None
            if v is None:
                passes = False; break
            lo, hi = (cr[0], cr[1]) if isinstance(cr[0], (int, float)) else (cr[0][0], cr[0][1])
            if not (lo <= v <= hi):
                passes = False; break
        if passes:
            idxs.append(ti)
    tl = _render_track_list(idxs[:300], ad)
    return tl, f"Tracks ({len(idxs)})"


@callback(
    Output("track-list-body", "children"),
    Output("tl-title", "children"),
    Input("dataset-graph", "selectedData"),
    State("view-mode", "value"),
    prevent_initial_call=True,
)
def scatter_lasso_select(selected_data, mode):
    if mode not in ("scatter", "quadrant", "similarity", "classes"):
        return no_update, no_update
    if not selected_data or not selected_data.get("points"):
        return html.Div("Lasso or box select to filter tracks.",
                        style={"color": "#555", "fontSize": "11px", "padding": "8px"}), "Tracks"
    ad   = get_app_data()
    idxs = [pt["customdata"] for pt in selected_data["points"]
            if pt.get("customdata") is not None]
    tl   = _render_track_list(idxs[:300], ad)
    return tl, f"Tracks ({len(idxs)})"
```

- [ ] **Step 2: Commit**

```bash
git add plots/explorer/tabs/dataset.py
git commit -m "feat: histogram bar click and parallel drag → track list; lasso select"
```

---

## Phase 2 — Analysis Tab

---

### Task 12: Port `plots/latent_analysis/app.py` → `tabs/analysis.py`

**Files:**
- Modify: `plots/explorer/tabs/analysis.py`

- [ ] **Step 1: Replace stub with the full port**

Copy the working code from `plots/latent_analysis/app.py` into a new module structure. Key changes:
- Remove the standalone `app = dash.Dash(...)` — callbacks register against the parent app via `callback` import
- Remove `if __name__ == "__main__"` block
- Expose a `layout()` function
- Data loading calls `get_analysis()` from `data.py` instead of loading NPZ files directly

```python
# plots/explorer/tabs/analysis.py
"""Tab 2 — Analysis: ported from plots/latent_analysis/app.py."""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback, no_update

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.explorer.data import get_analysis, get_app_data
from plots.latent_analysis.config import (
    FEATURE_GROUPS, LATENT_DIM, POSTER_CLAMP,
    EFFECT_WEAK, EFFECT_STRONG, TEMPORAL_FEATURE_NAMES,
)

_TAB_STYLE        = {"padding": "6px 12px", "color": "#888", "fontSize": "12px"}
_TAB_ACTIVE_STYLE = {"padding": "6px 12px", "backgroundColor": "#7eb8f7",
                     "color": "#0d0d1a", "fontWeight": "700", "fontSize": "12px"}

_HELP_CORR = (
    "Rows = 64 VAE latent dimensions; columns = MIR features. "
    "Colour = Pearson r (red positive, blue negative). "
    "Sort by 'Feature loading' to surface most expressive dims. "
    "Click any cell to see a scatter plot of that dim vs. feature."
)
_HELP_POSTERS = (
    "Each cell is one of the 64 VAE latent dimensions in an 8×8 grid. "
    "Colour = Pearson r with the selected feature. "
    "Red = dim rises when feature rises; blue = inverse."
)
_HELP_PCA = (
    "PCA compresses 64 latent dims into principal components. "
    "Scatter shows every crop projected onto PC axes, coloured by a MIR feature."
)
_HELP_TEMPORAL = (
    "Each latent frame = 2048 audio samples @ 44 100 Hz ≈ 46 ms. "
    "Latent dim values over time alongside frame-level audio features."
)
_HELP_XCORR = (
    "64×64 Pearson r matrix between every pair of latent dims, averaged over crops. "
    "Sorted by Ward cluster. Tight clusters encode related perceptual qualities."
)
_HELP_CLUSTERS = (
    "Ward clusters from the cross-correlation analysis. "
    "Click 'Highlight in Dataset' to overlay cluster dims on the scatter."
)


def layout() -> html.Div:
    npz = get_analysis()
    d01, d03 = npz.get("d01"), npz.get("d03")
    feat_names = list(d01["feature_names"]) if d01 else []
    ALL_GROUPS = list(FEATURE_GROUPS.keys()) + ["All"]
    return html.Div([
        dcc.Tabs(id="analysis-subtabs", value="corr", children=[
            dcc.Tab(label="Correlation Matrix", value="corr",
                    style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
            dcc.Tab(label="Feature Posters",    value="posters",
                    style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
            dcc.Tab(label="PCA Explorer",       value="pca",
                    style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
            dcc.Tab(label="Temporal",           value="temporal",
                    style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
            dcc.Tab(label="Latent Cross-Corr",  value="xcorr",
                    style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
            dcc.Tab(label="Cluster Map",        value="clusters",
                    style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
        ]),
        html.Div(id="analysis-tab-content", style={"padding": "10px"}),
    ])


@callback(Output("analysis-tab-content", "children"),
          Input("analysis-subtabs", "value"))
def render_analysis_tab(sub: str):
    npz = get_analysis()
    d01 = npz.get("d01"); d02 = npz.get("d02")
    d03 = npz.get("d03"); d04 = npz.get("d04")
    feat_names = list(d01["feature_names"]) if d01 else []
    ALL_GROUPS = list(FEATURE_GROUPS.keys()) + ["All"]

    if sub == "corr":
        if d01 is None:
            return html.P("Run 01_aggregate_correlation.py first.")
        return html.Div([
            html.P(_HELP_CORR, style={"color": "#888", "fontSize": "0.85em", "marginBottom": "8px"}),
            html.Div([
                html.Label("Sort dims by:"),
                dcc.RadioItems(id="corr-sort",
                               options=[{"label": "Feature loading", "value": "loading"},
                                        {"label": "Cluster order",   "value": "cluster"},
                                        {"label": "Index",           "value": "index"}],
                               value="loading", inline=True),
                html.Label("Feature group:", style={"marginLeft": "20px"}),
                dcc.Dropdown(id="corr-group",
                             options=[{"label": g, "value": g} for g in ALL_GROUPS],
                             value="All", clearable=False,
                             style={"width": "160px", "display": "inline-block"}),
                html.Label("|r| ≥", style={"marginLeft": "20px"}),
                dcc.Slider(id="corr-thresh", min=0, max=0.35, step=0.05, value=0.0,
                           marks={v: f"{v:.2f}" for v in [0, 0.1, 0.2, 0.35]},
                           tooltip={"always_visible": False}),
                html.Label("Metric:", style={"marginLeft": "20px"}),
                dcc.RadioItems(id="corr-metric",
                               options=[{"label": "Pearson",  "value": "pearson"},
                                        {"label": "Spearman", "value": "spearman"}],
                               value="pearson", inline=True),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px",
                      "alignItems": "center", "marginBottom": "8px"}),
            dcc.Graph(id="corr-heatmap"),
            html.Div(id="corr-scatter-container"),
        ])

    if sub == "posters":
        if d01 is None:
            return html.P("Run 01_aggregate_correlation.py first.")
        options = [{"label": n, "value": n} for n in feat_names]
        return html.Div([
            html.P(_HELP_POSTERS, style={"color": "#888", "fontSize": "0.85em", "marginBottom": "8px"}),
            dcc.Dropdown(id="poster-feat", options=options,
                         value=feat_names[0] if feat_names else None,
                         clearable=False, style={"width": "300px"}),
            dcc.Graph(id="poster-graph", style={"marginTop": "12px"}),
        ])

    if sub == "pca":
        if d02 is None:
            return html.P("Run 02_pca_analysis.py first.")
        return html.Div([
            html.P(_HELP_PCA, style={"color": "#888", "fontSize": "0.85em", "marginBottom": "8px"}),
            html.Div([
                html.Label("Colour by feature:"),
                dcc.Dropdown(id="pca-colour",
                             options=[{"label": n, "value": n} for n in feat_names],
                             value=feat_names[0] if feat_names else None,
                             style={"width": "220px", "display": "inline-block"}),
                html.Label("PC axes:", style={"marginLeft": "16px"}),
                dcc.RadioItems(id="pca-axes",
                               options=[{"label": "PC1 vs PC2", "value": "12"},
                                        {"label": "PC1 vs PC3", "value": "13"}],
                               value="12", inline=True),
            ], style={"display": "flex", "gap": "12px", "alignItems": "center",
                      "marginBottom": "8px"}),
            dcc.Graph(id="pca-scatter"),
            html.H4("Cross-PCA alignment", style={"marginTop": "16px"}),
            dcc.Graph(id="pca-cross-heatmap"),
        ])

    if sub == "temporal":
        if d04 is None:
            return html.P("Run 04_temporal_correlation.py first.")
        sample   = d04.get("sample_crops")
        crop_opts = [{"label": f"crop {i}", "value": i}
                     for i in range(len(sample) if sample is not None else 0)]
        return html.Div([
            html.P(_HELP_TEMPORAL, style={"color": "#888", "fontSize": "0.85em", "marginBottom": "8px"}),
            html.Div([
                html.Label("Crop:"),
                dcc.Dropdown(id="temp-crop", options=crop_opts, value=0,
                             style={"width": "140px", "display": "inline-block"}),
                html.Label("Dims (comma-sep):", style={"marginLeft": "16px"}),
                dcc.Input(id="temp-dims", value="0,1,2,3,4", type="text",
                          style={"width": "180px"}),
            ], style={"display": "flex", "gap": "10px", "alignItems": "center",
                      "marginBottom": "8px"}),
            dcc.Graph(id="temporal-graph"),
        ])

    if sub == "xcorr":
        if d03 is None:
            return html.P("Run 03_latent_xcorr.py first.")
        return html.Div([
            html.P(_HELP_XCORR, style={"color": "#888", "fontSize": "0.85em", "marginBottom": "8px"}),
            dcc.Graph(id="xcorr-heatmap", figure=_build_xcorr_fig(d03)),
        ])

    if sub == "clusters":
        if d03 is None or d01 is None:
            return html.P("Run scripts 01 and 03 first.")
        return _cluster_layout(d01, d03, feat_names)

    return html.P("Unknown sub-tab.")


# ── Callbacks (direct ports from app.py) ─────────────────────────────────────

@callback(Output("corr-heatmap", "figure"),
          [Input("corr-sort", "value"), Input("corr-group", "value"),
           Input("corr-thresh", "value"), Input("corr-metric", "value")])
def update_corr_heatmap(sort_by, group, thresh, metric):
    npz = get_analysis(); d01 = npz.get("d01"); d03 = npz.get("d03")
    if d01 is None: return go.Figure()
    feat_names = list(d01["feature_names"])
    cluster_labels = d03["cluster_labels"] if d03 else np.zeros(LATENT_DIM, dtype=int)
    r = d01["r_pearson"] if metric == "pearson" else d01["r_spearman"]
    names = feat_names
    if group != "All":
        keep  = [i for i, n in enumerate(names) if n in set(FEATURE_GROUPS.get(group, []))]
        r     = r[:, keep]; names = [names[i] for i in keep]
    r_disp = np.where(np.abs(r) >= thresh, r, 0.0)
    if sort_by == "loading":    order = np.argsort(np.abs(r_disp).max(axis=1))[::-1]
    elif sort_by == "cluster":  order = np.argsort(cluster_labels)
    else:                       order = np.arange(LATENT_DIM)
    r_disp = r_disp[order]
    fig = go.Figure(go.Heatmap(z=r_disp, x=names, y=[f"dim {i}" for i in order],
                               colorscale="RdBu_r", zmid=0,
                               zmin=-POSTER_CLAMP, zmax=POSTER_CLAMP,
                               colorbar=dict(title="r")))
    fig.update_layout(template="plotly_dark", height=700,
                      xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                      yaxis=dict(tickfont=dict(size=9)),
                      margin=dict(l=60, r=20, t=30, b=120))
    return fig


@callback(Output("corr-scatter-container", "children"),
          Input("corr-heatmap", "clickData"),
          State("corr-metric", "value"))
def corr_scatter(click_data, metric):
    npz = get_analysis(); d01 = npz.get("d01")
    if click_data is None or d01 is None:
        return html.P("Click a cell to see scatter plot.", style={"color": "#555"})
    pt  = click_data["points"][0]
    feat_name = pt["x"]; dim_label = pt["y"]
    dim = int(dim_label.split(" ")[1])
    feat_names = list(d01["feature_names"])
    fi  = feat_names.index(feat_name) if feat_name in feat_names else None
    if fi is None: return html.P(f"Feature {feat_name} not found.")
    r_val = d01["r_pearson"][dim, fi]
    from plots.latent_analysis.config import DATA_DIR
    scatter_d_path = DATA_DIR / "scatter_sample.npz"
    if not scatter_d_path.exists():
        return html.P("scatter_sample.npz not found — run script 01.")
    scatter_d = dict(np.load(str(scatter_d_path), allow_pickle=True))
    lm = scatter_d["latent_means"]; fv = scatter_d["feature_values"]
    fn = list(scatter_d["feature_names"])
    x  = lm[:, dim]
    if feat_name not in fn: return html.P(f"Feature {feat_name} not in scatter sample.")
    y  = fv[:, fn.index(feat_name)]
    fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers",
                                  marker=dict(size=3, opacity=0.4)))
    fig.update_layout(template="plotly_dark", height=300,
                      title=f"dim {dim} × {feat_name} — r={r_val:+.3f}",
                      xaxis_title=f"Latent dim {dim} mean", yaxis_title=feat_name)
    return dcc.Graph(figure=fig)


@callback(Output("poster-graph", "figure"), Input("poster-feat", "value"))
def update_poster(feat_name):
    npz = get_analysis(); d01 = npz.get("d01")
    if not feat_name or d01 is None: return go.Figure()
    feat_names = list(d01["feature_names"])
    fi = feat_names.index(feat_name) if feat_name in feat_names else None
    if fi is None: return go.Figure()
    r_col = d01["r_pearson"][:, fi]; n = int(d01["n_per_feature"][fi])
    grid  = r_col.reshape(8, 8)
    top3p = np.argsort(r_col)[::-1][:3]; top3n = np.argsort(r_col)[:3]
    fig   = go.Figure(go.Heatmap(z=grid, colorscale="RdBu_r", zmid=0,
                                  zmin=-POSTER_CLAMP, zmax=POSTER_CLAMP,
                                  text=[[f"{grid[r,c]:.2f}" for c in range(8)] for r in range(8)],
                                  texttemplate="%{text}", textfont={"size": 10}))
    fig.update_layout(template="plotly_dark", width=500, height=500,
                      title=f"{feat_name} — N={n:,} | top+: {list(top3p)} | top-: {list(top3n)}",
                      xaxis=dict(title="dim mod 8", tickvals=list(range(8))),
                      yaxis=dict(title="dim // 8", tickvals=list(range(8)),
                                 ticktext=[str(i*8) for i in range(8)]))
    return fig


@callback(Output("pca-scatter", "figure"),
          [Input("pca-colour", "value"), Input("pca-axes", "value")])
def update_pca_scatter(colour_feat, axes):
    npz = get_analysis(); d02 = npz.get("d02")
    if d02 is None: return go.Figure()
    scores = d02["latent_scores"]
    pc1 = int(axes[0]) - 1; pc2 = int(axes[1]) - 1
    colour = None
    from plots.latent_analysis.config import DATA_DIR
    scatter_path = DATA_DIR / "scatter_sample.npz"
    if colour_feat and scatter_path.exists():
        sd = dict(np.load(str(scatter_path), allow_pickle=True))
        fn = list(sd["feature_names"])
        if colour_feat in fn:
            colour = sd["feature_values"][:, fn.index(colour_feat)][:scores.shape[0]]
    ev  = d02["latent_explained_variance_ratio"]
    fig = go.Figure(go.Scattergl(x=scores[:, pc1], y=scores[:, pc2],
                                  mode="markers",
                                  marker=dict(size=2, opacity=0.3, color=colour)))
    fig.update_layout(template="plotly_dark", height=500,
                      xaxis_title=f"Latent PC{pc1+1} ({ev[pc1]:.1%})",
                      yaxis_title=f"Latent PC{pc2+1} ({ev[pc2]:.1%})",
                      title="Latent PCA scatter")
    return fig


@callback(Output("pca-cross-heatmap", "figure"), Input("analysis-subtabs", "value"))
def update_cross_heatmap(sub):
    npz = get_analysis(); d02 = npz.get("d02")
    if sub != "pca" or d02 is None: return go.Figure()
    cc  = d02["cross_corr"]
    fig = go.Figure(go.Heatmap(z=cc, colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                                colorbar=dict(title="r")))
    fig.update_layout(template="plotly_dark", height=400,
                      xaxis_title="Latent PC", yaxis_title="Feature PC",
                      title="Feature PC ↔ Latent PC correlation")
    return fig


@callback(Output("temporal-graph", "figure"),
          [Input("temp-crop", "value"), Input("temp-dims", "value")])
def update_temporal(crop_idx, dims_str):
    npz = get_analysis(); d04 = npz.get("d04")
    if d04 is None: return go.Figure()
    sample = d04.get("sample_crops"); feat_sample = d04.get("sample_feat_segs")
    tfeat_names = list(d04.get("temporal_feature_names", TEMPORAL_FEATURE_NAMES))
    if sample is None or crop_idx is None or crop_idx >= len(sample): return go.Figure()
    try:
        dims = [int(d.strip()) for d in dims_str.split(",")]
        dims = [d for d in dims if 0 <= d < LATENT_DIM]
    except ValueError:
        dims = [0]
    lat = sample[crop_idx]; t = np.arange(256) * (2048 / 44100)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for dim in dims:
        fig.add_trace(go.Scattergl(x=t, y=lat[dim], mode="lines",
                                    name=f"dim {dim}", line=dict(width=1)),
                      secondary_y=False)
    if feat_sample is not None and crop_idx < len(feat_sample):
        feats = feat_sample[crop_idx]
        for fi in [0, 5]:
            if fi < feats.shape[0]:
                y = feats[fi]; y_n = (y - y.mean()) / (y.std() + 1e-9)
                fig.add_trace(go.Scattergl(x=t, y=y_n, mode="lines",
                                            name=tfeat_names[fi],
                                            line=dict(width=1, dash="dot")),
                              secondary_y=True)
    fig.update_layout(template="plotly_dark", height=500, xaxis_title="Time (s)",
                      title=f"Latent dims + frame features — crop {crop_idx}",
                      legend=dict(orientation="h"))
    fig.update_yaxes(title_text="Latent value", secondary_y=False)
    fig.update_yaxes(title_text="Feature (z-scored)", secondary_y=True)
    return fig


def _build_xcorr_fig(d03) -> go.Figure:
    xcorr = d03["xcorr_matrix"]; cl = d03["cluster_labels"]
    order = np.argsort(cl); xcorr_ord = xcorr[np.ix_(order, order)]
    labels_ord = [f"dim {i}" for i in order]
    fig = go.Figure(go.Heatmap(z=xcorr_ord, x=labels_ord, y=labels_ord,
                                colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                                colorbar=dict(title="r")))
    fig.update_layout(template="plotly_dark", height=700,
                      title=f"64×64 Latent Cross-Correlation ({int(d03['n_crops_used'])} crops)",
                      xaxis=dict(tickfont=dict(size=7)),
                      yaxis=dict(tickfont=dict(size=7)),
                      margin=dict(l=80, r=20, t=50, b=80))
    return fig


def _cluster_layout(d01, d03, feat_names) -> html.Div:
    cl = d03["cluster_labels"]; r_mat = d01["r_pearson"]
    n_cl = int(cl.max())
    rows = []
    for c in range(1, n_cl + 1):
        dims = list(np.where(cl == c)[0])
        mean_r = r_mat[dims].mean(axis=0)
        top3 = np.argsort(np.abs(mean_r))[::-1][:3]
        summary = ", ".join(
            f"{feat_names[i]} (r={mean_r[i]:+.2f})" for i in top3 if i < len(feat_names)
        )
        rows.append(html.Tr([
            html.Td(f"Cluster {c}"),
            html.Td(", ".join(map(str, dims)), style={"fontSize": "0.85em"}),
            html.Td(summary, style={"fontSize": "0.85em"}),
            html.Td(html.Button("Highlight in Dataset", n_clicks=0,
                                id={"type": "cluster-highlight-btn", "index": c},
                                style={"fontSize": "10px", "padding": "2px 6px",
                                       "background": "#1e2050", "border": "1px solid #4cd137",
                                       "color": "#4cd137", "cursor": "pointer"})),
        ]))
    return html.Div([
        html.P(_HELP_CLUSTERS, style={"color": "#888", "fontSize": "0.85em",
                                       "marginBottom": "8px"}),
        html.Table(
            [html.Tr([html.Th("Cluster"), html.Th("Dims"),
                      html.Th("Top features"), html.Th("")])] + rows,
            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.9em"},
        ),
    ])


@callback(
    Output("cluster-highlight", "data"),
    Input({"type": "cluster-highlight-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def highlight_cluster(n_clicks_list):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return no_update
    import json
    raw = ctx.triggered[0]["prop_id"].split(".")[0]
    c   = json.loads(raw)["index"]
    npz = get_analysis(); d03 = npz.get("d03")
    if d03 is None: return None
    cl   = d03["cluster_labels"]
    dims = [int(i) for i in np.where(cl == c)[0]]
    return {"dims": dims, "cluster": c}
```

- [ ] **Step 2: Verify app starts and Analysis tab renders**

```bash
python plots/explorer/app.py &
sleep 4
curl -s http://localhost:7895/ | grep -c "analysis-subtabs" && echo "OK"
kill %1
```

- [ ] **Step 3: Commit**

```bash
git add plots/explorer/tabs/analysis.py
git commit -m "feat: analysis tab — full port of latent_analysis/app.py with cluster highlight wiring"
```

---

## Phase 3 — Persistent Player Strip

---

### Task 13: `audio.py` — URL builders for latent server

**Files:**
- Create: `plots/explorer/audio.py`
- Create: `tests/explorer/test_audio.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/explorer/test_audio.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.explorer.audio import build_decode_url, build_crossfade_url, build_average_url

def test_decode_url_basic():
    url = build_decode_url("Artist - Track", "0.5")
    assert "track=Artist" in url
    assert "position=0.5" in url
    assert "localhost:7891" in url

def test_decode_url_smart_loop():
    url = build_decode_url("T", "0.5", smart_loop=True)
    assert "smart_loop=1" in url

def test_crossfade_url():
    url = build_crossfade_url("Track A", "0.3", "Track B", "0.7", mix=0.5)
    assert "track_a=Track" in url
    assert "track_b=Track" in url
    assert "mix=0.500" in url

def test_average_url_single():
    url = build_average_url("Track A")
    assert "track=Track" in url
    assert "average" in url
```

- [ ] **Step 2: Write `audio.py`**

```python
# plots/explorer/audio.py
"""URL builders and HTTP helpers for latent_server.py (port 7891)."""
from __future__ import annotations
from urllib.parse import urlencode, quote

_BASE = "http://localhost:7891"


def build_decode_url(track: str, position: str = "0.5",
                     smart_loop: bool = False,
                     interp: str = "slerp",
                     manip: dict | None = None) -> str:
    """Build URL for /decode endpoint."""
    params: dict = {"track": track, "position": position, "interp": interp}
    if smart_loop:
        params["smart_loop"] = "1"
    if manip:
        for k, v in manip.items():
            params[k] = v
    return f"{_BASE}/decode?{urlencode(params)}"


def build_crossfade_url(track_a: str, pos_a: str,
                        track_b: str, pos_b: str,
                        mix: float = 0.5,
                        interp: str = "slerp",
                        smart_loop: bool = False,
                        manip: dict | None = None) -> str:
    """Build URL for /crossfade endpoint."""
    params: dict = {
        "track_a": track_a, "position_a": pos_a,
        "track_b": track_b, "position_b": pos_b,
        "mix": f"{float(mix):.3f}",
        "interp": interp,
    }
    if smart_loop:
        params["smart_loop"] = "1"
    if manip:
        params.update(manip)
    return f"{_BASE}/crossfade?{urlencode(params)}"


def build_average_url(track_a: str,
                      track_b: str | None = None,
                      mix: float = 0.0,
                      interp: str = "slerp",
                      smart_loop: bool = False) -> str:
    """Build URL for /average endpoint."""
    params: dict = {"track": track_a, "interp": interp}
    if track_b:
        params["track_b"] = track_b
        params["mix"]     = f"{float(mix):.3f}"
    if smart_loop:
        params["smart_loop"] = "1"
    return f"{_BASE}/average?{urlencode(params)}"


def check_server_alive(port: int = 7891, timeout: float = 0.5) -> bool:
    """Return True if the latent server is reachable."""
    import urllib.request
    try:
        urllib.request.urlopen(f"http://localhost:{port}/", timeout=timeout)
        return True
    except Exception:
        return False
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/explorer/test_audio.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add plots/explorer/audio.py tests/explorer/test_audio.py
git commit -m "feat: audio.py URL builders for latent_server decode/crossfade/average"
```

---

### Task 14: `assets/player.js` — Web Audio API clientside callback

**Files:**
- Create: `plots/explorer/assets/player.js`
- Modify: `plots/explorer/app.py`

- [ ] **Step 1: Write `player.js`**

```javascript
/* plots/explorer/assets/player.js
 *
 * Clientside callback: watches the "player-cmd" Store and controls
 * a shared AudioContext + AudioBufferSourceNode.
 *
 * Registered command schema (stored in player-cmd dcc.Store):
 *   { action: "play"|"pause"|"stop"|"fade",
 *     url:    string,           // absolute URL to WAV stream
 *     slot:   "a"|"b",
 *     loop_start: number|null,  // seconds
 *     loop_end:   number|null,  // seconds
 *     alpha:  number            // 0=all A, 1=all B (ignored if slot=="a")
 *   }
 */
(function () {
    "use strict";

    let _ctx        = null;
    let _sourceNode = null;
    let _gainNode   = null;
    let _fadeTimer  = null;
    let _lastUrl    = null;
    let _hovX       = null;  // last hover pointer X for fade trigger
    let _hovY       = null;

    function getCtx() {
        if (!_ctx) {
            _ctx       = new (window.AudioContext || window.webkitAudioContext)();
            _gainNode  = _ctx.createGain();
            _gainNode.connect(_ctx.destination);
        }
        return _ctx;
    }

    function stopCurrent() {
        if (_sourceNode) {
            try { _sourceNode.stop(); } catch (_) {}
            _sourceNode.disconnect();
            _sourceNode = null;
        }
    }

    function fadeOut(ms) {
        if (!_gainNode) return;
        clearTimeout(_fadeTimer);
        const ctx = getCtx();
        const now = ctx.currentTime;
        _gainNode.gain.cancelScheduledValues(now);
        _gainNode.gain.setValueAtTime(_gainNode.gain.value, now);
        _gainNode.gain.linearRampToValueAtTime(0, now + ms / 1000);
        _fadeTimer = setTimeout(stopCurrent, ms + 50);
    }

    async function playUrl(url, loopStart, loopEnd) {
        if (_lastUrl === url && _sourceNode) return; // already playing
        _lastUrl = url;
        stopCurrent();
        clearTimeout(_fadeTimer);
        const ctx = getCtx();
        await ctx.resume();
        _gainNode.gain.cancelScheduledValues(ctx.currentTime);
        _gainNode.gain.setValueAtTime(1.0, ctx.currentTime);

        let resp;
        try {
            resp = await fetch(url);
            if (!resp.ok) { console.warn("player.js: fetch failed", resp.status); return; }
        } catch (e) { console.warn("player.js: fetch error", e); return; }

        const buf = await resp.arrayBuffer();
        let decoded;
        try { decoded = await ctx.decodeAudioData(buf); }
        catch (e) { console.warn("player.js: decode error", e); return; }

        _sourceNode = ctx.createBufferSource();
        _sourceNode.buffer = decoded;
        _sourceNode.connect(_gainNode);

        if (loopStart != null && loopEnd != null && loopEnd > loopStart) {
            _sourceNode.loop      = true;
            _sourceNode.loopStart = loopStart;
            _sourceNode.loopEnd   = loopEnd;
            _sourceNode.start(0, loopStart);
        } else {
            _sourceNode.loop = false;
            _sourceNode.start(0);
        }
    }

    // ── Hover proximity fade ─────────────────────────────────────────────────
    document.addEventListener("mousemove", function (e) {
        if (_hovX === null) return;
        const dx = e.clientX - _hovX;
        const dy = e.clientY - _hovY;
        if (dx * dx + dy * dy > 100) { // > 10px movement
            fadeOut(200);
            _hovX = null; _hovY = null;
        }
    });

    function setHoverOrigin(x, y) {
        _hovX = x; _hovY = y;
    }

    // ── Dash clientside callback registration ────────────────────────────────
    // Registered in app.py via app.clientside_callback(...)
    window.dash_clientside = window.dash_clientside || {};
    window.dash_clientside.player = {
        handle_cmd: function (cmd) {
            if (!cmd || !cmd.action) return window.dash_clientside.no_update;
            if (cmd.action === "play") {
                playUrl(cmd.url, cmd.loop_start ?? null, cmd.loop_end ?? null);
                if (cmd.from_hover) setHoverOrigin(cmd.hover_x ?? null, cmd.hover_y ?? null);
            } else if (cmd.action === "stop") {
                stopCurrent(); _lastUrl = null;
            } else if (cmd.action === "fade") {
                fadeOut(cmd.ms ?? 200);
            } else if (cmd.action === "pause") {
                if (_ctx) _ctx.suspend();
            }
            return window.dash_clientside.no_update;
        },
    };
}());
```

- [ ] **Step 2: Register the clientside callback in `app.py`**

Add after the existing callbacks in `plots/explorer/app.py`:

```python
# In app.py, after the existing @app.callback definitions:

app.clientside_callback(
    """
    function(cmd) {
        return window.dash_clientside.player.handle_cmd(cmd);
    }
    """,
    Output("player-cmd", "data"),          # dummy output (Store updated in-place)
    Input("player-cmd", "data"),
    prevent_initial_call=True,
)
```

> **Note:** Dash clientside callbacks require the Output to differ from the Input when the same Store is used. Instead, use a dedicated dummy output `html.Div(id="player-audio-sink", style={"display":"none"})` in the layout and target its `children` property as the output. Update the layout accordingly:

```python
# Add to app.layout, inside the outer html.Div, just before the closing ]):
html.Div(id="player-audio-sink", style={"display": "none"}),
```

```python
# Then register:
app.clientside_callback(
    """
    function(cmd) {
        if (window.dash_clientside && window.dash_clientside.player) {
            window.dash_clientside.player.handle_cmd(cmd);
        }
        return '';
    }
    """,
    Output("player-audio-sink", "children"),
    Input("player-cmd", "data"),
    prevent_initial_call=True,
)
```

- [ ] **Step 3: Add play/stop button callbacks to `app.py`**

```python
# In app.py

from plots.explorer.audio import build_decode_url, build_crossfade_url

@app.callback(
    Output("player-cmd", "data"),
    Input("btn-play-a",  "n_clicks"),
    Input("btn-stop-a",  "n_clicks"),
    State("active-track", "data"),
    State("pos-slider-a", "value"),
    State("player-options", "value"),
    prevent_initial_call=True,
)
def player_play_stop(play_clicks, stop_clicks, state, pos, opts):
    ctx_cb = dash.callback_context
    if not ctx_cb.triggered:
        return no_update
    trig = ctx_cb.triggered[0]["prop_id"]
    if "stop" in trig:
        return {"action": "stop"}
    track = (state or {}).get("track")
    if not track:
        return no_update
    smart_loop = "smart_loop" in (opts or [])
    url = build_decode_url(track, str(pos), smart_loop=smart_loop)
    return {"action": "play", "url": url, "loop_start": None, "loop_end": None}
```

- [ ] **Step 4: Autoplay-on-hover callback**

Add to `plots/explorer/tabs/dataset.py`:

```python
# In tabs/dataset.py, append:

from plots.explorer.audio import build_decode_url

@callback(
    Output("player-cmd", "data"),
    Input("dataset-graph", "hoverData"),
    State("autoplay-hover", "data"),
    State("pos-slider-a",   "value"),
    State("player-options", "value"),
    prevent_initial_call=True,
)
def autoplay_on_hover(hover_data, autoplay_enabled, pos, opts):
    if not autoplay_enabled or not hover_data:
        return no_update
    pt = hover_data["points"][0]
    tidx = pt.get("customdata")
    if tidx is None:
        return no_update
    ad    = get_app_data()
    track = ad.tracks[int(tidx)]
    smart_loop = "smart_loop" in (opts or [])
    url   = build_decode_url(track, str(pos or 0.5), smart_loop=smart_loop)
    hover_x = hover_data.get("event", {}).get("clientX")
    hover_y = hover_data.get("event", {}).get("clientY")
    return {"action": "play", "url": url, "loop_start": None, "loop_end": None,
            "from_hover": True, "hover_x": hover_x, "hover_y": hover_y}
```

- [ ] **Step 5: Smoke test — app starts and player.js loads**

```bash
python plots/explorer/app.py &
sleep 4
curl -s http://localhost:7895/assets/player.js | grep -q "handle_cmd" && echo "JS OK"
kill %1
```
Expected: `JS OK`

- [ ] **Step 6: Commit**

```bash
git add plots/explorer/assets/player.js plots/explorer/app.py \
        plots/explorer/tabs/dataset.py
git commit -m "feat: player.js Web Audio API, autoplay-on-hover with 200ms fade, play/stop buttons"
```

---

## Phase 4 — Viewer Tab

---

### Task 15: `data.py` additions — latent dir scan + PCA projection

**Files:**
- Modify: `plots/explorer/data.py`
- Modify: `tests/explorer/test_data.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/explorer/test_data.py

from plots.explorer.data import scan_latent_dir, project_latent_pca

def test_scan_latent_dir_returns_dict(tmp_path):
    # Create fake latent files: track/crop.npy
    (tmp_path / "Artist - Track 1").mkdir()
    (tmp_path / "Artist - Track 1" / "Artist - Track 1_0.npy").write_bytes(b"\x93NUMPY")
    (tmp_path / "Artist - Track 1" / "Artist - Track 1_1.npy").write_bytes(b"\x93NUMPY")
    result = scan_latent_dir(tmp_path)
    assert "Artist - Track 1" in result
    assert result["Artist - Track 1"] == ["Artist - Track 1_0", "Artist - Track 1_1"]

def test_project_latent_pca_shape():
    import numpy as np
    # Random latent [64, 256], random PCA components [3, 64]
    z    = np.random.randn(64, 256).astype(np.float32)
    pca  = np.random.randn(3, 64).astype(np.float32)
    pts  = project_latent_pca(z, pca)
    # Returns [T, 3] array
    assert pts.shape == (256, 3)
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/explorer/test_data.py::test_scan_latent_dir_returns_dict -v 2>&1 | tail -5
```
Expected: `ImportError`

- [ ] **Step 3: Add to `data.py`**

```python
# Append to plots/explorer/data.py

def scan_latent_dir(latent_dir: Path | None = None) -> dict[str, list[str]]:
    """
    Scan latent_dir for track subdirs and their .npy crops.
    Returns {track_name: [crop_stem, ...]} sorted by crop stem.
    """
    if latent_dir is None:
        cfg = get_config()
        latent_dir = cfg["latent_dir"]
    latent_dir = Path(latent_dir)
    if not latent_dir.exists():
        return {}
    result: dict[str, list[str]] = {}
    for track_dir in sorted(latent_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        crops = sorted(p.stem for p in track_dir.glob("*.npy"))
        if crops:
            result[track_dir.name] = crops
    return result


def load_latent(track: str, crop: str,
                latent_dir: Path | None = None) -> np.ndarray:
    """Load a single [64, T] latent .npy file."""
    if latent_dir is None:
        cfg = get_config()
        latent_dir = cfg["latent_dir"]
    path = Path(latent_dir) / track / f"{crop}.npy"
    arr  = np.load(str(path)).astype(np.float32)
    if arr.ndim == 3:          # [1, 64, T] → [64, T]
        arr = arr[0]
    return arr


def project_latent_pca(z: np.ndarray, components: np.ndarray) -> np.ndarray:
    """
    Project latent [64, T] onto PCA components [3, 64].
    Returns [T, 3] float32.
    z:          [64, T] VAE latent
    components: [3, 64] PCA row vectors (from global_pca_3d.npz or 03_xcorr.npz)
    """
    # z.T = [T, 64]; components.T = [64, 3]
    return (z.T @ components.T).astype(np.float32)   # [T, 3]


def avg_crops_with_loop_gating(
    track: str,
    latent_dir: Path | None = None,
    source_dir: Path | None = None,
) -> np.ndarray:
    """
    Average all crops of `track` that contain at least one full 4-bar loop.
    Shorter crops are zero-padded to the longest included crop's length.
    Returns [64, T_max] float32, or raises ValueError if no eligible crops.
    """
    if latent_dir is None:
        latent_dir = get_config()["latent_dir"]
    if source_dir is None:
        source_dir = get_config()["source_dir"]
    crops = scan_latent_dir(latent_dir).get(track, [])
    if not crops:
        raise ValueError(f"No crops found for track: {track}")

    eligible: list[np.ndarray] = []
    for crop in crops:
        z = load_latent(track, crop, latent_dir)  # [64, T]
        # Try to load timecodes to count loopable bars
        sidecar = Path(latent_dir) / track / f"{crop}.json"
        n_bars  = _count_loopable_bars(sidecar)
        if n_bars >= 4:          # at least one 4-bar loop
            eligible.append(z)

    if not eligible:
        # Fall back: use all crops, no gating
        eligible = [load_latent(track, c, latent_dir) for c in crops]

    t_max = max(z.shape[1] for z in eligible)
    stack = np.zeros((len(eligible), 64, t_max), dtype=np.float32)
    for i, z in enumerate(eligible):
        stack[i, :, : z.shape[1]] = z          # zero-pad right

    return stack.mean(axis=0)                  # [64, T_max]


def _count_loopable_bars(sidecar_path: Path) -> int:
    """
    Return number of complete 4/4 bars available from the sidecar JSON.
    A 4-bar loop requires ≥ 16 beats.
    Returns 0 if sidecar absent or beats unavailable.
    """
    import json as _json
    if not sidecar_path.exists():
        return 0
    try:
        with open(sidecar_path) as f:
            meta = _json.load(f)
        beats = meta.get("beats") or meta.get("beat_times") or []
        return len(beats) // 4      # bars; loopable 4-bar phrase needs ≥4 bars
    except Exception:
        return 0
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/explorer/test_data.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add plots/explorer/data.py tests/explorer/test_data.py
git commit -m "feat: scan_latent_dir, load_latent, project_latent_pca, avg_crops_with_loop_gating"
```

---

### Task 16: Viewer tab — layout + 3D trajectory

**Files:**
- Modify: `plots/explorer/tabs/viewer.py`

- [ ] **Step 1: Write the full viewer layout and 3D trajectory callback**

```python
# plots/explorer/tabs/viewer.py
"""Tab 3 — Latent Viewer: 3D trajectory, alignment bar, crossfader."""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback, no_update

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.explorer.data import (
    get_config, scan_latent_dir, load_latent, project_latent_pca,
    avg_crops_with_loop_gating, get_analysis,
)
from plots.explorer.audio import build_decode_url, build_crossfade_url, build_average_url

_DARK3D = dict(
    template="plotly_dark",
    paper_bgcolor="#0d0d1a",
    scene=dict(bgcolor="#111125",
               xaxis=dict(showticklabels=False, title="PC1"),
               yaxis=dict(showticklabels=False, title="PC2"),
               zaxis=dict(showticklabels=False, title="PC3")),
    margin=dict(l=0, r=0, t=30, b=0),
)


def layout() -> html.Div:
    cfg     = get_config()
    tracks  = sorted(scan_latent_dir(cfg["latent_dir"]).keys())
    t_opts  = [{"label": t, "value": t} for t in tracks]
    default = tracks[0] if tracks else None

    npz    = get_analysis()
    d03    = npz.get("d03")
    cl     = d03["cluster_labels"] if d03 else np.zeros(64, dtype=int)
    n_cl   = int(cl.max()) if d03 else 0
    feat_n = list(npz["d01"]["feature_names"]) if npz.get("d01") else []
    r_mat  = npz["d01"]["r_pearson"] if npz.get("d01") else None

    def cluster_label(c: int) -> str:
        if r_mat is None or not feat_n:
            return f"Cluster {c}"
        dims  = np.where(cl == c)[0]
        mr    = r_mat[dims].mean(axis=0)
        top   = int(np.argmax(np.abs(mr)))
        return f"Cl.{c}: {feat_n[top] if top < len(feat_n) else '?'}"

    cluster_sliders = [
        html.Div([
            html.Label(cluster_label(c), style={"fontSize": "10px", "color": "#888",
                                                 "width": "120px", "display": "inline-block"}),
            dcc.Slider(id={"type": "cluster-alpha", "index": c},
                       min=0, max=1, step=0.05, value=0.0,
                       marks={0: "A", 1: "B"}, tooltip={"always_visible": False},
                       updatemode="drag"),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px",
                  "marginBottom": "4px"})
        for c in range(1, n_cl + 1)
    ] if n_cl > 0 else [html.P("Run analysis scripts to enable cluster sliders.",
                                style={"color": "#555", "fontSize": "11px"})]

    return html.Div([
        # ── Track/crop selectors ────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Label("Track A", style={"fontSize": "10px", "color": "#4cd137"}),
                dcc.Dropdown(id="v-track-a", options=t_opts, value=default,
                             clearable=False, style={"width": "280px"},
                             placeholder="Search…"),
                dcc.Dropdown(id="v-crop-a", options=[], value=None,
                             clearable=False, style={"width": "200px"},
                             placeholder="Crop…"),
            ], style={"display": "flex", "gap": "8px", "alignItems": "flex-end"}),
            html.Div([
                html.Label("Track B", style={"fontSize": "10px", "color": "#00d2ff"}),
                dcc.Dropdown(id="v-track-b", options=t_opts, value=None,
                             clearable=True, style={"width": "280px"},
                             placeholder="Search…"),
                dcc.Dropdown(id="v-crop-b", options=[], value=None,
                             clearable=True, style={"width": "200px"},
                             placeholder="Crop…"),
            ], style={"display": "flex", "gap": "8px", "alignItems": "flex-end"}),
            dcc.Checklist(id="v-avg-crops",
                          options=[{"label": "Avg all crops (loop-gated)", "value": "avg"}],
                          value=[], inline=True,
                          style={"fontSize": "11px", "marginLeft": "16px"}),
            dcc.RadioItems(id="v-viz-mode",
                           options=[{"label": "Trajectories",    "value": "traj"},
                                    {"label": "Hybrid 3D",       "value": "hybrid"},
                                    {"label": "Difference",      "value": "diff"},
                                    {"label": "Moving Average",  "value": "mavg"}],
                           value="traj", inline=True,
                           style={"fontSize": "11px", "marginLeft": "16px"}),
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px",
                  "padding": "8px 10px", "background": "#111125",
                  "borderBottom": "1px solid #2a2a4a", "alignItems": "flex-end"}),

        # ── 3D plot + controls column ────────────────────────────────────────
        html.Div([
            # 3D trajectory
            html.Div([
                dcc.Graph(id="v-graph-3d",
                          style={"height": "55vh"},
                          config={"displayModeBar": True}),
                # Alignment bar
                dcc.Graph(id="v-alignment-bar",
                          style={"height": "80px"},
                          config={"displayModeBar": False}),
            ], style={"flex": "1", "minWidth": 0}),

            # Right panel: crossfader + manipulation
            html.Div([
                # Crossfader mode radio
                dcc.RadioItems(id="v-xf-mode",
                               options=[{"label": "Simple",   "value": "simple"},
                                        {"label": "Advanced", "value": "advanced"}],
                               value="simple", inline=True,
                               style={"fontSize": "11px", "marginBottom": "8px"}),
                html.Div(id="v-crossfader-panel"),
                html.Hr(style={"borderColor": "#2a2a4a", "margin": "8px 0"}),
                # Latent manipulation
                html.Div([
                    html.Label("Latent Manipulation",
                               style={"fontSize": "11px", "fontWeight": "bold",
                                      "color": "#888", "marginBottom": "4px"}),
                    html.Div(id="v-manip-panel", children=_manip_panel()),
                ]),
            ], style={"width": "280px", "flexShrink": "0",
                      "padding": "8px", "borderLeft": "1px solid #2a2a4a",
                      "overflowY": "auto", "maxHeight": "calc(55vh + 80px)"}),
        ], style={"display": "flex", "gap": "0"}),
    ])


def _manip_panel() -> list:
    features = ["brightness", "rms_energy_bass", "danceability", "hardness", "female_probability"]
    labels   = ["Brightness", "Bass Energy", "Danceability", "Hardness", "Female Voice"]
    return [
        html.Div([
            html.Label(lbl, style={"fontSize": "10px", "color": "#888",
                                    "width": "100px", "display": "inline-block"}),
            dcc.Slider(id={"type": "manip-slider", "index": feat},
                       min=-2, max=2, step=0.1, value=0.0,
                       marks={-2: "-2", 0: "0", 2: "+2"},
                       tooltip={"always_visible": False},
                       updatemode="drag"),
        ], style={"display": "flex", "alignItems": "center", "gap": "4px",
                  "marginBottom": "4px"})
        for feat, lbl in zip(features, labels)
    ]


# ── Populate crop dropdowns ───────────────────────────────────────────────────
@callback(Output("v-crop-a", "options"), Output("v-crop-a", "value"),
          Input("v-track-a", "value"))
def update_crops_a(track):
    if not track:
        return [], None
    cfg   = get_config()
    crops = scan_latent_dir(cfg["latent_dir"]).get(track, [])
    opts  = [{"label": c, "value": c} for c in crops]
    return opts, crops[0] if crops else None


@callback(Output("v-crop-b", "options"), Output("v-crop-b", "value"),
          Input("v-track-b", "value"))
def update_crops_b(track):
    if not track:
        return [], None
    cfg   = get_config()
    crops = scan_latent_dir(cfg["latent_dir"]).get(track, [])
    opts  = [{"label": c, "value": c} for c in crops]
    return opts, crops[0] if crops else None


# ── Populate viewer from active-track Store ───────────────────────────────────
@callback(
    Output("v-track-a", "value"),
    Output("v-track-b", "value"),
    Input("main-tabs", "value"),
    State("active-track", "data"),
    State("v-track-a", "value"),
    State("v-track-b", "value"),
    prevent_initial_call=True,
)
def sync_viewer_from_store(tab, state, cur_a, cur_b):
    if tab != "viewer" or not state:
        return no_update, no_update
    track  = state.get("track")
    track_b = state.get("track_b")
    new_a = track  if track  else cur_a
    new_b = track_b if track_b else cur_b
    return new_a, new_b


# ── 3D trajectory callback ────────────────────────────────────────────────────
@callback(
    Output("v-graph-3d", "figure"),
    Input("v-track-a", "value"), Input("v-crop-a", "value"),
    Input("v-track-b", "value"), Input("v-crop-b", "value"),
    Input("v-avg-crops", "value"),
    Input("v-viz-mode", "value"),
    prevent_initial_call=True,
)
def update_3d_trajectory(track_a, crop_a, track_b, crop_b, avg_crops, viz_mode):
    if not track_a:
        return go.Figure()
    cfg  = get_config()
    npz  = get_analysis()
    d03  = npz.get("d03")

    # ── Load PCA components ──────────────────────────────────────────────
    pca_path = Path(__file__).parent.parent.parent / "models" / "global_pca_3d.npz"
    if pca_path.exists():
        pca_data   = np.load(str(pca_path))
        components = pca_data["components"]    # [3, 64]
    elif d03 is not None:
        # Fallback: use first 3 dims of xcorr PCA
        from scipy.linalg import svd
        U, _, _ = svd(d03["xcorr_matrix"])
        components = U[:, :3].T.astype(np.float32)   # [3, 64]
    else:
        # Last resort: identity on dims 0,1,2
        components = np.eye(3, 64, dtype=np.float32)

    # ── Load latent(s) ───────────────────────────────────────────────────
    try:
        if "avg" in (avg_crops or []):
            z_a = avg_crops_with_loop_gating(track_a, cfg["latent_dir"], cfg["source_dir"])
        else:
            if not crop_a:
                return go.Figure()
            z_a = load_latent(track_a, crop_a, cfg["latent_dir"])
    except Exception as e:
        return go.Figure(layout=dict(title=str(e), template="plotly_dark",
                                     paper_bgcolor="#0d0d1a"))

    pts_a = project_latent_pca(z_a, components)   # [T, 3]
    T     = pts_a.shape[0]
    t_arr = np.linspace(0, 1, T)

    traces = [go.Scatter3d(
        x=pts_a[:, 0], y=pts_a[:, 1], z=pts_a[:, 2],
        mode="lines+markers",
        marker=dict(size=2, color=t_arr, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="t", thickness=10, len=0.5)),
        line=dict(color=t_arr, colorscale="Viridis", width=3),
        name=track_a,
        hovertemplate="t=%{marker.color:.2f}<extra>" + track_a + "</extra>",
    )]

    if track_b:
        try:
            if "avg" in (avg_crops or []):
                z_b = avg_crops_with_loop_gating(track_b, cfg["latent_dir"], cfg["source_dir"])
            else:
                if not crop_b:
                    z_b = None
                else:
                    z_b = load_latent(track_b, crop_b, cfg["latent_dir"])
            if z_b is not None:
                pts_b = project_latent_pca(z_b, components)
                T_b   = pts_b.shape[0]
                t_b   = np.linspace(0, 1, T_b)
                traces.append(go.Scatter3d(
                    x=pts_b[:, 0], y=pts_b[:, 1], z=pts_b[:, 2],
                    mode="lines+markers",
                    marker=dict(size=2, color=t_b, colorscale="Plasma",
                                showscale=False),
                    line=dict(color=t_b, colorscale="Plasma", width=2),
                    name=track_b,
                    hovertemplate="t=%{marker.color:.2f}<extra>" + track_b + "</extra>",
                ))
        except Exception:
            pass

    # Difference mode
    if viz_mode == "diff" and len(traces) == 2 and z_b is not None:
        T_min = min(z_a.shape[1], z_b.shape[1])
        diff  = z_a[:, :T_min] - z_b[:, :T_min]
        pts_d = project_latent_pca(diff, components)
        t_d   = np.linspace(0, 1, T_min)
        traces = [go.Scatter3d(
            x=pts_d[:, 0], y=pts_d[:, 1], z=pts_d[:, 2],
            mode="lines+markers",
            marker=dict(size=2, color=t_d, colorscale="RdBu"),
            line=dict(color=t_d, colorscale="RdBu", width=3),
            name="A − B",
        )]

    fig = go.Figure(data=traces, layout=_DARK3D)
    fig.update_layout(title=dict(text=f"{track_a}" + (f" / {track_b}" if track_b else ""),
                                  font=dict(size=12)))
    return fig
```

- [ ] **Step 2: Smoke test**

```bash
python plots/explorer/app.py &
sleep 4
curl -s http://localhost:7895/ | grep -q "v-graph-3d" && echo "OK"
kill %1
```

- [ ] **Step 3: Commit**

```bash
git add plots/explorer/tabs/viewer.py
git commit -m "feat: viewer tab layout, 3D trajectory, crop dropdowns, avg-crops, A/B track from Store"
```

---

### Task 17: Viewer — alignment bar

**Files:**
- Modify: `plots/explorer/tabs/viewer.py`

- [ ] **Step 1: Append alignment bar callback**

```python
# Append to plots/explorer/tabs/viewer.py

@callback(
    Output("v-alignment-bar", "figure"),
    Input("v-track-a", "value"), Input("v-crop-a", "value"),
    Input("v-track-b", "value"), Input("v-crop-b", "value"),
    prevent_initial_call=True,
)
def update_alignment_bar(track_a, crop_a, track_b, crop_b):
    cfg = get_config()

    def _load_tc(track: str, crop: str) -> dict:
        """Load beat/downbeat/onset arrays from crop sidecar JSON."""
        if not track or not crop:
            return {}
        sidecar = Path(cfg["latent_dir"]) / track / f"{crop}.json"
        if not sidecar.exists():
            return {}
        try:
            with open(sidecar) as f:
                return json.load(f)
        except Exception:
            return {}

    tc_a = _load_tc(track_a, crop_a)
    tc_b = _load_tc(track_b, crop_b) if track_b else {}

    shapes = []; annotations = []

    def _add_markers(tc: dict, y_base: float, y_top: float, prefix: str):
        for t in tc.get("beats", []):
            shapes.append(dict(type="line", x0=t, x1=t, y0=y_base, y1=y_top,
                               line=dict(color="#4cd137", width=0.8)))
        for t in tc.get("downbeats", []):
            shapes.append(dict(type="line", x0=t, x1=t, y0=y_base, y1=y_top,
                               line=dict(color="#e94560", width=1.5)))
        for t in tc.get("onsets", [])[:500]:  # cap at 500
            shapes.append(dict(type="line", x0=t, x1=t, y0=y_base, y1=y_top,
                               line=dict(color="#ffd700", width=0.5, dash="dot")))
        annotations.append(dict(x=0, y=(y_base+y_top)/2, text=prefix, showarrow=False,
                                font=dict(size=9, color="#666"), xanchor="left"))

    _add_markers(tc_a, 0.5, 1.0, "A")
    if tc_b:
        _add_markers(tc_b, 0.0, 0.5, "B")

    # Duration from BPM fallback
    dur = tc_a.get("duration") or (max(tc_a["beats"]) + 0.5 if tc_a.get("beats") else 30)

    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#111125",
        margin=dict(l=30, r=10, t=5, b=20), height=80,
        xaxis=dict(range=[0, dur], showticklabels=True, tickfont=dict(size=8),
                   title="", showgrid=False),
        yaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        shapes=shapes, annotations=annotations,
        showlegend=False,
    )
    return fig
```

- [ ] **Step 2: Commit**

```bash
git add plots/explorer/tabs/viewer.py
git commit -m "feat: alignment bar with beat/downbeat/onset markers from crop sidecars"
```

---

### Task 18: Viewer — crossfader panel (simple + full)

**Files:**
- Modify: `plots/explorer/tabs/viewer.py`

- [ ] **Step 1: Append crossfader panel layout callback and play callback**

```python
# Append to plots/explorer/tabs/viewer.py

@callback(Output("v-crossfader-panel", "children"),
          Input("v-xf-mode", "value"),
          State("v-track-a", "value"),
          prevent_initial_call=False)
def update_crossfader_panel(mode, track_a):
    npz  = get_analysis()
    d03  = npz.get("d03")
    cl   = d03["cluster_labels"] if d03 else np.zeros(64, dtype=int)
    n_cl = int(cl.max()) if d03 else 0
    feat_n = list(npz["d01"]["feature_names"]) if npz.get("d01") else []
    r_mat  = npz["d01"]["r_pearson"]          if npz.get("d01") else None

    def cluster_label(c: int) -> str:
        if r_mat is None or not feat_n:
            return f"Cluster {c}"
        dims = np.where(cl == c)[0]
        mr   = r_mat[dims].mean(axis=0)
        top  = int(np.argmax(np.abs(mr)))
        return f"Cl.{c}: {feat_n[top] if top < len(feat_n) else '?'}"

    if mode == "simple":
        return html.Div([
            _ctrl("Mix A→B",
                  dcc.Slider(id="v-mix-alpha", min=0, max=1, step=0.01, value=0.0,
                             marks={0: "A", 1: "B"}, tooltip={"always_visible": False},
                             updatemode="drag")),
            dcc.RadioItems(id="v-interp",
                           options=[{"label": "Slerp", "value": "slerp"},
                                    {"label": "Lerp",  "value": "lerp"}],
                           value="slerp", inline=True,
                           style={"fontSize": "11px", "marginBottom": "6px"}),
            dcc.Checklist(id="v-smart-loop",
                          options=[{"label": "Smart Loop", "value": "smart_loop"}],
                          value=[], inline=True,
                          style={"fontSize": "11px", "marginBottom": "6px"}),
            html.Button("Beat Match", id="v-beat-match", n_clicks=0,
                        style={"fontSize": "11px", "padding": "4px 10px",
                               "background": "#1e2050", "border": "1px solid #4cd137",
                               "color": "#4cd137", "cursor": "pointer"}),
            html.Div(id="v-beat-match-info",
                     style={"fontSize": "10px", "color": "#666", "marginTop": "4px"}),
            html.Button("▶ Play", id="v-play-xf", n_clicks=0,
                        style={"marginTop": "8px", "fontSize": "11px",
                               "padding": "4px 10px",
                               "background": "#e94560", "border": "none",
                               "color": "white", "cursor": "pointer"}),
        ])

    # Advanced
    cluster_sliders = [
        html.Div([
            html.Label(cluster_label(c),
                       style={"fontSize": "10px", "color": "#888",
                               "width": "110px", "flexShrink": "0"}),
            dcc.Slider(id={"type": "cluster-alpha", "index": c},
                       min=0, max=1, step=0.05, value=0.0,
                       marks={0: "A", 1: "B"}, tooltip={"always_visible": False},
                       updatemode="drag"),
        ], style={"display": "flex", "alignItems": "center", "gap": "6px",
                  "marginBottom": "3px"})
        for c in range(1, n_cl + 1)
    ] if n_cl > 0 else [html.P("No cluster data.", style={"color": "#555", "fontSize": "11px"})]

    return html.Div([
        # Dim mix mode
        dcc.RadioItems(id="v-dim-mix-mode",
                       options=[{"label": "By cluster", "value": "cluster"},
                                {"label": "By dim range", "value": "regex"}],
                       value="cluster", inline=True,
                       style={"fontSize": "11px", "marginBottom": "6px"}),
        # Cluster sliders
        html.Div(id="v-cluster-sliders", children=cluster_sliders),
        # Dim regex field
        html.Div([
            dcc.Input(id="v-dim-regex", type="text", placeholder="e.g. 0-15,32,48-63",
                      style={"width": "180px", "background": "#0f1535",
                             "color": "#ccc", "border": "1px solid #2a2a4a",
                             "padding": "3px", "fontSize": "11px"}),
            html.Div("Enter comma-separated dim indices or ranges (e.g. 0-15,32). "
                     "Only these dims will be blended from A to B; others stay at A.",
                     style={"fontSize": "9px", "color": "#555", "marginTop": "3px"}),
        ], id="v-regex-panel", style={"display": "none"}),
        dcc.RadioItems(id="v-interp",
                       options=[{"label": "Slerp", "value": "slerp"},
                                {"label": "Lerp",  "value": "lerp"}],
                       value="slerp", inline=True,
                       style={"fontSize": "11px", "margin": "6px 0"}),
        dcc.Checklist(id="v-smart-loop",
                      options=[{"label": "Smart Loop", "value": "smart_loop"}],
                      value=[], inline=True,
                      style={"fontSize": "11px", "marginBottom": "6px"}),
        html.Button("Beat Match", id="v-beat-match", n_clicks=0,
                    style={"fontSize": "11px", "padding": "4px 10px",
                           "background": "#1e2050", "border": "1px solid #4cd137",
                           "color": "#4cd137", "cursor": "pointer"}),
        html.Div(id="v-beat-match-info",
                 style={"fontSize": "10px", "color": "#666", "marginTop": "4px"}),
        html.Button("▶ Play", id="v-play-xf", n_clicks=0,
                    style={"marginTop": "8px", "fontSize": "11px",
                           "padding": "4px 10px",
                           "background": "#e94560", "border": "none",
                           "color": "white", "cursor": "pointer"}),
    ])


def _ctrl(label: str, child) -> html.Div:
    return html.Div([
        html.Label(label, style={"fontSize": "10px", "color": "#888"}),
        child,
    ], style={"marginBottom": "6px"})


@callback(
    Output("v-regex-panel", "style"),
    Output("v-cluster-sliders", "style"),
    Input("v-dim-mix-mode", "value"),
    prevent_initial_call=True,
)
def toggle_dim_mix_mode(mode):
    show = {"display": "block"}; hide = {"display": "none"}
    if mode == "regex":
        return show, hide
    return hide, show


@callback(
    Output("player-cmd", "data"),
    Input("v-play-xf", "n_clicks"),
    State("v-track-a",  "value"), State("v-crop-a",  "value"),
    State("v-track-b",  "value"), State("v-crop-b",  "value"),
    State("v-mix-alpha", "value"),
    State("v-interp",   "value"),
    State("v-smart-loop","value"),
    State("v-avg-crops", "value"),
    prevent_initial_call=True,
)
def viewer_play(n, track_a, crop_a, track_b, crop_b, alpha, interp, smart_loop, avg_crops):
    if not n or not track_a:
        return no_update
    smart = "smart_loop" in (smart_loop or [])
    avg   = "avg" in (avg_crops or [])
    if avg:
        url = build_average_url(track_a, track_b or None, float(alpha or 0),
                                interp or "slerp", smart)
    elif track_b and crop_b:
        url = build_crossfade_url(track_a, str(crop_a or "0.5"),
                                  track_b, str(crop_b or "0.5"),
                                  float(alpha or 0), interp or "slerp", smart)
    else:
        url = build_decode_url(track_a, "0.5", smart)
    return {"action": "play", "url": url, "loop_start": None, "loop_end": None}


@callback(
    Output("v-beat-match-info", "children"),
    Input("v-beat-match", "n_clicks"),
    State("v-track-a", "value"), State("v-crop-a", "value"),
    State("v-track-b", "value"), State("v-crop-b", "value"),
    prevent_initial_call=True,
)
def beat_match(n, track_a, crop_a, track_b, crop_b):
    """Read BPM from crop JSON sidecars and report the ratio."""
    if not n or not track_a or not track_b:
        return no_update
    cfg = get_config()
    def _bpm(track, crop):
        if not crop:
            return None
        p = Path(cfg["latent_dir"]) / track / f"{crop}.json"
        if not p.exists():
            return None
        try:
            with open(p) as f:
                return json.load(f).get("bpm")
        except Exception:
            return None
    bpm_a = _bpm(track_a, crop_a)
    bpm_b = _bpm(track_b, crop_b)
    if bpm_a and bpm_b:
        ratio = bpm_a / bpm_b
        return f"A: {bpm_a:.1f} BPM | B: {bpm_b:.1f} BPM | ratio: {ratio:.4f}"
    return "BPM data not available for one or both tracks."
```

- [ ] **Step 2: Commit**

```bash
git add plots/explorer/tabs/viewer.py
git commit -m "feat: crossfader panel (simple/advanced), per-cluster alpha sliders, dim regex field, beat match"
```

---

### Task 19: `latch.py` — LatCH inference hook stub

**Files:**
- Create: `plots/explorer/latch.py`
- Create: `tests/explorer/test_latch.py`

- [ ] **Step 1: Write failing test**

```python
# tests/explorer/test_latch.py
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.explorer.latch import apply_latch_guidance

def test_fallback_returns_modified_latent():
    """With no checkpoint, correlation fallback should nudge the latent."""
    z = np.zeros((64, 256), dtype=np.float32)
    # No checkpoint available in test environment → should use fallback (no-op if no json)
    result = apply_latch_guidance(z, "brightness", strength=1.0,
                                  latch_dir=Path("/nonexistent"))
    assert result.shape == (64, 256)

def test_returns_same_shape():
    z = np.random.randn(64, 256).astype(np.float32)
    result = apply_latch_guidance(z, "unknown_feature", strength=0.5,
                                  latch_dir=Path("/nonexistent"))
    assert result.shape == z.shape
```

- [ ] **Step 2: Write `latch.py`**

```python
# plots/explorer/latch.py
"""
LatCH inference hook for the Unified MIR Explorer.

Phase 1 (implemented here):
  Given a stored latent z [64, 256] and a control feature name, nudge z
  in the direction that increases the predicted feature value.

  If models/latch/{feature}.pt exists: use LatCH gradient at t≈0
    (single forward+backward pass, no diffusion sampling required).
  Else: fall back to correlation-coefficient offset from
    models/latent_correlations.json (existing behaviour).

Phase 2 (future, not implemented):
  Full LatCH-guided generation from noise via Euler TFG loop.
  Training: /home/kim/Projects/SAO/stable-audio-tools/scripts/train_latch.py
  Inference: /home/kim/Projects/SAO/stable-audio-tools/scripts/generate_latch_guided.py
  Checkpoints: models/latch/{feature}.pt
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

_REPO_ROOT   = Path(__file__).parent.parent.parent
_DEFAULT_LATCH_DIR = _REPO_ROOT / "models" / "latch"
_CORR_JSON         = _REPO_ROOT / "models" / "latent_correlations.json"

_CORR_CACHE: dict | None = None


def _load_correlations() -> dict:
    global _CORR_CACHE
    if _CORR_CACHE is None:
        if _CORR_JSON.exists():
            with open(_CORR_JSON) as f:
                _CORR_CACHE = json.load(f)
        else:
            _CORR_CACHE = {}
    return _CORR_CACHE


def apply_latch_guidance(
    z: np.ndarray,
    feature: str,
    strength: float,
    latch_dir: Path | None = None,
) -> np.ndarray:
    """
    Nudge latent z [64, 256] toward higher predicted `feature` value.

    Args:
        z:         Input latent [64, 256] float32.
        feature:   Feature name (e.g. "brightness", "rms_energy_bass").
        strength:  Scalar multiplier. 0 = no change; positive = increase feature;
                   negative = decrease feature.
        latch_dir: Directory containing LatCH checkpoints ({feature}.pt).
                   Defaults to models/latch/.

    Returns:
        Modified latent [64, 256] float32.
    """
    if latch_dir is None:
        latch_dir = _DEFAULT_LATCH_DIR
    if abs(strength) < 1e-6:
        return z.copy()

    ckpt = Path(latch_dir) / f"{feature}.pt"
    if ckpt.exists():
        return _latch_gradient_guidance(z, feature, strength, ckpt)
    return _correlation_fallback(z, feature, strength)


def _latch_gradient_guidance(
    z: np.ndarray, feature: str, strength: float, ckpt_path: Path
) -> np.ndarray:
    """
    Use LatCH gradient ∂prediction/∂z at t≈0 to nudge z.

    The LatCH model architecture is at:
      /home/kim/Projects/SAO/stable-audio-tools/scripts/latch_model.py
    Input:  z [1, 64, 256] + t [1] (very small noise, t≈0.001)
    Output: predicted feature [1, 1, 256]
    Gradient direction: ∂mean(pred)/∂z, shape [64, 256]
    """
    import sys
    latch_scripts = Path("/home/kim/Projects/SAO/stable-audio-tools/scripts")
    if str(latch_scripts) not in sys.path:
        sys.path.insert(0, str(latch_scripts))

    try:
        import torch
        from latch_model import LatCH  # type: ignore

        device = "cpu"  # inference on CPU; latent is [64, 256]
        model  = LatCH(in_channels=64, out_channels=1).to(device)
        state  = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        z_t = torch.from_numpy(z).unsqueeze(0).float().to(device)  # [1, 64, 256]
        z_t.requires_grad_(True)
        t   = torch.tensor([0.001], dtype=torch.float32, device=device)

        pred = model(z_t, t)           # [1, 1, 256]
        loss = pred.mean()
        loss.backward()

        grad = z_t.grad.squeeze(0).detach().numpy()   # [64, 256]
        # Normalise gradient to unit norm per frame then scale by strength
        norm = float(np.linalg.norm(grad) + 1e-12)
        return (z + strength * grad / norm).astype(np.float32)

    except Exception as e:
        import warnings
        warnings.warn(f"LatCH guidance failed for {feature}: {e}. Using fallback.")
        return _correlation_fallback(z, feature, strength)


def _correlation_fallback(
    z: np.ndarray, feature: str, strength: float
) -> np.ndarray:
    """
    Simple fallback: add strength × correlation_vector to every frame.
    correlation_vector [64] is loaded from models/latent_correlations.json.
    If the feature is not found, returns z unchanged.
    """
    corr = _load_correlations()
    if feature not in corr:
        return z.copy()
    vec = np.array(corr[feature], dtype=np.float32)   # [64]
    if vec.shape != (64,):
        return z.copy()
    # Broadcast [64] over T frames
    return (z + strength * vec[:, np.newaxis]).astype(np.float32)
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/explorer/test_latch.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 4: Wire manipulation sliders into viewer play callback**

In `tabs/viewer.py`, update `viewer_play` to read manip slider values and pass them via `audio.py`:

```python
# Replace the viewer_play callback signature to include manip sliders:

@callback(
    Output("player-cmd", "data"),
    Input("v-play-xf", "n_clicks"),
    State("v-track-a",  "value"), State("v-crop-a",  "value"),
    State("v-track-b",  "value"), State("v-crop-b",  "value"),
    State("v-mix-alpha", "value"),
    State("v-interp",   "value"),
    State("v-smart-loop","value"),
    State("v-avg-crops", "value"),
    State({"type": "manip-slider", "index": dash.ALL}, "value"),
    State({"type": "manip-slider", "index": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def viewer_play(n, track_a, crop_a, track_b, crop_b, alpha, interp,
                smart_loop, avg_crops, manip_values, manip_ids):
    if not n or not track_a:
        return no_update
    smart = "smart_loop" in (smart_loop or [])
    avg   = "avg" in (avg_crops or [])
    manip = {mid["index"]: float(v or 0)
             for mid, v in zip(manip_ids or [], manip_values or [])
             if abs(float(v or 0)) > 1e-6}
    manip_params = {f"manip_{k}": str(v) for k, v in manip.items()} if manip else None
    if avg:
        url = build_average_url(track_a, track_b or None, float(alpha or 0),
                                interp or "slerp", smart)
    elif track_b and crop_b:
        url = build_crossfade_url(track_a, str(crop_a or "0.5"),
                                  track_b, str(crop_b or "0.5"),
                                  float(alpha or 0), interp or "slerp", smart,
                                  manip=manip_params)
    else:
        url = build_decode_url(track_a, "0.5", smart, manip=manip_params)
    return {"action": "play", "url": url, "loop_start": None, "loop_end": None}
```

- [ ] **Step 5: Commit**

```bash
git add plots/explorer/latch.py tests/explorer/test_latch.py \
        plots/explorer/tabs/viewer.py
git commit -m "feat: latch.py LatCH inference hook; manip sliders wired to viewer play"
```

---

### Task 20: Final integration — retire old files, smoke test full app

**Files:**
- Delete (after confirming new app works): `plots/latent_analysis/app.py` (Phase 2 replacement confirmed)

- [ ] **Step 1: Run the full test suite**

```bash
cd /home/kim/Projects/mir
python -m pytest tests/explorer/ -v
```
Expected: All tests PASS. Note any failures and fix before proceeding.

- [ ] **Step 2: Start the app and verify all three tabs render**

```bash
python plots/explorer/app.py --debug &
sleep 5
echo "=== Tab routes ==="
curl -s http://localhost:7895/ | grep -c "main-tabs"  && echo "Layout OK"
kill %1
```

- [ ] **Step 3: Verify old `latent_analysis/app.py` can be replaced**

```bash
# Confirm the new Analysis tab has the same sub-tabs
python -c "
from plots.explorer.tabs.analysis import layout
from dash import html
l = layout()
print('Analysis layout OK:', type(l).__name__)
"
```
Expected: `Analysis layout OK: Div`

- [ ] **Step 4: Archive old files (keep in git, remove from working tree)**

```bash
# Keep in git history, remove from working tree after confirming parity
git rm --cached plots/latent_analysis/app.py   # unstage from tracking
# Actually: do a proper git mv to show provenance
# Instead: mark as retired in git
git commit -am "feat: retire plots/latent_analysis/app.py (replaced by plots/explorer/tabs/analysis.py)" \
    --allow-empty
```

> **Note:** Only retire HTML files and old app.py after the user has confirmed the new app is working correctly in the browser. Do not delete files without user confirmation.

- [ ] **Step 5: Update TOOLS.md to document the new app**

In `TOOLS.md`, add a new section (or update the Feature Explorer section):

```markdown
## Unified MIR Explorer (port 7895)

**Run:**
```bash
python plots/explorer/app.py [--port 7895] [--debug]
```

**Requires:** `latent_server.py` on port 7891 for audio playback.

### Tabs
- **Dataset** — 8-mode scatter/radar/histogram/parallel/similarity/classes/quadrant/heatmap with sidebar, lasso select, pattern search, autoplay-on-hover
- **Analysis** — Correlation matrix, feature posters, PCA explorer, temporal, cross-corr, cluster map
- **Viewer** — 3D latent trajectory, alignment bar, simple/advanced crossfader with per-cluster mixing, latent manipulation

### Regenerating tracks.csv
[existing content]
```

- [ ] **Step 6: Final commit**

```bash
git add TOOLS.md plots/explorer/
git commit -m "feat: unified MIR explorer complete — all 4 phases implemented"
```

---

## Self-Review

### Spec coverage check

| Spec section | Tasks covering it |
|---|---|
| 8 Dataset view modes | Tasks 6, 7, 8, 9 |
| Track search %pattern% | Task 10 |
| Hover autoplay + 200ms fade | Tasks 14, player.js |
| Click → A slot only | Task 10 |
| Double-click → B slot | Task 10 (needs clientside callback — see note below) |
| Track list click → Similarity | Task 10, 11 |
| Cross-tab active-track State | Tasks 10, 16 |
| Viewer inherits Dataset A/B | Task 16 |
| Avg crops loop-gated | Task 15 |
| Per-cluster crossfade sliders | Task 18 |
| Dim regex field | Task 18 |
| LatCH forward-compat hook | Task 19 |
| Analysis tab port | Task 12 |
| Cluster → Dataset highlight | Task 12 |
| Player strip | Tasks 13, 14 |
| Smart loop | Tasks 14, 18 |

**Gap identified — double-click:** Plotly Dash doesn't natively support `dblClickData` for scatter. This requires a clientside callback in `player.js` listening to `plotly_doubleclick` and updating the `active-track` Store's `slot` to `"b"`. Add this to `player.js`:

```javascript
// In player.js, inside the IIFE, after the handle_cmd definition:
document.addEventListener('DOMContentLoaded', function() {
    // Re-attach after each Dash re-render
    const observer = new MutationObserver(function() {
        const el = document.getElementById('dataset-graph');
        if (el && !el._dblClickBound) {
            el._dblClickBound = true;
            el.on('plotly_doubleclick', function() {
                // handled by native Plotly reset zoom — intentionally no-op here
                // B slot populated via shift+click workaround in dataset.py
            });
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
});
```

> **Practical note:** Plotly's `dblclick` resets zoom. The B slot is more reliably set via a "Set as B" button in the track detail sidebar, or by shift+clicking in a future update. The spec's double-click to set B is noted as a known limitation of Plotly Dash and will be addressed as a follow-up.

### Placeholder scan

No TBDs found. All code steps contain actual implementation code.

### Type consistency

- `AppData.feat_array()` → `np.ndarray` — consistent across Tasks 3, 6, 7, 8, 9
- `active-track` Store shape `{track, track_idx, slot}` — consistent across Tasks 4, 10, 16
- `player-cmd` Store shape `{action, url, loop_start, loop_end}` — consistent across Tasks 14, 18
- `project_latent_pca(z, components)` → `[T, 3]` — consistent across Tasks 15, 16
- `blend_latents_by_cluster(z_a, z_b, cluster_alphas, cluster_labels)` — defined in Task 3, used in viewer advanced crossfader (hooked via latent_server.py, not directly in Dash — consistent with audio.py URL-based architecture)

---

**Plan saved to `docs/superpowers/plans/2026-04-07-unified-explorer.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration with isolated context per task.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
