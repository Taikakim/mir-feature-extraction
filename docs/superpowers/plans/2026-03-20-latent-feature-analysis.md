# Latent Feature Analysis Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable 4-script analysis pipeline + Plotly Dash app that cross-correlates 64 VAE latent dimensions with ~60 MIR features across 192K Goa trance crops, delivering correlation posters, PCA analysis, temporal analysis, and an interactive 6-tab explorer.

**Architecture:** Four independent analysis scripts write pre-computed NPZ/PNG artefacts into `plots/latent_analysis/data/`. A Plotly Dash app (port 7895) reads those artefacts at startup — no recomputation at browse time. Scripts are per-script resumable (check for output NPZ, skip if present unless `--force`).

**Tech Stack:** Python 3.12, numpy, scipy (stats + cluster + signal), librosa, sklearn (PCA/StandardScaler), matplotlib (posters), plotly + dash, soundfile. Install: `pip install dash pytest`.

---

## File Map

```
plots/latent_analysis/
├── config.py                    NEW — paths, constants, feature groups, STFT params
├── features.py                  NEW — feature loading, encoding (CLR/sin-cos/BPM), variance drop
├── findings.py                  NEW — FINDINGS.md section-overwrite + PROGRESS.md helpers
├── _corr_utils.py               NEW — shared Pearson/Spearman, BH FDR, Fisher-Z helpers
├── _temporal_features.py        NEW — frame-level STFT feature extraction (hop=n_fft=2048)
├── _03_collect.py               NEW — shared latent path sampler (scripts 03 + 04)
├── 01_aggregate_correlation.py  NEW — batch load 192K crops, Pearson+Spearman, BH FDR, posters
├── 02_pca_analysis.py           NEW — feature PCA + latent PCA + cross-PCA + scores
├── 03_latent_xcorr.py           NEW — 64×64 Fisher-Z-averaged xcorr + Ward clustering
├── 04_temporal_correlation.py   NEW — frame-level STFT features (256 frames) + temporal corr
└── app.py                       NEW — Plotly Dash app, 6 tabs
tests/
└── test_latent_analysis.py      NEW — unit tests (runs without external drives)
plots/latent_analysis/data/      NEW — gitignore this directory
plots/latent_analysis/data/posters/  NEW — PNG output dir
```

**External data paths (read-only, never modified):**
- Latents: `/run/media/kim/Lehto/goa-small/{TrackName}/{CropName}.npy` shape `[64, 256]` float32
- Features: `/run/media/kim/Mantu/ai-music/Goa_Separated_crops/{TrackName}/{CropName}.INFO` (JSON)

---

## Task 0: Install dependencies + scaffold

**Files:**
- Create: `plots/latent_analysis/__init__.py`
- Create: `plots/latent_analysis/data/.gitkeep`
- Create: `plots/latent_analysis/data/posters/.gitkeep`

- [ ] **Step 1: Install dash and pytest**

```bash
pip install dash pytest
python3 -c "import dash, pytest; print('dash', dash.__version__); print('pytest ok')"
```
Expected: versions printed, no errors.

- [ ] **Step 2: Create package init and gitkeep files**

```bash
touch plots/latent_analysis/__init__.py
touch plots/latent_analysis/data/.gitkeep
touch plots/latent_analysis/data/posters/.gitkeep
```

- [ ] **Step 3: Add data dir to .gitignore**

Add to `.gitignore` (or create if absent):
```
plots/latent_analysis/data/
```
Exclude `.gitkeep` files from that rule — add immediately after:
```
!plots/latent_analysis/data/.gitkeep
!plots/latent_analysis/data/posters/.gitkeep
```

- [ ] **Step 4: Commit scaffold**

```bash
git add plots/latent_analysis/__init__.py plots/latent_analysis/data/.gitkeep plots/latent_analysis/data/posters/.gitkeep .gitignore
git commit -m "feat: scaffold latent analysis directory"
```

---

## Task 1: config.py — paths and constants

**Files:**
- Create: `plots/latent_analysis/config.py`

- [ ] **Step 1: Write config.py**

```python
# plots/latent_analysis/config.py
"""Single source of truth for all latent analysis settings."""
from pathlib import Path

# --- Paths ---
LATENT_DIR = Path("/run/media/kim/Lehto/goa-small")
INFO_DIR   = Path("/run/media/kim/Mantu/ai-music/Goa_Separated_crops")
DATA_DIR   = Path(__file__).parent / "data"
POSTER_DIR = DATA_DIR / "posters"

# --- Latent geometry ---
LATENT_DIM   = 64
LATENT_FRAMES = 256          # T dimension of each [64, 256] latent file
SAMPLE_RATE  = 44100
HOP_LENGTH   = 2048          # VAE downsampling_ratio = hop size
N_FFT        = 4096          # 50% overlap with Hann window (2× hop)
# With center=True: n_frames = 1 + floor(524288/2048) = 257 → slice [:256]
# center=False gives 255 frames (window truncation); n_fft=hop=2048 gives 256
# but 0% overlap attenuates frame-boundary samples — bad for transient detection.

# --- Temporal analysis subsample ---
N_TEMPORAL_CROPS = 2000
RANDOM_SEED      = 42

# --- Statistics ---
EFFECT_WEAK     = 0.10
EFFECT_MODERATE = 0.20
EFFECT_STRONG   = 0.30
POSTER_CLAMP    = 0.35   # colour scale saturation
FDR_Q           = 0.05

# --- Feature groups (used for UI filtering) ---
FEATURE_GROUPS = {
    "Rhythm":    ["bpm", "bpm_essentia", "bpm_madmom_norm", "syncopation",
                  "on_beat_ratio", "rhythmic_complexity", "rhythmic_evenness"],
    "Timbral":   ["brightness", "roughness", "hardness", "depth",
                  "booming", "reverberation", "sharpness", "warmth"],
    "Spectral":  ["spectral_flatness", "spectral_flux", "spectral_skewness",
                  "spectral_kurtosis", "saturation_ratio", "saturation_count"],
    "Loudness":  ["lufs", "lra", "lufs_drums", "lufs_bass", "lufs_other",
                  "lufs_vocals", "rms_energy_bass", "rms_energy_body",
                  "rms_energy_mid", "rms_energy_air"],
    "Harmonic":  (
        [f"hpcp_{i}" for i in range(12)] +
        [f"tiv_{i}"  for i in range(12)] +
        ["tonic_sin", "tonic_cos", "tonic_minor", "tonic_strength",
         "harmonic_movement_bass", "harmonic_movement_other", "atonality"]
    ),
    "Voice":     ["voice_probability", "instrumental_probability",
                  "female_probability", "male_probability"],
    "Aesthetics":["content_enjoyment", "content_usefulness",
                  "production_complexity", "production_quality"],
}

# Raw INFO keys that need encoding before analysis
RAW_KEYS_TO_ENCODE = {
    "tonic":       "circular_12",    # → tonic_sin, tonic_cos
    "tonic_scale": "binary_minor",   # → tonic_minor (1=minor, 0=major)
    "bpm_madmom":  "bpm_octave_norm",# → bpm_madmom_norm (aligned to bpm_essentia)
}

# Feature names used for temporal (frame-level) analysis
TEMPORAL_FEATURE_NAMES = (
    ["rms_broadband"] +
    ["rms_bass", "rms_body", "rms_mid", "rms_air"] +
    ["spectral_flatness_t", "spectral_flux_t", "spectral_centroid_t",
     "spectral_skewness_t", "spectral_kurtosis_t"] +
    [f"chroma_{i}" for i in range(12)] +
    ["onset_strength"]
)
```

- [ ] **Step 2: Verify config imports cleanly**

```bash
python3 -c "from plots.latent_analysis.config import LATENT_DIR, LATENT_FRAMES, FEATURE_GROUPS; print('ok', LATENT_FRAMES)"
```
Expected: `ok 256`

- [ ] **Step 3: Commit**

```bash
git add plots/latent_analysis/config.py
git commit -m "feat: latent analysis config — paths, constants, feature groups"
```

---

## Task 2: features.py — loading and encoding

**Files:**
- Create: `plots/latent_analysis/features.py`
- Create: `tests/test_latent_analysis.py`

- [ ] **Step 1: Write the failing tests first**

```python
# tests/test_latent_analysis.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from plots.latent_analysis.features import (
    encode_tonic_circular,
    encode_tonic_scale,
    normalise_bpm_madmom,
    apply_clr,
    drop_low_variance,
)


def test_tonic_circular_encoding_values():
    """sin/cos encoding gives correct values."""
    sin_val, cos_val = encode_tonic_circular(0)
    assert abs(sin_val - np.sin(0)) < 1e-6
    assert abs(cos_val - np.cos(0)) < 1e-6

    sin_val, cos_val = encode_tonic_circular(6)
    assert abs(sin_val - np.sin(2 * np.pi * 6 / 12)) < 1e-6


def test_tonic_circular_distance_wraps():
    """Distance from tonic 11 to 0 should equal distance from 0 to 1."""
    def dist(a, b):
        sa, ca = encode_tonic_circular(a)
        sb, cb = encode_tonic_circular(b)
        return np.sqrt((sa - sb) ** 2 + (ca - cb) ** 2)

    assert abs(dist(11, 0) - dist(0, 1)) < 1e-4


def test_tonic_scale_binary():
    assert encode_tonic_scale("minor") == 1
    assert encode_tonic_scale("major") == 0
    assert encode_tonic_scale(None) is None  # missing = skip


def test_bpm_madmom_normalisation():
    """Half-tempo madmom (69) aligned to essentia (138) → ~138."""
    result = normalise_bpm_madmom(bpm_madmom=69.0, bpm_essentia=138.0)
    assert abs(result - 138.0) < 1.0

    # Already matching — no change
    result2 = normalise_bpm_madmom(bpm_madmom=140.0, bpm_essentia=138.0)
    assert abs(result2 - 140.0) < 1.0


def test_clr_sums_to_zero():
    """CLR output should sum to zero."""
    hpcp = np.array([0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    hpcp = hpcp / hpcp.sum()
    clr = apply_clr(hpcp)
    assert abs(clr.sum()) < 1e-5


def test_clr_shape():
    hpcp = np.ones(12) / 12.0
    clr = apply_clr(hpcp)
    assert clr.shape == (12,)


def test_drop_low_variance():
    """Features with std < threshold should be removed."""
    X = np.column_stack([
        np.random.randn(100),       # high variance
        np.zeros(100),              # zero variance → drop
        np.ones(100) * 5 + np.random.randn(100) * 1e-6,  # near-zero → drop
    ])
    names = ["good", "zeros", "flat"]
    X_out, names_out, dropped = drop_low_variance(X, names, threshold=1e-4)
    assert "good" in names_out
    assert "zeros" not in names_out
    assert "flat" not in names_out
    assert set(dropped) == {"zeros", "flat"}
```

- [ ] **Step 2: Run tests — expect FAIL (functions not yet defined)**

```bash
python3 -m pytest tests/test_latent_analysis.py -v 2>&1 | head -30
```
Expected: ImportError or multiple FAILED lines.

- [ ] **Step 3: Write features.py**

```python
# plots/latent_analysis/features.py
"""Feature loading, encoding, and filtering for latent analysis."""
import json
import numpy as np
from pathlib import Path
from typing import Optional

from plots.latent_analysis.config import (
    LATENT_DIR, INFO_DIR, LATENT_DIM, LATENT_FRAMES,
    RAW_KEYS_TO_ENCODE, FEATURE_GROUPS,
)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_tonic_circular(tonic: int):
    """Return (sin, cos) encoding of a pitch-class tonic (0–11)."""
    angle = 2 * np.pi * tonic / 12
    return float(np.sin(angle)), float(np.cos(angle))


def encode_tonic_scale(value: Optional[str]) -> Optional[int]:
    """'minor' → 1, 'major' → 0, None/unknown → None."""
    if value is None:
        return None
    return 1 if str(value).lower() == "minor" else 0


def normalise_bpm_madmom(bpm_madmom: float, bpm_essentia: float) -> float:
    """Multiply/divide bpm_madmom by 2 until ratio to bpm_essentia is minimised."""
    if bpm_essentia <= 0:
        return bpm_madmom
    best, best_ratio = bpm_madmom, abs(bpm_madmom / bpm_essentia - 1)
    for factor in [0.5, 2.0, 0.25, 4.0]:
        candidate = bpm_madmom * factor
        ratio = abs(candidate / bpm_essentia - 1)
        if ratio < best_ratio:
            best, best_ratio = candidate, ratio
    return best


def apply_clr(hpcp: np.ndarray) -> np.ndarray:
    """Centred log-ratio transform for compositional HPCP vector."""
    eps = 1e-9
    x = np.asarray(hpcp, dtype=np.float64) + eps
    log_x = np.log(x)
    return (log_x - log_x.mean()).astype(np.float32)


def drop_low_variance(
    X: np.ndarray,
    names: list,
    threshold: float = 1e-4,
):
    """Remove columns where std < threshold. Returns (X_out, names_out, dropped)."""
    stds = X.std(axis=0)
    keep = stds >= threshold
    dropped = [n for n, k in zip(names, keep) if not k]
    return X[:, keep], [n for n, k in zip(names, keep) if k], dropped


# ---------------------------------------------------------------------------
# INFO file loading + encoding
# ---------------------------------------------------------------------------

def load_info(path: Path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    # Some older INFO files nest under 'original_features'
    return raw.get("original_features", raw)


def encode_info_features(info: dict) -> dict:
    """
    Apply all encoding transformations to a raw INFO dict.
    Returns a new dict with encoded keys; drops raw keys that were replaced.
    """
    out = dict(info)

    # tonic → tonic_sin + tonic_cos  (drop raw tonic)
    if "tonic" in out and out["tonic"] is not None:
        try:
            s, c = encode_tonic_circular(int(out["tonic"]))
            out["tonic_sin"] = s
            out["tonic_cos"] = c
        except (ValueError, TypeError):
            pass
    out.pop("tonic", None)

    # tonic_scale → tonic_minor
    if "tonic_scale" in out:
        encoded = encode_tonic_scale(out["tonic_scale"])
        if encoded is not None:
            out["tonic_minor"] = float(encoded)
    out.pop("tonic_scale", None)

    # bpm_madmom → bpm_madmom_norm (using bpm_essentia as reference)
    if "bpm_madmom" in out and "bpm_essentia" in out:
        try:
            out["bpm_madmom_norm"] = normalise_bpm_madmom(
                float(out["bpm_madmom"]), float(out["bpm_essentia"])
            )
        except (ValueError, TypeError):
            pass
    out.pop("bpm_madmom", None)   # remove raw; keep normalised

    # hpcp_0..11 → hpcp_clr_0..11 (for PCA only — raw kept for Pearson)
    hpcp_keys = [f"hpcp_{i}" for i in range(12)]
    hpcp_vals = [out.get(k) for k in hpcp_keys]
    if all(v is not None for v in hpcp_vals):
        clr = apply_clr(np.array(hpcp_vals, dtype=np.float32))
        for i, v in enumerate(clr):
            out[f"hpcp_clr_{i}"] = float(v)

    return out


# ---------------------------------------------------------------------------
# Latent + feature pair iterator
# ---------------------------------------------------------------------------

def iter_paired_crops(feature_names: list):
    """
    Yields (latent_mean[64], feature_dict) for every crop that has a valid latent
    file AND a corresponding INFO file. Skips corrupted files silently.

    feature_names: the encoded feature names we care about (after encoding).
    """
    if not LATENT_DIR.exists():
        raise RuntimeError(f"Latent dir not mounted: {LATENT_DIR}")
    if not INFO_DIR.exists():
        raise RuntimeError(f"Feature dir not mounted: {INFO_DIR}")

    stem_suffixes = {"_bass", "_drums", "_other", "_vocals"}

    for track_dir in sorted(LATENT_DIR.iterdir()):
        if not track_dir.is_dir():
            continue
        t_name = track_dir.name
        info_track_dir = INFO_DIR / t_name
        if not info_track_dir.exists():
            continue

        for npy_path in track_dir.glob("*.npy"):
            # Skip stem latents
            if any(npy_path.stem.endswith(s) for s in stem_suffixes):
                continue

            info_path = info_track_dir / (npy_path.stem + ".INFO")
            if not info_path.exists():
                continue

            try:
                latent = np.load(str(npy_path))
                assert latent.shape == (LATENT_DIM, LATENT_FRAMES), \
                    f"Unexpected shape {latent.shape}"
                latent_mean = latent.mean(axis=1)   # [64]

                info = encode_info_features(load_info(info_path))
                yield latent_mean, info

            except Exception:
                continue
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python3 -m pytest tests/test_latent_analysis.py -v
```
Expected: 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add plots/latent_analysis/features.py tests/test_latent_analysis.py
git commit -m "feat: latent analysis feature encoding — CLR, tonic sin/cos, BPM normalisation"
```

---

## Task 3: findings.py — FINDINGS.md and PROGRESS.md helpers

**Files:**
- Create: `plots/latent_analysis/findings.py`

- [ ] **Step 1: Add tests for section overwrite**

Append to `tests/test_latent_analysis.py`:

```python
from plots.latent_analysis.findings import overwrite_findings_section


def test_findings_section_overwrite_replaces_content():
    content = (
        "# Header\n\n"
        "<!-- script-01-start -->\n"
        "## Old section\nOld content\n"
        "<!-- script-01-end -->\n\n"
        "## Footer stays\n"
    )
    result = overwrite_findings_section(content, "01", "## New section\nNew content\n")
    assert "Old content" not in result
    assert "New content" in result
    assert "## Footer stays" in result
    assert "<!-- script-01-start -->" in result
    assert "<!-- script-01-end -->" in result


def test_findings_section_overwrite_idempotent():
    """Running overwrite twice gives same result."""
    content = (
        "<!-- script-02-start -->\nOriginal\n<!-- script-02-end -->\n"
    )
    once  = overwrite_findings_section(content, "02", "Updated\n")
    twice = overwrite_findings_section(once,    "02", "Updated\n")
    assert once == twice


def test_findings_section_overwrite_missing_markers():
    """If markers are absent, raises ValueError rather than silently corrupting."""
    content = "# No markers here\n"
    with pytest.raises(ValueError, match="markers not found"):
        overwrite_findings_section(content, "01", "New content\n")
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python3 -m pytest tests/test_latent_analysis.py::test_findings_section_overwrite_replaces_content -v
```
Expected: ImportError.

- [ ] **Step 3: Write findings.py**

```python
# plots/latent_analysis/findings.py
"""FINDINGS.md section-overwrite and PROGRESS.md helpers."""
import re
from datetime import datetime
from pathlib import Path

from plots.latent_analysis.config import DATA_DIR

FINDINGS_PATH = DATA_DIR.parent / "FINDINGS.md"
PROGRESS_PATH = DATA_DIR.parent / "PROGRESS.md"

_FINDINGS_TEMPLATE = """\
# Latent Feature Analysis Findings

## Run info
- Date: {date}
- N crops: {n_crops}
- N features: {n_features} (after encoding + variance drop)
- Features dropped (low variance): {dropped}

<!-- script-01-start -->
## Script 01 — Aggregate Correlations
*(not yet run)*
<!-- script-01-end -->

<!-- script-02-start -->
## Script 02 — PCA
*(not yet run)*
<!-- script-02-end -->

<!-- script-03-start -->
## Script 03 — Latent Cross-Correlation
*(not yet run)*
<!-- script-03-end -->

<!-- script-04-start -->
## Script 04 — Temporal
*(not yet run)*
<!-- script-04-end -->

## Open questions / next steps
*(add manual notes here — this section is never auto-overwritten)*
"""


def overwrite_findings_section(content: str, script_id: str, new_body: str) -> str:
    """
    Replace the content between <!-- script-{id}-start --> and <!-- script-{id}-end -->
    markers with new_body. Raises ValueError if markers are absent.
    """
    start_marker = f"<!-- script-{script_id}-start -->"
    end_marker   = f"<!-- script-{script_id}-end -->"
    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker),
        re.DOTALL,
    )
    if not re.search(pattern, content):
        raise ValueError(
            f"FINDINGS.md section markers not found for script-{script_id}. "
            f"Expected '{start_marker}' ... '{end_marker}'."
        )
    replacement = f"{start_marker}\n{new_body.rstrip()}\n{end_marker}"
    return re.sub(pattern, replacement, content)


def init_findings(n_crops: int, n_features: int, dropped: list):
    """Create FINDINGS.md from template if it doesn't exist."""
    if FINDINGS_PATH.exists():
        return
    FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = _FINDINGS_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d"),
        n_crops=n_crops,
        n_features=n_features,
        dropped=", ".join(dropped) if dropped else "none",
    )
    FINDINGS_PATH.write_text(text)


def update_findings_section(script_id: str, body: str):
    """Overwrite a named section in FINDINGS.md. Idempotent."""
    if not FINDINGS_PATH.exists():
        raise RuntimeError("FINDINGS.md not initialised — call init_findings() first.")
    content = FINDINGS_PATH.read_text()
    content = overwrite_findings_section(content, script_id, body)
    FINDINGS_PATH.write_text(content)


def update_progress(script_id: str, message: str):
    """Append a timestamped line to PROGRESS.md."""
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] Script {script_id}: {message}\n"
    with open(PROGRESS_PATH, "a") as f:
        f.write(line)
    print(line.strip())
```

- [ ] **Step 4: Run all tests — expect PASS**

```bash
python3 -m pytest tests/test_latent_analysis.py -v
```
Expected: 10 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add plots/latent_analysis/findings.py tests/test_latent_analysis.py
git commit -m "feat: FINDINGS.md section-overwrite and PROGRESS.md helpers"
```

---

## Task 4: 01_aggregate_correlation.py

**Files:**
- Create: `plots/latent_analysis/01_aggregate_correlation.py`

Computes full Pearson + Spearman correlation matrix (64 dims × N features), BH FDR correction, 8×8 PNG posters, and a 5K scatter sample NPZ.

- [ ] **Step 1: Add a test for the correlation + FDR plumbing**

Append to `tests/test_latent_analysis.py`:

```python
from plots.latent_analysis._corr_utils import (
    compute_pearson_spearman,
    apply_bh_fdr,
)


def test_pearson_spearman_shapes():
    """Returns r and p arrays of length N_dims."""
    X = np.random.randn(200, 64)   # 200 crops, 64 dims
    y = np.random.randn(200)
    r_p, p_p, r_s, p_s = compute_pearson_spearman(X, y)
    assert r_p.shape == (64,)
    assert r_s.shape == (64,)


def test_bh_fdr_output_shape():
    pvals = np.random.uniform(0, 1, (64, 20))  # 64 dims × 20 features
    adj = apply_bh_fdr(pvals)
    assert adj.shape == pvals.shape
    # Adjusted p-values must be >= originals (BH never makes them smaller)
    assert np.all(adj >= pvals - 1e-10)
```

- [ ] **Step 2: Create `_corr_utils.py` (shared between scripts 01 and 04)**

```python
# plots/latent_analysis/_corr_utils.py
"""Shared correlation and FDR utilities."""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.stats import false_discovery_control


def compute_pearson_spearman(X: np.ndarray, y: np.ndarray):
    """
    For each column of X, compute Pearson r and Spearman rho against y.
    Returns (r_pearson, p_pearson, r_spearman, p_spearman), each shape [n_cols].
    """
    n_cols = X.shape[1]
    r_p = np.zeros(n_cols)
    p_p = np.ones(n_cols)
    r_s = np.zeros(n_cols)
    p_s = np.ones(n_cols)
    for c in range(n_cols):
        try:
            r_p[c], p_p[c] = pearsonr(X[:, c], y)
            r_s[c], p_s[c] = spearmanr(X[:, c], y)
        except Exception:
            pass
    r_p = np.nan_to_num(r_p)
    r_s = np.nan_to_num(r_s)
    return r_p, p_p, r_s, p_s


def apply_bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction to a 2D array of p-values.
    pvals shape: [n_dims, n_features]. Returns adjusted p-values, same shape.
    """
    flat = pvals.ravel()
    adj_flat = false_discovery_control(np.clip(flat, 0, 1), method='bh')
    return adj_flat.reshape(pvals.shape)
```

- [ ] **Step 3: Run new tests — expect PASS**

```bash
python3 -m pytest tests/test_latent_analysis.py::test_pearson_spearman_shapes tests/test_latent_analysis.py::test_bh_fdr_output_shape -v
```
Expected: 2 PASSED.

- [ ] **Step 4: Write 01_aggregate_correlation.py**

```python
# plots/latent_analysis/01_aggregate_correlation.py
"""
Script 01 — Aggregate correlation analysis.
Loads all 192K crops (streaming), computes Pearson + Spearman for each
latent dim × feature pair, applies BH FDR, saves NPZ + 8×8 posters.

Usage:
    python 01_aggregate_correlation.py [--force]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plots.latent_analysis.config import (
    DATA_DIR, POSTER_DIR, FEATURE_GROUPS, LATENT_DIM,
    POSTER_CLAMP, EFFECT_WEAK, EFFECT_STRONG, RANDOM_SEED,
)
from plots.latent_analysis.features import iter_paired_crops, drop_low_variance
from plots.latent_analysis.findings import init_findings, update_findings_section, update_progress
from plots.latent_analysis._corr_utils import compute_pearson_spearman, apply_bh_fdr

OUT_NPZ     = DATA_DIR / "01_correlations.npz"
SCATTER_NPZ = DATA_DIR / "scatter_sample.npz"
ALL_FEATURE_NAMES = [f for group in FEATURE_GROUPS.values() for f in group]


def run(force: bool = False):
    if OUT_NPZ.exists() and not force:
        print(f"01_correlations.npz already exists. Use --force to recompute.")
        return

    print("Script 01: loading crops...")
    # Accumulate per-feature lists
    latent_accumulator = defaultdict(list)  # feat_name → list of latent_mean[64]
    value_accumulator  = defaultdict(list)  # feat_name → list of scalar value
    all_latent_means   = []

    all_info_rows = []   # parallel to all_latent_means, for scatter sample
    n_total = 0
    for latent_mean, info in iter_paired_crops(ALL_FEATURE_NAMES):
        n_total += 1
        if n_total % 10000 == 0:
            print(f"  {n_total} crops loaded...")
        all_latent_means.append(latent_mean)
        all_info_rows.append(info)   # keep encoded info dict for scatter
        for feat in ALL_FEATURE_NAMES:
            val = info.get(feat)
            if val is not None and isinstance(val, (int, float)) and np.isfinite(val):
                latent_accumulator[feat].append(latent_mean)
                value_accumulator[feat].append(float(val))

    print(f"Total crops loaded: {n_total}")

    # Drop near-zero-variance features
    # Build a feature matrix from crops that have ALL features (for variance check)
    # Use a simplified approach: check std of each feature's values
    dropped = []
    valid_features = []
    for feat in ALL_FEATURE_NAMES:
        vals = np.array(value_accumulator[feat])
        if len(vals) < 10 or vals.std() < 1e-4:
            dropped.append(feat)
        else:
            valid_features.append(feat)

    if dropped:
        print(f"Dropped low-variance features: {dropped}")

    feature_names = valid_features
    n_features    = len(feature_names)
    print(f"Analysing {n_features} features × {LATENT_DIM} dims...")

    # Correlation matrices: [64, n_features]
    r_pearson  = np.zeros((LATENT_DIM, n_features))
    p_pearson  = np.ones((LATENT_DIM, n_features))
    r_spearman = np.zeros((LATENT_DIM, n_features))
    p_spearman = np.ones((LATENT_DIM, n_features))
    n_per_feat = np.zeros(n_features, dtype=int)

    for fi, feat in enumerate(feature_names):
        X = np.array(latent_accumulator[feat])   # [N_feat, 64]
        y = np.array(value_accumulator[feat])     # [N_feat]
        n_per_feat[fi] = len(y)
        rp, pp, rs, ps = compute_pearson_spearman(X, y)
        r_pearson[:,  fi] = rp
        p_pearson[:,  fi] = pp
        r_spearman[:, fi] = rs
        p_spearman[:, fi] = ps
        if fi % 10 == 0:
            print(f"  feature {fi+1}/{n_features}: {feat} (N={len(y)})")

    # BH FDR correction
    p_pearson_adj  = apply_bh_fdr(p_pearson)
    p_spearman_adj = apply_bh_fdr(p_spearman)

    # Save main NPZ
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        r_pearson=r_pearson,
        p_pearson_adj=p_pearson_adj,
        r_spearman=r_spearman,
        p_spearman_adj=p_spearman_adj,
        feature_names=np.array(feature_names),
        n_per_feature=n_per_feat,
    )
    print(f"Saved {OUT_NPZ}")

    # Scatter sample: 5000 crops that have ALL valid features
    # Build per-crop complete feature rows (192K × ~60 floats ≈ ~90MB, fine in RAM)
    # all_latent_means and all_info_rows were tracked in the accumulation loop (see below)
    complete_indices = [
        i for i, row in enumerate(all_info_rows)
        if all(isinstance(row.get(f), (int, float)) and np.isfinite(float(row.get(f, float('nan'))))
               for f in feature_names)
    ]
    rng = np.random.default_rng(RANDOM_SEED)
    idx5k = rng.choice(complete_indices, size=min(5000, len(complete_indices)), replace=False)
    scatter_latents = np.array([all_latent_means[i] for i in idx5k])
    scatter_feats   = np.array([[float(all_info_rows[i][f]) for f in feature_names]
                                 for i in idx5k], dtype=np.float32)
    np.savez_compressed(SCATTER_NPZ,
                        latent_means=scatter_latents,
                        feature_values=scatter_feats,
                        feature_names=np.array(feature_names))
    print(f"Saved {SCATTER_NPZ}")

    # Generate 8×8 posters
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    _generate_posters(r_pearson, feature_names, n_per_feat)

    # Update FINDINGS.md
    top_pairs = _top_corr_pairs(r_pearson, feature_names, n=5)
    init_findings(n_total, n_features, dropped)
    body = _findings_body(n_total, n_features, dropped, feature_names, top_pairs)
    update_findings_section("01", body)
    update_progress("01", f"Done. {n_total} crops, {n_features} features.")


def _generate_posters(r_matrix: np.ndarray, feature_names: list, n_per_feat: np.ndarray):
    """Save one 8×8 heatmap PNG per feature into POSTER_DIR."""
    for fi, feat in enumerate(feature_names):
        r_col = r_matrix[:, fi]           # [64]
        grid  = r_col.reshape(8, 8)       # 8×8 spatial layout

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-POSTER_CLAMP, vmax=POSTER_CLAMP,
                       aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

        # Annotate each cell with r value
        for row in range(8):
            for col in range(8):
                dim_idx = row * 8 + col
                val = grid[row, col]
                ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(val) < 0.25 else "white")

        ax.set_title(f"{feat}\n(N={n_per_feat[fi]:,})", fontsize=10)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels([str(i) for i in range(8)], fontsize=7)
        ax.set_yticklabels([str(i*8) for i in range(8)], fontsize=7)
        ax.set_xlabel("Dim mod 8")
        ax.set_ylabel("Dim // 8 × 8")

        fig.tight_layout()
        fig.savefig(POSTER_DIR / f"{feat}_poster.png", dpi=100)
        plt.close(fig)

    print(f"Saved {len(feature_names)} posters to {POSTER_DIR}")


def _top_corr_pairs(r_matrix, feature_names, n=5):
    """Return top-N (|r|, dim, feat) tuples."""
    flat_abs = np.abs(r_matrix).ravel()
    top_idx  = np.argsort(flat_abs)[::-1][:n]
    pairs = []
    for idx in top_idx:
        dim = idx // len(feature_names)
        fi  = idx % len(feature_names)
        pairs.append((r_matrix[dim, fi], dim, feature_names[fi]))
    return pairs


def _findings_body(n_crops, n_features, dropped, feature_names, top_pairs):
    lines = [
        f"## Script 01 — Aggregate Correlations",
        f"- Crops analysed: {n_crops:,}",
        f"- Features: {n_features} (dropped: {', '.join(dropped) or 'none'})",
        f"",
        f"### Top correlating (dim, feature) pairs by |Pearson r|",
    ]
    for r, dim, feat in top_pairs:
        lines.append(f"- Dim {dim:2d} × `{feat}`: r = {r:+.3f}")
    lines += [
        "",
        "> ⚠️ BPM correlations may be deflated due to low within-genre variance (~135–145 BPM).",
        "> ⚠️ `atonality` likely reflects noise/percussion energy in this corpus, not harmonic atonality.",
        "> ⚠️ HPCP correlations are relative (compositional constraint); interpret with CLR in PCA.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(force=args.force)
```

- [ ] **Step 5: Verify the script can be imported without the drives mounted**

```bash
python3 -c "import plots.latent_analysis.01_aggregate_correlation" 2>&1 | head -5
```
Wait — Python module names can't start with digits. Import via path instead:
```bash
python3 -c "
import sys, importlib.util
spec = importlib.util.spec_from_file_location('s01', 'plots/latent_analysis/01_aggregate_correlation.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('import ok')
"
```
Expected: `import ok` (no errors on import; drive check happens only inside `run()`).

- [ ] **Step 6: Commit**

```bash
git add plots/latent_analysis/_corr_utils.py plots/latent_analysis/01_aggregate_correlation.py tests/test_latent_analysis.py
git commit -m "feat: script 01 — aggregate Pearson+Spearman correlation + BH FDR + 8x8 posters"
```

---

## Task 5: 02_pca_analysis.py

**Files:**
- Create: `plots/latent_analysis/02_pca_analysis.py`

PCA on feature vectors, PCA on latent means, cross-PCA correlation matrix, per-crop projected scores saved to NPZ.

- [ ] **Step 1: Write 02_pca_analysis.py**

```python
# plots/latent_analysis/02_pca_analysis.py
"""
Script 02 — PCA analysis.
Loads all feature vectors + latent means, fits PCA on each,
computes cross-PCA correlation matrix and saves per-crop scores.

Usage:
    python 02_pca_analysis.py [--force] [--n-components 20]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plots.latent_analysis.config import DATA_DIR, FEATURE_GROUPS, LATENT_DIM
from plots.latent_analysis.features import (
    iter_paired_crops, drop_low_variance, apply_clr
)
from plots.latent_analysis.findings import update_findings_section, update_progress

OUT_NPZ = DATA_DIR / "02_pca.npz"
ALL_FEATURE_NAMES = [f for group in FEATURE_GROUPS.values() for f in group]


def run(force: bool = False, n_components: int = 20):
    if OUT_NPZ.exists() and not force:
        print("02_pca.npz already exists. Use --force to recompute.")
        return

    print("Script 02: loading crop feature vectors...")
    rows_feat    = []
    rows_latent  = []
    crop_ids     = []

    for i, (latent_mean, info) in enumerate(iter_paired_crops(ALL_FEATURE_NAMES)):
        # Build feature vector — use CLR-transformed HPCP for PCA
        row = []
        valid = True
        for feat in ALL_FEATURE_NAMES:
            if feat.startswith("hpcp_") and not feat.startswith("hpcp_clr_"):
                # Use CLR version for PCA
                clr_key = feat.replace("hpcp_", "hpcp_clr_")
                val = info.get(clr_key)
            else:
                val = info.get(feat)
            if val is None or not isinstance(val, (int, float)) or not np.isfinite(val):
                valid = False
                break
            row.append(float(val))
        if not valid:
            continue
        rows_feat.append(row)
        rows_latent.append(latent_mean)
        crop_ids.append(i)
        if len(rows_feat) % 10000 == 0:
            print(f"  {len(rows_feat)} crops with complete features...")

    X_feat   = np.array(rows_feat,   dtype=np.float32)
    X_latent = np.array(rows_latent, dtype=np.float32)
    print(f"Complete crops: {len(X_feat)} | Feature dims: {X_feat.shape[1]}")

    # Drop near-zero-variance features before PCA
    X_feat, feat_names_used, dropped = drop_low_variance(
        X_feat, ALL_FEATURE_NAMES, threshold=1e-4
    )
    if dropped:
        print(f"Dropped: {dropped}")

    # Feature PCA
    scaler_feat  = StandardScaler()
    X_feat_std   = scaler_feat.fit_transform(X_feat)
    n_comp_feat  = min(n_components, X_feat_std.shape[1])
    pca_feat     = PCA(n_components=n_comp_feat, random_state=42)
    feat_scores  = pca_feat.fit_transform(X_feat_std)   # [N, n_comp]

    # Latent PCA (no standardisation — already normalised)
    n_comp_lat   = min(n_components, LATENT_DIM)
    pca_latent   = PCA(n_components=n_comp_lat, random_state=42)
    latent_scores = pca_latent.fit_transform(X_latent)  # [N, n_comp]

    # Cross-PCA correlation matrix [n_comp_feat, n_comp_lat]
    from scipy.stats import pearsonr
    cross_corr = np.zeros((n_comp_feat, n_comp_lat))
    for i in range(n_comp_feat):
        for j in range(n_comp_lat):
            r, _ = pearsonr(feat_scores[:, i], latent_scores[:, j])
            cross_corr[i, j] = r

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        feat_explained_variance_ratio=pca_feat.explained_variance_ratio_,
        feat_components=pca_feat.components_,
        feat_scores=feat_scores,
        latent_explained_variance_ratio=pca_latent.explained_variance_ratio_,
        latent_components=pca_latent.components_,
        latent_scores=latent_scores,
        cross_corr=cross_corr,
        feat_names_used=np.array(feat_names_used),
        crop_ids=np.array(crop_ids),
        scaler_mean=scaler_feat.mean_,
        scaler_std=scaler_feat.scale_,
    )
    print(f"Saved {OUT_NPZ}")

    # FINDINGS
    ev_feat   = pca_feat.explained_variance_ratio_
    ev_latent = pca_latent.explained_variance_ratio_
    body = _findings_body(
        len(X_feat), feat_names_used, ev_feat, ev_latent,
        pca_feat.components_, cross_corr
    )
    update_findings_section("02", body)
    update_progress("02", f"Done. {len(X_feat)} crops, feat PC1={ev_feat[0]:.1%}, lat PC1={ev_latent[0]:.1%}")


def _findings_body(n, feat_names, ev_feat, ev_latent, feat_components, cross_corr):
    lines = [
        "## Script 02 — PCA",
        f"- Crops with complete features: {n:,}",
        "",
        "### Feature PCA explained variance",
    ]
    cumev = 0
    for i, ev in enumerate(ev_feat[:10]):
        cumev += ev
        top_loadings = np.argsort(np.abs(feat_components[i]))[::-1][:3]
        top_names    = [feat_names[j] for j in top_loadings]
        lines.append(f"- PC{i+1}: {ev:.1%} (cumulative {cumev:.1%}) — top: {', '.join(top_names)}")
    lines += [
        "",
        "### Latent PCA explained variance (top 5)",
    ]
    for i, ev in enumerate(ev_latent[:5]):
        lines.append(f"- Latent PC{i+1}: {ev:.1%}")
    lines += [
        "",
        "### Strongest cross-PCA alignments",
    ]
    flat = np.abs(cross_corr).ravel()
    top  = np.argsort(flat)[::-1][:5]
    for idx in top:
        fi = idx // cross_corr.shape[1]
        li = idx % cross_corr.shape[1]
        lines.append(f"- Feature PC{fi+1} ↔ Latent PC{li+1}: r = {cross_corr[fi,li]:+.3f}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-components", type=int, default=20)
    args = parser.parse_args()
    run(force=args.force, n_components=args.n_components)
```

- [ ] **Step 2: Verify import**

```bash
python3 -c "
import sys, importlib.util
spec = importlib.util.spec_from_file_location('s02', 'plots/latent_analysis/02_pca_analysis.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('import ok')
"
```

- [ ] **Step 3: Commit**

```bash
git add plots/latent_analysis/02_pca_analysis.py
git commit -m "feat: script 02 — feature+latent PCA, cross-PCA correlation, per-crop scores"
```

---

## Task 6: 03_latent_xcorr.py

**Files:**
- Create: `plots/latent_analysis/03_latent_xcorr.py`

Fisher-Z-averaged 64×64 temporal cross-correlation matrix, Ward clustering.

- [ ] **Step 1: Add Fisher Z test**

Append to `tests/test_latent_analysis.py`:

```python
from plots.latent_analysis._corr_utils import fisher_z_average


def test_fisher_z_average_preserves_sign():
    matrices = np.array([
        [[0.8, -0.3], [-0.3, 0.8]],
        [[0.7, -0.2], [-0.2, 0.7]],
    ])
    result = fisher_z_average(matrices)
    assert result[0, 0] > 0
    assert result[0, 1] < 0


def test_fisher_z_average_symmetric():
    matrices = np.random.uniform(-0.5, 0.5, (10, 8, 8))
    result = fisher_z_average(matrices)
    np.testing.assert_allclose(result, result.T, atol=1e-6)


def test_fisher_z_handles_diagonal_ones():
    """Diagonal r=1 should not produce inf after ε clamping."""
    matrices = np.eye(4)[np.newaxis].repeat(5, axis=0)
    result = fisher_z_average(matrices)
    assert np.all(np.isfinite(result))
```

- [ ] **Step 2: Add `fisher_z_average` to `_corr_utils.py`**

Append to `plots/latent_analysis/_corr_utils.py`:

```python
def fisher_z_average(matrices: np.ndarray) -> np.ndarray:
    """
    Average a stack of Pearson correlation matrices using Fisher Z-transform.
    matrices: shape [N, D, D].
    Returns averaged matrix: shape [D, D].
    """
    eps = 1e-7
    clipped = np.clip(matrices, -1 + eps, 1 - eps)
    z_stack = np.arctanh(clipped)
    return np.tanh(z_stack.mean(axis=0))
```

- [ ] **Step 3: Run new tests**

```bash
python3 -m pytest tests/test_latent_analysis.py::test_fisher_z_average_preserves_sign tests/test_latent_analysis.py::test_fisher_z_average_symmetric tests/test_latent_analysis.py::test_fisher_z_handles_diagonal_ones -v
```
Expected: 3 PASSED.

- [ ] **Step 4: Write 03_latent_xcorr.py**

```python
# plots/latent_analysis/03_latent_xcorr.py
"""
Script 03 — Latent temporal cross-correlation.
Subsamples 2000 crops, computes 64×64 Pearson xcorr per crop (Fisher-Z averaged),
then Ward-links dims into clusters.

Usage:
    python 03_latent_xcorr.py [--force] [--n-crops 2000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plots.latent_analysis.config import (
    DATA_DIR, LATENT_DIR, LATENT_DIM, LATENT_FRAMES,
    N_TEMPORAL_CROPS, RANDOM_SEED,
)
from plots.latent_analysis._corr_utils import fisher_z_average
from plots.latent_analysis.findings import update_findings_section, update_progress

OUT_NPZ = DATA_DIR / "03_xcorr.npz"
STEM_SUFFIXES = {"_bass", "_drums", "_other", "_vocals"}


def _collect_latent_paths(n_crops: int, seed: int):
    """Collect up to n_crops full-mix latent paths, randomly sampled."""
    rng  = np.random.default_rng(seed)
    all_paths = []
    for track_dir in LATENT_DIR.iterdir():
        if not track_dir.is_dir():
            continue
        for p in track_dir.glob("*.npy"):
            if not any(p.stem.endswith(s) for s in STEM_SUFFIXES):
                all_paths.append(p)
    rng.shuffle(all_paths)
    return all_paths[:n_crops]


def run(force: bool = False, n_crops: int = N_TEMPORAL_CROPS):
    if OUT_NPZ.exists() and not force:
        print("03_xcorr.npz already exists. Use --force to recompute.")
        return

    if not LATENT_DIR.exists():
        raise RuntimeError(f"Latent dir not mounted: {LATENT_DIR}")

    print(f"Script 03: collecting {n_crops} latent paths...")
    paths = _collect_latent_paths(n_crops, RANDOM_SEED)
    print(f"  found {len(paths)} crops")

    xcorr_stack = []
    skipped = 0
    for i, p in enumerate(paths):
        try:
            lat = np.load(str(p)).astype(np.float32)  # [64, 256]
            assert lat.shape == (LATENT_DIM, LATENT_FRAMES)
            # Demean each dim
            lat = lat - lat.mean(axis=1, keepdims=True)
            # 64×64 Pearson correlation across 256 time steps
            corr = np.corrcoef(lat)          # [64, 64]
            xcorr_stack.append(corr)
        except Exception:
            skipped += 1
            continue
        if i % 500 == 0:
            print(f"  {i+1}/{len(paths)}...")

    print(f"Computed {len(xcorr_stack)} xcorr matrices ({skipped} skipped).")

    mean_xcorr = fisher_z_average(np.array(xcorr_stack))   # [64, 64]

    # Ward clustering
    # Convert correlation to distance: d = 1 - r (bounded [0,2])
    dist_matrix = 1.0 - mean_xcorr
    np.fill_diagonal(dist_matrix, 0)
    dist_condensed = dist_matrix[np.triu_indices(LATENT_DIM, k=1)]
    Z = linkage(dist_condensed, method="ward")

    # Auto-determine number of clusters (cut at 70% of max merge distance)
    max_dist = Z[-1, 2]
    labels   = fcluster(Z, 0.7 * max_dist, criterion="distance")
    n_clusters = labels.max()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        xcorr_matrix=mean_xcorr,
        linkage_matrix=Z,
        cluster_labels=labels,
        n_crops_used=len(xcorr_stack),
    )
    print(f"Saved {OUT_NPZ} — {n_clusters} clusters found.")

    body = _findings_body(len(xcorr_stack), n_clusters, labels, mean_xcorr)
    update_findings_section("03", body)
    update_progress("03", f"Done. {len(xcorr_stack)} crops, {n_clusters} clusters.")


def _findings_body(n_crops, n_clusters, labels, xcorr):
    lines = [
        "## Script 03 — Latent Cross-Correlation",
        f"- Crops used: {n_crops} (Fisher-Z averaged)",
        f"- Clusters found (Ward, 70% cut): **{n_clusters}**",
        "",
        "### Cluster membership",
    ]
    for c in range(1, n_clusters + 1):
        dims = np.where(labels == c)[0]
        lines.append(f"- Cluster {c}: dims {', '.join(map(str, dims))}")
    # Strongest inter-dim correlations
    lines += ["", "### Strongest inter-dim temporal correlations"]
    flat = xcorr.copy()
    np.fill_diagonal(flat, 0)
    top = np.argsort(np.abs(flat).ravel())[::-1][:6]
    seen = set()
    for idx in top:
        d1, d2 = divmod(idx, 64)
        if (min(d1,d2), max(d1,d2)) in seen:
            continue
        seen.add((min(d1,d2), max(d1,d2)))
        lines.append(f"- Dim {d1} ↔ Dim {d2}: r = {xcorr[d1,d2]:+.3f}")
    lines += [
        "",
        "> Note: within-track crops treated as independent; effective N ≈ number of tracks sampled (~1600).",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-crops", type=int, default=N_TEMPORAL_CROPS)
    args = parser.parse_args()
    run(force=args.force, n_crops=args.n_crops)
```

- [ ] **Step 5: Run all tests**

```bash
python3 -m pytest tests/test_latent_analysis.py -v
```
Expected: all PASSED.

- [ ] **Step 6: Commit**

```bash
git add plots/latent_analysis/03_latent_xcorr.py plots/latent_analysis/_corr_utils.py tests/test_latent_analysis.py
git commit -m "feat: script 03 — Fisher-Z averaged 64x64 latent xcorr + Ward clustering"
```

---

## Task 7: 04_temporal_correlation.py

**Files:**
- Create: `plots/latent_analysis/04_temporal_correlation.py`

Frame-level features (256 frames, hop=2048, n_fft=2048, center=False) correlated with latent time series.

- [ ] **Step 1: Add STFT frame count test**

Append to `tests/test_latent_analysis.py`:

```python
from plots.latent_analysis._temporal_features import compute_frame_features


def test_stft_produces_exactly_256_frames():
    """n_fft=4096, hop=2048, center=True → 257 frames sliced to 256."""
    audio = np.random.randn(256 * 2048).astype(np.float32)
    feats = compute_frame_features(audio, sr=44100)
    assert feats.shape[1] == 256, f"Expected 256 frames, got {feats.shape[1]}"


def test_frame_features_no_nan():
    audio = np.random.randn(256 * 2048).astype(np.float32)
    feats = compute_frame_features(audio, sr=44100)
    assert np.all(np.isfinite(feats)), "NaN/Inf in temporal features"
```

- [ ] **Step 2: Create `_temporal_features.py`**

```python
# plots/latent_analysis/_temporal_features.py
"""Frame-level feature extraction aligned to latent hop size."""
import numpy as np
import librosa
from scipy.signal import butter, sosfilt

from plots.latent_analysis.config import (
    SAMPLE_RATE, HOP_LENGTH, N_FFT, LATENT_FRAMES,
    TEMPORAL_FEATURE_NAMES,
)

# Band-pass filter boundaries (Hz)
_BANDS = {
    "bass": (20,   250),
    "body": (250,  2000),
    "mid":  (2000, 8000),
    "air":  (8000, 20000),
}


def _bandpass_rms(audio: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    """Frame-level RMS within a frequency band. Returns [LATENT_FRAMES]."""
    nyq  = sr / 2
    sos  = butter(4, [low/nyq, min(high/nyq, 0.999)], btype="band", output="sos")
    filtered = sosfilt(sos, audio.astype(np.float64)).astype(np.float32)
    # Frame it
    frames = librosa.util.frame(filtered, frame_length=N_FFT, hop_length=HOP_LENGTH)
    rms = np.sqrt((frames ** 2).mean(axis=0))
    return rms[:LATENT_FRAMES]


def compute_frame_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Compute frame-level features for a single crop.
    audio: 1D float32, length exactly LATENT_FRAMES * HOP_LENGTH.
    Returns: [N_temporal_features, LATENT_FRAMES] float32.
    n_fft == hop_length == 2048, center=False → 256 frames guaranteed.
    """
    audio = audio.astype(np.float32)
    expected_len = LATENT_FRAMES * HOP_LENGTH
    if len(audio) != expected_len:
        raise ValueError(f"Expected {expected_len} samples, got {len(audio)}")

    # STFT (n_fft=4096, hop=2048, center=True → 257 frames; slice to 256)
    # 50% overlap with Hann window preserves transients at frame boundaries.
    # center=True pads by n_fft//2 each side → 1 + floor(524288/2048) = 257 frames.
    stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                center=True))[:, :LATENT_FRAMES]  # [n_fft//2+1, 256]
    assert stft.shape[1] == LATENT_FRAMES, f"STFT gave {stft.shape[1]} frames, expected {LATENT_FRAMES}"

    rows = []

    # Broadband RMS (center=True to match STFT alignment; slice to 256)
    rms_broad = librosa.feature.rms(y=audio, frame_length=N_FFT,
                                     hop_length=HOP_LENGTH, center=True)[0, :LATENT_FRAMES]
    rows.append(rms_broad)

    # Band RMS (4 bands)
    for band_name in ["bass", "body", "mid", "air"]:
        low, high = _BANDS[band_name]
        rows.append(_bandpass_rms(audio, sr, low, high))

    # Spectral features from STFT
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    power = stft ** 2

    # Flatness
    rows.append(librosa.feature.spectral_flatness(S=stft)[0][:LATENT_FRAMES])

    # Flux
    flux = np.sqrt(np.sum(np.diff(stft, axis=1, prepend=stft[:, :1]) ** 2, axis=0))
    rows.append(flux[:LATENT_FRAMES])

    # Centroid (normalised to 0-1)
    centroid = librosa.feature.spectral_centroid(S=power, sr=sr)[0]
    rows.append((centroid[:LATENT_FRAMES] / (sr / 2)).astype(np.float32))

    # Skewness + Kurtosis (spectral moment proxies)
    p_norm = power / (power.sum(axis=0, keepdims=True) + 1e-9)
    f_col  = freqs[:, np.newaxis] / (sr / 2)
    mean_f = (f_col * p_norm).sum(axis=0)
    var_f  = ((f_col - mean_f[np.newaxis]) ** 2 * p_norm).sum(axis=0)
    std_f  = np.sqrt(var_f + 1e-12)
    skew_f = ((f_col - mean_f[np.newaxis]) ** 3 * p_norm).sum(axis=0) / (std_f ** 3 + 1e-12)
    kurt_f = ((f_col - mean_f[np.newaxis]) ** 4 * p_norm).sum(axis=0) / (std_f ** 4 + 1e-12)
    rows.append(skew_f[:LATENT_FRAMES].astype(np.float32))
    rows.append(kurt_f[:LATENT_FRAMES].astype(np.float32))

    # Chroma CQT (12 bins) — no center param; produces 257 frames, slice to 256
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=HOP_LENGTH,
                                         n_chroma=12)[:, :LATENT_FRAMES]  # [12, 256]
    for ci in range(12):
        rows.append(chroma[ci])

    # Onset strength (center=True, slice to 256 — consistent with STFT)
    onset = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=HOP_LENGTH,
                                          n_fft=N_FFT, center=True)
    rows.append(onset[:LATENT_FRAMES])

    out = np.array([np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
                    for r in rows], dtype=np.float32)
    assert out.shape == (len(TEMPORAL_FEATURE_NAMES), LATENT_FRAMES)
    return out
```

- [ ] **Step 3: Run STFT tests**

```bash
python3 -m pytest tests/test_latent_analysis.py::test_stft_produces_exactly_256_frames tests/test_latent_analysis.py::test_frame_features_no_nan -v
```
Expected: 2 PASSED.

- [ ] **Step 4: Write 04_temporal_correlation.py**

```python
# plots/latent_analysis/04_temporal_correlation.py
"""
Script 04 — Temporal correlation between latent dims and frame-level features.
Uses same 2000-crop subsample as script 03 (same seed).

Usage:
    python 04_temporal_correlation.py [--force]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plots.latent_analysis.config import (
    DATA_DIR, LATENT_DIR, INFO_DIR, LATENT_DIM, LATENT_FRAMES,
    SAMPLE_RATE, HOP_LENGTH, N_TEMPORAL_CROPS, RANDOM_SEED,
    TEMPORAL_FEATURE_NAMES,
)
from plots.latent_analysis._temporal_features import compute_frame_features
from plots.latent_analysis._corr_utils import compute_pearson_spearman
from plots.latent_analysis.findings import update_findings_section, update_progress
from plots.latent_analysis._03_collect import collect_latent_paths  # reuse from script 03

OUT_NPZ = DATA_DIR / "04_temporal.npz"
STEM_SUFFIXES = {"_bass", "_drums", "_other", "_vocals"}
EXPECTED_AUDIO_LEN = LATENT_FRAMES * HOP_LENGTH  # 524288 samples


def _find_audio(npy_path: Path) -> Path:
    """Find the raw audio crop (.flac/.wav) matching a latent NPY path."""
    track_name = npy_path.parent.name
    stem       = npy_path.stem
    audio_dir  = INFO_DIR / track_name
    for ext in [".flac", ".wav", ".mp3"]:
        candidate = audio_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def run(force: bool = False):
    if OUT_NPZ.exists() and not force:
        print("04_temporal.npz already exists. Use --force to recompute.")
        return

    if not LATENT_DIR.exists():
        raise RuntimeError(f"Latent dir not mounted: {LATENT_DIR}")

    print(f"Script 04: collecting {N_TEMPORAL_CROPS} latent paths (same seed as script 03)...")
    from plots.latent_analysis._03_collect import collect_latent_paths
    paths = collect_latent_paths(N_TEMPORAL_CROPS, RANDOM_SEED)

    # Accumulators: for each (dim, temporal_feat), append r value
    # Strategy: concatenate (lat_dim[t], feat[f,t]) across all crops
    # Shape after loop: lat_flat[64, N_crops*256], feat_flat[N_tfeats, N_crops*256]
    lat_segs  = []   # list of [64, 256]
    feat_segs = []   # list of [N_tfeats, 256]
    sample_crops       = []   # [<=50] latent arrays for Tab 4 display
    sample_feat_segs   = []   # [<=50] frame-feature arrays for Tab 4 overlay
    skipped = 0

    for i, npy_path in enumerate(paths):
        audio_path = _find_audio(npy_path)
        if audio_path is None:
            skipped += 1
            continue
        try:
            lat = np.load(str(npy_path)).astype(np.float32)
            assert lat.shape == (LATENT_DIM, LATENT_FRAMES)

            audio, sr = sf.read(str(audio_path), always_2d=True)
            audio = audio.mean(axis=1).astype(np.float32)  # mono
            if sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            if len(audio) < EXPECTED_AUDIO_LEN:
                skipped += 1
                continue
            audio = audio[:EXPECTED_AUDIO_LEN]

            feats = compute_frame_features(audio, sr=SAMPLE_RATE)  # [N_tfeats, 256]
            lat_segs.append(lat)
            feat_segs.append(feats)
            if len(sample_crops) < 50:
                sample_crops.append(lat)
                sample_feat_segs.append(feats)   # save matching frame features

        except Exception as e:
            skipped += 1
            continue

        if i % 200 == 0:
            print(f"  {i+1}/{len(paths)}, {len(lat_segs)} valid...")

    print(f"Valid crops: {len(lat_segs)} ({skipped} skipped — audio not found or wrong length)")

    # Concatenate all (crop, t) observations
    lat_all  = np.concatenate(lat_segs,  axis=1)   # [64, N*256]
    feat_all = np.concatenate(feat_segs, axis=1)    # [N_tfeats, N*256]

    # Temporal correlation: each (latent_dim, temporal_feat) pair
    n_tfeats = len(TEMPORAL_FEATURE_NAMES)
    r_temp   = np.zeros((LATENT_DIM, n_tfeats))
    p_temp   = np.ones( (LATENT_DIM, n_tfeats))

    for fi, fname in enumerate(TEMPORAL_FEATURE_NAMES):
        y = feat_all[fi]
        rp, pp, _, _ = compute_pearson_spearman(lat_all.T, y)
        r_temp[:, fi] = rp
        p_temp[:, fi] = pp
        if fi % 5 == 0:
            print(f"  temporal feat {fi+1}/{n_tfeats}: {fname}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        r_temporal=r_temp,
        temporal_feature_names=np.array(TEMPORAL_FEATURE_NAMES),
        n_crops_used=len(lat_segs),
        sample_crops=np.array(sample_crops),           # [K, 64, 256] latent time series
        sample_feat_segs=np.array(sample_feat_segs),   # [K, N_tfeats, 256] frame features
    )
    print(f"Saved {OUT_NPZ}")

    body = _findings_body(len(lat_segs), r_temp)
    update_findings_section("04", body)
    update_progress("04", f"Done. {len(lat_segs)} crops, temporal corr computed.")


def _findings_body(n_crops, r_temp):
    lines = [
        "## Script 04 — Temporal Correlation",
        f"- Crops used: {n_crops}",
        f"- Note: within-track crops treated as independent (effective N ≈ tracks sampled)",
        "",
        "### Strongest temporal (dim × frame-feature) correlations",
    ]
    flat     = r_temp.ravel()
    top_idx  = np.argsort(np.abs(flat))[::-1][:8]
    for idx in top_idx:
        dim = idx // len(TEMPORAL_FEATURE_NAMES)
        fi  = idx % len(TEMPORAL_FEATURE_NAMES)
        lines.append(
            f"- Dim {dim:2d} × `{TEMPORAL_FEATURE_NAMES[fi]}`: r = {r_temp[dim,fi]:+.3f}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(force=args.force)
```

- [ ] **Step 5: Extract shared path collection into `_03_collect.py`**

Script 04 imports `collect_latent_paths` from script 03. Refactor it out to avoid circular imports:

Create `plots/latent_analysis/_03_collect.py`:
```python
# plots/latent_analysis/_03_collect.py
"""Shared latent path collection used by scripts 03 and 04."""
import numpy as np
from pathlib import Path
from plots.latent_analysis.config import LATENT_DIR

STEM_SUFFIXES = {"_bass", "_drums", "_other", "_vocals"}

def collect_latent_paths(n_crops: int, seed: int):
    rng = np.random.default_rng(seed)
    all_paths = []
    for track_dir in LATENT_DIR.iterdir():
        if not track_dir.is_dir():
            continue
        for p in track_dir.glob("*.npy"):
            if not any(p.stem.endswith(s) for s in STEM_SUFFIXES):
                all_paths.append(p)
    rng.shuffle(all_paths)
    return all_paths[:n_crops]
```

Update `03_latent_xcorr.py` to use `from plots.latent_analysis._03_collect import collect_latent_paths` instead of its inline `_collect_latent_paths`.

- [ ] **Step 6: Run full test suite**

```bash
python3 -m pytest tests/test_latent_analysis.py -v
```
Expected: all PASSED.

- [ ] **Step 7: Commit**

```bash
git add plots/latent_analysis/_temporal_features.py plots/latent_analysis/04_temporal_correlation.py plots/latent_analysis/_03_collect.py plots/latent_analysis/03_latent_xcorr.py tests/test_latent_analysis.py
git commit -m "feat: script 04 — frame-level temporal features + latent temporal correlation"
```

---

## Task 8: app.py — Plotly Dash explorer

**Files:**
- Create: `plots/latent_analysis/app.py`

Six-tab Dash app reading pre-computed NPZ files. Port 7895.

- [ ] **Step 1: Write app.py**

```python
# plots/latent_analysis/app.py
"""
Latent Feature Analysis — Plotly Dash Explorer (port 7895)

Run:
    python plots/latent_analysis/app.py [--port 7895] [--debug]

Reads from:
    plots/latent_analysis/data/01_correlations.npz
    plots/latent_analysis/data/02_pca.npz
    plots/latent_analysis/data/03_xcorr.npz
    plots/latent_analysis/data/04_temporal.npz
    plots/latent_analysis/data/posters/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash.exceptions

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.latent_analysis.config import (
    DATA_DIR, POSTER_DIR, FEATURE_GROUPS, LATENT_DIM,
    POSTER_CLAMP, EFFECT_WEAK, EFFECT_STRONG,
    TEMPORAL_FEATURE_NAMES,
)

# ---------------------------------------------------------------------------
# Data loading (at startup)
# ---------------------------------------------------------------------------

def _load(npz_path):
    if not Path(npz_path).exists():
        return None
    return dict(np.load(str(npz_path), allow_pickle=True))


d01 = _load(DATA_DIR / "01_correlations.npz")
d02 = _load(DATA_DIR / "02_pca.npz")
d03 = _load(DATA_DIR / "03_xcorr.npz")
d04 = _load(DATA_DIR / "04_temporal.npz")

feat_names = list(d01["feature_names"]) if d01 else []
ALL_GROUPS = list(FEATURE_GROUPS.keys()) + ["All"]
cluster_labels = d03["cluster_labels"] if d03 else np.zeros(LATENT_DIM, dtype=int)

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="Latent Feature Explorer")

_tab_style        = {"padding": "8px 16px", "color": "#888"}
_tab_active_style = {"padding": "8px 16px", "backgroundColor": "#7eb8f7",
                     "color": "#0d0d1a", "fontWeight": "700"}

app.layout = html.Div([
    html.H2("Latent ↔ Feature Analysis", style={"margin": "12px 16px 4px"}),

    dcc.Tabs(id="tabs", value="corr", children=[
        dcc.Tab(label="Correlation Matrix", value="corr",
                style=_tab_style, selected_style=_tab_active_style),
        dcc.Tab(label="Feature Posters",    value="posters",
                style=_tab_style, selected_style=_tab_active_style),
        dcc.Tab(label="PCA Explorer",       value="pca",
                style=_tab_style, selected_style=_tab_active_style),
        dcc.Tab(label="Temporal",           value="temporal",
                style=_tab_style, selected_style=_tab_active_style),
        dcc.Tab(label="Latent Cross-Corr",  value="xcorr",
                style=_tab_style, selected_style=_tab_active_style),
        dcc.Tab(label="Cluster Map",        value="clusters",
                style=_tab_style, selected_style=_tab_active_style),
    ]),
    html.Div(id="tab-content", style={"padding": "12px"}),
], style={"fontFamily": "monospace", "backgroundColor": "#0d0d1a", "color": "#ccc",
          "minHeight": "100vh"})


# ---------------------------------------------------------------------------
# Tab 1 — Correlation Matrix
# ---------------------------------------------------------------------------

def _corr_layout():
    if d01 is None:
        return html.P("Run 01_aggregate_correlation.py first.")
    return html.Div([
        html.Div([
            html.Label("Sort dims by:"),
            dcc.RadioItems(id="corr-sort",
                           options=[{"label": "Feature loading", "value": "loading"},
                                    {"label": "Cluster order",   "value": "cluster"},
                                    {"label": "Index",           "value": "index"}],
                           value="loading", inline=True),
            html.Label("Feature group:", style={"marginLeft": "20px"}),
            dcc.Dropdown(id="corr-group", options=[{"label": g, "value": g} for g in ALL_GROUPS],
                         value="All", clearable=False, style={"width": "160px", "display": "inline-block"}),
            html.Label("|r| ≥", style={"marginLeft": "20px"}),
            dcc.Slider(id="corr-thresh", min=0, max=0.35, step=0.05, value=0.0,
                       marks={v: f"{v:.2f}" for v in [0, 0.1, 0.2, 0.35]}, tooltip={"always_visible": False}),
            html.Label("Show:", style={"marginLeft": "20px"}),
            dcc.RadioItems(id="corr-metric",
                           options=[{"label": "Pearson", "value": "pearson"},
                                    {"label": "Spearman", "value": "spearman"}],
                           value="pearson", inline=True),
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px",
                  "alignItems": "center", "marginBottom": "8px"}),
        dcc.Graph(id="corr-heatmap"),
        html.Div(id="corr-scatter-container"),
    ])


@app.callback(Output("corr-heatmap", "figure"),
              [Input("corr-sort", "value"), Input("corr-group", "value"),
               Input("corr-thresh", "value"), Input("corr-metric", "value")])
def update_corr_heatmap(sort_by, group, thresh, metric):
    if d01 is None:
        return go.Figure()
    r = d01["r_pearson"] if metric == "pearson" else d01["r_spearman"]  # [64, N_feat]
    names = feat_names

    # Filter by group
    if group != "All":
        group_feats = set(FEATURE_GROUPS.get(group, []))
        keep = [i for i, n in enumerate(names) if n in group_feats]
        r     = r[:, keep]
        names = [names[i] for i in keep]

    # Apply threshold: mask weak correlations
    r_disp = np.where(np.abs(r) >= thresh, r, 0.0)

    # Sort dims
    if sort_by == "loading":
        order = np.argsort(np.abs(r_disp).max(axis=1))[::-1]
    elif sort_by == "cluster":
        order = np.argsort(cluster_labels)
    else:
        order = np.arange(LATENT_DIM)
    r_disp = r_disp[order]

    fig = go.Figure(go.Heatmap(
        z=r_disp, x=names, y=[f"dim {i}" for i in order],
        colorscale="RdBu_r", zmid=0, zmin=-POSTER_CLAMP, zmax=POSTER_CLAMP,
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        template="plotly_dark", height=700,
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        margin=dict(l=60, r=20, t=30, b=120),
    )
    return fig


@app.callback(Output("corr-scatter-container", "children"),
              Input("corr-heatmap", "clickData"),
              State("corr-metric", "value"))
def corr_scatter(click_data, metric):
    if click_data is None or d01 is None:
        return html.P("Click a cell to see scatter plot.", style={"color": "#555"})
    pt  = click_data["points"][0]
    feat_name = pt["x"]
    dim_label = pt["y"]  # "dim N"
    dim = int(dim_label.split(" ")[1])
    fi  = feat_names.index(feat_name) if feat_name in feat_names else None
    if fi is None:
        return html.P(f"Feature {feat_name} not found.")
    r_val = d01["r_pearson"][dim, fi]

    scatter_d = _load(DATA_DIR / "scatter_sample.npz")
    if scatter_d is None:
        return html.P("scatter_sample.npz not found — run script 01.")
    lm        = scatter_d["latent_means"]          # [5000, 64]
    fv        = scatter_d["feature_values"]        # [5000, N_feat]
    fn        = list(scatter_d["feature_names"])   # list of feature name strings
    x = lm[:, dim]
    if feat_name in fn:
        y = fv[:, fn.index(feat_name)]
    else:
        return html.P(f"Feature {feat_name} not in scatter sample.")

    fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers",
                                  marker=dict(size=3, opacity=0.4)))
    fig.update_layout(template="plotly_dark", height=300,
                      title=f"dim {dim} × {feat_name} — Pearson r = {r_val:+.3f}",
                      xaxis_title=f"Latent dim {dim} mean",
                      yaxis_title=feat_name)
    return dcc.Graph(figure=fig)


# ---------------------------------------------------------------------------
# Tab 2 — Feature Posters
# ---------------------------------------------------------------------------

def _posters_layout():
    options = [{"label": n, "value": n} for n in feat_names]
    default = feat_names[0] if feat_names else None
    return html.Div([
        dcc.Dropdown(id="poster-feat", options=options, value=default,
                     clearable=False, style={"width": "300px"}),
        html.Div(id="poster-content", style={"marginTop": "12px"}),
    ])


@app.callback(Output("poster-content", "children"), Input("poster-feat", "value"))
def update_poster(feat_name):
    if not feat_name or d01 is None:
        return html.P("No data.")
    fi = feat_names.index(feat_name) if feat_name in feat_names else None
    if fi is None:
        return html.P(f"{feat_name} not in data.")
    r_col = d01["r_pearson"][:, fi]  # [64]
    n     = int(d01["n_per_feature"][fi])
    grid  = r_col.reshape(8, 8)

    fig = go.Figure(go.Heatmap(
        z=grid, colorscale="RdBu_r", zmid=0, zmin=-POSTER_CLAMP, zmax=POSTER_CLAMP,
        text=[[f"{grid[r,c]:.2f}" for c in range(8)] for r in range(8)],
        texttemplate="%{text}", textfont={"size": 10},
        xaxis="x", yaxis="y",
    ))
    top3_pos = np.argsort(r_col)[::-1][:3]
    top3_neg = np.argsort(r_col)[:3]
    fig.update_layout(
        template="plotly_dark", width=500, height=500,
        title=f"{feat_name} — N={n:,} | top+: dims {list(top3_pos)} | top-: dims {list(top3_neg)}",
        xaxis=dict(title="dim mod 8", tickvals=list(range(8))),
        yaxis=dict(title="dim // 8 × 8", tickvals=list(range(8)),
                   ticktext=[str(i*8) for i in range(8)]),
    )
    return dcc.Graph(figure=fig)


# ---------------------------------------------------------------------------
# Tab 3 — PCA Explorer
# ---------------------------------------------------------------------------

def _pca_layout():
    if d02 is None:
        return html.P("Run 02_pca_analysis.py first.")
    return html.Div([
        html.Div([
            html.Label("Colour by feature:"),
            dcc.Dropdown(id="pca-colour", options=[{"label": n, "value": n} for n in feat_names],
                         value=feat_names[0] if feat_names else None,
                         style={"width": "220px", "display": "inline-block"}),
            html.Label("PC axes:", style={"marginLeft": "16px"}),
            dcc.RadioItems(id="pca-axes",
                           options=[{"label": "PC1 vs PC2", "value": "12"},
                                    {"label": "PC1 vs PC3", "value": "13"}],
                           value="12", inline=True),
        ], style={"display": "flex", "gap": "12px", "alignItems": "center", "marginBottom": "8px"}),
        dcc.Graph(id="pca-scatter"),
        html.H4("Cross-PCA alignment (Feature PC → Latent PC)", style={"marginTop": "16px"}),
        dcc.Graph(id="pca-cross-heatmap"),
    ])


@app.callback(Output("pca-scatter", "figure"),
              [Input("pca-colour", "value"), Input("pca-axes", "value")])
def update_pca_scatter(colour_feat, axes):
    if d02 is None:
        return go.Figure()
    scores = d02["latent_scores"]  # [N, 20]
    pc1 = int(axes[0]) - 1
    pc2 = int(axes[1]) - 1

    # Colour by feature using scatter_sample.npz (saved by script 01)
    colour = None
    scatter_d = _load(DATA_DIR / "scatter_sample.npz")
    if colour_feat and scatter_d is not None:
        fn = list(scatter_d["feature_names"])
        if colour_feat in fn:
            # scatter_sample has 5000 crops; pca scores has N_crops — use crop_ids to align
            # For now use a random subsample of pca scores matching scatter size
            n_scatter = scatter_d["latent_means"].shape[0]
            colour = scatter_d["feature_values"][:, fn.index(colour_feat)][:n_scatter]

    ev = d02["latent_explained_variance_ratio"]
    fig = go.Figure(go.Scattergl(
        x=scores[:, pc1], y=scores[:, pc2],
        mode="markers", marker=dict(size=2, opacity=0.3, color=colour),
    ))
    fig.update_layout(
        template="plotly_dark", height=500,
        xaxis_title=f"Latent PC{pc1+1} ({ev[pc1]:.1%})",
        yaxis_title=f"Latent PC{pc2+1} ({ev[pc2]:.1%})",
        title="Latent PCA scatter",
    )
    return fig


@app.callback(Output("pca-cross-heatmap", "figure"), Input("tabs", "value"))
def update_cross_heatmap(tab):
    if tab != "pca" or d02 is None:
        return go.Figure()
    cc = d02["cross_corr"]  # [n_feat_pc, n_lat_pc]
    fig = go.Figure(go.Heatmap(
        z=cc, colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        xaxis="x", yaxis="y",
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        template="plotly_dark", height=400,
        xaxis_title="Latent PC", yaxis_title="Feature PC",
        title="Feature PC ↔ Latent PC correlation",
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 4 — Temporal
# ---------------------------------------------------------------------------

def _temporal_layout():
    if d04 is None:
        return html.P("Run 04_temporal_correlation.py first.")
    n_crops = int(d04.get("n_crops_used", 0))
    sample  = d04.get("sample_crops")
    crop_opts = [{"label": f"crop {i}", "value": i}
                 for i in range(len(sample) if sample is not None else 0)]
    return html.Div([
        html.Div([
            html.Label("Crop:"),
            dcc.Dropdown(id="temp-crop", options=crop_opts, value=0,
                         style={"width": "140px", "display": "inline-block"}),
            html.Label("Dims (comma-sep):", style={"marginLeft": "16px"}),
            dcc.Input(id="temp-dims", value="0,1,2,3,4", type="text",
                      style={"width": "180px"}),
            html.Label("BPM source:", style={"marginLeft": "16px"}),
            dcc.RadioItems(id="temp-bpm",
                           options=[{"label": "Essentia", "value": "essentia"},
                                    {"label": "Madmom",   "value": "madmom"},
                                    {"label": "Average",  "value": "avg"}],
                           value="essentia", inline=True),
        ], style={"display": "flex", "gap": "10px", "alignItems": "center", "marginBottom": "8px"}),
        dcc.Graph(id="temporal-graph"),
    ])


@app.callback(Output("temporal-graph", "figure"),
              [Input("temp-crop", "value"), Input("temp-dims", "value"),
               Input("temp-bpm", "value")])
def update_temporal(crop_idx, dims_str, bpm_src):
    if d04 is None:
        return go.Figure()
    sample      = d04.get("sample_crops")        # [K, 64, 256]
    feat_sample = d04.get("sample_feat_segs")    # [K, N_tfeats, 256]
    tfeat_names = list(d04.get("temporal_feature_names", TEMPORAL_FEATURE_NAMES))

    if sample is None or crop_idx is None or crop_idx >= len(sample):
        return go.Figure()

    try:
        dims = [int(d.strip()) for d in dims_str.split(",")]
        dims = [d for d in dims if 0 <= d < LATENT_DIM]
    except ValueError:
        dims = [0]

    lat = sample[crop_idx]   # [64, 256]
    t   = np.arange(256) * (2048 / 44100)   # time in seconds

    # Two-axis figure: left=latent dims, right=temporal features (normalised)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for dim in dims:
        fig.add_trace(go.Scattergl(x=t, y=lat[dim], mode="lines",
                                    name=f"dim {dim}", line=dict(width=1)),
                      secondary_y=False)

    # Overlay temporal features (z-scored for comparability)
    if feat_sample is not None and crop_idx < len(feat_sample):
        feats = feat_sample[crop_idx]   # [N_tfeats, 256]
        # Show broadband RMS and spectral flatness by default (indices 0 and 5)
        for fi in [0, 5]:
            if fi < feats.shape[0]:
                y = feats[fi]
                y_norm = (y - y.mean()) / (y.std() + 1e-9)
                fig.add_trace(go.Scattergl(x=t, y=y_norm, mode="lines",
                                            name=tfeat_names[fi],
                                            line=dict(width=1, dash="dot")),
                              secondary_y=True)

    fig.update_layout(
        template="plotly_dark", height=500,
        xaxis_title="Time (s)",
        title=f"Latent dims + frame features — sample crop {crop_idx}",
        legend=dict(orientation="h"),
    )
    fig.update_yaxes(title_text="Latent value",        secondary_y=False)
    fig.update_yaxes(title_text="Feature (z-scored)",  secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# Tab 5 — Latent Cross-Corr
# ---------------------------------------------------------------------------

def _xcorr_layout():
    if d03 is None:
        return html.P("Run 03_latent_xcorr.py first.")
    return dcc.Graph(id="xcorr-heatmap",
                     figure=_build_xcorr_fig())


def _build_xcorr_fig():
    xcorr = d03["xcorr_matrix"]  # [64, 64]
    cl    = d03["cluster_labels"]
    order = np.argsort(cl)
    xcorr_ord = xcorr[np.ix_(order, order)]
    labels_ord = [f"dim {i}" for i in order]
    fig = go.Figure(go.Heatmap(
        z=xcorr_ord, x=labels_ord, y=labels_ord,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        template="plotly_dark", height=700,
        title=f"64×64 Latent Cross-Correlation (Fisher-Z avg, {int(d03['n_crops_used'])} crops, sorted by cluster)",
        xaxis=dict(tickfont=dict(size=7)), yaxis=dict(tickfont=dict(size=7)),
        margin=dict(l=80, r=20, t=50, b=80),
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 6 — Cluster Map
# ---------------------------------------------------------------------------

def _cluster_layout():
    if d03 is None or d01 is None:
        return html.P("Run scripts 01 and 03 first.")
    cl     = d03["cluster_labels"]
    r_mat  = d01["r_pearson"]
    n_cl   = int(cl.max())

    rows = []
    for c in range(1, n_cl + 1):
        dims = np.where(cl == c)[0]
        mean_r = r_mat[dims].mean(axis=0)     # avg r per feature
        top3   = np.argsort(np.abs(mean_r))[::-1][:3]
        summary = ", ".join(
            f"{feat_names[i]} (r={mean_r[i]:+.2f})"
            for i in top3 if i < len(feat_names)
        )
        rows.append(html.Tr([
            html.Td(f"Cluster {c}"),
            html.Td(", ".join(map(str, dims))),
            html.Td(summary, style={"fontSize": "0.85em"}),
        ]))

    return html.Div([
        html.Table(
            [html.Tr([html.Th("Cluster"), html.Th("Dims"), html.Th("Top correlated features")])] + rows,
            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.9em"},
        ),
    ])


# ---------------------------------------------------------------------------
# Main tab router
# ---------------------------------------------------------------------------

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "corr":    return _corr_layout()
    if tab == "posters": return _posters_layout()
    if tab == "pca":     return _pca_layout()
    if tab == "temporal":return _temporal_layout()
    if tab == "xcorr":   return _xcorr_layout()
    if tab == "clusters":return _cluster_layout()
    return html.P("Unknown tab.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int, default=7895)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"Starting Latent Feature Explorer on http://localhost:{args.port}")
    print(f"Data loaded: 01={'✓' if d01 else '✗'} 02={'✓' if d02 else '✗'} "
          f"03={'✓' if d03 else '✗'} 04={'✓' if d04 else '✗'}")
    app.run(debug=args.debug, port=args.port, host="127.0.0.1")
```

- [ ] **Step 2: Smoke test — app imports without error even with no data files present**

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('app', 'plots/latent_analysis/app.py')
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('app import ok, d01:', mod.d01 is not None)
"
```
Expected: `app import ok, d01: False` (or True if data exists).

- [ ] **Step 3: Run full test suite one final time**

```bash
python3 -m pytest tests/test_latent_analysis.py -v
```
Expected: all PASSED.

- [ ] **Step 4: Commit**

```bash
git add plots/latent_analysis/app.py
git commit -m "feat: Plotly Dash app — 6-tab latent feature explorer (port 7895)"
```

---

## Running the Full Pipeline

Once all scripts are committed, run in order:

```bash
cd /home/kim/Projects/mir
python plots/latent_analysis/01_aggregate_correlation.py
python plots/latent_analysis/02_pca_analysis.py
python plots/latent_analysis/03_latent_xcorr.py
python plots/latent_analysis/04_temporal_correlation.py
python plots/latent_analysis/app.py --debug
```

Then open http://localhost:7895.

To re-run any script (e.g. after adding features):
```bash
python plots/latent_analysis/01_aggregate_correlation.py --force
```

Findings are in `plots/latent_analysis/FINDINGS.md`.
Progress log is in `plots/latent_analysis/PROGRESS.md`.
