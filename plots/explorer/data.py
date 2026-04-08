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

FEATURE_GROUPS: dict[str, list[str]] = {
    "Rhythm":    ["bpm", "danceability", "beat_count"],
    "Spectral":  ["brightness", "roughness", "hardness", "depth",
                  "booming", "sharpness", "warmth",
                  "spectral_flatness", "spectral_flux",
                  "spectral_skewness", "spectral_kurtosis"],
    "Loudness":  ["lufs", "lra", "lufs_bass", "lufs_drums", "lufs_other", "lufs_vocals",
                  "rms_energy_bass", "rms_energy_body", "rms_energy_mid", "rms_energy_air"],
    "Timbral":   ["reverberation"],
    "Voice":     ["voice_probability", "female_probability", "male_probability",
                  "instrumental_probability"],
    "Aesthetic": ["content_enjoyment", "content_usefulness",
                  "production_complexity", "production_quality"],
    "Metadata":  ["atonality", "popularity", "release_year", "duration"],
}

# Columns treated as categorical class groupings for "Classes" mode
CLASS_FIELDS = ["essentia_genre", "essentia_instrument", "essentia_mood",
                "genres", "label"]

# Columns that pandas may misdetect as numeric due to all-NaN rows
_FORCE_CATEGORICAL = {"has_stems", "musicbrainz_id", "source", "stem_names",
                       "track_metadata_genre", "spotify_id"}

# ── Data structures ───────────────────────────────────────────────────────────
_META_COLS = ["spotify_id", "musicbrainz_id", "tidal_url", "tidal_id"]


@dataclass
class AppData:
    """Loaded and indexed dataset. Immutable after construction."""
    tracks: list[str]                   # display names, one per row
    artists: list[str]                  # lowercased artist string per row (for search)
    num_cols: list[str]                 # numeric feature column names
    class_cols: list[str]               # categorical column names
    _df_num: pd.DataFrame = field(repr=False)
    _df_class: pd.DataFrame = field(repr=False)
    _df_meta: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def feat_array(self, col: str) -> np.ndarray:
        """Return float64 array [n_tracks] for numeric column, NaN where missing."""
        return self._df_num[col].to_numpy(dtype=np.float64, na_value=np.nan)

    def class_array(self, col: str) -> list[str]:
        """Return list of raw string values for a class column."""
        return self._df_class[col].fillna("").tolist()

    def meta_val(self, col: str, idx: int) -> str:
        """Return metadata string value (e.g. spotify_id) for a track index, or ''."""
        if col not in self._df_meta.columns:
            return ""
        v = self._df_meta[col].iloc[idx]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return str(v).strip()

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

    def _artist_str(raw) -> str:
        if pd.isna(raw) or str(raw).strip() in ("", "nan"):
            return ""
        try:
            lst = ast.literal_eval(str(raw))
            return " ".join(lst).lower() if isinstance(lst, list) else str(raw).lower()
        except Exception:
            return str(raw).lower()

    artists = [_artist_str(a) for a in df.get("artists", pd.Series([""] * len(df)))]

    meta_present = [c for c in _META_COLS if c in df.columns]

    return AppData(
        tracks=tracks,
        artists=artists,
        num_cols=num_cols,
        class_cols=class_cols,
        _df_num=df[num_cols].copy(),
        _df_class=df[class_cols].copy() if class_cols else pd.DataFrame(),
        _df_meta=df[meta_present].copy() if meta_present else pd.DataFrame(),
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
    """
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
        sidecar = Path(latent_dir) / track / f"{crop}.json"
        n_bars  = _count_loopable_bars(sidecar)
        if n_bars >= 4:
            eligible.append(z)

    if not eligible:
        eligible = [load_latent(track, c, latent_dir) for c in crops]

    t_max = max(z.shape[1] for z in eligible)
    stack = np.zeros((len(eligible), 64, t_max), dtype=np.float32)
    for i, z in enumerate(eligible):
        stack[i, :, : z.shape[1]] = z

    return stack.mean(axis=0)                  # [64, T_max]


def _count_loopable_bars(sidecar_path: Path) -> int:
    """Return number of complete 4/4 bars from the sidecar JSON. 0 if absent."""
    import json as _json
    if not sidecar_path.exists():
        return 0
    try:
        with open(sidecar_path) as f:
            meta = _json.load(f)
        beats = meta.get("beats") or meta.get("beat_times") or []
        return len(beats) // 4
    except Exception:
        return 0


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
