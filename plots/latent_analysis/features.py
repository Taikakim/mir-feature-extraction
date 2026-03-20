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
