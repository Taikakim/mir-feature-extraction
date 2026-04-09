# plots/latent_analysis/_03_collect.py
"""Shared latent path collection used by scripts 03 and 04."""
import json
import numpy as np
from pathlib import Path
from plots.latent_analysis.config import LATENT_DIR, LATENT_FRAMES, MIN_VALID_FRACTION

STEM_SUFFIXES = {"_bass", "_drums", "_other", "_vocals"}
MIN_VALID_FRAMES = int(MIN_VALID_FRACTION * LATENT_FRAMES)


def _valid_crop(npy_path: Path) -> bool:
    """Return True if the companion .json padding_mask has enough valid frames."""
    json_path = npy_path.with_suffix(".json")
    if not json_path.exists():
        return True  # no metadata → accept (older crops have no json)
    try:
        with open(json_path) as f:
            meta = json.load(f)
        mask = meta.get("padding_mask")
        if mask is not None:
            return sum(mask) >= MIN_VALID_FRAMES
    except Exception:
        pass
    return True


def collect_latent_paths(n_crops: int, seed: int):
    rng = np.random.default_rng(seed)
    all_paths = []
    for track_dir in LATENT_DIR.iterdir():
        if not track_dir.is_dir():
            continue
        for p in track_dir.glob("*.npy"):
            if not any(p.stem.endswith(s) for s in STEM_SUFFIXES):
                if _valid_crop(p):
                    all_paths.append(p)
    rng.shuffle(all_paths)
    return all_paths[:n_crops]
