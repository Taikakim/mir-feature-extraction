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
