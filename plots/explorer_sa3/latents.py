"""Pure loaders for SA3 latents and their TIMESERIES sidecars."""
from __future__ import annotations
from pathlib import Path
import numpy as np

LATENT_DIM = 256
N_FRAMES = 4096
FRAME_RATE_HZ = 44100 / 4096   # 10.7666 Hz


def load_latent(latent_dir: Path, crop_id: str) -> np.ndarray:
    """Load NNNNNN.npy as float32 [256, T]. Squeezes a leading batch dim."""
    arr = np.load(Path(latent_dir) / f"{crop_id}.npy").astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] != LATENT_DIM:
        raise ValueError(
            f"{crop_id}: expected [{LATENT_DIM}, T], got {arr.shape}")
    return arr


def content_frames(meta: dict) -> int:
    """Number of non-padding frames from padding_mask; full T if absent."""
    mask = meta.get("padding_mask")
    if not mask:
        return N_FRAMES
    return int(sum(mask))


def load_timeseries(latent_dir: Path, crop_id: str) -> dict[str, np.ndarray]:
    p = Path(latent_dir) / f"{crop_id}.TIMESERIES.npz"
    with np.load(p) as z:
        return {k: z[k] for k in z.files}
