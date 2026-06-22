"""Live, on-demand latent analysis over a sampled subset of crops.

All functions take already-loaded latents [256, T] (variable T) and pool
frames across crops. No precompute, no disk artifacts.
"""
from __future__ import annotations
import numpy as np


def _stack_frames(latents: list[np.ndarray]) -> np.ndarray:
    """Concatenate all crops' frames → [n_frames_total, 256]."""
    return np.concatenate([z.T for z in latents], axis=0).astype(np.float64)


def pca_frames(latents: list[np.ndarray], k: int = 3):
    X = _stack_frames(latents)
    X = X - X.mean(axis=0, keepdims=True)
    # SVD of centered frames; components are right-singular vectors
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = (S ** 2) / max(1, X.shape[0] - 1)
    evr = var / var.sum()
    return Vt[:k].astype(np.float32), evr[:k].astype(np.float32)


def dim_xcorr(latents: list[np.ndarray]) -> np.ndarray:
    X = _stack_frames(latents)            # [N, 256]
    return np.corrcoef(X, rowvar=False).astype(np.float32)


def dim_feature_corr(latents: list[np.ndarray],
                     feature_ts: list[np.ndarray]) -> np.ndarray:
    """Per-dim Pearson r vs a per-frame feature, pooled across crops."""
    dim_cols, feat_vals = [], []
    for z, f in zip(latents, feature_ts):
        T = min(z.shape[1], len(f))
        dim_cols.append(z[:, :T].T)        # [T, 256]
        feat_vals.append(np.asarray(f[:T], dtype=np.float64))
    X = np.concatenate(dim_cols, axis=0).astype(np.float64)   # [N, 256]
    fv = np.concatenate(feat_vals, axis=0)                    # [N]
    Xc = X - X.mean(axis=0, keepdims=True)
    fc = fv - fv.mean()
    num = Xc.T @ fc
    den = np.sqrt((Xc ** 2).sum(axis=0) * (fc ** 2).sum())
    den = np.where(den < 1e-12, np.nan, den)
    return (num / den).astype(np.float32)
