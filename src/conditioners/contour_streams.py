"""Assemble a value stream plus a grid into a stream of contour symbols.

Two stages, deliberately separate:

  values + grid  ->  (symbols, anchors)   one symbol per window
  (symbols, anchors) -> per-frame array   piecewise-constant, held forward

The second stage exists because a conditioner is per-frame while the symbols
are per-window (and, on an event grid, per-note). Holding forward is the
conservative choice: a frame is described by the most recent window that
covers it, and frames before the first window are explicitly undefined rather
than silently zero.
"""

from __future__ import annotations

import numpy as np

from src.conditioners.contour_codes import symbol_of
from src.conditioners.morph_grids import sliding_windows

UNDEFINED: int = -1


def contour_stream(
    values: np.ndarray,
    points: np.ndarray,
    L: int,
    n_future: int = 0,
    tol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Contour symbols over sliding windows of ``L`` grid points.

    ``values`` is indexed *by frame*: ``values[f]`` is the morph value at
    latent frame ``f``. ``points`` selects which frames form the grid, so a
    sparse grid selects values rather than resampling them.
    """
    vals = np.asarray(values, dtype=np.float64).ravel()
    windows, anchors = sliding_windows(points, L=L, n_future=n_future)
    if windows.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    if windows.max() >= vals.size or windows.min() < 0:
        raise IndexError(
            f"grid points reference frames outside values of length {vals.size}"
        )

    symbols = np.empty(windows.shape[0], dtype=np.int64)
    for i in range(windows.shape[0]):
        symbols[i] = symbol_of(vals[windows[i]], L=L, tol=tol)
    return symbols, anchors


def expand_to_frames(
    symbols: np.ndarray, anchors: np.ndarray, n_frames: int
) -> np.ndarray:
    """Piecewise-constant expansion of a symbol stream onto every latent frame.

    Each symbol is held forward from its anchor frame until the next anchor.
    Frames before the first anchor are ``UNDEFINED``.
    """
    out = np.full(n_frames, UNDEFINED, dtype=np.int64)
    syms = np.asarray(symbols, dtype=np.int64).ravel()
    anch = np.asarray(anchors, dtype=np.int64).ravel()
    if syms.size == 0:
        return out

    order = np.argsort(anch, kind="stable")
    syms, anch = syms[order], anch[order]
    for i in range(syms.size):
        start = max(int(anch[i]), 0)
        stop = int(anch[i + 1]) if i + 1 < syms.size else n_frames
        stop = min(max(stop, start), n_frames)
        out[start:stop] = syms[i]
    return out
