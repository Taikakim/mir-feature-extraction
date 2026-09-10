"""Where to sample a morph: frame, event (note-onset), or beat grids.

A "grid" is just an ascending array of latent-frame indices. Contours are then
taken over sliding windows of L consecutive grid points. Choosing the grid is
the main design lever: a note-onset grid gives tempo-invariant contours and far
fewer near-ties than a fixed frame grid, because note pitches are well
separated where consecutive chroma frames are not.
"""

from __future__ import annotations

import numpy as np

# SAME / Stable Audio 3 latent rate. NOT 44100/2048 (Stable Audio Open).
SAMPLE_RATE: int = 44100
SAMPLES_PER_LATENT: int = 4096
N_FFT: int = 8192

FRAME_RATE_HZ: float = SAMPLE_RATE / SAMPLES_PER_LATENT      # 10.7666015625
FRAME_PERIOD_S: float = SAMPLES_PER_LATENT / SAMPLE_RATE     # 0.0928798...
ANALYSIS_WINDOW_S: float = N_FFT / SAMPLE_RATE               # 0.1857596...


def seconds_to_frames(t_s: np.ndarray) -> np.ndarray:
    """Convert times in seconds to nearest latent-frame indices."""
    t = np.asarray(t_s, dtype=np.float64).ravel()
    return np.rint(t * FRAME_RATE_HZ).astype(np.int64)


def frame_grid(n_frames: int, stride: int = 1) -> np.ndarray:
    """Every ``stride``-th latent frame."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return np.arange(0, n_frames, stride, dtype=np.int64)


def event_grid(event_times_s: np.ndarray, n_frames: int) -> np.ndarray:
    """Latent-frame indices of musical events (note onsets, beats, downbeats).

    Events are rounded to frames, sorted, de-duplicated, and clipped to
    ``[0, n_frames)``. De-duplication is not optional: at 16th notes and
    140 BPM events land ~107 ms apart, which is close to the 92.9 ms frame
    period and well inside the 185.8 ms analysis window, so collisions are
    the normal case rather than an edge case.
    """
    frames = seconds_to_frames(event_times_s)
    frames = frames[(frames >= 0) & (frames < n_frames)]
    return np.unique(frames).astype(np.int64)


def sliding_windows(
    points: np.ndarray, L: int, n_future: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows of ``L`` consecutive grid points, with anchor frames.

    ``n_future`` is how many of the L points lie *after* the anchor. 0 gives a
    trailing (causal) contour describing history; ``L-1`` gives a fully
    leading one. Acausal windows are legitimate here because flow matching is
    not autoregressive and the condition is supplied by the user rather than
    predicted from the output.

    Returns ``(windows, anchors)`` where ``windows`` has shape ``(n, L)`` of
    frame indices and ``anchors`` has shape ``(n,)``.
    """
    pts = np.asarray(points, dtype=np.int64).ravel()
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    if not (0 <= n_future <= L - 1):
        raise ValueError(f"n_future must be in [0, L-1]={L - 1}, got {n_future}")

    n = pts.size - L + 1
    if n <= 0:
        return np.empty((0, L), dtype=np.int64), np.empty(0, dtype=np.int64)

    idx = np.arange(n)[:, None] + np.arange(L)[None, :]
    windows = pts[idx]
    anchors = windows[:, L - 1 - n_future]
    return windows.astype(np.int64), anchors.astype(np.int64)
