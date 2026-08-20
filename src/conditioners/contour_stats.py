"""The three statistics that decide whether a contour code is worth using.

  entropy          -- does the code carry information, or is one symbol
                      dominating? (Bar-level codes on repetitive genres are
                      the expected failure.)
  agreement        -- are two candidate streams redundant? Redundancy only
                      buys robustness if the streams fail independently;
                      near-perfect agreement means we bought width, not
                      information.

                      CAUTION: agreement tests symbol *equality*, so it is
                      only meaningful for two streams over the same alphabet
                      with the same semantics. Across different L (or any two
                      different alphabets) it is worse than useless: a stream
                      that is a perfect deterministic function of another
                      scores agreement 0.0, which reads as "independent."
                      Use `normalised_conditional_entropy` for those.

  conditional      -- how much of stream A survives once B is known? This is
    entropy          the diagnostic that actually detects redundancy, because
                     it is invariant to relabelling and works across
                     alphabets. H(A|B) = 0 means A carries no information B
                     does not already have, whatever the symbols look like.
  flip rate        -- how often does a symbol change under small perturbation
                      of the underlying values? This is the fragility measure,
                      and the main reason to prefer an event grid (well
                      separated note pitches) over a frame grid (near-tied
                      consecutive chroma frames).
"""

from __future__ import annotations

import numpy as np

from src.conditioners.contour_streams import UNDEFINED, contour_stream


def symbol_entropy(symbols: np.ndarray) -> float:
    """Shannon entropy of a symbol stream in bits, ignoring ``UNDEFINED``."""
    s = np.asarray(symbols, dtype=np.int64).ravel()
    s = s[s != UNDEFINED]
    if s.size == 0:
        return 0.0
    _, counts = np.unique(s, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def normalised_entropy(symbols: np.ndarray, alphabet_size: int) -> float:
    """Entropy as a fraction of the maximum possible for the alphabet."""
    if alphabet_size < 2:
        return 0.0
    return symbol_entropy(symbols) / float(np.log2(alphabet_size))


def pairwise_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of positions where two streams carry the same symbol.

    Positions where either stream is ``UNDEFINED`` are skipped.
    """
    x = np.asarray(a, dtype=np.int64).ravel()
    y = np.asarray(b, dtype=np.int64).ravel()
    if x.size != y.size:
        raise ValueError(f"streams must have the same length, got {x.size} and {y.size}")
    mask = (x != UNDEFINED) & (y != UNDEFINED)
    if not np.any(mask):
        return 0.0
    return float(np.mean(x[mask] == y[mask]))


def conditional_entropy(a: np.ndarray, b: np.ndarray) -> float:
    """``H(a | b)`` in bits: information left in ``a`` once ``b`` is known.

    Zero means ``a`` is a deterministic function of ``b`` and therefore adds
    nothing as a parallel stream, no matter how different the two symbol
    alphabets look. Positions where either stream is ``UNDEFINED`` are skipped.
    """
    x = np.asarray(a, dtype=np.int64).ravel()
    y = np.asarray(b, dtype=np.int64).ravel()
    if x.size != y.size:
        raise ValueError(f"streams must have the same length, got {x.size} and {y.size}")
    mask = (x != UNDEFINED) & (y != UNDEFINED)
    x, y = x[mask], y[mask]
    if x.size == 0:
        return 0.0

    total = 0.0
    for vb in np.unique(y):
        sel = x[y == vb]
        _, counts = np.unique(sel, return_counts=True)
        p = counts.astype(np.float64) / counts.sum()
        total += (sel.size / x.size) * float(-np.sum(p * np.log2(p)))
    return total


def normalised_conditional_entropy(a: np.ndarray, b: np.ndarray) -> float:
    """``H(a|b) / H(a)`` — the fraction of ``a`` that ``b`` does not explain.

    1.0 means ``b`` tells us nothing about ``a``; 0.0 means ``a`` is redundant
    given ``b``. Returns 0.0 when ``a`` is constant (nothing to explain).
    """
    h_a = symbol_entropy(np.asarray(a, dtype=np.int64).ravel())
    if h_a <= 0.0:
        return 0.0
    return conditional_entropy(a, b) / h_a


def stream_redundancy(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Symmetric redundancy verdict for a pair of streams.

    ``min_norm_cond_entropy`` is the number to threshold on: it is small when
    *either* stream is nearly determined by the other, which is the condition
    under which the pair buys width rather than robustness.
    """
    ab = normalised_conditional_entropy(a, b)
    ba = normalised_conditional_entropy(b, a)
    return {
        "agreement": pairwise_agreement(a, b),
        "norm_cond_entropy_a_given_b": ab,
        "norm_cond_entropy_b_given_a": ba,
        "min_norm_cond_entropy": min(ab, ba),
    }


def flip_rate(
    values: np.ndarray,
    points: np.ndarray,
    L: int,
    sigma: float,
    n_trials: int = 20,
    seed: int = 0,
    n_future: int = 0,
    tol: float = 0.0,
) -> float:
    """Fraction of symbols that change when values are perturbed by N(0, sigma).

    ``sigma`` should be expressed in the same units as ``values``; scale it
    relative to the value spread to compare grids fairly.
    """
    vals = np.asarray(values, dtype=np.float64).ravel()
    base, _ = contour_stream(vals, points, L=L, n_future=n_future, tol=tol)
    if base.size == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    flips = 0
    total = 0
    for _ in range(n_trials):
        noisy = vals + rng.normal(scale=sigma, size=vals.size)
        trial, _ = contour_stream(noisy, points, L=L, n_future=n_future, tol=tol)
        flips += int(np.sum(trial != base))
        total += int(base.size)
    return float(flips) / float(total)
