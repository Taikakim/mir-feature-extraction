"""Contour alphabet: weak orderings of L values, enumerated and indexed.

A "contour" here is the Kant & Polansky combinatorial contour reduced to its
normal form -- the dense rank vector of the underlying values. The number of
such vectors on L points is the ordered Bell (Fubini) number, which is small
enough to enumerate for L <= 7, so a musical trajectory can be turned into a
token stream.

Monotone invariance is the point: dense rank is unchanged by any strictly
increasing transform of the values, which covers gain changes, log1p
compression, and any fixed EQ curve.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

# Ordered Bell / Fubini numbers a(L) = number of weak orderings on L elements.
# Index by L directly: ORDERED_BELL[3] == 13.
ORDERED_BELL: tuple[int, ...] = (1, 1, 3, 13, 75, 541, 4683, 47293, 545835)

# Enumerating L=8 means 8**8 = 16.7M candidates; refuse rather than hang.
MAX_ENUMERABLE_L = 7


def dense_rank(values: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Dense (gap-free) rank of ``values``, ascending, starting at 0.

    Values within ``tol`` of their sorted neighbour share a rank. ``tol=0.0``
    means exact equality only.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return np.empty(0, dtype=np.int64)

    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.int64)
    ranks[order[0]] = 0
    current = 0
    for i in range(1, v.size):
        if v[order[i]] - v[order[i - 1]] > tol:
            current += 1
        ranks[order[i]] = current
    return ranks


@lru_cache(maxsize=None)
def enumerate_contours(L: int) -> tuple[tuple[int, ...], ...]:
    """All dense rank vectors of length ``L``, in lexicographic order."""
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    if L > MAX_ENUMERABLE_L:
        raise ValueError(
            f"L={L} would enumerate {L ** L} candidates; "
            f"max supported is {MAX_ENUMERABLE_L}. Use a coarser grid instead."
        )
    out = []
    for cand in itertools.product(range(L), repeat=L):
        if set(cand) == set(range(max(cand) + 1)):
            out.append(cand)
    return tuple(out)


@lru_cache(maxsize=None)
def contour_table(L: int) -> dict[tuple[int, ...], int]:
    """Bijection from dense rank vector to symbol index ``0..n_contours(L)-1``."""
    return {c: i for i, c in enumerate(enumerate_contours(L))}


def n_contours(L: int) -> int:
    """Size of the contour alphabet for length ``L``."""
    return len(enumerate_contours(L))


def symbol_of(values: np.ndarray, L: int, tol: float = 0.0) -> int:
    """Symbol index of the contour traced by ``values`` (which must have length L)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size != L:
        raise ValueError(f"expected values of length L={L}, got {v.size}")
    key = tuple(int(x) for x in dense_rank(v, tol=tol))
    return contour_table(L)[key]
