"""Regression tests for src/conditioners (contour codes/streams/stats/grids).

Origin: external-agent drafts reviewed + verified by CONTINUITY 2026-08-21
(Kim's overnight review ask). The load-bearing checks:
  - alphabet sizes are the ordered Bell numbers (K&P combinatorial contours);
  - symbols are invariant under monotone transforms (gain / log1p / EQ offset);
  - the redundancy diagnostic: pairwise AGREEMENT cannot see cross-alphabet
    redundancy (a deterministic relabel scores ~chance), conditional entropy can
    (H(A|B)=0) — the defect the second-pass note demonstrated on nested-L pairs;
  - nested-L streams ARE deterministically redundant (H(L2|L3)=0 at shared anchors).
Run: mir/bin/python -m pytest tests/test_conditioners_contour.py -q
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.conditioners.contour_codes import n_contours, symbol_of
from src.conditioners.contour_streams import contour_stream, expand_to_frames, UNDEFINED
from src.conditioners.contour_stats import stream_redundancy, conditional_entropy
from src.conditioners.morph_grids import frame_grid, event_grid, sliding_windows, FRAME_RATE_HZ


def test_alphabet_sizes_are_fubini():
    assert [n_contours(L) for L in (1, 2, 3, 4, 5, 6)] == [1, 3, 13, 75, 541, 4683]


def test_monotone_invariance():
    v = np.array([3., 1., 4., 1.5, 5.])
    assert symbol_of(v, 5) == symbol_of(np.log1p(v * 7.3), 5) == symbol_of(v * 0.01 + 2, 5)


def test_agreement_blind_conditional_entropy_sees_relabel_redundancy():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 13, 5000)
    b = (a * 2 + 5) % 40                       # deterministic relabel, different alphabet
    r = stream_redundancy(a, b)
    assert r["agreement"] < 0.2                # the broken diagnostic reads "independent"
    assert r["min_norm_cond_entropy"] < 1e-9   # the fixed one reads "redundant"


def test_nested_L_streams_are_deterministically_redundant():
    rng = np.random.default_rng(1)
    vals = rng.normal(size=400)
    pts = frame_grid(400, stride=2)
    s3, a3 = contour_stream(vals, pts, L=3)
    s2, a2 = contour_stream(vals, pts, L=2)
    m3 = dict(zip(a3.tolist(), s3.tolist())); m2 = dict(zip(a2.tolist(), s2.tolist()))
    common = sorted(set(a3.tolist()) & set(a2.tolist()))
    x = np.array([m3[f] for f in common]); y = np.array([m2[f] for f in common])
    assert conditional_entropy(y, x) < 1e-9


def test_expand_holds_forward_and_marks_prefix_undefined():
    rng = np.random.default_rng(2)
    vals = rng.normal(size=100)
    s, a = contour_stream(vals, frame_grid(100, stride=5), L=3)
    fr = expand_to_frames(s, a, 100)
    assert (fr[:int(a[0])] == UNDEFINED).all()
    assert fr[-1] == s[-1]


def test_event_grid_dedups_and_frame_rate_constant():
    assert abs(FRAME_RATE_HZ - 10.7666015625) < 1e-9
    g = event_grid(np.array([0.5, 0.5001, 1.0, 99.0]), n_frames=50)
    assert g.tolist() == sorted(set(g.tolist()))
    assert g.max() < 50
