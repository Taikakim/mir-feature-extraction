# tests/test_latent_analysis.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from plots.latent_analysis.features import (
    encode_tonic_circular,
    encode_tonic_scale,
    normalise_bpm_madmom,
    apply_clr,
    drop_low_variance,
)


def test_tonic_circular_encoding_values():
    """sin/cos encoding gives correct values."""
    sin_val, cos_val = encode_tonic_circular(0)
    assert abs(sin_val - np.sin(0)) < 1e-6
    assert abs(cos_val - np.cos(0)) < 1e-6

    sin_val, cos_val = encode_tonic_circular(6)
    assert abs(sin_val - np.sin(2 * np.pi * 6 / 12)) < 1e-6


def test_tonic_circular_distance_wraps():
    """Distance from tonic 11 to 0 should equal distance from 0 to 1."""
    def dist(a, b):
        sa, ca = encode_tonic_circular(a)
        sb, cb = encode_tonic_circular(b)
        return np.sqrt((sa - sb) ** 2 + (ca - cb) ** 2)

    assert abs(dist(11, 0) - dist(0, 1)) < 1e-4


def test_tonic_scale_binary():
    assert encode_tonic_scale("minor") == 1
    assert encode_tonic_scale("major") == 0
    assert encode_tonic_scale(None) is None  # missing = skip


def test_bpm_madmom_normalisation():
    """Half-tempo madmom (69) aligned to essentia (138) → ~138."""
    result = normalise_bpm_madmom(bpm_madmom=69.0, bpm_essentia=138.0)
    assert abs(result - 138.0) < 1.0

    # Already matching — no change
    result2 = normalise_bpm_madmom(bpm_madmom=140.0, bpm_essentia=138.0)
    assert abs(result2 - 140.0) < 1.0


def test_clr_sums_to_zero():
    """CLR output should sum to zero."""
    hpcp = np.array([0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    hpcp = hpcp / hpcp.sum()
    clr = apply_clr(hpcp)
    assert abs(clr.sum()) < 1e-5


def test_clr_shape():
    hpcp = np.ones(12) / 12.0
    clr = apply_clr(hpcp)
    assert clr.shape == (12,)


def test_drop_low_variance():
    """Features with std < threshold should be removed."""
    X = np.column_stack([
        np.random.randn(100),       # high variance
        np.zeros(100),              # zero variance → drop
        np.ones(100) * 5 + np.random.randn(100) * 1e-6,  # near-zero → drop
    ])
    names = ["good", "zeros", "flat"]
    X_out, names_out, dropped = drop_low_variance(X, names, threshold=1e-4)
    assert "good" in names_out
    assert "zeros" not in names_out
    assert "flat" not in names_out
    assert set(dropped) == {"zeros", "flat"}


from plots.latent_analysis.findings import overwrite_findings_section


def test_findings_section_overwrite_replaces_content():
    content = (
        "# Header\n\n"
        "<!-- script-01-start -->\n"
        "## Old section\nOld content\n"
        "<!-- script-01-end -->\n\n"
        "## Footer stays\n"
    )
    result = overwrite_findings_section(content, "01", "## New section\nNew content\n")
    assert "Old content" not in result
    assert "New content" in result
    assert "## Footer stays" in result
    assert "<!-- script-01-start -->" in result
    assert "<!-- script-01-end -->" in result


def test_findings_section_overwrite_idempotent():
    """Running overwrite twice gives same result."""
    content = (
        "<!-- script-02-start -->\nOriginal\n<!-- script-02-end -->\n"
    )
    once  = overwrite_findings_section(content, "02", "Updated\n")
    twice = overwrite_findings_section(once,    "02", "Updated\n")
    assert once == twice


def test_findings_section_overwrite_missing_markers():
    """If markers are absent, raises ValueError rather than silently corrupting."""
    content = "# No markers here\n"
    with pytest.raises(ValueError, match="markers not found"):
        overwrite_findings_section(content, "01", "New content\n")


from plots.latent_analysis._corr_utils import (
    compute_pearson_spearman,
    apply_bh_fdr,
)


def test_pearson_spearman_shapes():
    """Returns r and p arrays of length N_dims."""
    X = np.random.randn(200, 64)   # 200 crops, 64 dims
    y = np.random.randn(200)
    r_p, p_p, r_s, p_s = compute_pearson_spearman(X, y)
    assert r_p.shape == (64,)
    assert r_s.shape == (64,)


def test_bh_fdr_output_shape():
    pvals = np.random.uniform(0, 1, (64, 20))  # 64 dims × 20 features
    adj = apply_bh_fdr(pvals)
    assert adj.shape == pvals.shape
    # Adjusted p-values must be >= originals (BH never makes them smaller)
    assert np.all(adj >= pvals - 1e-10)


from plots.latent_analysis._corr_utils import fisher_z_average


def test_fisher_z_average_preserves_sign():
    matrices = np.array([
        [[0.8, -0.3], [-0.3, 0.8]],
        [[0.7, -0.2], [-0.2, 0.7]],
    ])
    result = fisher_z_average(matrices)
    assert result[0, 0] > 0
    assert result[0, 1] < 0


def test_fisher_z_average_symmetric():
    """Output should be symmetric if all input matrices are symmetric."""
    # Create symmetric correlation matrices
    matrices = []
    for _ in range(10):
        A = np.random.uniform(-0.5, 0.5, (8, 8))
        sym = (A + A.T) / 2  # Force symmetry
        matrices.append(sym)
    matrices = np.array(matrices)
    result = fisher_z_average(matrices)
    np.testing.assert_allclose(result, result.T, atol=1e-6)


def test_fisher_z_handles_diagonal_ones():
    """Diagonal r=1 should not produce inf after ε clamping."""
    matrices = np.eye(4)[np.newaxis].repeat(5, axis=0)
    result = fisher_z_average(matrices)
    assert np.all(np.isfinite(result))
