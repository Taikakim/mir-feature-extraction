# plots/latent_analysis/_corr_utils.py
"""Shared correlation and FDR utilities."""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.stats import false_discovery_control


def compute_pearson_spearman(X: np.ndarray, y: np.ndarray):
    """
    For each column of X, compute Pearson r and Spearman rho against y.
    Returns (r_pearson, p_pearson, r_spearman, p_spearman), each shape [n_cols].
    """
    n_cols = X.shape[1]
    r_p = np.zeros(n_cols)
    p_p = np.ones(n_cols)
    r_s = np.zeros(n_cols)
    p_s = np.ones(n_cols)
    for c in range(n_cols):
        try:
            r_p[c], p_p[c] = pearsonr(X[:, c], y)
            r_s[c], p_s[c] = spearmanr(X[:, c], y)
        except Exception:
            pass
    r_p = np.nan_to_num(r_p)
    r_s = np.nan_to_num(r_s)
    return r_p, p_p, r_s, p_s


def apply_bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction to a 2D array of p-values.
    pvals shape: [n_dims, n_features]. Returns adjusted p-values, same shape.
    """
    flat = pvals.ravel()
    adj_flat = false_discovery_control(np.clip(flat, 0, 1), method='bh')
    return adj_flat.reshape(pvals.shape)
