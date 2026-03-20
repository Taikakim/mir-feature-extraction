#!/usr/bin/env python3
# plots/latent_analysis/01_aggregate_correlation.py
"""
Script 01 — Aggregate correlation analysis.
Loads all 192K crops (streaming), computes Pearson + Spearman for each
latent dim × feature pair, applies BH FDR, saves NPZ + 8×8 posters.

Usage:
    python 01_aggregate_correlation.py [--force]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plots.latent_analysis.config import (
    DATA_DIR, POSTER_DIR, FEATURE_GROUPS, LATENT_DIM,
    POSTER_CLAMP, EFFECT_WEAK, EFFECT_STRONG, RANDOM_SEED,
)
from plots.latent_analysis.features import iter_paired_crops, drop_low_variance
from plots.latent_analysis.findings import init_findings, update_findings_section, update_progress
from plots.latent_analysis._corr_utils import compute_pearson_spearman, apply_bh_fdr

OUT_NPZ     = DATA_DIR / "01_correlations.npz"
SCATTER_NPZ = DATA_DIR / "scatter_sample.npz"
ALL_FEATURE_NAMES = [f for group in FEATURE_GROUPS.values() for f in group]


def run(force: bool = False):
    if OUT_NPZ.exists() and not force:
        print(f"01_correlations.npz already exists. Use --force to recompute.")
        return

    print("Script 01: loading crops...")
    # Accumulate per-feature lists
    latent_accumulator = defaultdict(list)  # feat_name → list of latent_mean[64]
    value_accumulator  = defaultdict(list)  # feat_name → list of scalar value
    all_latent_means   = []

    all_info_rows = []   # parallel to all_latent_means, for scatter sample
    n_total = 0
    for latent_mean, info in iter_paired_crops(ALL_FEATURE_NAMES):
        n_total += 1
        if n_total % 10000 == 0:
            print(f"  {n_total} crops loaded...")
        all_latent_means.append(latent_mean)
        all_info_rows.append(info)   # keep encoded info dict for scatter
        for feat in ALL_FEATURE_NAMES:
            val = info.get(feat)
            if val is not None and isinstance(val, (int, float)) and np.isfinite(val):
                latent_accumulator[feat].append(latent_mean)
                value_accumulator[feat].append(float(val))

    print(f"Total crops loaded: {n_total}")

    # Drop near-zero-variance features
    # Build a feature matrix from crops that have ALL features (for variance check)
    # Use a simplified approach: check std of each feature's values
    dropped = []
    valid_features = []
    for feat in ALL_FEATURE_NAMES:
        vals = np.array(value_accumulator[feat])
        if len(vals) < 10 or vals.std() < 1e-4:
            dropped.append(feat)
        else:
            valid_features.append(feat)

    if dropped:
        print(f"Dropped low-variance features: {dropped}")

    feature_names = valid_features
    n_features    = len(feature_names)
    print(f"Analysing {n_features} features × {LATENT_DIM} dims...")

    # Correlation matrices: [64, n_features]
    r_pearson  = np.zeros((LATENT_DIM, n_features))
    p_pearson  = np.ones((LATENT_DIM, n_features))
    r_spearman = np.zeros((LATENT_DIM, n_features))
    p_spearman = np.ones((LATENT_DIM, n_features))
    n_per_feat = np.zeros(n_features, dtype=int)

    for fi, feat in enumerate(feature_names):
        X = np.array(latent_accumulator[feat])   # [N_feat, 64]
        y = np.array(value_accumulator[feat])     # [N_feat]
        n_per_feat[fi] = len(y)
        rp, pp, rs, ps = compute_pearson_spearman(X, y)
        r_pearson[:,  fi] = rp
        p_pearson[:,  fi] = pp
        r_spearman[:, fi] = rs
        p_spearman[:, fi] = ps
        if fi % 10 == 0:
            print(f"  feature {fi+1}/{n_features}: {feat} (N={len(y)})")

    # BH FDR correction
    p_pearson_adj  = apply_bh_fdr(p_pearson)
    p_spearman_adj = apply_bh_fdr(p_spearman)

    # Save main NPZ
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        r_pearson=r_pearson,
        p_pearson_adj=p_pearson_adj,
        r_spearman=r_spearman,
        p_spearman_adj=p_spearman_adj,
        feature_names=np.array(feature_names),
        n_per_feature=n_per_feat,
    )
    print(f"Saved {OUT_NPZ}")

    # Scatter sample: 5000 crops that have ALL valid features
    complete_indices = [
        i for i, row in enumerate(all_info_rows)
        if all(isinstance(row.get(f), (int, float)) and np.isfinite(float(row.get(f, float('nan'))))
               for f in feature_names)
    ]
    rng = np.random.default_rng(RANDOM_SEED)
    idx5k = rng.choice(complete_indices, size=min(5000, len(complete_indices)), replace=False)
    scatter_latents = np.array([all_latent_means[i] for i in idx5k])
    scatter_feats   = np.array([[float(all_info_rows[i][f]) for f in feature_names]
                                 for i in idx5k], dtype=np.float32)
    np.savez_compressed(SCATTER_NPZ,
                        latent_means=scatter_latents,
                        feature_values=scatter_feats,
                        feature_names=np.array(feature_names))
    print(f"Saved {SCATTER_NPZ}")

    # Generate 8×8 posters
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    _generate_posters(r_pearson, feature_names, n_per_feat)

    # Update FINDINGS.md
    top_pairs = _top_corr_pairs(r_pearson, feature_names, n=5)
    init_findings(n_total, n_features, dropped)
    body = _findings_body(n_total, n_features, dropped, feature_names, top_pairs)
    update_findings_section("01", body)
    update_progress("01", f"Done. {n_total} crops, {n_features} features.")


def _generate_posters(r_matrix: np.ndarray, feature_names: list, n_per_feat: np.ndarray):
    """Save one 8×8 heatmap PNG per feature into POSTER_DIR."""
    for fi, feat in enumerate(feature_names):
        r_col = r_matrix[:, fi]           # [64]
        grid  = r_col.reshape(8, 8)       # 8×8 spatial layout

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-POSTER_CLAMP, vmax=POSTER_CLAMP,
                       aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

        # Annotate each cell with r value
        for row in range(8):
            for col in range(8):
                dim_idx = row * 8 + col
                val = grid[row, col]
                ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(val) < 0.25 else "white")

        ax.set_title(f"{feat}\n(N={n_per_feat[fi]:,})", fontsize=10)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels([str(i) for i in range(8)], fontsize=7)
        ax.set_yticklabels([str(i*8) for i in range(8)], fontsize=7)
        ax.set_xlabel("Dim mod 8")
        ax.set_ylabel("Dim // 8 × 8")

        fig.tight_layout()
        fig.savefig(POSTER_DIR / f"{feat}_poster.png", dpi=100)
        plt.close(fig)

    print(f"Saved {len(feature_names)} posters to {POSTER_DIR}")


def _top_corr_pairs(r_matrix, feature_names, n=5):
    """Return top-N (|r|, dim, feat) tuples."""
    flat_abs = np.abs(r_matrix).ravel()
    top_idx  = np.argsort(flat_abs)[::-1][:n]
    pairs = []
    for idx in top_idx:
        dim = idx // len(feature_names)
        fi  = idx % len(feature_names)
        pairs.append((r_matrix[dim, fi], dim, feature_names[fi]))
    return pairs


def _findings_body(n_crops, n_features, dropped, feature_names, top_pairs):
    lines = [
        f"## Script 01 — Aggregate Correlations",
        f"- Crops analysed: {n_crops:,}",
        f"- Features: {n_features} (dropped: {', '.join(dropped) or 'none'})",
        f"",
        f"### Top correlating (dim, feature) pairs by |Pearson r|",
    ]
    for r, dim, feat in top_pairs:
        lines.append(f"- Dim {dim:2d} × `{feat}`: r = {r:+.3f}")
    lines += [
        "",
        "> ⚠️ BPM correlations may be deflated due to low within-genre variance (~135–145 BPM).",
        "> ⚠️ `atonality` likely reflects noise/percussion energy in this corpus, not harmonic atonality.",
        "> ⚠️ HPCP correlations are relative (compositional constraint); interpret with CLR in PCA.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(force=args.force)
