# plots/latent_analysis/02_pca_analysis.py
"""
Script 02 — PCA analysis.
Loads all feature vectors + latent means, fits PCA on each,
computes cross-PCA correlation matrix and saves per-crop scores.

Usage:
    python 02_pca_analysis.py [--force] [--n-components 20]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plots.latent_analysis.config import DATA_DIR, FEATURE_GROUPS, LATENT_DIM
from plots.latent_analysis.features import (
    iter_paired_crops, drop_low_variance, apply_clr
)
from plots.latent_analysis.findings import update_findings_section, update_progress

OUT_NPZ = DATA_DIR / "02_pca.npz"
ALL_FEATURE_NAMES = [f for group in FEATURE_GROUPS.values() for f in group]


def run(force: bool = False, n_components: int = 20):
    if OUT_NPZ.exists() and not force:
        print("02_pca.npz already exists. Use --force to recompute.")
        return

    print("Script 02: loading crop feature vectors...")
    rows_feat    = []
    rows_latent  = []
    crop_ids     = []
    n_total = 0

    for i, (latent_mean, info) in enumerate(iter_paired_crops(ALL_FEATURE_NAMES)):
        n_total += 1
        # Build feature vector — use CLR-transformed HPCP for PCA
        # Use NaN for missing/invalid values; we'll filter columns+rows after.
        row = []
        for feat in ALL_FEATURE_NAMES:
            if feat.startswith("hpcp_") and not feat.startswith("hpcp_clr_"):
                clr_key = feat.replace("hpcp_", "hpcp_clr_")
                val = info.get(clr_key)
            else:
                val = info.get(feat)
            if val is None or not isinstance(val, (int, float)) or not np.isfinite(val):
                row.append(np.nan)
            else:
                row.append(float(val))
        rows_feat.append(row)
        rows_latent.append(latent_mean)
        crop_ids.append(i)
        if n_total % 20000 == 0:
            print(f"  {n_total} crops loaded...")

    print(f"Total crops loaded: {n_total}")

    X_feat   = np.array(rows_feat,   dtype=np.float32)
    X_latent = np.array(rows_latent, dtype=np.float32)

    # Drop features missing in >10% of crops (e.g. features absent from this dataset)
    missing_frac = np.isnan(X_feat).mean(axis=0)
    keep_cols = missing_frac <= 0.10
    dropped_missing = [f for f, k in zip(ALL_FEATURE_NAMES, keep_cols) if not k]
    if dropped_missing:
        print(f"Dropped (>10% missing): {dropped_missing}")
    X_feat = X_feat[:, keep_cols]
    feat_names_filtered = [f for f, k in zip(ALL_FEATURE_NAMES, keep_cols) if k]

    # Drop crops with any remaining NaN
    complete_rows = ~np.isnan(X_feat).any(axis=1)
    X_feat   = X_feat[complete_rows]
    X_latent = X_latent[complete_rows]
    crop_ids = [crop_ids[i] for i, ok in enumerate(complete_rows) if ok]
    print(f"Complete crops: {len(X_feat)} | Feature dims: {X_feat.shape[1]}")

    # Drop near-zero-variance features before PCA
    X_feat, feat_names_used, dropped = drop_low_variance(
        X_feat, feat_names_filtered, threshold=1e-4
    )
    if dropped:
        print(f"Dropped (low variance): {dropped}")

    # Feature PCA
    scaler_feat  = StandardScaler()
    X_feat_std   = scaler_feat.fit_transform(X_feat)
    n_comp_feat  = min(n_components, X_feat_std.shape[1])
    pca_feat     = PCA(n_components=n_comp_feat, random_state=42)
    feat_scores  = pca_feat.fit_transform(X_feat_std)   # [N, n_comp]

    # Latent PCA (no standardisation — already normalised)
    n_comp_lat   = min(n_components, LATENT_DIM)
    pca_latent   = PCA(n_components=n_comp_lat, random_state=42)
    latent_scores = pca_latent.fit_transform(X_latent)  # [N, n_comp]

    # Cross-PCA correlation matrix [n_comp_feat, n_comp_lat]
    from scipy.stats import pearsonr
    cross_corr = np.zeros((n_comp_feat, n_comp_lat))
    for i in range(n_comp_feat):
        for j in range(n_comp_lat):
            r, _ = pearsonr(feat_scores[:, i], latent_scores[:, j])
            cross_corr[i, j] = r

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        feat_explained_variance_ratio=pca_feat.explained_variance_ratio_,
        feat_components=pca_feat.components_,
        feat_scores=feat_scores,
        latent_explained_variance_ratio=pca_latent.explained_variance_ratio_,
        latent_components=pca_latent.components_,
        latent_scores=latent_scores,
        cross_corr=cross_corr,
        feat_names_used=np.array(feat_names_used),
        crop_ids=np.array(crop_ids),
        scaler_mean=scaler_feat.mean_,
        scaler_std=scaler_feat.scale_,
    )
    print(f"Saved {OUT_NPZ}")

    # FINDINGS
    ev_feat   = pca_feat.explained_variance_ratio_
    ev_latent = pca_latent.explained_variance_ratio_
    body = _findings_body(
        len(X_feat), feat_names_used, ev_feat, ev_latent,
        pca_feat.components_, cross_corr
    )
    update_findings_section("02", body)
    update_progress("02", f"Done. {len(X_feat)} crops, feat PC1={ev_feat[0]:.1%}, lat PC1={ev_latent[0]:.1%}")


def _findings_body(n, feat_names, ev_feat, ev_latent, feat_components, cross_corr):
    lines = [
        "## Script 02 — PCA",
        f"- Crops with complete features: {n:,}",
        "",
        "### Feature PCA explained variance",
    ]
    cumev = 0
    for i, ev in enumerate(ev_feat[:10]):
        cumev += ev
        top_loadings = np.argsort(np.abs(feat_components[i]))[::-1][:3]
        top_names    = [feat_names[j] for j in top_loadings]
        lines.append(f"- PC{i+1}: {ev:.1%} (cumulative {cumev:.1%}) — top: {', '.join(top_names)}")
    lines += [
        "",
        "### Latent PCA explained variance (top 5)",
    ]
    for i, ev in enumerate(ev_latent[:5]):
        lines.append(f"- Latent PC{i+1}: {ev:.1%}")
    lines += [
        "",
        "### Strongest cross-PCA alignments",
    ]
    flat = np.abs(cross_corr).ravel()
    top  = np.argsort(flat)[::-1][:5]
    for idx in top:
        fi = idx // cross_corr.shape[1]
        li = idx % cross_corr.shape[1]
        lines.append(f"- Feature PC{fi+1} ↔ Latent PC{li+1}: r = {cross_corr[fi,li]:+.3f}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-components", type=int, default=20)
    args = parser.parse_args()
    run(force=args.force, n_components=args.n_components)
