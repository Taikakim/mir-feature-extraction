# plots/latent_analysis/03_latent_xcorr.py
"""
Script 03 — Latent temporal cross-correlation.
Subsamples 2000 crops, computes 64×64 Pearson xcorr per crop (Fisher-Z averaged),
then Ward-links dims into clusters.

Usage:
    python 03_latent_xcorr.py [--force] [--n-crops 2000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plots.latent_analysis.config import (
    DATA_DIR, LATENT_DIR, LATENT_DIM, LATENT_FRAMES,
    N_TEMPORAL_CROPS, RANDOM_SEED,
)
from plots.latent_analysis._corr_utils import fisher_z_average
from plots.latent_analysis.findings import update_findings_section, update_progress

OUT_NPZ = DATA_DIR / "03_xcorr.npz"
STEM_SUFFIXES = {"_bass", "_drums", "_other", "_vocals"}


def _collect_latent_paths(n_crops: int, seed: int):
    """Collect up to n_crops full-mix latent paths, randomly sampled."""
    rng  = np.random.default_rng(seed)
    all_paths = []
    for track_dir in LATENT_DIR.iterdir():
        if not track_dir.is_dir():
            continue
        for p in track_dir.glob("*.npy"):
            if not any(p.stem.endswith(s) for s in STEM_SUFFIXES):
                all_paths.append(p)
    rng.shuffle(all_paths)
    return all_paths[:n_crops]


def run(force: bool = False, n_crops: int = N_TEMPORAL_CROPS):
    if OUT_NPZ.exists() and not force:
        print("03_xcorr.npz already exists. Use --force to recompute.")
        return

    if not LATENT_DIR.exists():
        raise RuntimeError(f"Latent dir not mounted: {LATENT_DIR}")

    print(f"Script 03: collecting {n_crops} latent paths...")
    paths = _collect_latent_paths(n_crops, RANDOM_SEED)
    print(f"  found {len(paths)} crops")

    xcorr_stack = []
    skipped = 0
    for i, p in enumerate(paths):
        try:
            lat = np.load(str(p)).astype(np.float32)  # [64, 256]
            assert lat.shape == (LATENT_DIM, LATENT_FRAMES)
            # Demean each dim
            lat = lat - lat.mean(axis=1, keepdims=True)
            # 64×64 Pearson correlation across 256 time steps
            corr = np.corrcoef(lat)          # [64, 64]
            xcorr_stack.append(corr)
        except Exception:
            skipped += 1
            continue
        if i % 500 == 0:
            print(f"  {i+1}/{len(paths)}...")

    print(f"Computed {len(xcorr_stack)} xcorr matrices ({skipped} skipped).")

    mean_xcorr = fisher_z_average(np.array(xcorr_stack))   # [64, 64]

    # Ward clustering
    # Convert correlation to distance: d = 1 - r (bounded [0,2])
    dist_matrix = 1.0 - mean_xcorr
    np.fill_diagonal(dist_matrix, 0)
    dist_condensed = dist_matrix[np.triu_indices(LATENT_DIM, k=1)]
    Z = linkage(dist_condensed, method="ward")

    # Auto-determine number of clusters (cut at 70% of max merge distance)
    max_dist = Z[-1, 2]
    labels   = fcluster(Z, 0.7 * max_dist, criterion="distance")
    n_clusters = labels.max()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        xcorr_matrix=mean_xcorr,
        linkage_matrix=Z,
        cluster_labels=labels,
        n_crops_used=len(xcorr_stack),
    )
    print(f"Saved {OUT_NPZ} — {n_clusters} clusters found.")

    body = _findings_body(len(xcorr_stack), n_clusters, labels, mean_xcorr)
    update_findings_section("03", body)
    update_progress("03", f"Done. {len(xcorr_stack)} crops, {n_clusters} clusters.")


def _findings_body(n_crops, n_clusters, labels, xcorr):
    lines = [
        "## Script 03 — Latent Cross-Correlation",
        f"- Crops used: {n_crops} (Fisher-Z averaged)",
        f"- Clusters found (Ward, 70% cut): **{n_clusters}**",
        "",
        "### Cluster membership",
    ]
    for c in range(1, n_clusters + 1):
        dims = np.where(labels == c)[0]
        lines.append(f"- Cluster {c}: dims {', '.join(map(str, dims))}")
    # Strongest inter-dim correlations
    lines += ["", "### Strongest inter-dim temporal correlations"]
    flat = xcorr.copy()
    np.fill_diagonal(flat, 0)
    top = np.argsort(np.abs(flat).ravel())[::-1][:6]
    seen = set()
    for idx in top:
        d1, d2 = divmod(idx, 64)
        if (min(d1,d2), max(d1,d2)) in seen:
            continue
        seen.add((min(d1,d2), max(d1,d2)))
        lines.append(f"- Dim {d1} ↔ Dim {d2}: r = {xcorr[d1,d2]:+.3f}")
    lines += [
        "",
        "> Note: within-track crops treated as independent; effective N ≈ number of tracks sampled (~1600).",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-crops", type=int, default=N_TEMPORAL_CROPS)
    args = parser.parse_args()
    run(force=args.force, n_crops=args.n_crops)
