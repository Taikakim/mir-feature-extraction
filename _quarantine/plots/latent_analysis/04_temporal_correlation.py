# plots/latent_analysis/04_temporal_correlation.py
"""
Script 04 — Temporal correlation between latent dims and frame-level features.
Reads pre-computed timeseries from TimeseriesDB (data/timeseries.db) — no audio I/O.
Uses N_TEMPORAL_CROPS subsample; correlations are computed per-feature using only
crops that have that feature (handles partial DB coverage gracefully).

Usage:
    python 04_temporal_correlation.py [--force] [--n-crops N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plots.latent_analysis.config import (
    DATA_DIR, LATENT_DIR, LATENT_DIM, LATENT_FRAMES,
    N_TEMPORAL_CROPS, RANDOM_SEED,
    TEMPORAL_FEATURE_NAMES,
)
from plots.latent_analysis._corr_utils import compute_pearson_spearman
from plots.latent_analysis.findings import update_findings_section, update_progress
from plots.latent_analysis._03_collect import collect_latent_paths
from src.core.timeseries_db import TimeseriesDB

OUT_NPZ = DATA_DIR / "04_temporal.npz"
MIN_CROPS_PER_FEAT = 100   # skip feature if fewer crops have it


def _extract_ts(arrays: dict) -> dict:
    """Expand DB arrays into flat per-name (256,) arrays.
    hpcp_ts (256, 12) → hpcp_ts_0 .. hpcp_ts_11.
    """
    out = {}
    for name in TEMPORAL_FEATURE_NAMES:
        if name.startswith("hpcp_ts_"):
            idx = int(name.split("_")[-1])
            hpcp = arrays.get("hpcp_ts")
            if hpcp is not None and hpcp.shape == (LATENT_FRAMES, 12):
                out[name] = hpcp[:, idx]
        else:
            arr = arrays.get(name)
            if arr is not None and arr.shape == (LATENT_FRAMES,):
                out[name] = arr
    return out


def run(force: bool = False, n_crops: int = N_TEMPORAL_CROPS):
    if OUT_NPZ.exists() and not force:
        print("04_temporal.npz already exists. Use --force to recompute.")
        return

    if not LATENT_DIR.exists():
        raise RuntimeError(f"Latent dir not mounted: {LATENT_DIR}")

    db = TimeseriesDB.open()
    print(f"TimeseriesDB: {db.count():,} entries")

    print(f"Script 04: collecting {n_crops} latent paths...")
    paths = collect_latent_paths(n_crops, RANDOM_SEED)
    print(f"  {len(paths)} candidate crops")

    # --- Pass 1: load all crops into memory ---
    # crops: list of (lat[64,256], ts_dict{name: arr[256]})
    crops = []
    skipped = 0

    for i, npy_path in enumerate(paths):
        arrays = db.get(npy_path.stem)
        if arrays is None:
            skipped += 1
            continue
        ts = _extract_ts(arrays)
        if not ts:
            skipped += 1
            continue
        try:
            lat = np.load(str(npy_path), mmap_mode="r").astype(np.float32)
            if lat.shape != (LATENT_DIM, LATENT_FRAMES) or not np.all(np.isfinite(lat)):
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue
        crops.append((lat, ts))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(paths)}, {len(crops)} loaded...")

    print(f"Loaded: {len(crops)} crops ({skipped} skipped — no DB entry or bad latent)")

    # --- Pass 2: per-feature temporal correlation ---
    n_tfeats = len(TEMPORAL_FEATURE_NAMES)
    r_temp      = np.zeros((LATENT_DIM, n_tfeats))
    p_temp      = np.ones( (LATENT_DIM, n_tfeats))
    n_per_feat  = np.zeros(n_tfeats, dtype=int)

    sample_crops     = []
    sample_feat_segs = []

    for fi, fname in enumerate(TEMPORAL_FEATURE_NAMES):
        subset = [(lat, ts[fname]) for lat, ts in crops if fname in ts]
        n_per_feat[fi] = len(subset)
        if len(subset) < MIN_CROPS_PER_FEAT:
            print(f"  [{fi+1}/{n_tfeats}] {fname}: only {len(subset)} crops — skipped")
            continue

        lat_cat  = np.concatenate([lat for lat, _ in subset], axis=1)  # [64, N*256]
        feat_cat = np.concatenate([f   for _, f   in subset])           # [N*256]

        rp, pp, _, _ = compute_pearson_spearman(lat_cat.T, feat_cat)
        r_temp[:, fi] = rp
        p_temp[:, fi] = pp

        if fi % 5 == 0:
            print(f"  [{fi+1}/{n_tfeats}] {fname}: N={len(subset)}, max|r|={np.abs(rp).max():.3f}")

    # Build sample segments from first 50 crops (universal features only)
    universal = [n for n in TEMPORAL_FEATURE_NAMES if not n.startswith("hpcp_ts_")]
    for lat, ts in crops[:50]:
        if all(n in ts for n in universal):
            sample_crops.append(lat)
            sample_feat_segs.append(np.array([ts.get(n, np.zeros(LATENT_FRAMES))
                                               for n in TEMPORAL_FEATURE_NAMES]))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        r_temporal=r_temp,
        p_temporal=p_temp,
        temporal_feature_names=np.array(TEMPORAL_FEATURE_NAMES),
        n_crops_used=len(crops),
        n_per_feature=n_per_feat,
        sample_crops=np.array(sample_crops) if sample_crops else np.empty((0,)),
        sample_feat_segs=np.array(sample_feat_segs) if sample_feat_segs else np.empty((0,)),
    )
    print(f"Saved {OUT_NPZ}")

    body = _findings_body(len(crops), r_temp, n_per_feat)
    update_findings_section("04", body)
    update_progress("04", f"Done. {len(crops)} crops, temporal corr computed.")


def _findings_body(n_crops, r_temp, n_per_feat):
    lines = [
        "## Script 04 — Temporal Correlation",
        f"- Crops used: {n_crops:,}",
        "",
        "### Strongest temporal (dim × frame-feature) correlations",
    ]
    flat     = r_temp.ravel()
    top_idx  = np.argsort(np.abs(flat))[::-1][:8]
    for idx in top_idx:
        dim = idx // len(TEMPORAL_FEATURE_NAMES)
        fi  = idx % len(TEMPORAL_FEATURE_NAMES)
        n   = n_per_feat[fi]
        lines.append(
            f"- Dim {dim:2d} × `{TEMPORAL_FEATURE_NAMES[fi]}`: r = {r_temp[dim,fi]:+.3f} (N={n:,})"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-crops", type=int, default=N_TEMPORAL_CROPS)
    args = parser.parse_args()
    run(force=args.force, n_crops=args.n_crops)
