# plots/latent_analysis/04_temporal_correlation.py
"""
Script 04 — Temporal correlation between latent dims and frame-level features.
Uses same 2000-crop subsample as script 03 (same seed).

Usage:
    python 04_temporal_correlation.py [--force]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plots.latent_analysis.config import (
    DATA_DIR, LATENT_DIR, INFO_DIR, LATENT_DIM, LATENT_FRAMES,
    SAMPLE_RATE, HOP_LENGTH, N_TEMPORAL_CROPS, RANDOM_SEED,
    TEMPORAL_FEATURE_NAMES,
)
from plots.latent_analysis._temporal_features import compute_frame_features
from plots.latent_analysis._corr_utils import compute_pearson_spearman
from plots.latent_analysis.findings import update_findings_section, update_progress
from plots.latent_analysis._03_collect import collect_latent_paths

OUT_NPZ = DATA_DIR / "04_temporal.npz"
EXPECTED_AUDIO_LEN = LATENT_FRAMES * HOP_LENGTH  # 524288 samples


def _find_audio(npy_path: Path) -> Path:
    """Find the raw audio crop (.flac/.wav) matching a latent NPY path."""
    track_name = npy_path.parent.name
    stem       = npy_path.stem
    audio_dir  = INFO_DIR / track_name
    for ext in [".flac", ".wav", ".mp3"]:
        candidate = audio_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def run(force: bool = False):
    if OUT_NPZ.exists() and not force:
        print("04_temporal.npz already exists. Use --force to recompute.")
        return

    if not LATENT_DIR.exists():
        raise RuntimeError(f"Latent dir not mounted: {LATENT_DIR}")

    print(f"Script 04: collecting {N_TEMPORAL_CROPS} latent paths (same seed as script 03)...")
    paths = collect_latent_paths(N_TEMPORAL_CROPS, RANDOM_SEED)

    lat_segs  = []
    feat_segs = []
    sample_crops     = []
    sample_feat_segs = []
    skipped = 0

    for i, npy_path in enumerate(paths):
        audio_path = _find_audio(npy_path)
        if audio_path is None:
            skipped += 1
            continue
        try:
            lat = np.load(str(npy_path)).astype(np.float32)
            assert lat.shape == (LATENT_DIM, LATENT_FRAMES)
            if not np.all(np.isfinite(lat)):
                skipped += 1
                continue

            audio, sr = sf.read(str(audio_path), always_2d=True)
            audio = audio.mean(axis=1).astype(np.float32)  # mono
            if sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            if len(audio) < int(0.8 * EXPECTED_AUDIO_LEN):
                skipped += 1
                continue
            if len(audio) < EXPECTED_AUDIO_LEN:
                # Pad short crops with zeros (crop boundary rounding)
                pad = np.zeros(EXPECTED_AUDIO_LEN - len(audio), dtype=np.float32)
                audio = np.concatenate([audio, pad])
            audio = audio[:EXPECTED_AUDIO_LEN]

            feats = compute_frame_features(audio, sr=SAMPLE_RATE)  # [N_tfeats, 256]
            lat_segs.append(lat)
            feat_segs.append(feats)
            if len(sample_crops) < 50:
                sample_crops.append(lat)
                sample_feat_segs.append(feats)

        except Exception:
            skipped += 1
            continue

        if i % 200 == 0:
            print(f"  {i+1}/{len(paths)}, {len(lat_segs)} valid...")

    print(f"Valid crops: {len(lat_segs)} ({skipped} skipped — audio not found or wrong length)")

    # Concatenate all (crop, t) observations
    lat_all  = np.concatenate(lat_segs,  axis=1)   # [64, N*256]
    feat_all = np.concatenate(feat_segs, axis=1)    # [N_tfeats, N*256]

    # Temporal correlation: each (latent_dim, temporal_feat) pair
    n_tfeats = len(TEMPORAL_FEATURE_NAMES)
    r_temp   = np.zeros((LATENT_DIM, n_tfeats))
    p_temp   = np.ones( (LATENT_DIM, n_tfeats))

    for fi, fname in enumerate(TEMPORAL_FEATURE_NAMES):
        y = feat_all[fi]
        rp, pp, _, _ = compute_pearson_spearman(lat_all.T, y)
        r_temp[:, fi] = rp
        p_temp[:, fi] = pp
        if fi % 5 == 0:
            print(f"  temporal feat {fi+1}/{n_tfeats}: {fname}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        r_temporal=r_temp,
        temporal_feature_names=np.array(TEMPORAL_FEATURE_NAMES),
        n_crops_used=len(lat_segs),
        sample_crops=np.array(sample_crops),
        sample_feat_segs=np.array(sample_feat_segs),
    )
    print(f"Saved {OUT_NPZ}")

    body = _findings_body(len(lat_segs), r_temp)
    update_findings_section("04", body)
    update_progress("04", f"Done. {len(lat_segs)} crops, temporal corr computed.")


def _findings_body(n_crops, r_temp):
    lines = [
        "## Script 04 — Temporal Correlation",
        f"- Crops used: {n_crops}",
        f"- Note: within-track crops treated as independent (effective N ≈ tracks sampled)",
        "",
        "### Strongest temporal (dim × frame-feature) correlations",
    ]
    flat     = r_temp.ravel()
    top_idx  = np.argsort(np.abs(flat))[::-1][:8]
    for idx in top_idx:
        dim = idx // len(TEMPORAL_FEATURE_NAMES)
        fi  = idx % len(TEMPORAL_FEATURE_NAMES)
        lines.append(
            f"- Dim {dim:2d} × `{TEMPORAL_FEATURE_NAMES[fi]}`: r = {r_temp[dim,fi]:+.3f}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(force=args.force)
