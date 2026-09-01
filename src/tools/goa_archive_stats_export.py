#!/usr/bin/env python3
"""goa_archive_stats_export.py -- reduces the goa_archive_extracted per-track NPZ
time-series (mir/src/spectral/whole_track_expanded.py output, 20+ fields at native
per-field rates, some 2D) into flat scalar/categorical .INFO-style JSON files, so
the existing mir/src/tools/statistical_analysis.py (correlation/VIF/PCA/clustering/
MI -- "the usual statistics") can run over the archive unmodified.

Kim direct 2026-08-02: "mir has been run on the big goa set, might as well run the
usual statistics on it, including clustering." Clustering already exists
(goa_archive_curate.py, CONTINUITY, verified this session to reflect the complete
final 23231-track index). This script is the missing half: statistical_analysis.py
was built for flat .INFO features (one JSON object of scalar key->value pairs per
track/crop); the archive's real output is per-frame time-series + a couple of 2D
embedding/probability arrays. No existing per-track scalar-reduction consumer was
found for this format (checked: whole_track_expanded.py is producer-only, no
aggregator). This is a first-cut aggregation, not an authoritative methodology --
flagged to CONTINUITY (owns the field design) before running at scale.

Aggregation choices (documented, revisit if the numbers look off):
  - 1D scalar-rate fields (dissonance/pitch_salience/inharmonicity/novelty_curve/
    dyncomplexity(_loudness)/loudness_ebu_{momentary,shortterm}): mean + std over
    time -> two features each.
  - chroma_linmap / bass_chroma_linmap (T,12): mean-pooled 12-bin chroma vector,
    then TWO derived scalars -- entropy (spread across pitch classes: low =
    tonally centered, high = diffuse/atonal) and the vector's L2 norm (overall
    harmonic salience). The 12 raw bins are NOT expanded into 12 separate .INFO
    keys (would 24x the harmonic feature count for little interpretive value).
  - effnet_{genre400,moodtheme,instrument} (T,C): mean-pooled class-probability
    vector, then top-1 class NAME (categorical feature) + its probability
    (numeric confidence). Class names from the Essentia model JSONs (models/
    essentia/*-effnet-1.json "classes" list), not raw indices.
  - va_deam / va_emomusic (T,2): mean valence + mean arousal per model (2 models
    -> 4 features) -- kept SEPARATE (not averaged together) since they're two
    independently-trained emotion models, not two measurements of one thing.
  - maest_embed_ts (T,768): EXCLUDED from this export. It's a 768-d semantic
    embedding, not a musically-interpretable scalar; forcing it into one number
    (e.g. its norm) loses almost everything, and it's already used properly
    elsewhere (goa_archive_curate.py's cosine clustering on the full vector,
    index.jsonl's pooled maest_vec). This tool is for univariate/correlation-
    style stats on interpretable features, not embedding analysis.
  - r__*_ts (per-field sample rate in Hz): metadata about the data, not data
    about the track -- excluded.
  - index.jsonl scalars folded in as-is: dur_analyzed_s, loudness_ebu_integrated,
    loudness_ebu_range, stereo_source.

Output: <out>/<key>/<key>.INFO  (one dir+file per track, matching statistical_
analysis.py's `file_names.append(info_file.parent.name)` convention) so the
existing tool's `--build-db` / `--feature-select` / `--cluster` flags all work
against it exactly as documented, no code changes to that tool.

Run (mir venv, CPU): python src/tools/goa_archive_stats_export.py \
    --features /path/to/goa_archive_features --out /path/to/goa_archive_info
"""
import argparse
import json
from pathlib import Path

import numpy as np

MODELS_ESSENTIA = Path(__file__).resolve().parents[2] / "models/essentia"
CLASS_JSON = {
    "effnet_genre400_ts": "genre_discogs400-discogs-effnet-1.json",
    "effnet_moodtheme_ts": "mtg_jamendo_moodtheme-discogs-effnet-1.json",
    "effnet_instrument_ts": "mtg_jamendo_instrument-discogs-effnet-1.json",
}

SCALAR_1D = ["dissonance_ts", "pitch_salience_ts", "inharmonicity_ts", "novelty_curve_ts",
             "dyncomplexity_ts", "dyncomplexity_loudness_ts",
             "loudness_ebu_momentary_ts", "loudness_ebu_shortterm_ts"]
CHROMA_2D = ["chroma_linmap_ts", "bass_chroma_linmap_ts"]
VA_2D = ["va_deam_ts", "va_emomusic_ts"]


def _load_class_names():
    out = {}
    for field, fname in CLASS_JSON.items():
        p = MODELS_ESSENTIA / fname
        if p.exists():
            out[field] = json.loads(p.read_text())["classes"]
    return out


def _entropy(p):
    p = p / (p.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def track_features(npz_path, class_names, index_row):
    d = np.load(npz_path)
    feat = {}
    for f in SCALAR_1D:
        k = f"f__{f}"
        if k in d:
            v = d[k]
            if v.size:
                feat[f.removesuffix("_ts") + "_mean"] = float(np.mean(v))
                feat[f.removesuffix("_ts") + "_std"] = float(np.std(v))
    for f in CHROMA_2D:
        k = f"f__{f}"
        if k in d and d[k].size:
            pooled = d[k].mean(axis=0)  # (12,)
            feat[f.removesuffix("_ts") + "_entropy"] = _entropy(np.abs(pooled))
            feat[f.removesuffix("_ts") + "_norm"] = float(np.linalg.norm(pooled))
    for f in VA_2D:
        k = f"f__{f}"
        if k in d and d[k].size:
            pooled = d[k].mean(axis=0)  # (2,) = [valence, arousal]
            base = f.removesuffix("_ts")
            feat[f"{base}_valence"] = float(pooled[0])
            feat[f"{base}_arousal"] = float(pooled[1])
    for f, names in class_names.items():
        k = f"f__{f}"
        if k in d and d[k].size:
            pooled = d[k].mean(axis=0)
            top = int(np.argmax(pooled))
            base = f.removesuffix("_ts").removeprefix("effnet_")
            feat[f"{base}_top_class"] = names[top] if top < len(names) else str(top)
            feat[f"{base}_top_prob"] = float(pooled[top])
    for k in ("dur_analyzed_s", "loudness_ebu_integrated", "loudness_ebu_range"):
        if index_row.get(k) is not None:
            feat[k] = index_row[k]
    if "stereo_source" in index_row:
        feat["stereo_source"] = "stereo" if index_row["stereo_source"] else "mono"
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True, help="goa_archive_features dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap tracks (smoke test)")
    args = ap.parse_args()

    class_names = _load_class_names()
    print(f"[stats-export] class-name tables loaded: {list(class_names.keys())}")

    index = {}
    for line in open(args.features / "index.jsonl"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        index[row["key"]] = row

    npz_dir = args.features / "npz"
    npz_files = sorted(npz_dir.glob("*.npz"))
    if args.limit:
        npz_files = npz_files[: args.limit]
    print(f"[stats-export] {len(npz_files)} npz files, {len(index)} index rows")

    n_ok = n_skip = 0
    for i, npz_path in enumerate(npz_files):
        key = npz_path.stem
        row = index.get(key)
        if row is None:
            n_skip += 1
            continue
        try:
            feat = track_features(npz_path, class_names, row)
        except Exception as e:
            print(f"[stats-export] FAILED {key}: {e}")
            n_skip += 1
            continue
        track_dir = args.out / key
        track_dir.mkdir(parents=True, exist_ok=True)
        (track_dir / f"{key}.INFO").write_text(json.dumps(feat))
        n_ok += 1
        if (i + 1) % 2000 == 0:
            print(f"[stats-export] {i + 1}/{len(npz_files)}  ok={n_ok} skip={n_skip}")

    print(f"[stats-export] DONE: ok={n_ok} skip={n_skip} -> {args.out}")


if __name__ == "__main__":
    main()
