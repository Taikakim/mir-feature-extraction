#!/usr/bin/env python3
"""Classify + build the per-track feature table for a NEW (non-Goa) latents corpus.

The Goa corpus already carries genre/mood in the crop jsons + goa_crops sidecars,
so build_feature_table.py joins them. A freshly-encoded corpus (GHOST's variety
batch) has NONE of that — only bpm/onset/year in the crop json + per-crop
.TIMESERIES.npz. This tool fills the gap in one GPU pass per source:

  per source_track (grouped from the crop jsons):
    audio -> EffNet embedding (once) -> genre head (discogs-400) + mood head (56)
    - writes ``style_genre`` (12-label) back into every crop json (additive,
      resumable) so the training dataloader gets it too, matching the Goa latents
    - assembles the per-track row: genre_vec (12) + genre_discogs_vec (400) +
      mood_vec (56) + rms from the crop .TIMESERIES.npz + bpm/onset/year from json

FIXED vocabularies (not observed-union): all new-corpus tables share an identical
layout, and their labels are the canonical discogs / mtg-jamendo strings, so they
align with the Goa table (a subset of the same taxonomies) by label at cluster time.

Run in the MIR venv (essentia + effnet_onnx/gmi_onnx + MIGraphX):

    mir/bin/python src/tools/classify_new_corpus.py \\
        --latents-dir /home/kim/Projects/latents_organic_dance \\
        --source organic_dance \\
        --out-dir data/feature_tables/organic_dance \\
        --vocab /home/kim/Projects/SAO/control/sa3_control/genre_vocab.json
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # reach src/

from tools.build_feature_table import (  # noqa: E402 — reuse the pure helpers
    RMS_BANDS,
    agg_scalars,
    align_vector,
    parse_year,
    write_outputs,
)
from tools.crop_genre import genre_vector_from_probs  # noqa: E402

logger = logging.getLogger("classify_new_corpus")


def _track_rms(crop_paths: list[str]) -> dict[str, dict]:
    """Aggregate per-band rms from each crop's .TIMESERIES.npz -> per-track stats.

    Each crop npz carries rms_energy_<band>_ts (T,). We take each crop's per-band
    mean, then agg_scalars across the track's crops (mean/std)."""
    per_band: dict[str, list] = {b: [] for b in RMS_BANDS}
    for jp in crop_paths:
        npz = jp[:-5] + ".TIMESERIES.npz"
        if not os.path.exists(npz):
            continue
        try:
            z = np.load(npz, allow_pickle=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("bad npz %s: %s", npz, exc)
            continue
        for b in RMS_BANDS:
            key = f"rms_energy_{b}_ts"
            if key in z:
                per_band[b].append(float(np.mean(z[key])))
    return {b: agg_scalars(vals) for b, vals in per_band.items()}


def run(latents_dir: str, source: str, out_dir: Path, vocab_json: str) -> int:
    import essentia.standard as es
    from classification.effnet_onnx import get_effnet_migraphx
    from classification.essentia_features import (
        get_classification_labels, get_model_path)
    from classification.gmi_onnx import get_gmi_model
    from core.json_handler import safe_update
    from tools.genre_vocab import compute_track_mean400

    cfg = json.load(open(vocab_json))
    vocab12: list[str] = cfg["vocab"]
    labels400: list[str] = cfg["labels400"]
    mood_labels: list[str] = get_classification_labels()["mood"]        # 56
    genre_vocab = vocab12 + ["other"]                                   # 12 (matches style_genre keys)
    logger.info("vocabs — genre %d, discogs %d, mood %d",
                len(genre_vocab), len(labels400), len(mood_labels))

    effnet = get_effnet_migraphx(get_model_path("discogs-effnet-bsdynamic-1.onnx"))
    models_dir = Path(get_model_path("genre_discogs400-discogs-effnet-1.pb")).parent
    genre_model = get_gmi_model("genre", models_dir)
    mood_model = get_gmi_model("mood", models_dir)

    # group crops by source track
    by_track: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for jp in glob.glob(os.path.join(latents_dir, "*.json")):
        try:
            m = json.load(open(jp))
        except Exception:
            continue
        by_track[m.get("source_track") or jp].append((jp, m))
    logger.info("%s: %d crops, %d tracks", source,
                sum(len(v) for v in by_track.values()), len(by_track))

    rows = []
    n_missing_src = 0
    for i, (track, crops) in enumerate(sorted(by_track.items()), 1):
        src = crops[0][1].get("source_path") or crops[0][1].get("path")
        if not src or not os.path.exists(src):
            logger.warning("source missing for %s: %s", track, src)
            n_missing_src += 1
            continue

        # one embedding -> two heads
        audio = es.MonoLoader(filename=str(src), sampleRate=16000, resampleQuality=4)()
        emb = effnet(audio)
        mean400 = np.mean(genre_model(emb), axis=0).astype(np.float32)
        mood56 = np.mean(mood_model(emb), axis=0).astype(np.float32)

        style = genre_vector_from_probs(mean400, vocab12, labels400)      # dict{12}
        for jp, _ in crops:                                               # write back (additive)
            safe_update(jp, {"style_genre": style})

        onset = agg_scalars([m.get("onset_density") for _, m in crops])
        bpm = agg_scalars([(m.get("bpm_essentia") or m.get("bpm_madmom")) for _, m in crops])
        year, year_known = parse_year(
            next((m.get("track_metadata_year") for _, m in crops if m.get("track_metadata_year")), None))
        rms = _track_rms([jp for jp, _ in crops])

        row = {
            "source": source,
            "source_track": track,
            "latent_indices": [Path(jp).stem for jp, _ in crops],
            "n_crops": len(crops),
            "genre_vec": align_vector(style, genre_vocab),                # 12
            "genre_discogs_vec": [float(x) for x in mean400],             # 400, fixed order
            "mood_vec": [float(x) for x in mood56],                       # 56, fixed order
            "mood_present": True,
            "bpm_mean": bpm["mean"],
            "onset_density_mean": onset["mean"],
            "onset_density_std": onset["std"],
            "release_year": year,
            "year_known": year_known,
        }
        for b in RMS_BANDS:
            row[f"rms_energy_{b}_mean"] = rms[b]["mean"]
            row[f"rms_energy_{b}_std"] = rms[b]["std"]
        rows.append(row)
        if i % 25 == 0:
            logger.info("  %d/%d tracks classified", i, len(by_track))

    if not rows:
        logger.error("no rows produced (all sources missing?)")
        return 1
    vocabs = {"genre_vocab": genre_vocab,
              "genre_discogs_vocab": labels400,
              "mood_vocab": mood_labels}
    write_outputs(rows, vocabs, out_dir, source)
    logger.info("done — %d tracks, %d missing-source", len(rows), n_missing_src)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latents-dir", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--vocab", required=True,
                    help="genre_vocab.json (vocab + labels400)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.setLevel(logging.INFO)
    return run(args.latents_dir, args.source, args.out_dir, args.vocab)


if __name__ == "__main__":
    raise SystemExit(main())
