"""Store the fixed-vocab genre raw-prob vector + 'other' bucket into each crop .json.

Additive; genre is track-level so the inference head runs ONCE per source track.
A per-track softmax cache (``--cache``) lets subsequent incremental runs skip tracks
already scored without re-running the model.

Run in the MIR venv (essentia + effnet_onnx/gmi_onnx + MIGraphX):

    mir/bin/python src/tools/crop_genre.py \\
        --latents-dir /home/kim/Projects/latents_sa3 \\
        --vocab /home/kim/Projects/SAO/control/sa3_control/genre_vocab.json

The full corpus scan (~55 min) is deferred to a manual run when the GPU is free.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Pure helper — no I/O, fully testable
# ---------------------------------------------------------------------------

def genre_vector_from_probs(
    mean400,
    vocab: list[str],
    labels400: list[str],
) -> dict[str, float]:
    """Build the fixed-vocab genre vector with an 'other' bucket.

    Args:
        mean400:   np.ndarray of shape (N,) — per-label softmax probabilities
                   (N == len(labels400), typically 400 for discogs-400).
        vocab:     Ordered list of genre labels that form the target vocabulary
                   (a subset of labels400).
        labels400: Full ordered list of discogs-400 label strings.

    Returns:
        Dict ``{genre: prob for genre in vocab} | {"other": 1 − Σ(selected)}``.
        Guaranteed to form a proper simplex: all values ∈ [0, 1] and sum == 1.
        "other" is clamped to 0.0 if the selected probs sum to > 1 (numerical noise).
    """
    mean400 = np.asarray(mean400, dtype=np.float32)
    idx = {lbl: i for i, lbl in enumerate(labels400)}
    sel = {g: float(mean400[idx[g]]) for g in vocab}
    other = max(0.0, 1.0 - float(sum(sel.values())))
    return {**sel, "other": other}


# ---------------------------------------------------------------------------
# Corpus scanner — runs deferred (imports essentia / model weights on demand)
# ---------------------------------------------------------------------------

def store_genre(
    latents_dir: str,
    vocab_json: str,
    cache_file: str | None = None,
) -> None:
    """Additively write ``style_genre`` into every crop .json under *latents_dir*.

    Genre is a track-level property, so the discogs-400 head runs ONCE per
    source track and the resulting vector is broadcast to all its crops.

    Resumable: crops that already carry ``style_genre`` are skipped; tracks
    already in the per-track softmax cache are not re-scored.

    Args:
        latents_dir: Directory containing ``*.json`` crop metadata files.
        vocab_json:  Path to ``genre_vocab.json`` (Task 1 output).
        cache_file:  Optional path to a JSON file used as a persistent
                     per-track mean400 cache.  Loaded at startup; updated
                     incrementally so interrupted runs benefit immediately.
    """
    import glob
    import json
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # reach src/
    import essentia.standard as es
    from classification.effnet_onnx import get_effnet_migraphx
    from classification.essentia_features import get_model_path, get_classification_labels
    from classification.gmi_onnx import get_gmi_model
    from core.json_handler import safe_update
    from tools.genre_vocab import compute_track_mean400

    # --- load vocab ---
    cfg = json.load(open(vocab_json))
    vocab: list[str] = cfg["vocab"]
    labels400: list[str] = cfg["labels400"]

    # --- load models ---
    effnet = get_effnet_migraphx(get_model_path("discogs-effnet-bsdynamic-1.onnx"))
    models_dir = Path(get_model_path("genre_discogs400-discogs-effnet-1.pb")).parent
    genre_model = get_gmi_model("genre", models_dir)

    # --- per-track softmax cache (avoids re-running the model on repeated runs) ---
    cache: dict[str, list[float]] = {}
    if cache_file and os.path.exists(cache_file):
        try:
            cache = json.load(open(cache_file))
            print(f"[genre] loaded cache: {len(cache)} tracks from {cache_file}", flush=True)
        except Exception as e:
            print(f"[genre] cache load failed ({e}), starting fresh", flush=True)

    def _flush_cache() -> None:
        if not cache_file:
            return
        try:
            tmp = cache_file + ".tmp"
            json.dump(cache, open(tmp, "w"))
            os.replace(tmp, cache_file)
        except Exception as e:
            print(f"[genre] cache flush failed: {e}", flush=True)

    # --- group crop JSONs by source track → run the head ONCE per track ---
    by_track: dict[str, list[tuple[str, dict]]] = {}
    for j in glob.glob(os.path.join(latents_dir, "*.json")):
        try:
            m = json.load(open(j))
        except Exception:
            continue
        key = m.get("source_track") or m.get("path")
        by_track.setdefault(key, []).append((j, m))

    total_tracks = len(by_track)
    scored = 0
    skipped_already_done = 0
    skipped_no_src = 0

    for i, (trk, crops) in enumerate(sorted(by_track.items())):
        # Skip if every crop already has the field (resumable)
        if all("style_genre" in m for _, m in crops):
            skipped_already_done += 1
            continue

        src = crops[0][1].get("source_path") or crops[0][1].get("path")
        if not src or not os.path.exists(src):
            print(f"[skip] source missing: {src}", flush=True)
            skipped_no_src += 1
            continue

        # Use cached mean400 if available
        if str(src) in cache:
            mean400 = np.array(cache[str(src)], dtype=np.float32)
        else:
            mean400 = compute_track_mean400(src, effnet, genre_model, es)
            cache[str(src)] = mean400.tolist()
            _flush_cache()

        vec = genre_vector_from_probs(mean400, vocab, labels400)

        # Additive atomic write — never clobbers existing keys
        for j, _ in crops:
            safe_update(j, {"style_genre": vec})

        scored += 1
        if (i + 1) % 50 == 0:
            print(
                f"[genre] {i+1}/{total_tracks} tracks "
                f"(scored={scored}, already_done={skipped_already_done}, "
                f"missing_src={skipped_no_src})",
                flush=True,
            )

    _flush_cache()
    print(
        f"[genre] done — scored={scored}, already_done={skipped_already_done}, "
        f"missing_src={skipped_no_src}, total_tracks={total_tracks}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Write style_genre vector into every SA3 crop .json (additive)."
    )
    ap.add_argument(
        "--latents-dir",
        default="/home/kim/Projects/latents_sa3",
        help="Directory containing *.json crop metadata files.",
    )
    ap.add_argument(
        "--vocab",
        default="/home/kim/Projects/SAO/control/sa3_control/genre_vocab.json",
        help="Path to genre_vocab.json (Task 1 output).",
    )
    ap.add_argument(
        "--cache",
        default=None,
        help="Optional path to persist per-track mean400 cache (JSON).",
    )
    a = ap.parse_args()
    store_genre(a.latents_dir, a.vocab, cache_file=a.cache)
