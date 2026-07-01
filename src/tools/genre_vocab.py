"""Build the fixed genre vocabulary for the SA3 style adapter from the corpus
discogs-400 distribution. Runs in the MIR venv (essentia + gmi_onnx + MIGraphX)."""
from __future__ import annotations
import numpy as np


def genre_significant_labels(mean400, labels400, prob_thresh: float = 0.10):
    """Return labels whose mean probability >= prob_thresh."""
    mean400 = np.asarray(mean400, dtype=np.float32)
    return [labels400[i] for i in range(len(labels400)) if mean400[i] >= prob_thresh]


def select_genre_vocab(counts: dict, min_support: int = 303):
    """Return labels with count >= min_support, sorted by count descending then label ascending."""
    kept = [(lbl, n) for lbl, n in counts.items() if n >= min_support]
    kept.sort(key=lambda kv: (-kv[1], kv[0]))
    return [lbl for lbl, _ in kept]


# ---------------------------------------------------------------------------
# Corpus scan (runs only when executed as __main__ or called directly)
# ---------------------------------------------------------------------------

def scan_corpus(latents_dir: str, out_json: str, prob_thresh: float = 0.10, min_support: int = 303):
    """Scan the SA3 training corpus and emit a genre_vocab.json artifact.

    Loads each source track once, runs the discogs-400 genre head, counts how
    many crops carry each significant label (mean_prob >= prob_thresh), and
    writes the JSON artifact with the filtered vocabulary (count >= min_support).
    """
    import glob
    import json
    import os
    import sys
    from collections import Counter
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # reach src/
    import essentia.standard as es
    from classification.effnet_onnx import get_effnet_migraphx
    from classification.essentia_features import get_model_path, get_classification_labels
    from classification.gmi_onnx import get_gmi_model

    labels400 = get_classification_labels()["genre"]
    effnet = get_effnet_migraphx(get_model_path("discogs-effnet-bsdynamic-1.onnx"))
    models_dir = Path(get_model_path("genre_discogs400-discogs-effnet-1.pb")).parent
    genre = get_gmi_model("genre", models_dir)

    # Group crop JSONs by source track → run the head ONCE per track
    by_track: dict = {}
    for j in glob.glob(os.path.join(latents_dir, "*.json")):
        m = json.load(open(j))
        by_track.setdefault(m.get("source_track") or m.get("path"), []).append((j, m))

    counts: Counter = Counter()
    n_crops = 0
    skipped = 0
    for i, (trk, crops) in enumerate(sorted(by_track.items())):
        src = crops[0][1].get("source_path") or crops[0][1].get("path")
        if not src or not os.path.exists(src):
            print(f"[skip] source missing: {src}", flush=True)
            skipped += 1
            continue
        audio = es.MonoLoader(filename=src, sampleRate=16000, resampleQuality=4)()
        mean400 = np.mean(genre(effnet(audio)), axis=0)   # (400,) softmax, mean over patches
        sig = set(genre_significant_labels(mean400, labels400, prob_thresh))
        for _, _m in crops:
            for lbl in sig:
                counts[lbl] += 1
            n_crops += 1
        if (i + 1) % 50 == 0:
            print(f"[vocab] {i+1}/{len(by_track)} tracks, {n_crops} crops, {skipped} skipped", flush=True)

    vocab = select_genre_vocab(dict(counts), min_support)
    out = {
        "vocab": vocab,
        "min_support": min_support,
        "significance": f"mean_prob>={prob_thresh}",
        "counts": dict(counts),
        "n_crops": n_crops,
        "labels400": labels400,
    }
    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(out_json, "w"), indent=1)
    print(f"[vocab] K={len(vocab)}: {vocab}", flush=True)
    print(f"[vocab] wrote {out_json} ({n_crops} crops, {skipped} tracks skipped)", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build fixed genre vocabulary from SA3 corpus")
    ap.add_argument("--latents-dir", default="/home/kim/Projects/latents_sa3")
    ap.add_argument("--out", default="/home/kim/Projects/SAO/control/sa3_control/genre_vocab.json")
    ap.add_argument("--prob-thresh", type=float, default=0.10)
    ap.add_argument("--min-support", type=int, default=303)
    a = ap.parse_args()
    scan_corpus(a.latents_dir, a.out, a.prob_thresh, a.min_support)
