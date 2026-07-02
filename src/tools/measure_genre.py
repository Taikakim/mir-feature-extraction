"""measure_genre.py — score genre-eval renders with the discogs-400 head (MIR venv).

Reads a genre_eval manifest.json (produced by SA3 sa3_control/genre_eval.py), runs the
discogs-400 genre head on each rendered clip, maps the 400 probs to the fixed vocab, and
writes genre_scores.json. This is the measurement half of the style-adapter genre test:
does conditioning the fingerprint on genre X actually raise the OUTPUT's measured genre X?

Run in the MIR venv (essentia + effnet_onnx/gmi_onnx + MIGraphX):
    mir/bin/python src/tools/measure_genre.py --eval-dir <genre_eval_fpX> \\
        --vocab /home/kim/Projects/SAO/control/sa3_control/genre_vocab.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--eval-dir", required=True, help="dir with render WAVs + manifest.json")
    ap.add_argument("--vocab", required=True, help="genre_vocab.json")
    ap.add_argument("--out", default=None, help="default: <eval-dir>/genre_scores.json")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # reach src/
    import essentia.standard as es
    from classification.effnet_onnx import get_effnet_migraphx
    from classification.essentia_features import get_model_path
    from classification.gmi_onnx import get_gmi_model
    from tools.genre_vocab import compute_track_mean400
    from tools.crop_genre import genre_vector_from_probs

    cfg = json.load(open(args.vocab))
    vocab: list[str] = cfg["vocab"]
    labels400: list[str] = cfg["labels400"]

    effnet = get_effnet_migraphx(get_model_path("discogs-effnet-bsdynamic-1.onnx"))
    models_dir = Path(get_model_path("genre_discogs400-discogs-effnet-1.pb")).parent
    genre_model = get_gmi_model("genre", models_dir)

    manifest = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    results = []
    for clip in manifest["clips"]:
        wav = os.path.join(args.eval_dir, clip["file"])
        if not os.path.exists(wav):
            print(f"[measure] MISSING {clip['file']}", flush=True)
            continue
        mean400 = compute_track_mean400(wav, effnet, genre_model, es)
        vec = genre_vector_from_probs(mean400, vocab, labels400)   # {full-label: prob} + "other"
        measured = {g: float(vec.get(g, 0.0)) for g in vocab}      # vocab-11 only
        argmax_g = max(measured, key=measured.get)
        gi = clip.get("genre_idx", -1)
        target_label = vocab[gi] if 0 <= gi < len(vocab) else None
        results.append({
            **clip,
            "measured": measured,
            "argmax": argmax_g,
            "argmax_prob": measured[argmax_g],
            "target_label": target_label,
            "target_prob": float(measured.get(target_label, 0.0)) if target_label else None,
            "other": float(vec.get("other", 0.0)),
        })
        tp = f"{measured.get(target_label, 0.0):.3f}" if target_label else "  -  "
        print(f"[measure] {clip['file']:<42} cond={clip['genre']:<14} "
              f"-> argmax={argmax_g.split('---')[-1]:<16} ({measured[argmax_g]:.3f})  target={tp}",
              flush=True)

    out = args.out or os.path.join(args.eval_dir, "genre_scores.json")
    json.dump({"vocab": vocab, "eval_dir": args.eval_dir, "results": results}, open(out, "w"), indent=1)
    print(f"[measure] {len(results)} clips -> {out}", flush=True)


if __name__ == "__main__":
    main()
