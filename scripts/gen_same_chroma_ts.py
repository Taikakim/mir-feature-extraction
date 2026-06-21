"""Generate SAME 384-d chroma timeseries for latents_sa3 crops (LatCH/control-head targets).

Pipeline per crop:  latent .npy --[ONNX SAME decoder, MIGraphX]--> audio 44.1k
                    --[compute_same_chroma]--> (3,128,T)  --reshape--> (T,384)
                    --> sidecar  <crop>.CHROMA.npz  (key 'same_chroma_ts', non-destructive).

Why ONNX decode (not torch / not source audio):
  - source audio lives on Mantu (removable, often unmounted);
  - the ONNX SAME decoder runs ~39x RTF in ~2 GB on MIGraphX and coexists with a training job;
  - chroma from the *decoded* audio is arguably the *right* steering target — it's what the model's
    own decode produces, which is exactly what we'll re-measure at eval time.

Run in the MIR venv (it has onnxruntime_migraphx + same_chroma). One long session (the ~9-min
MIGraphX AOT compile happens once at start; caching isn't exposed in this ORT build):

    /home/kim/Projects/mir/mir/bin/python scripts/gen_same_chroma_ts.py \
        --encoded-dir /home/kim/Projects/latents_sa3 --subset-tracks 800

Resumable (skips crops whose .CHROMA.npz exists). UNTESTED end-to-end until a GPU+MIGraphX run;
the shapes/recipe are validated against same_chroma's self-test and the decode_onnx seam test.

Storage note: stored (T,384) band-major [band0[0..127], band1, band2] so _to_ct -> (384,T) and
ChromaAttributeEncoder.view(B,3,128,T) reads bands then 128 pitch bins — matches the conditioner.
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-3/scripts")   # decode_onnx
sys.path.insert(0, "/home/kim/Projects/mir-same-chroma/src/harmonic")  # same_chroma
import onnxruntime as ort                                  # noqa: E402
from decode_onnx import decode_chunked_onnx                # noqa: E402
import same_chroma as sc                                   # noqa: E402


def select_crops(encoded_dir, subset_tracks, seed):
    """All .npy, or a seeded subset of TRACKS (all their crops) — mirrors the trainer's split."""
    npys = sorted(glob.glob(os.path.join(encoded_dir, "*.npy")))
    if not subset_tracks:
        return npys
    by_track = defaultdict(list)
    for p in npys:
        try:
            m = json.load(open(p[:-4] + ".json"))
        except Exception:
            m = {}
        by_track[m.get("source_track") or m.get("path") or p].append(p)
    keys = sorted(by_track)
    np.random.default_rng(seed).shuffle(keys)
    keep = set(keys[: int(subset_tracks)])
    return [p for k in keep for p in by_track[k]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoded-dir", default="/home/kim/Projects/latents_sa3")
    ap.add_argument("--onnx", default="/home/kim/Projects/SAO/stable-audio-3/same_decoder_L128.onnx")
    ap.add_argument("--chunk-latents", type=int, default=128)     # matches the exported L128 graph
    ap.add_argument("--overlap", type=int, default=16)            # >= decoder receptive field (seam-tested)
    ap.add_argument("--max-frames", type=int, default=4096,
                    help="decode the first N latent frames (4096=full crop; store full, slice at train)")
    ap.add_argument("--subset-tracks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)              # same default as the trainer's subset
    ap.add_argument("--sr", type=int, default=44100)             # SAME trained at 44.1k
    ap.add_argument("--limit", type=int, default=0, help="debug: process at most N crops")
    args = ap.parse_args()

    crops = select_crops(args.encoded_dir, args.subset_tracks, args.seed)
    todo = [p for p in crops if not os.path.exists(p[:-4] + ".CHROMA.npz")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[chroma] {len(crops)} crops selected, {len(todo)} to do "
          f"({len(crops) - len(todo)} already have .CHROMA.npz)", flush=True)
    if not todo:
        return

    print(f"[ort] loading {args.onnx} (MIGraphX AOT compile ~9 min on first session)…", flush=True)
    sess = ort.InferenceSession(
        args.onnx, providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"])
    ep = sess.get_providers()[0]
    print(f"[ort] active EP: {ep}" + ("  ⚠ CPU — decode will be SLOW (MIGraphX EP missing?)"
                                      if "CPU" in ep else ""), flush=True)

    t0 = time.time()
    for i, p in enumerate(todo):
        lat = np.load(p).astype(np.float32)
        if lat.ndim == 2:
            lat = lat[None]                                  # (1, 256, T)
        lat = lat[:, :, : args.max_frames]
        audio = decode_chunked_onnx(sess, lat, args.chunk_latents, args.overlap)   # (1, 2, samples)
        chroma = sc.compute_same_chroma(audio[0], args.sr, align_to_latent=True)   # (3, 128, T)
        T = chroma.shape[-1]
        ts = chroma.reshape(3 * 128, T).T.astype(np.float32)                       # (T, 384) band-major
        out = p[:-4] + ".CHROMA.npz"
        tmp = out + ".tmp"
        np.savez_compressed(tmp, same_chroma_ts=ts)
        os.replace(tmp, out)                                 # atomic write
        if (i + 1) % 20 == 0 or i == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(todo) - i - 1) / max(rate, 1e-9) / 60
            print(f"[chroma] {i + 1}/{len(todo)}  {rate:.2f} crop/s  ETA {eta:.0f} min  "
                  f"(last (T,384)={ts.shape})", flush=True)
    print(f"[chroma] DONE {len(todo)} crops in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
