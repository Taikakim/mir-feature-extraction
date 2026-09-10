#!/usr/bin/env python
"""goa_archive_mir.py — resumable batch MIR feature extraction over the goa_archive_extracted
corpus (~23k MP3 + FLAC full-mix tracks), the backbone of the 4-day unattended data-prep run
(CONTINUITY 2026-07-29, Kim direct). CPU-only (Essentia/MAEST TF graphs run on CPU here), so it
does NOT block the GPU captioning stage that follows.

Per track: ExpandedExtractor.extract over the FULL track (no truncation — the --seconds arg is
vestigial/unused; verified live 2026-07-30: dur_analyzed_s median ~415s, max ~4770s. ~22s/track,
3 workers ≈ 42h for the 23k archive) -> 24 expanded fields incl. maest_embed
(the similarity-clustering vector Kim asked for — MERT-like embedding cosine, NOT bit-hash, since
same-track-different-master across compilations = real mastering augmentation to KEEP, not drop).

Output (on the UUID drive, 1.5T free — NOT Mantu at 90%, NOT the SAO tree):
  <out>/npz/<sha1(relpath)>.npz            per-track fields+rates+meta (resumable: skip if exists)
  <out>/index.jsonl                        one row/track: path, dur, loudness, maest_vec(768 mean),
                                           genre/mood top tags — read by the similarity+curation stage
  <out>/mir.log / failures.jsonl           progress + per-track failures (fail-soft, one bad mp3
                                           never kills the run)

Resumable + idempotent: re-running skips tracks whose npz exists. Safe to kill/restart.
Run (mir venv):
  mir/bin/python src/tools/goa_archive_mir.py --archive /run/media/kim/Mantu/goa_archive_extracted \
      --out /run/media/kim/9a410a1d-a4a8-4faf-8298-bcaa2576ea9d/goa_archive_features \
      --workers 3 [--limit N]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # mir/src on path

AUDIO_EXT = {".mp3", ".flac", ".wav", ".m4a", ".MP3", ".FLAC"}


def track_key(archive: Path, p: Path) -> str:
    return hashlib.sha1(str(p.relative_to(archive)).encode()).hexdigest()


def enumerate_tracks(archive: Path):
    for root, _, files in os.walk(archive):
        for f in files:
            if os.path.splitext(f)[1] in AUDIO_EXT:
                yield Path(root) / f


# one extractor per worker process (loads the ~10 TF graphs once, amortized over the shard)
_EX = None


def _get_extractor():
    global _EX
    if _EX is None:
        from spectral.whole_track_expanded import ExpandedExtractor
        _EX = ExpandedExtractor(enable_models=True)
    return _EX


def process_one(args):
    archive_s, path_s, out_s, seconds = args
    archive, path, out = Path(archive_s), Path(path_s), Path(out_s)
    key = track_key(archive, path)
    npz_path = out / "npz" / f"{key}.npz"
    if npz_path.exists():
        return ("skip", path_s, key, None)
    try:
        t0 = time.time()
        ex = _get_extractor()
        data, rates, extra = ex.extract(path, existing=None, existing_meta=None)
        # persist all fields + rates + meta
        save = {f"f__{k}": v for k, v in data.items()}
        save.update({f"r__{k}": np.float64(rates[k]) for k in rates})
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, meta=json.dumps(extra), **save)
        # compact index row (similarity vector + curation scalars)
        mv = data.get("maest_embed_ts")
        maest = mv.mean(axis=0).astype(np.float32).tolist() if mv is not None and mv.size else None
        dur = None
        if "stereo_width_ts" in rates and "stereo_width_ts" in data:
            dur = round(len(data["stereo_width_ts"]) / max(rates["stereo_width_ts"], 1e-6), 2)
        row = {
            "key": key, "path": path_s,
            "rel": str(path.relative_to(archive)),
            "dur_analyzed_s": dur,
            "loudness_ebu_integrated": extra.get("loudness_ebu_integrated"),
            "loudness_ebu_range": extra.get("loudness_ebu_range"),
            "stereo_source": extra.get("stereo_source"),
            "maest_vec": maest,   # 768-d, mean-pooled — the similarity/dedup vector (MERT-like)
            "wall_s": round(time.time() - t0, 1),
        }
        return ("ok", path_s, key, row)
    except Exception as e:
        return ("fail", path_s, key, f"{type(e).__name__}: {str(e)[:200]}\n{traceback.format_exc()[-600:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seconds", type=float, default=90.0, help="analyze first N s (corpus convention)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="process at most N (smoke test)")
    a = ap.parse_args()

    (a.out / "npz").mkdir(parents=True, exist_ok=True)
    idx_path = a.out / "index.jsonl"
    fail_path = a.out / "failures.jsonl"
    log_path = a.out / "mir.log"

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    tracks = list(enumerate_tracks(a.archive))
    if a.limit:
        tracks = tracks[:a.limit]
    log(f"enumerated {len(tracks)} audio tracks under {a.archive}")

    done = set()
    if idx_path.exists():
        for line in open(idx_path):
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    todo = [t for t in tracks if track_key(a.archive, t) not in done
            and not (a.out / "npz" / f"{track_key(a.archive, t)}.npz").exists()]
    log(f"{len(done)} already indexed; {len(todo)} to do; {a.workers} workers")

    payloads = [(str(a.archive), str(t), str(a.out), a.seconds) for t in todo]
    n_ok = n_fail = n_skip = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex, \
         open(idx_path, "a") as fidx, open(fail_path, "a") as ffail:
        for i, res in enumerate(ex.map(process_one, payloads, chunksize=1)):
            status, path_s, key, payload = res
            if status == "ok":
                n_ok += 1
                fidx.write(json.dumps(payload) + "\n"); fidx.flush()
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
                ffail.write(json.dumps({"key": key, "path": path_s, "err": payload}) + "\n"); ffail.flush()
            if (i + 1) % 50 == 0 or (i + 1) == len(payloads):
                rate = (i + 1) / max(time.time() - t_start, 1e-6)
                eta_h = (len(payloads) - (i + 1)) / max(rate, 1e-6) / 3600
                log(f"{i+1}/{len(payloads)}  ok={n_ok} fail={n_fail}  {rate:.2f}/s  eta {eta_h:.1f}h")
    log(f"DONE: ok={n_ok} fail={n_fail} skip={n_skip}  in {(time.time()-t_start)/3600:.2f}h -> {a.out}")


if __name__ == "__main__":
    main()
