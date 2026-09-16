#!/usr/bin/env python
"""goa_archive_fast_features.py — the FAST base features for the goa_archive big set.

Why this exists: `goa_archive_features/npz` (23,231 tracks, built 2026-07-29 by
goa_archive_mir.py) carries the 24 EXPANDED fields only, because that run's purpose was
curation/similarity (MAEST embeddings), not control-head targets. The base fields an
onset- or volume-density head needs were never run over this corpus.

This fills that gap with the CHEAP fields only (project guidance 2026-09-16):

    onset_envelope_ts                                   librosa onset strength
    rms_energy_{bass,body,mid,air}_ts                   multiband volume envelopes
    hpcp_ts                                    (12,)    essentia chroma
    spectral_{flatness,flux,skewness,kurtosis}_ts
    same_chroma_ts                        (3,128,T)     SAME-compatible 3-band 384-d chroma

**Madmom beat/downbeat activations are DELIBERATELY SKIPPED** — they are the slow part
(RNN over the whole track). `extract_whole_track` skips them for free when the processors
are not supplied, so a later pass can add them without redoing any of this.

CPU-ONLY. Nothing here touches the GPU: librosa/essentia are CPU, and compute_same_chroma's
torch path is optional (numpy+scipy is the default). It will not contend with rendering or
training on the RX 9070 XT. For reference the expanded pass did 23,228 tracks in 66.3 h at
3 workers; MEASURE this one with --limit before assuming a wall-clock.

Layout note: the archive is a flat tree of release folders, but extract_whole_track expects
`<track_dir>/full_mix.<ext>`. We HARDLINK (os.link, per mir/CLAUDE.md — not symlink, which
confuses some readers, and not copy, which would duplicate ~1 TB) into a scratch dir.
Hardlinks cost no extra disk and are removed after each track.

Keys match goa_archive_mir.py exactly: sha1(relpath) -> <out>/npz/<sha1>.npz, with f__/r__
prefixes, so this store JOINS the existing expanded store on the same key.

Run (mir venv — it has essentia + librosa; mir/bin/python is the ffmpeg8-compat wrapper):

    /home/kim/Projects/mir/mir/bin/python src/tools/goa_archive_fast_features.py \
        --archive /run/media/kim/Mantu/goa_archive_extracted \
        --out /run/media/kim/9a410a1d-a4a8-4faf-8298-bcaa2576ea9d/goa_archive_fast \
        --workers 6 --limit 50          # drop --limit for the full run

Resumable and idempotent: re-running skips tracks whose npz exists. Safe to kill/restart.
"""
from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # mir/src
# same_chroma lives ONLY in the mir-same-chroma checkout (branch same-chroma); canonical
# mir/src/harmonic/ is otherwise file-identical but lacks it. Same insert chroma384_eval.py uses.
SAME_CHROMA_SRC = "/home/kim/Projects/mir-same-chroma/src"

AUDIO_EXT = {".mp3", ".flac", ".wav", ".m4a", ".MP3", ".FLAC"}

# The cheap base fields. Anything not here is either slow (madmom) or already covered by
# the expanded store.
FAST_FIELDS = [
    "onset_envelope_ts",
    "rms_energy_bass_ts", "rms_energy_body_ts", "rms_energy_mid_ts", "rms_energy_air_ts",
    "hpcp_ts",
    "spectral_flatness_ts", "spectral_flux_ts",
    "spectral_skewness_ts", "spectral_kurtosis_ts",
]


def track_key(archive: Path, p: Path) -> str:
    """Identical to goa_archive_mir.py, so both stores share one key space."""
    return hashlib.sha1(str(p.relative_to(archive)).encode()).hexdigest()


def process_one(args):
    archive_s, path_s, out_s, work_s, want_chroma = args
    archive, path, out, work = Path(archive_s), Path(path_s), Path(out_s), Path(work_s)
    key = track_key(archive, path)
    npz_path = out / "npz" / f"{key}.npz"
    if npz_path.exists():
        return ("skip", path_s, key, None)

    tdir = work / key
    linked = None
    try:
        t0 = time.time()
        # --- hardlink into the <track_dir>/full_mix.<ext> layout the extractor needs ----
        tdir.mkdir(parents=True, exist_ok=True)
        linked = tdir / f"full_mix{path.suffix.lower()}"
        if not linked.exists():
            try:
                os.link(path, linked)
            except OSError:          # cross-device (archive and scratch on different drives)
                import shutil
                shutil.copy2(path, linked)

        from spectral.whole_track_timeseries import extract_whole_track
        # beat_proc/downbeat_proc omitted => madmom activations skipped (the slow part).
        data, meta = extract_whole_track(tdir, beat_proc=None, downbeat_proc=None)

        keep = {k: v for k, v in data.items() if k in FAST_FIELDS}
        fr = float(meta.get("frame_rate", 100))
        rates = {k: fr for k in keep}

        # --- SAME 384-d three-band chroma, straight from the source audio --------------
        if want_chroma:
            if SAME_CHROMA_SRC not in sys.path:
                sys.path.insert(0, SAME_CHROMA_SRC)
            from harmonic.same_chroma import compute_same_chroma          # noqa: E402
            from core.file_utils import read_audio                        # noqa: E402
            audio, sr = read_audio(str(linked))
            # align_to_latent=False: there is no fixed latent length for a whole track, so
            # keep the native STFT grid (hop 4096 => sr/4096 ~= 10.77 Hz) and record the
            # true rate. The consumer resamples per field anyway.
            sc = compute_same_chroma(audio, sr, align_to_latent=False)
            sc = np.asarray(sc, dtype=np.float32)                          # (3, 128, T)
            keep["same_chroma_ts"] = sc
            dur = (len(audio) / sr) if audio.ndim == 1 else (audio.shape[0] / sr)
            rates["same_chroma_ts"] = float(sc.shape[-1] / max(dur, 1e-6))

        if not keep:
            return ("fail", path_s, key, "no fields produced")

        save = {f"f__{k}": v for k, v in keep.items()}
        save.update({f"r__{k}": np.float64(rates[k]) for k in rates})
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, meta=json.dumps(meta), **save)

        n = keep["onset_envelope_ts"].shape[0] if "onset_envelope_ts" in keep else 0
        return ("ok", path_s, key, {"key": key, "rel": str(path.relative_to(archive)),
                                    "fields": sorted(keep), "n_frames": int(n),
                                    "wall_s": round(time.time() - t0, 1)})
    except Exception as e:
        return ("fail", path_s, key, f"{type(e).__name__}: {str(e)[:200]}\n{traceback.format_exc()[-500:]}")
    finally:
        try:
            if linked and linked.exists():
                linked.unlink()
            if tdir.exists():
                tdir.rmdir()
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=None,
                    help="scratch dir for the hardlink shims. MUST be on the same filesystem as "
                         "--archive or os.link fails and every track is COPIED instead (~1 TB of "
                         "needless I/O over the full archive — measured live: /tmp is a different "
                         "fs from Mantu and silently fell back to copy). Defaults to "
                         "<archive>/../.goa_fast_work, which is the right filesystem by "
                         "construction; hardlinks there cost no extra disk.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="process at most N (SMOKE FIRST)")
    ap.add_argument("--no-chroma", action="store_true", help="skip the 384-d SAME chroma")
    a = ap.parse_args()

    (a.out / "npz").mkdir(parents=True, exist_ok=True)
    # Must share a filesystem with the archive, or os.link fails and every track is COPIED.
    work = a.work or (a.archive.parent / ".goa_fast_work")
    work.mkdir(parents=True, exist_ok=True)
    log = (a.out / "fast.log").open("a")

    def say(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    if os.stat(work).st_dev != os.stat(a.archive).st_dev:
        say(f"WARNING: --work {work} is on a DIFFERENT filesystem from --archive — os.link "
            f"will fail and every track will be COPIED instead of hardlinked (~1 TB of I/O "
            f"over the full archive). Point --work at the archive's own filesystem.")

    tracks = [p for p in a.archive.rglob("*") if p.suffix in AUDIO_EXT]
    tracks.sort()
    say(f"enumerated {len(tracks)} audio tracks under {a.archive}")
    todo = [t for t in tracks if not (a.out / "npz" / f"{track_key(a.archive, t)}.npz").exists()]
    if a.limit:
        todo = todo[:a.limit]
    say(f"{len(tracks) - len(todo)} already done; {len(todo)} to do; {a.workers} workers")

    payload = [(str(a.archive), str(t), str(a.out), str(work), not a.no_chroma) for t in todo]
    ok = fail = skip = 0
    t0 = time.time()
    idx = (a.out / "index.jsonl").open("a")
    fails = (a.out / "failures.jsonl").open("a")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process_one, p) for p in payload]
        for i, f in enumerate(as_completed(futs), 1):
            status, path_s, key, row = f.result()
            if status == "ok":
                ok += 1
                idx.write(json.dumps(row) + "\n")
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                fails.write(json.dumps({"path": path_s, "key": key, "err": row}) + "\n")
            if i % 50 == 0 or i == len(futs):
                rate = i / max(time.time() - t0, 1e-6)
                eta = (len(futs) - i) / max(rate, 1e-9) / 3600
                idx.flush(); fails.flush()
                say(f"{i}/{len(futs)}  ok={ok} fail={fail}  {rate:.2f}/s  eta {eta:.1f}h")
    say(f"DONE: ok={ok} fail={fail} skip={skip} in {(time.time()-t0)/3600:.2f}h -> {a.out}")
    return 0 if fail < max(1, len(todo) // 10) else 1


if __name__ == "__main__":
    raise SystemExit(main())
