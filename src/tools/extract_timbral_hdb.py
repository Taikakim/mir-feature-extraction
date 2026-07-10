#!/usr/bin/env python3
"""
extract_timbral_hdb.py -- AudioCommons hardness/depth/booming (the safe subset --
NOT reverb, which can loop forever on pathological audio) over whole-track corpora
AND per-crop windows.

Kim's ask (via WINTERMUTE, 2026-07-10): DEPTH and BOOMING analysis (in addition to
hardness) over the goa and avp sets. Ground truth checked before writing this:
  - avp AUGMENTATION VARIANTS already carry all 8 AC features (WINTERMUTE's earlier
    reanalyze_variants.py run) -- but the avp SOURCE tracks themselves (the un-augmented
    <track>/full_mix.flac + its .INFO) do NOT; verified zero timbral fields on a sample.
  - Goa_Separated track .INFOs carry ZERO timbral fields at all; verified on a sample.
  - The 5400 latents_sa3 SA3 crops (T=4096, 10.767 Hz, 380s beat-aligned windows) have
    no timbral sidecar; WINTERMUTE needs per-crop hardness+depth+booming as a dependency
    for hardness LatCH head training.

Two modes, one script (shared feature-extraction core):

  --mode tracks --corpus name:/path/to/corpus_root [--corpus name2:/path2 ...]
      Whole-track full_mix.{flac,wav} -> safe_update() into the track's .INFO.
      Handles both <track>/full_mix.* (goa, avp source) layouts via glob.

  --mode crops --crops-dir /home/kim/Projects/latents_sa3
      For each <idx>.json crop companion: read ONLY the [start_sample:end_sample)
      window of source_path (soundfile partial read, no full-file decode), run
      hardness/depth/booming on the in-memory slice (no /dev/shm needed -- these 3
      features accept numpy arrays directly per analyze_all_timbral_features), write
      <idx>.TIMBRAL.json sidecar (mirrors the existing .TIMESERIES.npz convention).

Subprocess-isolated per item (mirrors reanalyze_variants.py's proven pattern: a stuck
or crashed child can't take the whole run down). hardness/depth/booming are NOT the
hang-prone AC feature (that's timbral_reverb, deliberately excluded here) but a
segfault-class failure is still possible on pathological audio, so isolation stays.
Resumable: tracks skip if .INFO already has all 3 fields; crops skip if the
.TIMBRAL.json sidecar already exists.

RUN WITH THE mir INTERPRETER (essentia not required, but keeps deps consistent):
    /home/kim/Projects/mir/mir/bin/python src/tools/extract_timbral_hdb.py \\
        --mode tracks --jobs 8 \\
        --corpus goa:/run/media/kim/Mantu/ai-music/Goa_Separated \\
        --corpus avp:/run/media/kim/9a410a1d-a4a8-4faf-8298-bcaa2576ea9d/avp-analyzed

    /home/kim/Projects/mir/mir/bin/python src/tools/extract_timbral_hdb.py \\
        --mode crops --jobs 8 --crops-dir /home/kim/Projects/latents_sa3
"""
import sys, os, json, time, argparse, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # reach src/

FEATURES = ["hardness", "depth", "booming"]  # deliberately NOT reverberation (hang risk)


# ---------- tracks mode ----------

def find_tracks(corpus, root):
    """Yield (corpus, track_name, full_mix_path, info_path)."""
    from core.json_handler import get_info_path
    root = Path(root)
    for fm in sorted(root.glob("*/full_mix.*")):
        if fm.suffix.lower() not in (".flac", ".wav", ".mp3", ".m4a"):
            continue
        yield (corpus, fm.parent.name, str(fm), str(get_info_path(fm)))


def _track_done(info_path):
    if not os.path.exists(info_path):
        return False
    try:
        d = json.load(open(info_path))
    except Exception:
        return False
    return all(k in d for k in FEATURES)


def process_track(fm_path, info_path):
    from timbral.audio_commons import analyze_all_timbral_features
    from core.json_handler import safe_update

    if _track_done(info_path):
        return "skip (done)"
    tmp_wav = None
    analyze_path = fm_path
    if fm_path.lower().endswith(".m4a"):
        # libsndfile (soundfile/timbral_models' underlying decoder) cannot open
        # m4a/AAC containers at all -- "Format not recognised", 100% reproducible,
        # nothing to do with system load (96/4470 goa tracks are m4a-sourced;
        # discovered 2026-07-10 debugging what looked like memory-pressure
        # failures but was this the whole time, coincidentally clustered near
        # the OOM incident in corpus sort order). Pre-transcode via ffmpeg to a
        # temp wav soundfile CAN read; mp3/ogg/aiff all work natively, no
        # transcode needed for those.
        tmp_wav = f"/dev/shm/mir_timbral_{os.getpid()}.wav"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", fm_path, tmp_wav],
                            capture_output=True, timeout=60)
        if r.returncode != 0:
            return f"ffmpeg transcode: {(r.stderr or b'').decode(errors='replace')[-150:]}"
        analyze_path = tmp_wav
    try:
        try:
            results = analyze_all_timbral_features(analyze_path, features=FEATURES)
        except Exception as e:
            return f"analyze: {type(e).__name__}: {e}"
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)
    missing = [f for f in FEATURES if f not in results]
    if missing:
        return f"partial (missing {missing})"
    try:
        safe_update(info_path, results)
    except Exception as e:
        return f"save: {e}"
    return "ok"


# ---------- crops mode ----------

def find_crops(crops_dir):
    """Yield (idx_str, json_path, sidecar_path) for every <idx>.json under crops_dir.
    Must exclude our OWN <idx>.TIMBRAL.json sidecar output -- it also ends in .json,
    so a naive '*.json' glob re-discovers it as a bogus pseudo-crop on the next run
    (caught 2026-07-10 in the first real launch: 000000.TIMBRAL.json got globbed and
    fed back in as if it were a crop companion, immediately erroring on missing
    source_path/start_sample/end_sample)."""
    crops_dir = Path(crops_dir)
    for jp in sorted(crops_dir.glob("*.json")):
        if jp.name.endswith(".lock") or jp.name.endswith(".TIMBRAL.json"):
            continue
        idx = jp.stem
        yield (idx, str(jp), str(crops_dir / f"{idx}.TIMBRAL.json"))


def process_crop(json_path, sidecar_path):
    import soundfile as sf
    from timbral.audio_commons import analyze_all_timbral_features

    if os.path.exists(sidecar_path):
        return "skip (done)"
    try:
        meta = json.load(open(json_path))
    except Exception as e:
        return f"meta: {e}"
    src = meta.get("source_path")
    start = meta.get("start_sample")
    end = meta.get("end_sample")
    if not (src and start is not None and end is not None):
        return "meta: missing source_path/start_sample/end_sample"
    if not os.path.exists(src):
        return f"source missing: {src}"
    try:
        audio, sr = sf.read(src, start=int(start), frames=int(end) - int(start), always_2d=False)
    except Exception as e:
        return f"read: {type(e).__name__}: {e}"
    if audio.size == 0:
        return "read: empty slice"
    try:
        results = analyze_all_timbral_features(json_path, features=FEATURES, audio=audio, sr=sr)
    except Exception as e:
        return f"analyze: {type(e).__name__}: {e}"
    missing = [f for f in FEATURES if f not in results]
    if missing:
        return f"partial (missing {missing})"
    try:
        tmp = sidecar_path + f".tmp{os.getpid()}"
        json.dump(results, open(tmp, "w"), indent=2)
        os.replace(tmp, sidecar_path)
    except Exception as e:
        return f"save: {e}"
    return "ok"


# ---------- subprocess isolation (mirrors reanalyze_variants.py) ----------

def run_one_subprocess(mode, item_key, hard_timeout):
    """item_key: for tracks, 'fm_path|info_path'; for crops, 'json_path|sidecar_path'."""
    try:
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--single", mode, item_key],
            timeout=hard_timeout, capture_output=True, text=True,
        )
        if r.returncode == 0:
            return (item_key, "ok", None)
        tail = (r.stdout or "").strip().splitlines()
        status = next((ln[7:] for ln in tail if ln.startswith("STATUS ")), None)
        return (item_key, status or f"exit{r.returncode}: {(r.stderr or '').strip()[-150:]}", None)
    except subprocess.TimeoutExpired:
        return (item_key, "TIMEOUT", None)
    except Exception as e:
        return (item_key, f"err: {e}", None)


def _is_done(mode, item_key):
    a, b = item_key.split("|", 1)
    if mode == "tracks":
        return _track_done(b)
    return os.path.exists(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tracks", "crops"], required=False)
    ap.add_argument("--corpus", action="append", default=[], help="name:/path (repeatable, tracks mode)")
    ap.add_argument("--crops-dir", default=None, help="crops mode: dir of <idx>.json companions")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--item-timeout", type=int, default=240,
                     help="hard wall-clock per item; a child exceeding this is SIGKILLed "
                          "(measured ~79s/full-track for hardness+depth+booming; crops are "
                          "similar duration to full tracks, ~380s beat-aligned windows)")
    ap.add_argument("--limit", type=int, default=0, help="process only first N (smoke test)")
    ap.add_argument("--max-passes", type=int, default=3,
                     help="retry not-done items this many passes; leftovers are quarantined")
    ap.add_argument("--single", nargs=2, metavar=("MODE", "ITEM_KEY"), default=None,
                     help="internal: process one item and exit")
    args = ap.parse_args()

    if args.single:
        smode, item_key = args.single
        a, b = item_key.split("|", 1)
        if smode == "tracks":
            status = process_track(a, b)
        else:
            status = process_crop(a, b)
        print(f"STATUS {status}")
        sys.exit(0 if status.startswith(("ok", "skip")) else 2)

    if not args.mode:
        ap.error("--mode is required (unless --single)")

    if args.mode == "tracks":
        items = []
        for spec in args.corpus:
            name, path = spec.split(":", 1)
            for corpus, track, fm, info in find_tracks(name, path):
                items.append(f"{fm}|{info}")
        if not items:
            print("[timbral] no --corpus given or no tracks found", flush=True)
            return
    else:
        if not args.crops_dir:
            ap.error("--crops-dir required for --mode crops")
        items = [f"{jp}|{sc}" for _idx, jp, sc in find_crops(args.crops_dir)]

    if args.limit:
        items = items[: args.limit]

    print(f"[timbral] mode={args.mode} {len(items)} items, jobs={args.jobs}, "
          f"item-timeout={args.item_timeout}s, subprocess-isolated, features={FEATURES}",
          flush=True)

    t0 = time.time()
    done_ok = 0
    for pass_no in range(1, args.max_passes + 1):
        remaining = [it for it in items if not _is_done(args.mode, it)]
        if not remaining:
            break
        print(f"[timbral] PASS {pass_no}/{args.max_passes}: {len(remaining)} remaining", flush=True)
        done_this_pass = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_one_subprocess, args.mode, it, args.item_timeout): it
                    for it in remaining}
            for f in as_completed(futs):
                item_key, status, _ = f.result()
                if status == "ok":
                    done_ok += 1
                    done_this_pass += 1
                if status != "ok" or done_ok % 50 == 0:
                    el = time.time() - t0
                    rate = done_ok / el if el > 0 else 0
                    ndone = sum(1 for it in items if _is_done(args.mode, it))
                    eta = (len(items) - ndone) / rate / 60 if rate > 0 else 0
                    label = Path(item_key.split("|", 1)[0]).parent.name if args.mode == "tracks" \
                        else Path(item_key.split("|", 1)[0]).stem
                    print(f"[timbral] ok={done_ok} {label[:30]} ({status}) pass={pass_no} "
                          f"done={ndone}/{len(items)} {rate:.2f}/s ETA~{eta:.0f}m", flush=True)
        print(f"[timbral] pass {pass_no} completed {done_this_pass} new", flush=True)

    ndone = sum(1 for it in items if _is_done(args.mode, it))
    leftover = [it for it in items if not _is_done(args.mode, it)]
    el = time.time() - t0
    print(f"[timbral] DONE {ndone}/{len(items)} complete, quarantined={len(leftover)} "
          f"(hung/crashed every pass) in {el/60:.1f}m", flush=True)
    for it in leftover[:40]:
        print(f"[timbral]   quarantined: {it}", flush=True)


if __name__ == "__main__":
    main()
