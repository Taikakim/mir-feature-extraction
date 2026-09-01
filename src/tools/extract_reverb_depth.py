#!/usr/bin/env python3
"""
extract_reverb_depth.py — AudioCommons RT60 (reverb dev_output) + depth over a corpus.

For the "pad-fill" hypothesis: the model papers over under-constrained regions with
reverby atmosphere, so RT60/depth become eval/rerank signals. This produces the corpus
BASELINE distributions (goa vs Kim's music) to meter outputs against.

Per track (full_mix): timbral_reverb(dev_output=True) -> (mean_RT60, reverb_prob),
timbral_depth -> depth. Continuous values (not the true/false default).

Robust: per-file SIGALRM timeout (timbral_reverb can hang on pathological audio),
ProcessPool parallel, resumable (appends to CSV, skips done). mir venv, CPU, GPU-free.

Usage:
  mir/bin/python src/tools/extract_reverb_depth.py --out <csv> --jobs 10 \
      --corpus goa:/run/media/kim/Mantu1/ai-music/Goa_Separated \
      --corpus avp:/run/media/kim/9a41.../avp-analyzed
"""
import sys, os, csv, time, signal, argparse, glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/home/kim/Projects/mir/repos/timbral_models")
TIMEOUT = 90  # seconds per file; timbral_reverb can loop forever otherwise


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def analyze_one(job):
    corpus, track, fm = job
    signal.signal(signal.SIGALRM, _alarm)
    import timbral_models as T
    rt60 = prob = depth = None
    status = "ok"
    try:
        signal.alarm(TIMEOUT)
        rt60, prob = T.timbral_reverb(fm, dev_output=True)
        signal.alarm(TIMEOUT)
        depth = T.timbral_depth(fm)
        signal.alarm(0)
    except _Timeout:
        status = "timeout"
    except Exception as e:
        status = f"err:{type(e).__name__}"
    finally:
        signal.alarm(0)
    return dict(corpus=corpus, track=track, rt60=rt60, reverb_prob=prob,
               depth=depth, status=status)


def find_tracks(corpus, root):
    """Yield (corpus, track_name, full_mix_path). Handles <track>/full_mix.* layout
    (avp, goa_separated) and flat <track>.flac layout."""
    root = Path(root)
    fms = sorted(root.glob("*/full_mix.*"))
    if fms:
        for fm in fms:
            yield (corpus, fm.parent.name, str(fm))
        return
    # flat: any audio file = a track
    for ext in ("flac", "wav", "mp3", "m4a"):
        for f in sorted(root.glob(f"*.{ext}")):
            yield (corpus, f.stem, str(f))


def load_done(out):
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for row in csv.DictReader(f):
                done.add((row["corpus"], row["track"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--corpus", action="append", default=[],
                    help="name:/path (repeatable)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    jobs = []
    for spec in args.corpus:
        name, path = spec.split(":", 1)
        jobs += list(find_tracks(name, path))
    done = load_done(args.out)
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(jobs)} tracks total, {len(done)} already done, {len(todo)} to do "
          f"({args.jobs} workers)", flush=True)

    new_file = not os.path.exists(args.out)
    fout = open(args.out, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=["corpus", "track", "rt60", "reverb_prob",
                                         "depth", "status"])
    if new_file:
        w.writeheader(); fout.flush()

    t0 = time.time(); n = 0; bad = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(analyze_one, j) for j in todo]
        for f in as_completed(futs):
            r = f.result()
            w.writerow(r); fout.flush()
            n += 1
            if r["status"] != "ok":
                bad += 1
            if n % 50 == 0:
                rate = n / (time.time() - t0)
                eta = (len(todo) - n) / rate / 60 if rate else 0
                print(f"  {n}/{len(todo)} done ({bad} bad) "
                      f"{rate*60:.0f}/min, ETA {eta:.0f}min", flush=True)
    fout.close()
    print(f"DONE: {n} processed, {bad} timeout/err. -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
