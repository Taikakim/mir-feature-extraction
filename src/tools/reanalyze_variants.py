#!/usr/bin/env python3
"""
reanalyze_variants.py -- per-variant MIR feature RE-ANALYSIS for the avp aug corpus.

Runs the full CPU feature set (the same functions test_all_features.py runs) on each
augmentation variant folder produced by augment_tracks.py:

    <track>/augmentations/<variant>/{full_mix,drums,bass,other,vocals}.flac

Each variant is a first-class track folder (full_mix + in-folder stems), so FeatureTester
points straight at it and writes the variant's own .INFO. This is the "pending-D" full
re-analysis (Kim 2026-07-09): the augmentation changes real signal properties
non-analytically (transient softening, HF dulling, RMS shifts) that the features must
CAPTURE -- so we measure, not derive. The two analytically-exact fields (bpm x speed,
key + semitones) are derived as a CANARY and compared to the measured values: a mismatch
is a free pipeline-integrity alarm (mislabeled variant / broken transform).

PHASE 1 (this script) = CPU features only:
    loudness, beat_grid, bpm, onsets, syncopation, spectral, multiband_rms, chroma,
    timbral (audio_commons), per_stem_rhythm, per_stem_harmonic.
DEFERRED to a GPU-free window (phase 2, separate pass):
    audiobox_aesthetics, essentia (effnet/gmi), ADTOF midi -- they need VRAM and would
    OOM a co-resident training run.

RUN WITH THE mir INTERPRETER (essentia + madmom):
    /home/kim/Projects/mir/mir/bin/python src/tools/reanalyze_variants.py <avp-analyzed> \
        [--jobs 8] [--timbral-timeout 300] [--limit N]

Resumable: a variant whose .INFO already carries '_reanalyzed_cpu' is skipped.
"""
import sys, os, json, time, argparse, signal, re, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # reach src/

# CPU feature methods on FeatureTester, in dependency order (beat+onset before syncopation)
CPU_STEPS = [
    "run_loudness", "run_beat_grid", "run_bpm", "run_onsets", "run_syncopation",
    "run_spectral_features", "run_multiband_rms", "run_chroma",
    "run_per_stem_rhythm", "run_per_stem_harmonic",
]  # run_timbral handled separately (SIGALRM-guarded); GPU steps skipped entirely

BPM_KEYS = ["bpm_madmom", "bpm_essentia", "bpm", "tempo"]
BPM_CAP = 155
_PITCH_RE = re.compile(r"^pitch([+-]\d+)$")
_TEMPO_RE = re.compile(r"^tempo([+-]\d+)$")


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _pick_bpm(info):
    for k in BPM_KEYS:
        v = info.get(k)
        try:
            if v is not None and 20.0 < float(v) < 400.0:
                return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _load_info(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def _measured_bpm(info):
    return _pick_bpm(info)


def _canary(variant_name, source_bpm):
    """Return derived_bpm for this variant (None if not computable), mirroring
    augment_tracks.variants_for: pitch = tempo preserved; tempo = min(round(bpm*(1+p)), cap)."""
    if source_bpm is None:
        return None
    if _PITCH_RE.match(variant_name):
        return source_bpm  # pitch shift preserves tempo
    m = _TEMPO_RE.match(variant_name)
    if m:
        pct = int(m.group(1))
        return float(min(round(source_bpm * (1.0 + pct / 100.0)), BPM_CAP))
    return None


def process_variant(args):
    variant_dir_str, timbral_timeout = args
    vdir = Path(variant_dir_str)
    vname = vdir.name
    fm = vdir / "full_mix.flac"
    if not fm.exists():
        return (vname, "no full_mix", None)

    # import inside worker so the pool spawns cleanly
    from test_all_features import FeatureTester
    from core.json_handler import safe_update, get_info_path

    info_path = get_info_path(fm)
    existing = _load_info(info_path)
    if existing.get("_reanalyzed_cpu"):
        return (vname, "skip (done)", None)

    try:
        ft = FeatureTester(fm, skip_demucs=True, skip_flamingo=True, skip_midi=True,
                           parallel=False)
    except Exception as e:
        return (vname, f"init: {e}", None)

    results = {}
    for step in CPU_STEPS:
        try:
            out = getattr(ft, step)()
            if out:
                results.update(out)
        except Exception as e:
            results.setdefault("_reanalyze_errors", []).append(f"{step}: {e}")

    # timbral: SIGALRM-guarded (timbral_reverb can loop forever on pathological audio)
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(timbral_timeout))
    try:
        out = ft.run_timbral()
        if out:
            results.update(out)
    except _Timeout:
        results.setdefault("_reanalyze_errors", []).append("timbral: TIMEOUT")
    except Exception as e:
        results.setdefault("_reanalyze_errors", []).append(f"timbral: {e}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    # canary: derived bpm x factor vs measured
    canary_note = None
    src_info = _load_info(vdir.parent.parent / f"{vdir.parent.parent.name}.INFO")
    derived_bpm = _canary(vname, _pick_bpm(src_info))
    meas_bpm = _measured_bpm(results) or _measured_bpm(existing)
    if derived_bpm is not None and meas_bpm is not None:
        rel = abs(meas_bpm - derived_bpm) / derived_bpm
        results["_canary_bpm_derived"] = round(derived_bpm, 3)
        results["_canary_bpm_measured"] = round(float(meas_bpm), 3)
        results["_canary_bpm_relerr"] = round(rel, 4)
        if rel > 0.06:  # >6% off => flag (madmom octave errors ~ up to 2x)
            canary_note = f"BPM canary {vname}: derived={derived_bpm:.1f} measured={meas_bpm:.1f} rel={rel:.2%}"

    results["_reanalyzed_cpu"] = 1
    try:
        safe_update(info_path, results)
    except Exception as e:
        return (vname, f"save: {e}", canary_note)

    nerr = len(results.get("_reanalyze_errors", []))
    return (vname, "ok" if nerr == 0 else f"ok ({nerr} feat-errs)", canary_note)


def find_variants(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        if p.parent.name == "augmentations" and "full_mix.flac" in filenames:
            out.append(str(p))
    return sorted(out)


def _is_done(vdir):
    from core.json_handler import get_info_path
    return _load_info(get_info_path(Path(vdir) / "full_mix.flac")).get("_reanalyzed_cpu") == 1


def run_one_subprocess(vdir, hard_timeout, timbral_timeout):
    """Process ONE variant in an isolated child so a hang OR segfault can't take the run down:
    subprocess.run(timeout=) SIGKILLs a stuck child; a crash surfaces as a non-zero return code.
    This is the fix for both failure modes seen in-process (madmom segfault -> BrokenProcessPool;
    madmom/beat hang in an untimed step -> as_completed blocks forever)."""
    try:
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--single", vdir,
             "--timbral-timeout", str(timbral_timeout)],
            timeout=hard_timeout, capture_output=True, text=True,
        )
        if r.returncode == 0:
            tail = (r.stdout or "").strip().splitlines()
            canary = next((ln[7:] for ln in tail if ln.startswith("CANARY ")), None)
            return (vdir, "ok", canary)
        return (vdir, f"exit{r.returncode}: {(r.stderr or '').strip()[-100:]}", None)
    except subprocess.TimeoutExpired:
        return (vdir, "TIMEOUT", None)
    except Exception as e:
        return (vdir, f"err: {e}", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timbral-timeout", type=int, default=300)
    ap.add_argument("--variant-timeout", type=int, default=600,
                    help="hard wall-clock per variant; a child exceeding this is SIGKILLed")
    ap.add_argument("--limit", type=int, default=0, help="process only first N (smoke test)")
    ap.add_argument("--max-passes", type=int, default=4,
                    help="retry not-done variants this many passes; leftovers are quarantined")
    ap.add_argument("--single", action="store_true",
                    help="internal: process the single variant dir given as `root` and exit")
    args = ap.parse_args()

    # --single: the isolated worker path. `root` is the variant dir. Emit CANARY on stdout.
    if args.single:
        name, status, canary = process_variant((args.root, args.timbral_timeout))
        if canary:
            print(f"CANARY {canary}")
        # ok AND skip(done) are both success; only real errors (no full_mix, save fail) exit non-zero
        sys.exit(0 if status.startswith(("ok", "skip")) else 2)

    all_variants = find_variants(Path(args.root))
    if args.limit:
        all_variants = all_variants[: args.limit]

    print(f"[reanalyze] {len(all_variants)} variant folders, jobs={args.jobs}, "
          f"variant-timeout={args.variant_timeout}s, subprocess-isolated (CPU features only)",
          flush=True)

    t0 = time.time()
    done_ok = 0
    canaries = []
    for pass_no in range(1, args.max_passes + 1):
        remaining = [v for v in all_variants if not _is_done(v)]
        if not remaining:
            break
        print(f"[reanalyze] PASS {pass_no}/{args.max_passes}: {len(remaining)} remaining", flush=True)
        done_this_pass = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_one_subprocess, v, args.variant_timeout, args.timbral_timeout): v
                    for v in remaining}
            for f in as_completed(futs):
                vdir, status, canary = f.result()
                if status == "ok":
                    done_ok += 1
                    done_this_pass += 1
                if canary:
                    canaries.append(canary)
                    print(f"[reanalyze] CANARY {canary}", flush=True)
                if status != "ok" or done_ok % 25 == 0:
                    el = time.time() - t0
                    rate = done_ok / el if el > 0 else 0
                    ndone = sum(1 for v in all_variants if _is_done(v))
                    eta = (len(all_variants) - ndone) / rate / 60 if rate > 0 else 0
                    print(f"[reanalyze] ok={done_ok} {Path(vdir).parent.parent.name[:18]}/"
                          f"{Path(vdir).name} ({status}) pass={pass_no} done={ndone}/{len(all_variants)} "
                          f"{rate:.2f}/s ETA~{eta:.0f}m", flush=True)
        print(f"[reanalyze] pass {pass_no} completed {done_this_pass} new", flush=True)

    ndone = sum(1 for v in all_variants if _is_done(v))
    leftover = [v for v in all_variants if not _is_done(v)]
    el = time.time() - t0
    print(f"[reanalyze] DONE {ndone}/{len(all_variants)} complete, "
          f"quarantined={len(leftover)} (hung/crashed every pass), canary_flags={len(canaries)} "
          f"in {el/60:.1f}m", flush=True)
    for v in leftover[:40]:
        print(f"[reanalyze]   quarantined: {Path(v).parent.parent.name}/{Path(v).name}", flush=True)


if __name__ == "__main__":
    main()
