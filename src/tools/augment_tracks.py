#!/usr/bin/env python3
"""
augment_tracks.py — pitch/tempo data augmentation for the avp corpus (Bungee engine).

RUN WITH THE .venv INTERPRETER (has bungee_python):
    /home/kim/Projects/mir/.venv/bin/python src/tools/augment_tracks.py <avp-analyzed> [--jobs N]

For every kept track folder (<track>/full_mix.* + stems drums/bass/other/vocals.flac,
and <track>/<track>.INFO with a BPM) it renders variants into per-variant TRACK FOLDERS
so each augmentation is first-class (full_mix + 4 stems), ready for whole_track_timeseries
and SA3 encode:

    <track>/augmentations/<variant>/{full_mix,drums,bass,other,vocals}.flac

Variants (8 max per track):
  pitch  : pitch-2, pitch-1, pitch+1, pitch+2   (semitones; tempo preserved)
  tempo  : tempo-10, tempo-5, tempo+5, tempo+10 (RELATIVE %; pitch preserved)
           target_bpm = round(bpm * (1 ± pct)), CAPPED at 155, then speed = target/bpm.
           Variants whose capped target duplicates another (or equals source) are skipped.

Bungee: set_pitch(2**(st/12)) changes pitch only; set_speed(x>1) speeds up (raises tempo).
Output is peak-normalized into [-1,1] before FLAC write. Fully resumable + parallel
(ProcessPool over tracks; each worker loads one file at a time to cap memory).
"""
import sys, os, json, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import soundfile as sf

PITCH_SEMITONES = [-2, -1, 1, 2]
TEMPO_PCT = [-10, -5, 5, 10]
BPM_CAP = 155.0
BPM_KEYS = ["bpm_madmom", "bpm_essentia", "bpm", "tempo"]
STEMS = ["drums.flac", "bass.flac", "other.flac", "vocals.flac"]


def pick_bpm(info: dict):
    for k in BPM_KEYS:
        v = info.get(k)
        try:
            if v is not None and 20.0 < float(v) < 400.0:
                return float(v)
        except (TypeError, ValueError):
            continue
    return None


def load_info(track_dir: Path) -> dict:
    ip = track_dir / f"{track_dir.name}.INFO"
    try:
        return json.load(open(ip))
    except Exception:
        return {}


def variants_for(bpm):
    """Return [(variant_name, semitones, speed)]. Tempo capped at 155, deduped."""
    out = [(f"pitch{s:+d}", s, 1.0) for s in PITCH_SEMITONES]
    if bpm:
        seen_targets = {round(bpm)}
        for pct in TEMPO_PCT:
            target = min(round(bpm * (1.0 + pct / 100.0)), int(BPM_CAP))
            if target in seen_targets:          # cap collapsed it, or no-op
                continue
            seen_targets.add(target)
            out.append((f"tempo{pct:+d}", 0, target / bpm))
    return out


def _peak_norm(x):
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return (x * (0.999 / p)).astype(np.float32) if p > 0.999 else x.astype(np.float32)


def _render(data, sr, semis, speed):
    from bungee_python import bungee as B
    ch = data.shape[1] if data.ndim > 1 else 1
    st = B.Bungee(sample_rate=sr, channels=ch)
    if semis:
        st.set_pitch(2.0 ** (semis / 12.0))
    if speed != 1.0:
        st.set_speed(speed)
    ai = data.astype(np.float32)
    if ai.ndim == 1:
        ai = ai.reshape(-1, 1)
    out = np.asarray(st.process(ai), dtype=np.float32)
    if ch == 1:
        out = out.reshape(-1)
    elif out.ndim == 1:
        out = np.column_stack([out] * ch)
    return _peak_norm(out)


def process_track(track_dir_str: str):
    track_dir = Path(track_dir_str)
    fm = next(iter(track_dir.glob("full_mix.*")), None)
    if fm is None:
        return (track_dir.name, 0, "no full_mix")
    bpm = pick_bpm(load_info(track_dir))
    vs = variants_for(bpm)
    # source files present (full_mix -> full_mix.flac; stems keep name)
    sources = [(fm, "full_mix.flac")] + [(track_dir / s, s) for s in STEMS
                                         if (track_dir / s).exists()]
    made = 0
    for src_path, out_name in sources:
        # which variants still need this file?
        need = [(n, se, sp) for (n, se, sp) in vs
                if not (track_dir / "augmentations" / n / out_name).exists()]
        if not need:
            continue
        try:
            data, sr = sf.read(str(src_path))
        except Exception as e:
            return (track_dir.name, made, f"read {out_name}: {e}")
        for name, semis, speed in need:
            od = track_dir / "augmentations" / name
            od.mkdir(parents=True, exist_ok=True)
            try:
                out = _render(data, sr, semis, speed)
                tmp = od / f".{out_name}.tmp"
                sf.write(str(tmp), out, sr, format="FLAC")
                tmp.rename(od / out_name)
                made += 1
            except Exception as e:
                return (track_dir.name, made, f"{name}/{out_name}: {e}")
    return (track_dir.name, made, "ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    root = Path(args.root)
    tracks = sorted(str(p.parent) for p in root.glob("*/full_mix.*"))
    print(f"[augment] {len(tracks)} tracks, {args.jobs} workers, engine=bungee", flush=True)
    total = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(process_track, t): t for t in tracks}
        done = 0
        for f in as_completed(futs):
            name, made, status = f.result()
            total += made
            done += 1
            if status != "ok" or done % 20 == 0:
                print(f"[augment] {done}/{len(tracks)} rendered={total} "
                      f"last={Path(name).name[:30]} ({status})", flush=True)
    print(f"[augment] DONE rendered={total} files across {len(tracks)} tracks", flush=True)


if __name__ == "__main__":
    main()
