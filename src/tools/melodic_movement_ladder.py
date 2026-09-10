#!/usr/bin/env python3
"""Melodic-movement metrics across an a2a noise-level ladder.

Kim's hypothesis (2026-07-08): at nl 0.4-0.55 a2a melodies go stereotypical — limited
note range, less movement. Discriminating test: U-shape vs monotone across the ladder.
  - U-shape (dip at mid-nl, recovery by 0.7)  -> mid-noising regime artifact
    (posterior averaging over destroyed melodic detail)
  - monotone decline toward high-nl           -> model prior itself is melodically static

Metrics per clip (central 120 s, harmonic component, beat-agnostic):
  chroma_flux      mean frame-to-frame L2 chroma change  (tonal movement)
  pc_trans_rate    dominant-pitch-class changes per second (melodic movement)
  pc_entropy       entropy of time-averaged chroma (bits)  (note-range breadth)
  pc_active        pitch classes holding >5% of mass       (range width)
"""
import sys, os, glob, csv
import numpy as np

sys.path.insert(0, "/home/kim/Projects/mir/src")
from core.file_utils import read_audio  # noqa: E402
import librosa  # noqa: E402

SR = 22050
SLICE = (60.0, 180.0)  # central 120 s


def metrics(path):
    y, sr = read_audio(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
        sr = SR
    s, e = (int(t * sr) for t in SLICE)
    y = y[s:e]
    if len(y) < sr * 30:
        return None
    yh = librosa.effects.harmonic(y, margin=4.0)
    C = librosa.feature.chroma_cqt(y=yh, sr=sr, hop_length=2048)  # (12, T)
    Cn = C / (C.sum(axis=0, keepdims=True) + 1e-9)
    fps = sr / 2048
    flux = float(np.linalg.norm(np.diff(Cn, axis=1), axis=0).mean())
    dom = Cn.argmax(axis=0)
    trans = float((np.diff(dom) != 0).sum() / (len(dom) / fps))
    avg = Cn.mean(axis=1); avg = avg / avg.sum()
    ent = float(-(avg * np.log2(avg + 1e-12)).sum())
    active = int((avg > 0.05).sum())
    return dict(chroma_flux=flux, pc_trans_rate=trans, pc_entropy=ent, pc_active=active)


def default_jobs():
    jobs = []  # (label, nl, path)
    base = os.path.expanduser("~/.cache/evals_aac/renders")
    for adapter in ("evr1x", "newstack"):
        for p in sorted(glob.glob(f"{base}/a2a_kaikkialla_{adapter}/a2a_nl*.m4a")):
            nl = int(os.path.basename(p).split("nl")[1].split(".")[0]) / 100
            jobs.append((f"a2a_{adapter}", nl, p))
    src = "/run/media/kim/9a410a1d-a4a8-4faf-8298-bcaa2576ea9d/avp-flac/009 goddess guerrilla (2006)/Aavepyora - Goddess Guerilla - Kaikki-Alla.flac"
    if os.path.exists(src):
        jobs.append(("source", 0.0, src))
    return jobs


def main():  # noqa: C901
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", default=[],
                    help="dir of clips to score (label = dirname, nl parsed from _nl<NN> "
                         "in filename when present; repeatable). Default: the "
                         "a2a_kaikkialla ladder + source.")
    ap.add_argument("--source", default=None, help="reference source audio to include")
    ap.add_argument("--out", default="/home/kim/Projects/mir/stats/melodic_movement_ladder.csv")
    args = ap.parse_args()

    if args.dir:
        jobs = []
        for d in args.dir:
            label = os.path.basename(os.path.normpath(d))
            for p in sorted(glob.glob(os.path.join(d, "*.m4a")) +
                            glob.glob(os.path.join(d, "*.wav")) +
                            glob.glob(os.path.join(d, "*.flac"))):
                name = os.path.basename(p)
                nl = 0.0
                if "nl" in name:
                    try:
                        nl = int(name.split("nl")[1][:2]) / 100
                    except ValueError:
                        pass
                jobs.append((label, nl, p))
        if args.source:
            jobs.append(("source", 0.0, args.source))
    else:
        jobs = default_jobs()
    out = args.out
    print(f"{len(jobs)} clips")

    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["label", "nl", "chroma_flux", "pc_trans_rate",
                                           "pc_entropy", "pc_active", "path"])
        w.writeheader()
        for label, nl, p in jobs:
            try:
                m = metrics(p)
            except Exception as ex:
                print(f"  ERR {os.path.basename(p)}: {ex}"); continue
            if m is None:
                continue
            w.writerow(dict(label=label, nl=nl, path=os.path.basename(p), **m))
            fo.flush()
            print(f"  {label} nl={nl:.2f}: flux={m['chroma_flux']:.4f} "
                  f"trans={m['pc_trans_rate']:.2f}/s ent={m['pc_entropy']:.2f}b "
                  f"active={m['pc_active']}", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
