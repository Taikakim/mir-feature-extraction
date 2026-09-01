#!/usr/bin/env python
"""goa_archive_quality.py — audio-quality audit of goa_archive_extracted. Header bitrate/codec/sr
(ffprobe) PLUS the real tell: SPECTRAL CUTOFF (lowpass brickwall of the averaged spectrum). Header
bitrate lies — a '320k' vintage file is often a transcode with real content only to ~16 kHz — so the
cutoff is what says how much genuine bandwidth is there vs a re-encoded 192-kbps MP3.

Tier by effective bandwidth (highest freq within 40 dB of the 1-4 kHz passband):
  A_near_lossless  >=20 kHz   (CD-sourced / lossless)
  B_256-320k       18-20 kHz
  C_~192k          16-18 kHz
  D_<=128k_lossy   <16 kHz    (heavy-lossy / transcode)

CPU, resumable (index.jsonl skip), parallel. --limit N + --stratify = even stride across the corpus
for a fast representative sample; omit for the full 23k. Prints an aggregate summary at the end.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXT = {".mp3", ".flac", ".wav", ".m4a", ".MP3", ".FLAC"}


def ffprobe(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=30).stdout
        j = json.loads(out)
        fmt, streams = j.get("format", {}), j.get("streams", [])
        a = next((s for s in streams if s.get("codec_type") == "audio"), {})
        br = fmt.get("bit_rate") or a.get("bit_rate")
        return {"bitrate_k": int(br) // 1000 if br else None, "codec": a.get("codec_name"),
                "sr": int(a.get("sample_rate", 0)) or None,
                "dur": round(float(fmt.get("duration", 0)), 1) or None}
    except Exception:
        return {"bitrate_k": None, "codec": None, "sr": None, "dur": None}


def spectral_cutoff(path, seg=20.0, n_windows=6):
    """Effective bandwidth = the MAXIMUM cutoff over several windows spread across the track.

    v2, 2026-08-21 (Kim caught it): v1 measured ONE 30 s window at offset 30 s and treated
    that as the file's bandwidth. Goa tracks routinely open with a long filtered/ambient
    intro, so that window measured the INTRO's low-pass, not the encode. Measured live:
    Koxbox "Space Interface" reads 9.6 kHz at offset 30 s and 21.5 kHz everywhere else --
    a near-lossless file that v1 tiered as D_<=128k_lossy. Four independent rips of that
    same track all landed in the "worst 100" for this reason, which is what exposed it.

    MAX, not median or mean, is the principled statistic here: lossy encoding imposes a
    hard CEILING, so a 128k file cannot exceed ~16 kHz in ANY window, while a lossless file
    reaches ~21 kHz in at least one. A central statistic would instead punish tracks that
    are legitimately quiet or filtered for much of their length -- i.e. ambient, breakdowns,
    and intros -- which is the very failure being fixed.
    """
    import librosa

    def _window(off):
        try:
            y, sr = librosa.load(str(path), sr=None, mono=True, offset=off, duration=seg)
        except Exception:
            return None, None
        if y is None or sr is None or len(y) < sr:
            return None, sr
        S = np.abs(librosa.stft(y, n_fft=8192)) ** 2
        p = S.mean(axis=1)
        freqs = np.linspace(0, sr / 2, len(p))
        pdb = 10 * np.log10(p + 1e-12)
        band = (freqs >= 1000) & (freqs <= 4000)
        # A near-silent window has no meaningful passband to reference; skip it rather
        # than emit a spurious cutoff from noise floor alone.
        if not band.any() or float(np.max(pdb)) < -80:
            return None, sr
        ref = np.median(pdb[band])
        above = np.where(pdb > ref - 40)[0]
        return (round(float(freqs[above[-1]]), 0) if len(above) else 0.0), sr

    try:
        dur = float(sf.info(str(path)).duration)
    except Exception:
        dur = 0.0
    if dur <= 0:                       # unreadable header (some vintage mp3): fall back
        offs = [30.0, 0.0]
    else:
        # spread over the middle 80%, avoiding the intro and the fade-out
        lo, hi = 0.10 * dur, 0.90 * dur
        if hi - lo < seg:
            offs = [max(0.0, (dur - seg) / 2)]
        else:
            offs = [lo + (hi - lo - seg) * i / max(1, n_windows - 1) for i in range(n_windows)]

    cuts, sr_seen = [], None
    for off in offs:
        c, sr = _window(off)
        sr_seen = sr or sr_seen
        if c is not None:
            cuts.append(c)
    if not cuts:                       # nothing readable anywhere
        return None, sr_seen
    return max(cuts), sr_seen


def tier(cut):
    if cut is None:
        return "unknown"
    if cut >= 20000:
        return "A_near_lossless"
    if cut >= 18000:
        return "B_256-320k"
    if cut >= 16000:
        return "C_~192k"
    return "D_<=128k_lossy"


def process(path_s):
    p = Path(path_s)
    h = ffprobe(p)
    cut, sr = spectral_cutoff(p)
    if sr and not h.get("sr"):
        h["sr"] = sr
    return {"path": path_s, "cutoff_hz": cut, "tier": tier(cut), **h}


def enumerate_tracks(archive, stem_filter=None):
    """stem_filter: only files whose basename (sans extension) matches. Needed for
    Goa_Separated-style layouts, where each track folder holds full_mix.* alongside
    drums/bass/other/vocals.mp3 -- measuring the separated stems would both 5x the work
    and corrupt the tier distribution, since the stems are lossy renders of the model
    output, not the source master."""
    for root, _, files in os.walk(archive):
        for f in files:
            if os.path.splitext(f)[1] not in AUDIO_EXT:
                continue
            if stem_filter and os.path.splitext(f)[0] != stem_filter:
                continue
            yield os.path.join(root, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stratify", action="store_true", help="even stride to --limit (representative sample)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only-name", default=None,
                    help="only files with this basename sans extension (e.g. full_mix) -- "
                         "use for Goa_Separated-style per-track folders that also hold stems")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    idx = os.path.join(a.out, "quality.jsonl")

    tracks = list(enumerate_tracks(a.archive, a.only_name))
    if a.limit and a.stratify and a.limit < len(tracks):
        step = len(tracks) / a.limit
        tracks = [tracks[int(i * step)] for i in range(a.limit)]
    elif a.limit:
        tracks = tracks[:a.limit]

    done = set()
    if os.path.exists(idx):
        for line in open(idx):
            try:
                done.add(json.loads(line)["path"])
            except Exception:
                pass
    todo = [t for t in tracks if t not in done]
    print(f"[quality] {len(tracks)} tracks ({len(done)} done, {len(todo)} to do), {a.workers} workers", flush=True)

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex, open(idx, "a") as f:
        for i, r in enumerate(ex.map(process, todo, chunksize=2)):
            rows.append(r)
            f.write(json.dumps(r) + "\n"); f.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(todo)}  {(i+1)/(time.time()-t0):.1f}/s", flush=True)

    # aggregate over EVERYTHING indexed (this run + prior)
    allrows = [json.loads(l) for l in open(idx)]
    n = len(allrows)
    print(f"\n===== QUALITY SUMMARY ({n} tracks) =====")
    tiers = Counter(r["tier"] for r in allrows)
    print("TIER (by spectral cutoff — the REAL bandwidth):")
    for t in ["A_near_lossless", "B_256-320k", "C_~192k", "D_<=128k_lossy", "unknown"]:
        c = tiers.get(t, 0)
        print(f"  {t:18s} {c:6d}  {100*c/max(n,1):5.1f}%")
    cuts = [r["cutoff_hz"] for r in allrows if r.get("cutoff_hz")]
    if cuts:
        cq = np.percentile(cuts, [10, 25, 50, 75, 90])
        print(f"cutoff kHz  p10/25/50/75/90 = {'/'.join(f'{x/1000:.1f}' for x in cq)}")
    print("CODEC:", dict(Counter(r.get("codec") for r in allrows)))
    brs = [r["bitrate_k"] for r in allrows if r.get("bitrate_k")]
    if brs:
        bq = np.percentile(brs, [10, 50, 90])
        print(f"header bitrate kbps  p10/50/90 = {'/'.join(f'{int(x)}' for x in bq)}  (NOTE: header, not real quality)")
    srs = Counter(r.get("sr") for r in allrows)
    print("SAMPLE RATE:", dict(srs))
    print(f"[quality] wrote {idx}")


if __name__ == "__main__":
    main()
