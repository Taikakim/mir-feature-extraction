#!/usr/bin/env python3
"""section_extractor.py -- corpus-wide section boundaries + energy arcs from the
whole-track timeseries (the "free road" of the section-conditioning plan,
Kim 2026-07-16/17; spec: SAO docs/superpowers/specs/2026-07-17-muscriptor-lumi-batch.md §7).

Foote checkerboard novelty on a cosine SSM over pooled multiband-RMS + spectral +
HPCP features (100 Hz npz -> 2 Hz), boundaries SNAPPED TO DOWNBEATS (the bar-snap
lesson: beat-grid vs bar-grid anchoring caused half-beat kick offsets in the
chroma-morph renders), per-section energy arc = mean(bass+body RMS) z-scored
against the track. Output feeds section_pos_ts / section_arc_ts conditioning
(v1 labels); MuScriptor MIDI-derived sections (the 5% sample) are the validator.

Usage (mir venv):
  python src/tools/section_extractor.py --limit 200            # sample pass
  python src/tools/section_extractor.py                        # full corpus
  python src/tools/section_extractor.py --track "Bic - Vermillion" -v
Outputs one <track>.sections.json per track under --out
(default /run/media/kim/Lehto/section_labels/).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TS_DIR = Path("/run/media/kim/Lehto/timeseries")
OUT_DIR = Path("/run/media/kim/Lehto/section_labels")
GOA = Path("/run/media/kim/Mantu/ai-music/Goa_Separated")

POOL = 50                      # 100 Hz -> 2 Hz
KERNEL_SEC = 32.0              # Foote checkerboard width
MIN_GAP_SEC = 12.0             # shortest section (goa sample: 8 bars ~ 13 s)
SCALARS = ["rms_energy_bass_ts", "rms_energy_body_ts", "rms_energy_mid_ts",
           "rms_energy_air_ts", "spectral_flatness_ts", "spectral_flux_ts",
           "spectral_skewness_ts", "spectral_kurtosis_ts", "onset_envelope_ts"]


def _features(z):
    cols = []
    n = min(len(z[k]) for k in SCALARS)
    for k in SCALARS:
        cols.append(np.asarray(z[k][:n], np.float32))
    hp = np.asarray(z["hpcp_ts"][:n], np.float32)          # (n, 12)
    X = np.column_stack(cols + [hp])                        # (n, 21)
    m = (len(X) // POOL) * POOL
    Xp = X[:m].reshape(-1, POOL, X.shape[1]).mean(1)        # 2 Hz
    mu, sd = Xp.mean(0, keepdims=True), Xp.std(0, keepdims=True)
    return (Xp - mu) / np.maximum(sd, 1e-6)


def foote_novelty(F, fps=2.0, kernel_sec=32.0):
    U = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)
    S = U @ U.T
    L = max(4, int(round(kernel_sec * fps / 2)))
    r = np.arange(-L, L)
    g = np.exp(-0.5 * (r / (L / 2.0)) ** 2)
    K = np.outer(g, g) * np.sign(np.outer(r, r) + 1e-9)     # checkerboard
    T = len(S)
    nov = np.zeros(T, np.float32)
    for t in range(L, T - L):
        nov[t] = float((S[t - L:t + L, t - L:t + L] * K).sum())
    nov -= nov.min()
    return nov / max(nov.max(), 1e-8)


def pick_boundaries(nov, fps=2.0, min_gap_sec=12.0, prominence=0.1):
    from scipy.signal import find_peaks
    peaks, props = find_peaks(nov, distance=int(min_gap_sec * fps),
                              prominence=prominence * (nov.max() - nov.mean() + 1e-8))
    return peaks / fps


def snap_to_downbeats(bounds_sec, track):
    db_file = GOA / track / f"{track}.DOWNBEATS"
    if not db_file.exists():
        return list(map(float, bounds_sec)), False
    db = np.array([float(x) for x in db_file.read_text().split()], np.float64)
    if db.size < 4:
        return list(map(float, bounds_sec)), False
    snapped = sorted({float(db[np.abs(db - b).argmin()]) for b in bounds_sec})
    return snapped, True


def extract_track(npz_path, verbose=False, kernel_sec=32.0, min_gap_sec=12.0, prominence=0.1):
    track = npz_path.name[: -len(".TIMESERIES.npz")]
    with np.load(npz_path, allow_pickle=True) as z:
        meta = json.loads(str(z["__meta__"]))
        F = _features(z)
        # arc source at 2 Hz: bass+body energy
        n = min(len(z["rms_energy_bass_ts"]), len(z["rms_energy_body_ts"]))
        e = (np.asarray(z["rms_energy_bass_ts"][:n], np.float32)
             + np.asarray(z["rms_energy_body_ts"][:n], np.float32))
    dur = float(meta.get("duration", len(F) / 2.0))
    nov = foote_novelty(F, kernel_sec=kernel_sec)
    bounds, snapped = snap_to_downbeats(
        pick_boundaries(nov, min_gap_sec=min_gap_sec, prominence=prominence), track)
    bounds = [b for b in bounds if 5.0 < b < dur - 5.0]
    edges = [0.0] + bounds + [dur]
    ez = (e - e.mean()) / max(e.std(), 1e-8)
    fps100 = 100.0
    arcs = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = ez[int(a * fps100): int(b * fps100)]
        arcs.append(round(float(seg.mean()) if len(seg) else 0.0, 3))
    out = {"track": track, "duration_sec": round(dur, 1),
           "boundaries_sec": [round(b, 2) for b in bounds],
           "n_sections": len(arcs), "section_arc_z": arcs,
           "downbeat_snapped": snapped,
           "method": f"foote(cos-SSM 21d@2Hz, kernel {kernel_sec:.0f}s, "
                     f"min-gap {min_gap_sec:.0f}s, prom {prominence:.2f}, downbeat-snap)"}
    if verbose:
        print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--track", default=None)
    ap.add_argument("--only-tracks-file", default=None,
                    help="newline list of track names to process (validation subset)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--kernel-sec", type=float, default=32.0)
    ap.add_argument("--min-gap-sec", type=float, default=12.0)
    ap.add_argument("--prominence", type=float, default=0.1)
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(TS_DIR.glob("*.TIMESERIES.npz"))
    if args.track:
        files = [f for f in files if f.name.startswith(args.track)]
    if args.only_tracks_file:
        keep = set(Path(args.only_tracks_file).read_text().splitlines())
        files = [f for f in files if f.name[: -len(".TIMESERIES.npz")] in keep]
    if args.limit:
        files = files[: args.limit]
    done = skip = fail = 0
    for f in files:
        dst = out_dir / (f.name[: -len(".TIMESERIES.npz")] + ".sections.json")
        if dst.exists():
            skip += 1
            continue
        try:
            dst.write_text(json.dumps(extract_track(f, args.verbose, args.kernel_sec, args.min_gap_sec, args.prominence), indent=1))
            done += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {f.name}: {type(e).__name__}: {e}", flush=True)
        if done and done % 200 == 0:
            print(f"[sections] {done} done / {skip} skipped / {fail} failed", flush=True)
    print(f"[sections] TOTAL {done} done / {skip} skipped / {fail} failed -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
