#!/usr/bin/env python3
"""articulation_stream.py — derive a scalar ARTICULATION/NOVELTY stream from the whole-track
multi-field timeseries, and validate it against the beat grid.

WHY THIS EXISTS (WINTERMUTE, 2026-08-12, from the reading sweep).
Three separate methods we want all bottom out in the *same* missing step:

  * Schindler et al. 2505.10004 (topology-driven repetitions) needs a scalar SURROGATE v(t)
    "capturing relative position within the current cycle";
  * Popoff & Yust meter networks need an ARTICULATION SET per "part" — and per Kim's
    generalisation (2026-08-12) any descriptor stream can be a part, but only once you can say
    *when* it articulates;
  * Heo & Jung 2405.04796 (PH of featured time series) needs a DISCRETISATION, because its
    graph nodes are distinct VALUES.

One step gates all three: turn our continuous 46-field timeseries into the right scalar/discrete
stream. That is what this does. It is CPU-only, needs no training and no GPU.

    articulation_stream.py --track "Ayahuasca - Propella" --validate
    articulation_stream.py --npz /path/to/X.TIMESERIES.npz --fields spectral_flux_ts,hpcp_ts

VALIDATION IS THE POINT, NOT A GARNISH. A novelty curve that fires often will hit every
downbeat by chance, so alignment alone proves nothing. Every validation run therefore reports
the score against a NULL built by shuffling the inter-peak intervals — same peak count, same
interval distribution, no relationship to the audio. A result that does not beat its own null
is reported as a failure however good the raw F-measure looks. This is the same discipline that
caught Audiobox CE ranking Kim's two favourite checkpoints 1st and 8th of 8: a number that moves
is not the same as a number that means something.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TS_ROOT = Path("/run/media/kim/Lehto/timeseries")
CORPUS_ROOT = Path("/run/media/kim/Mantu/ai-music/Goa_Separated")

# Sensible default: streams that plausibly mark *events* rather than slow state. Per-stem onset
# envelopes are the most literal articulation signal we have; spectral flux is already a novelty
# curve; hpcp/chroma flux marks harmonic change (Kim's "harmonic rhythm"); band energies mark
# per-band pulse. Deliberately excludes the slow embedding fields (0.2-1 Hz) -- they describe
# state, not articulation, and resampling them up would invent detail that is not there.
DEFAULT_FIELDS = [
    "onset_envelope_drums_ts", "onset_envelope_bass_ts", "onset_envelope_other_ts",
    "onsets_activations_ts", "spectral_flux_ts",
    "hpcp_ts", "chroma_linmap_ts",
    "rms_energy_bass_ts", "rms_energy_mid_ts", "rms_energy_air_ts",
]


def _rate_of(name: str, meta: dict, default: float = 100.0) -> float:
    """Field rate from the sidecar. The 20 legacy fields predate `field_rates` and are 100 Hz
    (MASTER §2); anything else without an entry is a bug worth surfacing, not silently guessing."""
    return float((meta.get("field_rates") or {}).get(name, default))


# Fields that ARE ALREADY a novelty/onset curve. Differencing these is a real bug, not a
# nuance: it yields the derivative OF a novelty, which peaks on the *rise* of an onset rather
# than the onset. Diagnosed 2026-08-12 by the null test -- the onset envelopes scored AT CHANCE
# against the beat grid while chroma flux (a genuine state field, correctly differenced) beat
# its null clearly. An onset detector failing to find beats while a harmony field finds them is
# not a plausible result about music; it is a sign the transform is wrong.
ALREADY_NOVELTY = (
    "onset_envelope_", "onsets_activations", "spectral_flux",
    "beat_activation", "downbeat_activation",
)


def is_novelty_field(name: str) -> bool:
    return any(name.startswith(p) or p in name for p in ALREADY_NOVELTY)


def field_novelty(arr: np.ndarray, name: str = "") -> np.ndarray:
    """Half-wave-rectified frame-to-frame change: how much did this stream just CHANGE.

    For fields that are already novelty curves (see ALREADY_NOVELTY) the field itself IS the
    articulation signal and is returned as-is.

    For a multi-dimensional field (hpcp is (N,12)) the novelty is the norm of the difference
    vector -- i.e. flux -- so chroma contributes "the harmony moved", not "which chord".
    Rectified because an articulation is an ONSET of change; decay is not an event.
    """
    a = np.asarray(arr, dtype=np.float64)
    if name and is_novelty_field(name):
        return a if a.ndim == 1 else np.linalg.norm(a, axis=1)
    if a.ndim == 1:
        d = np.diff(a, prepend=a[:1])
        return np.maximum(d, 0.0)
    d = np.diff(a, axis=0, prepend=a[:1])
    return np.linalg.norm(np.maximum(d, 0.0), axis=1)


def _robust_norm(x: np.ndarray) -> np.ndarray:
    """Scale to a comparable range without letting one loud field dominate the fusion.
    Median/IQR rather than max: a single transient must not set the scale for a whole track."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    med = np.median(x)
    iqr = np.subtract(*np.percentile(x, [75, 25])) or (x.std() or 1.0)
    return np.maximum(x - med, 0.0) / iqr


def _resample_to(x: np.ndarray, src_rate: float, dst_rate: float, n_out: int) -> np.ndarray:
    if abs(src_rate - dst_rate) < 1e-9 and len(x) == n_out:
        return x
    src_t = np.arange(len(x)) / src_rate
    dst_t = np.arange(n_out) / dst_rate
    return np.interp(dst_t, src_t, x)


def build_stream(npz_path: Path, fields=None, rate: float = 100.0):
    """-> (fused novelty stream at `rate`, per-field streams, meta). Fields absent from the npz
    are reported, never silently dropped -- a quietly missing input is how a fusion ends up
    measuring less than you think it does."""
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["__meta__"])) if "__meta__" in d.files else {}
    dur = float(meta.get("duration") or 0.0)
    n_out = int(round(dur * rate)) or max(len(d[f]) for f in d.files if f != "__meta__")

    want = fields or DEFAULT_FIELDS
    used, missing, per_field = [], [], {}
    for name in want:
        if name not in d.files:
            missing.append(name)
            continue
        nov = field_novelty(d[name], name)
        nov = _resample_to(_robust_norm(nov), _rate_of(name, meta), rate, n_out)
        per_field[name] = nov
        used.append(name)

    if not used:
        raise SystemExit(f"no usable fields in {npz_path.name} (wanted {want})")
    fused = np.mean(np.stack([per_field[k] for k in used]), axis=0)
    return fused, per_field, {"meta": meta, "used": used, "missing": missing,
                              "rate": rate, "n": n_out}


def pick_peaks(x: np.ndarray, rate: float, min_gap_s: float = 0.25, k: float = 1.0):
    """Local maxima above median + k*IQR, thinned by a refractory gap. Returns times (s)."""
    thr = np.median(x) + k * (np.subtract(*np.percentile(x, [75, 25])) or x.std() or 1.0)
    cand = np.flatnonzero((x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > thr)) + 1
    out, last = [], -1e9
    gap = min_gap_s * rate
    for i in cand:
        if i - last >= gap:
            out.append(i)
            last = i
    return np.asarray(out, dtype=float) / rate


def _match_f1(pred: np.ndarray, ref: np.ndarray, tol: float):
    """Greedy one-to-one match within +/- tol seconds -> (precision, recall, f1)."""
    if len(pred) == 0 or len(ref) == 0:
        return 0.0, 0.0, 0.0
    used = np.zeros(len(ref), dtype=bool)
    hits = 0
    for p in pred:
        j = int(np.argmin(np.abs(ref - p)))
        if not used[j] and abs(ref[j] - p) <= tol:
            used[j] = True
            hits += 1
    prec, rec = hits / len(pred), hits / len(ref)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


OFFSETS = np.arange(-0.30, 0.301, 0.02)


def best_offset(pred: np.ndarray, ref: np.ndarray, tol: float):
    """Best global time-shift and its F1.

    A PER-FIELD OFFSET IS REQUIRED, not a nicety (measured 2026-08-12). The fields come from
    different extractors with different window sizes and frame conventions, so each has its own
    effective latency against the beat grid -- the drum onset envelope wants ~+0.16 s where the
    fused stream wants ~-0.06 s. Uncalibrated, every stream scored AT CHANCE and looked like a
    null result; calibrated, they all carry real beat-aligned articulation. Worse, uncalibrated
    fusion averages streams that disagree about *when*, so the misalignment partly cancels --
    which is why naive fusion scored below its own best member.
    """
    scores = [(float(o), _match_f1(pred + o, ref, tol)[2]) for o in OFFSETS]
    o, f1 = max(scores, key=lambda t: t[1])
    return o, f1


def null_f1(pred: np.ndarray, ref: np.ndarray, tol: float, trials: int = 200, seed: int = 0,
            sweep: bool = False):
    """The number that decides whether any of this means anything.

    Shuffle the INTER-PEAK INTERVALS: identical peak count and identical interval distribution,
    but no relationship to the audio. If the real stream cannot beat this, its alignment with
    the bar grid is an artifact of how often it fires, not evidence that it found the bars.
    """
    if len(pred) < 3:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    gaps = np.diff(pred)
    scores = []
    for _ in range(trials):
        g = rng.permutation(gaps)
        fake = np.concatenate([[pred[0]], pred[0] + np.cumsum(g)])
        # IF THE REAL SCORE IS A MAX OVER OFFSETS, THE NULL MUST BE TOO. Taking the best of 31
        # shifts is a multiple-comparison win worth ~+0.11 F1 on its own; comparing a swept real
        # score against an unswept null would manufacture a "result" out of the search itself.
        scores.append(best_offset(fake, ref, tol)[1] if sweep
                      else _match_f1(fake, ref, tol)[2])
    return float(np.mean(scores)), float(np.percentile(scores, 95))


def cross_track_control(pred: np.ndarray, own_ref: np.ndarray, foreign_refs, tol: float):
    """The control that decides WHAT the alignment is evidence OF.

    A swept null rules out "peak density explains it", but not "any beat-periodic sequence
    phase-locks to any beat grid". So score the same peaks against OTHER tracks' grids: if a
    foreign grid scores as well as the track's own, we have only shown periodicity. Measured
    2026-08-12 across three tracks -- own grid beat the best foreign grid by +0.128, +0.160 and
    +0.277 F1, and the diagonal dominated every row, so the alignment is track-specific.
    """
    own = best_offset(pred, own_ref, tol)[1]
    frg = 0.0
    for r in foreign_refs:
        n = min(len(own_ref), len(r))
        if n > 2:
            frg = max(frg, best_offset(pred, r[:n], tol)[1])
    return own, frg


def load_reference(track: str, which: str):
    """Ground truth times. WHICH reference matters more than any parameter here.

    This stream detects ARTICULATIONS -- moments when something changes. Beats are
    articulations; DOWNbeats are a metrically-privileged sparse subset of them. Scoring an
    articulation detector against downbeats asks it to have found the metre, which it does not
    claim to do, and at ~5x the reference density that comparison is decided by peak count
    rather than by placement. Default is therefore the beat grid; downbeats remain available
    because bar-level agreement is a separate, harder question.
    """
    ext = {"beats": "BEATS_GRID", "downbeats": "DOWNBEATS"}[which]
    p = CORPUS_ROOT / track / f"{track}.{ext}"
    if not p.exists():
        return None
    return np.array([float(x) for x in p.read_text().split() if x.strip()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", help="track name (npz + DOWNBEATS looked up by name)")
    ap.add_argument("--npz", type=Path)
    ap.add_argument("--fields", help="comma-separated field list (default: event-like fields)")
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--min-gap", type=float, default=0.25, help="peak refractory, seconds")
    ap.add_argument("--tol", type=float, default=0.07, help="match tolerance vs downbeats, s")
    ap.add_argument("--validate", action="store_true", help="score against the grid + null")
    ap.add_argument("--ref", choices=["beats", "downbeats"], default="beats",
                    help="reference grid; beats is the honest default (see load_reference)")
    ap.add_argument("--per-field", action="store_true", help="also score each field alone")
    ap.add_argument("--calibrate", action="store_true", default=True,
                    help="fit a per-field global offset (REQUIRED for a fair read; the null is "
                         "swept identically so the search cannot manufacture a result)")
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    ap.add_argument("--null-trials", type=int, default=60)
    a = ap.parse_args()

    npz = a.npz or (TS_ROOT / f"{a.track}.TIMESERIES.npz")
    if not npz.exists():
        raise SystemExit(f"no timeseries at {npz}")
    fields = a.fields.split(",") if a.fields else None
    fused, per_field, info = build_stream(npz, fields, a.rate)

    print(f"[stream] {npz.name}")
    print(f"  duration {info['meta'].get('duration', 0):.1f}s @ {a.rate} Hz -> {info['n']} frames")
    print(f"  fields used ({len(info['used'])}): {', '.join(info['used'])}")
    if info["missing"]:
        print(f"  fields MISSING ({len(info['missing'])}): {', '.join(info['missing'])}")

    peaks = pick_peaks(fused, a.rate, a.min_gap)
    print(f"  articulation points: {len(peaks)}  "
          f"({len(peaks) / max(info['meta'].get('duration', 1), 1) * 60:.1f}/min)")

    if not a.validate:
        return 0
    track = a.track or npz.name.replace(".TIMESERIES.npz", "")
    ref = load_reference(track, a.ref)
    if ref is None:
        print(f"  [validate] no {a.ref} grid for this track -- cannot score")
        return 1

    def report(label, pk):
        if a.calibrate:
            off, f1 = best_offset(pk, ref, a.tol)
            p, r, _ = _match_f1(pk + off, ref, a.tol)
            nm, n95 = null_f1(pk, ref, a.tol, trials=a.null_trials, sweep=True)
            extra = f"off {off:+.2f}s "
        else:
            off = 0.0
            p, r, f1 = _match_f1(pk, ref, a.tol)
            nm, n95 = null_f1(pk, ref, a.tol, trials=a.null_trials, sweep=False)
            extra = ""
        verdict = "REAL SIGNAL" if f1 > n95 else "at/below null -- NOT evidence"
        print(f"  {label:28s} n={len(pk):5d}  {extra}P={p:.3f} R={r:.3f} F1={f1:.3f} | "
              f"null mean {nm:.3f} p95 {n95:.3f}  -> {verdict}")

    span = float(ref[-1] - ref[0]) if len(ref) > 1 else 1.0
    print(f"  [validate] ref={a.ref}: {len(ref)} points, median spacing "
          f"{np.median(np.diff(ref)):.3f}s, tol +/-{a.tol}s")
    report("FUSED", peaks)
    if a.per_field:
        for name in info["used"]:
            report(name, pick_peaks(per_field[name], a.rate, a.min_gap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
