#!/usr/bin/env python
"""avp_f0_augment_transform.py — SCALE-transform the AVP source f0 onto each bungee-augmented
variant (SAO D3 melody head, AVP iteration-2, Step 2 — F's scoped design, AGENT_DIALOGUE
2026-08-13 thread; built by CONTINUITY 2026-08-14, F not available).

WHY SCALE, NOT RECOMPUTE (W's settled reasoning, AGENT_DIALOGUE 2026-08-13 01:15): bungee
shifts the whole stem UNIFORMLY, so pitch+2's true melody IS source+2 semitones by definition;
any melodia deviation on a recompute is a fact about the extractor's salience under bungee's
spectral side-effects, not about the music. Ground-truth-consistency (scale) beats extractor-
consistency (recompute) for a head whose whole point is "make it higher."

PITCH variants (values only, frames unchanged — bungee pitch-shift preserves duration):
    f0_new = f0_source * 2**(semitones/12); voicing mask copied unchanged (same frame grid).

TEMPO variants (time axis remaps, values unchanged — bungee tempo-shift preserves pitch):
    f0_new = linear-resample(f0_source, T_variant), T_variant MEASURED from the variant's own
    sidecar — NEVER the nominal target_bpm pct (bungee's true bpm is per-track and can differ
    from the label, W's flag). Voicing mask resampled the same way (continuous [0,1] voiced-
    fraction, consistent with the crop resampler's masked-mean convention elsewhere).

Writes f0_other_ts / f0_bass_ts / f0_other_voiced_ts / f0_bass_voiced_ts into each variant's
OWN .TIMESERIES.npz (merged with existing fields) — no melodia re-run, no re-encode.
"""
import argparse
from pathlib import Path

import numpy as np

VARIANTS_PITCH = {"pitch+1": 1, "pitch+2": 2, "pitch-1": -1, "pitch-2": -2}
VARIANTS_TEMPO = {"tempo+5", "tempo+10", "tempo-5", "tempo-10"}
F0_FIELDS = ["f0_other_ts", "f0_bass_ts", "f0_other_voiced_ts", "f0_bass_voiced_ts"]


def _resample_1d(arr, n_out):
    n_in = len(arr)
    if n_in == n_out:
        return arr.astype(np.float32)
    src_x = np.linspace(0.0, 1.0, n_in, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, n_out, dtype=np.float64)
    return np.interp(dst_x, src_x, arr).astype(np.float32)


def transform_track(track_dir: Path, overwrite: bool = False) -> dict:
    """Transform one source track's f0 onto every augmentation variant beside it.

    Returns {variant_name: 'written'|'skipped'|'missing-variant'|'no-ref-field'
             |'unknown-variant-type'} for variants actually present; {} if the
    source itself has no f0 yet (Step 1 not done) or no augmentations/ dir exists.
    """
    track_dir = Path(track_dir)
    name = track_dir.name
    src_npz_path = track_dir / f"{name}.TIMESERIES.npz"
    aug_dir = track_dir / "augmentations"
    status = {}
    if not src_npz_path.exists() or not aug_dir.is_dir():
        return status
    with np.load(src_npz_path) as z:
        if not all(f in z.files for f in F0_FIELDS):
            return status
        src = {f: z[f].astype(np.float32) for f in F0_FIELDS}
    n_src = min(len(v) for v in src.values())
    src = {f: v[:n_src] for f, v in src.items()}

    for variant_dir in sorted(aug_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        vname = variant_dir.name
        variant_npz_path = variant_dir / f"{vname}.TIMESERIES.npz"
        if not variant_npz_path.exists():
            status[vname] = "missing-variant"
            continue
        with np.load(variant_npz_path) as z:
            existing = dict(z.items())
        if not overwrite and all(f in existing for f in F0_FIELDS):
            status[vname] = "skipped"
            continue
        # This variant's OWN legacy-field length is the ground truth for its duration —
        # never a nominal bpm pct (see module docstring).
        ref_key = next((k for k in ("rms_energy_bass_ts", "hpcp_ts") if k in existing), None)
        if ref_key is None:
            status[vname] = "no-ref-field"
            continue
        ref = existing[ref_key]
        n_var = ref.shape[-1] if ref.ndim > 1 else len(ref)

        if vname in VARIANTS_PITCH:
            factor = 2.0 ** (VARIANTS_PITCH[vname] / 12.0)
            new_fields = {
                "f0_other_ts": _resample_1d(src["f0_other_ts"] * factor, n_var),
                "f0_bass_ts": _resample_1d(src["f0_bass_ts"] * factor, n_var),
                "f0_other_voiced_ts": _resample_1d(src["f0_other_voiced_ts"], n_var),
                "f0_bass_voiced_ts": _resample_1d(src["f0_bass_voiced_ts"], n_var),
            }
        elif vname in VARIANTS_TEMPO:
            new_fields = {f: _resample_1d(src[f], n_var) for f in F0_FIELDS}
        else:
            status[vname] = "unknown-variant-type"
            continue

        existing.update(new_fields)
        np.savez(variant_npz_path, **existing)
        status[vname] = "written"
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="avp-analyzed root (per-track dirs with augmentations/ inside)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    root = Path(a.root)
    tracks = sorted(d for d in root.iterdir() if d.is_dir())
    if a.limit:
        tracks = tracks[: a.limit]
    totals = {}
    for i, td in enumerate(tracks):
        st = transform_track(td, overwrite=a.overwrite)
        for s in st.values():
            totals[s] = totals.get(s, 0) + 1
        print(f"[{i + 1}/{len(tracks)}] {td.name}: {st}", flush=True)
    print(f"\nDone. Totals: {totals}")


if __name__ == "__main__":
    main()
