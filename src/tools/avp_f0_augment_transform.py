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
import json
import os
from pathlib import Path

import numpy as np

VARIANTS_PITCH = {"pitch+1": 1, "pitch+2": 2, "pitch-1": -1, "pitch-2": -2}
VARIANTS_TEMPO = {"tempo+5", "tempo+10", "tempo-5", "tempo-10"}
F0_FIELDS = ["f0_other_ts", "f0_bass_ts", "f0_other_voiced_ts", "f0_bass_voiced_ts"]
F0_RATE = 100.0               # melodia grid, same as the source sidecar's base rate
F0_EXPANDED_VERSION = 2       # what the melody backfill denotes (whole_track_expanded.EXPANDED_VERSION)


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
        # UPDATE __meta__ TOO — writing arrays without announcing them makes the sidecar LIE.
        # Found by W 2026-08-18 auditing all 5035 Lehto sidecars: this wrote the four f0 arrays into
        # 1346 avp augmentation variants and never touched __meta__, so meta["fields"] listed 46
        # entries without them, field_rates had no f0, expanded_version stayed 1 and f0_source was
        # absent. A consumer following our OWN documented rule ("check field_rates/fields in
        # __meta__, never assume a field is present" — CLAUDE.md/MASTER) would conclude the melody
        # target does not exist on every augmented avp track while the arrays sit right there.
        # It failed silently rather than loudly because the f0 rate happens to fall back to
        # frame_rate=100 correctly. Nothing broke only because both live consumers happen to read
        # z.files rather than meta — that is luck, not design. merge_expanded already did this
        # correctly; this tool simply did not copy the pattern.
        announce_f0_in_meta(existing, sorted(new_fields), variant=vname)
        save_npz_atomic(variant_npz_path, existing)
        status[vname] = "written"
    return status


def announce_f0_in_meta(existing: dict, f0_fields, variant: str = None) -> bool:
    """Record newly-written f0 fields in the sidecar's __meta__ so it describes what it holds.

    Mirrors `whole_track_expanded.merge_expanded`: `fields` = every array actually present,
    `field_rates` gains the f0 entries (100 Hz), and the provenance keys live in the
    `expanded` SUB-DICT (`f0_source`, `expanded_version`) — not at top level, which is where
    the first version of this fix mistakenly put them.

    `f0_source` keeps merge_expanded's convention (the sorted list of `_voiced_ts` fields that
    exist) so a consumer's "is melody present" test works identically on source and variant;
    `f0_transform` records that these values were SCALED from the source, not re-extracted.

    Idempotent. Returns True if the meta changed. `existing` is mutated in place.
    """
    raw = existing.get("__meta__")
    if raw is None:
        return False
    try:
        meta = json.loads(str(raw))
        if isinstance(meta, str):          # doubly-encoded, seen on a few older sidecars
            meta = json.loads(meta)
    except Exception:
        return False
    before = json.dumps(meta, sort_keys=True)

    meta["fields"] = sorted(k for k in existing if k != "__meta__")
    rates = dict(meta.get("field_rates", {}))
    for f in f0_fields:
        rates[f] = F0_RATE
    meta["field_rates"] = rates

    expanded = dict(meta.get("expanded", {}))
    expanded["f0_source"] = sorted(f for f in f0_fields if f.endswith("_voiced_ts"))
    expanded["f0_transform"] = f"scaled-from-source:{variant}" if variant else "scaled-from-source"
    if expanded.get("expanded_version", 1) in (None, 1):
        expanded["expanded_version"] = F0_EXPANDED_VERSION
    meta["expanded"] = expanded
    # the first version of this fix wrote these two at top level; the canonical home is
    # meta["expanded"] (merge_expanded's `extra`), so drop the strays rather than keep two truths
    meta.pop("f0_source", None)
    meta.pop("expanded_version", None)

    if json.dumps(meta, sort_keys=True) == before:
        return False
    existing["__meta__"] = np.array(json.dumps(meta))
    return True


def save_npz_atomic(npz_path: Path, payload: dict) -> None:
    """Compressed + atomic, like merge_expanded (this used to be a bare uncompressed np.savez:
    a variant measured 10.7 MB against 8.5 MB for the same-shaped compressed source, and a
    crash mid-write left a truncated sidecar in place of a good one).

    The tmp name MUST end in .npz — np.savez appends the extension otherwise, leaving
    'x.tmp.npz' while os.replace looks for 'x.tmp' (pilot bug 2026-07-14).
    """
    npz_path = Path(npz_path)
    tmp = npz_path.parent / (npz_path.stem + ".tmp.npz")
    np.savez_compressed(str(tmp), **payload)
    os.replace(tmp, npz_path)


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
