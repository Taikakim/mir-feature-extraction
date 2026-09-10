#!/usr/bin/env python3
"""repair_timeseries_meta.py — make a .TIMESERIES.npz sidecar's __meta__ describe what it holds.

WHY. `avp_f0_augment_transform.py` wrote the four f0 melody arrays into every avp augmentation
variant with a bare `existing.update(...); np.savez(...)` and never touched `__meta__`. The
arrays are correct; the metadata says they do not exist — `fields` listed 46 entries without
them, `field_rates` had no f0 entry, and the `expanded` sub-dict still said version 1 with no
`f0_source`. Any consumer following mir's own documented rule ("check field_rates/fields in
__meta__, never assume a field is present") concludes the melody target is MISSING on every
augmented avp track. It fails silently; nothing broke only because the two live consumers read
`z.files` rather than the meta, which is luck, not design.

The writer is fixed (`announce_f0_in_meta` + `save_npz_atomic`), but re-running the transform
over already-written files would recompute and rewrite arrays that are already right. This
repairs `__meta__` IN PLACE from the arrays actually present instead.

WHAT IT REPAIRS (all derived from the file, nothing assumed):
  * `fields`        -> sorted list of every array in the npz (bar `__meta__`)
  * `field_rates`   -> f0_* entries added at 100 Hz (the melodia grid). Other missing entries
                       are left alone: base fields legitimately have NO field_rates entry and
                       fall back to top-level `frame_rate`.
  * `expanded`      -> `f0_source` (merge_expanded's convention: the sorted `_voiced_ts` names)
                       and `expanded_version` = 2, in the `expanded` SUB-DICT. Top-level
                       `f0_source`/`expanded_version` strays (written by the first, partial
                       version of the writer fix) are removed so there is one truth.

OPTIONAL, OFF BY DEFAULT (`--fix-vggish-rate`): `va_deam_ts` / `va_emomusic_ts` were stamped
at 16000/(96*160) = 1.041667 Hz, assuming a VGGish patchHopSize of 96. Essentia's
TensorflowPredictVGGish default patchHopSize is 93 (96 is the patch SIZE), so the true rate is
16000/(93*160) = 1.075269 Hz — a 3.1% error, which sits under the 5% warn threshold in
`crop_timeseries_resample._effective_rate` and is therefore accepted silently. The producer is
fixed; every sidecar written before that carries the wrong stated rate. This flag rewrites the
two entries. It is separate because it touches every sidecar in the store, not just the avp
variants, and rewriting a 37 GB store is a deliberate act.

Idempotent, dry-run by default.

    python src/tools/repair_timeseries_meta.py <root>              # report only
    python src/tools/repair_timeseries_meta.py <root> --apply
    python src/tools/repair_timeseries_meta.py <root> --apply --fix-vggish-rate
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

F0_FIELDS = ["f0_other_ts", "f0_bass_ts", "f0_other_voiced_ts", "f0_bass_voiced_ts"]
F0_RATE = 100.0
F0_EXPANDED_VERSION = 2

# essentia TensorflowPredictVGGish: patchHopSize default 93 (verified against the installed
# essentia 2.1-beta6-dev), mel frame hop 160 samples @ 16 kHz.
VGGISH_RATE_WRONG = 16000.0 / (96 * 160)     # 1.0416667 — what the producer used to stamp
VGGISH_RATE_TRUE = 16000.0 / (93 * 160)      # 1.0752688
VGGISH_FIELDS = ["va_deam_ts", "va_emomusic_ts"]


def repair_meta(arrays_present, meta: dict, fix_vggish_rate: bool = False):
    """Return (new_meta, [reasons]) — [] reasons means nothing needed changing."""
    meta = json.loads(json.dumps(meta))          # deep copy, keeps it JSON-clean
    reasons = []

    present = sorted(k for k in arrays_present if k != "__meta__")
    if list(meta.get("fields", [])) != present:
        missing = [k for k in present if k not in set(meta.get("fields", []))]
        phantom = [k for k in meta.get("fields", []) if k not in set(present)]
        meta["fields"] = present
        if phantom:
            reasons.append("fields-phantom")     # meta claims arrays the file does not have
        if missing:
            reasons.append("fields-f0-missing" if set(missing) <= set(F0_FIELDS)
                           else "fields-missing")
        if not missing and not phantom:
            reasons.append("fields-unsorted")

    f0_here = [f for f in F0_FIELDS if f in present]
    if f0_here:
        rates = dict(meta.get("field_rates", {}))
        if any(rates.get(f) != F0_RATE for f in f0_here):
            for f in f0_here:
                rates[f] = F0_RATE
            meta["field_rates"] = rates
            reasons.append("field_rates-f0")

        expanded = dict(meta.get("expanded", {}))
        want_src = sorted(f for f in f0_here if f.endswith("_voiced_ts"))
        if expanded.get("f0_source") != want_src:
            expanded["f0_source"] = want_src
            reasons.append("expanded.f0_source")
        if expanded.get("expanded_version", 1) in (None, 1):
            expanded["expanded_version"] = F0_EXPANDED_VERSION
            reasons.append("expanded.expanded_version")
        meta["expanded"] = expanded

    if "f0_source" in meta or "expanded_version" in meta:
        meta.pop("f0_source", None)
        meta.pop("expanded_version", None)
        reasons.append("toplevel-stray-removed")

    if fix_vggish_rate:
        rates = dict(meta.get("field_rates", {}))
        touched = False
        for f in VGGISH_FIELDS:
            if f in present and abs(float(rates.get(f, 0.0)) - VGGISH_RATE_WRONG) < 1e-6:
                rates[f] = VGGISH_RATE_TRUE
                touched = True
        if touched:
            meta["field_rates"] = rates
            reasons.append("vggish-rate")

    return meta, reasons


def repair_file(npz_path: Path, apply: bool = False, fix_vggish_rate: bool = False):
    """Returns (status, reasons). status: ok | repaired | would-repair | no-meta | error."""
    try:
        with np.load(str(npz_path), allow_pickle=False) as z:
            names = list(z.files)
            if "__meta__" not in names:
                return "no-meta", []
            meta = json.loads(str(z["__meta__"]))
            if isinstance(meta, str):
                meta = json.loads(meta)
            new_meta, reasons = repair_meta(names, meta, fix_vggish_rate)
            if not reasons:
                return "ok", []
            if not apply:
                return "would-repair", reasons
            payload = {k: z[k] for k in names if k != "__meta__"}
    except Exception as e:                       # noqa: BLE001 — report, never abort the walk
        return f"error:{type(e).__name__}:{e}", []

    payload["__meta__"] = np.array(json.dumps(new_meta))
    tmp = npz_path.parent / (npz_path.stem + ".tmp.npz")   # must end in .npz (np.savez appends)
    np.savez_compressed(str(tmp), **payload)
    os.replace(tmp, npz_path)
    return "repaired", reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory walked recursively for *.TIMESERIES.npz")
    ap.add_argument("--apply", action="store_true",
                    help="actually write (default: report only)")
    ap.add_argument("--fix-vggish-rate", action="store_true",
                    help="also correct va_deam_ts/va_emomusic_ts field_rates 1.041667 -> 1.075269")
    ap.add_argument("--variants-only", action="store_true",
                    help="only sidecars under an augmentations/ dir (the avp bug's blast radius)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", help="print every non-ok file")
    a = ap.parse_args()

    files = sorted(Path(a.root).rglob("*.TIMESERIES.npz"))
    if a.variants_only:
        files = [f for f in files if "augmentations" in f.parts]
    if a.limit:
        files = files[: a.limit]

    status_counts, reason_counts = Counter(), Counter()
    for i, f in enumerate(files):
        st, reasons = repair_file(f, apply=a.apply, fix_vggish_rate=a.fix_vggish_rate)
        status_counts[st.split(":")[0] if st.startswith("error") else st] += 1
        for r in reasons:
            reason_counts[r] += 1
        if a.verbose and st != "ok":
            print(f"  {st:14s} {','.join(reasons):50s} {f}", flush=True)
        if (i + 1) % 200 == 0:
            print(f"[{i + 1}/{len(files)}] {dict(status_counts)}", flush=True)

    print(f"\n{'APPLIED' if a.apply else 'DRY RUN'} — scanned {len(files)} sidecars under {a.root}")
    for k, v in sorted(status_counts.items()):
        print(f"  {k:14s} {v}")
    if reason_counts:
        print("  reasons:")
        for k, v in sorted(reason_counts.items()):
            print(f"    {k:26s} {v}")
    if not a.apply and status_counts.get("would-repair"):
        print("\n  re-run with --apply to write these.")


if __name__ == "__main__":
    main()
