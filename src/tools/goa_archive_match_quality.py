#!/usr/bin/env python3
"""goa_archive_match_quality.py -- build a big-goa file list whose AUDIO QUALITY matches
the old Goa_Separated corpus, by dropping the worst-bandwidth tail.

Kim 2026-08-21: "create a list of big goa files where you drop away the worst stuff so
that the quality matches [old] goa".

WHY A THRESHOLD AND NOT RESAMPLING. Two ways to make one distribution match another:
drop the bad tail until the summary statistics line up, or stratified-resample to
reproduce the target's exact shape. The second also throws away GOOD files (to match the
target's share of mediocre ones), which is the opposite of what is wanted from a corpus
you are trying to grow. So this cuts a single threshold on spectral cutoff and reports
where the distributions land.

QUALITY MEASURE. Spectral cutoff (effective bandwidth), not header bitrate -- a "320k"
vintage rip is often a transcode with real content only to ~16 kHz. Produced by
goa_archive_quality.py for both corpora, the same code path on both, so the comparison
is like-for-like.

DUPLICATES ARE A SEPARATE AXIS and are applied first: goa_archive_curate.py already
marked 8,383 near-identical tracks `duplicate_dropped`. Those go regardless of quality.
`master_variant` (same work, different mastering) is KEPT by Kim's explicit directive --
different masterings are real augmentation.

OVERLAP WITH OLD GOA is reported and optionally excluded (--drop-overlap): tracks the
curator matched against Goa_Separated are already in the old training corpus, so keeping
them in a "new data" list double-counts them.

RUN (mir venv, CPU, seconds):
  mir/bin/python src/tools/goa_archive_match_quality.py \
     --big  <features>/quality/quality.jsonl \
     --curated <features>/curated.jsonl \
     --old  stats/goa_old_quality/quality.jsonl \
     --out  stats/goa_big_quality_matched
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

ARCHIVE_ROOT = "/run/media/kim/Mantu/goa_archive_extracted/"
KEEP_ROLES = {"unique", "duplicate_best", "master_variant"}
TIER_ORDER = ["A_near_lossless", "B_256-320k", "C_~192k", "D_<=128k_lossy", "unknown"]


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def describe(cuts, label, tiers=None):
    if not cuts:
        print(f"  {label:34s} (empty)")
        return {}
    q = np.percentile(cuts, [10, 25, 50, 75, 90])
    print(f"  {label:34s} n={len(cuts):6d}  mean {np.mean(cuts)/1000:5.2f} kHz  "
          f"median {q[2]/1000:5.2f}  p10 {q[0]/1000:5.2f}  p25 {q[1]/1000:5.2f}")
    if tiers:
        n = sum(tiers.values())
        share = "  ".join(f"{t.split('_')[0]}:{100*tiers.get(t,0)/n:4.1f}%" for t in TIER_ORDER[:4])
        print(f"  {'':34s} {share}")
    return {"n": len(cuts), "mean": float(np.mean(cuts)), "median": float(q[2]),
            "p10": float(q[0]), "p25": float(q[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", required=True, help="quality.jsonl for goa_archive_extracted")
    ap.add_argument("--curated", required=True, help="curated.jsonl (roles + overlap)")
    ap.add_argument("--old", required=True, help="quality.jsonl for Goa_Separated (full_mix only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--match-on", default="mean", choices=["mean", "median", "p10", "p25"],
                    help="which statistic of the old corpus the filtered big set must reach")
    ap.add_argument("--drop-overlap", action="store_true",
                    help="also drop tracks the curator matched against Goa_Separated")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- old goa: the target ----
    old = [r for r in load_jsonl(a.old) if r.get("cutoff_hz")]
    old_cuts = [r["cutoff_hz"] for r in old]
    old_tiers = Counter(r["tier"] for r in old)

    # ---- big goa: quality joined onto curation ----
    qual = {}
    for r in load_jsonl(a.big):
        p = r["path"]
        qual[p[len(ARCHIVE_ROOT):] if p.startswith(ARCHIVE_ROOT) else p] = r
    cur = load_jsonl(a.curated)

    kept, dropped_dup, dropped_ov, no_q = [], 0, 0, 0
    for c in cur:
        if c.get("role") not in KEEP_ROLES:
            dropped_dup += 1
            continue
        if a.drop_overlap and c.get("goa_sep_overlap"):
            dropped_ov += 1
            continue
        q = qual.get(c.get("rel"))
        if not q or not q.get("cutoff_hz"):
            no_q += 1
            continue
        kept.append({**c, "cutoff_hz": q["cutoff_hz"], "bitrate_k": q.get("bitrate_k"),
                     "codec": q.get("codec"), "sr": q.get("sr"), "dur": q.get("dur")})

    print("=" * 92)
    print("BIG GOA -> matched to OLD GOA on spectral cutoff (effective bandwidth)")
    print("=" * 92)
    print(f"curated rows            : {len(cur)}")
    print(f"  dropped duplicate*    : {dropped_dup}")
    if a.drop_overlap:
        print(f"  dropped goa_sep overlap: {dropped_ov}")
    if no_q:
        print(f"  no cutoff measurement : {no_q}")
    print(f"  candidate pool        : {len(kept)}")
    print()

    tgt = describe(old_cuts, "OLD GOA (Goa_Separated, target)", old_tiers)
    describe([k["cutoff_hz"] for k in kept], "BIG GOA (dedup'd, unfiltered)",
             Counter(k["tier"] for k in kept))
    print()

    # ---- threshold ladder ----
    print("THRESHOLD LADDER -- drop everything below the cutoff:")
    print(f"  {'min cutoff':>12s} {'kept':>7s} {'% of pool':>10s} {'mean kHz':>9s} {'median':>8s}   vs old mean")
    ladder = [0, 14000, 15000, 16000, 17000, 17500, 18000, 18500, 19000, 19500,
              20000, 20500, 21000, 21500, 22000]
    rows_l = []
    for t in ladder:
        sel = [k["cutoff_hz"] for k in kept if k["cutoff_hz"] >= t]
        if not sel:
            continue
        m = np.mean(sel)
        rows_l.append((t, len(sel), m))
        flag = "  <-- reaches target" if m >= tgt[a.match_on] else ""
        print(f"  {t/1000:11.1f}k {len(sel):7d} {100*len(sel)/len(kept):9.1f}% "
              f"{m/1000:8.2f} {np.median(sel)/1000:8.2f}   {m-tgt[a.match_on]:+8.0f} Hz{flag}")

    # ---- pick the LOWEST threshold that reaches the target (keeps the most files) ----
    chosen, reached = None, False
    for t, n, m in rows_l:
        if m >= tgt[a.match_on]:
            chosen, reached = t, True
            break
    if chosen is None:
        chosen = ladder[-1]
    final = [k for k in kept if k["cutoff_hz"] >= chosen]
    print()
    if reached:
        print(f"CHOSEN THRESHOLD: cutoff >= {chosen/1000:.1f} kHz "
              f"(lowest that matches old-goa {a.match_on}, so it keeps the most files)")
    else:
        # Say so explicitly. Falling back to the strictest rung and still printing
        # "matches" would report a match that never happened.
        print(f"CHOSEN THRESHOLD: cutoff >= {chosen/1000:.1f} kHz -- NO threshold on this "
              f"ladder reaches the old-goa {a.match_on} ({tgt[a.match_on]/1000:.2f} kHz); "
              f"this is the strictest rung and it still falls short.")
    describe([k["cutoff_hz"] for k in final], "BIG GOA (filtered)",
             Counter(k["tier"] for k in final))
    describe(old_cuts, "OLD GOA (target)", old_tiers)

    # ---- write ----
    lst = os.path.join(a.out, "big_goa_quality_matched.txt")
    js = os.path.join(a.out, "big_goa_quality_matched.jsonl")
    with open(lst, "w") as f:
        for k in sorted(final, key=lambda x: x["rel"]):
            f.write(ARCHIVE_ROOT + k["rel"] + "\n")
    with open(js, "w") as f:
        for k in sorted(final, key=lambda x: x["rel"]):
            f.write(json.dumps(k) + "\n")
    meta = {
        "created": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "purpose": "Big-goa (goa_archive_extracted) file list filtered so its audio-quality "
                   "distribution matches the old Goa_Separated corpus.",
        "method": {
            "quality_measure": "spectral cutoff (effective bandwidth), goa_archive_quality.py",
            "duplicates": "goa_archive_curate.py roles; duplicate_dropped removed, "
                          "master_variant KEPT (different masterings = real augmentation)",
            "drop_overlap_with_old_goa": bool(a.drop_overlap),
            "matched_on": a.match_on,
            "threshold_hz": chosen,
        },
        "counts": {"curated_rows": len(cur), "after_dedup": len(kept), "final": len(final)},
        "old_goa_target": tgt,
        "threshold_reached_target": bool(reached),
        "big_goa_filtered": {"n": len(final),
                             "mean": float(np.mean([k["cutoff_hz"] for k in final])),
                             "median": float(np.median([k["cutoff_hz"] for k in final]))},
        "tier_shares_final": dict(Counter(k["tier"] for k in final)),
        "kim_feedback": None,
    }
    with open(os.path.join(a.out, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print()
    print(f"wrote {len(final)} paths -> {lst}")
    print(f"      per-track detail  -> {js}")


if __name__ == "__main__":
    main()
