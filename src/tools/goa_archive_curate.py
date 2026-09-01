#!/usr/bin/env python3
"""goa_archive_curate.py — Stage 1 curation for the goa archive (buildout plan,
CONTINUITY 2026-07-30). Kim's directives (verbatim intent, 2026-07-29):
  - SIMILARITY clustering, NOT bit-dedup: same track across compilations with different
    mastering = REAL augmentation to KEEP; drop only true near-identical.
  - Flag overlap with the existing Goa_Separated corpus (don't silently drop).
  - Quality-gate the ENCODE stage on tier >= B, FLAC preferred.

Method: MAEST 768-d whole-track embeddings (index.jsonl from goa_archive_mir.py),
cosine graph -> connected components at --variant-cos (same-work clusters); inside a
cluster, a tighter union-find at --dup-cos marks true near-identicals (keep the best:
FLAC > tier > bitrate; rest role=duplicate_dropped). Everything else in the cluster is
role=master_variant (KEPT). Goa_Separated overlap via mean-pooled maest_embed_ts from
the Lehto whole-track npz (cached after first run).

Idempotent over a GROWING index — run on the partial index anytime; final run after the
MIR pass completes. Outputs: curated.jsonl (one row/track), clusters_summary.json.

RUN (mir venv, CPU):
  mir/bin/python src/tools/goa_archive_curate.py \
      --features /run/media/kim/9a410a1d-a4a8-4faf-8298-bcaa2576ea9d/goa_archive_features
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TIER_RANK = {"A_near_lossless": 0, "B_256-320k": 1, "C_~192k": 2, "D_<=128k_lossy": 3, "unknown": 4}


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cos_pairs(X, thresh, chunk=2000):
    """Yield (i, j, cos) for i<j with cos >= thresh. X must be L2-normalized [N,768]."""
    n = X.shape[0]
    for s in range(0, n, chunk):
        G = X[s:s + chunk] @ X.T                      # [chunk, N]
        for li in range(G.shape[0]):
            i = s + li
            js = np.nonzero(G[li] >= thresh)[0]
            for j in js:
                if j > i:
                    yield i, int(j), float(G[li, j])


def pool_goa_sep(ts_dir: Path, cache: Path):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["M"], list(z["names"])
    files = sorted(ts_dir.glob("*.TIMESERIES.npz"))
    M, names = [], []
    for k, f in enumerate(files):
        try:
            with np.load(f, allow_pickle=True) as z:
                if "maest_embed_ts" not in z.files:
                    continue
                v = np.asarray(z["maest_embed_ts"], dtype=np.float32)
                if v.ndim != 2 or v.shape[0] < 2:
                    continue
                M.append(v.mean(axis=0))
                names.append(f.stem.replace(".TIMESERIES", ""))
        except Exception as e:
            print(f"[goa-sep] skip {f.name}: {e}", file=sys.stderr)
        if (k + 1) % 500 == 0:
            print(f"[goa-sep] pooled {k+1}/{len(files)}", flush=True)
    M = np.stack(M).astype(np.float32)
    np.savez(cache, M=M, names=np.array(names, dtype=object))
    print(f"[goa-sep] cached {M.shape} -> {cache}")
    return M, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True, help="goa_archive_features dir")
    ap.add_argument("--goa-sep-ts", type=Path,
                    default=Path("/run/media/kim/Lehto/timeseries"))
    ap.add_argument("--dup-cos", type=float, default=0.995,
                    help="true near-identical threshold (drop all but best)")
    ap.add_argument("--variant-cos", type=float, default=0.97,
                    help="same-work cluster threshold (members KEPT as master variants)")
    ap.add_argument("--goa-sep-cos", type=float, default=0.97)
    ap.add_argument("--min-dur", type=float, default=120.0)
    ap.add_argument("--max-dur", type=float, default=1500.0)
    args = ap.parse_args()

    # ---- load index (growing; skip malformed tails) + quality join on path
    rows = []
    for line in open(args.features / "index.jsonl"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    qual = {}
    qp = args.features / "quality" / "quality.jsonl"
    if qp.exists():
        for line in open(qp):
            try:
                r = json.loads(line)
                qual[r["path"]] = r
            except json.JSONDecodeError:
                continue
    print(f"[load] {len(rows)} index rows, {len(qual)} quality rows")

    no_embed = [r for r in rows if not (isinstance(r.get("maest_vec"), list) and len(r["maest_vec"]) == 768)]
    rows = [r for r in rows if isinstance(r.get("maest_vec"), list) and len(r["maest_vec"]) == 768]
    if no_embed:
        print(f"[load] {len(no_embed)} rows lack a 768-d maest_vec — emitted as role=no_embedding")

    X = np.array([r["maest_vec"] for r in rows], dtype=np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    # ---- same-work clustering + duplicate subgroups
    uf_var, uf_dup = UF(len(rows)), UF(len(rows))
    n_edges = 0
    for i, j, c in cos_pairs(X, args.variant_cos):
        uf_var.union(i, j)
        n_edges += 1
        if c >= args.dup_cos:
            uf_dup.union(i, j)
    print(f"[cluster] {n_edges} edges >= {args.variant_cos}")

    clusters = defaultdict(list)
    for i in range(len(rows)):
        clusters[uf_var.find(i)].append(i)
    dupgroups = defaultdict(list)
    for i in range(len(rows)):
        dupgroups[uf_dup.find(i)].append(i)

    def keep_rank(i):
        r = rows[i]
        q = qual.get(r["path"], {})
        flac = 0 if r["path"].lower().endswith(".flac") else 1
        return (flac, TIER_RANK.get(q.get("tier", "unknown"), 4), -q.get("bitrate_k", 0))

    role = {}
    for g in dupgroups.values():
        if len(g) == 1:
            continue
        best = min(g, key=keep_rank)
        for i in g:
            role[i] = "duplicate_best" if i == best else "duplicate_dropped"
    for cid, members in clusters.items():
        for i in members:
            if i not in role:
                role[i] = "master_variant" if len(members) > 1 else "unique"

    # ---- Goa_Separated overlap
    overlap = [None] * len(rows)
    if args.goa_sep_ts.exists():
        M, names = pool_goa_sep(args.goa_sep_ts, args.features / "goa_sep_maest_pooled.npz")
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        for s in range(0, len(rows), 2000):
            G = X[s:s + 2000] @ Mn.T
            am = G.argmax(axis=1)
            mx = G[np.arange(G.shape[0]), am]
            for li in range(G.shape[0]):
                if mx[li] >= args.goa_sep_cos:
                    overlap[s + li] = {"name": names[am[li]], "cos": round(float(mx[li]), 4)}
    else:
        print(f"[goa-sep] {args.goa_sep_ts} not mounted — overlap flags skipped this run")

    # ---- emit
    out = args.features / "curated.jsonl"
    stats = defaultdict(int)
    with open(out, "w") as f:
        for i, r in enumerate(rows):
            q = qual.get(r["path"], {})
            tier = q.get("tier", "unknown")
            dur_ok = args.min_dur <= r.get("dur_analyzed_s", 0) <= args.max_dur
            encode_ok = (role[i] != "duplicate_dropped" and dur_ok
                         and TIER_RANK.get(tier, 4) <= 1)
            rec = {"key": r["key"], "rel": r["rel"], "cluster": int(uf_var.find(i)),
                   "role": role[i], "tier": tier, "dur_s": round(r.get("dur_analyzed_s", 0), 1),
                   "goa_sep_overlap": overlap[i], "encode_ok": encode_ok}
            f.write(json.dumps(rec) + "\n")
            stats[role[i]] += 1
            stats["encode_ok"] += int(encode_ok)
            stats["goa_sep_overlap"] += int(overlap[i] is not None)
        for r in no_embed:
            q = qual.get(r["path"], {})
            tier = q.get("tier", "unknown")
            dur_ok = args.min_dur <= r.get("dur_analyzed_s", 0) <= args.max_dur
            f.write(json.dumps({"key": r["key"], "rel": r["rel"], "cluster": None,
                                "role": "no_embedding", "tier": tier,
                                "dur_s": round(r.get("dur_analyzed_s", 0), 1),
                                "goa_sep_overlap": None,
                                "encode_ok": dur_ok and TIER_RANK.get(tier, 4) <= 1}) + "\n")
            stats["no_embedding"] += 1

    multi = [m for m in clusters.values() if len(m) > 1]
    summary = {
        "n_tracks": len(rows), "roles": {k: v for k, v in stats.items() if k not in ("encode_ok", "goa_sep_overlap")},
        "encode_ok": stats["encode_ok"], "goa_sep_overlap": stats["goa_sep_overlap"],
        "clusters_multi": len(multi), "largest_cluster": max((len(m) for m in multi), default=1),
        "thresholds": {"variant_cos": args.variant_cos, "dup_cos": args.dup_cos,
                       "goa_sep_cos": args.goa_sep_cos},
        "note": "index may be partial (MIR still running) — rerun after completion; idempotent",
    }
    (args.features / "clusters_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[curate] -> {out}")


if __name__ == "__main__":
    main()
