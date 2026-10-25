#!/usr/bin/env python3
# Owner: WINTERMUTE (created 2026-10-25)
"""goa_tag_audit.py -- find Goa Dataset files whose (Picard-written) artist/title tags name the wrong track.

WHY: the untagged own rips in `ai-music/Goa Dataset/goa` were enriched with MusicBrainz Picard, which mis-identified
some of them (e.g. tracks tagged as Terry Pratchett audiobooks) and then RENAMED the files to the wrong tags, so the
filename cannot be checked against the tags. The original names survive in a different copy: most of these tracks
are also in the scene-release archive `goa_archive_extracted` (the source of the 12k-crop goa bigset), whose
filenames and ID3 tags were never touched by Picard.

HOW: chromaprint (fpcalc -raw, first 120 s) on both sets; each Goa Dataset file is matched to its best archive copy by
offset-voting over exact fingerprint words, then scored by bit agreement at that offset (same recording ~0.85+,
unrelated ~0.5). For matched pairs the Goa Dataset tags are compared with the archive copy's tags and filename
(normalised text similarity). Unmatched files still get the weaker in-set checks: tags vs filename, tags vs the
artist folder name, and no tags.

Stages are cached (tags + fingerprints as jsonl) so a rerun only fingerprints new files.
  mir/bin/python src/tools/goa_tag_audit.py --out /run/media/kim/Kosmos/goa_tag_audit --workers 16
Outputs in --out: goa_tag_audit.csv (every Goa Dataset file, with verdict), suspects.csv (the ones to fix).
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import numpy as np

GOA = "/run/media/kim/Mantu/ai-music/Goa Dataset"
ARCHIVE = "/run/media/kim/Mantu/goa_archive_extracted"
EXT = {".flac", ".mp3", ".ogg", ".m4a", ".wav", ".aiff", ".wma"}
FP_SECONDS = 120
MATCH_SIM = 0.70        # bit agreement at the best offset; unrelated audio sits near 0.5
TEXT_OK = 0.60          # normalised artist+title similarity below this = disagreement


def audio_files(root):
    for r, _, fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in EXT:
                yield os.path.join(r, f)


def read_tags(path):
    import mutagen
    try:
        m = mutagen.File(path, easy=True)
    except Exception:
        m = None
    t = dict(m.tags) if m is not None and m.tags else {}
    g = lambda k: str((t.get(k) or [""])[0])  # noqa: E731
    return dict(artist=g("artist"), title=g("title"), album=g("album"), albumartist=g("albumartist"),
                mb_trackid=g("musicbrainz_trackid"),
                duration=round(m.info.length, 1) if m is not None and getattr(m, "info", None) else None)


def fingerprint(path):
    try:
        r = subprocess.run(["fpcalc", "-raw", "-json", "-length", str(FP_SECONDS), path],
                           capture_output=True, text=True, timeout=120)
        d = json.loads(r.stdout)
        return d.get("fingerprint") or []
    except Exception:
        return []


def cached_scan(paths, cache, workers):
    """{path: {tags..., fp: [...]}}; appends new entries to the jsonl cache."""
    have = {}
    if os.path.exists(cache):
        with open(cache) as f:
            for line in f:
                d = json.loads(line)
                have[d["path"]] = d
    todo = [p for p in paths if p not in have]
    print(f"[audit] {os.path.basename(cache)}: {len(have)} cached, {len(todo)} to scan", flush=True)

    def one(p):
        d = read_tags(p)
        d["path"] = p
        d["fp"] = fingerprint(p)
        return d

    with open(cache, "a") as f, ThreadPoolExecutor(workers) as ex:
        for i, d in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            have[d["path"]] = d
            if i % 500 == 0:
                print(f"[audit]   {i}/{len(todo)}", flush=True)
    return {p: have[p] for p in paths if p in have}


def norm(s):
    s = s.lower()
    s = re.sub(r"\.(mp3|flac|ogg|m4a|wav|aiff|wma)$", "", s)
    s = re.sub(r"\(([a-z_]+_p|[a-z]{2,5})\)", " ", s)        # scene group tags like (montauk_p)
    s = re.sub(r"-(19|20)\d\d-[a-z0-9]{2,6}$", " ", s)        # trailing -1998-nbd
    s = re.sub(r"^\s*(cd\d+\s*)?[0-9]{1,4}[a-z]?[\s._)-]+", " ", s)  # leading track number
    s = s.replace("_", " ").replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def text_sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return None
    ta, tb = set(a.split()), set(b.split())
    tok = len(ta & tb) / max(1, min(len(ta), len(tb)))          # token containment (order/extra words tolerant)
    return round(max(tok, SequenceMatcher(None, a, b).ratio()), 3)


class FpIndex:
    """Inverted index of reference fingerprint words -> (ref, position), for offset voting."""

    def __init__(self, refs):
        self.refs = refs
        vals, ids, pos = [], [], []
        for i, r in enumerate(refs):
            fp = np.asarray(r["fp"], dtype=np.uint32)
            vals.append(fp)
            ids.append(np.full(len(fp), i, np.int32))
            pos.append(np.arange(len(fp), dtype=np.int32))
        v = np.concatenate(vals)
        order = np.argsort(v, kind="stable")
        self.v, self.id, self.pos = v[order], np.concatenate(ids)[order], np.concatenate(pos)[order]
        u, c = np.unique(self.v, return_counts=True)
        self.common = set(u[c > 200].tolist())                  # silence / generic words carry no identity

    def query(self, fp, top=5):
        q = np.asarray(fp, dtype=np.uint32)
        lo = np.searchsorted(self.v, q, "left")
        hi = np.searchsorted(self.v, q, "right")
        keep = (hi > lo) & ~np.isin(q, list(self.common))
        if not keep.any():
            return None
        qi, lo, hi = np.nonzero(keep)[0], lo[keep], hi[keep]
        cnt = hi - lo
        post = np.repeat(lo - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt) + np.arange(cnt.sum())
        rid = self.id[post].astype(np.int64)
        off = self.pos[post].astype(np.int64) - np.repeat(qi, cnt)
        keys, votes = np.unique(rid * 1_000_000 + (off + 500_000), return_counts=True)
        out = []
        for k in keys[np.argsort(-votes)[:top]]:
            r, o = int(k // 1_000_000), int(k % 1_000_000) - 500_000
            out.append((r, o, bit_sim(q, np.asarray(self.refs[r]["fp"], dtype=np.uint32), o)))
        return max(out, key=lambda t: t[2]) if out else None


def bit_sim(q, r, off):
    """Fraction of agreeing bits where q[i] aligns with r[i + off] (needs >= 80 overlapping words, ~10 s)."""
    if off >= 0:
        a, b = q, r[off:]
    else:
        a, b = q[-off:], r
    n = min(len(a), len(b))
    if n < 80:
        return 0.0
    x = np.bitwise_xor(a[:n], b[:n])
    diff = np.unpackbits(x.view(np.uint8)).sum()
    return round(1.0 - diff / (32.0 * n), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goa", default=GOA)
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    goa = cached_scan(sorted(audio_files(a.goa)), os.path.join(a.out, "goa_dataset_scan.jsonl"), a.workers)
    arc = cached_scan(sorted(audio_files(a.archive)), os.path.join(a.out, "archive_scan.jsonl"), a.workers)
    refs = [d for d in arc.values() if len(d["fp"]) >= 80]
    print(f"[audit] indexing {len(refs)} archive fingerprints", flush=True)
    idx = FpIndex(refs)

    rows = []
    for n, (p, d) in enumerate(goa.items(), 1):
        rel = os.path.relpath(p, a.goa)
        part = rel.split(os.sep)[0] if os.sep in rel else "root"
        stem = os.path.splitext(os.path.basename(p))[0]
        folder = os.path.basename(os.path.dirname(p)) if part != "root" else ""
        tag_at = f"{d['artist']} - {d['title']}" if d["artist"] or d["title"] else ""
        row = dict(part=part, rel=rel, tag_artist=d["artist"], tag_title=d["title"], tag_album=d["album"],
                   has_mbid=bool(d["mb_trackid"]), duration=d["duration"],
                   sim_tags_filename=text_sim(tag_at, stem) if tag_at else None,
                   sim_artist_folder=text_sim(d["artist"], folder) if d["artist"] and folder else None)
        m = idx.query(d["fp"]) if len(d["fp"]) >= 80 else None
        if m and m[2] >= MATCH_SIM:
            r = refs[m[0]]
            ref_at = f"{r['artist']} - {r['title']}" if r["artist"] or r["title"] else ""
            ref_name = os.path.splitext(os.path.basename(r["path"]))[0]
            row.update(match_sim=m[2], match_rel=os.path.relpath(r["path"], a.archive), match_tags=ref_at,
                       sim_tags_vs_match_tags=text_sim(tag_at, ref_at) if tag_at and ref_at else None,
                       sim_tags_vs_match_filename=text_sim(tag_at, ref_name) if tag_at else None)
        else:
            row.update(match_sim=m[2] if m else None, match_rel="", match_tags="",
                       sim_tags_vs_match_tags=None, sim_tags_vs_match_filename=None)
        row["verdict"] = verdict(row)
        rows.append(row)
        if n % 500 == 0:
            print(f"[audit] matched {n}/{len(goa)}", flush=True)

    cols = list(rows[0].keys())
    for name, sel in (("goa_tag_audit.csv", rows), ("suspects.csv", [r for r in rows if r["verdict"].startswith("SUSPECT")])):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(sel)
    from collections import Counter
    for part in ("root", "goa", "goa-unsorted"):
        print(part, dict(Counter(r["verdict"] for r in rows if r["part"] == part)), flush=True)


def verdict(r):
    if not (r["tag_artist"] or r["tag_title"]):
        return "untagged"
    if r["match_rel"]:
        best = max(x for x in (r["sim_tags_vs_match_tags"], r["sim_tags_vs_match_filename"], 0) if x is not None)
        return "ok (audio match agrees)" if best >= TEXT_OK else "SUSPECT: audio match names another track"
    weak = [x for x in (r["sim_tags_filename"], r["sim_artist_folder"]) if x is not None and x < TEXT_OK]
    return "SUSPECT (weak): tags disagree with filename/folder" if weak else "no audio match"


if __name__ == "__main__":
    sys.exit(main())
