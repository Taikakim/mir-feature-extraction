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
import glob
import json
import os
import re
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import numpy as np

GOA = "/run/media/kim/Mantu/ai-music/Goa Dataset"
ARCHIVE = "/run/media/kim/Mantu/goa_archive_extracted"
EXT = {".flac", ".mp3", ".ogg", ".m4a", ".wav", ".aiff", ".wma"}
FP_SECONDS = 120
# Bit agreement at the best offset. Calibrated 2026-10-25 on the root Tidal tracks (tags known good): pairs at >= 0.90
# agree by name 97% of the time, pairs under 0.80 almost never (unrelated 4/4 tracks still reach 0.6-0.8), and
# 0.88-0.97 also holds some unrelated pairs. So: >= SURE is the same recording, MATCH_SIM..SURE wants a listen.
MATCH_SIM = 0.88
SURE = 0.97
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


def cached_scan(paths, cache, workers, cache_only=False):
    """{path: {tags..., fp: [...]}}; appends new entries to the jsonl cache (cache_only: report on what is cached)."""
    have = {}
    if os.path.exists(cache):
        with open(cache) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:  # a line still being written by a concurrent scan
                    continue
                have[d["path"]] = d
    todo = [] if cache_only else [p for p in paths if p not in have]
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
    # fold accents and stylised letters so 'AΣIA'='asia', 'Tufáan'='tufaan', '7ème'='7eme'; drop apostrophes ('occams')
    s = s.replace("Σ", "S").replace("σ", "s")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"['’`´]", "", s.lower())
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

    def query(self, fp, top=10):
        """All of the `top` most-voted (ref, offset) candidates as (ref, offset, bit_sim), best first."""
        q = np.asarray(fp, dtype=np.uint32)
        lo = np.searchsorted(self.v, q, "left")
        hi = np.searchsorted(self.v, q, "right")
        keep = (hi > lo) & ~np.isin(q, list(self.common))
        if not keep.any():
            return []
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
        return sorted(out, key=lambda t: -t[2])


def name_parts(artist):
    """'Amanite FX & Prana feat. X' -> ['amanite fx', 'prana', 'x'] (normalised)."""
    return [p for p in (norm(x) for x in re.split(r"\s*(?:&|,|/|\bfeat\.?|\bft\.?|\bvs\.?|\band\b)\s*", artist, flags=re.I))
            if len(p) >= 2]


def fuzzy_share(phrase, ref_str):
    """Fraction of the phrase's words found in ref_str, a word counting when it is a one-typo variant of a ref word
    ('alphanut'~'alphanaut', 'subcouds'~'subclouds'); 1.0 also when the phrase is in ref_str with spaces removed
    ('tau factor'~'taufactor')."""
    toks = phrase.split()
    if not toks:
        return 0.0
    if phrase.replace(" ", "") in ref_str.replace(" ", ""):
        return 1.0
    rt = set(ref_str.split())
    hit = sum(1 for t in toks
              if t in rt or (len(t) >= 4 and any(SequenceMatcher(None, t, x).ratio() >= 0.8 for x in rt)))
    return hit / len(toks)


def in_corpus(s, corpus):
    return bool(s) and f" {s} " in corpus


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
    ap.add_argument("--separated", default="/run/media/kim/Mantu/ai-music/Goa_Separated")
    ap.add_argument("--latents", default="/run/media/kim/Kosmos/latents/latents_sa3")
    ap.add_argument("--cache-only", action="store_true", help="no new fingerprinting; report from the jsonl caches")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    goa = cached_scan(sorted(audio_files(a.goa)), os.path.join(a.out, "goa_dataset_scan.jsonl"), a.workers,
                      a.cache_only)
    arc = cached_scan(sorted(audio_files(a.archive)), os.path.join(a.out, "archive_scan.jsonl"), a.workers, a.cache_only)
    refs = [d for d in arc.values() if len(d["fp"]) >= 80]
    print(f"[audit] indexing {len(refs)} archive fingerprints", flush=True)
    idx = FpIndex(refs)
    # name check: the archive covers pre-2000 goa almost completely, so a Picard artist/title that appears nowhere in
    # its folder names, filenames or tags is suspect even without an audio match
    corpus = " " + " | ".join(norm(os.path.relpath(d["path"], a.archive).replace(os.sep, " ")) + " " +
                              norm(f"{d['artist']} {d['title']} {d['album']}") for d in arc.values()) + " "

    rows = []
    for n, (p, d) in enumerate(goa.items(), 1):
        rel = os.path.relpath(p, a.goa)
        part = rel.split(os.sep)[0] if os.sep in rel else "root"
        stem = os.path.splitext(os.path.basename(p))[0]
        folder = os.path.basename(os.path.dirname(p)) if part != "root" else ""
        name_from = "tags"
        if not (d["artist"] or d["title"]) and " - " in stem:
            # no readable tags (mutagen's easy mode reads none from AIFF): Picard's rename put its names in the filename
            d = dict(d, artist=stem.split(" - ", 1)[0].split(". ", 1)[-1], title=stem.split(" - ", 1)[1])
            name_from = "filename"
        tag_at = f"{d['artist']} - {d['title']}" if d["artist"] or d["title"] else ""
        row = dict(part=part, rel=rel, name_from=name_from, tag_artist=d["artist"], tag_title=d["title"],
                   tag_album=d["album"],
                   has_mbid=bool(d["mb_trackid"]), duration=d["duration"],
                   sim_tags_filename=text_sim(tag_at, stem) if tag_at else None,
                   sim_artist_folder=text_sim(d["artist"], folder) if d["artist"] and folder else None)
        parts = name_parts(d["artist"])
        title_core = norm(re.sub(r"[(\[].*?[)\]]", " ", d["title"]))
        row.update(artist_in_archive=any(in_corpus(p, corpus) for p in parts) if parts else None,
                   title_in_archive=in_corpus(title_core, corpus) if len(title_core) >= 3 else None)
        # every archive copy of the recording (compilations hold the same track several times, and a scene rip can
        # carry shifted filenames -- seen on The Green Compilation), so one agreeing copy confirms the tag
        cands = [c for c in (idx.query(d["fp"]) if len(d["fp"]) >= 80 else []) if c[2] >= MATCH_SIM]
        sure = [c for c in cands if c[2] >= SURE]
        copies = sure or cands
        scored = []
        for rid, _, sim in copies:
            r = refs[rid]
            ref_at = f"{r['artist']} - {r['title']}" if r["artist"] or r["title"] else ""
            ref_name = os.path.splitext(os.path.basename(r["path"]))[0]
            # artist and title are checked separately against the copy's tags + filename + release folder: a right title
            # under a wrong artist (Picard's 'Hawklords - Out of Phase' for Flippin Bixies) must not pass
            ref_str = " " + norm(f"{ref_at} {ref_name} {os.path.basename(os.path.dirname(r['path']))}") + " "
            artist_ok = any(fuzzy_share(p, ref_str) == 1.0 for p in parts) if parts else False
            title_ok = fuzzy_share(title_core, ref_str) >= TEXT_OK
            scored.append((int(artist_ok) + int(title_ok), sim, os.path.relpath(r["path"], a.archive), ref_at,
                           artist_ok, title_ok))
        scored.sort(key=lambda s: (-s[0], -s[1]))
        if scored:
            best = scored[0]
            row.update(match_sim=best[1], match_copies=len(scored), match_sure=bool(sure), copy_artist_ok=best[4],
                       copy_title_ok=best[5], match_rel=best[2], match_tags=best[3],
                       other_copies=" || ".join(s[2] for s in scored[1:4]))
        else:
            row.update(match_sim=None, match_copies=0, match_sure=False, copy_artist_ok=None, copy_title_ok=None,
                       match_rel="", match_tags="", other_copies="")
        row["verdict"] = verdict(row)
        rows.append(row)
        if n % 500 == 0:
            print(f"[audit] matched {n}/{len(goa)}", flush=True)

    # where a wrong name went downstream: Goa_Separated/<Artist - Title>/ and the latents_sa3 crops cut from it
    sep = set(os.listdir(a.separated)) if os.path.isdir(a.separated) else set()
    crops = {}
    for j in glob.glob(os.path.join(a.latents, "*.json")):
        try:
            st = json.load(open(j)).get("source_track")
        except Exception:
            continue
        crops[st] = crops.get(st, 0) + 1
    for r in rows:
        name = f"{r['tag_artist']} - {r['tag_title']}".replace("/", "_")
        r["in_goa_separated"] = name in sep
        r["latents_sa3_crops"] = crops.get(name, 0)

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
    if r["match_rel"] and r["copy_artist_ok"] and r["copy_title_ok"]:
        return "ok (an archive copy of the recording agrees)"
    if r["match_rel"]:
        if not r["match_sure"]:
            return "SUSPECT (listen): probable audio match names another track"
        if r["copy_title_ok"]:
            return "SUSPECT: same recording, title agrees but artist differs (alias?)"
        if r["copy_artist_ok"]:
            return "SUSPECT: same recording, artist agrees but title differs"
        return ("SUSPECT: same recording, archive names another artist and title"
                + ("; artist not in archive" if r["artist_in_archive"] is False else ""))
    if r["artist_in_archive"] is False:
        return "SUSPECT (name): artist not in archive"
    if r["title_in_archive"] is False:
        return "check (name): artist in archive, title not"
    weak = [x for x in (r["sim_tags_filename"], r["sim_artist_folder"]) if x is not None and x < TEXT_OK]
    return "SUSPECT (weak): tags disagree with filename/folder" if weak else "no audio match"


if __name__ == "__main__":
    sys.exit(main())
