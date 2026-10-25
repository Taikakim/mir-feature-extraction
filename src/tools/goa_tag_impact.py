#!/usr/bin/env python3
# Owner: WINTERMUTE (created 2026-10-25)
"""goa_tag_impact.py -- what the Goa_Separated naming errors did to the training stores, plus a duplicate check.

Reads the outputs of goa_tag_audit.py (same --out dir) and the training stores, writes:
  impact_table.csv      one row per Goa_Separated track whose name is suspect or has changed since separation:
                        why, the name it should probably have, and what carries the old name (latents_sa3 crops +
                        their prompt / track_metadata / release_year, the whole-track timeseries, the Flamingo caption
                        and the year its genre hint was given)
  duplicates_separated.csv   groups of Goa_Separated folders holding the same recording (fingerprint >= SURE), with
                        the latents_sa3 crops of each -- double weight, and train/val leakage under a name split
  overlap_sa3_bigset.csv     latents_sa3 tracks whose recording is also in latents_goa_bigset (under its archive name)
Nothing is modified. Names stay the KEYS of every store (crop sidecar, timeseries, captions join on them), so a fix
should rewrite metadata fields, not rename.

  mir/bin/python src/tools/goa_tag_impact.py --out /run/media/kim/Kosmos/goa_tag_audit
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goa_tag_audit import SURE, FpIndex  # noqa: E402

SEP = "/run/media/kim/Mantu/ai-music/Goa_Separated"
LATENTS = "/run/media/kim/Kosmos/latents/latents_sa3"
BIGSET = "/run/media/kim/Kosmos/latents/latents_goa_bigset"
TIMESERIES = "/run/media/kim/Kosmos/timeseries"
CAPTIONS = "/run/media/kim/Kosmos/captions/goa_src_captions/json"


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="goa_tag_audit.py output dir")
    a = ap.parse_args()

    audit = [r for r in csv.DictReader(open(os.path.join(a.out, "goa_tag_audit.csv"))) if r["part"] == "goa_separated"]
    single = {r["artist"]: r for r in csv.DictReader(open(os.path.join(a.out, "single_release_artists.csv")))}

    # latents_sa3: crops per source track, and what text they carry
    sa3 = defaultdict(list)
    for j in glob.glob(os.path.join(LATENTS, "[0-9]*.json")):
        if j.endswith(".TIMBRAL.json"):
            continue
        d = json.load(open(j))
        if d.get("source_track"):
            sa3[d["source_track"]].append(d)
    ts = {os.path.basename(p)[:-len(".TIMESERIES.npz")] for p in glob.glob(os.path.join(TIMESERIES, "*.TIMESERIES.npz"))}
    cap = {}
    for p in glob.glob(os.path.join(CAPTIONS, "*.json")):
        d = json.load(open(p))
        m = re.search(r"Goa_Separated/([^/]+)/", d.get("rel", ""))
        if m:
            y = re.search(r"release year: (\d{4})", d.get("genre_hint") or "")
            cap[m.group(1)] = y.group(1) if y else ""

    # ---- impact table
    rows = []
    for r in audit:
        name = r["rel"].split("/", 1)[1]
        why = []
        if r["renamed_since_separation"] == "True":
            why.append("renamed in Goa Dataset since separation")
        if r["verdict"].startswith("SUSPECT"):
            why.append(r["verdict"])
        s = single.get(r["tag_artist"])
        if s and s.get("lead"):
            why.append("single-release artist not in archive")
        if not why:
            continue
        crops = sa3.get(name, [])
        c0 = crops[0] if crops else {}
        # best guess at the right name: the Goa Dataset's current name when it is confirmed by the archive copy,
        # else the archive copy's name, else the current Goa Dataset name
        suggestion = r["goa_source_name_now"] or ""
        if r["match_rel"] and r["match_sure"] == "True" and not r["verdict"].startswith("ok"):
            suggestion = suggestion or ""
            arch = r["match_tags"] or os.path.splitext(os.path.basename(r["match_rel"]))[0]
            suggestion = f"{suggestion}  [archive: {arch}]" if suggestion else f"[archive: {arch}]"
        rows.append(dict(goa_separated=name, why=" ; ".join(why), name_now_in_goa_dataset=r["goa_source_name_now"],
                         goa_dataset_file=r["goa_source_rel"], archive_copy=r["match_rel"], suggested=suggestion,
                         latents_sa3_crops=len(crops), sa3_prompt=c0.get("prompt", ""),
                         sa3_meta=f"{c0.get('track_metadata_artist', '')} | {c0.get('track_metadata_title', '')} | "
                                  f"{c0.get('track_metadata_album', '')}" if c0 else "",
                         sa3_release_year=c0.get("release_year", ""), whole_track_timeseries=name in ts,
                         flamingo_caption=name in cap, caption_hint_year=cap.get(name, "")))
    rows.sort(key=lambda x: (-int(x["latents_sa3_crops"]), x["goa_separated"].lower()))
    write(os.path.join(a.out, "impact_table.csv"), rows)

    # ---- duplicates inside Goa_Separated
    scan = [d for d in load_jsonl(os.path.join(a.out, "goa_separated_scan.jsonl")) if len(d["fp"]) >= 80]
    idx = FpIndex(scan)
    parent = list(range(len(scan)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, d in enumerate(scan):
        for rid, _, sim in idx.query(d["fp"]):
            if rid != i and sim >= SURE:
                parent[find(rid)] = find(i)
    groups = defaultdict(list)
    for i in range(len(scan)):
        groups[find(i)].append(i)
    dup_rows = []
    for g, members in groups.items():
        if len(members) < 2:
            continue
        names = [os.path.basename(os.path.dirname(scan[i]["path"])) for i in members]
        for n, i in zip(names, members):
            dup_rows.append(dict(group=g, size=len(members), goa_separated=n,
                                 format=os.path.splitext(scan[i]["path"])[1], duration=scan[i]["duration"],
                                 latents_sa3_crops=len(sa3.get(n, [])),
                                 same_name_family=len({re.sub(r" \(\d+\)$", "", x) for x in names}) == 1))
    dup_rows.sort(key=lambda x: (x["group"], x["goa_separated"]))
    write(os.path.join(a.out, "duplicates_separated.csv"), dup_rows)

    # ---- latents_sa3 recordings that are also in the bigset (archive names)
    big = set()
    for j in glob.glob(os.path.join(BIGSET, "*.json")):
        rp = json.load(open(j)).get("relpath")
        if rp:
            big.add(rp)
    over = []
    for r in audit:
        name = r["rel"].split("/", 1)[1]
        if name in sa3 and r["match_sure"] == "True" and r["match_rel"]:
            copies = [r["match_rel"]] + [x for x in r["other_copies"].split(" || ") if x]
            inbig = [c for c in copies if c in big]
            if inbig:
                over.append(dict(goa_separated=name, latents_sa3_crops=len(sa3[name]), bigset_relpath=inbig[0],
                                 bigset_copies=len(inbig)))
    write(os.path.join(a.out, "overlap_sa3_bigset.csv"), over)

    n_dup_groups = len({r["group"] for r in dup_rows})
    extra = sum(r["size"] for r in {r["group"]: r for r in dup_rows}.values()) - n_dup_groups
    print(f"impact: {len(rows)} Goa_Separated tracks, {sum(1 for r in rows if r['latents_sa3_crops'])} of them in "
          f"latents_sa3 ({sum(r['latents_sa3_crops'] for r in rows)} crops)")
    print(f"duplicates: {n_dup_groups} groups, {extra} redundant folders; latents_sa3 crops in duplicate groups: "
          f"{sum(r['latents_sa3_crops'] for r in dup_rows)}")
    print(f"latents_sa3 tracks also in the bigset: {len(over)} ({sum(r['latents_sa3_crops'] for r in over)} crops) of "
          f"{len(sa3)} tracks")


if __name__ == "__main__":
    sys.exit(main())
