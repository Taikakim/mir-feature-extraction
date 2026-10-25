#!/usr/bin/env python3
# Owner: WINTERMUTE (created 2026-10-25)
# Last edited: WINTERMUTE 2026-10-25 -- NOT_SUSPECT: drop Aavepyörä and 01-GGE from the artist batches
"""goa_research_batches.py -- JSON files of 50 for an external web-research agent, from the goa_tag_audit.py outputs.

Two kinds of batch, both with an empty `research` block for the agent and nothing else to change:
  artists_batch_NNN.json   the single-release artists flagged "HALLUCINATED?" (not in the scene archive, no audio match):
                           is this a real act, does it make psytrance/goa, and if not, which act the tracks really belong to.
                           One entry per artist with every file and name it appears under.
  renamed_batch_NNN.json   Goa_Separated tracks whose source file in the Goa Dataset now carries a different name, and
                           whose recording has no copy in the archive (so the fingerprint cannot settle it): which of the
                           two names is the real release, or neither.
Merge-back keys: `artist` for the artist batches, `goa_separated` for the renamed ones. Nothing is modified.

  mir/bin/python src/tools/goa_research_batches.py --out /run/media/kim/Kosmos/goa_tag_audit
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

ARTIST_INSTRUCTIONS = (
    "These artist names came from automatic tagging (MusicBrainz Picard) of untagged goa / psytrance rips, and some are "
    "wrong matches: rock, jazz, blues or spoken-word acts, authors, even politicians, attached to psytrance audio. For "
    "each artist, check whether it is a real act that released the listed titles, and whether it makes goa / "
    "psytrance / related electronic music. Use Discogs, MusicBrainz, Bandcamp, Beatport, label pages. Fill `research`: "
    "exists (true/false/null), makes_psytrance (true/false/null), genre, active_years, label, and when the tag looks "
    "wrong, likely_real_artist / likely_real_title for each title if a search on the title + psytrance finds the real "
    "release. Leave a field null when you cannot verify it; do not guess. confidence: high / medium / low. sources: the "
    "URLs used. Do not change any field outside `research`."
)
RENAMED_INSTRUCTIONS = (
    "Each entry is one audio file that has carried two different names: `name_a` (an older snapshot) and `name_b` "
    "(the same file in the current library). At most one is right; sometimes neither. Durations are given to tell "
    "releases apart. Using Discogs / MusicBrainz / Bandcamp / Beatport, find which name matches a real release of that "
    "length and fill `research`: correct ('a' / 'b' / 'neither' / null), artist, title, album, year, label, genre. "
    "Leave fields null when you cannot verify; do not guess. confidence: high / medium / low. sources: the URLs used. "
    "Do not change any field outside `research`."
)
# real acts the archive just lacks, not tagging errors (2026-10-25): an own project, an untagged vinyl rip
NOT_SUSPECT = {"Aavepyörä", "01-GGE"}
ARTIST_RESEARCH = {"exists": None, "makes_psytrance": None, "genre": None, "active_years": None, "label": None,
                   "likely_real": [], "notes": None, "sources": [], "confidence": None}
RENAMED_RESEARCH = {"correct": None, "artist": None, "title": None, "album": None, "year": None, "label": None,
                    "genre": None, "notes": None, "sources": [], "confidence": None}


def dump(out, prefix, instructions, entries, size):
    n = 0
    for i in range(0, len(entries), size):
        n += 1
        with open(os.path.join(out, f"{prefix}_{n:03d}.json"), "w") as f:
            json.dump({"batch": n, "instructions": instructions, "entries": entries[i:i + size]}, f,
                      ensure_ascii=False, indent=1)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="goa_tag_audit.py output dir (batches go to <out>/research_batches)")
    ap.add_argument("--size", type=int, default=50)
    a = ap.parse_args()
    dst = os.path.join(a.out, "research_batches")
    os.makedirs(dst, exist_ok=True)

    audit = list(csv.DictReader(open(os.path.join(a.out, "goa_tag_audit.csv"))))
    by_artist = defaultdict(list)
    for r in audit:
        by_artist[r["tag_artist"]].append(r)

    # ---- artists
    artists = []
    for s in csv.DictReader(open(os.path.join(a.out, "single_release_artists.csv"))):
        if not s["lead"] or s["artist"] in NOT_SUSPECT:
            continue
        files, titles, other_names, crops = [], [], set(), 0
        for r in by_artist.get(s["artist"], []):
            files.append({"file": r["rel"], "title": r["tag_title"], "album": r["tag_album"] or None,
                          "duration_s": round(float(r["duration"])) if r["duration"] else None})
            if r["tag_title"] and r["tag_title"] not in titles:
                titles.append(r["tag_title"])
            if r["goa_source_name_now"]:
                other_names.add(r["goa_source_name_now"])
            crops += int(r["latents_sa3_crops"] or 0)
        artists.append({"artist": s["artist"], "titles": titles, "album": s["album"] or None, "files": files,
                        "also_named": sorted(other_names - {f"{s['artist']} - {t}" for t in titles}),
                        "in_training_crops": crops, "research": dict(ARTIST_RESEARCH)})
    artists.sort(key=lambda x: x["artist"].lower())

    # ---- renamed since separation, no archive copy to settle it
    renamed = []
    for r in audit:
        if r["part"] != "goa_separated" or r["renamed_since_separation"] != "True" or r["match_rel"]:
            continue
        renamed.append({"goa_separated": r["rel"].split("/", 1)[1],
                        "name_a": f"{r['tag_artist']} - {r['tag_title']}" if r["tag_artist"] else r["rel"].split("/", 1)[1],
                        "album_a": r["tag_album"] or None, "name_b": r["goa_source_name_now"],
                        "file_b": r["goa_source_rel"],
                        "duration_s": round(float(r["duration"])) if r["duration"] else None,
                        "in_training_crops": int(r["latents_sa3_crops"] or 0), "research": dict(RENAMED_RESEARCH)})
    renamed.sort(key=lambda x: x["goa_separated"].lower())

    na = dump(dst, "artists_batch", ARTIST_INSTRUCTIONS, artists, a.size)
    nr = dump(dst, "renamed_batch", RENAMED_INSTRUCTIONS, renamed, a.size)
    print(f"{len(artists)} artists -> {na} files, {len(renamed)} renamed tracks without an archive copy -> {nr} files, "
          f"in {dst}")


if __name__ == "__main__":
    sys.exit(main())
