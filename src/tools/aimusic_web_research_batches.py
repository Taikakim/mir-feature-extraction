#!/usr/bin/env python3
# Owner: WINTERMUTE (created 2026-10-26)
"""aimusic_web_research_batches.py -- input batches for web-research caption agents, one JSON per batch, grouped by artist.

Each item: id (track folder), set, artist, title, duration_s (ffprobe), year_hint (from the aimusic_captions genre hint),
in_training_set (the track has crops in latents_aimusic4/<set>), audio_caption (Music Flamingo caption from
Kosmos/captions/aimusic_captions), research = null for the agent to fill. Tracks of one artist stay in one batch (album
lookups are shared) unless the artist has more than --max-per-artist tracks. First used for the Chill set (2026-10-25,
batches built inline; see Kosmos/captions/chill_web_research/README.md).

  mir/bin/python src/tools/aimusic_web_research_batches.py --sets "Prog & Psytechno Dataset=prog_psytechno" \
      "Progressive Trance & Melodic Techno=prog_trance_melodic_techno" "organic dance=organic_dance" \
      --batches 15 --out /run/media/kim/Kosmos/captions/aimusic_web_research
"""
import argparse
import collections
import glob
import json
import os
import re
import subprocess

SRC = "/run/media/kim/Mantu/ai-music"
CAPS = "/run/media/kim/Kosmos/captions/aimusic_captions/json"
LATENTS = "/run/media/kim/Kosmos/latents/latents_aimusic4"


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    try:
        return round(float(out), 1)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, help='"<source folder>=<latents_aimusic4 subdir>" ...')
    ap.add_argument("--batches", type=int, default=15)
    ap.add_argument("--max-per-artist", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sets = dict(s.split("=", 1) for s in a.sets)

    caps = {}
    for f in glob.glob(f"{CAPS}/*.json"):
        j = json.load(open(f))
        top, _, rest = j["rel"].partition("/")
        if top in sets:
            caps[(top, os.path.splitext(rest)[0])] = j

    items = []
    for src, sub in sets.items():
        trained = {json.load(open(f))["source_track"] for f in glob.glob(f"{LATENTS}/{sub}/*.json")}
        root = f"{SRC}/{src}"
        for d in sorted(os.listdir(root)):
            if not os.path.isdir(f"{root}/{d}"):
                continue
            m = re.match(r"^(\d+)\.\s*(.+?) - (.+)$", d)
            artist, title = (m.group(2), m.group(3)) if m else (None, d)
            fm = glob.glob(glob.escape(f"{root}/{d}") + "/full_mix.*")
            c = caps.get((src, d), {})
            yh = re.search(r"release year: (\d{4})", c.get("genre_hint", "") or "")
            items.append({"id": d, "set": sub, "artist": artist, "title": title,
                          "duration_s": duration(fm[0]) if fm else None, "year_hint": yh.group(1) if yh else None,
                          "in_training_set": d in trained, "audio_caption": c.get("captions", {}).get("full"),
                          "research": None})

    by = collections.defaultdict(list)
    for it in items:
        by[(it["artist"] or "").split(",")[0].strip().lower()].append(it)
    bins = [[] for _ in range(a.batches)]
    for g in sorted(by.values(), key=len, reverse=True):
        for k in range(0, len(g), a.max_per_artist):
            min(bins, key=len).extend(g[k:k + a.max_per_artist])
    for n, b in enumerate(bins):
        json.dump({"batch": n, "items": b}, open(f"{a.out}/batch_{n:02d}.json", "w"), indent=1, ensure_ascii=False)
    print(f"{len(items)} tracks ({sum(i['in_training_set'] for i in items)} in training, "
          f"{sum(bool(i['audio_caption']) for i in items)} with audio caption) -> {a.batches} batches of "
          f"{min(map(len, bins))}-{max(map(len, bins))}")


if __name__ == "__main__":
    main()
