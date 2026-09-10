#!/usr/bin/env python3
"""
inject_trigger_caption.py — write the aavepyora / aavepyörä trigger caption into the
avp corpus INFOs, in place of Music-Flamingo descriptions (which are disabled for this
corpus). This is the training "prompt" for the personal-style LoRA.

Kim's spec: the caption is just the trigger word, "aavepyora" or "aavepyörä", 50/50.
Randomization is DETERMINISTIC per track (hash of track name) so re-runs are stable and
every crop / augmentation of the same track shares one spelling.

Writes the word into the fields a downstream encoder is likely to read as the text
prompt: `caption`, `music_flamingo_full`, `music_flamingo_short_genre`. (Confirm the
exact field the SA3/SAT encoder reads and trim this set if needed — cheap to re-run.)

Run AFTER crop_analysis has created the *.INFO files:
  mir/bin/python src/tools/inject_trigger_caption.py <avp-analyzed> [--dry-run]
"""
import sys, hashlib, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.json_handler import safe_update

WORDS = ["aavepyora", "aavepyörä"]
FIELDS = ["caption", "music_flamingo_full", "music_flamingo_short_genre"]


def track_name_of(info_path: Path) -> str:
    # crop INFOs live under <track>/... ; the track folder is the identity for 50/50.
    # Walk up to the folder directly under the corpus root.
    parts = info_path.parts
    return parts[-2] if len(parts) >= 2 else info_path.stem


def word_for(track: str) -> str:
    h = int(hashlib.sha1(track.encode("utf-8")).hexdigest(), 16)
    return WORDS[h % 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    infos = sorted(root.glob("**/*.INFO"))
    counts = {w: 0 for w in WORDS}
    for ip in infos:
        w = word_for(track_name_of(ip))
        counts[w] += 1
        if args.dry_run:
            continue
        safe_update(str(ip), {f: w for f in FIELDS})
    print(f"{'[dry-run] ' if args.dry_run else ''}captioned {len(infos)} INFO files "
          f"-> {counts}")


if __name__ == "__main__":
    main()
