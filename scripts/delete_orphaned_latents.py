#!/usr/bin/env python3
"""
delete_orphaned_latents.py — Delete .npy/.json files in goa-small and goa-stems
that have no corresponding source audio in Goa_Separated_crops.

Orphans arise from old encoding runs (different crop parameters, numbered folder
names embedded in filenames, etc.).  The crops directory is the source of truth.

Usage:
    python scripts/delete_orphaned_latents.py --dry-run
    python scripts/delete_orphaned_latents.py
"""

import argparse
from pathlib import Path

CROPS_DIR = Path("/run/media/kim/Mantu/ai-music/Goa_Separated_crops")
AUDIO_EXTS = {".flac", ".wav", ".mp3", ".ogg"}
STEM_SUFFIXES = ("_bass", "_drums", "_other", "_vocals")

TARGETS = [
    Path("/run/media/kim/Lehto/goa-small"),
    Path("/run/media/kim/Lehto/goa-stems"),
]


def build_valid_npy_names() -> dict[str, set[str]]:
    """Return {track_name: set of valid .npy basenames} from Goa_Separated_crops."""
    valid: dict[str, set[str]] = {}
    for track_dir in CROPS_DIR.iterdir():
        if not track_dir.is_dir():
            continue
        names: set[str] = set()
        for ext in AUDIO_EXTS:
            for f in track_dir.glob(f"*{ext}"):
                names.add(f.stem + ".npy")
        if names:
            valid[track_dir.name] = names
    return valid


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without making changes")
    args = parser.parse_args()
    dry = args.dry_run

    if dry:
        print("*** DRY RUN — no changes will be made ***\n")

    print("Building valid file index from Goa_Separated_crops...")
    valid = build_valid_npy_names()
    print(f"  {len(valid)} tracks with audio files\n")

    grand_total_deleted = 0
    grand_total_bytes   = 0

    for root_dir in TARGETS:
        deleted = 0
        deleted_bytes = 0
        print(f"=== {root_dir} ===")

        folders = sorted(d for d in root_dir.iterdir() if d.is_dir())
        for folder in folders:
            valid_names = valid.get(folder.name, set())
            # Also accept .json counterparts of valid .npy names
            valid_json  = {n.replace(".npy", ".json") for n in valid_names}

            for f in list(folder.iterdir()):
                if f.suffix == ".npy" and f.name not in valid_names:
                    sz = f.stat().st_size
                    if not dry:
                        f.unlink()
                    deleted += 1
                    deleted_bytes += sz
                elif f.suffix == ".json" and f.name not in valid_json:
                    sz = f.stat().st_size
                    if not dry:
                        f.unlink()
                    deleted += 1
                    deleted_bytes += sz

        print(f"  {'Would delete' if dry else 'Deleted'}: {deleted:,} files  "
              f"({deleted_bytes / 1e9:.2f} GB)")
        grand_total_deleted += deleted
        grand_total_bytes   += deleted_bytes

    print(f"\n{'DRY RUN ' if dry else ''}Total: {grand_total_deleted:,} files  "
          f"({grand_total_bytes / 1e9:.2f} GB freed)")


if __name__ == "__main__":
    main()
