#!/usr/bin/env python3
"""
rename_crop_files_strip_numbers.py — Strip numeric prefixes from .npy/.json filenames
inside goa-small and goa-stems track folders.

The crop files were generated when track folders were still numbered, so they embedded
the prefix in the filename: '0067 Astral Projection - Aurora Borealis_0_bass.npy'.
encode_dataset.py now looks for the unnumbered form, so these files are never matched.

Handles conflicts: if the unnumbered target already exists in the same folder (e.g. from
a later encoding pass), the numbered file is a duplicate and is deleted.

Usage:
    python scripts/rename_crop_files_strip_numbers.py --dry-run
    python scripts/rename_crop_files_strip_numbers.py
"""

import argparse
import re
from pathlib import Path

DIRS = [
    Path("/run/media/kim/Lehto/goa-small"),
    Path("/run/media/kim/Lehto/goa-stems"),
]

NUMBER_PREFIX = re.compile(r"^\d+\s+")


def strip_number(name: str) -> str:
    return NUMBER_PREFIX.sub("", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without making changes')
    args = parser.parse_args()
    dry = args.dry_run

    if dry:
        print("*** DRY RUN — no changes will be made ***\n")

    total_renamed = 0
    total_conflicts = 0
    total_skipped = 0  # non-.npy/.json files with number prefix (unexpected)

    for root_dir in DIRS:
        renamed = 0
        conflicts = 0

        print(f"\n=== {root_dir} ===")
        folders = sorted(d for d in root_dir.iterdir() if d.is_dir())
        print(f"  Scanning {len(folders)} folders...")

        for folder in folders:
            for f in list(folder.iterdir()):
                if not NUMBER_PREFIX.match(f.name):
                    continue
                if f.suffix not in (".npy", ".json"):
                    total_skipped += 1
                    continue

                new_name = strip_number(f.name)
                dst = folder / new_name

                if dst.exists():
                    # Conflict: unnumbered version already there — delete the numbered one
                    if not dry:
                        f.unlink()
                    conflicts += 1
                else:
                    if not dry:
                        f.rename(dst)
                    renamed += 1

        print(f"  Renamed : {renamed:,}")
        print(f"  Deleted (conflicts) : {conflicts:,}")
        total_renamed += renamed
        total_conflicts += conflicts

    print(f"\n{'DRY RUN ' if dry else ''}Summary")
    print(f"  Total renamed  : {total_renamed:,}")
    print(f"  Total deleted  : {total_conflicts:,}  (target already existed)")
    if total_skipped:
        print(f"  Skipped (non-.npy/.json with number prefix): {total_skipped}")


if __name__ == '__main__':
    main()
