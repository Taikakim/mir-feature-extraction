#!/usr/bin/env python3
"""
rename_small_strip_numbers.py — Strip numeric prefixes from goa-small track folders.

Renames folders like '0042 Artist - Title' → 'Artist - Title', cross-checking
against Goa_Separated_crops to confirm the target name is in the active dataset.

Categories
----------
RENAME    stripped name exists in crops, no conflict in goa-small → safe to rename
MERGE     stripped name already exists as an unnumbered folder in goa-small
          (encode_dataset.py may have started creating it) → merge .npy/.json files
          from the numbered folder into the unnumbered one, then delete numbered
DUPLICATE two numbered folders map to the same stripped name → legitimate separate
          tracks, leave both and report for manual review
ORPHAN    stripped name not found in crops at all → skip (not in active dataset)

Usage:
    python scripts/rename_small_strip_numbers.py --dry-run
    python scripts/rename_small_strip_numbers.py
    python scripts/rename_small_strip_numbers.py --verbose
"""

import argparse
import re
import shutil
from pathlib import Path

GOA_SMALL = Path("/run/media/kim/Lehto/goa-small")
CROPS_DIR = Path("/run/media/kim/Mantu/ai-music/Goa_Separated_crops")

NUMBER_PREFIX = re.compile(r"^\d+\s+")


def strip_number(name: str) -> str:
    return NUMBER_PREFIX.sub("", name)


def folder_npy_count(folder: Path) -> int:
    return sum(1 for _ in folder.glob("*.npy"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print every folder processed')
    args = parser.parse_args()

    dry     = args.dry_run
    verbose = args.verbose or dry

    # Canonical names from Goa_Separated_crops (source of truth)
    crops_names = {d.name for d in CROPS_DIR.iterdir() if d.is_dir()}

    stems_dirs          = [d for d in GOA_SMALL.iterdir() if d.is_dir()]
    numbered_dirs       = [d for d in stems_dirs if NUMBER_PREFIX.match(d.name)]
    unnumbered_existing = {d.name: d for d in stems_dirs
                           if not NUMBER_PREFIX.match(d.name)}

    print(f"goa-small folders total      : {len(stems_dirs)}")
    print(f"  Numbered (to process)      : {len(numbered_dirs)}")
    print(f"  Unnumbered (already clean) : {len(unnumbered_existing)}")
    print(f"Crops reference              : {len(crops_names)} folders")
    if dry:
        print("*** DRY RUN — no changes will be made ***")
    print()

    # Group numbered folders by stripped name to detect true duplicates
    by_stripped: dict[str, list[Path]] = {}
    for d in numbered_dirs:
        by_stripped.setdefault(strip_number(d.name), []).append(d)

    renames    = []   # (src, dst_path)
    merges     = []   # (src, existing_unnumbered_dir)
    duplicates = []   # (stripped_name, [folders])
    orphans    = []   # (stripped_name, folder)

    for stripped, folders in sorted(by_stripped.items()):
        in_crops      = stripped in crops_names
        already_clean = stripped in unnumbered_existing
        is_duplicate  = len(folders) > 1

        if is_duplicate:
            duplicates.append((stripped, folders))
        elif already_clean:
            merges.append((folders[0], unnumbered_existing[stripped]))
        elif not in_crops:
            orphans.append((stripped, folders[0]))
        else:
            renames.append((folders[0], GOA_SMALL / stripped))

    # -------------------------------------------------------------------------
    # RENAME
    # -------------------------------------------------------------------------
    print(f"RENAME — {len(renames)} folders")
    renamed = 0
    late_merges = 0
    for src, dst in renames:
        n = folder_npy_count(src)
        if not dry and dst.exists():
            # Destination appeared since the scan — merge
            existing_in_dst = {f.name for f in dst.iterdir()}
            files_to_move   = [f for f in src.iterdir()
                               if f.name not in existing_in_dst]
            if verbose:
                print(f"  LATE-MERGE {src.name}  →  {dst.name}  "
                      f"({len(files_to_move)} files to move)")
            for f in files_to_move:
                shutil.move(str(f), str(dst / f.name))
            remaining = list(src.iterdir())
            if not remaining:
                src.rmdir()
            late_merges += 1
            continue
        if verbose:
            print(f"  {src.name}  →  {dst.name}  ({n} files)")
        if not dry:
            src.rename(dst)
            renamed += 1

    # -------------------------------------------------------------------------
    # MERGE — numbered folder → already-existing unnumbered folder
    # -------------------------------------------------------------------------
    print(f"\nMERGE — {len(merges)} folders (numbered → unnumbered, move missing files)")
    merged_folders = 0
    merged_files   = 0
    for src, dst in merges:
        existing_in_dst = {f.name for f in dst.iterdir()}
        files_to_move   = [f for f in src.iterdir()
                           if f.name not in existing_in_dst]
        n_total = folder_npy_count(src)
        if verbose:
            print(f"  {src.name}  →  {dst.name}")
            print(f"    {len(files_to_move)} files to move in "
                  f"(dst already has {len(existing_in_dst)}, src has {n_total})")
        if not dry:
            for f in files_to_move:
                shutil.move(str(f), str(dst / f.name))
                merged_files += 1
            remaining = list(src.iterdir())
            if not remaining:
                src.rmdir()
            else:
                print(f"  WARNING: {src.name} not empty after merge "
                      f"({len(remaining)} files remain — conflicts)")
        merged_folders += 1

    # -------------------------------------------------------------------------
    # DUPLICATE — two numbered folders with same stripped name
    # -------------------------------------------------------------------------
    print(f"\nDUPLICATE — {len(duplicates)} groups (legitimate separate tracks, "
          f"left untouched)")
    for stripped, folders in duplicates:
        in_crops = stripped in crops_names
        print(f"  '{stripped}'"
              + (" [in crops]" if in_crops else " [NOT in crops]"))
        for f in sorted(folders):
            print(f"    {f.name}  ({folder_npy_count(f)} .npy files)")

    # -------------------------------------------------------------------------
    # ORPHAN — no matching crops folder
    # -------------------------------------------------------------------------
    print(f"\nORPHAN — {len(orphans)} folders (not in crops, skipped)")
    for stripped, folder in orphans[:30]:
        print(f"  {folder.name}")
    if len(orphans) > 30:
        print(f"  ... and {len(orphans) - 30} more")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    if dry:
        print(f"DRY RUN complete.")
        print(f"  Would rename  : {len(renames)} folders")
        print(f"  Would merge   : {len(merges)} folders")
        print(f"  Duplicates    : {len(duplicates)} groups (manual review)")
        print(f"  Orphans       : {len(orphans)} folders (skipped)")
    else:
        print(f"Done.")
        print(f"  Renamed       : {renamed} folders")
        print(f"  Merged        : {merged_folders + late_merges} folders "
              f"({merged_files} files moved, {late_merges} late-merge)")
        print(f"  Duplicates    : {len(duplicates)} groups (untouched — review manually)")
        print(f"  Orphans       : {len(orphans)} folders (untouched)")
    print()
    print("Next step: re-run encode_dataset.py — only genuinely missing files "
          "will be encoded.")


if __name__ == '__main__':
    main()
