#!/usr/bin/env python3
"""
delete_encoded_latents.py — Phase 4 of clean-source upgrade.

For each track in Goa_Separated with a .NEEDS_RESEPARATION marker, delete
the entire pre-encoded latent folder from goa-small and goa-stems so that
encode_dataset.py / encode_stems.py will re-encode from the clean source.

Both .npy latent files and their companion .json files are removed by deleting
the whole track subfolder.

Usage:
    python scripts/delete_encoded_latents.py [--dry-run] [--verbose]
"""

import argparse
import shutil
from pathlib import Path

GOA_SEPARATED = Path("/run/media/kim/Mantu/ai-music/Goa_Separated")
GOA_SMALL     = Path("/run/media/kim/Lehto/goa-small")
GOA_STEMS     = Path("/run/media/kim/Lehto/goa-stems")


def count_files(folder: Path) -> int:
    return sum(1 for _ in folder.iterdir())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print each folder removed')
    args = parser.parse_args()

    dry     = args.dry_run
    verbose = args.verbose or dry

    markers = sorted(GOA_SEPARATED.rglob('.NEEDS_RESEPARATION'))
    if not markers:
        print("No .NEEDS_RESEPARATION markers found — nothing to do.")
        return

    print(f"Found {len(markers)} tracks marked for re-separation.")
    if dry:
        print("*** DRY RUN — no changes will be made ***")
    print()

    small_deleted = 0
    stems_deleted = 0
    small_missing = 0
    stems_missing = 0

    for marker in markers:
        track_name = marker.parent.name
        small_dir  = GOA_SMALL / track_name
        stems_dir  = GOA_STEMS / track_name

        parts = []
        if small_dir.exists():
            n = count_files(small_dir)
            parts.append(f"goa-small ({n} files)")
            if not dry:
                shutil.rmtree(small_dir)
            small_deleted += 1
        else:
            small_missing += 1

        if stems_dir.exists():
            n = count_files(stems_dir)
            parts.append(f"goa-stems ({n} files)")
            if not dry:
                shutil.rmtree(stems_dir)
            stems_deleted += 1
        else:
            stems_missing += 1

        if verbose:
            status = "  remove  " + " + ".join(parts) if parts else "  (not in either dir)"
            print(f"[{track_name}]{status}")

    print()
    print("=" * 60)
    if dry:
        print("DRY RUN complete.")
    else:
        print("Done.")
    print(f"  goa-small folders deleted : {small_deleted}  (not found: {small_missing})")
    print(f"  goa-stems folders deleted : {stems_deleted}  (not found: {stems_missing})")
    print()
    print("Next step: re-run encode_dataset.py and encode_stems.py for these tracks")
    print("  (encode scripts skip existing folders — deleted folders will be re-encoded)")


if __name__ == '__main__':
    main()
