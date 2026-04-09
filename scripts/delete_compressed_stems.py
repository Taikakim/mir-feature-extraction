#!/usr/bin/env python3
"""
delete_compressed_stems.py — Phase 2 of clean-source upgrade.

For each track folder in Goa_Separated that has a .NEEDS_RESEPARATION marker,
delete the stem audio files (bass, drums, other, vocals) so that the main
pipeline will queue them for re-separation with BS-RoFormer.

full_mix.flac and all sidecar files (.INFO, .BEATS_GRID, .DOWNBEATS, .ONSETS)
are left untouched.

Usage:
    python scripts/delete_compressed_stems.py [--dry-run] [--verbose]
"""

import argparse
from pathlib import Path

GOA_SEPARATED = Path("/run/media/kim/Mantu/ai-music/Goa_Separated")

STEM_NAMES  = {'bass', 'drums', 'other', 'vocals', 'guitar', 'piano'}
AUDIO_EXTS  = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac'}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print every file deleted')
    args = parser.parse_args()

    dry     = args.dry_run
    verbose = args.verbose or dry

    markers = sorted(GOA_SEPARATED.rglob('.NEEDS_RESEPARATION'))
    if not markers:
        print("No .NEEDS_RESEPARATION markers found — nothing to do.")
        return

    print(f"Found {len(markers)} folders marked for re-separation.")
    if dry:
        print("*** DRY RUN — no changes will be made ***")
    print()

    total_deleted = 0
    total_missing = 0

    for marker in markers:
        folder = marker.parent
        stem_files = [
            f for f in folder.iterdir()
            if f.suffix.lower() in AUDIO_EXTS and f.stem.lower() in STEM_NAMES
        ]

        if not stem_files:
            if verbose:
                print(f"[{folder.name}]  no stems found (already clean)")
            total_missing += 1
            continue

        print(f"[{folder.name}]  {len(stem_files)} stem file(s)")
        for sf in sorted(stem_files):
            if verbose:
                print(f"  delete  {sf.name}")
            if not dry:
                sf.unlink()
            total_deleted += 1

    print()
    print("=" * 60)
    if dry:
        print(f"DRY RUN — would delete {total_deleted} stem files across "
              f"{len(markers) - total_missing} folders")
    else:
        print(f"Done.  Deleted {total_deleted} stem files.")
        print(f"  Folders with stems removed : {len(markers) - total_missing}")
        print(f"  Folders already stem-free  : {total_missing}")
    print()
    print("Next step: run master_pipeline.py with separation_backend: bs_roformer")
    print("  Folders with <4 stems will be queued automatically.")


if __name__ == '__main__':
    main()
