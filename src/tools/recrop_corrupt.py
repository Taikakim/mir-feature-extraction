#!/usr/bin/env python3
"""
Find corrupt/truncated crop MP3s (< SIZE_THRESHOLD bytes), delete them and
their .INFO files, then re-run create_training_crops for each affected track.

Usage:
    python src/tools/recrop_corrupt.py --dry-run          # list what would happen
    python src/tools/recrop_corrupt.py                    # delete + re-crop
    python src/tools/recrop_corrupt.py --delete-only      # delete without re-cropping
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

CROPS_ROOT  = Path('/run/media/kim/LostLands/ai-music/Goa_Separated_crops')
SOURCE_ROOT = Path('/run/media/kim/LostLands/ai-music/Goa_Separated')
SIZE_THRESHOLD = 10_000       # bytes — anything under 10KB is corrupt/truncated

# Crop settings — must match config/master_pipeline.yaml cropping section
LENGTH_SAMPLES = 524288       # crop length (~11.9s at 44.1kHz)
CROP_OVERLAP   = False        # config: overlap: false
CROP_DIV4      = False        # config: div4: false
CROP_SEQUENTIAL = False       # config: mode: beat-aligned  (sequential=False)
OUTPUT_DIR     = CROPS_ROOT   # re-cropped files go back to the same location


def find_corrupt(crops_root: Path) -> list[Path]:
    """Return all .mp3 crop files smaller than SIZE_THRESHOLD bytes."""
    corrupt = [
        p for p in crops_root.rglob('*.mp3')
        if p.stat().st_size < SIZE_THRESHOLD
    ]
    return sorted(corrupt)


def delete_corrupt(corrupt_files: list[Path], dry_run: bool) -> set[str]:
    """
    Delete each corrupt mp3 and its .INFO file.
    Returns the set of track folder names that had corrupt files.
    """
    affected_tracks: set[str] = set()

    for mp3 in corrupt_files:
        track_name = mp3.parent.name
        affected_tracks.add(track_name)

        info = mp3.with_suffix('.INFO')

        if dry_run:
            logger.info(f"  WOULD DELETE: {mp3.name}  ({mp3.stat().st_size} bytes)")
            if info.exists():
                logger.info(f"  WOULD DELETE: {info.name}")
        else:
            mp3.unlink()
            logger.info(f"  Deleted: {mp3.name}")
            if info.exists():
                info.unlink()
                logger.info(f"  Deleted: {info.name}")

    return affected_tracks


def recrop_tracks(track_names: set[str], dry_run: bool) -> None:
    """Re-run create_training_crops for each affected track."""
    from tools.create_training_crops import create_crops_for_file

    missing_sources = []
    to_recrop = []

    for name in sorted(track_names):
        source_folder = SOURCE_ROOT / name
        if not source_folder.exists():
            missing_sources.append(name)
            logger.warning(f"  Source not found: {source_folder}")
        else:
            to_recrop.append(source_folder)

    if missing_sources:
        logger.warning(f"{len(missing_sources)} source folders not found — skipped")

    logger.info(f"\nRe-cropping {len(to_recrop)} tracks...")

    for i, folder in enumerate(to_recrop, 1):
        logger.info(f"[{i}/{len(to_recrop)}] {folder.name}")
        if dry_run:
            logger.info(f"  WOULD RECROP: {folder}")
            continue
        try:
            create_crops_for_file(
                folder_path=folder,
                length_samples=LENGTH_SAMPLES,
                overlap=CROP_OVERLAP,
                div4=CROP_DIV4,
                sequential=CROP_SEQUENTIAL,
                output_dir=OUTPUT_DIR,
                overwrite=True,
            )
        except Exception as e:
            logger.error(f"  Failed to re-crop {folder.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Delete corrupt crops and re-crop from source')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--delete-only', action='store_true',
                        help='Delete corrupt files without re-cropping')
    args = parser.parse_args()

    logger.info(f"Scanning for corrupt crop files (< {SIZE_THRESHOLD:,} bytes)...")
    corrupt = find_corrupt(CROPS_ROOT)
    logger.info(f"Found {len(corrupt)} corrupt files across "
                f"{len({p.parent.name for p in corrupt})} tracks")

    if not corrupt:
        logger.info("Nothing to do.")
        return

    logger.info("\nDeleting corrupt files...")
    affected = delete_corrupt(corrupt, dry_run=args.dry_run)

    logger.info(f"\nAffected tracks: {len(affected)}")
    for name in sorted(affected):
        logger.info(f"  {name}")

    if args.delete_only:
        logger.info("--delete-only: skipping re-crop.")
        return

    recrop_tracks(affected, dry_run=args.dry_run)
    logger.info("\nDone.")


if __name__ == '__main__':
    main()
