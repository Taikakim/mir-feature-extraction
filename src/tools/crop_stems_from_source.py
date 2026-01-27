"""
Crop stems from source organized folders to match existing crops.

If you have:
  - Organized tracks with stems at /source/TrackName/{drums,bass,other,vocals}.{ext}
  - Crops at /crops/TrackName/TrackName_0.flac with position in .INFO

This script creates the corresponding stem crops:
  /crops/TrackName/TrackName_0_drums.flac
  /crops/TrackName/TrackName_0_bass.flac
  etc.

Usage:
    python src/tools/crop_stems_from_source.py /path/to/crops --source /path/to/organized

Example:
    python src/tools/crop_stems_from_source.py /audio/sunset_dance_crops --source /audio/sunset_dance
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import soundfile as sf
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.common import setup_logging
from core.file_utils import find_crop_folders, find_crop_files

logger = logging.getLogger(__name__)

STEM_NAMES = ['drums', 'bass', 'other', 'vocals']
CROP_DURATION = 30.0  # seconds


def find_source_stems(source_dir: Path, track_name: str) -> dict:
    """Find stem files in source organized folder."""
    source_folder = source_dir / track_name
    if not source_folder.exists():
        return {}

    stems = {}
    for stem_name in STEM_NAMES:
        for ext in ['.flac', '.mp3', '.wav']:
            stem_path = source_folder / f"{stem_name}{ext}"
            if stem_path.exists():
                stems[stem_name] = stem_path
                break

    return stems


def get_crop_position(info_path: Path) -> float | None:
    """Read position from crop's .INFO file."""
    if not info_path.exists():
        return None

    try:
        with open(info_path, 'r') as f:
            data = json.load(f)
            return data.get('position')
    except Exception as e:
        logger.warning(f"Failed to read {info_path}: {e}")
        return None


def crop_stem(
    source_stem: Path,
    output_path: Path,
    position: float,
    duration: float = CROP_DURATION,
) -> bool:
    """Crop a single stem file."""
    try:
        # Read source audio info
        info = sf.info(source_stem)
        sr = info.samplerate

        # Calculate frames
        start_frame = int(position * sr)
        n_frames = int(duration * sr)

        # Ensure we don't read past end of file
        max_frames = info.frames - start_frame
        if max_frames <= 0:
            logger.warning(f"Position {position}s is past end of {source_stem.name}")
            return False

        n_frames = min(n_frames, max_frames)

        # Read and write
        audio, sr = sf.read(source_stem, start=start_frame, frames=n_frames, dtype='float32')
        sf.write(output_path, audio, sr, subtype='PCM_16' if output_path.suffix == '.flac' else None)

        return True

    except Exception as e:
        logger.error(f"Failed to crop {source_stem.name}: {e}")
        return False


def process_crop_file(
    crop_path: Path,
    source_stems: dict,
    output_format: str = 'mp3',
    overwrite: bool = False,
) -> dict:
    """Process a single crop file - create all stem crops."""
    results = {'success': 0, 'skipped': 0, 'failed': 0}

    # Get crop base name (without extension)
    crop_base = crop_path.stem  # e.g., "TrackName_0"
    crop_dir = crop_path.parent

    # Get position from .INFO
    info_path = crop_path.with_suffix('.INFO')
    position = get_crop_position(info_path)

    if position is None:
        logger.warning(f"No position in {info_path.name}")
        results['failed'] = len(STEM_NAMES)
        return results

    # Crop each stem
    for stem_name, source_stem in source_stems.items():
        output_path = crop_dir / f"{crop_base}_{stem_name}.{output_format}"

        if output_path.exists() and not overwrite:
            results['skipped'] += 1
            continue

        if crop_stem(source_stem, output_path, position):
            results['success'] += 1
        else:
            results['failed'] += 1

    return results


def process_crop_folder(
    crop_folder: Path,
    source_dir: Path,
    output_format: str = 'mp3',
    overwrite: bool = False,
) -> dict:
    """Process all crops in a folder."""
    folder_name = crop_folder.name

    # Find source stems
    source_stems = find_source_stems(source_dir, folder_name)
    if not source_stems:
        logger.warning(f"No source stems for {folder_name}")
        return {'total': 0, 'success': 0, 'skipped': 0, 'failed': 0, 'no_source': True}

    # Find crop files
    crop_files = find_crop_files(crop_folder)
    if not crop_files:
        return {'total': 0, 'success': 0, 'skipped': 0, 'failed': 0}

    results = {
        'total': len(crop_files),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'stems_found': list(source_stems.keys()),
    }

    for crop_path in crop_files:
        crop_results = process_crop_file(
            crop_path,
            source_stems,
            output_format,
            overwrite,
        )
        results['success'] += crop_results['success']
        results['skipped'] += crop_results['skipped']
        results['failed'] += crop_results['failed']

    return results


def batch_crop_stems(
    crops_dir: Path,
    source_dir: Path,
    output_format: str = 'mp3',
    overwrite: bool = False,
    max_workers: int = 4,
) -> dict:
    """Batch process all crop folders."""
    crop_folders = find_crop_folders(crops_dir)

    logger.info("=" * 60)
    logger.info("BATCH STEM CROPPING")
    logger.info("=" * 60)
    logger.info(f"Crops directory: {crops_dir}")
    logger.info(f"Source directory: {source_dir}")
    logger.info(f"Folders to process: {len(crop_folders)}")
    logger.info(f"Output format: {output_format}")
    logger.info("=" * 60)

    stats = {
        'total_folders': len(crop_folders),
        'folders_processed': 0,
        'folders_no_source': 0,
        'total_stems': 0,
        'success': 0,
        'skipped': 0,
        'failed': 0,
    }

    for i, crop_folder in enumerate(crop_folders, 1):
        logger.info(f"[{i}/{len(crop_folders)}] {crop_folder.name}")

        results = process_crop_folder(
            crop_folder,
            source_dir,
            output_format,
            overwrite,
        )

        if results.get('no_source'):
            stats['folders_no_source'] += 1
            continue

        stats['folders_processed'] += 1
        stats['total_stems'] += results['total'] * len(STEM_NAMES)
        stats['success'] += results['success']
        stats['skipped'] += results['skipped']
        stats['failed'] += results['failed']

        if results['success'] > 0:
            logger.info(f"  ✓ Created {results['success']} stem crops")
        if results['skipped'] > 0:
            logger.info(f"  - Skipped {results['skipped']} (exist)")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Folders processed: {stats['folders_processed']}/{stats['total_folders']}")
    logger.info(f"Folders without source: {stats['folders_no_source']}")
    logger.info(f"Stems created: {stats['success']}")
    logger.info(f"Stems skipped: {stats['skipped']}")
    logger.info(f"Stems failed: {stats['failed']}")
    logger.info("=" * 60)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Crop stems from source folders to match existing crops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crop stems for all crops
  python src/tools/crop_stems_from_source.py /audio/crops --source /audio/organized

  # Overwrite existing stem crops
  python src/tools/crop_stems_from_source.py /audio/crops --source /audio/organized --overwrite

  # Output as MP3 instead of FLAC
  python src/tools/crop_stems_from_source.py /audio/crops --source /audio/organized --format mp3
        """
    )

    parser.add_argument('crops_dir', help='Directory containing crop folders')
    parser.add_argument('--source', '-s', required=True, help='Source directory with organized folders and stems')
    parser.add_argument('--format', '-f', default='mp3', choices=['mp3', 'flac', 'wav'],
                        help='Output format for stem crops (default: mp3 VBR 96kbps)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing stem crops')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    crops_dir = Path(args.crops_dir)
    source_dir = Path(args.source)

    if not crops_dir.exists():
        logger.error(f"Crops directory not found: {crops_dir}")
        sys.exit(1)

    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        sys.exit(1)

    stats = batch_crop_stems(
        crops_dir,
        source_dir,
        output_format=args.format,
        overwrite=args.overwrite,
    )

    if stats['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
