"""
Demucs Stem Separation for Crops

Wraps the standard Demucs separation to handle crop-specific stem naming.

For crop TrackName_0.flac, creates:
    TrackName_0_drums.flac
    TrackName_0_bass.flac
    TrackName_0_other.flac
    TrackName_0_vocals.flac

This is a FALLBACK for when stems weren't cropped with create_training_crops.py.
The preferred workflow is to have stems cropped along with the full mix.

Usage:
    python -m crops.demucs_sep /path/to/crops/TrackName/ --batch
    python -m crops.demucs_sep /path/to/TrackName_0.flac
"""

import shutil
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import find_crop_files, get_crop_stem_files, DEMUCS_STEMS
from core.common import setup_logging
from preprocessing.demucs_sep import separate_stems, check_demucs_installed

logger = logging.getLogger(__name__)


def separate_crop_stems(
    crop_file: str | Path,
    output_dir: Optional[Path] = None,
    device: str = 'cuda',
    output_format: str = 'mp3',
    overwrite: bool = False,
    **kwargs
) -> Dict[str, Path]:
    """
    Separate a single crop file into stems with prefixed naming.

    Args:
        crop_file: Path to the crop audio file (e.g., TrackName_0.flac)
        output_dir: Directory for stems (default: same as crop)
        device: 'cuda' or 'cpu'
        output_format: 'flac', 'mp3', etc.
        overwrite: Overwrite existing stems
        **kwargs: Additional arguments passed to separate_stems()

    Returns:
        Dict mapping stem names to paths (includes 'source' for original crop)
    """
    crop_file = Path(crop_file)
    if not crop_file.exists():
        raise FileNotFoundError(f"Crop file not found: {crop_file}")

    if output_dir is None:
        output_dir = crop_file.parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_stem = crop_file.stem  # e.g., "TrackName_0"

    # Check for existing stems
    if not overwrite:
        existing = get_crop_stem_files(crop_file)
        # Remove 'source' from count - we want actual stems
        stem_count = len([k for k in existing if k != 'source'])
        if stem_count == len(DEMUCS_STEMS):
            logger.info(f"Stems already exist for {crop_file.name}. Use --overwrite to regenerate.")
            return existing

    logger.info(f"Separating stems for crop: {crop_file.name}")

    # Get extension for output format
    ext_map = {
        'flac': '.flac',
        'mp3': '.mp3',
        'wav': '.wav',
        'ogg': '.ogg',
    }
    out_ext = ext_map.get(output_format, '.flac')

    # Use temp directory for demucs output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Run demucs separation
        stem_paths = separate_stems(
            crop_file,
            temp_path,
            device=device,
            output_format=output_format,
            **kwargs
        )

        # Rename and move stems to final location with prefixed names
        final_stems = {'source': crop_file}

        for stem_name, temp_stem_path in stem_paths.items():
            # Create prefixed name: TrackName_0_drums.flac
            final_name = f"{crop_stem}_{stem_name}{out_ext}"
            final_path = output_dir / final_name

            # Move from temp to final
            shutil.move(str(temp_stem_path), str(final_path))
            final_stems[stem_name] = final_path
            logger.info(f"  Created: {final_name}")

    logger.info(f"Successfully separated {len(final_stems) - 1} stems for {crop_file.name}")
    return final_stems


def batch_separate_crop_stems(
    crops_folder: str | Path,
    device: str = 'cuda',
    output_format: str = 'mp3',
    overwrite: bool = False,
    **kwargs
) -> Dict[str, any]:
    """
    Batch separate all crops in a folder.

    Loads Demucs model ONCE, processes all crops.

    Args:
        crops_folder: Folder containing crop files
        device: 'cuda' or 'cpu'
        output_format: 'flac', 'mp3', etc.
        overwrite: Overwrite existing stems
        **kwargs: Additional arguments passed to separate_stems()

    Returns:
        Statistics dict
    """
    crops_folder = Path(crops_folder)
    if not crops_folder.exists():
        raise FileNotFoundError(f"Folder not found: {crops_folder}")

    # Find all crop files
    crop_files = find_crop_files(crops_folder)

    if not crop_files:
        logger.warning(f"No crop files found in {crops_folder}")
        return {'total': 0, 'success': 0, 'skipped': 0, 'failed': 0, 'errors': []}

    logger.info(f"Found {len(crop_files)} crop files to process")

    stats = {
        'total': len(crop_files),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }

    for i, crop_file in enumerate(crop_files, 1):
        logger.info(f"Processing {i}/{stats['total']}: {crop_file.name}")

        try:
            # Check if stems already exist
            existing = get_crop_stem_files(crop_file)
            stem_count = len([k for k in existing if k != 'source'])

            if not overwrite and stem_count == len(DEMUCS_STEMS):
                logger.info(f"  Skipping (stems exist)")
                stats['skipped'] += 1
                continue

            # Separate stems
            separate_crop_stems(
                crop_file,
                device=device,
                output_format=output_format,
                overwrite=overwrite,
                **kwargs
            )
            stats['success'] += 1

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{crop_file.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"  Failed: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Crop Stem Separation Summary:")
    logger.info(f"  Total crops:  {stats['total']}")
    logger.info(f"  Successful:   {stats['success']}")
    logger.info(f"  Skipped:      {stats['skipped']}")
    logger.info(f"  Failed:       {stats['failed']}")
    logger.info("=" * 60)

    return stats


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Separate crop audio files into stems using Demucs"
    )

    parser.add_argument(
        'path',
        type=str,
        help='Path to crop file or folder containing crops'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all crops in folder'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='Device for processing (default: cuda)'
    )

    parser.add_argument(
        '--format', '-f',
        type=str,
        default='mp3',
        choices=['mp3', 'flac', 'wav', 'ogg'],
        help='Output format for stems (default: mp3 VBR 96kbps)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing stems'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if not check_demucs_installed():
        logger.error("Demucs is not installed. Install with: pip install demucs")
        sys.exit(1)

    path = Path(args.path)
    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)

    try:
        if args.batch or path.is_dir():
            stats = batch_separate_crop_stems(
                path,
                device=args.device,
                output_format=args.format,
                overwrite=args.overwrite
            )
            if stats['failed'] > 0:
                sys.exit(1)
        else:
            # Single crop file
            result = separate_crop_stems(
                path,
                device=args.device,
                output_format=args.format,
                overwrite=args.overwrite
            )
            print("\nStem Separation Complete:")
            for stem_name, stem_path in sorted(result.items()):
                print(f"  {stem_name}: {stem_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
