"""
File Organization for MIR Project

This script organizes audio files into the required folder structure:
- Creates a folder with the audio filename (minus extension)
- Moves the audio file into that folder and renames it to full_mix.{ext}

Example:
    /folder/song.flac  →  /folder/song/full_mix.flac

This structure is required for:
- Demucs stem separation (outputs to same folder)
- Feature extraction (saves .INFO, .BEATS_GRID, etc. to same folder)
- Dataset config for Stable Audio Tools

Dependencies:
- pathlib
- shutil
- src.core.common (AUDIO_EXTENSIONS)

Usage:
    python organize_files.py <path> [--batch] [--overwrite] [-v]

Options:
    --batch: Process all audio files recursively in directory tree
    --overwrite: Re-organize files even if already organized
    -v, --verbose: Enable verbose logging
"""

import shutil
from pathlib import Path
from typing import List, Dict
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.common import AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)


def is_already_organized(audio_file: Path) -> bool:
    """
    Check if an audio file is already organized.

    An audio file is considered organized if:
    1. It's named full_mix.{ext}, OR
    2. Its parent folder contains a full_mix.{ext} file (it's a stem/related file)

    Args:
        audio_file: Path to audio file

    Returns:
        True if already organized, False otherwise
    """
    # Check if this file is named full_mix
    if audio_file.stem == "full_mix":
        return True

    # Check if parent folder contains a full_mix file (any extension)
    parent_dir = audio_file.parent
    for ext in AUDIO_EXTENSIONS:
        full_mix_path = parent_dir / f"full_mix{ext}"
        if full_mix_path.exists():
            # This file is in an organized folder (probably a stem)
            return True

    return False


def organize_single_file(audio_file: Path, overwrite: bool = False) -> bool:
    """
    Organize a single audio file into the required folder structure.

    Args:
        audio_file: Path to audio file
        overwrite: Whether to re-organize if already organized

    Returns:
        True if organized successfully, False if skipped or failed

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio file is in root directory (can't create parent folder)
    """
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # Check if already organized
    if is_already_organized(audio_file) and not overwrite:
        logger.info(f"Already organized: {audio_file}")
        return False

    # Get the parent directory and filename
    parent_dir = audio_file.parent
    filename_stem = audio_file.stem
    extension = audio_file.suffix

    # If it's already named full_mix, use the parent folder name
    if filename_stem == "full_mix":
        folder_name = parent_dir.name
        target_folder = parent_dir
    else:
        # Create target folder with the filename (minus extension)
        folder_name = filename_stem
        target_folder = parent_dir / folder_name

    # Check if target folder already exists and has full_mix
    target_file = target_folder / f"full_mix{extension}"

    if target_file.exists() and not overwrite:
        logger.warning(f"Target already exists: {target_file}")
        if target_file != audio_file:
            logger.warning(f"Skipping to avoid overwriting. Use --overwrite to force.")
            return False
        else:
            # Same file, already organized
            return False

    # Create target folder if it doesn't exist
    if not target_folder.exists():
        logger.info(f"Creating folder: {target_folder}")
        target_folder.mkdir(parents=True, exist_ok=True)

    # Move and rename the file
    if audio_file != target_file:
        logger.info(f"Moving: {audio_file.name}")
        logger.info(f"    To: {target_file.relative_to(parent_dir)}")
        shutil.move(str(audio_file), str(target_file))

    return True


def find_audio_files(root_path: Path, recursive: bool = True) -> List[Path]:
    """
    Find all audio files in a directory.

    Args:
        root_path: Root directory to search
        recursive: Whether to search recursively

    Returns:
        List of audio file paths
    """
    audio_files = []

    if recursive:
        for ext in AUDIO_EXTENSIONS:
            audio_files.extend(root_path.rglob(f"*{ext}"))
    else:
        for ext in AUDIO_EXTENSIONS:
            audio_files.extend(root_path.glob(f"*{ext}"))

    # Filter out files that are already organized (named full_mix)
    # unless we're in overwrite mode (handled later)
    return sorted(audio_files)


def batch_organize_files(root_directory: str | Path,
                          overwrite: bool = False) -> Dict[str, int]:
    """
    Batch organize all audio files in a directory tree.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to re-organize files that are already organized

    Returns:
        Dictionary with statistics about the batch processing
    """
    root_directory = Path(root_directory)
    logger.info(f"Starting batch file organization: {root_directory}")

    # Find all audio files
    audio_files = find_audio_files(root_directory, recursive=True)

    # Separate already organized files
    files_to_organize = []
    already_organized = []

    for file in audio_files:
        if is_already_organized(file):
            already_organized.append(file)
        else:
            files_to_organize.append(file)

    stats = {
        'total': len(audio_files),
        'already_organized': len(already_organized),
        'to_organize': len(files_to_organize),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }

    logger.info(f"Found {stats['total']} audio files")
    logger.info(f"  Already organized: {stats['already_organized']}")
    logger.info(f"  Need organizing:   {stats['to_organize']}")

    if stats['to_organize'] == 0:
        logger.info("All files are already organized!")
        return stats

    # Process each file
    for i, audio_file in enumerate(files_to_organize, 1):
        logger.info(f"\nProcessing {i}/{stats['to_organize']}: {audio_file.name}")

        try:
            success = organize_single_file(audio_file, overwrite=overwrite)
            if success:
                stats['success'] += 1
            else:
                stats['skipped'] += 1

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{audio_file.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to organize {audio_file.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch File Organization Summary:")
    logger.info(f"  Total files:        {stats['total']}")
    logger.info(f"  Already organized:  {stats['already_organized']}")
    logger.info(f"  Newly organized:    {stats['success']}")
    logger.info(f"  Skipped:            {stats['skipped']}")
    logger.info(f"  Failed:             {stats['failed']}")
    logger.info("=" * 60)

    return stats


# Command-line interface
if __name__ == "__main__":
    import argparse
    from core.common import setup_logging

    parser = argparse.ArgumentParser(
        description="Organize audio files into required folder structure"
    )

    parser.add_argument(
        'path',
        type=str,
        help='Path to audio file or directory'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all audio files recursively in directory tree'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Re-organize files even if already organized'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    path = Path(args.path)

    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)

    try:
        if args.batch:
            # Batch processing
            stats = batch_organize_files(path, overwrite=args.overwrite)

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} files failed to organize")
                sys.exit(1)

        elif path.is_file():
            # Single file
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                logger.error(f"Not an audio file: {path}")
                sys.exit(1)

            success = organize_single_file(path, overwrite=args.overwrite)

            if success:
                logger.info("File organized successfully")
            else:
                logger.info("File was skipped (already organized)")

        else:
            # Directory without --batch flag
            logger.error("For directories, use --batch flag")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
