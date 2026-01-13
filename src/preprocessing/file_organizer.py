"""
File Organizer for MIR Project

This script organizes audio files into the required folder structure for the MIR pipeline.

Input: Audio files in any location
Output: Organized structure with files moved to their own folders

Structure:
    /path/to/file/filename.flac -> /path/to/file/filename/full_mix.flac

Dependencies:
- src.core.file_utils
- src.core.common

Usage:
    python file_organizer.py /path/to/audio/files
    python file_organizer.py /path/to/audio/files --dry-run  # Test without moving
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import find_audio_files, get_audio_folder_structure, is_organized
from core.common import AUDIO_EXTENSIONS, setup_logging

logger = logging.getLogger(__name__)


def organize_file(audio_file: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Organize a single audio file into the required folder structure.

    Args:
        audio_file: Path to the audio file
        dry_run: If True, don't actually move files (just simulate)

    Returns:
        Tuple of (success: bool, message: str)
    """
    # Check if already organized
    if is_organized(audio_file):
        return True, f"Already organized: {audio_file}"

    # Get expected structure
    structure = get_audio_folder_structure(audio_file)
    folder_path = structure['folder']
    full_mix_path = structure['full_mix']

    # Check if target already exists
    if full_mix_path.exists():
        return False, f"Target already exists: {full_mix_path}"

    if dry_run:
        logger.info(f"[DRY RUN] Would move: {audio_file} -> {full_mix_path}")
        return True, f"Would organize: {audio_file}"

    try:
        # Create the folder
        folder_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created folder: {folder_path}")

        # Move the file
        shutil.move(str(audio_file), str(full_mix_path))
        logger.info(f"Organized: {audio_file} -> {full_mix_path}")

        return True, f"Successfully organized: {audio_file}"

    except Exception as e:
        logger.error(f"Error organizing {audio_file}: {e}")
        return False, f"Error: {e}"


def organize_directory(directory: Path,
                        dry_run: bool = False,
                        recursive: bool = True) -> dict:
    """
    Organize all audio files in a directory.

    Args:
        directory: Root directory containing audio files
        dry_run: If True, don't actually move files
        recursive: If True, search subdirectories

    Returns:
        Dictionary with statistics:
            - 'total': Total files found
            - 'organized': Number of files successfully organized
            - 'skipped': Number of files skipped (already organized)
            - 'failed': Number of files that failed to organize
    """
    logger.info(f"Starting organization of directory: {directory}")
    logger.info(f"Dry run: {dry_run}, Recursive: {recursive}")

    # Find all audio files
    audio_files = find_audio_files(directory, recursive=recursive)

    if not audio_files:
        logger.warning(f"No audio files found in {directory}")
        return {'total': 0, 'organized': 0, 'skipped': 0, 'failed': 0}

    stats = {
        'total': len(audio_files),
        'organized': 0,
        'skipped': 0,
        'failed': 0
    }

    logger.info(f"Found {stats['total']} audio files")

    # Process each file
    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"Processing {i}/{stats['total']}: {audio_file.name}")

        success, message = organize_file(audio_file, dry_run=dry_run)

        if success:
            if "Already organized" in message:
                stats['skipped'] += 1
            else:
                stats['organized'] += 1
        else:
            stats['failed'] += 1
            logger.warning(message)

    # Print summary
    logger.info("=" * 60)
    logger.info("Organization Summary:")
    logger.info(f"  Total files found:    {stats['total']}")
    logger.info(f"  Successfully organized: {stats['organized']}")
    logger.info(f"  Already organized:    {stats['skipped']}")
    logger.info(f"  Failed:               {stats['failed']}")
    logger.info("=" * 60)

    return stats


def main():
    """Main entry point for the file organizer script."""
    parser = argparse.ArgumentParser(
        description="Organize audio files into the MIR pipeline folder structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Organize all audio files in a directory:
    python file_organizer.py /path/to/audio

  Test organization without moving files:
    python file_organizer.py /path/to/audio --dry-run

  Organize only files in the root directory (not recursive):
    python file_organizer.py /path/to/audio --no-recursive

  Enable debug logging:
    python file_organizer.py /path/to/audio --verbose
        """
    )

    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing audio files to organize'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate organization without actually moving files'
    )

    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories (only process root level)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (debug) logging'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        help='Write logs to specified file'
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, log_file=args.log_file)

    # Validate directory
    directory = Path(args.directory)
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return 1

    if not directory.is_dir():
        logger.error(f"Path is not a directory: {directory}")
        return 1

    # Run organization
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be moved")

    stats = organize_directory(
        directory,
        dry_run=args.dry_run,
        recursive=not args.no_recursive
    )

    # Return appropriate exit code
    if stats['failed'] > 0:
        logger.warning("Some files failed to organize")
        return 1

    logger.info("Organization completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
