"""
File Organizer for MIR Project

This script organizes audio files into the required folder structure for the MIR pipeline.

Input: Audio files in any location
Output: Organized structure with files copied to output directory (preserves originals by default)

Structure:
    /input/path/filename.flac -> /output/path/filename/full_mix.flac

Features:
- Automatically fixes "Various Artists" track names using Spotify/MusicBrainz lookup
- Skips already organized files (full_mix, stems, etc.)

Dependencies:
- src.core.file_utils
- src.core.common

Usage:
    # Copy files to output directory (preserves originals - RECOMMENDED)
    python file_organizer.py /path/to/audio/files --output-dir /path/to/organized

    # With metadata lookup for Various Artists tracks
    python file_organizer.py /path/to/audio/files --output-dir /path/to/organized --fix-various

    # Move files in place (DESTRUCTIVE - deletes originals)
    python file_organizer.py /path/to/audio/files --move

    # Test without copying/moving
    python file_organizer.py /path/to/audio/files --dry-run
"""

import argparse
import shutil
import re
from pathlib import Path
from typing import List, Tuple, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import find_audio_files, get_audio_folder_structure, is_organized
from core.common import AUDIO_EXTENSIONS, setup_logging
from preprocessing.filename_cleanup import clean_filename

logger = logging.getLogger(__name__)

# Various Artists aliases (case-insensitive)
VARIOUS_ARTISTS_ALIASES = {
    'various artists', 'various', 'va', 'v/a', 'v.a.',
    'compilation', 'unknown artist', 'unknown'
}

# Module-level Spotify client (initialized once)
_spotify_client = None


def _init_metadata_lookup():
    """Initialize Spotify client for metadata lookup (called once)."""
    global _spotify_client
    if _spotify_client is not None:
        return _spotify_client

    try:
        from tools.track_metadata_lookup import init_spotify
        _spotify_client = init_spotify()
        if _spotify_client:
            logger.info("Spotify API initialized for metadata lookup")
        return _spotify_client
    except ImportError:
        logger.warning("track_metadata_lookup not available")
        return None


def _is_various_artists(filename: str) -> bool:
    """Check if filename contains a 'Various Artists' pattern."""
    # Remove leading track numbers
    name = re.sub(r'^\d+\.?\s*', '', filename).strip()

    # Check for patterns like "Various Artists - Track" or "VA - Track"
    parts = name.split(' - ')
    if parts:
        artist_part = parts[0].strip().lower()
        return artist_part in VARIOUS_ARTISTS_ALIASES

    return False


def _lookup_correct_name(filename: str, audio_path: Path = None) -> Optional[str]:
    """
    Look up the correct artist/track name for a Various Artists file.

    Returns the corrected folder name, or None if lookup fails.
    """
    sp = _init_metadata_lookup()
    if not sp:
        return None

    try:
        from tools.track_metadata_lookup import extract_track_name, lookup_track

        # Extract track name from filename
        artist, album, track_name = extract_track_name(filename)

        if not track_name:
            logger.debug(f"Could not extract track name from: {filename}")
            return None

        logger.info(f"Looking up: {track_name}")

        # Look up metadata
        result = lookup_track(track_name, artist_hint=None, sp=sp)

        if not result:
            logger.warning(f"No match found for: {track_name}")
            return None

        # Build corrected folder name
        corrected_name = f"{result['artist']} - {result.get('track', track_name)}"
        # Clean up for filesystem
        corrected_name = re.sub(r'[<>:"/\\|?*]', '', corrected_name)

        logger.info(f"  Found: {corrected_name}")
        return corrected_name

    except Exception as e:
        logger.debug(f"Metadata lookup failed: {e}")
        return None


def organize_file(audio_file: Path,
                  output_dir: Path = None,
                  move: bool = False,
                  dry_run: bool = False,
                  fix_various: bool = False) -> Tuple[bool, str]:
    """
    Organize a single audio file into the required folder structure.

    Args:
        audio_file: Path to the audio file
        output_dir: Output directory for organized files (None = organize in place)
        move: If True, move files (destructive). If False, copy files (preserves originals)
        dry_run: If True, don't actually copy/move files (just simulate)
        fix_various: If True, look up correct artist for "Various Artists" tracks

    Returns:
        Tuple of (success: bool, message: str)
    """
    # Skip files that are already organized (full_mix.*)
    if audio_file.stem == 'full_mix':
        return True, f"Already organized (full_mix): {audio_file}"

    # Skip stem files (drums, bass, other, vocals)
    stem_names = {'drums', 'bass', 'other', 'vocals', 'piano', 'guitar'}
    if audio_file.stem.lower() in stem_names:
        return True, f"Skipped stem file: {audio_file}"

    # Skip files that look like crop stems (e.g., TrackName_0_drums)
    for stem in stem_names:
        if audio_file.stem.lower().endswith(f'_{stem}'):
            return True, f"Skipped crop stem file: {audio_file}"

    # Skip files inside already-organized folders (folder contains full_mix.*)
    if any(audio_file.parent.glob('full_mix.*')):
        return True, f"Skipped (in organized folder): {audio_file}"

    # Check if already organized (legacy check for in-place)
    if output_dir is None and is_organized(audio_file):
        return True, f"Already organized: {audio_file}"

    # Get expected structure
    if output_dir is not None:
        # Create structure in output directory
        # Extract folder name from audio file
        folder_name = audio_file.stem  # filename without extension

        # Remove leading track number (e.g., "013 Track", "02. Track", "1 - Track")
        track_num_match = re.match(r'^(\d{1,3})[\.\s\-_]+', folder_name)
        if track_num_match:
            original_name = folder_name
            folder_name = folder_name[track_num_match.end():].lstrip()
            logger.debug(f"Removed track number: {original_name} -> {folder_name}")

        # Fix "Various Artists" names if requested
        if fix_various and _is_various_artists(folder_name):
            corrected_name = _lookup_correct_name(folder_name, audio_file)
            if corrected_name:
                logger.info(f"  Corrected: {folder_name} -> {corrected_name}")
                folder_name = corrected_name

        folder_path = output_dir / folder_name
        full_mix_path = folder_path / f"full_mix{audio_file.suffix}"

        # Also check for cleaned version of the folder name (for resume after filename cleanup)
        cleaned_folder_name = clean_filename(folder_name)
        cleaned_folder_path = output_dir / cleaned_folder_name
    else:
        # Use existing logic for in-place organization
        structure = get_audio_folder_structure(audio_file)
        folder_path = structure['folder']
        full_mix_path = structure['full_mix']
        cleaned_folder_path = None  # Not needed for in-place organization

    # Check if target already exists (skip, not fail)
    if full_mix_path.exists():
        return True, f"Already exists at destination: {full_mix_path}"

    # Check if folder already exists with different content (potential conflict)
    if folder_path.exists() and any(folder_path.glob('full_mix.*')):
        return True, f"Folder already has full_mix: {folder_path}"

    # Check if cleaned version of folder already exists (resume after filename cleanup)
    if cleaned_folder_path and cleaned_folder_path != folder_path:
        if cleaned_folder_path.exists() and any(cleaned_folder_path.glob('full_mix.*')):
            return True, f"Already exists (cleaned name): {cleaned_folder_path}"

    # Determine operation
    operation = "move" if move else "copy"

    if dry_run:
        logger.info(f"[DRY RUN] Would {operation}: {audio_file} -> {full_mix_path}")
        return True, f"Would organize: {audio_file}"

    try:
        # Create the folder
        folder_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created folder: {folder_path}")

        # Copy or move the file
        if move:
            shutil.move(str(audio_file), str(full_mix_path))
            logger.info(f"Moved: {audio_file} -> {full_mix_path}")
        else:
            shutil.copy2(str(audio_file), str(full_mix_path))
            logger.info(f"Copied: {audio_file} -> {full_mix_path}")

        return True, f"Successfully organized: {audio_file}"

    except Exception as e:
        logger.error(f"Error organizing {audio_file}: {e}")
        return False, f"Error: {e}"


def organize_directory(directory: Path,
                        output_dir: Path = None,
                        move: bool = False,
                        dry_run: bool = False,
                        recursive: bool = True,
                        fix_various: bool = False) -> dict:
    """
    Organize all audio files in a directory.

    Args:
        directory: Root directory containing audio files
        output_dir: Output directory for organized files (None = organize in place)
        move: If True, move files (destructive). If False, copy files (preserves originals)
        dry_run: If True, don't actually copy/move files
        recursive: If True, search subdirectories
        fix_various: If True, look up correct artist for "Various Artists" tracks

    Returns:
        Dictionary with statistics:
            - 'total': Total files found
            - 'organized': Number of files successfully organized
            - 'skipped': Number of files skipped (already organized)
            - 'failed': Number of files that failed to organize
    """
    logger.info(f"Starting organization of directory: {directory}")
    if output_dir:
        logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mode: {'MOVE (destructive)' if move else 'COPY (preserves originals)'}")
    logger.info(f"Dry run: {dry_run}, Recursive: {recursive}, Fix Various Artists: {fix_various}")

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

        success, message = organize_file(
            audio_file,
            output_dir=output_dir,
            move=move,
            dry_run=dry_run,
            fix_various=fix_various
        )

        if success:
            # Check for various skip conditions
            skip_keywords = ['Already', 'Skipped', 'exists']
            if any(kw in message for kw in skip_keywords):
                stats['skipped'] += 1
                logger.debug(message)
            else:
                stats['organized'] += 1
        else:
            stats['failed'] += 1
            logger.warning(message)

    # Print summary
    logger.info("=" * 60)
    logger.info("Organization Summary:")
    logger.info(f"  Total files found:      {stats['total']}")
    logger.info(f"  Successfully organized: {stats['organized']}")
    logger.info(f"  Already organized:      {stats['skipped']}")
    logger.info(f"  Failed:                 {stats['failed']}")
    logger.info("=" * 60)

    return stats


def main():
    """Main entry point for the file organizer script."""
    parser = argparse.ArgumentParser(
        description="Organize audio files into the MIR pipeline folder structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Copy files to output directory (RECOMMENDED - preserves originals):
    python file_organizer.py /path/to/audio --output-dir /path/to/organized

  Fix "Various Artists" names during organization (RECOMMENDED):
    python file_organizer.py /path/to/audio --output-dir /path/to/organized --fix-various

  Organize in place by copying (preserves originals):
    python file_organizer.py /path/to/audio

  Organize in place by moving (DESTRUCTIVE - deletes originals):
    python file_organizer.py /path/to/audio --move

  Test organization without copying/moving files:
    python file_organizer.py /path/to/audio --dry-run --output-dir /path/to/organized --fix-various

  Organize only files in the root directory (not recursive):
    python file_organizer.py /path/to/audio --output-dir /path/to/organized --no-recursive

  Enable debug logging:
    python file_organizer.py /path/to/audio --output-dir /path/to/organized --verbose
        """
    )

    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing audio files to organize'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for organized files (preserves originals in source location)'
    )

    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying (DESTRUCTIVE - deletes originals)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate organization without actually copying/moving files'
    )

    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories (only process root level)'
    )

    parser.add_argument(
        '--fix-various',
        action='store_true',
        help='Look up correct artist for "Various Artists" tracks using Spotify/MusicBrainz'
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

    # Parse output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")

    # Warn about destructive operations
    if args.move:
        logger.warning("WARNING: --move flag is DESTRUCTIVE and will delete original files!")
        logger.warning("Original files will NOT be recoverable after moving.")
        if not args.dry_run:
            logger.warning("Press Ctrl+C within 3 seconds to cancel...")
            import time
            time.sleep(3)

    # Run organization
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be copied/moved")
    elif args.move:
        logger.info("MOVE MODE - Original files will be deleted")
    else:
        logger.info("COPY MODE - Original files will be preserved")

    # Initialize metadata lookup if fixing Various Artists
    if args.fix_various:
        logger.info("FIX VARIOUS ARTISTS - Will look up correct names via Spotify/MusicBrainz")
        _init_metadata_lookup()

    stats = organize_directory(
        directory,
        output_dir=output_dir,
        move=args.move,
        dry_run=args.dry_run,
        recursive=not args.no_recursive,
        fix_various=args.fix_various
    )

    # Return appropriate exit code
    if stats['failed'] > 0:
        logger.warning("Some files failed to organize")
        return 1

    logger.info("Organization completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
