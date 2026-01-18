"""
Batch Processing Utilities

Provides utilities for robust batch processing:
- File locking to prevent conflicts
- Resume capability (skip already-processed files)
- Feature presence checking
- Progress tracking

Usage:
    from core.batch_utils import should_process_folder, FileLock

    for folder in folders:
        # Check if should process (not locked, features missing or overwrite)
        if not should_process_folder(folder, required_features=['lufs', 'lra'], overwrite=False):
            logger.info(f"Skipping {folder.name} - already processed")
            continue

        # Acquire lock
        with FileLock(folder) as lock:
            if not lock.acquired:
                logger.info(f"Skipping {folder.name} - being processed by another worker")
                continue

            # Process folder
            process_folder(folder)
"""

import json
from pathlib import Path
from typing import List, Optional, Set, Dict
import logging

from core.file_locks import FileLock, is_locked
from core.json_handler import get_info_path
from core.file_utils import get_stem_files

logger = logging.getLogger(__name__)


def has_features(folder: Path, required_features: List[str]) -> bool:
    """
    Check if a folder already has the required features in its .INFO file.

    Args:
        folder: Path to organized folder
        required_features: List of feature keys to check for

    Returns:
        True if all required features exist, False otherwise
    """
    # Find .INFO file
    stems = get_stem_files(folder, include_full_mix=True)
    if 'full_mix' not in stems:
        return False

    info_path = get_info_path(stems['full_mix'])

    if not info_path.exists():
        return False

    try:
        with open(info_path, 'r') as f:
            data = json.load(f)

        # Check if all required features are present and not None
        for feature in required_features:
            if feature not in data or data[feature] is None:
                return False

        return True

    except Exception as e:
        logger.debug(f"Error reading .INFO file for {folder.name}: {e}")
        return False


def get_missing_features(folder: Path, required_features: List[str]) -> Set[str]:
    """
    Get list of missing features from a folder's .INFO file.

    Args:
        folder: Path to organized folder
        required_features: List of feature keys to check for

    Returns:
        Set of missing feature keys
    """
    missing = set()

    # Find .INFO file
    stems = get_stem_files(folder, include_full_mix=True)
    if 'full_mix' not in stems:
        return set(required_features)

    info_path = get_info_path(stems['full_mix'])

    if not info_path.exists():
        return set(required_features)

    try:
        with open(info_path, 'r') as f:
            data = json.load(f)

        # Check each feature
        for feature in required_features:
            if feature not in data or data[feature] is None:
                missing.add(feature)

        return missing

    except Exception as e:
        logger.debug(f"Error reading .INFO file for {folder.name}: {e}")
        return set(required_features)


def should_process_folder(
    folder: Path,
    required_features: Optional[List[str]] = None,
    overwrite: bool = False,
    check_lock: bool = True
) -> bool:
    """
    Determine if a folder should be processed.

    A folder should be processed if:
    1. overwrite=True, OR
    2. Required features are missing, AND
    3. (check_lock=False OR folder is not locked)

    Args:
        folder: Path to organized folder
        required_features: List of features to check for (None = always process)
        overwrite: If True, process even if features exist
        check_lock: If True, skip if folder is locked

    Returns:
        True if folder should be processed
    """
    # If overwrite, always process (unless locked)
    if overwrite:
        if check_lock and is_locked(folder):
            logger.debug(f"Skipping {folder.name} - locked by another process")
            return False
        return True

    # Check if features exist
    if required_features:
        if has_features(folder, required_features):
            logger.debug(f"Skipping {folder.name} - features already exist")
            return False

    # Check if locked
    if check_lock and is_locked(folder):
        logger.debug(f"Skipping {folder.name} - locked by another process")
        return False

    return True


def filter_folders_to_process(
    folders: List[Path],
    required_features: Optional[List[str]] = None,
    overwrite: bool = False,
    check_lock: bool = True
) -> tuple[List[Path], Dict[str, int]]:
    """
    Filter list of folders to only those that need processing.

    Args:
        folders: List of folder paths
        required_features: Features to check for
        overwrite: Process even if features exist
        check_lock: Skip locked folders

    Returns:
        Tuple of (folders_to_process, stats_dict)
        where stats_dict contains counts of skipped/locked/ready folders
    """
    folders_to_process = []
    stats = {
        'total': len(folders),
        'ready': 0,
        'skipped_complete': 0,
        'skipped_locked': 0
    }

    for folder in folders:
        # Check if locked
        if check_lock and is_locked(folder):
            stats['skipped_locked'] += 1
            logger.info(f"Skipping {folder.name} - locked by another process")
            continue

        # If overwrite, add to queue
        if overwrite:
            folders_to_process.append(folder)
            stats['ready'] += 1
            continue

        # Check if features exist
        if required_features and has_features(folder, required_features):
            stats['skipped_complete'] += 1
            logger.info(f"Skipping {folder.name} - features already exist")
            continue

        # Need processing
        folders_to_process.append(folder)
        stats['ready'] += 1

    return folders_to_process, stats


class BatchProcessor:
    """
    Base class for batch processing with locking and resume support.

    Handles:
    - File locking
    - Skip already-processed files
    - Progress tracking
    - Error handling

    Usage:
        processor = BatchProcessor(
            required_features=['lufs', 'lra'],
            lock_timeout=3600
        )

        results = processor.process_batch(
            folders=folders,
            process_func=analyze_loudness,
            overwrite=False
        )
    """

    def __init__(
        self,
        required_features: Optional[List[str]] = None,
        lock_timeout: Optional[float] = None
    ):
        """
        Initialize batch processor.

        Args:
            required_features: Features to check for when resuming
            lock_timeout: Lock timeout in seconds (None = use default)
        """
        self.required_features = required_features
        self.lock_timeout = lock_timeout

    def process_batch(
        self,
        folders: List[Path],
        process_func: callable,
        overwrite: bool = False,
        skip_errors: bool = True
    ) -> Dict[str, any]:
        """
        Process a batch of folders with locking and resume support.

        Args:
            folders: List of folders to process
            process_func: Function to call for each folder (takes folder Path)
            overwrite: Process even if features exist
            skip_errors: Continue processing on errors

        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'total': len(folders),
            'success': 0,
            'skipped_complete': 0,
            'skipped_locked': 0,
            'failed': 0,
            'errors': []
        }

        for i, folder in enumerate(folders, 1):
            logger.info(f"Processing {i}/{stats['total']}: {folder.name}")

            # Check if should process
            if not overwrite and self.required_features:
                if has_features(folder, self.required_features):
                    stats['skipped_complete'] += 1
                    logger.info(f"  Skipping - features already exist")
                    continue

            # Try to acquire lock
            with FileLock(folder, timeout=self.lock_timeout) as lock:
                if not lock.acquired:
                    stats['skipped_locked'] += 1
                    logger.info(f"  Skipping - locked by another process")
                    continue

                # Process folder
                try:
                    process_func(folder)
                    stats['success'] += 1
                    logger.info(f"  ✓ Completed")

                except Exception as e:
                    stats['failed'] += 1
                    error_msg = f"{folder.name}: {str(e)}"
                    stats['errors'].append(error_msg)
                    logger.error(f"  ✗ Failed: {e}")

                    if not skip_errors:
                        raise

        return stats


def print_batch_summary(stats: Dict[str, any], operation_name: str = "Processing"):
    """
    Print a formatted summary of batch processing results.

    Args:
        stats: Statistics dictionary from batch processing
        operation_name: Name of the operation (e.g., "Loudness Analysis")
    """
    logger.info("=" * 60)
    logger.info(f"{operation_name} Summary:")
    logger.info(f"  Total folders:      {stats['total']}")
    logger.info(f"  Successfully processed: {stats['success']}")

    if 'skipped_complete' in stats and stats['skipped_complete'] > 0:
        logger.info(f"  Skipped (complete): {stats['skipped_complete']}")

    if 'skipped_locked' in stats and stats['skipped_locked'] > 0:
        logger.info(f"  Skipped (locked):   {stats['skipped_locked']}")

    if 'failed' in stats and stats['failed'] > 0:
        logger.info(f"  Failed:             {stats['failed']}")

    logger.info("=" * 60)

    if stats.get('errors'):
        logger.warning("Errors encountered:")
        for error in stats['errors'][:10]:  # Show first 10 errors
            logger.warning(f"  {error}")
        if len(stats['errors']) > 10:
            logger.warning(f"  ... and {len(stats['errors']) - 10} more errors")


def get_progress_stats(root_directory: Path, required_features: List[str]) -> Dict[str, int]:
    """
    Get statistics on processing progress for a dataset.

    Args:
        root_directory: Root directory to scan
        required_features: Features to check for

    Returns:
        Dictionary with progress statistics
    """
    from core.file_utils import find_organized_folders

    folders = find_organized_folders(root_directory)

    stats = {
        'total': len(folders),
        'complete': 0,
        'incomplete': 0,
        'locked': 0
    }

    for folder in folders:
        if is_locked(folder):
            stats['locked'] += 1
        elif has_features(folder, required_features):
            stats['complete'] += 1
        else:
            stats['incomplete'] += 1

    return stats


# Command-line interface
if __name__ == "__main__":
    import argparse
    from core.common import setup_logging
    from core.file_utils import find_organized_folders

    parser = argparse.ArgumentParser(
        description="Batch processing utilities"
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Progress command
    progress_parser = subparsers.add_parser(
        'progress',
        help='Check processing progress'
    )
    progress_parser.add_argument(
        'directory',
        type=str,
        help='Directory to scan'
    )
    progress_parser.add_argument(
        '--features',
        nargs='+',
        required=True,
        help='Features to check for (e.g., lufs lra bpm)'
    )

    # Check command
    check_parser = subparsers.add_parser(
        'check',
        help='Check which folders need processing'
    )
    check_parser.add_argument(
        'directory',
        type=str,
        help='Directory to scan'
    )
    check_parser.add_argument(
        '--features',
        nargs='+',
        required=True,
        help='Features to check for'
    )

    # Missing command
    missing_parser = subparsers.add_parser(
        'missing',
        help='Show missing features for each folder'
    )
    missing_parser.add_argument(
        'directory',
        type=str,
        help='Directory to scan'
    )
    missing_parser.add_argument(
        '--features',
        nargs='+',
        required=True,
        help='Features to check for'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    if args.command == 'progress':
        stats = get_progress_stats(Path(args.directory), args.features)
        print(f"\nProcessing Progress:")
        print(f"  Total folders:  {stats['total']}")
        print(f"  Complete:       {stats['complete']} ({stats['complete']/stats['total']*100:.1f}%)")
        print(f"  Incomplete:     {stats['incomplete']}")
        print(f"  Locked:         {stats['locked']}")

    elif args.command == 'check':
        folders = find_organized_folders(Path(args.directory))
        to_process, stats = filter_folders_to_process(
            folders,
            required_features=args.features
        )

        print(f"\nFolders to Process: {len(to_process)}")
        print(f"  Total:          {stats['total']}")
        print(f"  Ready:          {stats['ready']}")
        print(f"  Skipped (done): {stats['skipped_complete']}")
        print(f"  Skipped (lock): {stats['skipped_locked']}")

    elif args.command == 'missing':
        folders = find_organized_folders(Path(args.directory))

        print(f"\nMissing Features Analysis:")
        incomplete_count = 0

        for folder in folders:
            missing = get_missing_features(folder, args.features)
            if missing:
                incomplete_count += 1
                print(f"\n{folder.name}:")
                print(f"  Missing: {', '.join(sorted(missing))}")

        print(f"\nSummary: {incomplete_count}/{len(folders)} folders incomplete")

    else:
        parser.print_help()
