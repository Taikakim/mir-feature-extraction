"""
File Locking System for Parallel Processing

Provides safe file locking to prevent multiple processes from processing
the same file simultaneously. Essential for parallel batch processing.

Features:
- Exclusive lock acquisition with fcntl
- Dead lock detection and cleanup
- Process tracking (PID in lock file)
- Context manager for automatic cleanup
- Timeout support for lock acquisition

Usage:
    # Method 1: Context manager (recommended)
    with FileLock(audio_folder) as lock:
        if lock.acquired:
            process_folder(audio_folder)
        else:
            logger.info(f"Skipping {audio_folder.name} - already being processed")

    # Method 2: Manual locking
    lock = FileLock(audio_folder)
    if lock.acquire():
        try:
            process_folder(audio_folder)
        finally:
            lock.release()
"""

import fcntl
import os
import time
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FileLock:
    """
    File-based lock for coordinating parallel processing.

    Uses fcntl for exclusive file locking, with dead lock detection
    and automatic cleanup.

    Attributes:
        file_path: Path to the file/folder being locked
        lock_path: Path to the .lock file
        acquired: Whether the lock was successfully acquired
        lock_file: File handle for the lock file
    """

    # How old a lock file can be before considered "dead" (in seconds)
    LOCK_TIMEOUT = 3600  # 1 hour - adjust based on expected processing time

    def __init__(self, file_path: Path, timeout: Optional[float] = None):
        """
        Initialize a file lock.

        Args:
            file_path: Path to file or folder to lock
            timeout: Maximum age (in seconds) for considering a lock "dead"
                    If None, uses class default LOCK_TIMEOUT
        """
        self.file_path = Path(file_path)
        self.lock_path = self.file_path.with_suffix('.lock')
        self.acquired = False
        self.lock_file = None
        self.timeout = timeout if timeout is not None else self.LOCK_TIMEOUT

    def _is_lock_dead(self) -> bool:
        """
        Check if an existing lock file is "dead" (stale).

        A lock is considered dead if:
        1. The lock file is older than timeout
        2. The process that created it no longer exists

        Returns:
            True if the lock is dead and can be removed
        """
        if not self.lock_path.exists():
            return False

        try:
            # Check lock file age
            lock_age = time.time() - self.lock_path.stat().st_mtime

            if lock_age > self.timeout:
                logger.warning(
                    f"Found stale lock file (age: {lock_age/60:.1f} minutes): {self.lock_path.name}"
                )

                # Try to read PID from lock file
                try:
                    with open(self.lock_path, 'r') as f:
                        content = f.read()
                        # Extract PID if present
                        if 'PID:' in content:
                            pid = int(content.split('PID:')[1].split()[0])

                            # Check if process still exists
                            try:
                                os.kill(pid, 0)  # Signal 0 checks if process exists
                                # Process exists - lock might be valid
                                logger.warning(f"  Process {pid} still exists - may be slow processing")
                                return False  # Don't remove - process might be working
                            except OSError:
                                # Process doesn't exist - lock is definitely dead
                                logger.warning(f"  Process {pid} no longer exists - removing dead lock")
                                return True
                except Exception as e:
                    logger.warning(f"  Could not check lock file process: {e}")

                # If we can't verify the process, assume lock is dead after timeout
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking lock file: {e}")
            return False

    def acquire(self, wait: bool = False, poll_interval: float = 1.0) -> bool:
        """
        Acquire the lock.

        Args:
            wait: If True, wait for lock to become available
            poll_interval: Seconds to wait between lock attempts (if wait=True)

        Returns:
            True if lock was acquired, False otherwise
        """
        # Clean up dead locks
        if self._is_lock_dead():
            logger.info(f"Removing dead lock: {self.lock_path.name}")
            try:
                self.lock_path.unlink()
            except Exception as e:
                logger.error(f"Failed to remove dead lock: {e}")
                return False

        # Try to acquire lock
        while True:
            try:
                # Create lock file with exclusive access
                # Use 'x' mode to fail if file exists
                self.lock_file = open(self.lock_path, 'x')

                # Try to get an exclusive lock
                try:
                    fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                    # Write process info to lock file
                    lock_info = (
                        f"Processing started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"PID: {os.getpid()}\n"
                        f"File: {self.file_path.name}\n"
                    )
                    self.lock_file.write(lock_info)
                    self.lock_file.flush()

                    self.acquired = True
                    logger.debug(f"Acquired lock: {self.lock_path.name}")
                    return True

                except IOError:
                    # Another process has the lock
                    self.lock_file.close()
                    self.lock_file = None

                    if not wait:
                        logger.debug(f"Lock already held: {self.lock_path.name}")
                        return False

                    # Wait and retry
                    time.sleep(poll_interval)
                    continue

            except FileExistsError:
                # Lock file already exists
                if not wait:
                    logger.debug(f"Lock file exists: {self.lock_path.name}")
                    return False

                # Wait and retry
                time.sleep(poll_interval)
                continue

            except Exception as e:
                logger.error(f"Error acquiring lock for {self.file_path.name}: {e}")
                if self.lock_file:
                    try:
                        self.lock_file.close()
                    except:
                        pass
                    self.lock_file = None
                return False

    def release(self) -> None:
        """Release the lock."""
        if not self.acquired:
            return

        try:
            # Release fcntl lock
            if self.lock_file:
                try:
                    fcntl.flock(self.lock_file, fcntl.LOCK_UN)
                except:
                    pass

                # Close file
                try:
                    self.lock_file.close()
                except:
                    pass

                self.lock_file = None

            # Remove lock file
            if self.lock_path.exists():
                self.lock_path.unlink()
                logger.debug(f"Released lock: {self.lock_path.name}")

            self.acquired = False

        except Exception as e:
            logger.error(f"Error releasing lock for {self.file_path.name}: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False

    def __del__(self):
        """Destructor - ensure lock is released."""
        self.release()


def cleanup_dead_locks(root_directory: Path, timeout: Optional[float] = None) -> int:
    """
    Scan directory tree for dead lock files and remove them.

    Useful for cleaning up after crashes or interrupted processing.

    Args:
        root_directory: Directory to scan
        timeout: Maximum lock age (seconds) before considering dead

    Returns:
        Number of dead locks removed
    """
    root_directory = Path(root_directory)
    removed_count = 0

    logger.info(f"Scanning for dead locks in: {root_directory}")

    # Find all .lock files
    lock_files = list(root_directory.rglob('*.lock'))

    logger.info(f"Found {len(lock_files)} lock files")

    for lock_file in lock_files:
        # Get the original file path
        original_path = lock_file.with_suffix('')

        # Create FileLock instance to check if dead
        lock = FileLock(original_path, timeout=timeout)

        if lock._is_lock_dead():
            try:
                lock_file.unlink()
                logger.info(f"Removed dead lock: {lock_file.name}")
                removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove lock {lock_file.name}: {e}")

    logger.info(f"Removed {removed_count} dead lock files")
    return removed_count


def remove_all_locks(root_directory: Path) -> int:
    """
    Remove ALL lock files in a directory tree, regardless of state.

    Use this for recovery after crashes or interrupted processing.
    WARNING: This will remove locks even if processes are still running.

    Args:
        root_directory: Directory to scan

    Returns:
        Number of lock files removed
    """
    root_directory = Path(root_directory)
    removed_count = 0

    lock_files = list(root_directory.rglob('*.lock'))

    for lock_file in lock_files:
        try:
            lock_file.unlink()
            removed_count += 1
        except Exception as e:
            logger.warning(f"Failed to remove {lock_file.name}: {e}")

    if removed_count > 0:
        logger.info(f"Removed {removed_count} lock files from {root_directory}")

    return removed_count


def is_locked(file_path: Path) -> bool:
    """
    Check if a file is currently locked (without acquiring lock).

    Args:
        file_path: Path to check

    Returns:
        True if file has an active lock, False otherwise
    """
    lock_path = Path(file_path).with_suffix('.lock')

    # No lock file = not locked
    if not lock_path.exists():
        return False

    # Check if it's a dead lock
    lock = FileLock(file_path)
    if lock._is_lock_dead():
        return False

    # Lock file exists and is not dead
    return True


# Command-line interface for lock management
if __name__ == "__main__":
    import argparse
    from core.common import setup_logging

    parser = argparse.ArgumentParser(
        description="File lock management utilities"
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Cleanup command
    cleanup_parser = subparsers.add_parser(
        'cleanup',
        help='Clean up dead lock files'
    )
    cleanup_parser.add_argument(
        'directory',
        type=str,
        help='Directory to scan for dead locks'
    )
    cleanup_parser.add_argument(
        '--timeout',
        type=float,
        default=3600,
        help='Lock timeout in seconds (default: 3600 = 1 hour)'
    )

    # List command
    list_parser = subparsers.add_parser(
        'list',
        help='List all lock files'
    )
    list_parser.add_argument(
        'directory',
        type=str,
        help='Directory to scan'
    )

    # Check command
    check_parser = subparsers.add_parser(
        'check',
        help='Check if a specific file is locked'
    )
    check_parser.add_argument(
        'file',
        type=str,
        help='File to check'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    if args.command == 'cleanup':
        removed = cleanup_dead_locks(
            Path(args.directory),
            timeout=args.timeout
        )
        print(f"Removed {removed} dead lock files")

    elif args.command == 'list':
        lock_files = list(Path(args.directory).rglob('*.lock'))
        print(f"Found {len(lock_files)} lock files:")
        for lock_file in lock_files:
            lock = FileLock(lock_file.with_suffix(''))
            status = "DEAD" if lock._is_lock_dead() else "ACTIVE"
            age = time.time() - lock_file.stat().st_mtime
            print(f"  [{status}] {lock_file.name} (age: {age/60:.1f} min)")

    elif args.command == 'check':
        file_path = Path(args.file)
        if is_locked(file_path):
            print(f"LOCKED: {file_path.name}")
            lock_path = file_path.with_suffix('.lock')
            if lock_path.exists():
                with open(lock_path, 'r') as f:
                    print(f.read())
        else:
            print(f"NOT LOCKED: {file_path.name}")

    else:
        parser.print_help()
