"""
JSON Handler for MIR Project

This module provides safe JSON read/write operations for .INFO and .MIR files.
It ensures that existing data is never accidentally destroyed when adding new features.

Process Safety:
- Uses filelock.FileLock for cross-process locking (works with ProcessPoolExecutor)
- Safe for use with both ThreadPoolExecutor and ProcessPoolExecutor
- Uses atomic writes (temp file + rename) for crash safety

Dependencies:
- filelock (pip install filelock)

Functions:
- read_info(file_path): Read .INFO file, return dict or empty dict if not exists
- write_info(file_path, data, merge=True): Write data to .INFO file with optional merge
- read_mir(file_path): Read .MIR file (temporal data)
- write_mir(file_path, data, merge=True): Write data to .MIR file with optional merge
- safe_update(file_path, updates): Safely update specific keys without destroying others
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

try:
    from filelock import FileLock
    FILELOCK_AVAILABLE = True
except ImportError:
    FILELOCK_AVAILABLE = False

# Always import threading for fallback registry
import threading
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe lock registry (fallback when filelock not available)
_thread_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
_registry_lock = threading.Lock()


def _get_lock_path(file_path: Path) -> Path:
    """Get the lock file path for a given file."""
    return file_path.with_suffix(file_path.suffix + '.lock')


def _get_thread_lock(file_path: Path) -> threading.Lock:
    """Get or create a thread lock for a file (fallback for single-process use)."""
    key = str(file_path.resolve())
    with _registry_lock:
        if key not in _thread_locks:
            _thread_locks[key] = threading.Lock()
        return _thread_locks[key]


def read_info(file_path: str | Path) -> Dict[str, Any]:
    """
    Read a .INFO file and return its contents as a dictionary.

    Args:
        file_path: Path to the .INFO file

    Returns:
        Dictionary with file contents, or empty dict if file doesn't exist

    Raises:
        json.JSONDecodeError: If file exists but contains invalid JSON
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.debug(f"INFO file does not exist: {file_path}")
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.debug(f"Successfully read {len(data)} keys from {file_path}")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        raise


def write_info(file_path: str | Path, data: Dict[str, Any], merge: bool = True) -> None:
    """
    Write data to a .INFO file. By default, merges with existing data.

    Process-safe: uses filelock for cross-process locking (works with ProcessPoolExecutor).

    Args:
        file_path: Path to the .INFO file
        data: Dictionary to write
        merge: If True, merge with existing data (preserving existing keys)
               If False, completely replace file contents

    Raises:
        IOError: If file cannot be written
    """
    file_path = Path(file_path)

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Process-safe: acquire file lock for the entire read-modify-write cycle
    if FILELOCK_AVAILABLE:
        lock_path = _get_lock_path(file_path)
        lock = FileLock(lock_path, timeout=60)  # 60 second timeout
    else:
        # Fallback for single-process use (thread-safe only)
        lock = _get_thread_lock(file_path)

    with lock:
        # If merging, read existing data first
        if merge and file_path.exists():
            try:
                existing_data = read_info(file_path)
                # Merge: new data overwrites existing keys
                existing_data.update(data)
                data = existing_data
                logger.debug(f"Merged data with existing {file_path}")
            except Exception as e:
                logger.warning(f"Could not read existing data from {file_path}, will overwrite: {e}")

        # Write atomically using a temporary file
        temp_path = file_path.with_suffix('.INFO.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_path.replace(file_path)
            logger.debug(f"Successfully wrote {len(data)} keys to {file_path}")

        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Error writing {file_path}: {e}")
            raise


def read_mir(file_path: str | Path) -> Dict[str, Any]:
    """
    Read a .MIR file (temporal data) and return its contents.

    Args:
        file_path: Path to the .MIR file

    Returns:
        Dictionary with temporal data, or empty dict if file doesn't exist

    Raises:
        json.JSONDecodeError: If file exists but contains invalid JSON
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.debug(f"MIR file does not exist: {file_path}")
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.debug(f"Successfully read temporal data from {file_path}")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        raise


def write_mir(file_path: str | Path, data: Dict[str, Any], merge: bool = True) -> None:
    """
    Write temporal data to a .MIR file. By default, merges with existing data.

    Process-safe: uses filelock for cross-process locking (works with ProcessPoolExecutor).

    Args:
        file_path: Path to the .MIR file
        data: Dictionary with temporal data to write
        merge: If True, merge with existing data (preserving existing keys)
               If False, completely replace file contents

    Raises:
        IOError: If file cannot be written
    """
    file_path = Path(file_path)

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Process-safe: acquire file lock for the entire read-modify-write cycle
    if FILELOCK_AVAILABLE:
        lock_path = _get_lock_path(file_path)
        lock = FileLock(lock_path, timeout=60)
    else:
        lock = _get_thread_lock(file_path)

    with lock:
        # If merging, read existing data first
        if merge and file_path.exists():
            try:
                existing_data = read_mir(file_path)
                # Merge: new data overwrites existing keys
                existing_data.update(data)
                data = existing_data
                logger.debug(f"Merged temporal data with existing {file_path}")
            except Exception as e:
                logger.warning(f"Could not read existing data from {file_path}, will overwrite: {e}")

        # Write atomically using a temporary file
        temp_path = file_path.with_suffix('.MIR.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_path.replace(file_path)
            logger.debug(f"Successfully wrote temporal data to {file_path}")

        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Error writing {file_path}: {e}")
            raise


def safe_update(file_path: str | Path, updates: Dict[str, Any], file_type: str = 'INFO') -> None:
    """
    Safely update specific keys in a .INFO or .MIR file without affecting other keys.

    This is a convenience function that ensures merge=True and provides clear logging.

    Args:
        file_path: Path to the file
        updates: Dictionary with keys to update
        file_type: Either 'INFO' or 'MIR' to determine which type of file

    Raises:
        ValueError: If file_type is not 'INFO' or 'MIR'
        IOError: If file cannot be written
    """
    if file_type == 'INFO':
        write_info(file_path, updates, merge=True)
    elif file_type == 'MIR':
        write_mir(file_path, updates, merge=True)
    else:
        raise ValueError(f"file_type must be 'INFO' or 'MIR', got: {file_type}")

    logger.info(f"Safely updated {len(updates)} keys in {file_path}")


def should_process(info_path: str | Path, output_keys: list, overwrite: bool = False) -> bool:
    """
    Universal check: Should we process this file?

    This function encapsulates the standard "check before processing" pattern
    that should be used consistently across all feature extraction operations.

    Returns True (should process) if ANY of:
    - overwrite is True
    - info_path doesn't exist
    - ANY of the output_keys are missing from the .INFO file

    Returns False (skip) if:
    - overwrite is False AND
    - info_path exists AND
    - ALL output_keys are present in the .INFO file

    Args:
        info_path: Path to the .INFO file to check
        output_keys: List of keys that the operation will produce
                     (e.g., ['lufs', 'lra'] for loudness analysis)
        overwrite: If True, always returns True (force processing)

    Returns:
        True if processing should occur, False if it should be skipped

    Example:
        >>> info_path = get_info_path(audio_file)
        >>> if should_process(info_path, ['spectral_flatness', 'spectral_flux'], overwrite=False):
        ...     results = analyze_spectral_features(audio_file)
        ...     safe_update(info_path, results)
        ... else:
        ...     logger.debug(f"Skipping {audio_file.name} - already processed")
    """
    if overwrite:
        return True

    info_path = Path(info_path)
    if not info_path.exists():
        return True

    try:
        existing = read_info(info_path)
        # Process if ANY output key is missing
        return any(key not in existing for key in output_keys)
    except Exception:
        # If we can't read the file, process it
        return True


def batch_write_info(writes: list, merge: bool = True) -> int:
    """
    Batch write multiple .INFO files efficiently.

    Optimized for HDD by grouping writes. Still uses atomic writes for safety.

    Args:
        writes: List of (file_path, data_dict) tuples
        merge: If True, merge with existing data. If False, overwrite.

    Returns:
        Number of files successfully written

    Example:
        >>> writes = [
        ...     (Path('/path/to/crop1.INFO'), {'bpm': 120, 'position': 0.0}),
        ...     (Path('/path/to/crop2.INFO'), {'bpm': 120, 'position': 0.5}),
        ... ]
        >>> batch_write_info(writes, merge=False)
        2
    """
    success_count = 0

    for file_path, data in writes:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if merge and file_path.exists():
                # Need to read-modify-write with locking
                safe_update(file_path, data)
            else:
                # New file or overwrite mode - use atomic write for safety
                # This avoids leaving partial files if interrupted
                temp_path = file_path.with_suffix('.INFO.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                temp_path.replace(file_path)  # Atomic rename

            success_count += 1

        except Exception as e:
            logger.warning(f"Failed to write {file_path}: {e}")
            # Clean up temp file if it exists
            temp_path = file_path.with_suffix('.INFO.tmp')
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    if success_count > 0:
        logger.debug(f"Batch wrote {success_count}/{len(writes)} .INFO files")

    return success_count


def should_process_file(file_path: str | Path, output_keys: list, overwrite: bool = False) -> bool:
    """
    Convenience wrapper: Check if an audio file needs processing.

    Automatically derives the .INFO path from the audio file path.

    Args:
        file_path: Path to audio file (full_mix.flac, crop.flac, etc.)
        output_keys: List of keys that the operation will produce
        overwrite: If True, always returns True (force processing)

    Returns:
        True if processing should occur, False if it should be skipped
    """
    info_path = get_info_path(file_path)
    return should_process(info_path, output_keys, overwrite)


def get_info_path(audio_file: str | Path) -> Path:
    """
    Get the corresponding .INFO file path for an audio file.

    Args:
        audio_file: Path to audio file (e.g., /path/to/song/full_mix.flac)

    Returns:
        Path to the .INFO file (e.g., /path/to/song/song.INFO)
    """
    audio_file = Path(audio_file)
    # Get the parent directory name (song folder)
    folder_name = audio_file.parent.name
    return audio_file.parent / f"{folder_name}.INFO"


def get_mir_path(audio_file: str | Path) -> Path:
    """
    Get the corresponding .MIR file path for an audio file.

    Args:
        audio_file: Path to audio file (e.g., /path/to/song/full_mix.flac)

    Returns:
        Path to the .MIR file (e.g., /path/to/song/song.MIR)
    """
    audio_file = Path(audio_file)
    # Get the parent directory name (song folder)
    folder_name = audio_file.parent.name
    return audio_file.parent / f"{folder_name}.MIR"


def get_crop_info_path(crop_file: str | Path) -> Path:
    """
    Get the corresponding .INFO file path for a crop file.

    Unlike full tracks (which use folder name), crops use the crop filename
    for the .INFO file name.

    Args:
        crop_file: Path to crop file (e.g., /path/to/TrackName/TrackName_0.flac)

    Returns:
        Path to the .INFO file (e.g., /path/to/TrackName/TrackName_0.INFO)
    """
    crop_file = Path(crop_file)
    return crop_file.with_suffix('.INFO')


# Example usage
if __name__ == "__main__":
    # Test basic functionality
    test_dir = Path("/tmp/mir_test")
    test_dir.mkdir(exist_ok=True)

    test_info = test_dir / "test.INFO"

    # Write some initial data
    initial_data = {
        "lufs": -14.5,
        "bpm": 128,
        "brightness": 75.3
    }
    write_info(test_info, initial_data, merge=False)
    print(f"Wrote initial data: {initial_data}")

    # Update with new data (merge)
    new_data = {
        "danceability": 0.85,
        "bpm": 130  # This will overwrite the old BPM
    }
    safe_update(test_info, new_data)

    # Read and display final result
    final_data = read_info(test_info)
    print(f"Final data: {final_data}")
    print(f"Expected: lufs=-14.5, bpm=130 (updated), brightness=75.3, danceability=0.85")

    # Clean up
    test_info.unlink()
    test_dir.rmdir()
