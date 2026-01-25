"""
JSON Handler for MIR Project

This module provides safe JSON read/write operations for .INFO and .MIR files.
It ensures that existing data is never accidentally destroyed when adding new features.

Thread Safety:
- Uses per-file threading.Lock() to prevent race conditions when multiple threads
  write to the same file simultaneously
- Safe for use with ThreadPoolExecutor and concurrent processing
- Uses atomic writes (temp file + rename) for crash safety

Dependencies:
- None (standard library only)

Functions:
- read_info(file_path): Read .INFO file, return dict or empty dict if not exists
- write_info(file_path, data, merge=True): Write data to .INFO file with optional merge
- read_mir(file_path): Read .MIR file (temporal data)
- write_mir(file_path, data, merge=True): Write data to .MIR file with optional merge
- safe_update(file_path, updates): Safely update specific keys without destroying others
"""

import json
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe lock registry: one lock per file path
# Uses defaultdict so accessing a new path automatically creates a lock
_file_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
_registry_lock = threading.Lock()  # Protects the _file_locks dict itself


def _get_file_lock(file_path: Path) -> threading.Lock:
    """
    Get or create a thread lock for a specific file path.

    Thread-safe: uses a registry lock to protect the lock dictionary.
    """
    key = str(file_path.resolve())
    with _registry_lock:
        if key not in _file_locks:
            _file_locks[key] = threading.Lock()
        return _file_locks[key]


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

    Thread-safe: uses per-file locking to prevent race conditions.

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

    # Thread-safe: acquire per-file lock for the entire read-modify-write cycle
    file_lock = _get_file_lock(file_path)
    with file_lock:
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
            logger.info(f"Successfully wrote {len(data)} keys to {file_path}")

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

    Thread-safe: uses per-file locking to prevent race conditions.

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

    # Thread-safe: acquire per-file lock for the entire read-modify-write cycle
    file_lock = _get_file_lock(file_path)
    with file_lock:
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
            logger.info(f"Successfully wrote temporal data to {file_path}")

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
