"""
File Utilities for MIR Project

This module provides file and path handling utilities for the MIR pipeline.
Handles audio file discovery, path conversions, and file organization queries.

Dependencies:
- None (standard library only)

Functions:
- find_audio_files(directory, extensions): Recursively find audio files
- get_audio_folder_structure(audio_file): Get expected organized folder structure
- is_organized(audio_file): Check if file is in organized structure
- get_stem_files(audio_folder): Get all stem files for a song
- get_grid_files(audio_folder): Get beat/onset grid files
- calculate_position_in_file(clip_path): Calculate relative position in original file
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported audio file extensions
AUDIO_EXTENSIONS = {'.flac', '.wav', '.mp3', '.ogg', '.m4a', '.aiff', '.aif'}

# Standard stem names from Demucs
DEMUCS_STEMS = ['drums', 'bass', 'other', 'vocals']

# DrumSep outputs (these will vary based on the model)
DRUMSEP_STEMS = ['kick', 'snare', 'cymbals', 'toms', 'percussion']


def find_audio_files(directory: str | Path,
                     extensions: Optional[set] = None,
                     recursive: bool = True) -> List[Path]:
    """
    Find all audio files in a directory.

    Args:
        directory: Root directory to search
        extensions: Set of file extensions to search for (default: AUDIO_EXTENSIONS)
        recursive: Whether to search recursively (default: True)

    Returns:
        List of Path objects for found audio files
    """
    directory = Path(directory)
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return []

    if extensions is None:
        extensions = AUDIO_EXTENSIONS

    # Normalize extensions (ensure they start with .)
    extensions = {ext if ext.startswith('.') else f'.{ext}' for ext in extensions}

    audio_files = []
    pattern = '**/*' if recursive else '*'

    for ext in extensions:
        audio_files.extend(directory.glob(f'{pattern}{ext}'))

    logger.info(f"Found {len(audio_files)} audio files in {directory}")
    return sorted(audio_files)


def get_audio_folder_structure(audio_file: str | Path) -> Dict[str, Path]:
    """
    Get the expected organized folder structure for an audio file.

    According to the MIR plan, files should be organized as:
    /path/to/original/filename.flac -> /path/to/original/filename/full_mix.flac

    Args:
        audio_file: Path to the audio file

    Returns:
        Dictionary with:
            - 'original': Original file path
            - 'folder': Expected organized folder path
            - 'full_mix': Expected full_mix.flac path
            - 'folder_name': Folder name
    """
    audio_file = Path(audio_file)

    # Get the filename without extension for folder name
    folder_name = audio_file.stem

    # Expected organized structure
    folder_path = audio_file.parent / folder_name
    full_mix_path = folder_path / f"full_mix{audio_file.suffix}"

    return {
        'original': audio_file,
        'folder': folder_path,
        'full_mix': full_mix_path,
        'folder_name': folder_name
    }


def is_organized(audio_file: str | Path) -> bool:
    """
    Check if an audio file is already in the organized structure.

    Organized means: /path/to/folder/full_mix.ext

    Args:
        audio_file: Path to check

    Returns:
        True if file is in organized structure, False otherwise
    """
    audio_file = Path(audio_file)

    # Check if the file is named full_mix.*
    if audio_file.stem != 'full_mix':
        return False

    # Check if it's in a dedicated folder (not at root level)
    if audio_file.parent == audio_file.parent.parent:
        return False

    return True


def get_stem_files(audio_folder: str | Path,
                   include_full_mix: bool = False) -> Dict[str, Path]:
    """
    Get all stem files for a song folder.

    Args:
        audio_folder: Path to the organized audio folder
        include_full_mix: Whether to include full_mix in the results

    Returns:
        Dictionary mapping stem names to file paths
        Example: {'drums': Path(...), 'bass': Path(...), ...}
    """
    audio_folder = Path(audio_folder)

    if not audio_folder.exists():
        logger.warning(f"Audio folder does not exist: {audio_folder}")
        return {}

    stems = {}

    # Find full_mix first (to get the extension)
    full_mix = None
    for ext in AUDIO_EXTENSIONS:
        potential_path = audio_folder / f"full_mix{ext}"
        if potential_path.exists():
            full_mix = potential_path
            if include_full_mix:
                stems['full_mix'] = full_mix
            break

    if full_mix is None:
        logger.warning(f"No full_mix file found in {audio_folder}")
        return stems

    # Find Demucs stems (check all audio extensions)
    for stem_name in DEMUCS_STEMS:
        for ext in AUDIO_EXTENSIONS:
            stem_path = audio_folder / f"{stem_name}{ext}"
            if stem_path.exists():
                stems[stem_name] = stem_path
                break

    # Find DrumSep stems (check all audio extensions)
    for stem_name in DRUMSEP_STEMS:
        for ext in AUDIO_EXTENSIONS:
            stem_path = audio_folder / f"{stem_name}{ext}"
            if stem_path.exists():
                stems[stem_name] = stem_path
                break

    logger.debug(f"Found {len(stems)} stem files in {audio_folder}")
    return stems


def get_grid_files(audio_folder: str | Path) -> Dict[str, Path]:
    """
    Get beat and onset grid files for a song folder.

    Args:
        audio_folder: Path to the organized audio folder

    Returns:
        Dictionary with 'beats' and 'onsets' grid file paths (if they exist)
    """
    audio_folder = Path(audio_folder)
    folder_name = audio_folder.name

    grids = {}

    # Beat grid file
    beat_grid = audio_folder / f"{folder_name}.BEATS_GRID"
    if beat_grid.exists():
        grids['beats'] = beat_grid

    # Onset grid file
    onset_grid = audio_folder / f"{folder_name}.ONSETS_GRID"
    if onset_grid.exists():
        grids['onsets'] = onset_grid

    return grids


def get_chroma_file(audio_folder: str | Path) -> Optional[Path]:
    """
    Get the chroma data file for a song folder.

    Args:
        audio_folder: Path to the organized audio folder

    Returns:
        Path to chroma file if it exists, None otherwise
    """
    audio_folder = Path(audio_folder)
    folder_name = audio_folder.name
    chroma_file = audio_folder / f"{folder_name}.CHROMA"

    return chroma_file if chroma_file.exists() else None


def calculate_position_in_file(clip_path: str | Path,
                                 full_file_duration: float,
                                 clip_start_time: float) -> float:
    """
    Calculate the relative position of a clip in the original file.

    This is used for the position_in_file conditioning feature.

    Args:
        clip_path: Path to the clip file
        full_file_duration: Duration of the full original file in seconds
        clip_start_time: Start time of the clip in the original file (seconds)

    Returns:
        Position value between 0.0 (start) and 1.0 (end)
    """
    if full_file_duration <= 0:
        logger.warning("Invalid full_file_duration, returning 0.0")
        return 0.0

    position = clip_start_time / full_file_duration

    # Clamp to [0, 1] range
    position = max(0.0, min(1.0, position))

    return position


def parse_section_number(filename: str) -> Optional[int]:
    """
    Parse the section number from a cropped filename.

    According to the plan, cropped files have format: filename_section_N.ext

    Args:
        filename: The filename to parse

    Returns:
        Section number if found, None otherwise
    """
    # Look for pattern: section_N where N is a number
    parts = filename.split('_section_')
    if len(parts) == 2:
        # Get the number part (before the extension)
        num_part = parts[1].split('.')[0]
        try:
            return int(num_part)
        except ValueError:
            pass

    return None


def get_crops_folder(audio_folder: str | Path) -> Path:
    """
    Get the crops subfolder for a song folder.

    Args:
        audio_folder: Path to the organized audio folder

    Returns:
        Path to the crops subfolder
    """
    audio_folder = Path(audio_folder)
    return audio_folder / "crops"


# Directories to skip when searching for organized folders
SKIP_DIRECTORIES = {
    '.venv', 'venv', '.env', 'env',
    'node_modules', '__pycache__', '.git',
    '.tox', '.pytest_cache', '.mypy_cache',
    'site-packages', 'dist-packages',
    '.cache', 'build', 'dist', 'eggs',
    'repos', 'crops', '_crops', 'test data',
    'bs roformer output', 'bs roformer optimized output',
}

# Pattern to detect crop folders (end with _0, _1, _2, etc.)
import re
CROP_FOLDER_PATTERN = re.compile(r'_\d+$')


def find_organized_folders(root_directory: str | Path) -> List[Path]:
    """
    Find all organized audio folders in a directory tree.

    An organized folder contains a full_mix.* file.
    Skips common non-audio directories (.venv, node_modules, etc.)

    Args:
        root_directory: Root directory to search

    Returns:
        List of Path objects for organized folders
    """
    root_directory = Path(root_directory)
    organized_folders = []

    def should_skip(path: Path) -> bool:
        """Check if any parent directory should be skipped or if folder is a crop."""
        for part in path.parts:
            if part in SKIP_DIRECTORIES:
                return True
        # Skip crop folders (end with _0, _1, etc.)
        if CROP_FOLDER_PATTERN.search(path.name):
            return True
        return False

    # Search for all folders containing full_mix files
    for ext in AUDIO_EXTENSIONS:
        full_mix_files = root_directory.glob(f'**/full_mix{ext}')
        for full_mix_file in full_mix_files:
            folder = full_mix_file.parent
            if folder not in organized_folders and not should_skip(folder):
                organized_folders.append(folder)

    logger.info(f"Found {len(organized_folders)} organized folders in {root_directory}")
    return sorted(organized_folders)


def get_midi_files(audio_folder: str | Path,
                   stem: Optional[str] = None) -> List[Path]:
    """
    Get MIDI files in an audio folder, optionally filtered by stem.

    Args:
        audio_folder: Path to the organized audio folder
        stem: Optional stem name to filter (e.g., 'drums', 'bass')

    Returns:
        List of MIDI file paths
    """
    audio_folder = Path(audio_folder)
    midi_files = list(audio_folder.glob('*.mid')) + list(audio_folder.glob('*.midi'))

    if stem:
        # Filter to only include MIDI files with the stem name
        midi_files = [f for f in midi_files if stem in f.stem.lower()]

    return sorted(midi_files)


# ============================================================================
# Crop File Utilities
# ============================================================================

def parse_crop_number(filename: str) -> Optional[int]:
    """
    Parse the crop number from a crop filename.

    Crop files follow pattern: TrackName_N.ext where N is a number.

    Args:
        filename: The filename to parse (e.g., "TrackName_0.flac")

    Returns:
        Crop number if found, None otherwise

    Examples:
        "TrackName_0.flac" -> 0
        "My Song_15.flac" -> 15
        "full_mix.flac" -> None
    """
    stem = Path(filename).stem
    parts = stem.rsplit('_', 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


def find_crop_files(directory: str | Path,
                    extensions: Optional[set] = None) -> List[Path]:
    """
    Find all crop audio files in a directory.

    A crop file matches pattern: *_N.{ext} where N is a number.

    Args:
        directory: Directory to search
        extensions: Set of file extensions (default: AUDIO_EXTENSIONS)

    Returns:
        List of crop audio file paths, sorted by name
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    if extensions is None:
        extensions = AUDIO_EXTENSIONS

    crop_files = []

    for ext in extensions:
        for audio_file in directory.glob(f'*{ext}'):
            # Check if it's a crop file (ends with _N)
            if parse_crop_number(audio_file.name) is not None:
                # Exclude stem files (TrackName_0_drums.flac)
                stem = audio_file.stem
                parts = stem.rsplit('_', 1)
                if len(parts) == 2:
                    # Check if second-to-last part is a number (crop file)
                    # or a stem name (stem file)
                    if parts[1] not in DEMUCS_STEMS:
                        crop_files.append(audio_file)

    logger.debug(f"Found {len(crop_files)} crop files in {directory}")
    return sorted(crop_files)


def get_crop_stem_files(crop_path: str | Path) -> Dict[str, Path]:
    """
    Get stem files for a crop.

    For crop TrackName_0.flac, looks for:
        TrackName_0_drums.flac
        TrackName_0_bass.flac
        TrackName_0_other.flac
        TrackName_0_vocals.flac

    Args:
        crop_path: Path to the crop audio file

    Returns:
        Dict mapping stem name to path (includes 'source' for the crop itself)
    """
    crop_path = Path(crop_path)
    if not crop_path.exists():
        logger.warning(f"Crop file does not exist: {crop_path}")
        return {}

    stems = {'source': crop_path}
    crop_stem = crop_path.stem  # e.g., "TrackName_0"
    crop_dir = crop_path.parent

    # Find stems with prefixed naming
    for stem_name in DEMUCS_STEMS:
        for ext in AUDIO_EXTENSIONS:
            stem_path = crop_dir / f"{crop_stem}_{stem_name}{ext}"
            if stem_path.exists():
                stems[stem_name] = stem_path
                break

    logger.debug(f"Found {len(stems) - 1} stems for crop {crop_path.name}")
    return stems


def find_crop_folders(root_directory: str | Path) -> List[Path]:
    """
    Find all folders that contain crop files.

    A crop folder contains files matching *_N.{ext} pattern.

    Args:
        root_directory: Root directory to search

    Returns:
        List of folder paths containing crops
    """
    root_directory = Path(root_directory)
    crop_folders = set()

    for ext in AUDIO_EXTENSIONS:
        # Find all audio files and check if they're crops
        for audio_file in root_directory.glob(f'**/*{ext}'):
            if parse_crop_number(audio_file.name) is not None:
                # Exclude stem files
                stem = audio_file.stem
                parts = stem.rsplit('_', 1)
                if len(parts) == 2 and parts[1] not in DEMUCS_STEMS:
                    crop_folders.add(audio_file.parent)

    logger.info(f"Found {len(crop_folders)} crop folders in {root_directory}")
    return sorted(crop_folders)


def is_crop_file(audio_file: str | Path) -> bool:
    """
    Check if an audio file is a crop file.

    Args:
        audio_file: Path to check

    Returns:
        True if file is a crop (matches *_N.ext pattern and not a stem)
    """
    audio_file = Path(audio_file)
    stem = audio_file.stem

    # Check for _N pattern
    if parse_crop_number(audio_file.name) is None:
        return False

    # Exclude stem files
    parts = stem.rsplit('_', 1)
    if len(parts) == 2 and parts[1] in DEMUCS_STEMS:
        return False

    return True


# Example usage and testing
if __name__ == "__main__":
    # Test path operations
    test_file = Path("/path/to/music/awesome_song.flac")

    structure = get_audio_folder_structure(test_file)
    print("Expected structure:")
    for key, value in structure.items():
        print(f"  {key}: {value}")

    # Test position calculation
    position = calculate_position_in_file(
        "song_section_2.flac",
        full_file_duration=240.0,  # 4 minutes
        clip_start_time=120.0      # Start at 2 minutes
    )
    print(f"\nPosition in file: {position} (expected: 0.5)")

    # Test section parsing
    section = parse_section_number("awesome_song_section_3.flac")
    print(f"Section number: {section} (expected: 3)")
