"""
Beat Grid Creation for MIR Project

This module creates beat grids using beat tracking algorithms.
The beat grid serves as the foundation for BPM detection, syncopation analysis,
and MIDI quantization.

Dependencies:
- madmom (primary)
- librosa (fallback)
- soundfile
- src.core.file_utils
- src.core.common

Output:
- filename.BEATS_GRID: Text file with beat timestamps (one per line)
- filename.DOWNBEATS: Text file with downbeat timestamps (one per line)
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import get_stem_files
from core.common import BEATS_GRID_EXT

logger = logging.getLogger(__name__)

# Try to import madmom (preferred for electronic music)
try:
    import madmom
    MADMOM_AVAILABLE = True
    logger.debug("Madmom available")
except ImportError:
    MADMOM_AVAILABLE = False
    logger.warning("Madmom not available, will use librosa as fallback")

# Import librosa as fallback
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.error("Neither madmom nor librosa available!")


def detect_beats_madmom(audio_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect beats using Madmom's beat tracking algorithms.

    Args:
        audio_path: Path to audio file

    Returns:
        Tuple of (beat_times, downbeat_times) in seconds
        downbeat_times may be empty if downbeat detection is not available

    Raises:
        ImportError: If madmom is not installed
        Exception: If beat detection fails
    """
    if not MADMOM_AVAILABLE:
        raise ImportError("Madmom is not installed")

    audio_path = str(audio_path)

    logger.info(f"Detecting beats with Madmom: {Path(audio_path).name}")

    try:
        # Use DBNBeatTracker which is good for general music
        # For electronic music, RNNBeatProcessor + DBNBeatTrackingProcessor works well
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(audio_path)
        beat_times = proc(act)

        logger.info(f"Detected {len(beat_times)} beats")

        # Try to detect downbeats (bars)
        downbeat_times = np.array([])
        try:
            dbn_proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
                beats_per_bar=[3, 4],  # Common time signatures
                fps=100
            )
            down_act = madmom.features.downbeats.RNNDownBeatProcessor()(audio_path)
            downbeats = dbn_proc(down_act)

            # downbeats is Nx2 array: [time, beat_number]
            # downbeat_number = 1 indicates downbeat
            if len(downbeats) > 0:
                downbeat_times = downbeats[downbeats[:, 1] == 1, 0]
                logger.info(f"Detected {len(downbeat_times)} downbeats")

        except Exception as e:
            logger.debug(f"Downbeat detection failed: {e}")

        return beat_times, downbeat_times

    except Exception as e:
        logger.error(f"Madmom beat detection failed: {e}")
        raise


def detect_beats_librosa(audio_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect beats using Librosa (fallback method).

    Args:
        audio_path: Path to audio file

    Returns:
        Tuple of (beat_times, downbeat_times) in seconds
        downbeat_times will be empty (librosa doesn't detect downbeats)

    Raises:
        ImportError: If librosa is not installed
        Exception: If beat detection fails
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("Librosa is not installed")

    logger.info(f"Detecting beats with Librosa: {Path(audio_path).name}")

    try:
        # Load audio
        y, sr = librosa.load(str(audio_path))

        # Detect beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # Convert frames to time
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Extract tempo value (can be array or scalar)
        tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0]) if len(tempo) > 0 else 0.0

        logger.info(f"Detected {len(beat_times)} beats (tempo: {tempo_val:.1f} BPM)")

        # Librosa doesn't provide downbeat detection
        downbeat_times = np.array([])

        return beat_times, downbeat_times

    except Exception as e:
        logger.error(f"Librosa beat detection failed: {e}")
        raise


def detect_beats(audio_path: str | Path,
                  method: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect beats using the best available method.

    Args:
        audio_path: Path to audio file
        method: 'auto', 'madmom', or 'librosa'

    Returns:
        Tuple of (beat_times, downbeat_times) in seconds

    Raises:
        ValueError: If specified method is not available
        Exception: If beat detection fails
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if method == 'auto':
        # Prefer madmom for electronic music
        if MADMOM_AVAILABLE:
            method = 'madmom'
        elif LIBROSA_AVAILABLE:
            method = 'librosa'
        else:
            raise ImportError("No beat tracking library available")

    if method == 'madmom':
        return detect_beats_madmom(audio_path)
    elif method == 'librosa':
        return detect_beats_librosa(audio_path)
    else:
        raise ValueError(f"Unknown method: {method}")


def save_beat_grid(beat_times: np.ndarray,
                   output_path: str | Path) -> None:
    """
    Save beat grid to a text file.

    Args:
        beat_times: Array of beat timestamps in seconds
        output_path: Path to output .BEATS_GRID file

    Raises:
        IOError: If file cannot be written
    """
    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Write one timestamp per line
        with open(output_path, 'w') as f:
            for beat_time in beat_times:
                f.write(f"{beat_time:.6f}\n")

        logger.info(f"Saved {len(beat_times)} beats to {output_path.name}")

    except Exception as e:
        logger.error(f"Error saving beat grid: {e}")
        raise


def load_beat_grid(grid_path: str | Path) -> np.ndarray:
    """
    Load beat grid from a text file.

    Args:
        grid_path: Path to .BEATS_GRID file

    Returns:
        Array of beat timestamps in seconds

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file contains invalid data
    """
    grid_path = Path(grid_path)

    if not grid_path.exists():
        raise FileNotFoundError(f"Beat grid file not found: {grid_path}")

    try:
        beat_times = np.loadtxt(grid_path)

        # Ensure it's 1D array
        if beat_times.ndim == 0:
            beat_times = np.array([beat_times])

        logger.debug(f"Loaded {len(beat_times)} beats from {grid_path.name}")
        return beat_times

    except Exception as e:
        logger.error(f"Error loading beat grid: {e}")
        raise


def create_beat_grid(audio_path: str | Path,
                     save_grid: bool = True,
                     method: str = 'auto') -> Tuple[np.ndarray, Optional[Path]]:
    """
    Create beat grid for an audio file.

    Args:
        audio_path: Path to audio file
        save_grid: Whether to save the grid to a file
        method: Beat detection method ('auto', 'madmom', 'librosa')

    Returns:
        Tuple of (beat_times, grid_file_path)
        grid_file_path is None if save_grid=False

    Raises:
        FileNotFoundError: If audio file doesn't exist
        Exception: If beat detection fails
    """
    audio_path = Path(audio_path)

    # Detect beats
    beat_times, downbeat_times = detect_beats(audio_path, method=method)

    grid_file_path = None

    if save_grid:
        # Determine output path
        # For organized structure: /path/to/song/full_mix.flac -> /path/to/song/song.BEATS_GRID
        folder_name = audio_path.parent.name
        grid_file_path = audio_path.parent / f"{folder_name}{BEATS_GRID_EXT}"

        # Save the grid
        save_beat_grid(beat_times, grid_file_path)
        
        # Save downbeats if available
        if len(downbeat_times) > 0:
            downbeat_path = audio_path.parent / f"{folder_name}.DOWNBEATS"
            try:
                np.savetxt(downbeat_path, downbeat_times, fmt='%.6f')
                logger.info(f"Saved {len(downbeat_times)} downbeats to {downbeat_path.name}")
            except Exception as e:
                logger.error(f"Failed to save downbeats: {e}")

    return beat_times, grid_file_path


def _process_folder_beats(args: Tuple) -> Tuple[str, str, int, str]:
    """
    Worker function for parallel beat grid creation.

    Args:
        args: Tuple of (folder_path, method, overwrite)

    Returns:
        Tuple of (folder_name, status, num_beats, error_msg)
        status: 'success', 'skipped', 'failed'
    """
    folder_path, method, overwrite = args
    folder = Path(folder_path)

    try:
        # Check if grid already exists
        grid_file = folder / f"{folder.name}{BEATS_GRID_EXT}"

        if grid_file.exists() and not overwrite:
            return (folder.name, 'skipped', 0, '')

        # Find full_mix file
        stems = get_stem_files(folder, include_full_mix=True)
        if 'full_mix' not in stems:
            return (folder.name, 'failed', 0, 'No full_mix found')

        beat_times, grid_path = create_beat_grid(
            stems['full_mix'],
            save_grid=True,
            method=method
        )

        return (folder.name, 'success', len(beat_times), '')

    except Exception as e:
        return (folder.name, 'failed', 0, str(e))


def batch_create_beat_grids(root_directory: str | Path,
                             method: str = 'auto',
                             overwrite: bool = False,
                             workers: int = 4) -> dict:
    """
    Batch create beat grids for all organized folders.

    Args:
        root_directory: Root directory to search
        method: Beat detection method
        overwrite: Whether to overwrite existing grids
        workers: Number of parallel workers (default: 4)

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch beat grid creation: {root_directory}")

    # Find all organized folders
    folders = find_organized_folders(root_directory)

    stats = {
        'total': len(folders),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }

    logger.info(f"Found {stats['total']} organized folders")

    if workers <= 1:
        # Sequential processing
        for i, folder in enumerate(folders, 1):
            logger.info(f"Processing {i}/{stats['total']}: {folder.name}")

            # Check if grid already exists
            grid_file = folder / f"{folder.name}{BEATS_GRID_EXT}"

            if grid_file.exists() and not overwrite:
                logger.info("Beat grid already exists. Use --overwrite to regenerate.")
                stats['skipped'] += 1
                continue

            # Find full_mix file
            stems = get_stem_files(folder, include_full_mix=True)
            if 'full_mix' not in stems:
                logger.warning(f"No full_mix found in {folder.name}")
                stats['failed'] += 1
                continue

            try:
                beat_times, grid_path = create_beat_grid(
                    stems['full_mix'],
                    save_grid=True,
                    method=method
                )
                stats['success'] += 1
                logger.info(f"Created beat grid with {len(beat_times)} beats")

            except Exception as e:
                stats['failed'] += 1
                error_msg = f"{folder.name}: {str(e)}"
                stats['errors'].append(error_msg)
                logger.error(f"Failed to process {folder.name}: {e}")
    else:
        # Parallel processing
        logger.info(f"Using {workers} parallel workers")

        # Prepare arguments for workers
        work_args = [(str(folder), method, overwrite) for folder in folders]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_folder_beats, args): args[0]
                      for args in work_args}

            completed = 0
            for future in as_completed(futures):
                completed += 1
                folder_name, status, num_beats, error_msg = future.result()

                if status == 'success':
                    stats['success'] += 1
                    logger.info(f"[{completed}/{stats['total']}] {folder_name}: {num_beats} beats")
                elif status == 'skipped':
                    stats['skipped'] += 1
                    logger.debug(f"[{completed}/{stats['total']}] {folder_name}: skipped (exists)")
                else:
                    stats['failed'] += 1
                    stats['errors'].append(f"{folder_name}: {error_msg}")
                    logger.error(f"[{completed}/{stats['total']}] {folder_name}: {error_msg}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Beat Grid Creation Summary:")
    logger.info(f"  Total folders:  {stats['total']}")
    logger.info(f"  Successful:     {stats['success']}")
    logger.info(f"  Skipped:        {stats['skipped']}")
    logger.info(f"  Failed:         {stats['failed']}")
    logger.info("=" * 60)

    return stats


# Command-line interface
if __name__ == "__main__":
    import argparse
    from core.common import setup_logging

    parser = argparse.ArgumentParser(
        description="Create beat grids for audio files"
    )

    parser.add_argument(
        'path',
        type=str,
        help='Path to audio file or organized folder'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process all organized folders in directory tree'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='auto',
        choices=['auto', 'madmom', 'librosa'],
        help='Beat detection method (default: auto)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing beat grids'
    )

    parser.add_argument(
        '--workers', '-j',
        type=int,
        default=4,
        help='Number of parallel workers for batch processing (default: 4)'
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
            stats = batch_create_beat_grids(
                path,
                method=args.method,
                overwrite=args.overwrite,
                workers=args.workers
            )

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} folders failed to process")
                sys.exit(1)

        elif path.is_dir():
            # Single folder - find full_mix
            stems = get_stem_files(path, include_full_mix=True)
            if 'full_mix' not in stems:
                logger.error(f"No full_mix file found in {path}")
                sys.exit(1)

            beat_times, grid_path = create_beat_grid(
                stems['full_mix'],
                save_grid=True,
                method=args.method
            )

            print(f"\nBeat Grid Created:")
            print(f"  Beats detected: {len(beat_times)}")
            print(f"  Grid file: {grid_path}")
            print(f"  First few beats: {beat_times[:5]}")

        else:
            # Single file
            beat_times, grid_path = create_beat_grid(
                path,
                save_grid=True,
                method=args.method
            )

            print(f"\nBeat Grid Created:")
            print(f"  Beats detected: {len(beat_times)}")
            if grid_path:
                print(f"  Grid file: {grid_path}")

        logger.info("Beat grid creation completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
