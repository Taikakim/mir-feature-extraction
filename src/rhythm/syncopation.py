"""
Syncopation Analysis for MIR Project

This module analyzes syncopation (off-beat emphasis) in audio by comparing
onset times with beat positions. Syncopation occurs when strong musical events
emphasize weak metric positions.

Dependencies:
- numpy
- src.rhythm.beat_grid (for loading beat grids)
- src.rhythm.onsets (for detecting onsets)
- src.core.file_utils
- src.core.common

Output:
- syncopation: Syncopation measure 0.0-1.0 (NumberConditioner)
- on_beat_ratio: Ratio of onsets on beats vs off-beats (NumberConditioner)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import get_stem_files
from core.common import clamp_feature_value
from core.json_handler import safe_update, get_info_path

logger = logging.getLogger(__name__)


def load_beat_grid(beat_grid_path: str | Path) -> np.ndarray:
    """
    Load beat grid from .BEATS_GRID file.

    Args:
        beat_grid_path: Path to .BEATS_GRID file

    Returns:
        Array of beat timestamps in seconds
    """
    beat_grid_path = Path(beat_grid_path)

    if not beat_grid_path.exists():
        raise FileNotFoundError(f"Beat grid not found: {beat_grid_path}")

    beat_times = np.loadtxt(beat_grid_path)

    if beat_times.ndim == 0:
        beat_times = np.array([beat_times])

    return beat_times


def load_onsets(onsets_path: str | Path) -> np.ndarray:
    """
    Load onset times from .ONSETS file.

    Args:
        onsets_path: Path to .ONSETS file

    Returns:
        Array of onset timestamps in seconds
    """
    onsets_path = Path(onsets_path)

    if not onsets_path.exists():
        raise FileNotFoundError(f"Onsets file not found: {onsets_path}")

    onset_times = np.loadtxt(onsets_path)

    if onset_times.ndim == 0:
        onset_times = np.array([onset_times])

    return onset_times


def calculate_beat_strength(onset_times: np.ndarray,
                            beat_times: np.ndarray,
                            tolerance: float = 0.07) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate which onsets are on-beat vs off-beat.

    Args:
        onset_times: Array of onset timestamps
        beat_times: Array of beat timestamps
        tolerance: Time tolerance in seconds for considering an onset "on-beat"

    Returns:
        Tuple of (on_beat_mask, off_beat_mask)
        on_beat_mask: Boolean array indicating which onsets are on-beat
        off_beat_mask: Boolean array indicating which onsets are off-beat
    """
    if len(onset_times) == 0 or len(beat_times) == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)

    # For each onset, find the closest beat
    on_beat_mask = np.zeros(len(onset_times), dtype=bool)

    for i, onset_time in enumerate(onset_times):
        # Find closest beat
        distances = np.abs(beat_times - onset_time)
        min_distance = np.min(distances)

        # If within tolerance, it's on-beat
        if min_distance <= tolerance:
            on_beat_mask[i] = True

    off_beat_mask = ~on_beat_mask

    return on_beat_mask, off_beat_mask


def calculate_syncopation(onset_times: np.ndarray,
                          beat_times: np.ndarray,
                          onset_strengths: Optional[np.ndarray] = None,
                          tolerance: float = 0.07) -> float:
    """
    Calculate syncopation measure.

    Syncopation is calculated as the ratio of off-beat onset energy to total onset energy.
    If onset strengths are not provided, all onsets are weighted equally.

    Args:
        onset_times: Array of onset timestamps
        beat_times: Array of beat timestamps
        onset_strengths: Optional array of onset strength values
        tolerance: Time tolerance for on-beat detection

    Returns:
        Syncopation measure (0.0 = all on-beat, 1.0 = all off-beat)
    """
    if len(onset_times) == 0:
        return 0.0

    # Get on/off beat masks
    on_beat_mask, off_beat_mask = calculate_beat_strength(
        onset_times, beat_times, tolerance
    )

    # If strengths not provided, use equal weights
    if onset_strengths is None:
        onset_strengths = np.ones(len(onset_times))

    # Calculate weighted syncopation
    on_beat_energy = np.sum(onset_strengths[on_beat_mask]) if np.any(on_beat_mask) else 0.0
    off_beat_energy = np.sum(onset_strengths[off_beat_mask]) if np.any(off_beat_mask) else 0.0
    total_energy = on_beat_energy + off_beat_energy

    if total_energy == 0:
        return 0.0

    syncopation = off_beat_energy / total_energy

    return float(syncopation)


def calculate_on_beat_ratio(onset_times: np.ndarray,
                            beat_times: np.ndarray,
                            tolerance: float = 0.07) -> float:
    """
    Calculate the ratio of on-beat onsets to total onsets.

    Args:
        onset_times: Array of onset timestamps
        beat_times: Array of beat timestamps
        tolerance: Time tolerance for on-beat detection

    Returns:
        Ratio of on-beat onsets (0.0 to 1.0)
    """
    if len(onset_times) == 0:
        return 0.0

    on_beat_mask, _ = calculate_beat_strength(onset_times, beat_times, tolerance)

    on_beat_count = np.sum(on_beat_mask)
    total_count = len(onset_times)

    return float(on_beat_count / total_count)


def analyze_syncopation(folder_path: str | Path,
                       tolerance: float = 0.07) -> Dict[str, float]:
    """
    Analyze syncopation for an organized music folder.

    Requires:
    - .BEATS_GRID file (from beat_grid.py)
    - .ONSETS file (from onsets.py)

    Args:
        folder_path: Path to organized folder
        tolerance: Time tolerance for on-beat detection (seconds)

    Returns:
        Dictionary with syncopation features:
        - syncopation: Syncopation measure (0.0-1.0)
        - on_beat_ratio: Ratio of on-beat onsets (0.0-1.0)
    """
    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    logger.info(f"Analyzing syncopation: {folder_path.name}")

    # Find required files
    folder_name = folder_path.name
    beat_grid_file = folder_path / f"{folder_name}.BEATS_GRID"
    onsets_file = folder_path / f"{folder_name}.ONSETS"

    if not beat_grid_file.exists():
        raise FileNotFoundError(
            f"Beat grid not found: {beat_grid_file}\n"
            "Run beat_grid.py first to generate beat grid."
        )

    if not onsets_file.exists():
        raise FileNotFoundError(
            f"Onsets file not found: {onsets_file}\n"
            "Run onsets.py first to generate onsets."
        )

    # Load beat grid and onsets
    beat_times = load_beat_grid(beat_grid_file)
    onset_times = load_onsets(onsets_file)

    logger.info(f"Loaded {len(beat_times)} beats and {len(onset_times)} onsets")

    # Calculate syncopation
    syncopation = calculate_syncopation(onset_times, beat_times, tolerance=tolerance)
    on_beat_ratio = calculate_on_beat_ratio(onset_times, beat_times, tolerance=tolerance)

    # Compile results
    results = {
        'syncopation': float(syncopation),
        'on_beat_ratio': float(on_beat_ratio)
    }

    # Clamp values to valid ranges
    for key, value in results.items():
        results[key] = clamp_feature_value(key, value)

    logger.info(f"Syncopation: {results['syncopation']:.3f}")
    logger.info(f"On-beat ratio: {results['on_beat_ratio']:.3f}")

    return results


def batch_analyze_syncopation(root_directory: str | Path,
                              overwrite: bool = False,
                              tolerance: float = 0.07) -> dict:
    """
    Batch analyze syncopation for all organized folders.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing syncopation data
        tolerance: Time tolerance for on-beat detection

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch syncopation analysis: {root_directory}")

    folders = find_organized_folders(root_directory)

    stats = {
        'total': len(folders),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }

    logger.info(f"Found {stats['total']} organized folders")

    for i, folder in enumerate(folders, 1):
        logger.info(f"Processing {i}/{stats['total']}: {folder.name}")

        # Find full_mix file
        stems = get_stem_files(folder, include_full_mix=True)
        if 'full_mix' not in stems:
            logger.warning(f"No full_mix found in {folder.name}")
            stats['failed'] += 1
            continue

        # Check if already processed
        info_path = get_info_path(stems['full_mix'])
        if info_path.exists() and not overwrite:
            try:
                import json
                with open(info_path, 'r') as f:
                    data = json.load(f)
                if 'syncopation' in data:
                    logger.info("Syncopation data already exists. Use --overwrite to regenerate.")
                    stats['skipped'] += 1
                    continue
            except Exception:
                pass

        try:
            results = analyze_syncopation(folder, tolerance=tolerance)

            # Save to .INFO file
            safe_update(info_path, results)

            stats['success'] += 1
            logger.info(f"Syncopation: {results['syncopation']:.3f}, "
                       f"On-beat ratio: {results['on_beat_ratio']:.3f}")

        except FileNotFoundError as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: Missing prerequisite - {str(e)}"
            stats['errors'].append(error_msg)
            logger.warning(error_msg)

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Syncopation Analysis Summary:")
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
        description="Analyze syncopation in audio files"
    )

    parser.add_argument(
        'path',
        type=str,
        help='Path to organized folder'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process all organized folders in directory tree'
    )

    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.07,
        help='Time tolerance for on-beat detection in seconds (default: 0.07)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing syncopation data'
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
            stats = batch_analyze_syncopation(
                path,
                overwrite=args.overwrite,
                tolerance=args.tolerance
            )

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} folders failed to process")
                sys.exit(1)

        else:
            # Single folder
            if not path.is_dir():
                logger.error(f"Path must be a directory: {path}")
                sys.exit(1)

            results = analyze_syncopation(path, tolerance=args.tolerance)

            # Save to .INFO
            info_path = get_info_path(path)
            safe_update(info_path, results)

            print(f"\nSyncopation Analysis Results:")
            print(f"  Syncopation:    {results['syncopation']:.3f}")
            print(f"  On-beat ratio:  {results['on_beat_ratio']:.3f}")

        logger.info("Syncopation analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
