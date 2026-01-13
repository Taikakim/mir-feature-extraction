"""
BPM Detection and Validation for MIR Project

This module calculates BPM from beat grids and determines whether a valid
BPM is present (for rhythmic vs non-rhythmic content).

Dependencies:
- numpy
- src.rhythm.beat_grid
- src.core.json_handler
- src.core.file_utils
- src.core.common

Features extracted:
- {bpm}: Detected BPM value (or default if undefined)
- {bpm_is_defined}: 1 if valid BPM detected, 0 for non-rhythmic content
- {beat_count}: Total number of beats detected
- {beat_regularity}: Consistency measure (std dev of beat intervals)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rhythm.beat_grid import load_beat_grid, create_beat_grid
from core.json_handler import safe_update, get_info_path
from core.file_utils import get_stem_files
from core.common import BEAT_TRACKING_CONFIG, clamp_feature_value

logger = logging.getLogger(__name__)


def calculate_bpm_from_beats(beat_times: np.ndarray,
                              audio_duration: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
    """
    Calculate BPM from beat timestamps.

    Args:
        beat_times: Array of beat timestamps in seconds
        audio_duration: Optional total duration for better estimation

    Returns:
        Tuple of (bpm, stats_dict) where stats_dict contains:
            - 'beat_count': Number of beats
            - 'mean_interval': Mean beat interval in seconds
            - 'std_interval': Std dev of beat intervals
            - 'regularity': Regularity score (for validation)

    Raises:
        ValueError: If insufficient beats for BPM calculation
    """
    if len(beat_times) < 2:
        raise ValueError("Need at least 2 beats to calculate BPM")

    # Calculate inter-beat intervals (IBIs)
    intervals = np.diff(beat_times)

    if len(intervals) == 0:
        raise ValueError("No intervals available")

    # Calculate statistics
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)

    # Calculate BPM from mean interval
    # BPM = 60 / interval_in_seconds
    bpm = 60.0 / mean_interval if mean_interval > 0 else 0.0

    # Regularity score (for validation)
    # Lower std = more regular
    regularity = std_interval

    stats = {
        'beat_count': len(beat_times),
        'mean_interval': float(mean_interval),
        'std_interval': float(std_interval),
        'regularity': float(regularity)
    }

    return float(bpm), stats


def validate_bpm(beat_count: int,
                 regularity: float,
                 bpm: float) -> bool:
    """
    Determine if detected BPM is valid (rhythmic content).

    Uses thresholds from BEAT_TRACKING_CONFIG:
    - beat_count_threshold: Minimum number of beats
    - regularity_threshold: Maximum std dev of beat intervals

    Args:
        beat_count: Number of beats detected
        regularity: Regularity score (std dev of intervals)
        bpm: Detected BPM value

    Returns:
        True if BPM is valid, False for non-rhythmic content
    """
    beat_count_threshold = BEAT_TRACKING_CONFIG['beat_count_threshold']
    regularity_threshold = BEAT_TRACKING_CONFIG['regularity_threshold']

    # Check if enough beats
    if beat_count < beat_count_threshold:
        logger.info(f"Insufficient beats: {beat_count} < {beat_count_threshold}")
        return False

    # Check if beats are regular enough
    if regularity > regularity_threshold:
        logger.info(f"Beats too irregular: {regularity:.3f} > {regularity_threshold}")
        return False

    # Check if BPM is in reasonable range
    if bpm < 40 or bpm > 300:
        logger.info(f"BPM out of range: {bpm:.1f}")
        return False

    logger.info(f"Valid BPM detected: {bpm:.1f} (beats: {beat_count}, regularity: {regularity:.3f})")
    return True


def analyze_bpm(beat_times: np.ndarray,
                audio_duration: Optional[float] = None) -> Dict[str, float]:
    """
    Analyze BPM and create conditioning features.

    Args:
        beat_times: Array of beat timestamps in seconds
        audio_duration: Optional total audio duration

    Returns:
        Dictionary with BPM features:
            - 'bpm': BPM value (or default if undefined)
            - 'bpm_is_defined': 1 or 0
            - 'beat_count': Number of beats
            - 'beat_regularity': Regularity measure

    Raises:
        ValueError: If beat_times is empty or invalid
    """
    if len(beat_times) == 0:
        logger.warning("No beats detected")
        return {
            'bpm': BEAT_TRACKING_CONFIG['default_bpm'],
            'bpm_is_defined': 0,
            'beat_count': 0,
            'beat_regularity': 1.0  # Maximum irregularity
        }

    if len(beat_times) == 1:
        logger.warning("Only one beat detected")
        return {
            'bpm': BEAT_TRACKING_CONFIG['default_bpm'],
            'bpm_is_defined': 0,
            'beat_count': 1,
            'beat_regularity': 1.0
        }

    # Calculate BPM
    try:
        bpm, stats = calculate_bpm_from_beats(beat_times, audio_duration)
    except Exception as e:
        logger.error(f"Error calculating BPM: {e}")
        return {
            'bpm': BEAT_TRACKING_CONFIG['default_bpm'],
            'bpm_is_defined': 0,
            'beat_count': len(beat_times),
            'beat_regularity': 1.0
        }

    # Validate BPM
    is_valid = validate_bpm(stats['beat_count'], stats['regularity'], bpm)

    # Prepare results
    results = {
        'bpm': bpm if is_valid else BEAT_TRACKING_CONFIG['default_bpm'],
        'bpm_is_defined': 1 if is_valid else 0,
        'beat_count': stats['beat_count'],
        'beat_regularity': stats['regularity']
    }

    # Clamp values to valid ranges
    results['bpm'] = clamp_feature_value('bpm', results['bpm'])
    results['beat_regularity'] = clamp_feature_value('beat_regularity', results['beat_regularity'])

    return results


def analyze_folder_bpm(audio_folder: str | Path,
                        create_grid_if_missing: bool = True,
                        save_to_info: bool = True) -> Dict[str, float]:
    """
    Analyze BPM for an organized audio folder.

    Args:
        audio_folder: Path to organized folder
        create_grid_if_missing: Whether to create beat grid if it doesn't exist
        save_to_info: Whether to save results to .INFO file

    Returns:
        Dictionary with BPM features

    Raises:
        FileNotFoundError: If folder or beat grid doesn't exist
    """
    audio_folder = Path(audio_folder)

    if not audio_folder.exists():
        raise FileNotFoundError(f"Folder not found: {audio_folder}")

    logger.info(f"Analyzing BPM for folder: {audio_folder.name}")

    # Look for beat grid file
    grid_file = audio_folder / f"{audio_folder.name}.BEATS_GRID"

    if not grid_file.exists():
        if create_grid_if_missing:
            logger.info("Beat grid not found, creating...")
            # Find full_mix
            stems = get_stem_files(audio_folder, include_full_mix=True)
            if 'full_mix' not in stems:
                raise FileNotFoundError(f"No full_mix file found in {audio_folder}")

            # Create beat grid
            beat_times, _ = create_beat_grid(stems['full_mix'], save_grid=True)
        else:
            raise FileNotFoundError(f"Beat grid not found: {grid_file}")
    else:
        # Load existing beat grid
        beat_times = load_beat_grid(grid_file)

    # Get audio duration (optional, for better estimation)
    audio_duration = None
    try:
        import soundfile as sf
        stems = get_stem_files(audio_folder, include_full_mix=True)
        if 'full_mix' in stems:
            info = sf.info(str(stems['full_mix']))
            audio_duration = info.duration
    except Exception as e:
        logger.debug(f"Could not get audio duration: {e}")

    # Analyze BPM
    results = analyze_bpm(beat_times, audio_duration)

    logger.info(f"BPM Analysis Results:")
    logger.info(f"  BPM: {results['bpm']:.1f}")
    logger.info(f"  BPM defined: {'Yes' if results['bpm_is_defined'] else 'No'}")
    logger.info(f"  Beat count: {results['beat_count']}")
    logger.info(f"  Regularity: {results['beat_regularity']:.3f}")

    # Save to .INFO file if requested
    if save_to_info and results:
        try:
            stems = get_stem_files(audio_folder, include_full_mix=True)
            if 'full_mix' in stems:
                info_path = get_info_path(stems['full_mix'])
                safe_update(info_path, results)
                logger.info(f"Saved BPM features to {info_path.name}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    return results


def batch_analyze_bpm(root_directory: str | Path,
                      create_grid_if_missing: bool = True,
                      save_to_info: bool = True) -> Dict[str, any]:
    """
    Batch analyze BPM for all organized folders in a directory tree.

    Args:
        root_directory: Root directory to search
        create_grid_if_missing: Whether to create beat grids if missing
        save_to_info: Whether to save results to .INFO files

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch BPM analysis: {root_directory}")

    # Find all organized folders
    folders = find_organized_folders(root_directory)

    stats = {
        'total': len(folders),
        'success': 0,
        'failed': 0,
        'rhythmic': 0,
        'non_rhythmic': 0,
        'errors': []
    }

    logger.info(f"Found {stats['total']} organized folders")

    # Process each folder
    for i, folder in enumerate(folders, 1):
        logger.info(f"Processing {i}/{stats['total']}: {folder.name}")

        try:
            results = analyze_folder_bpm(
                folder,
                create_grid_if_missing=create_grid_if_missing,
                save_to_info=save_to_info
            )
            stats['success'] += 1

            # Track rhythmic vs non-rhythmic
            if results['bpm_is_defined'] == 1:
                stats['rhythmic'] += 1
            else:
                stats['non_rhythmic'] += 1

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch BPM Analysis Summary:")
    logger.info(f"  Total folders:    {stats['total']}")
    logger.info(f"  Successful:       {stats['success']}")
    logger.info(f"  Rhythmic:         {stats['rhythmic']}")
    logger.info(f"  Non-rhythmic:     {stats['non_rhythmic']}")
    logger.info(f"  Failed:           {stats['failed']}")
    logger.info("=" * 60)

    return stats


# Command-line interface
if __name__ == "__main__":
    import argparse
    from core.common import setup_logging

    parser = argparse.ArgumentParser(
        description="Analyze BPM for audio files"
    )

    parser.add_argument(
        'path',
        type=str,
        help='Path to organized folder or root directory for batch processing'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process all organized folders in directory tree'
    )

    parser.add_argument(
        '--no-create-grid',
        action='store_true',
        help='Do not create beat grid if missing (will fail instead)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to .INFO file'
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
            stats = batch_analyze_bpm(
                path,
                create_grid_if_missing=not args.no_create_grid,
                save_to_info=not args.no_save
            )

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} folders failed to process")
                sys.exit(1)

        else:
            # Single folder
            results = analyze_folder_bpm(
                path,
                create_grid_if_missing=not args.no_create_grid,
                save_to_info=not args.no_save
            )

            # Print results
            print("\nBPM Analysis Results:")
            print(f"  BPM:           {results['bpm']:.1f}")
            print(f"  BPM Defined:   {'Yes' if results['bpm_is_defined'] else 'No'}")
            print(f"  Beat Count:    {results['beat_count']}")
            print(f"  Regularity:    {results['beat_regularity']:.3f}")

        logger.info("BPM analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
