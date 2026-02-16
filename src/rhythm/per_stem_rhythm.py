"""
Per-Stem Rhythmic Features for MIR Project

This module calculates rhythmic features for separated stems.
Analyzes onset density, syncopation, complexity, and evenness for each stem.

Features calculated for bass, drums, and other stems:
- Onset density (average and variance over time windows)
- Syncopation (off-beat emphasis)
- Rhythmic complexity (IOI entropy + coefficient of variation)
- Rhythmic evenness (regularity of rhythm)

Dependencies:
- librosa
- numpy
- scipy
- src.core.file_utils
- src.core.common
- src.rhythm.onsets
- src.rhythm.syncopation
- src.rhythm.complexity

Output:
- onset_density_average_{stem}: Average onset density (NumberConditioner)
- onset_density_variance_{stem}: Variance of onset density (NumberConditioner)
- syncopation_{stem}: Syncopation measure (NumberConditioner)
- rhythmic_complexity_{stem}: Complexity measure (NumberConditioner)
- rhythmic_evenness_{stem}: Evenness/regularity (NumberConditioner)
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import get_stem_files
from core.common import clamp_feature_value
from core.json_handler import safe_update, get_info_path

# Import functions from other rhythm modules
from rhythm.onsets import detect_onsets
from rhythm.syncopation import calculate_syncopation
from rhythm.complexity import (
    calculate_inter_onset_intervals,
    calculate_rhythmic_complexity,
    calculate_rhythmic_evenness
)

logger = logging.getLogger(__name__)


def load_beats(folder_path: Path) -> np.ndarray:
    """
    Load beat times from .BEATS_GRID file.

    Args:
        folder_path: Path to organized folder

    Returns:
        Array of beat timestamps in seconds
    """
    folder_name = folder_path.name
    beats_file = folder_path / f"{folder_name}.BEATS_GRID"

    if not beats_file.exists():
        raise FileNotFoundError(f"Beats file not found: {beats_file}")

    beat_times = np.loadtxt(beats_file)

    if beat_times.ndim == 0:
        beat_times = np.array([beat_times])

    return beat_times


def calculate_onset_density_stats(onset_times: np.ndarray,
                                   duration: float,
                                   window_size: float = 4.0) -> Tuple[float, float]:
    """
    Calculate average and variance of onset density over time windows.

    Args:
        onset_times: Array of onset timestamps (seconds)
        duration: Total audio duration (seconds)
        window_size: Size of time window for density calculation (seconds)

    Returns:
        Tuple of (average_density, variance_density)
    """
    if len(onset_times) == 0 or duration == 0:
        return 0.0, 0.0

    # Calculate density in overlapping windows
    hop = window_size / 2  # 50% overlap
    densities = []

    window_start = 0
    while window_start < duration:
        window_end = min(window_start + window_size, duration)

        # Count onsets in window
        onsets_in_window = np.sum(
            (onset_times >= window_start) & (onset_times < window_end)
        )

        # Density = onsets per second
        window_duration = window_end - window_start
        density = onsets_in_window / window_duration if window_duration > 0 else 0.0
        densities.append(density)

        window_start += hop

    if len(densities) == 0:
        return 0.0, 0.0

    densities = np.array(densities)
    avg_density = float(np.mean(densities))
    var_density = float(np.var(densities))

    return avg_density, var_density


def analyze_stem_rhythm(audio_path: str | Path,
                         beat_times: np.ndarray,
                         hop_length: int = 512) -> Dict[str, float]:
    """
    Analyze rhythmic features for a single stem.

    Args:
        audio_path: Path to stem audio file
        beat_times: Array of beat timestamps (seconds)
        hop_length: Hop length for onset detection

    Returns:
        Dictionary with rhythmic features:
        - onset_density_average: Average onset density
        - onset_density_variance: Variance of onset density
        - syncopation: Syncopation measure
        - rhythmic_complexity: Complexity measure
        - rhythmic_evenness: Evenness measure
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration = len(audio) / sr

    # Detect onsets
    onset_times, onset_strengths = detect_onsets(audio, sr, hop_length)

    results = {}

    # Onset density stats
    avg_density, var_density = calculate_onset_density_stats(
        onset_times, duration
    )
    results['onset_density_average'] = avg_density
    results['onset_density_variance'] = var_density

    # Syncopation (requires beats)
    if len(beat_times) > 0 and len(onset_times) > 0:
        syncopation = calculate_syncopation(
            onset_times,
            beat_times,
            onset_strengths
        )
        results['syncopation'] = syncopation
    else:
        results['syncopation'] = 0.0

    # Rhythmic complexity and evenness (requires IOIs)
    if len(onset_times) >= 2:
        iois = calculate_inter_onset_intervals(onset_times)
        complexity = calculate_rhythmic_complexity(iois)
        evenness = calculate_rhythmic_evenness(iois)

        results['rhythmic_complexity'] = complexity
        results['rhythmic_evenness'] = evenness
    else:
        results['rhythmic_complexity'] = 0.0
        results['rhythmic_evenness'] = 1.0  # Default to perfectly even

    return results


def analyze_per_stem_rhythm(folder_path: str | Path,
                             hop_length: int = 512) -> Dict[str, float]:
    """
    Analyze per-stem rhythmic features for an organized folder.

    Analyzes bass, drums, and other stems.

    Args:
        folder_path: Path to organized folder with separated stems
        hop_length: Hop length for onset detection

    Returns:
        Dictionary with per-stem rhythmic features
    """
    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    logger.info(f"Analyzing per-stem rhythm: {folder_path.name}")

    # Load beat times
    try:
        beat_times = load_beats(folder_path)
        logger.info(f"Loaded {len(beat_times)} beats")
    except FileNotFoundError as e:
        logger.warning(f"No beats file found: {e}")
        logger.warning("Syncopation features will be unavailable")
        beat_times = np.array([])

    # Get stem files
    stems = get_stem_files(folder_path, include_full_mix=False)

    # Analyze these stems
    target_stems = ['bass', 'drums', 'other']

    all_results = {}

    for stem_name in target_stems:
        if stem_name not in stems:
            logger.warning(f"Missing {stem_name} stem")
            # Use default values
            all_results[f'onset_density_average_{stem_name}'] = 0.0
            all_results[f'onset_density_variance_{stem_name}'] = 0.0
            all_results[f'syncopation_{stem_name}'] = 0.0
            all_results[f'rhythmic_complexity_{stem_name}'] = 0.0
            all_results[f'rhythmic_evenness_{stem_name}'] = 1.0
            continue

        logger.info(f"Processing {stem_name} stem")

        try:
            results = analyze_stem_rhythm(
                stems[stem_name],
                beat_times,
                hop_length=hop_length
            )

            # Add stem suffix to keys
            for key, value in results.items():
                all_results[f'{key}_{stem_name}'] = value

            logger.info(f"  {stem_name} density avg: {results['onset_density_average']:.2f}")
            logger.info(f"  {stem_name} syncopation: {results['syncopation']:.3f}")

        except Exception as e:
            logger.error(f"Failed to process {stem_name}: {e}")
            all_results[f'onset_density_average_{stem_name}'] = 0.0
            all_results[f'onset_density_variance_{stem_name}'] = 0.0
            all_results[f'syncopation_{stem_name}'] = 0.0
            all_results[f'rhythmic_complexity_{stem_name}'] = 0.0
            all_results[f'rhythmic_evenness_{stem_name}'] = 1.0

    # Clamp values to valid ranges
    for key, value in all_results.items():
        all_results[key] = clamp_feature_value(key, value)

    return all_results


def batch_analyze_per_stem_rhythm(root_directory: str | Path,
                                   overwrite: bool = False,
                                   hop_length: int = 512) -> dict:
    """
    Batch analyze per-stem rhythmic features for all organized folders.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing data
        hop_length: Hop length for onset detection

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch per-stem rhythm analysis: {root_directory}")

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

        # Check if already processed — require ALL target stems to have data
        info_path = get_info_path(stems['full_mix'])
        target_stems = ['bass', 'drums', 'other']
        if info_path.exists() and not overwrite:
            try:
                import json
                with open(info_path, 'r') as f:
                    data = json.load(f)
                sentinel_keys = [f'onset_density_average_{s}' for s in target_stems]
                if all(k in data for k in sentinel_keys):
                    logger.info("Per-stem rhythm data already exists. Use --overwrite to regenerate.")
                    stats['skipped'] += 1
                    continue
            except Exception:
                pass

        try:
            results = analyze_per_stem_rhythm(
                folder,
                hop_length=hop_length
            )

            # Save to .INFO file
            safe_update(info_path, results)

            stats['success'] += 1

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Per-Stem Rhythm Analysis Summary:")
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
        description="Analyze per-stem rhythmic features"
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
        '--hop-length',
        type=int,
        default=512,
        help='Hop length for onset detection (default: 512)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing data'
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
            stats = batch_analyze_per_stem_rhythm(
                path,
                overwrite=args.overwrite,
                hop_length=args.hop_length
            )

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} folders failed to process")
                sys.exit(1)

        else:
            # Single folder
            if not path.is_dir():
                logger.error(f"Path must be a directory: {path}")
                sys.exit(1)

            results = analyze_per_stem_rhythm(
                path,
                hop_length=args.hop_length
            )

            # Save to .INFO
            info_path = get_info_path(path)
            safe_update(info_path, results)

            print(f"\nPer-Stem Rhythmic Features:")
            for stem in ['bass', 'drums', 'other']:
                print(f"\n  {stem.capitalize()}:")
                print(f"    Density avg:  {results[f'onset_density_average_{stem}']:.2f}")
                print(f"    Density var:  {results[f'onset_density_variance_{stem}']:.2f}")
                print(f"    Syncopation:  {results[f'syncopation_{stem}']:.3f}")
                print(f"    Complexity:   {results[f'rhythmic_complexity_{stem}']:.3f}")
                print(f"    Evenness:     {results[f'rhythmic_evenness_{stem}']:.3f}")

        logger.info("Per-stem rhythm analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
