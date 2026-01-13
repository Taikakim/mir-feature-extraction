"""
Per-Stem Harmonic Features for MIR Project

This module calculates harmonic movement and variance for separated stems.
Harmonic features track how pitch content changes over time.

Features:
- Harmonic movement: Average change in chroma between consecutive frames
- Harmonic variance: Overall variability of chroma across time

Only bass and other stems are analyzed (drums and vocals excluded).

Dependencies:
- librosa
- numpy
- src.core.file_utils
- src.core.common

Output:
- harmonic_movement_bass: Average chroma change in bass (0-1, NumberConditioner)
- harmonic_movement_other: Average chroma change in other (0-1, NumberConditioner)
- harmonic_variance_bass: Chroma variability in bass (0-1, NumberConditioner)
- harmonic_variance_other: Chroma variability in other (0-1, NumberConditioner)
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Tuple
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import get_stem_files
from core.common import clamp_feature_value
from core.json_handler import safe_update, get_info_path

logger = logging.getLogger(__name__)


def calculate_chromagram(audio: np.ndarray,
                          sample_rate: int,
                          hop_length: int = 512,
                          n_chroma: int = 12) -> np.ndarray:
    """
    Calculate chromagram for audio.

    Args:
        audio: Audio signal (mono)
        sample_rate: Sample rate in Hz
        hop_length: Hop length in samples
        n_chroma: Number of chroma bins (12 for semitones)

    Returns:
        Chromagram matrix (n_chroma x n_frames), normalized per frame
    """
    # Calculate constant-Q chromagram
    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        n_chroma=n_chroma
    )

    # Normalize each frame to sum to 1
    chroma_norm = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-8)

    return chroma_norm


def calculate_harmonic_movement(chroma: np.ndarray) -> float:
    """
    Calculate harmonic movement (average chroma change between frames).

    Measures how much the pitch content changes over time.

    Args:
        chroma: Chromagram matrix (n_chroma x n_frames)

    Returns:
        Average movement (0-1, where higher means more change)
    """
    if chroma.shape[1] < 2:
        return 0.0

    # Calculate Euclidean distance between consecutive frames
    diff = np.diff(chroma, axis=1)
    distances = np.sqrt(np.sum(diff ** 2, axis=0))

    # Average distance
    mean_movement = np.mean(distances)

    # Normalize to 0-1 range
    # Max possible distance for normalized chroma is sqrt(2)
    normalized_movement = min(mean_movement / np.sqrt(2), 1.0)

    return float(normalized_movement)


def calculate_harmonic_variance(chroma: np.ndarray) -> float:
    """
    Calculate harmonic variance (overall variability of chroma).

    Measures how much the pitch content varies across the entire track.

    Args:
        chroma: Chromagram matrix (n_chroma x n_frames)

    Returns:
        Variance measure (0-1, where higher means more variable)
    """
    if chroma.shape[1] < 2:
        return 0.0

    # Calculate variance for each chroma bin across time
    variances = np.var(chroma, axis=1)

    # Average variance across all bins
    mean_variance = np.mean(variances)

    # Normalize to 0-1 range
    # For normalized chroma, variance is typically < 0.1
    normalized_variance = min(mean_variance / 0.1, 1.0)

    return float(normalized_variance)


def analyze_stem_harmonics(audio_path: str | Path,
                            hop_length: int = 512) -> Tuple[float, float]:
    """
    Analyze harmonic features for a single stem.

    Args:
        audio_path: Path to stem audio file
        hop_length: Hop length in samples

    Returns:
        Tuple of (movement, variance)
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)

    # Calculate chromagram
    chroma = calculate_chromagram(audio, sr, hop_length)

    # Calculate features
    movement = calculate_harmonic_movement(chroma)
    variance = calculate_harmonic_variance(chroma)

    return movement, variance


def analyze_per_stem_harmonics(folder_path: str | Path,
                                hop_length: int = 512) -> Dict[str, float]:
    """
    Analyze per-stem harmonic features for an organized folder.

    Analyzes bass and other stems (drums and vocals excluded).

    Args:
        folder_path: Path to organized folder with separated stems
        hop_length: Hop length in samples

    Returns:
        Dictionary with harmonic features:
        - harmonic_movement_bass: Average chroma change in bass
        - harmonic_movement_other: Average chroma change in other
        - harmonic_variance_bass: Chroma variability in bass
        - harmonic_variance_other: Chroma variability in other
    """
    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    logger.info(f"Analyzing per-stem harmonics: {folder_path.name}")

    # Get stem files
    stems = get_stem_files(folder_path, include_full_mix=False)

    # We only analyze bass and other
    target_stems = ['bass', 'other']

    results = {}

    for stem_name in target_stems:
        if stem_name not in stems:
            logger.warning(f"Missing {stem_name} stem")
            # Use default values
            results[f'harmonic_movement_{stem_name}'] = 0.0
            results[f'harmonic_variance_{stem_name}'] = 0.0
            continue

        logger.info(f"Processing {stem_name} stem")

        try:
            movement, variance = analyze_stem_harmonics(
                stems[stem_name],
                hop_length=hop_length
            )

            results[f'harmonic_movement_{stem_name}'] = movement
            results[f'harmonic_variance_{stem_name}'] = variance

            logger.info(f"  {stem_name} movement: {movement:.3f}")
            logger.info(f"  {stem_name} variance: {variance:.3f}")

        except Exception as e:
            logger.error(f"Failed to process {stem_name}: {e}")
            results[f'harmonic_movement_{stem_name}'] = 0.0
            results[f'harmonic_variance_{stem_name}'] = 0.0

    # Clamp values to valid ranges
    for key, value in results.items():
        results[key] = clamp_feature_value(key, value)

    return results


def batch_analyze_per_stem_harmonics(root_directory: str | Path,
                                      overwrite: bool = False,
                                      hop_length: int = 512) -> dict:
    """
    Batch analyze per-stem harmonic features for all organized folders.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing data
        hop_length: Hop length in samples

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch per-stem harmonic analysis: {root_directory}")

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
                if 'harmonic_movement_bass' in data:
                    logger.info("Per-stem harmonic data already exists. Use --overwrite to regenerate.")
                    stats['skipped'] += 1
                    continue
            except Exception:
                pass

        try:
            results = analyze_per_stem_harmonics(
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
    logger.info("Batch Per-Stem Harmonic Analysis Summary:")
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
        description="Analyze per-stem harmonic features"
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
        help='Hop length in samples (default: 512)'
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
            stats = batch_analyze_per_stem_harmonics(
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

            results = analyze_per_stem_harmonics(
                path,
                hop_length=args.hop_length
            )

            # Save to .INFO
            info_path = get_info_path(path)
            safe_update(info_path, results)

            print(f"\nPer-Stem Harmonic Features:")
            print(f"  Bass movement:  {results['harmonic_movement_bass']:.3f}")
            print(f"  Bass variance:  {results['harmonic_variance_bass']:.3f}")
            print(f"  Other movement: {results['harmonic_movement_other']:.3f}")
            print(f"  Other variance: {results['harmonic_variance_other']:.3f}")

        logger.info("Per-stem harmonic analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
