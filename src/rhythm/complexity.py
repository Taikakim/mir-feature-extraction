"""
Rhythmic Complexity and Evenness Analysis for MIR Project

This module analyzes rhythmic complexity and evenness by examining inter-onset
intervals (IOI) and their variability.

Complexity measures:
- IOI entropy: Entropy of inter-onset interval distribution
- IOI coefficient of variation: Std/mean of IOIs

Evenness measures:
- Regularity: How evenly spaced onsets are (inverse of CV)

Dependencies:
- numpy
- scipy
- src.rhythm.onsets
- src.core.file_utils
- src.core.common

Output:
- rhythmic_complexity: Normalized complexity measure (NumberConditioner)
- rhythmic_evenness: Evenness/regularity of rhythm (NumberConditioner)
- ioi_mean: Mean inter-onset interval in seconds (NumberConditioner)
- ioi_std: Std of inter-onset intervals (NumberConditioner)
"""

import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import get_stem_files
from core.common import clamp_feature_value
from core.json_handler import safe_update, get_info_path

logger = logging.getLogger(__name__)


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


def calculate_inter_onset_intervals(onset_times: np.ndarray) -> np.ndarray:
    """
    Calculate inter-onset intervals (IOIs).

    Args:
        onset_times: Array of onset timestamps in seconds

    Returns:
        Array of inter-onset intervals in seconds
    """
    if len(onset_times) < 2:
        return np.array([])

    # Sort onsets (should already be sorted, but ensure)
    onset_times = np.sort(onset_times)

    # Calculate differences between consecutive onsets
    iois = np.diff(onset_times)

    return iois


def calculate_ioi_entropy(iois: np.ndarray,
                          num_bins: int = 50) -> float:
    """
    Calculate entropy of inter-onset interval distribution.

    Higher entropy indicates more complex/varied rhythm.

    Args:
        iois: Array of inter-onset intervals
        num_bins: Number of bins for histogram

    Returns:
        Entropy value
    """
    if len(iois) < 2:
        return 0.0

    # Create histogram
    hist, _ = np.histogram(iois, bins=num_bins)

    # Normalize to get probability distribution
    hist = hist / np.sum(hist)

    # Remove zero bins
    hist = hist[hist > 0]

    # Calculate entropy
    entropy = stats.entropy(hist)

    return float(entropy)


def calculate_coefficient_of_variation(iois: np.ndarray) -> float:
    """
    Calculate coefficient of variation (CV) of IOIs.

    CV = std / mean
    Higher CV indicates more variable/complex rhythm.

    Args:
        iois: Array of inter-onset intervals

    Returns:
        Coefficient of variation
    """
    if len(iois) < 2:
        return 0.0

    mean_ioi = np.mean(iois)

    if mean_ioi == 0:
        return 0.0

    std_ioi = np.std(iois)
    cv = std_ioi / mean_ioi

    return float(cv)


def calculate_rhythmic_complexity(iois: np.ndarray) -> float:
    """
    Calculate overall rhythmic complexity measure.

    Combines IOI entropy and coefficient of variation.

    Args:
        iois: Array of inter-onset intervals

    Returns:
        Complexity measure (0.0-1.0, normalized)
    """
    if len(iois) < 2:
        return 0.0

    # Calculate both measures
    entropy = calculate_ioi_entropy(iois)
    cv = calculate_coefficient_of_variation(iois)

    # Normalize entropy to 0-1 (typical max entropy is around 4-5 for 50 bins)
    normalized_entropy = min(entropy / 5.0, 1.0)

    # Normalize CV to 0-1 (typical CV range is 0-2+)
    normalized_cv = min(cv / 2.0, 1.0)

    # Combine both measures (average)
    complexity = (normalized_entropy + normalized_cv) / 2.0

    return float(complexity)


def calculate_rhythmic_evenness(iois: np.ndarray) -> float:
    """
    Calculate rhythmic evenness (regularity).

    Evenness is the inverse of variation - higher values mean more regular rhythm.

    Args:
        iois: Array of inter-onset intervals

    Returns:
        Evenness measure (0.0-1.0)
    """
    if len(iois) < 2:
        return 1.0  # Single onset or no onsets is maximally "even"

    cv = calculate_coefficient_of_variation(iois)

    # Evenness is inverse of CV, normalized to 0-1
    # Use 1 / (1 + cv) so that:
    # - cv=0 -> evenness=1.0 (perfectly regular)
    # - cv->inf -> evenness->0.0 (highly irregular)
    evenness = 1.0 / (1.0 + cv)

    return float(evenness)


def analyze_rhythmic_complexity(folder_path: str | Path) -> Dict[str, float]:
    """
    Analyze rhythmic complexity and evenness for an organized music folder.

    Requires:
    - .ONSETS file (from onsets.py)

    Args:
        folder_path: Path to organized folder

    Returns:
        Dictionary with complexity features:
        - rhythmic_complexity: Overall complexity (0.0-1.0)
        - rhythmic_evenness: Regularity of rhythm (0.0-1.0)
        - ioi_mean: Mean inter-onset interval (seconds)
        - ioi_std: Std of inter-onset intervals (seconds)
    """
    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    logger.info(f"Analyzing rhythmic complexity: {folder_path.name}")

    # Find onsets file
    folder_name = folder_path.name
    onsets_file = folder_path / f"{folder_name}.ONSETS"

    if not onsets_file.exists():
        raise FileNotFoundError(
            f"Onsets file not found: {onsets_file}\n"
            "Run onsets.py first to generate onsets."
        )

    # Load onsets
    onset_times = load_onsets(onsets_file)

    logger.info(f"Loaded {len(onset_times)} onsets")

    # Calculate inter-onset intervals
    iois = calculate_inter_onset_intervals(onset_times)

    if len(iois) == 0:
        logger.warning("Not enough onsets to calculate complexity")
        return {
            'rhythmic_complexity': 0.0,
            'rhythmic_evenness': 1.0,
            'ioi_mean': 0.0,
            'ioi_std': 0.0
        }

    # Calculate complexity and evenness
    complexity = calculate_rhythmic_complexity(iois)
    evenness = calculate_rhythmic_evenness(iois)

    # Calculate IOI statistics
    ioi_mean = float(np.mean(iois))
    ioi_std = float(np.std(iois))

    # Compile results
    results = {
        'rhythmic_complexity': float(complexity),
        'rhythmic_evenness': float(evenness),
        'ioi_mean': float(ioi_mean),
        'ioi_std': float(ioi_std)
    }

    # Clamp values to valid ranges
    for key, value in results.items():
        results[key] = clamp_feature_value(key, value)

    logger.info(f"Rhythmic complexity: {results['rhythmic_complexity']:.3f}")
    logger.info(f"Rhythmic evenness: {results['rhythmic_evenness']:.3f}")
    logger.info(f"Mean IOI: {results['ioi_mean']:.3f}s (std: {results['ioi_std']:.3f}s)")

    return results


def batch_analyze_complexity(root_directory: str | Path,
                             overwrite: bool = False) -> dict:
    """
    Batch analyze rhythmic complexity for all organized folders.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing complexity data

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch rhythmic complexity analysis: {root_directory}")

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
                if 'rhythmic_complexity' in data:
                    logger.info("Complexity data already exists. Use --overwrite to regenerate.")
                    stats['skipped'] += 1
                    continue
            except Exception:
                pass

        try:
            results = analyze_rhythmic_complexity(folder)

            # Save to .INFO file
            safe_update(info_path, results)

            stats['success'] += 1
            logger.info(f"Complexity: {results['rhythmic_complexity']:.3f}, "
                       f"Evenness: {results['rhythmic_evenness']:.3f}")

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
    logger.info("Batch Rhythmic Complexity Analysis Summary:")
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
        description="Analyze rhythmic complexity and evenness in audio files"
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
        '--overwrite',
        action='store_true',
        help='Overwrite existing complexity data'
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
            stats = batch_analyze_complexity(
                path,
                overwrite=args.overwrite
            )

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} folders failed to process")
                sys.exit(1)

        else:
            # Single folder
            if not path.is_dir():
                logger.error(f"Path must be a directory: {path}")
                sys.exit(1)

            results = analyze_rhythmic_complexity(path)

            # Save to .INFO
            info_path = get_info_path(path)
            safe_update(info_path, results)

            print(f"\nRhythmic Complexity Analysis Results:")
            print(f"  Complexity:     {results['rhythmic_complexity']:.3f}")
            print(f"  Evenness:       {results['rhythmic_evenness']:.3f}")
            print(f"  Mean IOI:       {results['ioi_mean']:.3f}s")
            print(f"  IOI std:        {results['ioi_std']:.3f}s")

        logger.info("Rhythmic complexity analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
