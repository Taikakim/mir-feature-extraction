"""
Onset Detection and Density Analysis for MIR Project

This module detects onsets (note/percussion attacks) in audio and calculates
onset density for rhythmic complexity analysis.

Dependencies:
- librosa
- numpy
- soundfile
- src.core.file_utils
- src.core.common

Output:
- onset_count: Total number of detected onsets (IntConditioner)
- onset_density: Onsets per second (NumberConditioner)
- onset_strength_mean: Average onset strength (NumberConditioner)
- onset_strength_std: Onset strength variability (NumberConditioner)
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import get_stem_files
from core.common import clamp_feature_value
from core.json_handler import safe_update, get_info_path

logger = logging.getLogger(__name__)


def detect_onsets(audio: np.ndarray,
                  sample_rate: int,
                  hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect onsets in audio signal using librosa.

    Args:
        audio: Audio signal (mono)
        sample_rate: Sample rate in Hz
        hop_length: Hop length in samples for onset detection

    Returns:
        Tuple of (onset_times, onset_strengths)
        onset_times: Array of onset timestamps in seconds
        onset_strengths: Array of onset strength values
    """
    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length
    )

    # Detect onset frames
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop_length,
        backtrack=True  # Refine onset positions
    )

    # Convert frames to time
    onset_times = librosa.frames_to_time(
        onset_frames,
        sr=sample_rate,
        hop_length=hop_length
    )

    # Get onset strengths at detected positions
    onset_strengths = onset_env[onset_frames]

    return onset_times, onset_strengths


def calculate_onset_density(onset_times: np.ndarray,
                            duration: float) -> float:
    """
    Calculate onset density (onsets per second).

    Args:
        onset_times: Array of onset timestamps in seconds
        duration: Total duration of audio in seconds

    Returns:
        Onset density in onsets per second
    """
    if duration <= 0:
        return 0.0

    return float(len(onset_times) / duration)


def analyze_onset_statistics(onset_strengths: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics about onset strengths.

    Args:
        onset_strengths: Array of onset strength values

    Returns:
        Dictionary with mean and std of onset strengths
    """
    if len(onset_strengths) == 0:
        return {
            'onset_strength_mean': 0.0,
            'onset_strength_std': 0.0
        }

    return {
        'onset_strength_mean': float(np.mean(onset_strengths)),
        'onset_strength_std': float(np.std(onset_strengths))
    }


def analyze_onsets(audio_path: str | Path,
                   hop_length: int = 512) -> Dict[str, float]:
    """
    Analyze onsets in an audio file.

    Args:
        audio_path: Path to audio file
        hop_length: Hop length for onset detection

    Returns:
        Dictionary with onset features:
        - onset_count: Number of detected onsets
        - onset_density: Onsets per second
        - onset_strength_mean: Mean onset strength
        - onset_strength_std: Std of onset strength
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Analyzing onsets: {audio_path.name}")

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration = len(audio) / sr

    # Detect onsets
    onset_times, onset_strengths = detect_onsets(audio, sr, hop_length)

    logger.info(f"Detected {len(onset_times)} onsets in {duration:.1f}s")

    # Calculate onset density
    onset_density = calculate_onset_density(onset_times, duration)

    # Calculate onset strength statistics
    strength_stats = analyze_onset_statistics(onset_strengths)

    # Compile results
    results = {
        'onset_count': int(len(onset_times)),
        'onset_density': float(onset_density),
        **strength_stats
    }

    # Clamp values to valid ranges
    for key, value in results.items():
        results[key] = clamp_feature_value(key, value)

    logger.info(f"Onset density: {results['onset_density']:.2f} onsets/sec")
    logger.info(f"Onset strength: {results['onset_strength_mean']:.3f} ± {results['onset_strength_std']:.3f}")

    return results


def save_onset_times(onset_times: np.ndarray,
                     output_path: str | Path) -> None:
    """
    Save onset timestamps to a text file.

    Args:
        onset_times: Array of onset timestamps in seconds
        output_path: Path to output .ONSETS file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'w') as f:
            for onset_time in onset_times:
                f.write(f"{onset_time:.6f}\n")

        logger.debug(f"Saved {len(onset_times)} onsets to {output_path.name}")

    except Exception as e:
        logger.error(f"Error saving onsets: {e}")
        raise


def analyze_onsets_with_save(audio_path: str | Path,
                             save_onsets: bool = True,
                             hop_length: int = 512) -> Tuple[Dict[str, float], Optional[Path]]:
    """
    Analyze onsets and optionally save onset timestamps.

    Args:
        audio_path: Path to audio file
        save_onsets: Whether to save onset timestamps to file
        hop_length: Hop length for onset detection

    Returns:
        Tuple of (results_dict, onsets_file_path)
        onsets_file_path is None if save_onsets=False
    """
    audio_path = Path(audio_path)

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration = len(audio) / sr

    # Detect onsets
    onset_times, onset_strengths = detect_onsets(audio, sr, hop_length)

    # Calculate features
    onset_density = calculate_onset_density(onset_times, duration)
    strength_stats = analyze_onset_statistics(onset_strengths)

    results = {
        'onset_count': int(len(onset_times)),
        'onset_density': float(onset_density),
        **strength_stats
    }

    # Clamp values
    for key, value in results.items():
        results[key] = clamp_feature_value(key, value)

    onsets_file_path = None

    if save_onsets:
        # Save onsets to .ONSETS file
        folder_name = audio_path.parent.name
        onsets_file_path = audio_path.parent / f"{folder_name}.ONSETS"
        save_onset_times(onset_times, onsets_file_path)

    return results, onsets_file_path


def batch_analyze_onsets(root_directory: str | Path,
                         overwrite: bool = False,
                         save_onsets: bool = True,
                         hop_length: int = 512) -> dict:
    """
    Batch analyze onsets for all organized folders.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing onset data
        save_onsets: Whether to save onset timestamps
        hop_length: Hop length for onset detection

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch onset analysis: {root_directory}")

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
                if 'onset_count' in data:
                    logger.info("Onset data already exists. Use --overwrite to regenerate.")
                    stats['skipped'] += 1
                    continue
            except Exception:
                pass


        try:
            results, onsets_file = analyze_onsets_with_save(
                stems['full_mix'],
                save_onsets=save_onsets,
                hop_length=hop_length
            )

            # Save to .INFO file
            safe_update(info_path, results)

            stats['success'] += 1
            logger.info(f"Analyzed {results['onset_count']} onsets "
                       f"({results['onset_density']:.2f} onsets/sec)")

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Onset Analysis Summary:")
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
        description="Analyze onsets in audio files"
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
        '--hop-length',
        type=int,
        default=512,
        help='Hop length for onset detection (default: 512)'
    )

    parser.add_argument(
        '--no-save-onsets',
        action='store_true',
        help='Do not save onset timestamps to .ONSETS file'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing onset data'
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
            stats = batch_analyze_onsets(
                path,
                overwrite=args.overwrite,
                save_onsets=not args.no_save_onsets,
                hop_length=args.hop_length
            )

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} folders failed to process")
                sys.exit(1)

        elif path.is_dir():
            # Single folder
            stems = get_stem_files(path, include_full_mix=True)
            if 'full_mix' not in stems:
                logger.error(f"No full_mix file found in {path}")
                sys.exit(1)

            results, onsets_file = analyze_onsets_with_save(
                stems['full_mix'],
                save_onsets=not args.no_save_onsets,
                hop_length=args.hop_length
            )

            # Save to .INFO
            info_path = get_info_path(path)
            safe_update(info_path, results)

            print(f"\nOnset Analysis Results:")
            print(f"  Onset count:    {results['onset_count']}")
            print(f"  Onset density:  {results['onset_density']:.2f} onsets/sec")
            print(f"  Onset strength: {results['onset_strength_mean']:.3f} ± {results['onset_strength_std']:.3f}")
            if onsets_file:
                print(f"  Onsets file:    {onsets_file}")

        else:
            # Single file
            results, onsets_file = analyze_onsets_with_save(
                path,
                save_onsets=not args.no_save_onsets,
                hop_length=args.hop_length
            )

            print(f"\nOnset Analysis Results:")
            print(f"  Onset count:    {results['onset_count']}")
            print(f"  Onset density:  {results['onset_density']:.2f} onsets/sec")
            print(f"  Onset strength: {results['onset_strength_mean']:.3f} ± {results['onset_strength_std']:.3f}")

        logger.info("Onset analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
