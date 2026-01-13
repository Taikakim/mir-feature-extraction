"""
Loudness Analysis for MIR Project

This module extracts loudness features using ITU-R BS.1770 standard.
Calculates LUFS (Loudness Units relative to Full Scale) and LRA (Loudness Range).

Dependencies:
- pyloudnorm
- soundfile
- src.core.json_handler
- src.core.file_utils
- src.core.common

Features extracted:
- {lufs}: Integrated loudness (-40 to 0 dB)
- {lra}: Loudness range (0 to 25 LU)
- Per-stem: {lufs_drums}, {lufs_bass}, {lufs_other}, {lufs_vocals}
- Per-stem: {lra_drums}, {lra_bass}, {lra_other}, {lra_vocals}
"""

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.json_handler import safe_update, get_info_path
from core.file_utils import get_stem_files
from core.common import clamp_feature_value, DEMUCS_STEMS

logger = logging.getLogger(__name__)


def calculate_loudness(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """
    Calculate LUFS and LRA for an audio signal.

    Args:
        audio: Audio samples (can be mono or multi-channel)
        sample_rate: Sample rate in Hz

    Returns:
        Dictionary with 'lufs' and 'lra' values

    Raises:
        ValueError: If audio is empty or sample rate is invalid
    """
    if audio.size == 0:
        raise ValueError("Empty audio array")

    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")

    # Ensure audio is 2D (channels, samples) or 1D for mono
    if audio.ndim == 1:
        # Mono audio
        audio_2d = audio.reshape(1, -1)
    elif audio.ndim == 2:
        # Multi-channel - transpose if needed (pyloudnorm expects samples x channels)
        if audio.shape[0] < audio.shape[1]:
            # Likely (channels, samples) - transpose to (samples, channels)
            audio_2d = audio.T
        else:
            audio_2d = audio
    else:
        raise ValueError(f"Unexpected audio dimensions: {audio.ndim}")

    # Create loudness meter
    meter = pyln.Meter(sample_rate)

    try:
        # Calculate integrated loudness (LUFS)
        lufs = meter.integrated_loudness(audio_2d)

        # Calculate loudness range (LRA)
        # Note: pyloudnorm doesn't have built-in LRA, we'll use a simplified version
        # For full EBU R128 LRA, would need more complex implementation
        # Using percentile-based approach as approximation
        block_size = int(0.4 * sample_rate)  # 400ms blocks
        hop_size = int(0.1 * sample_rate)    # 100ms hop

        if audio_2d.shape[0] < block_size:
            # Audio too short for proper LRA calculation
            lra = 0.0
            logger.warning("Audio too short for proper LRA calculation")
        else:
            # Calculate loudness for overlapping blocks
            block_loudnesses = []
            for i in range(0, audio_2d.shape[0] - block_size, hop_size):
                block = audio_2d[i:i+block_size]
                try:
                    block_loudness = meter.integrated_loudness(block)
                    if not np.isinf(block_loudness):  # Skip infinite values
                        block_loudnesses.append(block_loudness)
                except:
                    pass

            if len(block_loudnesses) > 0:
                # LRA is the difference between 95th and 10th percentiles
                loudnesses = np.array(block_loudnesses)
                p95 = np.percentile(loudnesses, 95)
                p10 = np.percentile(loudnesses, 10)
                lra = p95 - p10
            else:
                lra = 0.0
                logger.warning("Could not calculate block loudnesses for LRA")

    except Exception as e:
        logger.error(f"Error calculating loudness: {e}")
        # Return sensible defaults
        lufs = -40.0
        lra = 0.0

    # Clamp values to valid ranges
    lufs = clamp_feature_value('lufs', lufs)
    lra = clamp_feature_value('lra', lra)

    return {
        'lufs': float(lufs),
        'lra': float(lra)
    }


def analyze_file_loudness(audio_path: str | Path,
                           prefix: str = '') -> Dict[str, float]:
    """
    Analyze loudness of a single audio file.

    Args:
        audio_path: Path to audio file
        prefix: Optional prefix for feature keys (e.g., 'drums' -> 'lufs_drums')

    Returns:
        Dictionary with loudness features

    Raises:
        FileNotFoundError: If audio file doesn't exist
        Exception: If audio cannot be loaded
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Analyzing loudness: {audio_path.name}")

    try:
        # Load audio
        audio, sr = sf.read(str(audio_path))

        # Calculate loudness
        loudness = calculate_loudness(audio, sr)

        # Add prefix if specified
        if prefix:
            return {
                f'lufs_{prefix}': loudness['lufs'],
                f'lra_{prefix}': loudness['lra']
            }
        else:
            return loudness

    except Exception as e:
        logger.error(f"Error analyzing {audio_path}: {e}")
        raise


def analyze_folder_loudness(audio_folder: str | Path,
                             analyze_stems: bool = True,
                             save_to_info: bool = True) -> Dict[str, float]:
    """
    Analyze loudness for all files in an organized audio folder.

    Args:
        audio_folder: Path to organized folder (contains full_mix.flac, etc.)
        analyze_stems: Whether to analyze individual stems
        save_to_info: Whether to save results to .INFO file

    Returns:
        Dictionary with all loudness features

    Raises:
        FileNotFoundError: If folder or full_mix doesn't exist
    """
    audio_folder = Path(audio_folder)

    if not audio_folder.exists():
        raise FileNotFoundError(f"Folder not found: {audio_folder}")

    logger.info(f"Analyzing loudness for folder: {audio_folder.name}")

    results = {}

    # Get all audio files
    stems = get_stem_files(audio_folder, include_full_mix=True)

    if 'full_mix' not in stems:
        raise FileNotFoundError(f"No full_mix file found in {audio_folder}")

    # Analyze full mix (no prefix)
    try:
        full_mix_loudness = analyze_file_loudness(stems['full_mix'])
        results.update(full_mix_loudness)
        logger.info(f"Full mix: LUFS={full_mix_loudness['lufs']:.1f} dB, LRA={full_mix_loudness['lra']:.1f} LU")
    except Exception as e:
        logger.error(f"Error analyzing full_mix: {e}")

    # Analyze stems if requested
    if analyze_stems:
        for stem_name in DEMUCS_STEMS:
            if stem_name in stems:
                try:
                    stem_loudness = analyze_file_loudness(stems[stem_name], prefix=stem_name)
                    results.update(stem_loudness)
                    logger.info(f"  {stem_name}: LUFS={stem_loudness[f'lufs_{stem_name}']:.1f} dB")
                except Exception as e:
                    logger.warning(f"Could not analyze {stem_name}: {e}")

    # Save to .INFO file if requested
    if save_to_info and results:
        try:
            info_path = get_info_path(stems['full_mix'])
            safe_update(info_path, results)
            logger.info(f"Saved {len(results)} loudness features to {info_path.name}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    return results


def batch_analyze_loudness(root_directory: str | Path,
                            analyze_stems: bool = True,
                            save_to_info: bool = True) -> Dict[str, Any]:
    """
    Batch analyze loudness for all organized folders in a directory tree.

    Args:
        root_directory: Root directory to search
        analyze_stems: Whether to analyze individual stems
        save_to_info: Whether to save results to .INFO files

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch loudness analysis: {root_directory}")

    # Find all organized folders
    folders = find_organized_folders(root_directory)

    stats = {
        'total': len(folders),
        'success': 0,
        'failed': 0,
        'errors': []
    }

    logger.info(f"Found {stats['total']} organized folders")

    # Process each folder
    for i, folder in enumerate(folders, 1):
        logger.info(f"Processing {i}/{stats['total']}: {folder.name}")

        try:
            analyze_folder_loudness(folder, analyze_stems=analyze_stems, save_to_info=save_to_info)
            stats['success'] += 1
        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Loudness Analysis Summary:")
    logger.info(f"  Total folders: {stats['total']}")
    logger.info(f"  Successful:    {stats['success']}")
    logger.info(f"  Failed:        {stats['failed']}")
    logger.info("=" * 60)

    return stats


# Command-line interface
if __name__ == "__main__":
    import argparse
    from core.common import setup_logging

    parser = argparse.ArgumentParser(
        description="Analyze loudness (LUFS/LRA) for audio files"
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
        '--no-stems',
        action='store_true',
        help='Do not analyze individual stems (only full mix)'
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
            stats = batch_analyze_loudness(
                path,
                analyze_stems=not args.no_stems,
                save_to_info=not args.no_save
            )

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} folders failed to process")
                sys.exit(1)

        elif path.is_dir():
            # Single folder
            results = analyze_folder_loudness(
                path,
                analyze_stems=not args.no_stems,
                save_to_info=not args.no_save
            )

            # Print results
            print("\nLoudness Analysis Results:")
            for key, value in sorted(results.items()):
                print(f"  {key}: {value:.2f}")

        else:
            # Single file
            results = analyze_file_loudness(path)

            print("\nLoudness Analysis Results:")
            print(f"  LUFS: {results['lufs']:.2f} dB")
            print(f"  LRA:  {results['lra']:.2f} LU")

        logger.info("Loudness analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
