"""
Loudness Analysis for MIR Project

This module calculates LUFS (Loudness Units Full Scale) and LRA (Loudness Range)
using the ITU-R BS.1770 standard.

These measurements are essential for:
- Understanding the perceived loudness of audio
- Normalizing audio for consistent playback
- Conditioning audio generation models

Measurements calculated:
- LUFS: Integrated loudness in LUFS (-40 to 0 typical range)
- LRA: Loudness range in LU (0 to 25+ typical range)

Both global (full_mix) and per-stem measurements are supported.

Dependencies:
- pyloudnorm
- librosa
- numpy
- soundfile
- src.core.file_utils
- src.core.common

Output:
- lufs: Integrated loudness (NumberConditioner, -40 to 0 dB)
- lra: Loudness range (NumberConditioner, 0 to 25 LU)
- Per-stem: lufs_bass, lufs_drums, lufs_other, lufs_vocals
- Per-stem: lra_bass, lra_drums, lra_other, lra_vocals
"""

import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from pathlib import Path
from typing import Dict, Tuple
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import get_stem_files
from core.common import clamp_feature_value, DEMUCS_STEMS
from core.json_handler import safe_update, get_info_path

logger = logging.getLogger(__name__)


def calculate_loudness(audio: np.ndarray, sample_rate: int) -> Tuple[float, float]:
    """
    Calculate LUFS and LRA for audio signal.

    Uses pyloudnorm's ITU-R BS.1770 implementation.

    Args:
        audio: Audio signal (mono or stereo)
        sample_rate: Sample rate in Hz

    Returns:
        Tuple of (lufs, lra)
    """
    # Create loudness meter
    meter = pyln.Meter(sample_rate)

    # Ensure audio is in correct shape for pyloudnorm
    # pyloudnorm expects (samples,) for mono or (samples, channels) for stereo
    if audio.ndim == 1:
        audio_for_meter = audio
    elif audio.ndim == 2:
        # If shape is (channels, samples), transpose to (samples, channels)
        if audio.shape[0] < audio.shape[1]:
            audio_for_meter = audio.T
        else:
            audio_for_meter = audio
    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")

    # Calculate integrated loudness (LUFS)
    try:
        lufs = meter.integrated_loudness(audio_for_meter)
    except ValueError as e:
        # Handle silent or very quiet audio
        logger.warning(f"Could not calculate LUFS: {e}")
        lufs = -70.0  # Very quiet

    # Calculate loudness range (LRA)
    # LRA requires computing short-term loudness over time
    # pyloudnorm doesn't have direct LRA function, so we compute it manually
    try:
        # Use short blocks (400ms as per standard)
        block_size = int(0.4 * sample_rate)
        hop_size = block_size // 4  # 75% overlap

        # Calculate short-term loudness for each block
        short_term_loudness = []
        for i in range(0, len(audio_for_meter) - block_size, hop_size):
            block = audio_for_meter[i:i+block_size]
            if audio_for_meter.ndim == 2:
                block_loudness = meter.integrated_loudness(block)
            else:
                block_loudness = meter.integrated_loudness(block)

            if not np.isinf(block_loudness):
                short_term_loudness.append(block_loudness)

        if len(short_term_loudness) > 0:
            # LRA is the difference between 95th and 10th percentile
            short_term_loudness = np.array(short_term_loudness)
            lra = np.percentile(short_term_loudness, 95) - np.percentile(short_term_loudness, 10)
        else:
            lra = 0.0

    except Exception as e:
        logger.warning(f"Could not calculate LRA: {e}")
        lra = 0.0

    return float(lufs), float(lra)


def analyze_loudness(audio_path: str | Path) -> Dict[str, float]:
    """
    Analyze loudness for a single audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with loudness features:
        - lufs: Integrated loudness
        - lra: Loudness range
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Analyzing loudness: {audio_path.name}")

    # Load audio (keep stereo if available for better LUFS accuracy)
    audio, sr = librosa.load(str(audio_path), sr=None, mono=False)

    logger.debug(f"Loaded audio: {audio.shape} @ {sr} Hz")

    # Calculate loudness
    lufs, lra = calculate_loudness(audio, sr)

    # Compile results
    results = {
        'lufs': float(lufs),
        'lra': float(lra)
    }

    # Clamp values to valid ranges
    for key, value in results.items():
        results[key] = clamp_feature_value(key, value)

    logger.info(f"LUFS: {results['lufs']:.2f} dB")
    logger.info(f"LRA: {results['lra']:.2f} LU")

    return results


def analyze_per_stem_loudness(folder_path: str | Path) -> Dict[str, float]:
    """
    Analyze loudness for all stems in an organized folder.

    Calculates LUFS and LRA for each Demucs stem (drums, bass, other, vocals).

    Args:
        folder_path: Path to organized folder with separated stems

    Returns:
        Dictionary with per-stem loudness features:
        - lufs_bass, lufs_drums, lufs_other, lufs_vocals
        - lra_bass, lra_drums, lra_other, lra_vocals
    """
    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    logger.info(f"Analyzing per-stem loudness: {folder_path.name}")

    # Get stem files
    stems = get_stem_files(folder_path, include_full_mix=False)

    results = {}

    for stem_name in DEMUCS_STEMS:
        if stem_name not in stems:
            logger.warning(f"Missing {stem_name} stem")
            # Use default values
            results[f'lufs_{stem_name}'] = -70.0
            results[f'lra_{stem_name}'] = 0.0
            continue

        logger.info(f"Processing {stem_name} stem")

        try:
            # Load audio (keep stereo for LUFS accuracy)
            audio, sr = librosa.load(str(stems[stem_name]), sr=None, mono=False)

            # Calculate loudness
            lufs, lra = calculate_loudness(audio, sr)

            results[f'lufs_{stem_name}'] = lufs
            results[f'lra_{stem_name}'] = lra

            logger.info(f"  {stem_name} LUFS: {lufs:.2f} dB, LRA: {lra:.2f} LU")

        except Exception as e:
            logger.error(f"Failed to process {stem_name}: {e}")
            results[f'lufs_{stem_name}'] = -70.0
            results[f'lra_{stem_name}'] = 0.0

    # Clamp values to valid ranges
    for key, value in results.items():
        results[key] = clamp_feature_value(key, value)

    return results


def batch_analyze_loudness(root_directory: str | Path,
                            overwrite: bool = False,
                            include_stems: bool = False) -> dict:
    """
    Batch analyze loudness for all organized folders.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing loudness data
        include_stems: Whether to also analyze per-stem loudness

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch loudness analysis: {root_directory}")

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

                check_key = 'lufs' if not include_stems else 'lufs_bass'
                if check_key in data:
                    logger.info("Loudness data already exists. Use --overwrite to regenerate.")
                    stats['skipped'] += 1
                    continue
            except Exception:
                pass

        try:
            # Analyze full_mix loudness
            results = analyze_loudness(stems['full_mix'])

            # Optionally analyze per-stem loudness
            if include_stems:
                stem_results = analyze_per_stem_loudness(folder)
                results.update(stem_results)

            # Save to .INFO file
            safe_update(info_path, results)

            stats['success'] += 1
            logger.info(f"Full mix LUFS: {results['lufs']:.2f} dB, LRA: {results['lra']:.2f} LU")

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Loudness Analysis Summary:")
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
        description="Analyze loudness (LUFS/LRA) in audio files"
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
        '--include-stems',
        action='store_true',
        help='Also analyze per-stem loudness'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing loudness data'
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
                overwrite=args.overwrite,
                include_stems=args.include_stems
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

            # Analyze full_mix
            results = analyze_loudness(stems['full_mix'])

            # Optionally analyze stems
            if args.include_stems:
                stem_results = analyze_per_stem_loudness(path)
                results.update(stem_results)

            # Save to .INFO
            info_path = get_info_path(path)
            safe_update(info_path, results)

            print(f"\nLoudness Results:")
            print(f"  LUFS: {results['lufs']:.2f} dB")
            print(f"  LRA:  {results['lra']:.2f} LU")

            if args.include_stems:
                print(f"\nPer-Stem Loudness:")
                for stem in DEMUCS_STEMS:
                    if f'lufs_{stem}' in results:
                        print(f"  {stem.capitalize()}:")
                        print(f"    LUFS: {results[f'lufs_{stem}']:.2f} dB")
                        print(f"    LRA:  {results[f'lra_{stem}']:.2f} LU")

        else:
            # Single file
            results = analyze_loudness(path)

            print(f"\nLoudness Results:")
            print(f"  LUFS: {results['lufs']:.2f} dB")
            print(f"  LRA:  {results['lra']:.2f} LU")

        logger.info("Loudness analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
