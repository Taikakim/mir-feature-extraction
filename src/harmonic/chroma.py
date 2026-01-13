"""
Chroma Features Analysis for MIR Project

This module extracts average chroma features from audio signals.
Chroma represents the pitch class distribution across 12 semitones.

For now, this calculates a simple global average chroma vector for the
entire clip, with each semitone having a weight from 0-1.

Later versions may implement time-variant chroma features.

Dependencies:
- librosa
- numpy
- soundfile
- src.core.file_utils
- src.core.common

Output:
- chroma_0 through chroma_11: Average pitch class weights (0-1) for each semitone
  (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import get_stem_files
from core.common import clamp_feature_value
from core.json_handler import safe_update, get_info_path

logger = logging.getLogger(__name__)

# Pitch class names for reference
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def calculate_average_chroma(audio: np.ndarray,
                              sample_rate: int,
                              n_fft: int = 2048,
                              hop_length: int = 512,
                              n_chroma: int = 12) -> np.ndarray:
    """
    Calculate average chroma vector for entire audio clip.

    Uses constant-Q chromagram for better pitch resolution.

    Args:
        audio: Audio signal (mono)
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Hop length in samples
        n_chroma: Number of chroma bins (12 for semitones)

    Returns:
        Array of 12 chroma values, each 0-1
    """
    # Calculate constant-Q chromagram
    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        n_chroma=n_chroma
    )

    # Average across time
    avg_chroma = np.mean(chroma, axis=1)

    # Normalize to 0-1 range
    # The chromagram is already normalized per frame, but we'll ensure 0-1 range
    max_val = np.max(avg_chroma)
    if max_val > 0:
        avg_chroma = avg_chroma / max_val

    return avg_chroma


def analyze_chroma(audio_path: str | Path,
                   n_fft: int = 2048,
                   hop_length: int = 512) -> Dict[str, float]:
    """
    Analyze chroma features for an audio file.

    Calculates a global average chroma vector representing the
    distribution of pitch classes across the entire clip.

    Args:
        audio_path: Path to audio file
        n_fft: FFT window size
        hop_length: Hop length in samples

    Returns:
        Dictionary with chroma features:
        - chroma_0 through chroma_11: Weights for each semitone (0-1)
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Analyzing chroma: {audio_path.name}")

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)

    logger.debug(f"Loaded audio: {len(audio)} samples @ {sr} Hz")

    # Calculate average chroma
    avg_chroma = calculate_average_chroma(audio, sr, n_fft, hop_length)

    # Compile results
    results = {}
    for i in range(12):
        feature_name = f'chroma_{i}'
        results[feature_name] = float(avg_chroma[i])

    # Clamp values to valid ranges
    for key, value in results.items():
        results[key] = clamp_feature_value(key, value)

    # Log results with pitch class names
    logger.info("Average chroma distribution:")
    for i in range(12):
        pitch_class = PITCH_CLASSES[i]
        weight = results[f'chroma_{i}']
        logger.info(f"  {pitch_class:>2}: {weight:.3f}")

    return results


def batch_analyze_chroma(root_directory: str | Path,
                          overwrite: bool = False,
                          n_fft: int = 2048,
                          hop_length: int = 512) -> dict:
    """
    Batch analyze chroma features for all organized folders.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing chroma data
        n_fft: FFT window size
        hop_length: Hop length in samples

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch chroma analysis: {root_directory}")

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
                if 'chroma_0' in data:
                    logger.info("Chroma data already exists. Use --overwrite to regenerate.")
                    stats['skipped'] += 1
                    continue
            except Exception:
                pass


        try:
            results = analyze_chroma(
                stems['full_mix'],
                n_fft=n_fft,
                hop_length=hop_length
            )

            # Save to .INFO file
            safe_update(info_path, results)

            stats['success'] += 1

            # Log dominant pitch classes
            sorted_chroma = sorted(results.items(), key=lambda x: x[1], reverse=True)
            top_3 = [(int(k.split('_')[1]), v) for k, v in sorted_chroma[:3]]
            top_3_str = ', '.join([f"{PITCH_CLASSES[idx]}:{val:.2f}" for idx, val in top_3])
            logger.info(f"Top 3 pitch classes: {top_3_str}")

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Chroma Analysis Summary:")
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
        description="Analyze chroma features in audio files"
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
        '--n-fft',
        type=int,
        default=2048,
        help='FFT window size (default: 2048)'
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
        help='Overwrite existing chroma data'
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
            stats = batch_analyze_chroma(
                path,
                overwrite=args.overwrite,
                n_fft=args.n_fft,
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

            results = analyze_chroma(
                stems['full_mix'],
                n_fft=args.n_fft,
                hop_length=args.hop_length
            )

            # Save to .INFO
            info_path = get_info_path(path)
            safe_update(info_path, results)

            print(f"\nChroma Features Results:")
            for i in range(12):
                pitch_class = PITCH_CLASSES[i]
                weight = results[f'chroma_{i}']
                print(f"  {pitch_class:>2} (chroma_{i:2d}): {weight:.3f}")

        else:
            # Single file
            results = analyze_chroma(
                path,
                n_fft=args.n_fft,
                hop_length=args.hop_length
            )

            print(f"\nChroma Features Results:")
            for i in range(12):
                pitch_class = PITCH_CLASSES[i]
                weight = results[f'chroma_{i}']
                print(f"  {pitch_class:>2} (chroma_{i:2d}): {weight:.3f}")

        logger.info("Chroma analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
