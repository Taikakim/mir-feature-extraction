"""
Chroma Features Analysis for MIR Project (Essentia HPCP)

This module extracts average chroma features using Essentia's HPCP
(Harmonic Pitch Class Profile) algorithm.

Uses unitSum normalization so values represent a probability distribution
of pitch classes - comparable across different audio files for AI training.

Dependencies:
- essentia
- numpy
- soundfile
- src.core.file_utils
- src.core.common

Output:
- chroma_0 through chroma_11: Pitch class weights (sum to 1.0) for each semitone
  (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
"""

import numpy as np
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


def calculate_hpcp(audio: np.ndarray,
                   sample_rate: int,
                   frame_size: int = 4096,
                   hop_size: int = 2048) -> np.ndarray:
    """
    Calculate average HPCP (Harmonic Pitch Class Profile) for audio.

    Uses Essentia's HPCP algorithm with unitSum normalization for
    cross-file comparability in AI training.

    Args:
        audio: Audio signal (mono, float32)
        sample_rate: Sample rate in Hz
        frame_size: Analysis frame size
        hop_size: Hop size between frames

    Returns:
        Array of 12 chroma values that sum to 1.0
    """
    import essentia
    from essentia.standard import (
        Windowing, Spectrum, SpectralPeaks, HPCP, FrameGenerator
    )

    # Ensure float32 for Essentia
    audio = audio.astype(np.float32)

    # Initialize Essentia algorithms
    windowing = Windowing(type='blackmanharris62')
    spectrum = Spectrum()
    spectral_peaks = SpectralPeaks(
        sampleRate=sample_rate,
        maxPeaks=100,
        magnitudeThreshold=0.00001,
        minFrequency=20,
        maxFrequency=sample_rate / 2 - 1
    )
    hpcp = HPCP(
        size=12,
        sampleRate=sample_rate,
        normalized='unitSum',  # Values sum to 1.0 - comparable across files
        harmonics=4,           # Include harmonics for better pitch detection
        bandPreset=False       # Disable band preset to avoid warnings with unitSum
    )

    # Process frames and accumulate HPCP
    hpcp_frames = []

    for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size,
                                 startFromZero=True):
        windowed = windowing(frame)
        spec = spectrum(windowed)
        frequencies, magnitudes = spectral_peaks(spec)

        # Skip frames with no peaks
        if len(frequencies) > 0:
            frame_hpcp = hpcp(frequencies, magnitudes)
            hpcp_frames.append(frame_hpcp)

    if not hpcp_frames:
        logger.warning("No valid HPCP frames extracted, returning uniform distribution")
        return np.ones(12) / 12

    # Average across all frames
    avg_hpcp = np.mean(hpcp_frames, axis=0)

    # Re-normalize to ensure sum = 1.0 after averaging
    hpcp_sum = np.sum(avg_hpcp)
    if hpcp_sum > 0:
        avg_hpcp = avg_hpcp / hpcp_sum

    return avg_hpcp


def analyze_chroma(audio_path: str | Path,
                   frame_size: int = 4096,
                   hop_size: int = 2048) -> Dict[str, float]:
    """
    Analyze chroma features for an audio file using Essentia HPCP.

    Calculates a global average HPCP vector representing the
    probability distribution of pitch classes across the entire clip.

    Args:
        audio_path: Path to audio file
        frame_size: Analysis frame size
        hop_size: Hop size between frames

    Returns:
        Dictionary with chroma features:
        - chroma_0 through chroma_11: Weights for each semitone (sum to 1.0)
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Analyzing chroma (HPCP): {audio_path.name}")

    # Load audio with soundfile (avoids librosa/numba issues)
    audio, sr = sf.read(str(audio_path), dtype='float32')

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    logger.debug(f"Loaded audio: {len(audio)} samples @ {sr} Hz")

    # Calculate average HPCP
    avg_hpcp = calculate_hpcp(audio, sr, frame_size, hop_size)

    # Compile results
    results = {}
    for i in range(12):
        feature_name = f'chroma_{i}'
        results[feature_name] = float(avg_hpcp[i])

    # Clamp values to valid ranges
    for key, value in results.items():
        results[key] = clamp_feature_value(key, value)

    # Log results with pitch class names
    logger.info("HPCP pitch class distribution (unitSum normalized):")
    for i in range(12):
        pitch_class = PITCH_CLASSES[i]
        weight = results[f'chroma_{i}']
        bar = '█' * int(weight * 40)
        logger.info(f"  {pitch_class:>2}: {weight:.3f} {bar}")

    # Verify sum ≈ 1.0
    total = sum(results.values())
    logger.debug(f"Sum of chroma values: {total:.4f}")

    return results


def batch_analyze_chroma(root_directory: str | Path,
                          overwrite: bool = False,
                          frame_size: int = 4096,
                          hop_size: int = 2048) -> dict:
    """
    Batch analyze chroma features for all organized folders.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing chroma data
        frame_size: Analysis frame size
        hop_size: Hop size between frames

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch HPCP chroma analysis: {root_directory}")

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
                frame_size=frame_size,
                hop_size=hop_size
            )

            # Save to .INFO file
            safe_update(info_path, results)

            stats['success'] += 1

            # Log dominant pitch classes
            sorted_chroma = sorted(results.items(), key=lambda x: x[1], reverse=True)
            top_3 = [(int(k.split('_')[1]), v) for k, v in sorted_chroma[:3]]
            top_3_str = ', '.join([f"{PITCH_CLASSES[idx]}:{val:.3f}" for idx, val in top_3])
            logger.info(f"Top 3 pitch classes: {top_3_str}")

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch HPCP Chroma Analysis Summary:")
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
        description="Analyze chroma features using Essentia HPCP"
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
        '--frame-size',
        type=int,
        default=4096,
        help='Analysis frame size (default: 4096)'
    )

    parser.add_argument(
        '--hop-size',
        type=int,
        default=2048,
        help='Hop size in samples (default: 2048)'
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
                frame_size=args.frame_size,
                hop_size=args.hop_size
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
                frame_size=args.frame_size,
                hop_size=args.hop_size
            )

            # Save to .INFO
            info_path = get_info_path(path)
            safe_update(info_path, results)

            print(f"\nHPCP Chroma Features (unitSum normalized, sum ≈ 1.0):")
            for i in range(12):
                pitch_class = PITCH_CLASSES[i]
                weight = results[f'chroma_{i}']
                bar = '█' * int(weight * 40)
                print(f"  {pitch_class:>2} (chroma_{i:2d}): {weight:.3f} {bar}")
            print(f"\n  Sum: {sum(results.values()):.4f}")

        else:
            # Single file
            results = analyze_chroma(
                path,
                frame_size=args.frame_size,
                hop_size=args.hop_size
            )

            print(f"\nHPCP Chroma Features (unitSum normalized, sum ≈ 1.0):")
            for i in range(12):
                pitch_class = PITCH_CLASSES[i]
                weight = results[f'chroma_{i}']
                bar = '█' * int(weight * 40)
                print(f"  {pitch_class:>2} (chroma_{i:2d}): {weight:.3f} {bar}")
            print(f"\n  Sum: {sum(results.values()):.4f}")

        logger.info("HPCP chroma analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
