"""
Multiband RMS Energy Analysis for MIR Project

This module calculates RMS energy levels across different frequency bands:
- Bass: 20-120 Hz
- Body: 120-600 Hz
- Mid: 600-2500 Hz
- Air: 2500-22000 Hz

This provides a frequency-domain energy profile useful for conditioning.

Dependencies:
- librosa
- numpy
- scipy
- soundfile
- src.core.file_utils
- src.core.common

Output:
- rms_energy_bass: RMS energy in bass band (dB, NumberConditioner)
- rms_energy_body: RMS energy in body band (dB, NumberConditioner)
- rms_energy_mid: RMS energy in mid band (dB, NumberConditioner)
- rms_energy_air: RMS energy in air band (dB, NumberConditioner)
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt
from pathlib import Path
from typing import Dict, Tuple
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import get_stem_files
from core.common import clamp_feature_value, FREQUENCY_BANDS
from core.json_handler import safe_update, get_info_path

logger = logging.getLogger(__name__)


def create_bandpass_filter(lowcut: float,
                           highcut: float,
                           sample_rate: int,
                           order: int = 5) -> np.ndarray:
    """
    Create a Butterworth bandpass filter.

    Args:
        lowcut: Low frequency cutoff (Hz)
        highcut: High frequency cutoff (Hz)
        sample_rate: Sample rate (Hz)
        order: Filter order

    Returns:
        Second-order sections representation of the filter
    """
    nyquist = sample_rate / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist

    # Clamp to valid range
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))

    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def calculate_rms_db(audio: np.ndarray) -> float:
    """
    Calculate RMS energy in dB.

    Args:
        audio: Audio signal

    Returns:
        RMS energy in dB (relative to full scale)
    """
    if len(audio) == 0:
        return -60.0  # Minimum value

    rms = np.sqrt(np.mean(audio ** 2))

    if rms == 0:
        return -60.0

    # Convert to dB (relative to 1.0 = 0 dB)
    rms_db = 20 * np.log10(rms)

    return float(rms_db)


def calculate_band_rms(audio: np.ndarray,
                       sample_rate: int,
                       lowcut: float,
                       highcut: float) -> float:
    """
    Calculate RMS energy for a specific frequency band.

    Args:
        audio: Audio signal (mono)
        sample_rate: Sample rate in Hz
        lowcut: Low frequency cutoff (Hz)
        highcut: High frequency cutoff (Hz)

    Returns:
        RMS energy in dB
    """
    # Create bandpass filter
    sos = create_bandpass_filter(lowcut, highcut, sample_rate)

    # Apply filter
    filtered = sosfilt(sos, audio)

    # Calculate RMS in dB
    rms_db = calculate_rms_db(filtered)

    return rms_db


def analyze_multiband_rms(audio_path: str | Path) -> Dict[str, float]:
    """
    Analyze multiband RMS energy for an audio file.

    Uses frequency bands defined in core.common.FREQUENCY_BANDS:
    - Bass: 20-120 Hz
    - Body: 120-600 Hz
    - Mid: 600-2500 Hz
    - Air: 2500-22000 Hz

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with RMS energy levels in dB:
        - rms_energy_bass: Bass band RMS
        - rms_energy_body: Body band RMS
        - rms_energy_mid: Mid band RMS
        - rms_energy_air: Air band RMS
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Analyzing multiband RMS: {audio_path.name}")

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)

    logger.debug(f"Loaded audio: {len(audio)} samples @ {sr} Hz")

    # Calculate RMS for each band
    results = {}

    for band_name, (lowcut, highcut) in FREQUENCY_BANDS.items():
        logger.debug(f"Processing {band_name} band: {lowcut}-{highcut} Hz")

        rms_db = calculate_band_rms(audio, sr, lowcut, highcut)

        feature_name = f'rms_energy_{band_name}'
        results[feature_name] = rms_db

        logger.info(f"{band_name.capitalize()} RMS: {rms_db:.2f} dB")

    # Clamp values to valid ranges
    for key, value in results.items():
        results[key] = clamp_feature_value(key, value)

    return results


def batch_analyze_multiband_rms(root_directory: str | Path,
                                 overwrite: bool = False) -> dict:
    """
    Batch analyze multiband RMS energy for all organized folders.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing RMS data

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch multiband RMS analysis: {root_directory}")

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
                if 'rms_energy_bass' in data:
                    logger.info("Multiband RMS data already exists. Use --overwrite to regenerate.")
                    stats['skipped'] += 1
                    continue
            except Exception:
                pass


        try:
            results = analyze_multiband_rms(stems['full_mix'])

            # Save to .INFO file
            safe_update(info_path, results)

            stats['success'] += 1
            logger.info(f"Bass: {results['rms_energy_bass']:.2f} dB, "
                       f"Body: {results['rms_energy_body']:.2f} dB")

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Multiband RMS Analysis Summary:")
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
        description="Analyze multiband RMS energy in audio files"
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
        '--overwrite',
        action='store_true',
        help='Overwrite existing RMS data'
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
            stats = batch_analyze_multiband_rms(
                path,
                overwrite=args.overwrite
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

            results = analyze_multiband_rms(stems['full_mix'])

            # Save to .INFO
            info_path = get_info_path(path)
            safe_update(info_path, results)

            print(f"\nMultiband RMS Energy Results:")
            print(f"  Bass (20-120 Hz):      {results['rms_energy_bass']:.2f} dB")
            print(f"  Body (120-600 Hz):     {results['rms_energy_body']:.2f} dB")
            print(f"  Mid (600-2500 Hz):     {results['rms_energy_mid']:.2f} dB")
            print(f"  Air (2500-22000 Hz):   {results['rms_energy_air']:.2f} dB")

        else:
            # Single file
            results = analyze_multiband_rms(path)

            print(f"\nMultiband RMS Energy Results:")
            print(f"  Bass (20-120 Hz):      {results['rms_energy_bass']:.2f} dB")
            print(f"  Body (120-600 Hz):     {results['rms_energy_body']:.2f} dB")
            print(f"  Mid (600-2500 Hz):     {results['rms_energy_mid']:.2f} dB")
            print(f"  Air (2500-22000 Hz):   {results['rms_energy_air']:.2f} dB")

        logger.info("Multiband RMS analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
