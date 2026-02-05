"""
Spectral Features Analysis for MIR Project

This module extracts spectral features from audio signals:
- Spectral flatness: Measure of noisiness vs tonality
- Spectral flux: Rate of spectral change over time
- Spectral skewness: Asymmetry of the spectral distribution
- Spectral kurtosis: Peakedness of the spectral distribution

Dependencies:
- librosa
- numpy
- scipy
- soundfile
- src.core.file_utils
- src.core.common

Output:
- spectral_flatness: Flatness measure 0.0-1.0 (NumberConditioner)
- spectral_flux: Flux measure (NumberConditioner)
- spectral_skewness: Skewness of spectrum (NumberConditioner)
- spectral_kurtosis: Kurtosis of spectrum (NumberConditioner)
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import stats
from pathlib import Path
from typing import Dict, Tuple
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import get_stem_files
from core.common import clamp_feature_value
from core.json_handler import safe_update, get_info_path

logger = logging.getLogger(__name__)


def calculate_spectral_flatness(audio: np.ndarray,
                                sample_rate: int,
                                n_fft: int = 2048,
                                hop_length: int = 512) -> float:
    """
    Calculate spectral flatness (Wiener entropy).

    Flatness measures how noise-like (flat spectrum) vs tonal (peaked spectrum)
    the audio is. Values near 1 indicate noise, near 0 indicate tones.

    Args:
        audio: Audio signal (mono)
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Hop length in samples

    Returns:
        Mean spectral flatness (0.0-1.0)
    """
    # Calculate spectral flatness
    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Return median across time (more robust to outlier frames)
    return float(np.median(flatness))


def calculate_spectral_flux(audio: np.ndarray,
                            sample_rate: int,
                            n_fft: int = 2048,
                            hop_length: int = 512) -> float:
    """
    Calculate spectral flux.

    Flux measures the rate of change in the spectrum over time.
    Higher values indicate more rapid spectral changes.

    Args:
        audio: Audio signal (mono)
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Hop length in samples

    Returns:
        Mean spectral flux
    """
    # Compute spectrogram
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

    # Calculate flux as sum of squared differences between consecutive frames
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))

    # Return median flux (more robust to outlier frames during transitions)
    return float(np.median(flux))


def calculate_spectral_moments(audio: np.ndarray,
                               sample_rate: int,
                               n_fft: int = 2048,
                               hop_length: int = 512) -> Tuple[float, float]:
    """
    Calculate spectral skewness and kurtosis.

    Skewness measures the asymmetry of the spectral distribution.
    Kurtosis measures the peakedness/tailedness of the spectral distribution.

    Args:
        audio: Audio signal (mono)
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Hop length in samples

    Returns:
        Tuple of (skewness, kurtosis)
    """
    # Compute spectrogram
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

    # For each time frame, treat the spectrum as a distribution
    skewness_list = []
    kurtosis_list = []

    for frame_idx in range(S.shape[1]):
        spectrum = S[:, frame_idx]

        # Normalize spectrum to be a probability distribution
        if np.sum(spectrum) > 0:
            spectrum = spectrum / np.sum(spectrum)

            # Calculate moments
            # Create frequency bins
            freq_bins = np.arange(len(spectrum))

            # Calculate skewness and kurtosis
            mean = np.sum(freq_bins * spectrum)
            variance = np.sum(((freq_bins - mean) ** 2) * spectrum)
            std = np.sqrt(variance)

            if std > 0:
                skewness = np.sum(((freq_bins - mean) ** 3) * spectrum) / (std ** 3)
                kurtosis = np.sum(((freq_bins - mean) ** 4) * spectrum) / (variance ** 2)

                skewness_list.append(skewness)
                kurtosis_list.append(kurtosis)

    # Return median values (more robust to outlier frames during transitions)
    median_skewness = float(np.median(skewness_list)) if len(skewness_list) > 0 else 0.0
    median_kurtosis = float(np.median(kurtosis_list)) if len(kurtosis_list) > 0 else 0.0

    return median_skewness, median_kurtosis


def analyze_spectral_features(audio_path: str | Path,
                              n_fft: int = 2048,
                              hop_length: int = 512) -> Dict[str, float]:
    """
    Analyze spectral features for an audio file.

    Args:
        audio_path: Path to audio file
        n_fft: FFT window size
        hop_length: Hop length in samples

    Returns:
        Dictionary with spectral features:
        - spectral_flatness: Flatness measure (0.0-1.0)
        - spectral_flux: Rate of spectral change
        - spectral_skewness: Asymmetry of spectrum
        - spectral_kurtosis: Peakedness of spectrum
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Analyzing spectral features: {audio_path.name}")

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)

    logger.debug(f"Loaded audio: {len(audio)} samples @ {sr} Hz")

    # Calculate spectral features
    flatness = calculate_spectral_flatness(audio, sr, n_fft, hop_length)
    flux = calculate_spectral_flux(audio, sr, n_fft, hop_length)
    skewness, kurtosis = calculate_spectral_moments(audio, sr, n_fft, hop_length)

    # Compile results
    results = {
        'spectral_flatness': float(flatness),
        'spectral_flux': float(flux),
        'spectral_skewness': float(skewness),
        'spectral_kurtosis': float(kurtosis)
    }

    # Clamp values to valid ranges
    for key, value in results.items():
        results[key] = clamp_feature_value(key, value)

    logger.info(f"Spectral flatness: {results['spectral_flatness']:.3f}")
    logger.info(f"Spectral flux: {results['spectral_flux']:.3f}")
    logger.info(f"Spectral skewness: {results['spectral_skewness']:.3f}")
    logger.info(f"Spectral kurtosis: {results['spectral_kurtosis']:.3f}")

    return results


def batch_analyze_spectral_features(root_directory: str | Path,
                                    overwrite: bool = False,
                                    n_fft: int = 2048,
                                    hop_length: int = 512) -> dict:
    """
    Batch analyze spectral features for all organized folders.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing spectral data
        n_fft: FFT window size
        hop_length: Hop length in samples

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch spectral features analysis: {root_directory}")

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
                if 'spectral_flatness' in data:
                    logger.info("Spectral data already exists. Use --overwrite to regenerate.")
                    stats['skipped'] += 1
                    continue
            except Exception:
                pass


        try:
            results = analyze_spectral_features(
                stems['full_mix'],
                n_fft=n_fft,
                hop_length=hop_length
            )

            # Save to .INFO file
            safe_update(info_path, results)

            stats['success'] += 1
            logger.info(f"Flatness: {results['spectral_flatness']:.3f}, "
                       f"Flux: {results['spectral_flux']:.3f}")

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Spectral Features Analysis Summary:")
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
        description="Analyze spectral features in audio files"
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
        help='Overwrite existing spectral data'
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
            stats = batch_analyze_spectral_features(
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

            results = analyze_spectral_features(
                stems['full_mix'],
                n_fft=args.n_fft,
                hop_length=args.hop_length
            )

            # Save to .INFO
            info_path = get_info_path(path)
            safe_update(info_path, results)

            print(f"\nSpectral Features Results:")
            print(f"  Flatness:       {results['spectral_flatness']:.3f}")
            print(f"  Flux:           {results['spectral_flux']:.3f}")
            print(f"  Skewness:       {results['spectral_skewness']:.3f}")
            print(f"  Kurtosis:       {results['spectral_kurtosis']:.3f}")

        else:
            # Single file
            results = analyze_spectral_features(
                path,
                n_fft=args.n_fft,
                hop_length=args.hop_length
            )

            print(f"\nSpectral Features Results:")
            print(f"  Flatness:       {results['spectral_flatness']:.3f}")
            print(f"  Flux:           {results['spectral_flux']:.3f}")
            print(f"  Skewness:       {results['spectral_skewness']:.3f}")
            print(f"  Kurtosis:       {results['spectral_kurtosis']:.3f}")

        logger.info("Spectral features analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
