"""
Spectral Features Analysis for MIR Project

This module extracts spectral features from audio signals using Essentia:
- Spectral flatness: Measure of noisiness vs tonality
- Spectral flux: Rate of spectral change over time
- Spectral skewness: Asymmetry of the spectral distribution
- Spectral kurtosis: Peakedness of the spectral distribution

Dependencies:
- essentia
- numpy
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
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import get_stem_files
from core.common import clamp_feature_value
from core.json_handler import safe_update, get_info_path

logger = logging.getLogger(__name__)

# Try to import Essentia
try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    logger.warning("Essentia not available - spectral features will use fallback")


def analyze_spectral_features_essentia(audio: np.ndarray,
                                       sample_rate: int,
                                       frame_size: int = 2048,
                                       hop_size: int = 512) -> Dict[str, float]:
    """
    Calculate spectral features using Essentia (faster than librosa).

    Args:
        audio: Audio signal (mono, float32)
        sample_rate: Sample rate in Hz
        frame_size: FFT window size
        hop_size: Hop size in samples

    Returns:
        Dictionary with spectral features
    """
    if not ESSENTIA_AVAILABLE:
        raise ImportError("Essentia is required for spectral feature extraction")

    # Ensure float32 for Essentia
    audio = audio.astype(np.float32)

    # Create Essentia algorithms
    windowing = es.Windowing(type='hann', size=frame_size)
    spectrum_algo = es.Spectrum(size=frame_size)
    flatness_algo = es.Flatness()
    flux_algo = es.Flux(norm='L2')  # Essentia Flux maintains internal state
    central_moments = es.CentralMoments(range=frame_size // 2 + 1)
    dist_shape = es.DistributionShape()

    # Frame the audio
    frame_generator = es.FrameGenerator(
        audio,
        frameSize=frame_size,
        hopSize=hop_size,
        startFromZero=True
    )

    # Collect per-frame values
    flatness_values: List[float] = []
    flux_values: List[float] = []
    skewness_values: List[float] = []
    kurtosis_values: List[float] = []

    for frame in frame_generator:
        # Apply windowing
        windowed = windowing(frame)

        # Compute spectrum
        spectrum = spectrum_algo(windowed)

        # Flatness
        flatness = flatness_algo(spectrum)
        flatness_values.append(flatness)

        # Flux (Essentia maintains internal state for previous spectrum)
        flux = flux_algo(spectrum)
        flux_values.append(flux)

        # Central moments for skewness and kurtosis
        moments = central_moments(spectrum)
        spread, skewness, kurtosis = dist_shape(moments)
        skewness_values.append(skewness)
        kurtosis_values.append(kurtosis)

    # Aggregate using median (more robust to outlier frames)
    results = {
        'spectral_flatness': float(np.median(flatness_values)) if flatness_values else 0.0,
        'spectral_flux': float(np.median(flux_values)) if flux_values else 0.0,
        'spectral_skewness': float(np.median(skewness_values)) if skewness_values else 0.0,
        'spectral_kurtosis': float(np.median(kurtosis_values)) if kurtosis_values else 0.0,
    }

    return results


def analyze_spectral_features_librosa(audio: np.ndarray,
                                      sample_rate: int,
                                      n_fft: int = 2048,
                                      hop_length: int = 512) -> Dict[str, float]:
    """
    Fallback: Calculate spectral features using librosa.

    Args:
        audio: Audio signal (mono)
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Hop length in samples

    Returns:
        Dictionary with spectral features
    """
    import librosa

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(
        y=audio, n_fft=n_fft, hop_length=hop_length
    )
    median_flatness = float(np.median(flatness))

    # Spectral flux
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    median_flux = float(np.median(flux))

    # Spectral moments (skewness, kurtosis)
    skewness_list = []
    kurtosis_list = []

    for frame_idx in range(S.shape[1]):
        spectrum = S[:, frame_idx]
        if np.sum(spectrum) > 0:
            spectrum = spectrum / np.sum(spectrum)
            freq_bins = np.arange(len(spectrum))
            mean = np.sum(freq_bins * spectrum)
            variance = np.sum(((freq_bins - mean) ** 2) * spectrum)
            std = np.sqrt(variance)

            if std > 0:
                skewness = np.sum(((freq_bins - mean) ** 3) * spectrum) / (std ** 3)
                kurtosis = np.sum(((freq_bins - mean) ** 4) * spectrum) / (variance ** 2)
                skewness_list.append(skewness)
                kurtosis_list.append(kurtosis)

    median_skewness = float(np.median(skewness_list)) if skewness_list else 0.0
    median_kurtosis = float(np.median(kurtosis_list)) if kurtosis_list else 0.0

    return {
        'spectral_flatness': median_flatness,
        'spectral_flux': median_flux,
        'spectral_skewness': median_skewness,
        'spectral_kurtosis': median_kurtosis,
    }


def analyze_spectral_features(audio_path: str | Path,
                              frame_size: int = 2048,
                              hop_size: int = 512,
                              audio: Optional[np.ndarray] = None,
                              sr: Optional[int] = None) -> Dict[str, float]:
    """
    Analyze spectral features for an audio file.

    Uses Essentia if available (faster), falls back to librosa.

    Args:
        audio_path: Path to audio file
        frame_size: FFT window size (default: 2048)
        hop_size: Hop size in samples (default: 512)
        audio: Pre-loaded mono audio array (skips disk read if provided)
        sr: Sample rate (required if audio is provided)

    Returns:
        Dictionary with spectral features:
        - spectral_flatness: Flatness measure (0.0-1.0)
        - spectral_flux: Rate of spectral change
        - spectral_skewness: Asymmetry of spectrum
        - spectral_kurtosis: Peakedness of spectrum
    """
    audio_path = Path(audio_path)

    if audio is None:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Analyzing spectral features: {audio_path.name}")

        # Load audio
        audio, sr = sf.read(str(audio_path))

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    else:
        logger.info(f"Analyzing spectral features: {audio_path.name} (pre-loaded)")

    logger.debug(f"Loaded audio: {len(audio)} samples @ {sr} Hz")

    # Calculate spectral features
    if ESSENTIA_AVAILABLE:
        results = analyze_spectral_features_essentia(audio, sr, frame_size, hop_size)
    else:
        logger.warning("Using librosa fallback (Essentia not available)")
        results = analyze_spectral_features_librosa(audio, sr, frame_size, hop_size)

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
                                    frame_size: int = 2048,
                                    hop_size: int = 512) -> dict:
    """
    Batch analyze spectral features for all organized folders.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing spectral data
        frame_size: FFT window size
        hop_size: Hop size in samples

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch spectral features analysis: {root_directory}")
    logger.info(f"Using {'Essentia' if ESSENTIA_AVAILABLE else 'librosa (fallback)'}")

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

        # Check if already processed - must check ALL output keys
        info_path = get_info_path(stems['full_mix'])
        from core.json_handler import should_process
        SPECTRAL_KEYS = ['spectral_flatness', 'spectral_flux', 'spectral_skewness', 'spectral_kurtosis']
        if not should_process(info_path, SPECTRAL_KEYS, overwrite):
            logger.info("Spectral data already exists. Use --overwrite to regenerate.")
            stats['skipped'] += 1
            continue

        try:
            results = analyze_spectral_features(
                stems['full_mix'],
                frame_size=frame_size,
                hop_size=hop_size
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
        '--frame-size',
        type=int,
        default=2048,
        help='FFT window size (default: 2048)'
    )

    parser.add_argument(
        '--hop-size',
        type=int,
        default=512,
        help='Hop size in samples (default: 512)'
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

            results = analyze_spectral_features(
                stems['full_mix'],
                frame_size=args.frame_size,
                hop_size=args.hop_size
            )

            # Save to .INFO
            info_path = get_info_path(path)
            safe_update(info_path, results)

            print(f"\nSpectral Features Results (using {'Essentia' if ESSENTIA_AVAILABLE else 'librosa'}):")
            print(f"  Flatness:       {results['spectral_flatness']:.3f}")
            print(f"  Flux:           {results['spectral_flux']:.3f}")
            print(f"  Skewness:       {results['spectral_skewness']:.3f}")
            print(f"  Kurtosis:       {results['spectral_kurtosis']:.3f}")

        else:
            # Single file
            results = analyze_spectral_features(
                path,
                frame_size=args.frame_size,
                hop_size=args.hop_size
            )

            print(f"\nSpectral Features Results (using {'Essentia' if ESSENTIA_AVAILABLE else 'librosa'}):")
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
