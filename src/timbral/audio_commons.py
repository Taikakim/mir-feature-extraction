"""
Audio Commons Timbral Features for MIR Project

This module extracts perceptual timbral features using the Audio Commons timbral models.
These features bridge the gap between acoustic measurements and perceptual qualities.

Dependencies:
- timbral_models
- src.core.json_handler
- src.core.file_utils
- src.core.common

Features extracted (all 0-100 scale):
- {brightness}: High-frequency content perception
- {roughness}: Harshness/beating perception
- {hardness}: Soft vs metallic perception
- {depth}: Low-frequency spaciousness
- {booming}: Low-frequency resonance (100-200 Hz)
- {reverberation}: Wet/dry balance
- {sharpness}: High-frequency harshness
- {warmth}: Mid-low frequency richness
"""

from pathlib import Path
from typing import Dict, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.json_handler import safe_update, get_info_path
from core.file_utils import get_stem_files
from core.common import clamp_feature_value

logger = logging.getLogger(__name__)

# Try to import timbral_models
try:
    import timbral_models
    TIMBRAL_MODELS_AVAILABLE = True
    logger.debug("Timbral models available")
except ImportError:
    TIMBRAL_MODELS_AVAILABLE = False
    logger.error("Timbral models not available!")


def analyze_brightness(audio_path: str | Path) -> float:
    """
    Analyze perceived brightness of audio.

    Brightness correlates with spectral centroid - higher values indicate
    more high-frequency content.

    Args:
        audio_path: Path to audio file

    Returns:
        Brightness score (0-100)

    Raises:
        ImportError: If timbral_models is not available
        Exception: If analysis fails
    """
    if not TIMBRAL_MODELS_AVAILABLE:
        raise ImportError("timbral_models is not installed")

    audio_path = str(audio_path)
    logger.info(f"Analyzing brightness: {Path(audio_path).name}")

    try:
        brightness = timbral_models.timbral_brightness(audio_path)
        brightness = clamp_feature_value('brightness', float(brightness))

        logger.debug(f"Brightness: {brightness:.1f}")
        return brightness

    except Exception as e:
        logger.error(f"Error analyzing brightness: {e}")
        raise


def analyze_roughness(audio_path: str | Path) -> float:
    """
    Analyze perceived roughness/harshness of audio.

    High roughness indicates beating, modulation, or harsh textures.

    Args:
        audio_path: Path to audio file

    Returns:
        Roughness score (0-100)
    """
    if not TIMBRAL_MODELS_AVAILABLE:
        raise ImportError("timbral_models is not installed")

    audio_path = str(audio_path)
    logger.info(f"Analyzing roughness: {Path(audio_path).name}")

    try:
        roughness = timbral_models.timbral_roughness(audio_path)
        roughness = clamp_feature_value('roughness', float(roughness))

        logger.debug(f"Roughness: {roughness:.1f}")
        return roughness

    except Exception as e:
        logger.error(f"Error analyzing roughness: {e}")
        raise


def analyze_hardness(audio_path: str | Path) -> float:
    """
    Analyze perceived hardness of audio.

    Distinguishes soft, mellow sounds from hard, metallic ones.

    Args:
        audio_path: Path to audio file

    Returns:
        Hardness score (0-100)
    """
    if not TIMBRAL_MODELS_AVAILABLE:
        raise ImportError("timbral_models is not installed")

    audio_path = str(audio_path)
    logger.info(f"Analyzing hardness: {Path(audio_path).name}")

    try:
        hardness = timbral_models.timbral_hardness(audio_path)
        hardness = clamp_feature_value('hardness', float(hardness))

        logger.debug(f"Hardness: {hardness:.1f}")
        return hardness

    except Exception as e:
        logger.error(f"Error analyzing hardness: {e}")
        raise


def analyze_depth(audio_path: str | Path) -> float:
    """
    Analyze perceived depth/spaciousness of audio.

    High depth suggests large spaces and powerful sub-bass.

    Args:
        audio_path: Path to audio file

    Returns:
        Depth score (0-100)
    """
    if not TIMBRAL_MODELS_AVAILABLE:
        raise ImportError("timbral_models is not installed")

    audio_path = str(audio_path)
    logger.info(f"Analyzing depth: {Path(audio_path).name}")

    try:
        depth = timbral_models.timbral_depth(audio_path)
        depth = clamp_feature_value('depth', float(depth))

        logger.debug(f"Depth: {depth:.1f}")
        return depth

    except Exception as e:
        logger.error(f"Error analyzing depth: {e}")
        raise


def analyze_booming(audio_path: str | Path) -> float:
    """
    Analyze perceived boominess of audio.

    Measures low-frequency resonance around 100-200 Hz.

    Args:
        audio_path: Path to audio file

    Returns:
        Booming score (0-100)
    """
    if not TIMBRAL_MODELS_AVAILABLE:
        raise ImportError("timbral_models is not installed")

    audio_path = str(audio_path)
    logger.info(f"Analyzing booming: {Path(audio_path).name}")

    try:
        booming = timbral_models.timbral_booming(audio_path)
        booming = clamp_feature_value('booming', float(booming))

        logger.debug(f"Booming: {booming:.1f}")
        return booming

    except Exception as e:
        logger.error(f"Error analyzing booming: {e}")
        raise


def analyze_reverb(audio_path: str | Path) -> float:
    """
    Analyze perceived reverberation of audio.

    Quantifies wet/dry balance and decay characteristics.

    Args:
        audio_path: Path to audio file

    Returns:
        Reverberation score (0-100)
    """
    if not TIMBRAL_MODELS_AVAILABLE:
        raise ImportError("timbral_models is not installed")

    audio_path = str(audio_path)
    logger.info(f"Analyzing reverberation: {Path(audio_path).name}")

    try:
        # timbral_reverb returns binary 0/1 by default.
        # Use dev_output=True to get (RT60, probability)
        # We use probability * 100 as the score.
        rt60, probability = timbral_models.timbral_reverb(audio_path, dev_output=True)
        
        reverb = float(probability) * 100.0
        reverb = clamp_feature_value('reverberation', reverb)

        logger.debug(f"Reverberation: {reverb:.1f} (RT60: {rt60:.2f}s)")
        return reverb

    except Exception as e:
        logger.error(f"Error analyzing reverberation: {e}")
        raise


def analyze_sharpness(audio_path: str | Path) -> float:
    """
    Analyze perceived sharpness of audio.

    Related to brightness but specifically captures fatiguing high-frequency harshness.

    Args:
        audio_path: Path to audio file

    Returns:
        Sharpness score (0-100)
    """
    if not TIMBRAL_MODELS_AVAILABLE:
        raise ImportError("timbral_models is not installed")

    audio_path = str(audio_path)
    logger.info(f"Analyzing sharpness: {Path(audio_path).name}")

    try:
        sharpness = timbral_models.timbral_sharpness(audio_path)
        sharpness = clamp_feature_value('sharpness', float(sharpness))

        logger.debug(f"Sharpness: {sharpness:.1f}")
        return sharpness

    except Exception as e:
        logger.error(f"Error analyzing sharpness: {e}")
        raise


def analyze_warmth(audio_path: str | Path) -> float:
    """
    Analyze perceived warmth of audio.

    Captures mid-low frequency richness that makes sounds feel "warm" vs "cold".

    Args:
        audio_path: Path to audio file

    Returns:
        Warmth score (0-100)
    """
    if not TIMBRAL_MODELS_AVAILABLE:
        raise ImportError("timbral_models is not installed")

    audio_path = str(audio_path)
    logger.info(f"Analyzing warmth: {Path(audio_path).name}")

    try:
        warmth = timbral_models.timbral_warmth(audio_path)
        warmth = clamp_feature_value('warmth', float(warmth))

        logger.debug(f"Warmth: {warmth:.1f}")
        return warmth

    except Exception as e:
        logger.error(f"Error analyzing warmth: {e}")
        raise


def analyze_all_timbral_features(audio_path: str | Path,
                                  features: Optional[list] = None,
                                  audio: Optional['np.ndarray'] = None,
                                  sr: Optional[int] = None) -> Dict[str, float]:
    """
    Analyze all (or selected) Audio Commons timbral features.

    When audio/sr are provided, the audio is written to /dev/shm (tmpfs RAM disk)
    once, and all 8 timbral_models calls read from RAM instead of HDD.
    This eliminates 7 redundant disk reads per analysis.

    Args:
        audio_path: Path to audio file
        features: Optional list of features to extract. If None, extracts all.
                 Valid: ['brightness', 'roughness', 'hardness', 'depth',
                        'booming', 'reverberation', 'sharpness', 'warmth']
        audio: Pre-loaded audio array (written to /dev/shm for timbral_models)
        sr: Sample rate (required if audio is provided)

    Returns:
        Dictionary with timbral features

    Raises:
        ImportError: If timbral_models is not available
    """
    if not TIMBRAL_MODELS_AVAILABLE:
        raise ImportError("timbral_models is not installed")

    audio_path = Path(audio_path)
    tmpfs_path = None

    # If pre-loaded audio provided, write to /dev/shm for timbral_models
    # (library only accepts file paths, so we use RAM disk to avoid HDD I/O)
    if audio is not None and sr is not None:
        try:
            import soundfile as sf
            import os
            tmpfs_path = Path(f'/dev/shm/mir_timbral_{os.getpid()}.wav')
            sf.write(str(tmpfs_path), audio, sr)
            analysis_path = tmpfs_path
            logger.info(f"Analyzing {audio_path.name} timbral features (via /dev/shm)")
        except Exception as e:
            logger.debug(f"/dev/shm write failed ({e}), falling back to disk path")
            analysis_path = audio_path
    else:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        analysis_path = audio_path

    # Define all available features and their analysis functions
    all_features = {
        'brightness': analyze_brightness,
        'roughness': analyze_roughness,
        'hardness': analyze_hardness,
        'depth': analyze_depth,
        'booming': analyze_booming,
        'reverberation': analyze_reverb,
        'sharpness': analyze_sharpness,
        'warmth': analyze_warmth,
    }

    # Use all features if none specified
    if features is None:
        features = list(all_features.keys())

    logger.info(f"Analyzing {len(features)} timbral features: {audio_path.name}")

    results = {}
    try:
        for feature_name in features:
            if feature_name not in all_features:
                logger.warning(f"Unknown feature: {feature_name}")
                continue

            try:
                value = all_features[feature_name](analysis_path)
                results[feature_name] = value
            except Exception as e:
                logger.error(f"Could not analyze {feature_name}: {e}")
    finally:
        # Clean up tmpfs file
        if tmpfs_path is not None and tmpfs_path.exists():
            tmpfs_path.unlink()

    logger.info(f"Extracted {len(results)}/{len(features)} timbral features")
    return results


def analyze_folder_timbral_features(audio_folder: str | Path,
                                     features: Optional[list] = None,
                                     save_to_info: bool = True) -> Dict[str, float]:
    """
    Analyze Audio Commons timbral features for an organized audio folder.

    Args:
        audio_folder: Path to organized folder
        features: Optional list of features to extract (default: all)
        save_to_info: Whether to save results to .INFO file

    Returns:
        Dictionary with timbral features

    Raises:
        FileNotFoundError: If folder or full_mix doesn't exist
    """
    audio_folder = Path(audio_folder)

    if not audio_folder.exists():
        raise FileNotFoundError(f"Folder not found: {audio_folder}")

    logger.info(f"Analyzing Audio Commons features for folder: {audio_folder.name}")

    # Find full_mix file
    stems = get_stem_files(audio_folder, include_full_mix=True)
    if 'full_mix' not in stems:
        raise FileNotFoundError(f"No full_mix file found in {audio_folder}")

    # Analyze features
    results = analyze_all_timbral_features(stems['full_mix'], features=features)

    # Save to .INFO file if requested
    if save_to_info and results:
        try:
            info_path = get_info_path(stems['full_mix'])
            safe_update(info_path, results)
            logger.info(f"Saved {len(results)} timbral features to {info_path.name}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    return results


def batch_analyze_timbral_features(root_directory: str | Path,
                                   features: Optional[list] = None,
                                   save_to_info: bool = True,
                                   overwrite: bool = False) -> Dict[str, any]:
    """
    Batch analyze Audio Commons features for all organized folders.

    Args:
        root_directory: Root directory to search
        features: Optional list of features to extract (default: all)
        save_to_info: Whether to save results to .INFO files
        overwrite: Whether to overwrite existing timbral data

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders
    from core.json_handler import read_info

    root_directory = Path(root_directory)
    logger.info(f"Starting batch Audio Commons analysis: {root_directory}")

    # Find all organized folders
    folders = find_organized_folders(root_directory)

    stats = {
        'total': len(folders),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }

    logger.info(f"Found {stats['total']} organized folders")

    # Process each folder
    for i, folder in enumerate(folders, 1):
        logger.info(f"Processing {i}/{stats['total']}: {folder.name}")

        # Check for existing data - must check ALL output keys
        stems = get_stem_files(folder, include_full_mix=True)
        if 'full_mix' in stems:
            info_path = get_info_path(stems['full_mix'])
            from core.json_handler import should_process
            TIMBRAL_KEYS = ['brightness', 'roughness', 'hardness', 'depth',
                           'booming', 'reverberation', 'sharpness', 'warmth']
            if not should_process(info_path, TIMBRAL_KEYS, overwrite):
                logger.info(f"  Timbral data already exists. Use --overwrite to regenerate.")
                stats['skipped'] += 1
                continue

        try:
            analyze_folder_timbral_features(folder, features=features, save_to_info=save_to_info)
            stats['success'] += 1

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Audio Commons Analysis Summary:")
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
        description="Extract Audio Commons timbral features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard usage (put filename first to avoid confusion):
  python src/timbral/audio_commons.py /path/to/file.flac --features brightness roughness

  # Or use '--' to separate flags from positional arguments:
  python src/timbral/audio_commons.py --features reverberation -- /path/to/file.flac

  # Batch processing:
  python src/timbral/audio_commons.py /path/to/dataset --batch
        """
    )

    parser.add_argument(
        'path',
        type=str,
        help='Path to audio file, organized folder, or root directory for batch'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process all organized folders in directory tree'
    )

    parser.add_argument(
        '--features',
        nargs='+',
        choices=['brightness', 'roughness', 'hardness', 'depth',
                 'booming', 'reverberation', 'sharpness', 'warmth'],
        help='Specific features to extract (default: all)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to .INFO file'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing timbral data'
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

    if not TIMBRAL_MODELS_AVAILABLE:
        logger.error("timbral_models is not installed")
        logger.error("Install with: pip install timbral_models")
        sys.exit(1)

    path = Path(args.path)

    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)

    try:
        if args.batch:
            # Batch processing
            stats = batch_analyze_timbral_features(
                path,
                features=args.features,
                save_to_info=not args.no_save,
                overwrite=args.overwrite
            )

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} folders failed to process")
                sys.exit(1)

        elif path.is_dir():
            # Single folder
            results = analyze_folder_timbral_features(
                path,
                features=args.features,
                save_to_info=not args.no_save
            )

            # Print results
            print("\nAudio Commons Timbral Features:")
            for key, value in sorted(results.items()):
                print(f"  {key}: {value:.1f}")

        else:
            # Single file
            results = analyze_all_timbral_features(path, features=args.features)

            print("\nAudio Commons Timbral Features:")
            for key, value in sorted(results.items()):
                print(f"  {key}: {value:.1f}")

        logger.info("Audio Commons analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
