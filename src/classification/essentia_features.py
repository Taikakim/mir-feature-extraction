"""
Essentia High-Level Features for MIR Project

This module extracts high-level features using Essentia TensorFlow models:
- Danceability
- Atonality
- Genre (saved to text prompts)
- Mood (saved to text prompts)
- Instrumentation (saved to text prompts)

Dependencies:
- essentia.standard
- essentia-tensorflow (TensorflowPredict2D)
- numpy
- src.core.json_handler
- src.core.file_utils
- src.core.common

Features extracted:
- {danceability}: 0-1 float
- {atonality}: 0-1 float
- Genre, mood, instrumentation saved as text for prompts
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.json_handler import safe_update, get_info_path
from core.file_utils import get_stem_files
from core.common import clamp_feature_value

logger = logging.getLogger(__name__)

# Try to import essentia
try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    logger.error("Essentia not available!")

# Try to import essentia-tensorflow
try:
    from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs, TensorflowPredictVGGish
    ESSENTIA_TF_AVAILABLE = True
except ImportError:
    ESSENTIA_TF_AVAILABLE = False
    logger.warning("Essentia-TensorFlow not available - high-level features will not work")


def get_model_path(model_filename: str) -> str:
    """
    Find the path to an Essentia model file.

    Searches in the following order:
    1. ESSENTIA_MODELS_DIR environment variable
    2. ~/Projects/mir/models/essentia
    3. Current directory

    Args:
        model_filename: Name of the model file (e.g., "danceability-vggish-audioset-1.pb")

    Returns:
        Full path to the model file

    Raises:
        FileNotFoundError: If model file cannot be found
    """
    import os

    # Check environment variable
    env_dir = os.getenv('ESSENTIA_MODELS_DIR')
    if env_dir:
        model_path = Path(env_dir) / model_filename
        if model_path.exists():
            return str(model_path)

    # Check default project location
    project_models_dir = Path.home() / "Projects" / "mir" / "models" / "essentia"
    model_path = project_models_dir / model_filename
    if model_path.exists():
        return str(model_path)

    # Check current directory
    model_path = Path(model_filename)
    if model_path.exists():
        return str(model_path)

    # Not found
    raise FileNotFoundError(
        f"Model file not found: {model_filename}\n"
        f"Searched in:\n"
        f"  - ESSENTIA_MODELS_DIR: {env_dir or 'not set'}\n"
        f"  - {project_models_dir}\n"
        f"  - Current directory\n"
        f"Please run: scripts/download_essentia_models.sh"
    )


def load_audio_essentia(audio_path: str | Path) -> Tuple[np.ndarray, int]:
    """
    Load audio file using Essentia.

    Args:
        audio_path: Path to audio file

    Returns:
        Tuple of (audio_samples, sample_rate)

    Raises:
        ImportError: If essentia is not available
        Exception: If audio cannot be loaded
    """
    if not ESSENTIA_AVAILABLE:
        raise ImportError("Essentia is not installed")

    audio_path = str(audio_path)

    try:
        # Load audio with Essentia
        loader = es.MonoLoader(filename=audio_path, sampleRate=16000)
        audio = loader()

        return audio, 16000
    except Exception as e:
        logger.error(f"Error loading audio with Essentia: {e}")
        raise


def analyze_danceability(audio_path: str | Path) -> float:
    """
    Analyze danceability using Essentia's pre-trained VGGish model.

    Based on the original Essentia code for danceability classification.

    Args:
        audio_path: Path to audio file

    Returns:
        Danceability score (0-1)

    Raises:
        ImportError: If essentia-tensorflow is not available
        Exception: If analysis fails
    """
    if not ESSENTIA_TF_AVAILABLE:
        raise ImportError("Essentia-TensorFlow is not installed")

    logger.info(f"Analyzing danceability: {Path(audio_path).name}")

    try:
        # Load audio with MonoLoader at 16000 Hz (as per original code)
        loader = es.MonoLoader(filename=str(audio_path), sampleRate=16000)
        audio = loader()

        # Use VGGish-based danceability model
        model_path = get_model_path("danceability-vggish-audioset-1.pb")
        model = TensorflowPredictVGGish(
            graphFilename=model_path
        )

        # Get activations from model
        activations = model(audio)

        # Extract danceability score (mean of first column)
        # This is the exact method from the original code
        danceability_score = float(activations.mean(axis=0)[0])

        # Clamp to valid range
        danceability_score = clamp_feature_value('danceability', danceability_score)

        logger.info(f"Danceability: {danceability_score:.3f}")
        return danceability_score

    except Exception as e:
        logger.error(f"Error analyzing danceability: {e}")
        # Try alternative approach using rhythm features
        logger.warning("Attempting fallback danceability estimation...")
        return estimate_danceability_fallback(audio_path)


def estimate_danceability_fallback(audio_path: str | Path) -> float:
    """
    Estimate danceability using rhythm features (fallback method).

    Args:
        audio_path: Path to audio file

    Returns:
        Estimated danceability score (0-1)
    """
    try:
        audio, sr = load_audio_essentia(audio_path)

        # Extract rhythm features
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

        # Estimate danceability from rhythm features
        # Higher BPM in dance range (90-140) -> higher danceability
        # More regular beats -> higher danceability

        bpm_score = 0.0
        if 90 <= bpm <= 140:
            # Optimal dance range
            bpm_score = 1.0
        elif 70 <= bpm < 90 or 140 < bpm <= 180:
            # Moderate dance range
            bpm_score = 0.7
        else:
            bpm_score = 0.3

        # Regularity from beat intervals
        if len(beats_intervals) > 1:
            regularity = 1.0 - min(1.0, np.std(beats_intervals) / np.mean(beats_intervals))
        else:
            regularity = 0.5

        # Combine scores
        danceability = 0.6 * bpm_score + 0.4 * regularity
        danceability = clamp_feature_value('danceability', danceability)

        logger.info(f"Fallback danceability estimate: {danceability:.3f}")
        return danceability

    except Exception as e:
        logger.error(f"Fallback danceability estimation failed: {e}")
        return 0.5  # Neutral default


def analyze_atonality(audio_path: str | Path) -> float:
    """
    Analyze atonality using Essentia's pre-trained VGGish tonality model.

    Based on the original Essentia code for tonality classification.

    Args:
        audio_path: Path to audio file

    Returns:
        Atonality score (0-1, higher = more atonal)

    Raises:
        ImportError: If essentia-tensorflow is not available
        Exception: If analysis fails
    """
    if not ESSENTIA_TF_AVAILABLE:
        raise ImportError("Essentia-TensorFlow is not installed")

    logger.info(f"Analyzing atonality: {Path(audio_path).name}")

    try:
        # Load audio with MonoLoader at 16000 Hz (as per original code)
        loader = es.MonoLoader(filename=str(audio_path), sampleRate=16000)
        audio = loader()

        # Use VGGish-based tonality model
        model_path = get_model_path("tonal_atonal-vggish-audioset-1.pb")
        model = TensorflowPredictVGGish(
            graphFilename=model_path,
            output="model/Sigmoid"
        )

        # Get activations from model
        activations = model(audio)

        # Get mean predictions across time frames
        predictions_mean = np.mean(activations, axis=0)

        # Classes are ["tonal", "atonal"]
        # We want the atonal probability (index 1)
        tonal_prob = float(predictions_mean[0])
        atonal_prob = float(predictions_mean[1])

        # Use atonal probability as the atonality score
        atonality = atonal_prob

        # Clamp to valid range
        atonality = clamp_feature_value('atonality', atonality)

        logger.info(f"Atonality: {atonality:.3f} (tonal: {tonal_prob:.3f}, atonal: {atonal_prob:.3f})")
        return atonality

    except Exception as e:
        logger.error(f"Error analyzing atonality with VGGish model: {e}")
        # Try fallback approach using key detection
        logger.warning("Attempting fallback atonality estimation...")
        return estimate_atonality_fallback(audio_path)


def estimate_atonality_fallback(audio_path: str | Path) -> float:
    """
    Estimate atonality using key detection (fallback method).

    Args:
        audio_path: Path to audio file

    Returns:
        Estimated atonality score (0-1)
    """
    if not ESSENTIA_AVAILABLE:
        return 0.5  # Neutral default

    try:
        # Load audio
        loader = es.MonoLoader(filename=str(audio_path), sampleRate=16000)
        audio = loader()

        # Extract key and key strength
        key_extractor = es.KeyExtractor()
        key, scale, key_strength = key_extractor(audio)

        # Atonality is inverse of key strength
        # Low key strength = high atonality
        atonality = 1.0 - key_strength

        atonality = clamp_feature_value('atonality', atonality)

        logger.info(f"Fallback atonality: {atonality:.3f} (key: {key} {scale}, strength: {key_strength:.3f})")
        return atonality

    except Exception as e:
        logger.error(f"Fallback atonality estimation failed: {e}")
        return 0.5  # Neutral default


def analyze_voice_instrumental(audio_path: str | Path) -> Dict[str, float]:
    """
    Analyze voice vs instrumental content using Essentia's VGGish model.

    Based on the original Essentia code for voice/instrumental classification.

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with:
            - 'voice_probability': Probability of voice presence (0-1)
            - 'instrumental_probability': Probability of instrumental content (0-1)

    Raises:
        ImportError: If essentia-tensorflow is not available
    """
    if not ESSENTIA_TF_AVAILABLE:
        raise ImportError("Essentia-TensorFlow is not installed")

    logger.info(f"Analyzing voice/instrumental: {Path(audio_path).name}")

    try:
        # Load audio with MonoLoader at 16000 Hz
        loader = es.MonoLoader(filename=str(audio_path), sampleRate=16000)
        audio = loader()

        # Use VGGish-based voice/instrumental model
        model_path = get_model_path("voice_instrumental-vggish-audioset-1.pb")
        model = TensorflowPredictVGGish(
            graphFilename=model_path,
            output="model/Sigmoid"
        )

        # Get activations from model
        activations = model(audio)

        # Get mean predictions across time frames
        predictions_mean = np.mean(activations, axis=0)

        # Classes are ["instrumental", "voice"]
        instrumental_prob = float(predictions_mean[0])
        voice_prob = float(predictions_mean[1])

        logger.info(f"Voice: {voice_prob:.3f}, Instrumental: {instrumental_prob:.3f}")

        return {
            'voice_probability': voice_prob,
            'instrumental_probability': instrumental_prob
        }

    except Exception as e:
        logger.error(f"Error analyzing voice/instrumental: {e}")
        raise


def analyze_vocal_gender(audio_path: str | Path) -> Dict[str, float]:
    """
    Analyze vocal gender using Essentia's VGGish model.

    Based on the original Essentia code for gender classification.
    Note: Only meaningful for audio with vocals present.

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with:
            - 'female_probability': Probability of female vocals (0-1)
            - 'male_probability': Probability of male vocals (0-1)

    Raises:
        ImportError: If essentia-tensorflow is not available
    """
    if not ESSENTIA_TF_AVAILABLE:
        raise ImportError("Essentia-TensorFlow is not installed")

    logger.info(f"Analyzing vocal gender: {Path(audio_path).name}")

    try:
        # Load audio with MonoLoader at 16000 Hz
        loader = es.MonoLoader(filename=str(audio_path), sampleRate=16000)
        audio = loader()

        # Use VGGish-based gender model
        model_path = get_model_path("gender-vggish-audioset-1.pb")
        model = TensorflowPredictVGGish(
            graphFilename=model_path,
            output="model/Sigmoid"
        )

        # Get activations from model
        activations = model(audio)

        # Get mean predictions across time frames
        predictions_mean = np.mean(activations, axis=0)

        # Classes are ["female", "male"]
        female_prob = float(predictions_mean[0])
        male_prob = float(predictions_mean[1])

        logger.info(f"Female: {female_prob:.3f}, Male: {male_prob:.3f}")

        return {
            'female_probability': female_prob,
            'male_probability': male_prob
        }

    except Exception as e:
        logger.error(f"Error analyzing vocal gender: {e}")
        raise


def analyze_folder_essentia_features(audio_folder: str | Path,
                                     save_to_info: bool = True,
                                     include_voice_analysis: bool = False,
                                     include_gender: bool = False) -> Dict[str, float]:
    """
    Analyze Essentia high-level features for an organized audio folder.

    Args:
        audio_folder: Path to organized folder
        save_to_info: Whether to save results to .INFO file
        include_voice_analysis: Whether to analyze voice/instrumental content
        include_gender: Whether to analyze vocal gender (only if voice detected)

    Returns:
        Dictionary with Essentia features

    Raises:
        FileNotFoundError: If folder or full_mix doesn't exist
    """
    audio_folder = Path(audio_folder)

    if not audio_folder.exists():
        raise FileNotFoundError(f"Folder not found: {audio_folder}")

    logger.info(f"Analyzing Essentia features for folder: {audio_folder.name}")

    # Find full_mix file
    stems = get_stem_files(audio_folder, include_full_mix=True)
    if 'full_mix' not in stems:
        raise FileNotFoundError(f"No full_mix file found in {audio_folder}")

    full_mix_path = stems['full_mix']
    results = {}

    # Analyze danceability
    try:
        danceability = analyze_danceability(full_mix_path)
        results['danceability'] = danceability
    except Exception as e:
        logger.error(f"Could not analyze danceability: {e}")

    # Analyze atonality
    try:
        atonality = analyze_atonality(full_mix_path)
        results['atonality'] = atonality
    except Exception as e:
        logger.error(f"Could not analyze atonality: {e}")

    # Analyze voice/instrumental if requested
    if include_voice_analysis:
        try:
            voice_results = analyze_voice_instrumental(full_mix_path)
            results.update(voice_results)

            # Only analyze gender if voice is detected (threshold: > 0.5)
            if include_gender and voice_results.get('voice_probability', 0) > 0.5:
                try:
                    gender_results = analyze_vocal_gender(full_mix_path)
                    results.update(gender_results)
                except Exception as e:
                    logger.warning(f"Could not analyze vocal gender: {e}")

        except Exception as e:
            logger.error(f"Could not analyze voice/instrumental: {e}")

    # Save to .INFO file if requested
    if save_to_info and results:
        try:
            info_path = get_info_path(full_mix_path)
            safe_update(info_path, results)
            logger.info(f"Saved {len(results)} Essentia features to {info_path.name}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    return results


def batch_analyze_essentia_features(root_directory: str | Path,
                                    save_to_info: bool = True,
                                    include_voice_analysis: bool = False,
                                    include_gender: bool = False) -> Dict[str, any]:
    """
    Batch analyze Essentia features for all organized folders.

    Args:
        root_directory: Root directory to search
        save_to_info: Whether to save results to .INFO files
        include_voice_analysis: Whether to analyze voice/instrumental content
        include_gender: Whether to analyze vocal gender

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch Essentia features analysis: {root_directory}")

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
            analyze_folder_essentia_features(
                folder,
                save_to_info=save_to_info,
                include_voice_analysis=include_voice_analysis,
                include_gender=include_gender
            )
            stats['success'] += 1

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Essentia Features Analysis Summary:")
    logger.info(f"  Total folders:  {stats['total']}")
    logger.info(f"  Successful:     {stats['success']}")
    logger.info(f"  Failed:         {stats['failed']}")
    logger.info("=" * 60)

    return stats


# Command-line interface
if __name__ == "__main__":
    import argparse
    from core.common import setup_logging

    parser = argparse.ArgumentParser(
        description="Extract Essentia high-level features (danceability, atonality)"
    )

    parser.add_argument(
        'path',
        type=str,
        help='Path to organized folder or root directory for batch processing'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process all organized folders in directory tree'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to .INFO file'
    )

    parser.add_argument(
        '--voice',
        action='store_true',
        help='Include voice/instrumental analysis'
    )

    parser.add_argument(
        '--gender',
        action='store_true',
        help='Include vocal gender analysis (only if voice detected)'
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

    if not ESSENTIA_AVAILABLE:
        logger.error("Essentia is not installed. Install with: pip install essentia")
        sys.exit(1)

    path = Path(args.path)

    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)

    try:
        if args.batch:
            # Batch processing
            stats = batch_analyze_essentia_features(
                path,
                save_to_info=not args.no_save,
                include_voice_analysis=args.voice,
                include_gender=args.gender
            )

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} folders failed to process")
                sys.exit(1)

        else:
            # Single folder
            results = analyze_folder_essentia_features(
                path,
                save_to_info=not args.no_save,
                include_voice_analysis=args.voice,
                include_gender=args.gender
            )

            # Print results
            print("\nEssentia Features Analysis Results:")
            for key, value in sorted(results.items()):
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

        logger.info("Essentia features analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
