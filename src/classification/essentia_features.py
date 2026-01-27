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
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os

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


# =============================================================================
# Genre, Mood, and Instrument Classification (Discogs-EffNet based)
# =============================================================================

def load_classification_labels() -> Dict[str, List[str]]:
    """
    Load class labels from model metadata JSON files.

    Returns:
        Dictionary with 'genre', 'mood', 'instrument' keys containing class lists
    """
    import json

    labels = {}

    # Load genre labels (400 classes)
    try:
        genre_json = get_model_path("genre_discogs400-discogs-effnet-1.json")
        with open(genre_json) as f:
            labels['genre'] = json.load(f)['classes']
    except Exception as e:
        logger.warning(f"Could not load genre labels: {e}")
        labels['genre'] = []

    # Load mood/theme labels (56 classes)
    try:
        mood_json = get_model_path("mtg_jamendo_moodtheme-discogs-effnet-1.json")
        with open(mood_json) as f:
            labels['mood'] = json.load(f)['classes']
    except Exception as e:
        logger.warning(f"Could not load mood labels: {e}")
        labels['mood'] = []

    # Load instrument labels (40 classes)
    try:
        instrument_json = get_model_path("mtg_jamendo_instrument-discogs-effnet-1.json")
        with open(instrument_json) as f:
            labels['instrument'] = json.load(f)['classes']
    except Exception as e:
        logger.warning(f"Could not load instrument labels: {e}")
        labels['instrument'] = []

    return labels


# Cache for labels (loaded once)
_CLASSIFICATION_LABELS = None

def get_classification_labels() -> Dict[str, List[str]]:
    """Get cached classification labels."""
    global _CLASSIFICATION_LABELS
    if _CLASSIFICATION_LABELS is None:
        _CLASSIFICATION_LABELS = load_classification_labels()
    return _CLASSIFICATION_LABELS


def analyze_genre_mood_instrument(audio_path: str | Path,
                                   threshold: float = 0.1,
                                   top_k: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Analyze genre, mood, and instrument using Discogs-EffNet embeddings.

    Uses the following models:
    - discogs-effnet-bs64-1.pb for embeddings
    - genre_discogs400-discogs-effnet-1.pb for genre (400 classes)
    - mtg_jamendo_moodtheme-discogs-effnet-1.pb for mood (56 classes)
    - mtg_jamendo_instrument-discogs-effnet-1.pb for instrument (40 classes)

    Args:
        audio_path: Path to audio file
        threshold: Minimum probability threshold for including predictions
        top_k: Maximum number of top predictions to return per category

    Returns:
        Dictionary with 'genre', 'mood', 'instrument' keys, each containing
        a dict of {label: probability} for predictions above threshold

    Raises:
        ImportError: If essentia-tensorflow is not available
    """
    if not ESSENTIA_TF_AVAILABLE:
        raise ImportError("Essentia-TensorFlow is not installed")

    logger.info(f"Analyzing genre/mood/instrument: {Path(audio_path).name}")

    try:
        # Load audio at 16kHz with high quality resampling
        loader = es.MonoLoader(filename=str(audio_path), sampleRate=16000, resampleQuality=4)
        audio = loader()

        # Get embeddings using Discogs-EffNet
        embedding_model_path = get_model_path("discogs-effnet-bs64-1.pb")
        embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename=embedding_model_path,
            output="PartitionedCall:1"
        )
        embeddings = embedding_model(audio)

        # Load classification models
        genre_model_path = get_model_path("genre_discogs400-discogs-effnet-1.pb")
        mood_model_path = get_model_path("mtg_jamendo_moodtheme-discogs-effnet-1.pb")
        instrument_model_path = get_model_path("mtg_jamendo_instrument-discogs-effnet-1.pb")

        genre_model = TensorflowPredict2D(
            graphFilename=genre_model_path,
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0"
        )
        mood_model = TensorflowPredict2D(graphFilename=mood_model_path)
        instrument_model = TensorflowPredict2D(graphFilename=instrument_model_path)

        # Get predictions
        genre_predictions = genre_model(embeddings)
        mood_predictions = mood_model(embeddings)
        instrument_predictions = instrument_model(embeddings)

        # Get labels
        labels = get_classification_labels()

        # Filter and format predictions
        results = {
            'essentia_genre': _filter_predictions(genre_predictions, labels['genre'], threshold, top_k),
            'essentia_mood': _filter_predictions(mood_predictions, labels['mood'], threshold, top_k),
            'essentia_instrument': _filter_predictions(instrument_predictions, labels['instrument'], threshold, top_k),
        }

        # Log summary
        logger.info(f"  Genres: {len(results['essentia_genre'])} above threshold")
        logger.info(f"  Moods: {len(results['essentia_mood'])} above threshold")
        logger.info(f"  Instruments: {len(results['essentia_instrument'])} above threshold")

        return results

    except Exception as e:
        logger.error(f"Error analyzing genre/mood/instrument: {e}")
        raise


def _filter_predictions(predictions: np.ndarray,
                        class_labels: List[str],
                        threshold: float = 0.1,
                        top_k: int = 10) -> Dict[str, float]:
    """
    Filter predictions by threshold and return top-k results.

    Args:
        predictions: Model output array (frames x classes)
        class_labels: List of class names
        threshold: Minimum probability to include
        top_k: Maximum results to return

    Returns:
        Dictionary of {label: probability} sorted by probability descending
    """
    # Average predictions across time frames
    predictions_mean = np.mean(predictions, axis=0)

    # Get sorted indices (descending)
    sorted_indices = np.argsort(predictions_mean)[::-1]

    result = {}
    count = 0

    for idx in sorted_indices:
        if count >= top_k:
            break

        prob = float(predictions_mean[idx])
        if prob < threshold:
            break

        if idx < len(class_labels):
            label = class_labels[idx]
            result[label] = round(prob, 4)
            count += 1

    return result


def format_genre_for_prompt(genre_dict: Dict[str, float], max_items: int = 5) -> str:
    """
    Format genre predictions as a text string suitable for prompts.

    Args:
        genre_dict: Dictionary of {genre: probability}
        max_items: Maximum number of genres to include

    Returns:
        Formatted string like "Electronic---Techno (0.85), Electronic---House (0.72)"
    """
    if not genre_dict:
        return ""

    # Sort by probability and take top items
    sorted_items = sorted(genre_dict.items(), key=lambda x: x[1], reverse=True)[:max_items]

    # Format as string
    parts = [f"{genre} ({prob:.2f})" for genre, prob in sorted_items]
    return ", ".join(parts)


def format_mood_for_prompt(mood_dict: Dict[str, float], max_items: int = 5) -> str:
    """
    Format mood predictions as a text string suitable for prompts.

    Args:
        mood_dict: Dictionary of {mood: probability}
        max_items: Maximum number of moods to include

    Returns:
        Formatted string like "energetic (0.82), upbeat (0.75)"
    """
    if not mood_dict:
        return ""

    sorted_items = sorted(mood_dict.items(), key=lambda x: x[1], reverse=True)[:max_items]
    parts = [f"{mood} ({prob:.2f})" for mood, prob in sorted_items]
    return ", ".join(parts)


def format_instrument_for_prompt(instrument_dict: Dict[str, float], max_items: int = 5) -> str:
    """
    Format instrument predictions as a text string suitable for prompts.

    Args:
        instrument_dict: Dictionary of {instrument: probability}
        max_items: Maximum number of instruments to include

    Returns:
        Formatted string like "synthesizer (0.91), drums (0.88)"
    """
    if not instrument_dict:
        return ""

    sorted_items = sorted(instrument_dict.items(), key=lambda x: x[1], reverse=True)[:max_items]
    parts = [f"{inst} ({prob:.2f})" for inst, prob in sorted_items]
    return ", ".join(parts)


def analyze_essentia_features(audio_path: str | Path,
                              include_voice_analysis: bool = False,
                              include_gender: bool = False,
                              include_gmi: bool = False) -> Dict[str, Any]:
    """
    Analyze Essentia high-level features for a single audio file.

    Args:
        audio_path: Path to audio file
        include_voice_analysis: Whether to analyze voice/instrumental content
        include_gender: Whether to analyze vocal gender
        include_gmi: Whether to analyze genre, mood, and instrument

    Returns:
        Dictionary with Essentia features
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Analyzing Essentia features: {audio_path.name}")

    results = {}

    # Analyze danceability
    try:
        danceability = analyze_danceability(audio_path)
        results['danceability'] = danceability
    except Exception as e:
        logger.error(f"Could not analyze danceability: {e}")

    # Analyze atonality
    try:
        atonality = analyze_atonality(audio_path)
        results['atonality'] = atonality
    except Exception as e:
        logger.error(f"Could not analyze atonality: {e}")

    # Analyze voice/instrumental if requested
    if include_voice_analysis:
        try:
            voice_results = analyze_voice_instrumental(audio_path)
            results.update(voice_results)

            # Only analyze gender if voice is detected
            if include_gender and voice_results.get('voice_probability', 0) > 0.5:
                try:
                    gender_results = analyze_vocal_gender(audio_path)
                    results.update(gender_results)
                except Exception as e:
                    logger.warning(f"Could not analyze vocal gender: {e}")
        except Exception as e:
            logger.error(f"Could not analyze voice/instrumental: {e}")

    # Analyze genre, mood, and instrument if requested
    if include_gmi:
        try:
            gmi_results = analyze_genre_mood_instrument(audio_path)
            results.update(gmi_results)
        except Exception as e:
            logger.error(f"Could not analyze genre/mood/instrument: {e}")

    return results


def analyze_folder_essentia_features(audio_folder: str | Path,
                                     save_to_info: bool = True,
                                     include_voice_analysis: bool = False,
                                     include_gender: bool = False,
                                     include_gmi: bool = False) -> Dict[str, Any]:
    """
    Analyze Essentia high-level features for an organized audio folder.

    Args:
        audio_folder: Path to organized folder
        save_to_info: Whether to save results to .INFO file
        include_voice_analysis: Whether to analyze voice/instrumental content
        include_gender: Whether to analyze vocal gender (only if voice detected)
        include_gmi: Whether to analyze genre, mood, and instrument

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

    # Analyze genre, mood, and instrument if requested
    if include_gmi:
        try:
            gmi_results = analyze_genre_mood_instrument(full_mix_path)
            results.update(gmi_results)
        except Exception as e:
            logger.error(f"Could not analyze genre/mood/instrument: {e}")

    # Save to .INFO file if requested
    if save_to_info and results:
        try:
            info_path = get_info_path(full_mix_path)
            safe_update(info_path, results)
            logger.info(f"Saved {len(results)} Essentia features to {info_path.name}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    return results


# =============================================================================
# Parallel Processing Worker Function
# =============================================================================

def _process_folder_essentia(args: Tuple[Path, bool, bool, bool, bool, bool]) -> Tuple[str, str, Optional[str]]:
    """
    Worker function for parallel Essentia processing.

    Must be at module level for pickling by ProcessPoolExecutor.

    Args:
        args: Tuple of (folder_path, save_to_info, include_voice, include_gender, include_gmi, overwrite)

    Returns:
        Tuple of (folder_name, status, error_message)
        status is one of: 'success', 'skipped', 'failed'
    """
    folder, save_to_info, include_voice, include_gender, include_gmi, overwrite = args

    try:
        # Check for existing data if not overwriting
        if not overwrite:
            stems = get_stem_files(folder, include_full_mix=True)
            if 'full_mix' in stems:
                info_path = get_info_path(stems['full_mix'])
                if info_path.exists():
                    from core.json_handler import read_info
                    existing = read_info(info_path)
                    if 'danceability' in existing:
                        return (folder.name, 'skipped', None)

        # Process the folder
        analyze_folder_essentia_features(
            folder,
            save_to_info=save_to_info,
            include_voice_analysis=include_voice,
            include_gender=include_gender,
            include_gmi=include_gmi
        )
        return (folder.name, 'success', None)

    except Exception as e:
        return (folder.name, 'failed', str(e))


def batch_analyze_essentia_features(root_directory: str | Path,
                                    save_to_info: bool = True,
                                    include_voice_analysis: bool = False,
                                    include_gender: bool = False,
                                    include_gmi: bool = False,
                                    overwrite: bool = False,
                                    workers: int = 0) -> Dict[str, any]:
    """
    Batch analyze Essentia features for all organized folders.

    Args:
        root_directory: Root directory to search
        save_to_info: Whether to save results to .INFO files
        include_voice_analysis: Whether to analyze voice/instrumental content
        include_gender: Whether to analyze vocal gender
        include_gmi: Whether to analyze genre, mood, and instrument
        overwrite: Whether to overwrite existing Essentia data
        workers: Number of parallel workers (0 = auto based on CPU count, 1 = sequential)

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
        'skipped': 0,
        'failed': 0,
        'errors': []
    }

    logger.info(f"Found {stats['total']} organized folders")

    if stats['total'] == 0:
        return stats

    # Determine number of workers
    if workers == 0:
        # Auto: use 75% of CPU cores (Essentia is CPU-bound)
        workers = max(1, int(os.cpu_count() * 0.75))
    elif workers < 0:
        workers = 1

    # Build work arguments
    work_args = [
        (folder, save_to_info, include_voice_analysis, include_gender, include_gmi, overwrite)
        for folder in folders
    ]

    # Sequential processing if workers == 1
    if workers == 1:
        logger.info("Processing sequentially (workers=1)")
        for i, args in enumerate(work_args, 1):
            folder_name, status, error = _process_folder_essentia(args)
            logger.info(f"[{i}/{stats['total']}] {folder_name}: {status}")

            if status == 'success':
                stats['success'] += 1
            elif status == 'skipped':
                stats['skipped'] += 1
            else:
                stats['failed'] += 1
                if error:
                    stats['errors'].append(f"{folder_name}: {error}")
    else:
        # Parallel processing
        logger.info(f"Processing with {workers} parallel workers")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_process_folder_essentia, args): args[0]
                for args in work_args
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(futures):
                completed += 1
                folder = futures[future]

                try:
                    folder_name, status, error = future.result()

                    if status == 'success':
                        stats['success'] += 1
                        logger.info(f"[{completed}/{stats['total']}] {folder_name}: success")
                    elif status == 'skipped':
                        stats['skipped'] += 1
                        logger.debug(f"[{completed}/{stats['total']}] {folder_name}: skipped")
                    else:
                        stats['failed'] += 1
                        if error:
                            stats['errors'].append(f"{folder_name}: {error}")
                        logger.error(f"[{completed}/{stats['total']}] {folder_name}: failed - {error}")

                except Exception as e:
                    stats['failed'] += 1
                    stats['errors'].append(f"{folder.name}: {str(e)}")
                    logger.error(f"[{completed}/{stats['total']}] {folder.name}: exception - {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Essentia Features Analysis Summary:")
    logger.info(f"  Total folders:  {stats['total']}")
    logger.info(f"  Successful:     {stats['success']}")
    logger.info(f"  Skipped:        {stats['skipped']}")
    logger.info(f"  Failed:         {stats['failed']}")
    logger.info(f"  Workers used:   {workers}")
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
        '--gmi',
        action='store_true',
        help='Include genre, mood, and instrument classification (Discogs-EffNet)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing Essentia data'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=0,
        help='Number of parallel workers (0=auto ~75%% CPU cores, 1=sequential)'
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
                include_gender=args.gender,
                include_gmi=args.gmi,
                overwrite=args.overwrite,
                workers=args.workers
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
                include_gender=args.gender,
                include_gmi=args.gmi
            )

            # Print results
            print("\nEssentia Features Analysis Results:")
            for key, value in sorted(results.items()):
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                elif isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in sorted(value.items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"    {k}: {v:.3f}")
                else:
                    print(f"  {key}: {value}")

        logger.info("Essentia features analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
