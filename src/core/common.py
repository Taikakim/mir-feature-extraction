"""
Common Constants and Utilities for MIR Project

This module provides shared constants, configuration, and utility functions
used across the MIR pipeline.

Dependencies:
- None (standard library only)

Constants:
- Audio file extensions
- Stem names
- Feature value ranges
- Conditioner configurations
"""

from pathlib import Path
from typing import Dict, Any
import logging

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO

# Audio file extensions
AUDIO_EXTENSIONS = {'.flac', '.wav', '.mp3', '.ogg', '.m4a', '.aiff', '.aif'}

# Standard file naming
FULL_MIX_NAME = 'full_mix'
INFO_EXT = '.INFO'
MIR_EXT = '.MIR'
BEATS_GRID_EXT = '.BEATS_GRID'
ONSETS_GRID_EXT = '.ONSETS_GRID'
CHROMA_EXT = '.CHROMA'

# Demucs stem names
DEMUCS_STEMS = ['drums', 'bass', 'other', 'vocals']

# DrumSep stem names (approximate - will depend on model)
DRUMSEP_STEMS = ['kick', 'snare', 'cymbals', 'toms', 'percussion']

# Feature value ranges for conditioners
# Format: 'feature_name': {'min': float, 'max': float, 'type': 'number'/'int'}
FEATURE_RANGES: Dict[str, Dict[str, Any]] = {
    # Loudness features
    'lufs': {'min': -40.0, 'max': 0.0, 'type': 'number'},
    'lra': {'min': 0.0, 'max': 25.0, 'type': 'number'},

    # Per-stem loudness features
    'lufs_bass': {'min': -70.0, 'max': 0.0, 'type': 'number'},
    'lufs_drums': {'min': -70.0, 'max': 0.0, 'type': 'number'},
    'lufs_other': {'min': -70.0, 'max': 0.0, 'type': 'number'},
    'lufs_vocals': {'min': -70.0, 'max': 0.0, 'type': 'number'},
    'lra_bass': {'min': 0.0, 'max': 25.0, 'type': 'number'},
    'lra_drums': {'min': 0.0, 'max': 25.0, 'type': 'number'},
    'lra_other': {'min': 0.0, 'max': 25.0, 'type': 'number'},
    'lra_vocals': {'min': 0.0, 'max': 25.0, 'type': 'number'},

    # BPM features
    'bpm': {'min': 40.0, 'max': 300.0, 'type': 'number'},
    'bpm_is_defined': {'min': 0, 'max': 1, 'type': 'int'},
    'beat_count': {'min': 0, 'max': 1000, 'type': 'int'},
    'beat_regularity': {'min': 0.0, 'max': 1.0, 'type': 'number'},

    # Onset features
    'onset_count': {'min': 0, 'max': 10000, 'type': 'int'},
    'onset_density': {'min': 0.0, 'max': 50.0, 'type': 'number'},
    'onset_strength_mean': {'min': 0.0, 'max': 10.0, 'type': 'number'},
    'onset_strength_std': {'min': 0.0, 'max': 10.0, 'type': 'number'},

    # Syncopation features
    'syncopation': {'min': 0.0, 'max': 1.0, 'type': 'number'},
    'on_beat_ratio': {'min': 0.0, 'max': 1.0, 'type': 'number'},

    # Rhythmic complexity features
    'rhythmic_complexity': {'min': 0.0, 'max': 1.0, 'type': 'number'},
    'rhythmic_evenness': {'min': 0.0, 'max': 1.0, 'type': 'number'},
    'ioi_mean': {'min': 0.0, 'max': 10.0, 'type': 'number'},
    'ioi_std': {'min': 0.0, 'max': 10.0, 'type': 'number'},

    # Position
    'position_in_file': {'min': 0.0, 'max': 1.0, 'type': 'number'},

    # Audio Commons timbral features (0-100 scale)
    'brightness': {'min': 0.0, 'max': 100.0, 'type': 'number'},
    'roughness': {'min': 0.0, 'max': 100.0, 'type': 'number'},
    'hardness': {'min': 0.0, 'max': 100.0, 'type': 'number'},
    'depth': {'min': 0.0, 'max': 100.0, 'type': 'number'},
    'booming': {'min': 0.0, 'max': 100.0, 'type': 'number'},
    'reverberation': {'min': 0.0, 'max': 100.0, 'type': 'number'},
    'sharpness': {'min': 0.0, 'max': 100.0, 'type': 'number'},
    'warmth': {'min': 0.0, 'max': 100.0, 'type': 'number'},

    # Spectral features
    'spectral_flatness': {'min': 0.0, 'max': 1.0, 'type': 'number'},
    'spectral_flux': {'min': 0.0, 'max': 3.0, 'type': 'number'},
    'spectral_skewness': {'min': -3.0, 'max': 3.0, 'type': 'number'},
    'spectral_kurtosis': {'min': 0.0, 'max': 10.0, 'type': 'number'},

    # Multiband RMS (in dB)
    'rms_energy_bass': {'min': -60.0, 'max': 0.0, 'type': 'number'},
    'rms_energy_body': {'min': -60.0, 'max': 0.0, 'type': 'number'},
    'rms_energy_mid': {'min': -60.0, 'max': 0.0, 'type': 'number'},
    'rms_energy_air': {'min': -60.0, 'max': 0.0, 'type': 'number'},

    # Chroma features (12 semitones, 0-1 weights)
    'chroma_0': {'min': 0.0, 'max': 1.0, 'type': 'number'},  # C
    'chroma_1': {'min': 0.0, 'max': 1.0, 'type': 'number'},  # C#
    'chroma_2': {'min': 0.0, 'max': 1.0, 'type': 'number'},  # D
    'chroma_3': {'min': 0.0, 'max': 1.0, 'type': 'number'},  # D#
    'chroma_4': {'min': 0.0, 'max': 1.0, 'type': 'number'},  # E
    'chroma_5': {'min': 0.0, 'max': 1.0, 'type': 'number'},  # F
    'chroma_6': {'min': 0.0, 'max': 1.0, 'type': 'number'},  # F#
    'chroma_7': {'min': 0.0, 'max': 1.0, 'type': 'number'},  # G
    'chroma_8': {'min': 0.0, 'max': 1.0, 'type': 'number'},  # G#
    'chroma_9': {'min': 0.0, 'max': 1.0, 'type': 'number'},  # A
    'chroma_10': {'min': 0.0, 'max': 1.0, 'type': 'number'},  # A#
    'chroma_11': {'min': 0.0, 'max': 1.0, 'type': 'number'},  # B

    # Harmonic features
    'harmonic_movement_bass': {'min': 0.0, 'max': 1.0, 'type': 'number'},
    'harmonic_movement_other': {'min': 0.0, 'max': 1.0, 'type': 'number'},
    'harmonic_variance_bass': {'min': 0.0, 'max': 1.0, 'type': 'number'},
    'harmonic_variance_other': {'min': 0.0, 'max': 1.0, 'type': 'number'},

    # Rhythmic features (per-stem)
    'onset_density_average_bass': {'min': 0.0, 'max': 50.0, 'type': 'number'},
    'onset_density_average_drums': {'min': 0.0, 'max': 50.0, 'type': 'number'},
    'onset_density_average_other': {'min': 0.0, 'max': 50.0, 'type': 'number'},
    'onset_density_variance_bass': {'min': 0.0, 'max': 10.0, 'type': 'number'},
    'onset_density_variance_drums': {'min': 0.0, 'max': 10.0, 'type': 'number'},
    'onset_density_variance_other': {'min': 0.0, 'max': 10.0, 'type': 'number'},

    'syncopation_bass': {'min': 0.0, 'max': 100.0, 'type': 'number'},
    'syncopation_drums': {'min': 0.0, 'max': 100.0, 'type': 'number'},
    'syncopation_other': {'min': 0.0, 'max': 100.0, 'type': 'number'},

    'rhythmic_complexity_bass': {'min': 0.0, 'max': 5.0, 'type': 'number'},
    'rhythmic_complexity_drums': {'min': 0.0, 'max': 5.0, 'type': 'number'},
    'rhythmic_complexity_other': {'min': 0.0, 'max': 5.0, 'type': 'number'},

    'rhythmic_evenness_bass': {'min': 0.0, 'max': 1.0, 'type': 'number'},
    'rhythmic_evenness_drums': {'min': 0.0, 'max': 1.0, 'type': 'number'},
    'rhythmic_evenness_other': {'min': 0.0, 'max': 1.0, 'type': 'number'},

    # Essentia high-level features
    'danceability': {'min': 0.0, 'max': 1.0, 'type': 'number'},
    'atonality': {'min': 0.0, 'max': 1.0, 'type': 'number'},

    # AudioBox aesthetics (1-10 scale)
    'content_enjoyment': {'min': 1.0, 'max': 10.0, 'type': 'number'},
    'content_usefulness': {'min': 1.0, 'max': 10.0, 'type': 'number'},
    'production_complexity': {'min': 1.0, 'max': 10.0, 'type': 'number'},
    'production_quality': {'min': 1.0, 'max': 10.0, 'type': 'number'},
}

# Frequency band definitions for multiband RMS
FREQUENCY_BANDS = {
    'bass': (20, 120),      # 20-120 Hz
    'body': (120, 600),     # 120-600 Hz
    'mid': (600, 2500),     # 600-2500 Hz
    'air': (2500, 22000),   # 2500-22000 Hz
}

# Beat tracking configuration
BEAT_TRACKING_CONFIG = {
    'beat_count_threshold': 15,      # Minimum beats for BPM to be considered valid
    'regularity_threshold': 0.1,     # Maximum std dev of beat intervals (seconds)
    'default_bpm': 120,              # Default BPM when undefined (TODO: decide on 120 vs 0)
}

# Chroma configuration
CHROMA_CONFIG = {
    'n_chroma': 12,                  # 12 semitones
    'interval_beats': 32,            # Calculate at 32nd note intervals
    'fallback_interval_ms': 100,     # Fallback to 100ms if no BPM
}

# Demucs configuration
DEMUCS_CONFIG = {
    'model': 'htdemucs_ft',          # Demucs HT v4 fine-tuned (4x slower, better quality)
    'shifts': 1,
    'filetype': 'flac',
    'jobs': 4,                       # Concurrent jobs
}

# Spectral analysis configuration
SPECTRAL_CONFIG = {
    'frame_size': 2048,
    'hop_size': 512,
    'window': 'hann',
    'aggregation': 'median',         # Use median instead of mean
}


def setup_logging(level: int = LOG_LEVEL, log_file: str = None) -> None:
    """
    Set up logging configuration for the project.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=handlers
    )


def get_feature_range(feature_name: str) -> Dict[str, Any]:
    """
    Get the value range configuration for a feature.

    Args:
        feature_name: Name of the feature

    Returns:
        Dictionary with min, max, and type information

    Raises:
        KeyError: If feature name is not found
    """
    if feature_name not in FEATURE_RANGES:
        raise KeyError(f"Unknown feature: {feature_name}")

    return FEATURE_RANGES[feature_name].copy()


def clamp_feature_value(feature_name: str, value: float) -> float:
    """
    Clamp a feature value to its valid range.

    Args:
        feature_name: Name of the feature
        value: Value to clamp

    Returns:
        Clamped value within the valid range
    """
    range_info = get_feature_range(feature_name)
    min_val = range_info['min']
    max_val = range_info['max']

    return max(min_val, min(max_val, value))


# Example usage
if __name__ == "__main__":
    # Set up logging
    setup_logging()

    # Test feature ranges
    print("Sample feature ranges:")
    for feature in ['lufs', 'bpm', 'brightness', 'danceability']:
        range_info = get_feature_range(feature)
        print(f"  {feature}: {range_info}")

    # Test clamping
    print("\nTesting clamping:")
    test_value = 150
    clamped = clamp_feature_value('bpm', test_value)
    print(f"  BPM {test_value} -> {clamped} (max is 300)")

    test_value = -100
    clamped = clamp_feature_value('lufs', test_value)
    print(f"  LUFS {test_value} -> {clamped} (min is -40)")
