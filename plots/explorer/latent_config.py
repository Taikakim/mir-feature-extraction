"""Constants extracted from the old plots/latent_analysis/config.py.

The latent_analysis directory was quarantined; these values are now
the authoritative copy for the explorer's Analysis tab.
"""
from pathlib import Path

# NPZ data files live in the quarantine (already computed, no need to regenerate)
DATA_DIR = Path(__file__).parent.parent.parent / "_quarantine" / "plots" / "latent_analysis" / "data"

# --- Latent geometry ---
LATENT_DIM   = 64
POSTER_CLAMP = 0.35

# --- Feature groups (used for UI filtering) ---
FEATURE_GROUPS = {
    "Rhythm":    ["bpm", "bpm_essentia", "bpm_madmom_norm", "syncopation",
                  "on_beat_ratio", "rhythmic_complexity", "rhythmic_evenness"],
    "Timbral":   ["brightness", "roughness", "hardness", "depth",
                  "booming", "reverberation", "sharpness", "warmth"],
    "Spectral":  ["spectral_flatness", "spectral_flux", "spectral_skewness",
                  "spectral_kurtosis", "saturation_ratio", "saturation_count"],
    "Loudness":  ["lufs", "lra", "lufs_drums", "lufs_bass", "lufs_other",
                  "lufs_vocals", "rms_energy_bass", "rms_energy_body",
                  "rms_energy_mid", "rms_energy_air"],
    "Harmonic":  (
        [f"hpcp_{i}" for i in range(12)] +
        [f"tiv_{i}"  for i in range(12)] +
        ["tonic_sin", "tonic_cos", "tonic_minor", "tonic_strength",
         "harmonic_movement_bass", "harmonic_movement_other", "atonality"]
    ),
    "Voice":     ["voice_probability", "instrumental_probability",
                  "female_probability", "male_probability"],
    "Aesthetics": ["content_enjoyment", "content_usefulness",
                   "production_complexity", "production_quality"],
}

# Feature names stored in TimeseriesDB (frame-level arrays)
TEMPORAL_FEATURE_NAMES = (
    ["rms_energy_bass_ts", "rms_energy_body_ts", "rms_energy_mid_ts", "rms_energy_air_ts"] +
    ["spectral_flatness_ts", "spectral_flux_ts", "spectral_skewness_ts", "spectral_kurtosis_ts"] +
    ["beat_activations_ts", "downbeat_activations_ts", "onsets_activations_ts"] +
    [f"hpcp_ts_{i}" for i in range(12)] +
    ["tonic_ts", "tonic_strength_ts"]
)
