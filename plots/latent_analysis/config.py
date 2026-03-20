# plots/latent_analysis/config.py
"""Single source of truth for all latent analysis settings."""
from pathlib import Path

# --- Paths ---
LATENT_DIR = Path("/run/media/kim/Lehto/goa-small")
INFO_DIR   = Path("/run/media/kim/Mantu/ai-music/Goa_Separated_crops")
DATA_DIR   = Path(__file__).parent / "data"
POSTER_DIR = DATA_DIR / "posters"

# --- Latent geometry ---
LATENT_DIM    = 64
LATENT_FRAMES = 256          # T dimension of each [64, 256] latent file
SAMPLE_RATE   = 44100
HOP_LENGTH    = 2048          # VAE downsampling_ratio = hop size
N_FFT         = 4096          # 50% overlap with Hann window (2× hop)
# With center=True: n_frames = 1 + floor(524288/2048) = 257 → slice [:256]
# center=False gives 255 frames (window truncation); n_fft=hop=2048 gives 256
# but 0% overlap attenuates frame-boundary samples — bad for transient detection.

# --- Temporal analysis subsample ---
N_TEMPORAL_CROPS = 2000
RANDOM_SEED      = 42

# --- Quality filter ---
# Drop crops where valid (non-padded) frames < 75% of LATENT_FRAMES.
# Checked via padding_mask in the companion .json file.
MIN_VALID_FRACTION = 0.75

# --- Statistics ---
EFFECT_WEAK     = 0.10
EFFECT_MODERATE = 0.20
EFFECT_STRONG   = 0.30
POSTER_CLAMP    = 0.35   # colour scale saturation
FDR_Q           = 0.05

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
    "Aesthetics":["content_enjoyment", "content_usefulness",
                  "production_complexity", "production_quality"],
}

# Raw INFO keys that need encoding before analysis
RAW_KEYS_TO_ENCODE = {
    "tonic":       "circular_12",    # → tonic_sin, tonic_cos
    "tonic_scale": "binary_minor",   # → tonic_minor (1=minor, 0=major)
    "bpm_madmom":  "bpm_octave_norm",# → bpm_madmom_norm (aligned to bpm_essentia)
}

# Feature names used for temporal (frame-level) analysis
TEMPORAL_FEATURE_NAMES = (
    ["rms_broadband"] +
    ["rms_bass", "rms_body", "rms_mid", "rms_air"] +
    ["spectral_flatness_t", "spectral_flux_t", "spectral_centroid_t",
     "spectral_skewness_t", "spectral_kurtosis_t"] +
    [f"chroma_{i}" for i in range(12)] +
    ["onset_strength"]
)
