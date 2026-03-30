#!/usr/bin/env python3
"""
Build all dataset statistics for Feature Explorer.

Replaces generate_explorer_data.py.  Three stages:
  1. Scalar pass   — scan .INFO files, average per track
  2. Timeseries pass — query TimeseriesDB, compute shape vectors + mini curves
  3. Similarity pass — cosine similarity, emit TS_NEIGHBORS

Usage:
    python plots/build_dataset_stats.py --source /path/to/crops
    python plots/build_dataset_stats.py --source /path/to/crops --skip-timeseries
    python plots/build_dataset_stats.py --source /path/to/crops --skip-scalars --skip-curves
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.spatial.distance

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PLOTS_DIR   = Path(__file__).resolve().parent
REPO_ROOT   = PLOTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.timeseries_db import TimeseriesDB  # noqa: E402 (after sys.path setup)

DEFAULT_SRC    = Path("/run/media/kim/Mantu/ai-music/Goa_Separated_crops")
DEFAULT_OUTDIR = Path("/run/media/kim/Lehto")
DEFAULT_DB     = REPO_ROOT / "data" / "timeseries.db"

# ---------------------------------------------------------------------------
# Timeseries field lists
# ---------------------------------------------------------------------------
# 1D fields used for display curves and shape scalars (9 total)
TS_1D_FIELDS = [
    "rms_energy_bass_ts",
    "rms_energy_body_ts",
    "rms_energy_mid_ts",
    "rms_energy_air_ts",
    "spectral_flatness_ts",
    "spectral_flux_ts",
    "beat_activations_ts",
    "onsets_activations_ts",
    "tonic_strength_ts",
]
# Embedding layout: [mean,std]×9 + hpcp_raw×12 + hpcp_rot×12 + sin/cos = 44
EMBEDDING_DIMS = 44

# ---------------------------------------------------------------------------
# Metadata dicts for ts-derived scalar features (appended to UNITS/DESCS/METHODS)
# ---------------------------------------------------------------------------
TS_UNITS: dict = {}
TS_DESCS: dict = {}
TS_METHODS: dict = {}

for _f in TS_1D_FIELDS:
    _base = _f.replace("_ts", "")
    TS_UNITS[_f + "_mean"] = "-"
    TS_UNITS[_f + "_std"]  = "-"
    TS_DESCS[_f + "_mean"] = f"[ts] Mean of {_base} over time (averaged across crops)"
    TS_DESCS[_f + "_std"]  = f"[ts] Std of {_base} over time (averaged across crops)"
    TS_METHODS[_f + "_mean"] = ["spectral/timeseries_features.py", "TimeseriesDB"]
    TS_METHODS[_f + "_std"]  = ["spectral/timeseries_features.py", "TimeseriesDB"]

for _i in range(12):
    TS_UNITS[f"hpcp_raw_{_i}"] = "-"
    TS_UNITS[f"hpcp_rot_{_i}"] = "-"
    TS_DESCS[f"hpcp_raw_{_i}"] = f"[ts] Raw chroma pitch class {_i} mean (key-sensitive)"
    TS_DESCS[f"hpcp_rot_{_i}"] = f"[ts] Tonic-rotated chroma pitch class {_i} (key-invariant)"
    TS_METHODS[f"hpcp_raw_{_i}"] = ["harmonic/chroma.py", "TimeseriesDB hpcp_ts"]
    TS_METHODS[f"hpcp_rot_{_i}"] = ["harmonic/chroma.py", "TimeseriesDB hpcp_ts"]

TS_UNITS["tonic_sin"]  = "-"
TS_UNITS["tonic_cos"]  = "-"
TS_DESCS["tonic_sin"]  = "[ts] sin(2π·tonic/12) — circular tonic encoding"
TS_DESCS["tonic_cos"]  = "[ts] cos(2π·tonic/12) — circular tonic encoding"
TS_METHODS["tonic_sin"] = ["harmonic/chroma.py", "TimeseriesDB tonic_ts"]
TS_METHODS["tonic_cos"] = ["harmonic/chroma.py", "TimeseriesDB tonic_ts"]

# ts scalars that go into the scatter plot (NUMERIC_FEATURES extension)
TS_NUMERIC_FEATURES = (
    [f + "_mean" for f in TS_1D_FIELDS] +
    [f + "_std"  for f in TS_1D_FIELDS] +
    [f"hpcp_raw_{i}" for i in range(12)] +
    [f"hpcp_rot_{i}" for i in range(12)] +
    ["tonic_sin", "tonic_cos"]
)
assert len(TS_NUMERIC_FEATURES) == EMBEDDING_DIMS, \
    f"TS_NUMERIC_FEATURES length {len(TS_NUMERIC_FEATURES)} != EMBEDDING_DIMS {EMBEDDING_DIMS}"

# ---------------------------------------------------------------------------
# Scalar feature metadata (from generate_explorer_data.py)
# ---------------------------------------------------------------------------
STEM_SUFFIXES = {"_bass", "_drums", "_other", "_vocals"}

UNITS = {
    "atonality":                    "0-1",
    "beat_count":                   "count",
    "beat_regularity":              "sec",
    "booming":                      "0-100",
    "bpm":                          "BPM",
    "bpm_essentia":                 "BPM",
    "bpm_madmom":                   "BPM",
    "brightness":                   "0-100",
    "chroma_0":  "-", "chroma_1":  "-", "chroma_2":  "-", "chroma_3":  "-",
    "chroma_4":  "-", "chroma_5":  "-", "chroma_6":  "-", "chroma_7":  "-",
    "chroma_8":  "-", "chroma_9":  "-", "chroma_10": "-", "chroma_11": "-",
    "content_enjoyment":            "1-10",
    "content_usefulness":           "1-10",
    "danceability":                 "0-1",
    "depth":                        "0-100",
    "downbeats":                    "count",
    "duration":                     "sec",
    "end_sample":                   "samples",
    "end_time":                     "sec",
    "female_probability":           "0-1",
    "hardness":                     "0-100",
    "harmonic_movement_bass":       "0-1",
    "harmonic_movement_other":      "0-1",
    "harmonic_variance_bass":       "0-1",
    "harmonic_variance_other":      "0-1",
    "instrumental_probability":     "0-1",
    "ioi_mean":                     "sec",
    "ioi_std":                      "sec",
    "lra":                          "LU",
    "lra_bass":                     "LU",
    "lra_drums":                    "LU",
    "lra_other":                    "LU",
    "lra_vocals":                   "LU",
    "lufs":                         "LUFS",
    "lufs_bass":                    "LUFS",
    "lufs_drums":                   "LUFS",
    "lufs_other":                   "LUFS",
    "lufs_vocals":                  "LUFS",
    "male_probability":             "0-1",
    "on_beat_ratio":                "0-1",
    "onset_count":                  "count",
    "onset_density":                "/sec",
    "onset_density_average_bass":   "/sec",
    "onset_density_average_drums":  "/sec",
    "onset_density_average_other":  "/sec",
    "onset_density_variance_bass":  "-",
    "onset_density_variance_drums": "-",
    "onset_density_variance_other": "-",
    "onset_strength_mean":          "-",
    "onset_strength_std":           "-",
    "popularity":                   "0-100",
    "position":                     "0-1",
    "production_complexity":        "1-10",
    "production_quality":           "1-10",
    "release_year":                 "year",
    "reverberation":                "0-100",
    "rhythmic_complexity":          "0-1",
    "rhythmic_complexity_bass":     "0-5",
    "rhythmic_complexity_drums":    "0-5",
    "rhythmic_complexity_other":    "0-5",
    "rhythmic_evenness":            "0-1",
    "rhythmic_evenness_bass":       "0-1",
    "rhythmic_evenness_drums":      "0-1",
    "rhythmic_evenness_other":      "0-1",
    "rms_energy_air":               "dB",
    "rms_energy_bass":              "dB",
    "rms_energy_body":              "dB",
    "rms_energy_mid":               "dB",
    "roughness":                    "0-100",
    "samples":                      "count",
    "saturation_count":             "count",
    "saturation_ratio":             "0-1",
    "sharpness":                    "0-100",
    "spectral_flatness":            "0-1",
    "spectral_flux":                "-",
    "spectral_kurtosis":            "-",
    "spectral_skewness":            "-",
    "start_sample":                 "samples",
    "start_time":                   "sec",
    "syncopation":                  "0-1",
    "syncopation_bass":             "0-1",
    "syncopation_drums":            "0-1",
    "syncopation_other":            "0-1",
    "track_metadata_year":          "year",
    "voice_probability":            "0-1",
    "warmth":                       "0-100",
}

DESCS = {
    "atonality":                    "How atonal/dissonant the harmonic content is (0=tonal, 1=atonal)",
    "beat_count":                   "Number of detected beats in the clip",
    "beat_regularity":              "Standard deviation of beat intervals — lower = more regular",
    "booming":                      "Perceived boominess / low-frequency resonance (0-100, from AudioCommons)",
    "bpm":                          "Beats per minute — estimated tempo",
    "bpm_essentia":                 "BPM estimate from Essentia RhythmExtractor2013",
    "bpm_madmom":                   "BPM estimate from madmom DBNBeatTracker (CPU, slow)",
    "brightness":                   "Perceived brightness of the sound (0-100, from AudioCommons)",
    "chroma_0":  None, "chroma_1":  None, "chroma_2":  None, "chroma_3":  None,
    "chroma_4":  None, "chroma_5":  None, "chroma_6":  None, "chroma_7":  None,
    "chroma_8":  None, "chroma_9":  None, "chroma_10": None, "chroma_11": None,
    "content_enjoyment":            "AudioBox aesthetic score: how enjoyable/pleasant the audio is (1-10)",
    "content_usefulness":           "AudioBox aesthetic score: how useful/purposeful the audio is (1-10)",
    "danceability":                 "Essentia danceability estimate (0-1)",
    "depth":                        "Perceived depth / spatial fullness (0-100, from AudioCommons)",
    "downbeats":                    "Number of detected downbeats (bar starts)",
    "duration":                     "Clip duration in seconds",
    "end_sample":                   "End sample index in the source file",
    "end_time":                     "End time (s) in the source file",
    "female_probability":           "Probability that the vocal source is female",
    "hardness":                     "Perceived hardness / attack sharpness (0-100, from AudioCommons)",
    "harmonic_movement_bass":       "Rate of harmonic change in the bass stem (chroma flux, 0-1)",
    "harmonic_movement_other":      "Rate of harmonic change in the 'other' stem (chroma flux, 0-1)",
    "harmonic_variance_bass":       "Variance of chroma across time in the bass stem (0-1)",
    "harmonic_variance_other":      "Variance of chroma across time in the 'other' stem (0-1)",
    "instrumental_probability":     "Probability that the track has no vocals",
    "ioi_mean":                     "Mean inter-onset interval in seconds",
    "ioi_std":                      "Standard deviation of inter-onset intervals in seconds",
    "lra":                          "Loudness range of the full mix (LU)",
    "lra_bass":                     "Loudness range of the bass stem (LU)",
    "lra_drums":                    "Loudness range of the drums stem (LU)",
    "lra_other":                    "Loudness range of the 'other' stem (LU)",
    "lra_vocals":                   "Loudness range of the vocals stem (LU)",
    "lufs":                         "Integrated loudness of the full mix (LUFS)",
    "lufs_bass":                    "Integrated loudness of the bass stem (LUFS)",
    "lufs_drums":                   "Integrated loudness of the drums stem (LUFS)",
    "lufs_other":                   "Integrated loudness of the 'other' stem (LUFS)",
    "lufs_vocals":                  "Integrated loudness of the vocals stem (LUFS)",
    "male_probability":             "Probability that the vocal source is male",
    "on_beat_ratio":                "Fraction of onsets landing on strong beat positions (0-1)",
    "onset_count":                  "Total number of detected onsets in the clip",
    "onset_density":                "Onsets per second",
    "onset_density_average_bass":   "Average onset density across bass stem crops (onsets/sec)",
    "onset_density_average_drums":  "Average onset density across drums stem crops (onsets/sec)",
    "onset_density_average_other":  "Average onset density across other stem crops (onsets/sec)",
    "onset_density_variance_bass":  "Variance in onset density across bass stem crops",
    "onset_density_variance_drums": "Variance in onset density across drums stem crops",
    "onset_density_variance_other": "Variance in onset density across other stem crops",
    "onset_strength_mean":          "Mean onset strength envelope value",
    "onset_strength_std":           "Standard deviation of onset strength",
    "popularity":                   "Track popularity (Spotify 0-100 or Tidal)",
    "position":                     "Crop position within the track (0=start, 1=end)",
    "production_complexity":        "AudioBox aesthetic score: production complexity (1-10)",
    "production_quality":           "AudioBox aesthetic score: production quality (1-10)",
    "release_year":                 "Year the track was released",
    "reverberation":                "Perceived reverberation / room size (0-100, from AudioCommons)",
    "rhythmic_complexity":          "Rhythmic complexity of the full mix (0-1; higher = more complex patterns)",
    "rhythmic_complexity_bass":     "Rhythmic complexity of the bass stem",
    "rhythmic_complexity_drums":    "Rhythmic complexity of the drums stem",
    "rhythmic_complexity_other":    "Rhythmic complexity of the 'other' stem",
    "rhythmic_evenness":            "Rhythmic evenness / regularity (0-1; higher = more even timing)",
    "rhythmic_evenness_bass":       "Rhythmic evenness of the bass stem",
    "rhythmic_evenness_drums":      "Rhythmic evenness of the drums stem",
    "rhythmic_evenness_other":      "Rhythmic evenness of the 'other' stem",
    "rms_energy_air":               "RMS energy in the 'air' band (8-20 kHz), in dB",
    "rms_energy_bass":              "RMS energy in the bass band (20-120 Hz), in dB",
    "rms_energy_body":              "RMS energy in the 'body' band (120-2500 Hz), in dB",
    "rms_energy_mid":               "RMS energy in the mid band (2500-8000 Hz), in dB",
    "roughness":                    "Perceived roughness / harshness (0-100, from AudioCommons)",
    "samples":                      "Clip length in samples",
    "saturation_count":             "Number of detected saturation/clipping events",
    "saturation_ratio":             "Fraction of frames with saturation detected (0-1)",
    "sharpness":                    "Perceived sharpness / high-frequency transient content (0-100, from AudioCommons)",
    "spectral_flatness":            "How noise-like the spectrum is (0=tonal, 1=white noise)",
    "spectral_flux":                "Average change in the magnitude spectrum between frames",
    "spectral_kurtosis":            "Kurtosis of the spectral distribution (peakedness)",
    "spectral_skewness":            "Skewness of the spectral distribution",
    "start_sample":                 "Start sample index in the source file",
    "start_time":                   "Start time (s) in the source file",
    "syncopation":                  "Degree to which beats land off the main pulse (0-1; higher = more syncopated)",
    "syncopation_bass":             "Syncopation of the bass stem",
    "syncopation_drums":            "Syncopation of the drums stem",
    "syncopation_other":            "Syncopation of the 'other' stem",
    "track_metadata_year":          "Year from track metadata tags",
    "voice_probability":            "Probability that the track contains a human voice",
    "warmth":                       "Perceived warmth / low-frequency richness (0-100, from AudioCommons)",
}

METHODS = {
    "atonality":                    ["classification/essentia_features.py", "Essentia TF VGGish + ONNX"],
    "beat_count":                   ["rhythm/bpm.py", "librosa beat_track"],
    "beat_regularity":              ["rhythm/bpm.py", "librosa beat_track"],
    "booming":                      ["timbral/audio_commons.py", "timbral_models (AudioCommons)"],
    "bpm":                          ["rhythm/bpm.py", "librosa beat_track"],
    "bpm_essentia":                 ["rhythm/bpm.py", "Essentia RhythmExtractor2013"],
    "bpm_madmom":                   ["rhythm/bpm.py", "madmom DBNBeatTracker"],
    "brightness":                   ["timbral/audio_commons.py", "timbral_models (AudioCommons)"],
    "content_enjoyment":            ["timbral/audiobox_aesthetics.py", "AudioBox Aesthetics"],
    "content_usefulness":           ["timbral/audiobox_aesthetics.py", "AudioBox Aesthetics"],
    "danceability":                 ["classification/essentia_features.py", "Essentia TF ONNX"],
    "depth":                        ["timbral/audio_commons.py", "timbral_models (AudioCommons)"],
    "female_probability":           ["classification/essentia_features.py", "Essentia voice gender"],
    "hardness":                     ["timbral/audio_commons.py", "timbral_models (AudioCommons)"],
    "harmonic_movement_bass":       ["harmonic/per_stem_harmonic.py", "Librosa chroma flux"],
    "harmonic_movement_other":      ["harmonic/per_stem_harmonic.py", "Librosa chroma flux"],
    "harmonic_variance_bass":       ["harmonic/per_stem_harmonic.py", "Librosa chroma variance"],
    "harmonic_variance_other":      ["harmonic/per_stem_harmonic.py", "Librosa chroma variance"],
    "instrumental_probability":     ["classification/essentia_features.py", "Essentia vocal detector"],
    "ioi_mean":                     ["rhythm/onsets.py", "Librosa onset detection"],
    "ioi_std":                      ["rhythm/onsets.py", "Librosa onset detection"],
    "lra":                          ["timbral/loudness.py", "pyloudnorm / ebur128"],
    "lra_bass":                     ["timbral/loudness.py", "pyloudnorm on stem"],
    "lra_drums":                    ["timbral/loudness.py", "pyloudnorm on stem"],
    "lra_other":                    ["timbral/loudness.py", "pyloudnorm on stem"],
    "lra_vocals":                   ["timbral/loudness.py", "pyloudnorm on stem"],
    "lufs":                         ["timbral/loudness.py", "pyloudnorm / ebur128"],
    "lufs_bass":                    ["timbral/loudness.py", "pyloudnorm on stem"],
    "lufs_drums":                   ["timbral/loudness.py", "pyloudnorm on stem"],
    "lufs_other":                   ["timbral/loudness.py", "pyloudnorm on stem"],
    "lufs_vocals":                  ["timbral/loudness.py", "pyloudnorm on stem"],
    "male_probability":             ["classification/essentia_features.py", "Essentia voice gender"],
    "on_beat_ratio":                ["rhythm/syncopation.py", "Beat grid analysis"],
    "onset_count":                  ["rhythm/onsets.py", "Librosa onset detection"],
    "onset_density":                ["rhythm/onsets.py", "Librosa onset detection"],
    "onset_density_average_bass":   ["rhythm/per_stem_rhythm.py", "Librosa onset detection on stem"],
    "onset_density_average_drums":  ["rhythm/per_stem_rhythm.py", "Librosa onset detection on stem"],
    "onset_density_average_other":  ["rhythm/per_stem_rhythm.py", "Librosa onset detection on stem"],
    "onset_density_variance_bass":  ["rhythm/per_stem_rhythm.py", "Librosa onset detection on stem"],
    "onset_density_variance_drums": ["rhythm/per_stem_rhythm.py", "Librosa onset detection on stem"],
    "onset_density_variance_other": ["rhythm/per_stem_rhythm.py", "Librosa onset detection on stem"],
    "onset_strength_mean":          ["rhythm/onsets.py", "Librosa onset strength"],
    "onset_strength_std":           ["rhythm/onsets.py", "Librosa onset strength"],
    "production_complexity":        ["timbral/audiobox_aesthetics.py", "AudioBox Aesthetics"],
    "production_quality":           ["timbral/audiobox_aesthetics.py", "AudioBox Aesthetics"],
    "reverberation":                ["timbral/audio_commons.py", "timbral_models (AudioCommons)"],
    "rhythmic_complexity":          ["rhythm/complexity.py", "Librosa beat grid analysis"],
    "rhythmic_complexity_bass":     ["rhythm/per_stem_rhythm.py", "Librosa beat grid on stem"],
    "rhythmic_complexity_drums":    ["rhythm/per_stem_rhythm.py", "Librosa beat grid on stem"],
    "rhythmic_complexity_other":    ["rhythm/per_stem_rhythm.py", "Librosa beat grid on stem"],
    "rhythmic_evenness":            ["rhythm/complexity.py", "Librosa beat grid analysis"],
    "rhythmic_evenness_bass":       ["rhythm/per_stem_rhythm.py", "Librosa beat grid on stem"],
    "rhythmic_evenness_drums":      ["rhythm/per_stem_rhythm.py", "Librosa beat grid on stem"],
    "rhythmic_evenness_other":      ["rhythm/per_stem_rhythm.py", "Librosa beat grid on stem"],
    "rms_energy_air":               ["spectral/multiband_rms.py", "numpy RMS per band"],
    "rms_energy_bass":              ["spectral/multiband_rms.py", "numpy RMS per band"],
    "rms_energy_body":              ["spectral/multiband_rms.py", "numpy RMS per band"],
    "rms_energy_mid":               ["spectral/multiband_rms.py", "numpy RMS per band"],
    "roughness":                    ["timbral/audio_commons.py", "timbral_models (AudioCommons)"],
    "saturation_count":             ["spectral/saturation.py", "Essentia SaturationDetector"],
    "saturation_ratio":             ["spectral/saturation.py", "Essentia SaturationDetector"],
    "sharpness":                    ["timbral/audio_commons.py", "timbral_models (AudioCommons)"],
    "spectral_flatness":            ["spectral/spectral_features.py", "Librosa spectral_flatness"],
    "spectral_flux":                ["spectral/spectral_features.py", "numpy spectrum diff"],
    "spectral_kurtosis":            ["spectral/spectral_features.py", "scipy kurtosis"],
    "spectral_skewness":            ["spectral/spectral_features.py", "scipy skew"],
    "syncopation":                  ["rhythm/syncopation.py", "Beat grid analysis"],
    "syncopation_bass":             ["rhythm/per_stem_rhythm.py", "Beat grid on stem"],
    "syncopation_drums":            ["rhythm/per_stem_rhythm.py", "Beat grid on stem"],
    "syncopation_other":            ["rhythm/per_stem_rhythm.py", "Beat grid on stem"],
    "voice_probability":            ["classification/essentia_features.py", "Essentia voice detector"],
    "warmth":                       ["timbral/audio_commons.py", "timbral_models (AudioCommons)"],
}

PERCEPTUAL_FEATURES = {
    "booming", "brightness", "content_enjoyment", "content_usefulness",
    "danceability", "depth", "hardness", "production_complexity",
    "production_quality", "reverberation", "roughness", "sharpness", "warmth",
}

NUMERIC_FEATURES = [
    "atonality", "beat_count", "beat_regularity", "booming", "bpm", "bpm_essentia", "bpm_madmom", "brightness",
    "content_enjoyment", "content_usefulness", "danceability", "depth",
    "downbeats", "duration", "end_sample", "end_time",
    "female_probability", "hardness",
    "harmonic_movement_bass", "harmonic_movement_other",
    "harmonic_variance_bass", "harmonic_variance_other",
    "instrumental_probability", "ioi_mean", "ioi_std",
    "lra", "lra_bass", "lra_drums", "lra_other", "lra_vocals",
    "lufs", "lufs_bass", "lufs_drums", "lufs_other", "lufs_vocals",
    "male_probability", "on_beat_ratio",
    "onset_count", "onset_density",
    "onset_density_average_bass", "onset_density_average_drums", "onset_density_average_other",
    "onset_density_variance_bass", "onset_density_variance_drums", "onset_density_variance_other",
    "onset_strength_mean", "onset_strength_std",
    "popularity", "position", "production_complexity", "production_quality",
    "release_year", "reverberation",
    "rhythmic_complexity", "rhythmic_complexity_bass", "rhythmic_complexity_drums", "rhythmic_complexity_other",
    "rhythmic_evenness", "rhythmic_evenness_bass", "rhythmic_evenness_drums", "rhythmic_evenness_other",
    "rms_energy_air", "rms_energy_bass", "rms_energy_body", "rms_energy_mid",
    "roughness", "samples", "saturation_count", "saturation_ratio",
    "sharpness", "spectral_flatness", "spectral_flux", "spectral_kurtosis", "spectral_skewness",
    "start_sample", "start_time",
    "syncopation", "syncopation_bass", "syncopation_drums", "syncopation_other",
    "track_metadata_year", "voice_probability", "warmth",
] + [f"chroma_{i}" for i in range(12)]

# ---------------------------------------------------------------------------
# Pure helper functions (all tested in tests/test_build_dataset_stats.py)
# ---------------------------------------------------------------------------

def _strip_crop_suffix(key: str) -> str:
    """'Artist - Title_0' -> 'Artist - Title'"""
    return re.sub(r'_\d+$', '', key)


def _interp32(arr: np.ndarray) -> np.ndarray:
    """Interpolate a 1-D array of any length to exactly 32 steps."""
    n = len(arr)
    if n == 32:
        return arr.astype(np.float32)
    xp = np.linspace(0.0, 1.0, n)
    x  = np.linspace(0.0, 1.0, 32)
    return np.interp(x, xp, arr).astype(np.float32)


def _rotate_hpcp(hpcp: np.ndarray, tonic: int) -> np.ndarray:
    """Roll a [12] chroma vector so tonic lands at index 0."""
    return np.roll(hpcp.astype(np.float32), -tonic)


def _dominant_tonic(tonic_arr: np.ndarray) -> int:
    """Mode of rounded tonic values, clamped to [0, 11]."""
    if len(tonic_arr) == 0:
        return 0
    rounded = np.round(tonic_arr).astype(int) % 12
    return int(np.argmax(np.bincount(rounded, minlength=12)))


def _cosine_top_k(emb: np.ndarray, names: list, k: int = 20) -> dict:
    """Return top-k cosine neighbors per track (self excluded).

    Args:
        emb:   [N, D] float32 embedding matrix (already z-scored)
        names: list of N track names
        k:     number of neighbors to return

    Returns:
        {track_name: [[neighbor_name, score], ...]} sorted descending
    """
    dist = scipy.spatial.distance.cdist(emb, emb, metric='cosine')  # [N, N]
    sim  = (1.0 - dist).astype(np.float32)
    np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.fill_diagonal(sim, -2.0)  # exclude self

    result = {}
    for i, name in enumerate(names):
        top_idx = np.argsort(sim[i])[::-1][:k]
        result[name] = [[names[j], round(float(sim[i, j]), 4)] for j in top_idx]
    return result


# ---------------------------------------------------------------------------
# Stage 1 — scalar pass
# ---------------------------------------------------------------------------

def is_full_mix_info(path: Path) -> bool:
    stem = path.stem
    return not any(stem.endswith(s) for s in STEM_SUFFIXES)


def load_track_data(track_dir: Path) -> dict | None:
    """Average all full-mix crop INFO values for a single track directory."""
    infos = sorted([p for p in track_dir.glob("*.INFO") if is_full_mix_info(p)])
    if not infos:
        return None

    feature_values: dict[str, list] = defaultdict(list)
    metadata: dict = {}

    for info_path in infos:
        try:
            with open(info_path) as f:
                data = json.load(f)
        except Exception:
            continue
        for k, v in data.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
                feature_values[k].append(float(v))
            elif k in ("spotify_id", "musicbrainz_id", "track_metadata_artist",
                       "track_metadata_title", "music_flamingo_short_genre",
                       "music_flamingo_short_mood", "music_flamingo_short_technical",
                       "label", "album", "isrc", "tidal_id", "tidal_url") \
                    and k not in metadata and v:
                metadata[k] = v
            elif k in ("genres", "artists") and k not in metadata \
                    and isinstance(v, list) and v:
                metadata[k] = ", ".join(str(x) for x in v)

    if not feature_values:
        return None

    averaged = {k: float(np.mean(vals)) for k, vals in feature_values.items() if vals}
    averaged.update(metadata)
    return averaged


def run_scalar_pass(src: Path, output_dir: Path) -> tuple[list, dict]:
    """Scan .INFO files and write feature_explorer_data.js (scalars only).

    Returns (sorted_tracks, tracks_data) for use by later stages.
    """
    src = Path(src)
    track_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    print(f"Stage 1: scanning {len(track_dirs)} track dirs in {src}")

    tracks_data: dict[str, dict] = {}
    for i, d in enumerate(track_dirs):
        if i % 200 == 0:
            print(f"  {i}/{len(track_dirs)} ...", end="\r")
        result = load_track_data(d)
        if result:
            tracks_data[d.name] = result

    print(f"\n  {len(tracks_data)} tracks loaded")
    sorted_tracks = sorted(tracks_data.keys())

    all_keys = set()
    for td in tracks_data.values():
        all_keys.update(td.keys())
    features_in_data = [f for f in NUMERIC_FEATURES if f in all_keys]

    write_data_js(sorted_tracks, tracks_data, features_in_data, output_dir)
    return sorted_tracks, tracks_data


def write_data_js(sorted_tracks: list, tracks_data: dict,
                  features_in_data: list, output_dir: Path) -> None:
    """Serialise all scalar features (including ts-derived) to feature_explorer_data.js."""
    out = Path(output_dir) / "feature_explorer_data.js"

    DATA = {feat: [tracks_data[t].get(feat) for t in sorted_tracks]
            for feat in features_in_data}

    def _s(key):
        return [tracks_data[t].get(key, "") or "" for t in sorted_tracks]

    all_units   = {**UNITS,   **TS_UNITS}
    all_descs   = {**DESCS,   **TS_DESCS}
    all_methods = {**METHODS, **TS_METHODS}

    with open(out, "w") as fh:
        fh.write("// Auto-generated — re-generate: python plots/build_dataset_stats.py\n\n")
        fh.write(f"const DATA = {json.dumps(DATA, separators=(',', ':'))};\n")
        fh.write(f"const TRACKS = {json.dumps(sorted_tracks, separators=(',', ':'))};\n")
        fh.write(f"const FEATURES = {json.dumps(sorted(features_in_data), separators=(',', ':'))};\n")
        fh.write(f"const UNITS = {json.dumps({f: all_units.get(f, '-') for f in features_in_data}, separators=(',', ':'))};\n")
        fh.write(f"const DESCS = {json.dumps({f: all_descs.get(f) for f in features_in_data}, separators=(',', ':'))};\n")
        fh.write(f"const METHODS = {json.dumps({f: all_methods.get(f, []) for f in features_in_data}, separators=(',', ':'))};\n")
        fh.write(f"const SPOTIFY = {json.dumps(_s('spotify_id'), separators=(',', ':'))};\n")
        fh.write(f"const MBIDS = {json.dumps(_s('musicbrainz_id'), separators=(',', ':'))};\n")
        fh.write(f"const LABELS = {json.dumps(_s('label'), separators=(',', ':'))};\n")
        fh.write(f"const ALBUMS = {json.dumps(_s('album'), separators=(',', ':'))};\n")
        fh.write(f"const GENRES = {json.dumps(_s('genres'), separators=(',', ':'))};\n")
        fh.write(f"const ARTISTS = {json.dumps(_s('artists'), separators=(',', ':'))};\n")
        fh.write(f"const ISRC = {json.dumps(_s('isrc'), separators=(',', ':'))};\n")
        fh.write(f"const TIDAL_IDS = {json.dumps(_s('tidal_id'), separators=(',', ':'))};\n")
        fh.write(f"const TIDAL_URLS = {json.dumps(_s('tidal_url'), separators=(',', ':'))};\n")
        fh.write(f"const FG_GENRE = {json.dumps(_s('music_flamingo_short_genre'), separators=(',', ':'))};\n")
        fh.write(f"const FG_MOOD = {json.dumps(_s('music_flamingo_short_mood'), separators=(',', ':'))};\n")
        fh.write(f"const FG_TECH = {json.dumps(_s('music_flamingo_short_technical'), separators=(',', ':'))};\n")
        fh.write(f"const PERCEPTUAL = new Set({json.dumps(sorted(PERCEPTUAL_FEATURES), separators=(',', ':'))});\n")

    size = out.stat().st_size
    print(f"  Wrote {out.name}: {size:,} bytes  ({len(sorted_tracks)} tracks, {len(features_in_data)} features)")


# ---------------------------------------------------------------------------
# Stage 2 — timeseries pass
# ---------------------------------------------------------------------------

def process_track_ts(crop_keys: list, db) -> dict | None:
    """Aggregate timeseries for all crops of one track.

    Returns dict with:
      'curves' : {field: list[32]}          — L∞-normalised to [0,1]
      'shape'  : {field_mean/std/hpcp/tonic: float}
      'hpcp_raw': list[12]
      'hpcp_rot': list[12]
    Returns None if no timeseries data found.
    """
    accumulated: dict[str, list] = {f: [] for f in TS_1D_FIELDS}
    hpcp_vecs: list = []
    tonic_vals: list = []

    for key in crop_keys:
        arrays = db.get(key)
        if arrays is None:
            continue
        for f in TS_1D_FIELDS:
            if f in arrays and len(arrays[f]) > 0:
                accumulated[f].append(_interp32(arrays[f]))
        if "hpcp_ts" in arrays:
            h = arrays["hpcp_ts"]
            if h.ndim == 2 and h.shape[1] == 12:
                hpcp_vecs.append(h.mean(axis=0).astype(np.float32))
        if "tonic_ts" in arrays and len(arrays["tonic_ts"]) > 0:
            tonic_vals.extend(arrays["tonic_ts"].tolist())

    has_1d = any(len(v) > 0 for v in accumulated.values())
    if not has_1d and not hpcp_vecs:
        return None

    curves: dict = {}
    shape: dict = {}

    for f in TS_1D_FIELDS:
        if not accumulated[f]:
            continue
        stack = np.stack(accumulated[f], axis=0)   # [n_crops, 32]
        mean_curve = stack.mean(axis=0)             # [32]
        mx = mean_curve.max()
        curves[f] = (mean_curve / mx if mx > 0 else mean_curve).tolist()
        shape[f + "_mean"] = float(stack.mean())
        shape[f + "_std"]  = float(stack.std())

    hpcp_raw_list = None
    hpcp_rot_list = None
    tonic = 0

    if hpcp_vecs:
        hpcp_raw_arr = np.stack(hpcp_vecs, axis=0).mean(axis=0)   # [12]
        hpcp_raw_list = hpcp_raw_arr.tolist()
        for i, v in enumerate(hpcp_raw_list):
            shape[f"hpcp_raw_{i}"] = float(v)

    if tonic_vals:
        tonic_arr = np.array(tonic_vals, dtype=np.float32)
        tonic = _dominant_tonic(tonic_arr)
        shape["tonic_sin"] = float(np.sin(2 * np.pi * tonic / 12))
        shape["tonic_cos"] = float(np.cos(2 * np.pi * tonic / 12))
        if hpcp_raw_list is not None:
            hpcp_rot_arr = _rotate_hpcp(np.array(hpcp_raw_list, dtype=np.float32), tonic)
            hpcp_rot_list = hpcp_rot_arr.tolist()
            for i, v in enumerate(hpcp_rot_list):
                shape[f"hpcp_rot_{i}"] = float(v)

    result: dict = {"curves": curves, "shape": shape}
    if hpcp_raw_list is not None:
        result["hpcp_raw"] = hpcp_raw_list
    if hpcp_rot_list is not None:
        result["hpcp_rot"] = hpcp_rot_list
    return result


def run_timeseries_pass(sorted_tracks: list, output_dir: Path, db_path: Path) -> dict:
    """Query TimeseriesDB for all tracks, build shape vectors and mini curves.

    Saves .ts_cache.npz to output_dir for Stage 3 reuse.
    Returns ts_data: {track_name: process_track_ts result}.
    """
    output_dir = Path(output_dir)
    db = TimeseriesDB.open(db_path)
    print(f"Stage 2: TimeseriesDB has {db.count():,} entries")

    # Map crop keys to track names
    all_keys = db.all_keys()
    crops_by_track: dict[str, list] = defaultdict(list)
    track_set = set(sorted_tracks)
    for key in all_keys:
        track = _strip_crop_suffix(key)
        if track in track_set:
            crops_by_track[track].append(key)

    print(f"  {len(crops_by_track)} tracks have timeseries data")

    ts_data: dict[str, dict] = {}
    for i, name in enumerate(sorted_tracks):
        if i % 200 == 0:
            print(f"  {i}/{len(sorted_tracks)} ...", end="\r")
        crop_keys = crops_by_track.get(name, [])
        if not crop_keys:
            continue
        result = process_track_ts(crop_keys, db)
        if result is not None:
            ts_data[name] = result

    db.close()
    print(f"\n  {len(ts_data)} tracks with ts data")

    # Save cache for Stage 3 reuse — raw (un-normalised) embedding rows
    N = len(sorted_tracks)
    name_arr = np.array(sorted_tracks)
    raw_emb = np.zeros((N, EMBEDDING_DIMS), dtype=np.float32)
    mini_curves = np.zeros((N, len(TS_1D_FIELDS), 32), dtype=np.float32)

    for i, name in enumerate(sorted_tracks):
        td = ts_data.get(name)
        if td is None:
            continue
        s = td.get("shape", {})
        col = 0
        for f in TS_1D_FIELDS:
            raw_emb[i, col]   = s.get(f + "_mean", 0.0)
            raw_emb[i, col+1] = s.get(f + "_std",  0.0)
            col += 2
        for j in range(12):
            raw_emb[i, col+j] = s.get(f"hpcp_raw_{j}", 0.0)
        col += 12
        for j in range(12):
            raw_emb[i, col+j] = s.get(f"hpcp_rot_{j}", 0.0)
        col += 12
        raw_emb[i, col]   = s.get("tonic_sin", 0.0)
        raw_emb[i, col+1] = s.get("tonic_cos", 0.0)

        for fi, f in enumerate(TS_1D_FIELDS):
            c = td["curves"].get(f)
            if c:
                mini_curves[i, fi] = np.array(c, dtype=np.float32)

    cache_path = output_dir / ".ts_cache.npz"
    np.savez_compressed(cache_path,
        track_names=name_arr,
        raw_embedding=raw_emb,
        mini_curves=mini_curves)
    print(f"  Cache saved: {cache_path}")
    return ts_data
