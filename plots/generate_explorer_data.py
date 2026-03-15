#!/usr/bin/env python3
"""
Generate feature_explorer_data.js from .INFO crop files.

Scans Goa_Separated_crops for full-mix .INFO files, averages numeric features
across all crops for each track, then writes JS arrays to feature_explorer_data.js.

Usage:
    python plots/generate_explorer_data.py
    python plots/generate_explorer_data.py --source /path/to/Goa_Separated_crops --output /run/media/kim/Lehto/feature_explorer_data.js
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PLOTS_DIR   = Path(__file__).resolve().parent
DEFAULT_SRC = Path("/run/media/kim/Mantu/ai-music/Goa_Separated_crops")
DEFAULT_OUT = Path("/run/media/kim/Lehto/feature_explorer_data.js")

# Stems — INFO files for stems are skipped
STEM_SUFFIXES = {"_bass", "_drums", "_other", "_vocals"}

# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------
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

# Features shown in the perceptual radar (curated subjective qualities)
PERCEPTUAL_FEATURES = {
    "booming", "brightness", "content_enjoyment", "content_usefulness",
    "danceability", "depth", "hardness", "production_complexity",
    "production_quality", "reverberation", "roughness", "sharpness", "warmth",
}

# Numeric features to include in DATA (must be plottable scalars)
# Chroma features are generated programmatically below
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
# Load INFO files
# ---------------------------------------------------------------------------

def is_full_mix_info(path: Path) -> bool:
    stem = path.stem  # e.g. "0001 ... _0"
    for s in STEM_SUFFIXES:
        if stem.endswith(s):
            return False
    return True


def load_track_data(track_dir: Path) -> dict | None:
    """Average all crop INFO values for a single track directory."""
    infos = sorted([p for p in track_dir.glob("*.INFO") if is_full_mix_info(p)])
    if not infos:
        return None

    # Per-feature list of values across crops
    feature_values: dict[str, list] = defaultdict(list)
    # Single-valued metadata from first crop
    metadata: dict = {}

    for info_path in infos:
        try:
            with open(info_path) as f:
                data = json.load(f)
        except Exception:
            continue

        for k, v in data.items():
            if isinstance(v, (int, float)) and v is not None:
                feature_values[k].append(float(v))
            elif k in ("spotify_id", "musicbrainz_id", "track_metadata_artist",
                       "track_metadata_title", "music_flamingo_short_genre",
                       "music_flamingo_short_mood", "music_flamingo_short_technical",
                       "label", "album", "isrc", "tidal_id", "tidal_url") \
                    and k not in metadata and v:
                metadata[k] = v
            elif k in ("genres", "artists") and k not in metadata and isinstance(v, list) and v:
                # Join lists to comma-separated strings for display
                metadata[k] = ", ".join(str(x) for x in v)

    if not feature_values:
        return None

    # Average numeric features
    averaged = {k: float(np.mean(vals)) for k, vals in feature_values.items() if vals}
    averaged.update(metadata)
    return averaged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate feature_explorer_data.js")
    parser.add_argument("--source", default=str(DEFAULT_SRC),
                        help=f"Root of Goa_Separated_crops (default: {DEFAULT_SRC})")
    parser.add_argument("--output", default=str(DEFAULT_OUT),
                        help=f"Output JS file (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.output)

    if not src.exists():
        print(f"Error: source not found: {src}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {src} ...")
    track_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    print(f"  {len(track_dirs)} track directories found")

    # Load per-track averaged data
    tracks_data: dict[str, dict] = {}
    for i, d in enumerate(track_dirs):
        if i % 200 == 0:
            print(f"  Loading {i}/{len(track_dirs)} ...", end="\r")
        result = load_track_data(d)
        if result:
            tracks_data[d.name] = result

    print(f"\n  {len(tracks_data)} tracks loaded")

    # Determine which numeric features are actually present
    all_feature_keys = set()
    for td in tracks_data.values():
        all_feature_keys.update(td.keys())

    features_in_data = [f for f in NUMERIC_FEATURES if f in all_feature_keys]
    print(f"  {len(features_in_data)} numeric features present")

    # Build DATA arrays (sorted by track name)
    sorted_tracks = sorted(tracks_data.keys())
    DATA = {}
    for feat in features_in_data:
        arr = []
        for track in sorted_tracks:
            v = tracks_data[track].get(feat)
            arr.append(v)  # None if missing
        DATA[feat] = arr

    # Spotify IDs and MBIDs
    spotify_ids = [tracks_data[t].get("spotify_id", "") or "" for t in sorted_tracks]
    mbids       = [tracks_data[t].get("musicbrainz_id", "") or "" for t in sorted_tracks]

    # Additional string metadata arrays
    def _str_arr(key):
        return [tracks_data[t].get(key, "") or "" for t in sorted_tracks]

    labels     = _str_arr("label")
    albums     = _str_arr("album")
    genres     = _str_arr("genres")
    artists    = _str_arr("artists")
    isrc_arr   = _str_arr("isrc")
    tidal_ids  = _str_arr("tidal_id")
    tidal_urls = _str_arr("tidal_url")

    # Flamingo short captions (already extracted into metadata)
    fg_genre   = _str_arr("music_flamingo_short_genre")
    fg_mood    = _str_arr("music_flamingo_short_mood")
    fg_tech    = _str_arr("music_flamingo_short_technical")

    # Filter metadata dicts to what we have
    units_out   = {f: UNITS.get(f, "-")   for f in features_in_data}
    descs_out   = {f: DESCS.get(f)        for f in features_in_data}
    methods_out = {f: METHODS.get(f, [])  for f in features_in_data}

    # Write JS
    print(f"Writing {out} ...")
    with open(out, "w") as fh:
        fh.write("// Auto-generated data for Feature Explorer\n")
        fh.write("// Re-generate with: python plots/generate_explorer_data.py\n")
        fh.write("\n")
        fh.write(f"const DATA = {json.dumps(DATA, separators=(',', ':'))};\n")
        fh.write(f"const TRACKS = {json.dumps(sorted_tracks, separators=(',', ':'))};\n")
        fh.write(f"const FEATURES = {json.dumps(sorted(features_in_data), separators=(',', ':'))};\n")
        fh.write(f"const UNITS = {json.dumps(units_out, separators=(',', ':'))};\n")
        fh.write(f"const DESCS = {json.dumps(descs_out, separators=(',', ':'))};\n")
        fh.write(f"const METHODS = {json.dumps(methods_out, separators=(',', ':'))};\n")
        fh.write(f"const SPOTIFY = {json.dumps(spotify_ids, separators=(',', ':'))};\n")
        fh.write(f"const MBIDS = {json.dumps(mbids, separators=(',', ':'))};\n")
        fh.write(f"const LABELS = {json.dumps(labels, separators=(',', ':'))};\n")
        fh.write(f"const ALBUMS = {json.dumps(albums, separators=(',', ':'))};\n")
        fh.write(f"const GENRES = {json.dumps(genres, separators=(',', ':'))};\n")
        fh.write(f"const ARTISTS = {json.dumps(artists, separators=(',', ':'))};\n")
        fh.write(f"const ISRC = {json.dumps(isrc_arr, separators=(',', ':'))};\n")
        fh.write(f"const TIDAL_IDS = {json.dumps(tidal_ids, separators=(',', ':'))};\n")
        fh.write(f"const TIDAL_URLS = {json.dumps(tidal_urls, separators=(',', ':'))};\n")
        fh.write(f"const FG_GENRE = {json.dumps(fg_genre, separators=(',', ':'))};\n")
        fh.write(f"const FG_MOOD = {json.dumps(fg_mood, separators=(',', ':'))};\n")
        fh.write(f"const FG_TECH = {json.dumps(fg_tech, separators=(',', ':'))};\n")
        fh.write(f"const PERCEPTUAL = new Set({json.dumps(sorted(PERCEPTUAL_FEATURES), separators=(',', ':'))});\n")

    size = out.stat().st_size
    print(f"Done. {size:,} bytes written to {out}")
    print(f"  Tracks: {len(sorted_tracks)}   Features: {len(features_in_data)}")


if __name__ == "__main__":
    main()
