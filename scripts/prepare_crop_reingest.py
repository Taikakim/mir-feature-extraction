#!/usr/bin/env python3
"""
prepare_crop_reingest.py — Phase 3 of clean-source upgrade.

For each track in Goa_Separated with a .NEEDS_RESEPARATION marker:

  1. Delete all crop audio files (.flac / .wav / .mp3) from Goa_Separated_crops
     so the pipeline will re-crop from the clean full_mix.flac
  2. Strip compression-sensitive keys from every .INFO file in those crop
     folders so the pipeline re-computes them from the clean source

Sidecar files (.BEATS_GRID, .DOWNBEATS, .ONSETS, .INFO.lock) and all
metadata / rhythm / tonal INFO keys are preserved.

Keys stripped — grouped by source model:
  Spectral signal      spectral_* (flatness, flux, kurtosis, skewness + _ts)
  Saturation           saturation_count, saturation_ratio
  AudioCommons timbral booming, brightness, depth, hardness, reverberation,
                       roughness, sharpness, warmth
  AudioBox / quality   atonality, content_enjoyment, content_usefulness,
                       danceability, production_quality, production_complexity
  Essentia classifiers essentia_genre, essentia_instrument, essentia_mood
  Voice / gender       voice_probability, instrumental_probability,
                       female_probability, male_probability

Usage:
    python scripts/prepare_crop_reingest.py [--dry-run] [--verbose]
"""

import argparse
import json
from pathlib import Path

GOA_SEPARATED       = Path("/run/media/kim/Mantu/ai-music/Goa_Separated")
GOA_SEPARATED_CROPS = Path("/run/media/kim/Mantu/ai-music/Goa_Separated_crops")

AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac'}

# Keys to strip from .INFO files — re-run pipeline fills these in from clean audio
STRIP_PREFIXES = {
    'spectral_',    # spectral_flatness, spectral_flux, spectral_kurtosis, spectral_skewness (+ _ts)
    'saturation_',  # saturation_count, saturation_ratio
}

STRIP_EXACT = {
    # AudioCommons perceptual timbral descriptors
    'booming', 'brightness', 'depth', 'hardness', 'reverberation',
    'roughness', 'sharpness', 'warmth',
    # AudioBox / production quality models
    'atonality', 'content_enjoyment', 'content_usefulness',
    'danceability', 'production_quality', 'production_complexity',
    # Essentia classifier outputs
    'essentia_genre', 'essentia_instrument', 'essentia_mood',
    # Voice / gender detection
    'voice_probability', 'instrumental_probability',
    'female_probability', 'male_probability',
}


def should_strip(key: str) -> bool:
    if key in STRIP_EXACT:
        return True
    return any(key.startswith(p) for p in STRIP_PREFIXES)


def strip_info_file(info_path: Path, dry: bool) -> tuple[int, int]:
    """Load, strip bad keys, re-save. Returns (keys_removed, total_keys)."""
    try:
        data = json.loads(info_path.read_text())
    except Exception as e:
        print(f"    WARNING: could not read {info_path.name}: {e}")
        return 0, 0

    bad_keys = [k for k in data if should_strip(k)]
    if not bad_keys:
        return 0, len(data)

    for k in bad_keys:
        del data[k]

    if not dry:
        info_path.write_text(json.dumps(data, indent=2))

    return len(bad_keys), len(data) + len(bad_keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print each file deleted / INFO stripped')
    args = parser.parse_args()

    dry     = args.dry_run
    verbose = args.verbose or dry

    markers = sorted(GOA_SEPARATED.rglob('.NEEDS_RESEPARATION'))
    if not markers:
        print("No .NEEDS_RESEPARATION markers found — nothing to do.")
        return

    print(f"Found {len(markers)} tracks marked for re-separation.")
    if dry:
        print("*** DRY RUN — no changes will be made ***")
    print()

    no_crops_dir  = []
    audio_deleted = 0
    info_stripped = 0
    info_keys_removed = 0
    tracks_done   = 0

    for marker in markers:
        track_name = marker.parent.name
        crops_dir  = GOA_SEPARATED_CROPS / track_name

        if not crops_dir.exists():
            if verbose:
                print(f"[{track_name}]  NO CROPS FOLDER — skipping")
            no_crops_dir.append(track_name)
            continue

        audio_files = [f for f in crops_dir.iterdir()
                       if f.suffix.lower() in AUDIO_EXTS]
        info_files  = sorted(crops_dir.glob('*.INFO'))

        print(f"[{track_name}]")
        print(f"  {len(audio_files)} crop audio files, {len(info_files)} INFO files")

        # --- delete crop audio ---
        for af in sorted(audio_files):
            if verbose:
                print(f"  delete  {af.name}")
            if not dry:
                af.unlink()
            audio_deleted += 1

        # --- strip bad keys from each INFO ---
        for info_path in info_files:
            removed, total = strip_info_file(info_path, dry)
            if removed:
                if verbose:
                    print(f"  strip   {info_path.name}  ({removed}/{total} keys removed)")
                info_stripped     += 1
                info_keys_removed += removed

        tracks_done += 1

    print()
    print("=" * 60)
    if dry:
        print("DRY RUN complete.")
    else:
        print("Done.")
    print(f"  Tracks processed         : {tracks_done}")
    print(f"  Crop audio files deleted : {audio_deleted}")
    print(f"  INFO files modified      : {info_stripped}")
    print(f"  INFO keys stripped total : {info_keys_removed}")
    if no_crops_dir:
        print(f"  Tracks with no crops dir : {len(no_crops_dir)}")
        for t in no_crops_dir:
            print(f"    {t}")
    print()
    print("Next steps:")
    print("  1. Run cropping step to rebuild audio from clean full_mix.flac")
    print("  2. Run master_pipeline.py — missing keys will be filled in automatically")
    print("  3. Re-encode latents: encode_dataset.py and encode_stems.py for affected tracks")


if __name__ == '__main__':
    main()
