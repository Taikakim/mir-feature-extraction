#!/usr/bin/env python3
"""
replace_compressed_sources.py — Phase 1 of clean-source upgrade.

For each track in Goa_Separated whose full_mix is a lossy file (mp3/ogg/m4a)
AND a clean FLAC exists in goa-new:

  1. Quarantine all audio files (full_mix + stems) to
     Goa_Separated_compressed_quarantine/<TrackFolder>/
  2. Copy the clean FLAC from goa-new as full_mix.flac
  3. Leave sidecar files (.INFO, .BEATS_GRID, .DOWNBEATS, .ONSETS) in place
  4. Write a .NEEDS_RESEPARATION marker so the pipeline knows to re-run Demucs

Sidecar files (rhythm/beat analysis) are kept because they depend on the
audio content which doesn't change significantly with compression on goa trance,
and re-analysis is expensive. Spectral/audiobox/timbral will be cleaned from
crop INFOs in a separate phase.

Usage:
    python scripts/replace_compressed_sources.py [--dry-run] [--verbose]
"""

import argparse
import shutil
import sys
from pathlib import Path

GOA_SEPARATED  = Path("/run/media/kim/Mantu/ai-music/Goa_Separated")
GOA_NEW        = Path("/run/media/kim/Lehto/goa-new")
QUARANTINE_ROOT = Path("/run/media/kim/Mantu/ai-music/Goa_Separated_compressed_quarantine")

LOSSY_EXTS = {'.mp3', '.ogg', '.m4a', '.aac', '.wma', '.opus'}
AUDIO_EXTS = LOSSY_EXTS | {'.flac', '.wav'}

# Stem basenames that Demucs produces
STEM_NAMES = {'bass', 'drums', 'other', 'vocals', 'guitar', 'piano'}
# Audio basenames to quarantine (full_mix + all stems)
AUDIO_BASENAMES = {'full_mix'} | STEM_NAMES


def build_clean_map() -> dict[str, Path]:
    """Return {track_name_lower: clean_flac_path} from goa-new."""
    m = {}
    for flac in GOA_NEW.rglob("*.flac"):
        m[flac.stem.lower()] = flac
    return m


def find_compressed_tracks() -> list[tuple[Path, str]]:
    """Return list of (track_folder, lossy_ext) for folders with a lossy full_mix."""
    results = []
    for folder in sorted(GOA_SEPARATED.iterdir()):
        if not folder.is_dir():
            continue
        for ext in LOSSY_EXTS:
            if (folder / f"full_mix{ext}").exists():
                results.append((folder, ext))
                break
    return results


def audio_files_in_folder(folder: Path) -> list[Path]:
    """Return all audio files that belong to full_mix or stems."""
    found = []
    for f in folder.iterdir():
        if f.suffix.lower() in AUDIO_EXTS and f.stem.lower() in AUDIO_BASENAMES:
            found.append(f)
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print every file moved/copied')
    args = parser.parse_args()

    dry = args.dry_run
    verbose = args.verbose or dry

    clean_map = build_clean_map()
    compressed_tracks = find_compressed_tracks()

    # Match compressed tracks to clean replacements
    replacements = []
    no_match = []
    for folder, lossy_ext in compressed_tracks:
        clean_flac = clean_map.get(folder.name.lower())
        if clean_flac:
            replacements.append((folder, lossy_ext, clean_flac))
        else:
            no_match.append(folder.name)

    print(f"Compressed tracks found : {len(compressed_tracks)}")
    print(f"Clean replacements found: {len(replacements)}")
    print(f"No clean version yet    : {len(no_match)}")
    if dry:
        print("*** DRY RUN — no changes will be made ***")
    print()

    quarantined_tracks = 0
    quarantined_files  = 0
    skipped            = 0
    errors             = []

    for folder, lossy_ext, clean_flac in replacements:
        q_dir = QUARANTINE_ROOT / folder.name
        audio_files = audio_files_in_folder(folder)

        print(f"[{folder.name}]")
        print(f"  source : {clean_flac}")
        print(f"  lossy  : full_mix{lossy_ext}  ({len(audio_files)} audio files total)")

        # Skip if quarantine already has files (was already processed)
        if q_dir.exists() and any(q_dir.iterdir()):
            print(f"  SKIP   : quarantine already exists")
            skipped += 1
            continue

        if not dry:
            q_dir.mkdir(parents=True, exist_ok=True)

        # Move all audio files to quarantine
        for af in audio_files:
            dest = q_dir / af.name
            if verbose:
                print(f"  move → quarantine/{af.name}")
            if not dry:
                try:
                    shutil.move(str(af), str(dest))
                    quarantined_files += 1
                except Exception as e:
                    errors.append(f"{folder.name}/{af.name}: {e}")
                    print(f"  ERROR moving {af.name}: {e}")

        # Copy clean FLAC as full_mix.flac
        target = folder / "full_mix.flac"
        if verbose:
            print(f"  copy → full_mix.flac ({clean_flac.stat().st_size // 1024 // 1024} MB)")
        if not dry:
            try:
                shutil.copy2(str(clean_flac), str(target))
            except Exception as e:
                errors.append(f"{folder.name}/full_mix.flac: {e}")
                print(f"  ERROR copying full_mix.flac: {e}")
                continue

        # Write marker for re-separation
        marker = folder / ".NEEDS_RESEPARATION"
        if verbose:
            print(f"  write  → .NEEDS_RESEPARATION")
        if not dry:
            marker.write_text(f"Source: {clean_flac}\nLossy was: full_mix{lossy_ext}\n")

        quarantined_tracks += 1

    print()
    print("=" * 60)
    if dry:
        print(f"DRY RUN complete — would process {len(replacements)} tracks")
    else:
        print(f"Done.")
        print(f"  Tracks processed : {quarantined_tracks}")
        print(f"  Files quarantined: {quarantined_files}")
        print(f"  Skipped (already done): {skipped}")
        if errors:
            print(f"  Errors ({len(errors)}):")
            for e in errors:
                print(f"    {e}")
        print()
        print(f"Quarantine location: {QUARANTINE_ROOT}")
        print(f"Next step: run Demucs re-separation on folders with .NEEDS_RESEPARATION")
        marker_count = sum(1 for f in GOA_SEPARATED.rglob(".NEEDS_RESEPARATION"))
        print(f"  ({marker_count} folders marked)")


if __name__ == "__main__":
    main()
