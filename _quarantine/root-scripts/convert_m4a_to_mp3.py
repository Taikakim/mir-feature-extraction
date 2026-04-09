#!/usr/bin/env python3
"""Convert all .m4a files to .mp3 at comparable bitrate using multiprocessing."""

import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/run/media/kim/LostLands/ai-music/Goa_Separated_crops")
NUM_WORKERS = 12


def get_bitrate(filepath: Path) -> int | None:
    """Get bitrate in bits/s from an audio file using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=bit_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(filepath),
            ],
            capture_output=True, text=True, timeout=30,
        )
        val = result.stdout.strip()
        if val and val != "N/A":
            return int(val)
    except Exception:
        pass
    return None


def map_to_mp3_bitrate(aac_bitrate: int | None) -> str:
    """Map AAC bitrate to comparable MP3 bitrate (AAC is ~20-30% more efficient)."""
    if aac_bitrate is None:
        return "128k"
    if aac_bitrate < 80_000:
        return "96k"
    if aac_bitrate < 120_000:
        return "128k"
    if aac_bitrate < 180_000:
        return "192k"
    if aac_bitrate < 260_000:
        return "256k"
    return "320k"


def convert_file(m4a_path: Path) -> str:
    """Convert a single m4a file to mp3. Returns a status message."""
    mp3_path = m4a_path.with_suffix(".mp3")

    if mp3_path.exists():
        return f"SKIP (exists): {m4a_path.name}"

    bitrate = get_bitrate(m4a_path)
    mp3_br = map_to_mp3_bitrate(bitrate)

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(m4a_path),
                "-b:a", mp3_br,
                "-y", "-loglevel", "error",
                str(mp3_path),
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return f"ERROR: {m4a_path.name} — {result.stderr.strip()}"
        return f"OK ({mp3_br}): {m4a_path.name}"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: {m4a_path.name}"
    except Exception as e:
        return f"ERROR: {m4a_path.name} — {e}"


def collect_m4a_files(root: Path) -> list[Path]:
    """Walk the directory tree and collect all .m4a files."""
    print(f"Scanning {root} for .m4a files...")
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".m4a"):
                files.append(Path(dirpath) / f)
    print(f"Found {len(files)} .m4a files.")
    return files


def main():
    if not ROOT.exists():
        print(f"ERROR: Directory not found: {ROOT}")
        sys.exit(1)

    files = collect_m4a_files(ROOT)
    if not files:
        print("No .m4a files found.")
        sys.exit(0)

    converted = 0
    skipped = 0
    errors = 0

    print(f"Converting with {NUM_WORKERS} parallel workers...\n")
    with Pool(processes=NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(convert_file, files), 1):
            print(f"[{i}/{len(files)}] {result}")
            if result.startswith("OK"):
                converted += 1
            elif result.startswith("SKIP"):
                skipped += 1
            else:
                errors += 1

    print(f"\nDone! Converted: {converted}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
