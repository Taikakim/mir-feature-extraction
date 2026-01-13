#!/usr/bin/env python3
"""
Download Essentia Models for MIR Project

This script downloads all required Essentia TensorFlow models for:
- Danceability
- Tonality/Atonality
- Voice/Instrumental
- Vocal Gender
- Genre
- Mood/Theme
- Instrument

Usage:
    python scripts/download_essentia_models.py
    python scripts/download_essentia_models.py --models-dir /custom/path
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Model URLs
MODELS = {
    # VGGish-based classifiers
    "danceability-vggish-audioset-1.pb":
        "https://essentia.upf.edu/models/classifiers/danceability/danceability-vggish-audioset-1.pb",
    "danceability-vggish-audioset-1.json":
        "https://essentia.upf.edu/models/classifiers/danceability/danceability-vggish-audioset-1.json",

    "tonal_atonal-vggish-audioset-1.pb":
        "https://essentia.upf.edu/models/classifiers/tonal_atonal/tonal_atonal-vggish-audioset-1.pb",
    "tonal_atonal-vggish-audioset-1.json":
        "https://essentia.upf.edu/models/classifiers/tonal_atonal/tonal_atonal-vggish-audioset-1.json",

    "voice_instrumental-vggish-audioset-1.pb":
        "https://essentia.upf.edu/models/classifiers/voice_instrumental/voice_instrumental-vggish-audioset-1.pb",
    "voice_instrumental-vggish-audioset-1.json":
        "https://essentia.upf.edu/models/classifiers/voice_instrumental/voice_instrumental-vggish-audioset-1.json",

    "gender-vggish-audioset-1.pb":
        "https://essentia.upf.edu/models/legacy/classifiers/gender/gender-vggish-audioset-1.pb",
    "gender-vggish-audioset-1.json":
        "https://essentia.upf.edu/models/legacy/classifiers/gender/gender-vggish-audioset-1.json",

    # Effnet-based classifiers
    "genre_discogs400-discogs-effnet-1.pb":
        "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    "discogs-effnet-bs64-1.pb":
        "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
    "mtg_jamendo_moodtheme-discogs-effnet-1.pb":
        "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    "mtg_jamendo_instrument-discogs-effnet-1.pb":
        "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb",
}


def download_file(url: str, dest_path: Path, resume: bool = True) -> bool:
    """
    Download a file with progress indication.

    Args:
        url: URL to download from
        dest_path: Destination file path
        resume: Whether to resume partial downloads

    Returns:
        True if successful, False otherwise
    """
    # Check if file already exists
    if dest_path.exists() and not resume:
        logger.info(f"Skipping {dest_path.name} (already exists)")
        return True

    try:
        logger.info(f"Downloading {dest_path.name}...")

        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')

        urlretrieve(url, dest_path, reporthook=progress_hook)
        print()  # New line after progress
        logger.info(f"✓ Downloaded {dest_path.name}")
        return True

    except URLError as e:
        logger.error(f"✗ Failed to download {dest_path.name}: {e}")
        return False
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user")
        if dest_path.exists():
            logger.info(f"Removing partial file: {dest_path}")
            dest_path.unlink()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download Essentia models for MIR project",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--models-dir',
        type=str,
        default=None,
        help='Directory to save models (default: ~/Projects/mir/models/essentia)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume partial downloads (default: True)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download of existing files'
    )

    args = parser.parse_args()

    # Determine models directory
    if args.models_dir:
        models_dir = Path(args.models_dir)
    else:
        models_dir = Path.home() / "Projects" / "mir" / "models" / "essentia"

    # Create directory
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Downloading {len(MODELS)} files...")
    logger.info("")

    # Download each model
    success_count = 0
    failed_count = 0

    for filename, url in MODELS.items():
        dest_path = models_dir / filename

        # Skip if exists and not forcing
        if dest_path.exists() and not args.force:
            logger.info(f"Skipping {filename} (already exists)")
            success_count += 1
            continue

        if download_file(url, dest_path, resume=args.resume and not args.force):
            success_count += 1
        else:
            failed_count += 1
        logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("Download Summary:")
    logger.info(f"  Total files:  {len(MODELS)}")
    logger.info(f"  Downloaded:   {success_count}")
    logger.info(f"  Failed:       {failed_count}")
    logger.info("=" * 60)

    if failed_count > 0:
        logger.warning(f"{failed_count} files failed to download")
        logger.info("You can re-run this script to retry failed downloads")
        sys.exit(1)

    logger.info("All models downloaded successfully!")
    logger.info("")
    logger.info(f"Models location: {models_dir}")
    logger.info("")
    logger.info("To use these models, set the environment variable:")
    logger.info(f'export ESSENTIA_MODELS_DIR="{models_dir}"')
    logger.info("")
    logger.info("Or add this to your ~/.bashrc or ~/.zshrc")


if __name__ == "__main__":
    main()
