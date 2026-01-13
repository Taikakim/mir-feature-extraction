"""
Demucs Stem Separation for MIR Project

This module wraps Demucs HT v4 for source separation.
Separates audio into: drums, bass, other, vocals

Dependencies:
- demucs
- src.core.file_utils
- src.core.common

Output:
- drums.mp3
- bass.mp3
- other.mp3
- vocals.mp3

Configuration (from MIR plan):
- Model: htdemucs (HT v4)
- Shifts: 1
- File type: mp3 @ 320kbps (changed from flac due to TorchCodec/FFmpeg issues)
- Concurrent jobs: 4
"""

import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import find_audio_files, is_organized, get_audio_folder_structure
from core.common import DEMUCS_CONFIG, DEMUCS_STEMS, AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)


def check_demucs_installed() -> bool:
    """
    Check if demucs is installed and available.

    Returns:
        True if demucs is available, False otherwise
    """
    return shutil.which('demucs') is not None


def separate_stems(audio_file: str | Path,
                   output_dir: str | Path,
                   model: str = None,
                   shifts: int = None,
                   jobs: int = None,
                   device: str = 'cuda') -> Dict[str, Path]:
    """
    Separate an audio file into stems using Demucs.

    Args:
        audio_file: Path to input audio file
        output_dir: Directory where stems will be saved
        model: Demucs model to use (default: from DEMUCS_CONFIG)
        shifts: Number of random shifts for prediction (default: from config)
        jobs: Number of parallel jobs (default: from config)
        device: Device to use ('cuda', 'cpu', 'mps')

    Returns:
        Dictionary mapping stem names to output file paths

    Raises:
        FileNotFoundError: If demucs is not installed
        subprocess.CalledProcessError: If demucs fails
    """
    audio_file = Path(audio_file)
    output_dir = Path(output_dir)

    if not check_demucs_installed():
        raise FileNotFoundError("Demucs not found. Install with: pip install demucs")

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # Use defaults from config if not specified
    if model is None:
        model = DEMUCS_CONFIG['model']
    if shifts is None:
        shifts = DEMUCS_CONFIG['shifts']
    if jobs is None:
        jobs = DEMUCS_CONFIG['jobs']

    logger.info(f"Separating stems: {audio_file.name}")
    logger.info(f"Model: {model}, Shifts: {shifts}, Jobs: {jobs}, Device: {device}")

    # Prepare demucs command
    cmd = [
        'demucs',
        '--two-stems=vocals',  # This separates into 4 stems with htdemucs
        '-n', model,
        '--shifts', str(shifts),
        '-j', str(jobs),
        '--out', str(output_dir),
        '--filename', '{track}.{stem}.flac',  # Output format
        '-d', device,
        str(audio_file)
    ]

    # Note: If using --two-stems, remove it for full 4-stem separation
    # For htdemucs, we want all 4 stems, so let's adjust:
    cmd = [
        'demucs',
        '-n', model,
        '--shifts', str(shifts),
        '-j', str(jobs),
        '--out', str(output_dir),
        '--filename', '{stem}.mp3',
        '--mp3',
        '--mp3-bitrate', '320',
        '-d', device,
        str(audio_file)
    ]

    try:
        # Run demucs
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        logger.debug(f"Demucs stdout: {result.stdout}")

        # Find output files
        # Demucs MP3 output: output_dir/model_name/stem.mp3 (flat structure)
        # We need to move them to the correct location

        # Check flat structure first (MP3 output)
        demucs_output_flat = output_dir / model
        demucs_output_nested = output_dir / model / audio_file.stem

        stem_paths = {}

        # Try flat structure first (MP3)
        if demucs_output_flat.exists():
            for stem_name in DEMUCS_STEMS:
                stem_file = demucs_output_flat / f"{stem_name}.mp3"
                if stem_file.exists():
                    # Move to final location (parent of demucs output)
                    final_path = output_dir / f"{stem_name}.mp3"
                    shutil.move(str(stem_file), str(final_path))
                    stem_paths[stem_name] = final_path
                    logger.info(f"  Created: {stem_name}.mp3")

        # Try nested structure (WAV/FLAC)
        if not stem_paths and demucs_output_nested.exists():
            for stem_name in DEMUCS_STEMS:
                for ext in ['.mp3', '.wav', '.flac']:
                    stem_file = demucs_output_nested / f"{stem_name}{ext}"
                    if stem_file.exists():
                        final_path = output_dir / f"{stem_name}{ext}"
                        shutil.move(str(stem_file), str(final_path))
                        stem_paths[stem_name] = final_path
                        logger.info(f"  Created: {stem_name}{ext}")
                        break

        # Clean up demucs output directory structure
        try:
            shutil.rmtree(output_dir / model)
        except:
            pass

        if not stem_paths:
            raise RuntimeError("No stem files were created by demucs")

        logger.info(f"Successfully separated {len(stem_paths)} stems")
        return stem_paths

    except subprocess.CalledProcessError as e:
        logger.error(f"Demucs failed: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Error during separation: {e}")
        raise


def separate_organized_folder(audio_folder: str | Path,
                               overwrite: bool = False,
                               **kwargs) -> Dict[str, Path]:
    """
    Separate stems for an organized audio folder.

    Args:
        audio_folder: Path to organized folder (contains full_mix.flac)
        overwrite: Whether to overwrite existing stems
        **kwargs: Additional arguments passed to separate_stems()

    Returns:
        Dictionary mapping stem names to output file paths

    Raises:
        FileNotFoundError: If folder or full_mix doesn't exist
    """
    audio_folder = Path(audio_folder)

    if not audio_folder.exists():
        raise FileNotFoundError(f"Folder not found: {audio_folder}")

    # Find full_mix file
    full_mix = None
    for ext in AUDIO_EXTENSIONS:
        potential_path = audio_folder / f"full_mix{ext}"
        if potential_path.exists():
            full_mix = potential_path
            break

    if full_mix is None:
        raise FileNotFoundError(f"No full_mix file found in {audio_folder}")

    # Check if stems already exist (check multiple extensions)
    if not overwrite:
        existing_stems = {}
        for stem_name in DEMUCS_STEMS:
            # Check for any audio extension
            for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                stem_path = audio_folder / f"{stem_name}{ext}"
                if stem_path.exists():
                    existing_stems[stem_name] = stem_path
                    break

        if existing_stems:
            logger.info(f"Stems already exist: {list(existing_stems.keys())}. Use --overwrite to regenerate.")
            return existing_stems

    # Separate stems
    return separate_stems(full_mix, audio_folder, **kwargs)


def batch_separate_stems(root_directory: str | Path,
                          overwrite: bool = False,
                          **kwargs) -> Dict[str, any]:
    """
    Batch separate stems for all organized folders in a directory tree.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing stems
        **kwargs: Additional arguments passed to separate_stems()

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch stem separation: {root_directory}")

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

    # Process each folder
    for i, folder in enumerate(folders, 1):
        logger.info(f"Processing {i}/{stats['total']}: {folder.name}")

        try:
            result = separate_organized_folder(folder, overwrite=overwrite, **kwargs)

            if result and any('full_mix' not in str(p) for p in result.values()):
                stats['success'] += 1
            else:
                stats['skipped'] += 1

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Stem Separation Summary:")
    logger.info(f"  Total folders:  {stats['total']}")
    logger.info(f"  Successful:     {stats['success']}")
    logger.info(f"  Skipped:        {stats['skipped']}")
    logger.info(f"  Failed:         {stats['failed']}")
    logger.info("=" * 60)

    return stats


# Command-line interface
if __name__ == "__main__":
    import argparse
    from core.common import setup_logging

    parser = argparse.ArgumentParser(
        description="Separate audio into stems using Demucs"
    )

    parser.add_argument(
        'path',
        type=str,
        help='Path to audio file or organized folder'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process all organized folders in directory tree'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help=f'Demucs model to use (default: {DEMUCS_CONFIG["model"]})'
    )

    parser.add_argument(
        '--shifts',
        type=int,
        default=None,
        help=f'Number of random shifts (default: {DEMUCS_CONFIG["shifts"]})'
    )

    parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=None,
        help=f'Number of parallel jobs (default: {DEMUCS_CONFIG["jobs"]})'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='Device to use for processing (default: cuda)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing stems'
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

    # Check if demucs is installed
    if not check_demucs_installed():
        logger.error("Demucs is not installed. Install with: pip install demucs")
        sys.exit(1)

    path = Path(args.path)

    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)

    try:
        kwargs = {
            'model': args.model,
            'shifts': args.shifts,
            'jobs': args.jobs,
            'device': args.device
        }

        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        if args.batch:
            # Batch processing
            stats = batch_separate_stems(path, overwrite=args.overwrite, **kwargs)

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} folders failed to process")
                sys.exit(1)

        elif path.is_dir():
            # Single folder
            result = separate_organized_folder(path, overwrite=args.overwrite, **kwargs)

            print("\nStem Separation Complete:")
            for stem_name, stem_path in sorted(result.items()):
                print(f"  {stem_name}: {stem_path}")

        else:
            logger.error("Please provide a directory containing organized folders")
            sys.exit(1)

        logger.info("Stem separation completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
