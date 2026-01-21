"""
Demucs Stem Separation for MIR Project

This module wraps Demucs HT v4 for source separation.
Separates audio into: drums, bass, other, vocals

Dependencies:
- demucs
- soundfile (for OGG conversion and fallback saving)
- src.core.file_utils
- src.core.common

Output formats supported:
- FLAC (lossless, default)
- MP3 (lossy, CBR or VBR)
- OGG (lossy, via post-conversion)
- WAV (16-bit, 24-bit, or 32-bit float)

Configuration (from MIR plan):
- Model: htdemucs (HT v4)
- Shifts: 1
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


# Output format configurations
OUTPUT_FORMATS = {
    'flac': {'ext': '.flac', 'demucs_flag': '--flac'},
    'mp3': {'ext': '.mp3', 'demucs_flag': '--mp3'},
    'wav': {'ext': '.wav', 'demucs_flag': None},  # Default
    'wav24': {'ext': '.wav', 'demucs_flag': '--int24'},
    'wav32': {'ext': '.wav', 'demucs_flag': '--float32'},
    'ogg': {'ext': '.ogg', 'demucs_flag': None, 'post_convert': True},
}


def check_demucs_installed() -> bool:
    """
    Check if demucs is installed and available.

    Returns:
        True if demucs is available, False otherwise
    """
    return shutil.which('demucs') is not None


def convert_to_ogg(input_path: Path, output_path: Path, quality: float = 0.5) -> bool:
    """
    Convert audio file to OGG Vorbis using soundfile.

    Args:
        input_path: Path to input audio file
        output_path: Path for output OGG file
        quality: OGG quality (0.0-1.0, default 0.5 ~ 160kbps)

    Returns:
        True if successful, False otherwise
    """
    try:
        import soundfile as sf
        import numpy as np

        # Read input file
        data, sr = sf.read(str(input_path))

        # Write as OGG
        sf.write(str(output_path), data, sr, format='OGG', subtype='VORBIS')

        logger.debug(f"Converted {input_path.name} to OGG")
        return True

    except Exception as e:
        logger.error(f"Failed to convert to OGG: {e}")
        return False


def convert_wav_to_flac_soundfile(input_path: Path, output_path: Path) -> bool:
    """
    Convert WAV to FLAC using soundfile (fallback for torchcodec issues).

    Args:
        input_path: Path to input WAV file
        output_path: Path for output FLAC file

    Returns:
        True if successful, False otherwise
    """
    try:
        import soundfile as sf

        # Read WAV
        data, sr = sf.read(str(input_path))

        # Write as FLAC
        sf.write(str(output_path), data, sr, format='FLAC')

        logger.debug(f"Converted {input_path.name} to FLAC via soundfile")
        return True

    except Exception as e:
        logger.error(f"Failed to convert to FLAC: {e}")
        return False


def separate_stems(audio_file: str | Path,
                   output_dir: str | Path,
                   model: str = None,
                   shifts: int = None,
                   jobs: int = None,
                   device: str = 'cuda',
                   output_format: str = 'flac',
                   mp3_bitrate: int = 320,
                   mp3_preset: int = 2,
                   ogg_quality: float = 0.5,
                   clip_mode: str = 'rescale') -> Dict[str, Path]:
    """
    Separate an audio file into stems using Demucs.

    Args:
        audio_file: Path to input audio file
        output_dir: Directory where stems will be saved
        model: Demucs model to use (default: from DEMUCS_CONFIG)
        shifts: Number of random shifts for prediction (default: from config)
        jobs: Number of parallel jobs (default: from config)
        device: Device to use ('cuda', 'cpu', 'mps')
        output_format: Output format - 'flac', 'mp3', 'ogg', 'wav', 'wav24', 'wav32'
        mp3_bitrate: Bitrate for MP3 output (64-320, default 320)
        mp3_preset: VBR preset for MP3 (2-7, 2=best quality, 7=fastest)
        ogg_quality: Quality for OGG output (0.0-1.0, default 0.5)
        clip_mode: Clipping strategy - 'rescale' or 'clamp'

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

    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {output_format}. "
                        f"Supported: {list(OUTPUT_FORMATS.keys())}")

    # Use defaults from config if not specified
    if model is None:
        model = DEMUCS_CONFIG['model']
    if shifts is None:
        shifts = DEMUCS_CONFIG['shifts']
    if jobs is None:
        jobs = DEMUCS_CONFIG['jobs']

    format_config = OUTPUT_FORMATS[output_format]
    final_ext = format_config['ext']
    needs_post_convert = format_config.get('post_convert', False)

    logger.info(f"Separating stems: {audio_file.name}")
    logger.info(f"Model: {model}, Shifts: {shifts}, Device: {device}, Format: {output_format}")

    # Build demucs command
    cmd = [
        'demucs',
        '-n', model,
        '--shifts', str(shifts),
        '-j', str(jobs),
        '--out', str(output_dir),
        '--clip-mode', clip_mode,
        '-d', device,
    ]

    # Determine demucs output format
    # Strategy: Always use WAV for FLAC/OGG to bypass torchcodec issues entirely
    # MP3 uses lameenc (reliable), WAV variants are native
    if output_format == 'mp3':
        # MP3 uses lameenc encoder (reliable, no torchcodec)
        demucs_ext = '.mp3'
        cmd.extend(['--mp3', '--mp3-bitrate', str(mp3_bitrate)])
        if mp3_preset != 2:  # Only add if not default
            cmd.extend(['--mp3-preset', str(mp3_preset)])
        cmd.extend(['--filename', '{stem}.mp3'])
    elif output_format in ('flac', 'ogg'):
        # Output WAV, convert to FLAC/OGG with soundfile (bypasses torchcodec)
        demucs_ext = '.wav'
        cmd.extend(['--float32', '--filename', '{stem}.wav'])  # 32-bit for quality
        needs_post_convert = True
    elif output_format == 'wav24':
        demucs_ext = '.wav'
        cmd.extend(['--int24', '--filename', '{stem}.wav'])
    elif output_format == 'wav32':
        demucs_ext = '.wav'
        cmd.extend(['--float32', '--filename', '{stem}.wav'])
    else:  # wav (16-bit default)
        demucs_ext = '.wav'
        cmd.extend(['--filename', '{stem}.wav'])

    cmd.append(str(audio_file))

    logger.debug(f"Running: {' '.join(cmd)}")

    # Run demucs
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True
    )

    logger.debug(f"Demucs stdout: {result.stdout}")
    if result.stderr:
        logger.debug(f"Demucs stderr: {result.stderr}")

    # Find and move output files
    demucs_output = output_dir / model
    stem_paths = {}

    if demucs_output.exists():
        for stem_name in DEMUCS_STEMS:
            stem_file = demucs_output / f"{stem_name}{demucs_ext}"
            if stem_file.exists():
                final_path = output_dir / f"{stem_name}{final_ext}"

                # Post-conversion for FLAC/OGG (using soundfile)
                if needs_post_convert:
                    if output_format == 'flac':
                        if convert_wav_to_flac_soundfile(stem_file, final_path):
                            stem_file.unlink()  # Remove WAV
                            stem_paths[stem_name] = final_path
                            logger.info(f"  Created: {stem_name}.flac")
                        else:
                            # Keep WAV as fallback
                            wav_final = output_dir / f"{stem_name}.wav"
                            shutil.move(str(stem_file), str(wav_final))
                            stem_paths[stem_name] = wav_final
                            logger.warning(f"  FLAC conversion failed, kept as: {stem_name}.wav")
                    elif output_format == 'ogg':
                        if convert_to_ogg(stem_file, final_path, ogg_quality):
                            stem_file.unlink()  # Remove WAV
                            stem_paths[stem_name] = final_path
                            logger.info(f"  Created: {stem_name}.ogg")
                        else:
                            # Keep WAV as fallback
                            wav_final = output_dir / f"{stem_name}.wav"
                            shutil.move(str(stem_file), str(wav_final))
                            stem_paths[stem_name] = wav_final
                            logger.warning(f"  OGG conversion failed, kept as: {stem_name}.wav")
                else:
                    # Direct move (MP3, WAV variants)
                    shutil.move(str(stem_file), str(final_path))
                    stem_paths[stem_name] = final_path
                    logger.info(f"  Created: {stem_name}{final_ext}")

    # Clean up demucs output directory structure
    try:
        shutil.rmtree(output_dir / model)
    except:
        pass

    if not stem_paths:
        raise RuntimeError("No stem files were created by demucs")

    logger.info(f"Successfully separated {len(stem_paths)} stems")
    return stem_paths


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
            for ext in ['.flac', '.mp3', '.ogg', '.wav', '.m4a']:
                stem_path = audio_folder / f"{stem_name}{ext}"
                if stem_path.exists():
                    existing_stems[stem_name] = stem_path
                    break

        if len(existing_stems) == len(DEMUCS_STEMS):
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
        description="Separate audio into stems using Demucs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output format examples:
  --format flac              Lossless FLAC (default)
  --format mp3 --bitrate 320 High quality MP3 CBR
  --format mp3 --preset 2    High quality MP3 VBR
  --format ogg --ogg-quality 0.6  OGG Vorbis (~192kbps)
  --format wav               16-bit WAV
  --format wav24             24-bit WAV
  --format wav32             32-bit float WAV
"""
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

    # Output format options
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='flac',
        choices=['flac', 'mp3', 'ogg', 'wav', 'wav24', 'wav32'],
        help='Output format (default: flac)'
    )

    parser.add_argument(
        '--bitrate',
        type=int,
        default=320,
        help='MP3 bitrate in kbps for CBR mode (64-320, default: 320)'
    )

    parser.add_argument(
        '--preset',
        type=int,
        default=2,
        choices=[2, 3, 4, 5, 6, 7],
        help='MP3 VBR preset (2=best quality, 7=fastest, default: 2)'
    )

    parser.add_argument(
        '--ogg-quality',
        type=float,
        default=0.5,
        help='OGG Vorbis quality (0.0-1.0, default: 0.5 ~ 160kbps)'
    )

    parser.add_argument(
        '--clip-mode',
        type=str,
        default='rescale',
        choices=['rescale', 'clamp'],
        help='Clipping strategy (default: rescale)'
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
            'device': args.device,
            'output_format': args.format,
            'mp3_bitrate': args.bitrate,
            'mp3_preset': args.preset,
            'ogg_quality': args.ogg_quality,
            'clip_mode': args.clip_mode,
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
