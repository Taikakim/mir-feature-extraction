"""
Pipeline Worker Functions for MIR Project

Parallel processing worker functions for the master pipeline.
These are designed to work with ProcessPoolExecutor for batch operations.

Usage:
    from core.pipeline_workers import process_folder_features, process_folder_crops
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_folder_features, (folder, False)) for folder in folders]
"""

import subprocess
import time
from pathlib import Path
from typing import Tuple


def process_folder_features(args: Tuple[Path, bool]) -> Tuple[str, bool, str]:
    """
    Worker function for parallel feature extraction.

    Args:
        args: Tuple of (folder_path, overwrite)

    Returns:
        Tuple of (folder_name, success, message)
    """
    folder, overwrite = args
    folder_name = folder.name

    # Define output keys for each feature type (must check ALL, not just one)
    LOUDNESS_KEYS = ['lufs', 'lra', 'peak_dbfs', 'true_peak_dbfs']
    SPECTRAL_KEYS = ['spectral_flatness', 'spectral_flux', 'spectral_skewness', 'spectral_kurtosis']

    # Import here to avoid issues with multiprocessing
    from core.file_locks import FileLock
    from core.file_utils import get_stem_files
    from core.json_handler import get_info_path, safe_update, should_process

    # Use file lock to prevent race conditions
    with FileLock(folder) as lock:
        if not lock.acquired:
            return (folder_name, False, "Could not acquire lock")

        try:
            from spectral.spectral_features import analyze_spectral_features
            from timbral.loudness import analyze_file_loudness

            stems = get_stem_files(folder, include_full_mix=True)
            if 'full_mix' not in stems:
                return (folder_name, False, "No full_mix found")

            full_mix = stems['full_mix']
            info_path = get_info_path(full_mix)
            results = {}

            # Loudness - check ALL output keys
            if should_process(info_path, LOUDNESS_KEYS, overwrite):
                try:
                    results.update(analyze_file_loudness(full_mix))
                except Exception:
                    pass  # Non-critical

            # Spectral - check ALL output keys
            if should_process(info_path, SPECTRAL_KEYS, overwrite):
                try:
                    results.update(analyze_spectral_features(full_mix))
                except Exception:
                    pass  # Non-critical

            if results:
                safe_update(info_path, results)
                return (folder_name, True, f"Extracted {len(results)} features")
            else:
                return (folder_name, True, "Already complete")

        except Exception as e:
            return (folder_name, False, str(e))


def process_folder_crops(args: Tuple[Path, Path, int, bool, bool, bool, bool]) -> Tuple[str, int, str]:
    """
    Worker function for parallel crop creation.

    Args:
        args: Tuple of (folder_path, crops_dir, length_samples, sequential, overlap, div4, overwrite)

    Returns:
        Tuple of (folder_name, crop_count, message)
    """
    folder, crops_dir, length_samples, sequential, overlap, div4, overwrite = args
    folder_name = folder.name

    # Import here to avoid issues with multiprocessing
    from core.file_locks import FileLock

    # Use file lock to prevent race conditions
    with FileLock(folder) as lock:
        if not lock.acquired:
            return (folder_name, 0, "Could not acquire lock")

        try:
            from tools.create_training_crops import create_crops_for_file

            count = create_crops_for_file(
                folder,
                length_samples=length_samples,
                output_dir=crops_dir,
                sequential=sequential,
                overlap=overlap,
                div4=div4,
                overwrite=overwrite,
            )
            return (folder_name, count, f"Created {count} crops")

        except Exception as e:
            return (folder_name, 0, str(e))


def process_demucs_subprocess(args: Tuple[Path, Path, str, int, str, int]) -> Tuple[str, bool, float, str]:
    """
    Worker function for parallel Demucs separation via subprocess.

    Each subprocess gets its own GPU context, allowing true parallel processing.

    Args:
        args: Tuple of (audio_path, output_dir, model, shifts, format, bitrate)

    Returns:
        Tuple of (folder_name, success, elapsed_time, message)
    """
    audio_path, output_dir, model, shifts, output_format, bitrate = args
    folder_name = audio_path.parent.name

    start_time = time.time()

    try:
        # Build demucs command
        cmd = [
            'demucs',
            '-n', model,
            '--shifts', str(shifts),
            '-j', '1',  # Single-threaded within each instance
            '--out', str(output_dir),
        ]

        # Add format options
        if output_format == 'mp3':
            cmd.extend(['--mp3', '--mp3-bitrate', str(bitrate)])
        elif output_format == 'flac':
            cmd.append('--flac')

        cmd.append(str(audio_path))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per track
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            return (folder_name, True, elapsed, f"Done in {elapsed:.1f}s")
        else:
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"
            return (folder_name, False, elapsed, error_msg)

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return (folder_name, False, elapsed, "Timeout (>10min)")
    except Exception as e:
        elapsed = time.time() - start_time
        return (folder_name, False, elapsed, str(e))


# Aliases for backward compatibility with underscore-prefixed names
_process_folder_features = process_folder_features
_process_folder_crops = process_folder_crops
_process_demucs_subprocess = process_demucs_subprocess
