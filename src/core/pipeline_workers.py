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


def process_folder_features(args) -> Tuple[str, bool, str]:
    """
    Worker function for parallel feature extraction.

    Args:
        args: Tuple of (folder_path, overwrite, per_feature_overwrite[, skip_flags])
              overwrite: global overwrite flag
              per_feature_overwrite: dict mapping feature names to overwrite bools
              skip_flags: optional dict of skip_<feature>: bool flags

    Returns:
        Tuple of (folder_name, success, message)
    """
    if len(args) == 4:
        folder, overwrite, per_feature_overwrite, skip_flags = args
    else:
        folder, overwrite, per_feature_overwrite = args
        skip_flags = {}
    folder_name = folder.name

    def should_overwrite(feature: str) -> bool:
        return overwrite or per_feature_overwrite.get(feature, False)

    # Define output keys for each feature type (must check ALL, not just one)
    LOUDNESS_KEYS = ['lufs', 'lra']
    SPECTRAL_KEYS = ['spectral_flatness', 'spectral_flux', 'spectral_skewness', 'spectral_kurtosis']
    SATURATION_KEYS = ['saturation_ratio', 'saturation_count']
    SYNCOPATION_KEYS = ['syncopation', 'on_beat_ratio']
    COMPLEXITY_KEYS = ['rhythmic_complexity', 'rhythmic_evenness']
    PER_STEM_RHYTHM_KEYS = ['onset_density_average_bass', 'onset_density_average_other']
    PER_STEM_HARMONIC_KEYS = ['harmonic_movement_bass', 'harmonic_movement_other']
    BPM_KEYS = ['bpm_madmom', 'bpm_essentia']

    # Import here to avoid issues with multiprocessing
    from core.file_locks import FileLock
    from core.file_utils import get_stem_files
    from core.json_handler import get_info_path, read_info, safe_update, should_process

    # Use file lock to prevent race conditions
    with FileLock(folder) as lock:
        if not lock.acquired:
            return (folder_name, False, "Could not acquire lock")

        try:
            from spectral.spectral_features import analyze_spectral_features
            from spectral.saturation import analyze_saturation
            from timbral.loudness import analyze_file_loudness
            from rhythm.syncopation import analyze_syncopation
            from rhythm.complexity import analyze_rhythmic_complexity
            from rhythm.per_stem_rhythm import analyze_per_stem_rhythm
            from harmonic.per_stem_harmonic import analyze_per_stem_harmonics

            stems = get_stem_files(folder, include_full_mix=True)
            if 'full_mix' not in stems:
                return (folder_name, False, "No full_mix found")

            full_mix = stems['full_mix']
            info_path = get_info_path(full_mix)
            # Read .INFO once; pass to all should_process() calls to avoid
            # repeated HDD seeks on the same file.
            existing = read_info(info_path) if info_path.exists() else {}
            results = {}

            # Loudness - check ALL output keys
            if not skip_flags.get('skip_loudness') and should_process(info_path, LOUDNESS_KEYS, should_overwrite('loudness'), existing=existing):
                try:
                    results.update(analyze_file_loudness(full_mix))
                except Exception:
                    pass  # Non-critical

            # Spectral - check ALL output keys
            if not skip_flags.get('skip_spectral') and should_process(info_path, SPECTRAL_KEYS, should_overwrite('spectral'), existing=existing):
                try:
                    results.update(analyze_spectral_features(full_mix))
                except Exception:
                    pass  # Non-critical

            # Saturation / hard-clipping detection
            if not skip_flags.get('skip_saturation') and should_process(info_path, SATURATION_KEYS, should_overwrite('saturation'), existing=existing):
                try:
                    results.update(analyze_saturation(full_mix))
                except Exception:
                    pass  # Non-critical

            # Syncopation
            if not skip_flags.get('skip_syncopation') and should_process(info_path, SYNCOPATION_KEYS, should_overwrite('syncopation'), existing=existing):
                try:
                    results.update(analyze_syncopation(folder))
                except FileNotFoundError:
                    pass  # Normal if beats/onsets aren't available
                except Exception:
                    pass

            # Complexity
            if not skip_flags.get('skip_complexity') and should_process(info_path, COMPLEXITY_KEYS, should_overwrite('complexity'), existing=existing):
                try:
                    results.update(analyze_rhythmic_complexity(folder))
                except FileNotFoundError:
                    pass
                except Exception:
                    pass

            # Per Stem Rhythm
            if not skip_flags.get('skip_per_stem_rhythm') and should_process(info_path, PER_STEM_RHYTHM_KEYS, should_overwrite('per_stem_rhythm'), existing=existing):
                try:
                    res = analyze_per_stem_rhythm(folder)
                    if res: results.update(res)
                except Exception:
                    pass

            # Per Stem Harmonic
            if not skip_flags.get('skip_per_stem_harmonic') and should_process(info_path, PER_STEM_HARMONIC_KEYS, should_overwrite('per_stem_harmonic'), existing=existing):
                try:
                    res = analyze_per_stem_harmonics(folder)
                    if res: results.update(res)
                except Exception:
                    pass

            # BPM — Madmom TempoEstimationProcessor + Essentia RhythmExtractor2013
            if should_process(info_path, BPM_KEYS, should_overwrite('bpm'), existing=existing):
                try:
                    from rhythm.bpm import estimate_bpm_madmom, estimate_bpm_essentia
                    results['bpm_madmom'] = estimate_bpm_madmom(full_mix)
                    results['bpm_essentia'] = estimate_bpm_essentia(full_mix)
                except Exception:
                    pass

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


def process_folder_onsets(args: Tuple[Path, bool]) -> Tuple[str, str, int, str]:
    """
    Worker function for parallel onset detection.

    Detects onsets in a track's full_mix, writes a .ONSETS timestamp file
    alongside the audio, and merges onset statistics into the .INFO file.

    Args:
        args: Tuple of (folder_path, overwrite)

    Returns:
        Tuple of (folder_name, status, onset_count, message)
        status: 'success' | 'skipped' | 'failed'
    """
    folder, overwrite = args
    folder_name = folder.name

    from core.file_locks import FileLock

    with FileLock(folder) as lock:
        if not lock.acquired:
            return (folder_name, 'failed', 0, 'Could not acquire lock')

        try:
            from core.file_utils import get_stem_files
            from core.json_handler import get_info_path, safe_update
            from rhythm.onsets import analyze_onsets_with_save

            onsets_file = folder / f"{folder_name}.ONSETS"

            # Skip if .ONSETS file already exists and not overwriting
            if onsets_file.exists() and not overwrite:
                return (folder_name, 'skipped', 0, '')

            stems = get_stem_files(folder, include_full_mix=True)
            if 'full_mix' not in stems:
                return (folder_name, 'failed', 0, 'No full_mix found')

            results, _ = analyze_onsets_with_save(stems['full_mix'], save_onsets=True)

            info_path = get_info_path(stems['full_mix'])
            safe_update(info_path, results)

            return (folder_name, 'success', results.get('onset_count', 0), '')

        except Exception as e:
            return (folder_name, 'failed', 0, str(e))


def process_folder_rhythm(args: Tuple[Path, bool]) -> Tuple[str, str, str]:
    """
    Combined worker: beat grid + onset detection for one track folder.

    Runs whatever is still missing (.BEATS_GRID and/or .ONSETS).
    Loads audio once with librosa for onset detection; madmom (used for beat
    detection) reads the file independently via its own reader.

    Args:
        args: Tuple of (folder_path, overwrite)

    Returns:
        Tuple of (folder_name, status, message)
        status: 'success' | 'skipped' | 'failed'
    """
    folder, overwrite = args
    folder_name = folder.name

    from core.file_locks import FileLock

    with FileLock(folder) as lock:
        if not lock.acquired:
            return (folder_name, 'failed', 'Could not acquire lock')

        try:
            from core.file_utils import get_stem_files
            from core.json_handler import get_info_path, safe_update

            beats_file = folder / f"{folder_name}.BEATS_GRID"
            onsets_file = folder / f"{folder_name}.ONSETS"
            beats_needed = not beats_file.exists() or overwrite
            onsets_needed = not onsets_file.exists() or overwrite

            if not beats_needed and not onsets_needed:
                return (folder_name, 'skipped', '')

            stems = get_stem_files(folder, include_full_mix=True)
            if 'full_mix' not in stems:
                return (folder_name, 'failed', 'No full_mix found')

            full_mix = stems['full_mix']
            info_path = get_info_path(full_mix)
            results = {}

            # Beat detection — madmom reads the file via its own reader
            if beats_needed:
                from rhythm.beat_grid import create_beat_grid
                create_beat_grid(full_mix, save_grid=True)

            # Onset detection — load audio once with librosa, reuse for analysis
            if onsets_needed:
                import librosa
                from core.common import clamp_feature_value
                from rhythm.onsets import (detect_onsets, calculate_onset_density,
                                           analyze_onset_statistics, save_onset_times)

                audio, sr = librosa.load(str(full_mix), sr=None, mono=True)
                duration = len(audio) / sr
                onset_times, onset_strengths = detect_onsets(audio, sr)

                onset_density = calculate_onset_density(onset_times, duration)
                strength_stats = analyze_onset_statistics(onset_strengths)

                onset_results = {
                    'onset_count': int(len(onset_times)),
                    'onset_density': float(onset_density),
                    **strength_stats,
                }
                for k, v in onset_results.items():
                    onset_results[k] = clamp_feature_value(k, v)

                save_onset_times(onset_times, onsets_file)
                results.update(onset_results)

            if results:
                safe_update(info_path, results)

            done = []
            if beats_needed: done.append('beats')
            if onsets_needed: done.append('onsets')
            return (folder_name, 'success', '+'.join(done))

        except Exception as e:
            return (folder_name, 'failed', str(e))


# Aliases for backward compatibility with underscore-prefixed names
_process_folder_features = process_folder_features
_process_folder_crops = process_folder_crops
_process_demucs_subprocess = process_demucs_subprocess
