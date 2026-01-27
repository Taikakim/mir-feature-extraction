"""
Tool to create training crops for Stable Audio Tools training.

Usage:
    # Sequential mode (fixed sample length, no beat alignment)
    python src/tools/create_training_crops.py /path/to/data --length 2097152 --sequential

    # Beat-aligned mode
    python src/tools/create_training_crops.py /path/to/data --length 2097152
    python src/tools/create_training_crops.py /path/to/data --length 2097152 --overlap
    python src/tools/create_training_crops.py /path/to/data --length 2097152 --overlap --div4

    # With output directory (creates per-track folders)
    python src/tools/create_training_crops.py /path/to/data --output-dir /path/to/crops --sequential
    python src/tools/create_training_crops.py /path/to/data -o /path/to/crops --overlap

Features:
- --output-dir / -o: Save crops to destination with per-track folders (e.g., /output/TrackName/)
- --sequential: Simple sequential crops at exact sample length (no beat logic)
- Beat-aligned mode: Start times snap to closest downbeat
- End times snap BACKWARDS to last downbeat before target end (never exceeds length)
- When --div4: ensures each crop contains downbeats divisible by 4
- When --overlap: next crop starts at (last_start + length/2), snapped to closest downbeat
- 10ms fade-in and fade-out for clean transitions
- First crop starts without zero-crossing snap (preserves exact position)
- Silence detection at -72dB threshold
- Creates .INFO file for each crop with position metadata
"""

import argparse
import logging
import json
import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import find_organized_folders, get_stem_files
from core.common import setup_logging
from rhythm.beat_grid import load_beat_grid
from core.json_handler import get_info_path, read_info, safe_update
from core.file_locks import FileLock
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Features to check for in source .INFO and transfer to crop .INFO
TRANSFERRABLE_FEATURES = [
    # Core Metadata (from track_metadata_lookup.py)
    'release_year',
    'release_date',
    'artists',
    'label',
    'genres',
    'popularity',
    'spotify_id',
    'musicbrainz_id',
    
    # Music Flamingo descriptions
    'music_flamingo_full',
    'music_flamingo_genre_mood',
    'music_flamingo_instrumentation',
    'music_flamingo_technical',
    'music_flamingo_structure',
    
    # Optional Spotify audio features
    'spotify_acousticness',
    'spotify_energy',
    'spotify_instrumentalness',
    'spotify_time_signature',
    'spotify_valence',
    'spotify_danceability',
    'spotify_speechiness',
    'spotify_liveness',
    'spotify_key',
    'spotify_mode',
    'spotify_tempo',
]

# Features that are good to have but don't warn if missing
OPTIONAL_TRANSFERRABLE = [
    'bpm',
    'beat_count',
]

# Demucs stem names to crop along with full_mix
STEM_NAMES = ['drums', 'bass', 'other', 'vocals']


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate using scipy.

    Args:
        audio: Audio array of shape (samples,) or (samples, channels)
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    # Calculate resampling ratio
    ratio = target_sr / orig_sr
    new_length = int(audio.shape[0] * ratio)

    if audio.ndim == 1:
        return signal.resample(audio, new_length)
    else:
        # Resample each channel
        resampled = np.zeros((new_length, audio.shape[1]), dtype=audio.dtype)
        for ch in range(audio.shape[1]):
            resampled[:, ch] = signal.resample(audio[:, ch], new_length)
        return resampled


def crop_stem_file(stem_path: Path, crop_base_path: Path, stem_name: str,
                   start_sample: int, end_sample: int, sr: int,
                   fade_len: int) -> Optional[Path]:
    """
    Crop a stem file at the same sample positions as the full mix crop.

    Args:
        stem_path: Path to source stem (e.g., drums.flac)
        crop_base_path: Base path for output (e.g., TrackName_0.flac)
        stem_name: Name of stem (drums, bass, other, vocals)
        start_sample: Start sample index
        end_sample: End sample index
        sr: Sample rate (target)
        fade_len: Fade length in samples

    Returns:
        Path to cropped stem, or None if failed
    """
    if not stem_path.exists():
        return None

    try:
        # Load stem audio
        stem_audio, stem_sr = sf.read(str(stem_path))

        # Resample if needed (e.g., 32kHz MP3 stems to 44.1kHz)
        if stem_sr != sr:
            logger.debug(f"Resampling {stem_name}: {stem_sr} -> {sr} Hz")
            stem_audio = resample_audio(stem_audio, stem_sr, sr)

        # Ensure shape is (samples, channels)
        if stem_audio.ndim == 1:
            stem_audio = stem_audio[:, np.newaxis]
        elif stem_audio.shape[0] < stem_audio.shape[1]:
            stem_audio = stem_audio.T

        # Check bounds
        if end_sample > stem_audio.shape[0]:
            logger.warning(f"Stem {stem_name} too short: {stem_audio.shape[0]} < {end_sample}")
            return None

        # Extract crop
        stem_crop = stem_audio[start_sample:end_sample].copy()

        # Apply fades
        stem_crop = apply_fades(stem_crop, fade_len)

        # Save with prefixed name: TrackName_0_drums.flac
        crop_stem_name = f"{crop_base_path.stem}_{stem_name}{crop_base_path.suffix}"
        crop_stem_path = crop_base_path.parent / crop_stem_name
        sf.write(str(crop_stem_path), stem_crop, sr)

        return crop_stem_path

    except Exception as e:
        logger.warning(f"Failed to crop stem {stem_name}: {e}")
        return None


def crop_all_stems(folder_path: Path, crop_base_path: Path,
                   start_sample: int, end_sample: int, sr: int,
                   fade_len: int) -> Dict[str, Path]:
    """
    Crop all available stems at the same positions as the full mix.

    Returns:
        Dict mapping stem names to cropped paths
    """
    cropped_stems = {}

    for stem_name in STEM_NAMES:
        # Find stem file (check multiple extensions)
        stem_path = None
        for ext in ['.flac', '.wav', '.mp3']:
            potential = folder_path / f"{stem_name}{ext}"
            if potential.exists():
                stem_path = potential
                break

        if stem_path is None:
            continue

        cropped = crop_stem_file(
            stem_path, crop_base_path, stem_name,
            start_sample, end_sample, sr, fade_len
        )
        if cropped:
            cropped_stems[stem_name] = cropped

    return cropped_stems


def slice_rhythm_file(source_path: Path, dest_path: Path,
                      start_time: float, end_time: float):
    """
    Read timestamps from source_path, filter those within [start_time, end_time],
    shift them by -start_time, and write to dest_path.
    """
    if not source_path.exists():
        return
        
    try:
        # Read timestamps (handle both single-column and multi-column if needed)
        with open(source_path, 'r') as f:
            lines = f.readlines()
            
        valid_lines = []
        for line in lines:
            if not line.strip(): continue
            try:
                # Assume first token is timestamp
                parts = line.strip().split()
                if not parts: continue
                t = float(parts[0])
                
                if start_time <= t <= end_time:
                    # Shift timestamp
                    new_t = t - start_time
                    # Reconstruct line with new timestamp
                    if len(parts) > 1:
                        new_line = f"{new_t:.6f} {' '.join(parts[1:])}"
                    else:
                        new_line = f"{new_t:.6f}"
                    valid_lines.append(new_line)
            except ValueError:
                continue
                
        # Write only if we have data (or write empty file? Empty is fine)
        with open(dest_path, 'w') as f:
            f.write('\n'.join(valid_lines))
            if valid_lines: f.write('\n')
            
    except Exception as e:
        logger.warning(f"Failed to slice rhythm file {source_path.name}: {e}")


def get_start_offset_above_threshold(audio: np.ndarray,
                                     threshold_db: float = -72.0,
                                     window_size_samples: int = 4410) -> int:
    """
    Find the sample index where audio first rises above threshold_db.
    Uses -72dB as default threshold for detecting silence.
    """
    threshold_linear = 10 ** (threshold_db / 20)

    num_samples = audio.shape[0]

    # If stereo, mix to mono for detection
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio.flatten()

    for i in range(0, num_samples, window_size_samples):
        chunk = mono[i: i + window_size_samples]
        if np.max(np.abs(chunk)) > threshold_linear:
            return i

    return 0


def find_downbeat_before(downbeat_times: np.ndarray, target_time: float) -> Optional[float]:
    """Find the latest downbeat that is <= target_time (snap backwards)."""
    if downbeat_times is None or len(downbeat_times) == 0:
        return None

    candidates = downbeat_times[downbeat_times <= target_time + 0.001]
    if len(candidates) == 0:
        return None

    return candidates[-1]


def find_closest_downbeat(downbeat_times: np.ndarray, target_time: float) -> Optional[float]:
    """Find the closest downbeat to target_time (for start alignment)."""
    if downbeat_times is None or len(downbeat_times) == 0:
        return None

    idx = (np.abs(downbeat_times - target_time)).argmin()
    return downbeat_times[idx]


def find_zero_crossing_backwards(audio: np.ndarray, target_idx: int, search_window: int = 2205) -> int:
    """
    Find the first zero crossing BACKWARDS from target_idx.
    Returns the zero crossing index, or target_idx if none found.
    """
    start = max(0, target_idx - search_window)
    end = target_idx

    if start >= end:
        return target_idx

    window = audio[start:end]
    zero_crossings = np.where(np.diff(np.signbit(window)))[0]

    if len(zero_crossings) == 0:
        return target_idx

    zero_crossings = zero_crossings + start
    return zero_crossings[-1]


def count_downbeats_in_range(downbeat_times: np.ndarray, start_sec: float, end_sec: float) -> int:
    """Count how many downbeats fall within [start_sec, end_sec]."""
    if downbeat_times is None or len(downbeat_times) == 0:
        return 0
    mask = (downbeat_times >= start_sec - 0.001) & (downbeat_times <= end_sec + 0.001)
    return int(np.sum(mask))


def find_end_for_div4_downbeats(downbeat_times: np.ndarray, start_sec: float,
                                 target_end_sec: float, min_downbeats: int = 4) -> Optional[float]:
    """
    Find an end time such that the number of downbeats in [start, end] is divisible by 4.
    Searches backwards from target_end_sec.
    """
    if downbeat_times is None or len(downbeat_times) == 0:
        return None

    candidates = downbeat_times[downbeat_times > start_sec + 0.001]
    candidates = candidates[candidates <= target_end_sec + 0.001]

    if len(candidates) < min_downbeats:
        return None

    for i in range(len(candidates) - 1, -1, -1):
        end_time = candidates[i]
        count = count_downbeats_in_range(downbeat_times, start_sec, end_time)
        if count >= min_downbeats and count % 4 == 0:
            return end_time

    return None


def apply_fades(crop_audio: np.ndarray, fade_len: int) -> np.ndarray:
    """Apply 10ms fade-in and fade-out to crop."""
    if crop_audio.shape[0] <= fade_len * 2:
        return crop_audio

    # Fade in
    fade_in_curve = np.linspace(0, 1, fade_len)
    if crop_audio.ndim > 1:
        crop_audio[:fade_len, :] *= fade_in_curve[:, np.newaxis]
    else:
        crop_audio[:fade_len] *= fade_in_curve

    # Fade out
    fade_out_curve = np.linspace(1, 0, fade_len)
    if crop_audio.ndim > 1:
        crop_audio[-fade_len:, :] *= fade_out_curve[:, np.newaxis]
    else:
        crop_audio[-fade_len:] *= fade_out_curve

    return crop_audio


def create_sequential_crops(folder_path: Path, length_samples: int, sr: int,
                            audio: np.ndarray, full_mix_path: Path,
                            output_dir: Optional[Path] = None) -> int:
    """
    Create simple sequential crops at fixed sample length.
    No beat alignment, just exact sample boundaries with fades.
    """
    total_samples = audio.shape[0]
    duration_sec = total_samples / sr
    fade_len = int(0.01 * sr)  # 10ms fade

    # Find start after silence (-72dB threshold)
    start_sample = get_start_offset_above_threshold(audio, threshold_db=-72.0)
    if start_sample > 0:
        logger.info(f"Skipping initial silence: {start_sample / sr:.2f}s ({start_sample} samples)")

    # Determine output directory
    if output_dir:
        # Create per-track folder in output directory
        track_name = folder_path.name
        crops_dir = output_dir / track_name
    else:
        crops_dir = folder_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    crop_count = 0
    current_sample = start_sample

    while current_sample + length_samples <= total_samples:
        end_sample = current_sample + length_samples

        # Extract crop (no ZC snap for sequential mode)
        crop_audio = audio[current_sample:end_sample].copy()

        # Apply fades
        crop_audio = apply_fades(crop_audio, fade_len)

        # Save
        # Use parent folder name (Track Name) for consistent naming
        track_name = full_mix_path.parent.name
        crop_name = f"{track_name}_{crop_count}.flac"
        crop_path = crops_dir / crop_name
        sf.write(str(crop_path), crop_audio, sr)

        # Crop stems at the same positions
        cropped_stems = crop_all_stems(
            folder_path, crop_path,
            current_sample, end_sample, sr, fade_len
        )
        if cropped_stems:
            logger.debug(f"Cropped {len(cropped_stems)} stems for crop {crop_count}")

        # Metadata
        start_sec = current_sample / sr
        end_sec = end_sample / sr
        position = start_sec / duration_sec

        # Slice rhythm files (beats, downbeats, onsets)
        rhythm_suffixes = ['.BEATS_GRID', '.ONSETS', '.DOWNBEATS']
        for suffix in rhythm_suffixes:
            candidates = list(folder_path.glob(f"*{suffix}"))
            if candidates:
                source_file = candidates[0]
                dest_file = crop_path.with_suffix(suffix)
                slice_rhythm_file(source_file, dest_file, start_sec, end_sec)

        meta = {
            "position": position,
            "start_time": start_sec,
            "end_time": end_sec,
            "start_sample": current_sample,
            "end_sample": end_sample,
            "duration": end_sec - start_sec,
            "samples": length_samples,
            "source": str(full_mix_path.name),
            "has_stems": len(cropped_stems) > 0,
            "stem_names": list(cropped_stems.keys())
        }

        # Save JSON metadata
        with open(crop_path.with_suffix('.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        # Create .INFO file with position key
        info_path = get_info_path(crop_path)
        safe_update(info_path, {"position": position})

        crop_count += 1
        current_sample = end_sample  # Sequential: next starts where this ended

    return crop_count


def create_crops_for_file(folder_path: Path,
                          length_samples: int,
                          overlap: bool = True,
                          div4: bool = False,
                          sequential: bool = False,
                          output_dir: Optional[Path] = None,
                          overwrite: bool = False) -> int:
    """
    Generate crops for a single folder.

    Args:
        folder_path: Path to organized folder
        length_samples: Target crop length in samples
        overlap: If True, next crop starts at last_start + length/2
        div4: If True, ensure each crop contains downbeats divisible by 4
        sequential: If True, use simple sequential mode (no beat alignment)
        output_dir: Optional output directory for crops
        overwrite: If False, skip folders that already have crops
    """
    stems = get_stem_files(folder_path, include_full_mix=True)
    if 'full_mix' not in stems:
        logger.warning(f"No full_mix in {folder_path.name}")
        return 0

    # Determine crops output directory
    if output_dir:
        track_name = folder_path.name
        crops_dir = output_dir / track_name
    else:
        crops_dir = folder_path / "crops"

    # Check if crops already exist (skip if not overwriting)
    if not overwrite and crops_dir.exists():
        existing_crops = list(crops_dir.glob("*_[0-9].flac")) + list(crops_dir.glob("*_[0-9][0-9].flac"))
        if existing_crops:
            logger.info(f"Crops already exist ({len(existing_crops)} files): {folder_path.name}. Use --overwrite to regenerate.")
            return 0

    full_mix_path = stems['full_mix']

    # Load audio
    try:
        audio, sr = sf.read(str(full_mix_path))
    except Exception as e:
        logger.error(f"Failed to load {full_mix_path}: {e}")
        return 0

    # Ensure shape is (samples, channels)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    elif audio.shape[0] < audio.shape[1]:
        audio = audio.T

    total_samples = audio.shape[0]
    duration_sec = total_samples / sr
    length_sec = length_samples / sr

    logger.info(f"{folder_path.name}: {total_samples} samples, {duration_sec:.2f}s, sr={sr}")

    # Sequential mode: simple fixed-length crops
    if sequential:
        return create_sequential_crops(folder_path, length_samples, sr, audio, full_mix_path, output_dir)

    # Beat-aligned mode - load beat grid (BPM not needed when grids are available)
    beat_times = None
    downbeat_times = None

    grid_path = folder_path / f"{folder_path.name}.BEATS_GRID"
    if grid_path.exists():
        beat_times = load_beat_grid(grid_path)

        downbeat_path = folder_path / f"{folder_path.name}.DOWNBEATS"
        if downbeat_path.exists():
            try:
                downbeat_times = np.loadtxt(downbeat_path)
                if downbeat_times.ndim == 0:
                    downbeat_times = np.array([downbeat_times])
                logger.info(f"{folder_path.name}: Loaded {len(downbeat_times)} downbeats.")
            except Exception as e:
                logger.warning(f"Failed to load downbeats: {e}")

        if downbeat_times is None or len(downbeat_times) < 2:
            if len(beat_times) >= 4:
                downbeat_times = beat_times[::4]
                logger.info(f"{folder_path.name}: Inferred {len(downbeat_times)} downbeats.")
            else:
                downbeat_times = beat_times
    else:
        logger.warning(f"No beat grid for {folder_path.name}, using sequential fallback.")
        return create_sequential_crops(folder_path, length_samples, sr, audio, full_mix_path, output_dir)

    # Get BPM from INFO file or calculate from beat times
    info_path = get_info_path(full_mix_path)
    info_data = read_info(info_path)
    bpm = info_data.get('bpm', 0)

    # If BPM not in INFO, calculate from beat times
    if not bpm and beat_times is not None and len(beat_times) > 1:
        ibis = np.diff(beat_times)
        mean_ibi = np.median(ibis)  # Use median to avoid outliers
        if mean_ibi > 0:
            bpm = 60.0 / mean_ibi

    # Find start after silence (-72dB threshold)
    start_offset_samples = get_start_offset_above_threshold(audio, threshold_db=-72.0)
    start_offset_sec = start_offset_samples / sr
    if start_offset_samples > 0:
        logger.info(f"Skipping initial silence: {start_offset_sec:.2f}s")

    # Determine output directory
    if output_dir:
        # Create per-track folder in output directory
        track_name = folder_path.name
        crops_dir = output_dir / track_name
    else:
        crops_dir = folder_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    crop_count = 0
    align_grid = downbeat_times if (downbeat_times is not None and len(downbeat_times) > 0) else beat_times
    global_bpm = float(bpm) if bpm else 0.0

    # Initial start: snap to closest downbeat after silence
    aligned_start = find_closest_downbeat(align_grid, start_offset_sec)
    if aligned_start is not None:
        current_start_sec = aligned_start
    else:
        current_start_sec = start_offset_sec

    zc_window = int(0.05 * sr)  # 50ms
    fade_len = int(0.01 * sr)   # 10ms
    is_first_crop = True

    while current_start_sec + 1.0 < duration_sec:
        target_start_sample = int(current_start_sec * sr)
        target_end_sec = current_start_sec + length_sec

        if target_end_sec > duration_sec:
            target_end_sec = duration_sec

        # Snap end BACKWARDS to downbeat
        final_end_sec = target_end_sec

        if div4:
            div4_end = find_end_for_div4_downbeats(align_grid, current_start_sec, target_end_sec)
            if div4_end is not None and div4_end > current_start_sec + 1.0:
                final_end_sec = div4_end
            else:
                # Skip if can't achieve div4
                if overlap:
                    next_target = current_start_sec + (length_sec / 2.0)
                else:
                    next_target = current_start_sec + 1.0
                next_start = find_closest_downbeat(align_grid, next_target)
                if next_start is not None and next_start > current_start_sec:
                    current_start_sec = next_start
                else:
                    current_start_sec = next_target
                continue
        else:
            snapped_end = find_downbeat_before(align_grid, target_end_sec)
            if snapped_end is not None and snapped_end > current_start_sec + 1.0:
                final_end_sec = snapped_end

        final_end_sample = int(final_end_sec * sr)

        # Determine actual start sample
        mono_audio = audio.mean(axis=1) if audio.ndim > 1 else audio.flatten()

        if is_first_crop:
            # First crop: NO zero-crossing snap for start
            actual_start_sample = target_start_sample
            is_first_crop = False
        else:
            # Subsequent crops: snap start to ZC backwards
            actual_start_sample = find_zero_crossing_backwards(mono_audio, target_start_sample, zc_window)

        # End: snap to ZC backwards
        search_start = max(actual_start_sample + int(0.1 * sr), final_end_sample - zc_window)
        search_end = final_end_sample

        if search_start < search_end:
            detect_mono = mono_audio[search_start:search_end]
            zcs = np.where(np.diff(np.signbit(detect_mono)))[0]
            if len(zcs) > 0:
                actual_end_sample = search_start + zcs[-1]
            else:
                actual_end_sample = search_end
        else:
            actual_end_sample = final_end_sample

        # Extract crop
        crop_audio = audio[actual_start_sample:actual_end_sample].copy()

        # Skip if too short
        if crop_audio.shape[0] < sr:
            if overlap:
                next_target = current_start_sec + (length_sec / 2.0)
            else:
                next_target = current_start_sec + 1.0
            next_start = find_closest_downbeat(align_grid, next_target)
            if next_start is not None and next_start > current_start_sec:
                current_start_sec = next_start
            else:
                current_start_sec = next_target
            continue

        # Apply fades
        crop_audio = apply_fades(crop_audio, fade_len)

        # Save
        # Use parent folder name (Track Name) for consistent naming
        track_name = full_mix_path.parent.name
        crop_name = f"{track_name}_{crop_count}.flac"
        crop_path = crops_dir / crop_name
        sf.write(str(crop_path), crop_audio, sr)

        # Crop stems at the same positions
        cropped_stems = crop_all_stems(
            folder_path, crop_path,
            actual_start_sample, actual_end_sample, sr, fade_len
        )
        if cropped_stems:
            logger.debug(f"Cropped {len(cropped_stems)} stems for crop {crop_count}")

        # Metadata
        actual_start_sec = float(actual_start_sample / sr)
        actual_end_sec = float(actual_end_sample / sr)

        # Slice Rhythm Files
        rhythm_suffixes = ['.BEATS_GRID', '.ONSETS', '.DOWNBEATS']
        folder = full_mix_path.parent
        
        for suffix in rhythm_suffixes:
            # Try to find source file
            candidates = list(folder.glob(f"*{suffix}"))
            if candidates:
                source_file = candidates[0]
                dest_file = crop_path.with_suffix(suffix)
                slice_rhythm_file(source_file, dest_file, actual_start_sec, actual_end_sec)

        num_downbeats = count_downbeats_in_range(align_grid, current_start_sec, final_end_sec)
        
        # Calculate local rhythmic features
        # Filter beats within this crop
        crop_beats = [b for b in beat_times if actual_start_sec <= b <= actual_end_sec]
        beat_count = len(crop_beats)
        
        # Estimate local BPM
        local_bpm = global_bpm
        if len(crop_beats) > 1:
            ibis = np.diff(crop_beats)
            mean_ibi = np.mean(ibis)
            if mean_ibi > 0:
                local_bpm = 60.0 / mean_ibi

        position = float(actual_start_sec / duration_sec)

        meta = {
            "position": position,
            "start_time": actual_start_sec,
            "end_time": actual_end_sec,
            "start_sample": int(actual_start_sample),
            "end_sample": int(actual_end_sample),
            "duration": float(actual_end_sec - actual_start_sec),
            "samples": int(actual_end_sample - actual_start_sample),
            "downbeats": int(num_downbeats),
            "source": str(full_mix_path.name),
            "has_stems": len(cropped_stems) > 0,
            "stem_names": list(cropped_stems.keys())
        }

        # Save JSON metadata
        with open(crop_path.with_suffix('.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        # Create .INFO file with position key and transferred features
        # Use crop_path.with_suffix('.INFO') for per-crop INFO file
        info_path = crop_path.with_suffix('.INFO')
        info_data = {
            "position": position,
            "bpm": round(local_bpm, 1),
            "beat_count": beat_count,
            "downbeats": num_downbeats
        }
        
        # Load source .INFO if available
        source_info_path = get_info_path(full_mix_path)
        if source_info_path.exists():
            try:
                # Read source .INFO
                with open(source_info_path, 'r') as f:
                    source_info = json.load(f)
                
                # Transfer features
                transferred_count = 0
                for feature in TRANSFERRABLE_FEATURES:
                    if feature in source_info:
                        info_data[feature] = source_info[feature]
                        transferred_count += 1
                    else:
                        # Warn about missing transferrable features (except optional ones)
                        if feature not in OPTIONAL_TRANSFERRABLE:
                            # Don't spam warnings for every crop, just log debug
                            # logger.debug(f"Missing transferrable feature '{feature}' in source")
                            pass
                
                # Also check optional features
                for feature in OPTIONAL_TRANSFERRABLE:
                    if feature in source_info:
                        info_data[feature] = source_info[feature]
                        transferred_count += 1
                        
            except Exception as e:
                logger.warning(f"Failed to read source info: {e}")
        
        safe_update(info_path, info_data)

        crop_count += 1

        # Advance
        if overlap:
            next_target = current_start_sec + (length_sec / 2.0)
        else:
            next_target = actual_end_sec

        next_start = find_closest_downbeat(align_grid, next_target)
        if next_start is not None and next_start > current_start_sec:
            current_start_sec = next_start
        else:
            current_start_sec = next_target

    return crop_count


def main():
    parser = argparse.ArgumentParser(description="Create training crops for audio files.")
    parser.add_argument("path", type=str, help="Root directory containing organized audio folders")
    parser.add_argument("--length", type=int, default=2097152,
                        help="Crop length in SAMPLES (default: 2097152 = ~47.5s at 44.1kHz)")
    parser.add_argument("--overlap", action="store_true",
                        help="Enable 50%% overlap (next crop starts at last_start + length/2)")
    parser.add_argument("--div4", action="store_true",
                        help="Ensure each crop contains a number of downbeats divisible by 4")
    parser.add_argument("--sequential", action="store_true",
                        help="Sequential mode: fixed sample length, no beat alignment")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory for crops (creates per-track folders)")
    parser.add_argument("--workers", "-j", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing crops (default: skip if crops exist)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    input_path = Path(args.path)

    # Resolve output directory first (needed for filtering)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    # Snapshot folders BEFORE any processing to prevent infinite loops
    folders = []

    if input_path.is_file():
        logger.info(f"Processing single file: {input_path}")
        folders = [input_path.parent]
    elif input_path.is_dir():
        stems = get_stem_files(input_path, include_full_mix=True)
        if 'full_mix' in stems:
            logger.info(f"Processing single folder: {input_path}")
            folders = [input_path]
        else:
            all_folders = find_organized_folders(input_path)

            # Filter out folders that are inside the output directory or look like crop folders
            for folder in all_folders:
                folder_resolved = folder.resolve()

                # Skip if folder is inside output_dir
                if output_dir:
                    try:
                        folder_resolved.relative_to(output_dir)
                        logger.debug(f"Skipping (inside output_dir): {folder.name}")
                        continue
                    except ValueError:
                        pass  # Not inside output_dir, keep it

                # Skip folders that look like crop folders (contain _N.flac files)
                crop_files = list(folder.glob("*_[0-9].flac")) + list(folder.glob("*_[0-9][0-9].flac"))
                if crop_files and not any(folder.glob("full_mix.*")):
                    logger.debug(f"Skipping (looks like crop folder): {folder.name}")
                    continue

                # Skip "crops" subfolders
                if folder.name == "crops":
                    logger.debug(f"Skipping (crops subfolder): {folder}")
                    continue

                folders.append(folder)

    if not folders:
        logger.error(f"No valid organized folders found for: {input_path}")
        return

    # Freeze the folder list - this snapshot prevents processing newly created folders
    folders = list(folders)
    logger.info(f"Found {len(folders)} folders to process (snapshot taken).")
    logger.info(f"Crop length: {args.length} samples")

    if args.sequential:
        logger.info("Mode: Sequential (fixed sample length, no beat alignment)")
    else:
        logger.info(f"Mode: Beat-aligned (overlap={args.overlap}, div4={args.div4})")

    def process_folder_with_lock(folder):
        """Process a single folder with file locking."""
        with FileLock(folder) as lock:
            if not lock.acquired:
                logger.warning(f"Skipping {folder.name} - locked by another process")
                return 0
            return create_crops_for_file(
                folder, args.length, args.overlap, args.div4, args.sequential,
                output_dir, overwrite=args.overwrite
            )

    total_crops = 0
    failed = 0

    if args.workers > 1:
        logger.info(f"Using {args.workers} parallel workers")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_folder_with_lock, folder): folder 
                for folder in folders
            }
            for future in as_completed(futures):
                folder = futures[future]
                try:
                    count = future.result()
                    if count > 0:
                        logger.info(f"{folder.name}: Generated {count} crops.")
                    total_crops += count
                except Exception as e:
                    logger.error(f"Failed to process {folder.name}: {e}")
                    failed += 1
    else:
        # Sequential processing
        for folder in folders:
            try:
                count = process_folder_with_lock(folder)
                if count > 0:
                    logger.info(f"{folder.name}: Generated {count} crops.")
                total_crops += count
            except Exception as e:
                logger.error(f"Failed to process {folder.name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    logger.info(f"Finished. Total crops generated: {total_crops}, Failed: {failed}")


if __name__ == "__main__":
    main()
