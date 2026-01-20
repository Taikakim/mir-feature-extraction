"""
Tool to create training crops for Stable Audio Tools training.

Usage:
    # Sequential mode (fixed sample length, no beat alignment)
    python src/tools/create_training_crops.py /path/to/data --length 2097152 --sequential

    # Beat-aligned mode
    python src/tools/create_training_crops.py /path/to/data --length 2097152
    python src/tools/create_training_crops.py /path/to/data --length 2097152 --overlap
    python src/tools/create_training_crops.py /path/to/data --length 2097152 --overlap --div4

Features:
- --sequential: Simple sequential crops at exact sample length (no beat logic)
- Beat-aligned mode: Start times snap to closest downbeat
- End times snap BACKWARDS to last downbeat before target end (never exceeds length)
- When --div4: ensures each crop contains downbeats divisible by 4
- When --overlap: next crop starts at (last_start + length/2), snapped to closest downbeat
- 10ms fade-in and fade-out for clean transitions
- First crop starts without zero-crossing snap (preserves exact position)
- Silence detection at -72dB threshold
"""

import argparse
import logging
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import find_organized_folders, get_stem_files
from core.common import setup_logging
from rhythm.beat_grid import load_beat_grid
from core.json_handler import get_info_path, read_info

logger = logging.getLogger(__name__)


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
                            audio: np.ndarray, full_mix_path: Path) -> int:
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

    crops_dir = folder_path / "crops"
    crops_dir.mkdir(exist_ok=True)

    crop_count = 0
    current_sample = start_sample

    while current_sample + length_samples <= total_samples:
        end_sample = current_sample + length_samples

        # Extract crop (no ZC snap for sequential mode)
        crop_audio = audio[current_sample:end_sample].copy()

        # Apply fades
        crop_audio = apply_fades(crop_audio, fade_len)

        # Save
        crop_name = f"section_{crop_count:03d}_{full_mix_path.stem}.flac"
        crop_path = crops_dir / crop_name
        sf.write(str(crop_path), crop_audio, sr)

        # Metadata
        start_sec = current_sample / sr
        end_sec = end_sample / sr

        meta = {
            "position": start_sec / duration_sec,
            "start_time": start_sec,
            "end_time": end_sec,
            "start_sample": current_sample,
            "end_sample": end_sample,
            "duration": end_sec - start_sec,
            "samples": length_samples,
            "source": str(full_mix_path.name)
        }

        with open(crop_path.with_suffix('.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        crop_count += 1
        current_sample = end_sample  # Sequential: next starts where this ended

    return crop_count


def create_crops_for_file(folder_path: Path,
                          length_samples: int,
                          overlap: bool = True,
                          div4: bool = False,
                          sequential: bool = False) -> int:
    """
    Generate crops for a single folder.

    Args:
        folder_path: Path to organized folder
        length_samples: Target crop length in samples
        overlap: If True, next crop starts at last_start + length/2
        div4: If True, ensure each crop contains downbeats divisible by 4
        sequential: If True, use simple sequential mode (no beat alignment)
    """
    stems = get_stem_files(folder_path, include_full_mix=True)
    if 'full_mix' not in stems:
        logger.warning(f"No full_mix in {folder_path.name}")
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
        return create_sequential_crops(folder_path, length_samples, sr, audio, full_mix_path)

    # Beat-aligned mode
    # Load BPM Info
    info_path = get_info_path(full_mix_path)
    info_data = read_info(info_path)
    bpm = info_data.get('bpm', 0)

    use_beat_logic = True
    if bpm is None or float(bpm) <= 0:
        use_beat_logic = False
        logger.info(f"{folder_path.name}: BPM not defined, using sequential fallback.")
        return create_sequential_crops(folder_path, length_samples, sr, audio, full_mix_path)

    # Load beat grid
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
        return create_sequential_crops(folder_path, length_samples, sr, audio, full_mix_path)

    # Find start after silence (-72dB threshold)
    start_offset_samples = get_start_offset_above_threshold(audio, threshold_db=-72.0)
    start_offset_sec = start_offset_samples / sr
    if start_offset_samples > 0:
        logger.info(f"Skipping initial silence: {start_offset_sec:.2f}s")

    crops_dir = folder_path / "crops"
    crops_dir.mkdir(exist_ok=True)

    crop_count = 0
    align_grid = downbeat_times if (downbeat_times is not None and len(downbeat_times) > 0) else beat_times

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
        crop_name = f"section_{crop_count:03d}_{full_mix_path.stem}.flac"
        crop_path = crops_dir / crop_name
        sf.write(str(crop_path), crop_audio, sr)

        # Metadata
        actual_start_sec = float(actual_start_sample / sr)
        actual_end_sec = float(actual_end_sample / sr)

        num_downbeats = count_downbeats_in_range(align_grid, current_start_sec, final_end_sec)

        meta = {
            "position": float(actual_start_sec / duration_sec),
            "start_time": actual_start_sec,
            "end_time": actual_end_sec,
            "start_sample": int(actual_start_sample),
            "end_sample": int(actual_end_sample),
            "duration": float(actual_end_sec - actual_start_sec),
            "samples": int(actual_end_sample - actual_start_sample),
            "downbeats": int(num_downbeats),
            "source": str(full_mix_path.name)
        }

        with open(crop_path.with_suffix('.json'), 'w') as f:
            json.dump(meta, f, indent=2)

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
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    input_path = Path(args.path)
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
            folders = find_organized_folders(input_path)

    if not folders:
        logger.error(f"No valid organized folders found for: {input_path}")
        return

    logger.info(f"Found {len(folders)} folders to process.")
    logger.info(f"Crop length: {args.length} samples")

    if args.sequential:
        logger.info("Mode: Sequential (fixed sample length, no beat alignment)")
    else:
        logger.info(f"Mode: Beat-aligned (overlap={args.overlap}, div4={args.div4})")

    total_crops = 0
    for folder in folders:
        try:
            count = create_crops_for_file(
                folder, args.length, args.overlap, args.div4, args.sequential
            )
            logger.info(f"{folder.name}: Generated {count} crops.")
            total_crops += count
        except Exception as e:
            logger.error(f"Failed to process {folder.name}: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"Finished. Total crops generated: {total_crops}")


if __name__ == "__main__":
    main()
