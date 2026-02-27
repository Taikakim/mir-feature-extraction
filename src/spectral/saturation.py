"""
Saturation / Overcompression Detection for MIR Project

Detects regions where the audio ENVELOPE stays near full scale — characteristic of
soft limiting, brick-wall limiting, or hard digital clipping.

Algorithm (frame-level peak envelope — pure numpy):
  1. Compute per-frame peak amplitude (max |sample|) over short frames (~5.8ms each)
  2. Flag frames where peak >= amplitude_threshold (default 0.95 ≈ -0.45 dBFS)
  3. Apply HOLD time: bridge flagged regions separated by <= hold_duration (default 20ms)
     This is essential: inside a soft-limited section the audio IS oscillating, so the
     envelope touches the ceiling only at each wave cycle's peak, not continuously.
     Without hold, a 200 Hz sine wave clamped at -0.5 dBFS would produce one 1ms event
     per cycle instead of one continuous event for the whole limited section.
  4. Keep merged regions lasting >= min_duration (default 10ms) — discards isolated
     transient peaks that briefly touch the ceiling.

Why frame-level peaks instead of Essentia SaturationDetector?
  - Essentia's differentialThreshold checks consecutive-SAMPLE variation, which is large
    inside any oscillating waveform — it only catches perfectly flat DC-like clipping.
  - The "flat tops" visible in a waveform zoom are flat at the ENVELOPE level, not at the
    individual sample level. Frame-level peak tracking captures exactly this.
  - Also avoids Essentia's length bug (events dropped for audio > ~1 second).

Two outputs, both useful as conditioning parameters:
  saturation_ratio   — severity: fraction of total duration saturated (0–1)
  saturation_count   — style: number of discrete events (slow vs fast limiter)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import read_audio
from core.common import clamp_feature_value

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection parameters
# ---------------------------------------------------------------------------

# Amplitude threshold for "near ceiling" — frames with peak amplitude >= this are flagged.
# 0.95 ≈ -0.45 dBFS  captures soft limiting at -0.5 dB ceiling
# 0.99 ≈ -0.087 dBFS captures hard clipping only
AMPLITUDE_THRESHOLD = 0.95      # linear, 0–1

# Hold time: merge flagged regions separated by at most this gap.
# Bridges the "valleys" between wave-cycle peaks inside a sustained limited section.
# At 200 Hz, each wave cycle is 5ms, so 20ms hold merges ~4 cycles into one event.
HOLD_DURATION       = 0.020     # seconds — 20ms gap fill

# Minimum event duration AFTER hold merging.  Filters isolated loud transients.
MINIMUM_DURATION    = 0.010     # seconds — 10ms

# Frame hop for peak computation (smaller = finer time resolution).
# 256 samples ≈ 5.8ms at 44.1 kHz — fine enough to resolve 100 Hz wave peaks.
_FRAME_HOP          = 256


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _detect_saturation(
    audio: np.ndarray,
    sr: int,
    amplitude_threshold: float = AMPLITUDE_THRESHOLD,
    hold_duration: float       = HOLD_DURATION,
    min_duration: float        = MINIMUM_DURATION,
    frame_hop: int             = _FRAME_HOP,
) -> Tuple[List[float], List[float]]:
    """
    Find sustained near-ceiling envelope regions using frame-level peak tracking.

    Parameters
    ----------
    audio              : mono float32 array
    sr                 : sample rate
    amplitude_threshold: linear peak threshold for "near full scale" (0–1)
    hold_duration      : merge gap — bridge flagged regions <= this apart (seconds)
    min_duration       : keep merged regions >= this duration (seconds)
    frame_hop          : frame hop in samples (controls time resolution)

    Returns
    -------
    (starts, ends) — boundary timestamps in seconds
    """
    if len(audio) < frame_hop:
        return [], []

    # -- 1. Per-frame peak amplitude ----------------------------------------
    # Reshape to frames and compute max |sample| per frame (vectorised).
    n_complete = len(audio) // frame_hop
    frames = audio[:n_complete * frame_hop].reshape(n_complete, frame_hop)
    frame_peaks = np.max(np.abs(frames), axis=1)

    frame_dt = frame_hop / sr  # seconds per frame

    # -- 2. Flag frames above threshold -------------------------------------
    flagged = frame_peaks >= amplitude_threshold

    if not flagged.any():
        return [], []

    # -- 3. Hold / merge: bridge gaps <= hold_duration ----------------------
    hold_frames = max(1, int(round(hold_duration / frame_dt)))

    # Find raw flagged spans [start_idx, end_idx)
    padded = np.concatenate([[False], flagged, [False]])
    diff   = np.diff(padded.astype(np.int8))
    raw_starts = list(np.where(diff ==  1)[0])
    raw_ends   = list(np.where(diff == -1)[0])

    # Merge spans whose gap <= hold_frames
    merged_s: List[int] = []
    merged_e: List[int] = []
    cs, ce = raw_starts[0], raw_ends[0]
    for s, e in zip(raw_starts[1:], raw_ends[1:]):
        if s - ce <= hold_frames:
            ce = e          # extend current span
        else:
            merged_s.append(cs)
            merged_e.append(ce)
            cs, ce = s, e
    merged_s.append(cs)
    merged_e.append(ce)

    # -- 4. Apply minimum duration filter -----------------------------------
    min_frames = max(1, int(round(min_duration / frame_dt)))

    starts: List[float] = []
    ends:   List[float] = []
    for s, e in zip(merged_s, merged_e):
        if (e - s) >= min_frames:
            starts.append(float(s) * frame_dt)
            ends.append(float(e) * frame_dt)

    return starts, ends


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_saturation(
    audio_path: str | Path,
    audio: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    amplitude_threshold: float = AMPLITUDE_THRESHOLD,
    hold_duration: float       = HOLD_DURATION,
    min_duration: float        = MINIMUM_DURATION,
) -> Dict[str, float]:
    """
    Measure saturation / overcompression in an audio file.

    Detects sustained near-ceiling envelope characteristic of soft limiting,
    brick-wall limiting, or hard clipping.

    Parameters
    ----------
    audio_path         : path to audio file (used for loading when audio is None)
    audio              : pre-loaded mono float32 array (skips disk read)
    sr                 : sample rate in Hz (required when audio is provided)
    amplitude_threshold: linear peak amplitude for "near ceiling" (0–1, default 0.95)
    hold_duration      : merge gap in seconds; bridges the valleys between wave-cycle
                         peaks inside a soft-limited section (default 20ms)
    min_duration       : minimum event duration after merging (default 10ms)

    Returns
    -------
    dict with keys:
        saturation_ratio  — fraction of total duration with saturated envelope (0–1)
        saturation_count  — number of discrete saturation events (int ≥ 0)
    """
    audio_path = Path(audio_path)

    if audio is None:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio, sr = read_audio(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
    else:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

    total_duration = len(audio) / sr

    starts, ends = _detect_saturation(
        audio, sr,
        amplitude_threshold=amplitude_threshold,
        hold_duration=hold_duration,
        min_duration=min_duration,
    )

    if not starts:
        return {'saturation_ratio': 0.0, 'saturation_count': 0}

    saturated_duration = sum(e - s for s, e in zip(starts, ends))
    ratio = saturated_duration / total_duration if total_duration > 0 else 0.0

    results = {
        'saturation_ratio': float(clamp_feature_value('saturation_ratio', ratio)),
        'saturation_count': int(clamp_feature_value('saturation_count', len(starts))),
    }

    if results['saturation_ratio'] > 0.001:
        logger.debug(
            f"Saturation: {results['saturation_ratio']:.4f} ratio "
            f"({saturated_duration:.3f}s / {total_duration:.1f}s total), "
            f"{results['saturation_count']} events "
            f"[thresh={amplitude_threshold:.3f}, hold={hold_duration*1000:.0f}ms]"
        )

    return results


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def batch_analyze_saturation(root_directory: str | Path, overwrite: bool = False) -> dict:
    from core.file_utils import find_organized_folders, get_stem_files
    from core.json_handler import safe_update, get_info_path, should_process

    SATURATION_KEYS = ['saturation_ratio', 'saturation_count']

    root_directory = Path(root_directory)
    folders = find_organized_folders(root_directory)
    stats = {'total': len(folders), 'success': 0, 'skipped': 0, 'failed': 0}

    for i, folder in enumerate(folders, 1):
        stems = get_stem_files(folder, include_full_mix=True)
        if 'full_mix' not in stems:
            stats['failed'] += 1
            continue

        info_path = get_info_path(stems['full_mix'])
        if not should_process(info_path, SATURATION_KEYS, overwrite):
            stats['skipped'] += 1
            continue

        try:
            results = analyze_saturation(stems['full_mix'])
            safe_update(info_path, results)
            stats['success'] += 1
            logger.info(
                f"[{i}/{stats['total']}] {folder.name}: "
                f"ratio={results['saturation_ratio']:.4f}, "
                f"count={results['saturation_count']}"
            )
        except Exception as e:
            stats['failed'] += 1
            logger.error(f"[{i}/{stats['total']}] {folder.name}: {e}")

    logger.info(
        f"Saturation batch done: {stats['success']} ok, "
        f"{stats['skipped']} skipped, {stats['failed']} failed / {stats['total']} total"
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from core.common import setup_logging

    parser = argparse.ArgumentParser(
        description='Detect audio saturation / overcompression (soft limiting + hard clipping)'
    )
    parser.add_argument('path', help='Audio file or folder to analyse')
    parser.add_argument('--batch', action='store_true',
                        help='Batch-process all organised folders under path')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-analyse even if saturation data already exists')
    parser.add_argument('--threshold', type=float, default=AMPLITUDE_THRESHOLD,
                        help=f'Linear amplitude threshold for near-ceiling (default {AMPLITUDE_THRESHOLD},'
                             f' ≈ {20*np.log10(AMPLITUDE_THRESHOLD):.1f} dBFS)')
    parser.add_argument('--hold', type=float, default=HOLD_DURATION,
                        help=f'Hold/merge gap in seconds (default {HOLD_DURATION},'
                             f' {HOLD_DURATION*1000:.0f}ms). Bridges wave-cycle valleys'
                             f' within a soft-limited region.')
    parser.add_argument('--min-duration', type=float, default=MINIMUM_DURATION,
                        help=f'Minimum event duration after merging (default {MINIMUM_DURATION},'
                             f' {MINIMUM_DURATION*1000:.0f}ms)')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    path = Path(args.path)
    if not path.exists():
        logger.error(f'Path not found: {path}')
        sys.exit(1)

    if args.batch:
        batch_analyze_saturation(path, overwrite=args.overwrite)
    else:
        from core.file_utils import get_stem_files
        from core.json_handler import safe_update, get_info_path

        audio_path = path if path.is_file() else None
        if audio_path is None:
            stems = get_stem_files(path, include_full_mix=True)
            if 'full_mix' not in stems:
                logger.error('No full_mix found')
                sys.exit(1)
            audio_path = stems['full_mix']

        results = analyze_saturation(
            audio_path,
            amplitude_threshold=args.threshold,
            hold_duration=args.hold,
            min_duration=args.min_duration,
        )
        pct = results['saturation_ratio'] * 100
        print(f"\nSaturation Analysis  ({audio_path.name})")
        print(f"  saturation_ratio  : {results['saturation_ratio']:.4f}  ({pct:.2f}% of duration)")
        print(f"  saturation_count  : {results['saturation_count']} events")
        print(f"  threshold         : {args.threshold:.3f}"
              f"  ({20*np.log10(args.threshold):.2f} dBFS)")
        print(f"  hold              : {args.hold*1000:.0f} ms")
        print(f"  min_duration      : {args.min_duration*1000:.0f} ms")

        if path.is_dir():
            info_path = get_info_path(audio_path)
            safe_update(info_path, results)
            print(f"  Saved to: {info_path}")
