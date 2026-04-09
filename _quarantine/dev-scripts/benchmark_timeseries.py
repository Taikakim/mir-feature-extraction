"""
Benchmark: Time-Series Feature Extraction

Tests each feature group on 10 crop audio files and reports per-group timings.
Extrapolates estimated wall time to re-analyse the full dataset.

Usage:
    python benchmark_timeseries.py
    python benchmark_timeseries.py --crops-dir /path/to/crops --n 10
    python benchmark_timeseries.py --include-timbral        # also benchmark timbral (slow)
    python benchmark_timeseries.py --steps 64               # test at non-default step count
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- Project path setup -------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# ROCm env must be set before any torch import (we don't use torch here,
# but downstream imports might; set it defensively)
try:
    from core.rocm_env import setup_rocm_env
    setup_rocm_env()
except Exception:
    pass

from core.file_utils import read_audio

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('benchmark_timeseries')

# ---------------------------------------------------------------------------
# Crop discovery
# ---------------------------------------------------------------------------

DEFAULT_CROPS_DIRS = [
    '/run/media/kim/Mantu/ai-music/Goa_Separated_crops',
]

AUDIO_EXTS = {'.flac', '.wav', '.mp3'}


def find_crop_files(root: str, n: int = 10, seed: int = 42) -> List[Path]:
    """
    Find up to *n* crop audio files under *root*.

    Skips stem files (names ending with _bass, _drums, _other, _vocals)
    and files shorter than 5 seconds.
    """
    STEM_SUFFIXES = ('_bass', '_drums', '_other', '_vocals')
    candidates: List[Path] = []

    for ext in AUDIO_EXTS:
        for p in Path(root).rglob(f'*{ext}'):
            if any(p.stem.endswith(s) for s in STEM_SUFFIXES):
                continue
            candidates.append(p)

    if not candidates:
        return []

    rng = random.Random(seed)
    rng.shuffle(candidates)

    # Verify audio is loadable and long enough, take first n good ones
    selected: List[Path] = []
    for p in candidates:
        if len(selected) >= n:
            break
        try:
            import soundfile as sf
            info = sf.info(str(p))
            if info.duration >= 5.0:
                selected.append(p)
        except Exception:
            continue

    return selected


# ---------------------------------------------------------------------------
# Per-group timing helpers
# ---------------------------------------------------------------------------

def _time(fn, *args, **kwargs) -> Tuple[float, any]:
    """Call fn(*args, **kwargs), return (elapsed_seconds, result)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


def benchmark_one_crop(
    audio_path: Path,
    n_steps: int,
    include_timbral: bool,
    timbral_n_steps: int,
) -> Dict[str, float]:
    """
    Run each feature group in isolation and return a dict of timings (seconds).
    """
    import librosa

    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    from spectral.timeseries_features import (
        _compute_multiband_rms_ts,
        _compute_spectral_ts,
        _compute_activation_ts,
        _compute_onset_ts,
        _compute_hpcp_ts,
        _compute_timbral_ts,
        ESSENTIA_AVAILABLE,
        TIMBRAL_AVAILABLE,
    )

    # Load audio once
    t_load, (audio_raw, sr) = _time(read_audio, str(audio_path))
    audio = (audio_raw.mean(axis=1) if audio_raw.ndim > 1 else audio_raw).astype(np.float32)
    audio_duration = len(audio) / sr

    timings: Dict[str, float] = {'audio_load': t_load}

    # 1. Multiband RMS
    t, _ = _time(_compute_multiband_rms_ts, audio, sr, n_steps)
    timings['multiband_rms_ts'] = t

    # 2. Spectral
    t, _ = _time(_compute_spectral_ts, audio, sr, n_steps)
    timings['spectral_ts'] = t

    # 3. Beat detection + activations
    def _beats_and_activations():
        _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        beats = librosa.frames_to_time(beat_frames, sr=sr)
        downbeats = beats[::4] if len(beats) >= 4 else beats
        _compute_activation_ts(beats,     audio_duration, n_steps)
        _compute_activation_ts(downbeats, audio_duration, n_steps)

    t, _ = _time(_beats_and_activations)
    timings['beat_activations_ts'] = t

    # 4. Onset strength
    t, _ = _time(_compute_onset_ts, audio, sr, n_steps)
    timings['onsets_activations_ts'] = t

    # 5. HPCP + tonic
    if ESSENTIA_AVAILABLE:
        t, _ = _time(_compute_hpcp_ts, audio, sr, n_steps)
        timings['hpcp_tonic_ts'] = t
    else:
        timings['hpcp_tonic_ts'] = float('nan')

    # 6. Timbral (optional)
    if include_timbral:
        if TIMBRAL_AVAILABLE:
            feats = ['brightness', 'roughness', 'hardness', 'depth', 'reverb']
            t, _ = _time(_compute_timbral_ts, audio, sr, feats, timbral_n_steps)
            timings['timbral_ts'] = t
        else:
            timings['timbral_ts'] = float('nan')

    # Total (excluding timbral for the base estimate)
    base_keys = ['multiband_rms_ts', 'spectral_ts', 'beat_activations_ts',
                 'onsets_activations_ts', 'hpcp_tonic_ts']
    timings['total_no_timbral'] = sum(
        v for k, v in timings.items() if k in base_keys and not np.isnan(v))

    return timings


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    if np.isnan(v):
        return '   N/A  '
    return f'{v:7.3f}s'


def _fmt_hms(seconds: float) -> str:
    if np.isnan(seconds):
        return 'N/A'
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f'{h}h {m:02d}m {s:02d}s'
    if m > 0:
        return f'{m}m {s:02d}s'
    return f'{s}s'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Benchmark time-series feature extraction')
    parser.add_argument('--crops-dir', type=str, default=None,
                        help='Root directory to search for crop audio files')
    parser.add_argument('--n', type=int, default=10,
                        help='Number of crops to benchmark (default: 10)')
    parser.add_argument('--steps', type=int, default=256,
                        help='n_steps for time-series extraction (default: 256)')
    parser.add_argument('--timbral-steps', type=int, default=16,
                        help='timbral_n_steps (default: 16, each ~750 ms)')
    parser.add_argument('--include-timbral', action='store_true',
                        help='Also benchmark timbral time-series (slow)')
    parser.add_argument('--dataset-size', type=int, default=4494,
                        help='Total crops in full dataset for extrapolation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # -- Find crops ------------------------------------------------------------
    print(f'\nSearching for {args.n} crop files...')
    crops: List[Path] = []

    search_dirs = [args.crops_dir] if args.crops_dir else DEFAULT_CROPS_DIRS
    for d in search_dirs:
        if Path(d).exists():
            found = find_crop_files(d, n=args.n, seed=args.seed)
            crops.extend(found)
            if len(crops) >= args.n:
                break

    crops = crops[:args.n]

    if not crops:
        print('ERROR: No crop files found. Check that your external drives are mounted.')
        print(f'  Searched: {search_dirs}')
        sys.exit(1)

    print(f'Found {len(crops)} crops.\n')

    # -- Run benchmarks --------------------------------------------------------
    all_timings: List[Dict[str, float]] = []

    for idx, crop_path in enumerate(crops):
        print(f'  [{idx+1:2d}/{len(crops)}] {crop_path.name} ... ', end='', flush=True)
        try:
            t = benchmark_one_crop(
                crop_path,
                n_steps=args.steps,
                include_timbral=args.include_timbral,
                timbral_n_steps=args.timbral_steps,
            )
            all_timings.append(t)
            print(f"total={t['total_no_timbral']:.2f}s")
        except Exception as e:
            print(f'FAILED: {e}')
            logger.exception(f'Failed on {crop_path}')

    if not all_timings:
        print('\nNo successful benchmarks.')
        sys.exit(1)

    # -- Aggregate results -----------------------------------------------------
    all_keys = [k for k in all_timings[0].keys() if k != 'audio_load']
    if args.include_timbral:
        all_keys = [k for k in all_keys]   # keep timbral_ts

    rows = []
    for key in all_keys:
        vals = [t[key] for t in all_timings if key in t and not np.isnan(t[key])]
        if not vals:
            rows.append((key, float('nan'), float('nan'), float('nan')))
        else:
            rows.append((key, np.mean(vals), np.min(vals), np.max(vals)))

    # -- Print table -----------------------------------------------------------
    sep = '-' * 70
    header_fmt = f'  {"Feature group":<30} {"Mean":>10} {"Min":>10} {"Max":>10}'

    print(f'\n{"="*70}')
    print(f'  Time-Series Benchmark  (n_steps={args.steps}, crops={len(all_timings)})')
    print(f'{"="*70}')
    print(header_fmt)
    print(sep)

    for name, mean, mn, mx in rows:
        label = name.replace('_', ' ').replace('ts', '(ts)').strip()
        print(f'  {label:<30} {_fmt(mean)} {_fmt(mn)} {_fmt(mx)}')

    print(sep)

    # Extrapolation
    n_crops   = args.dataset_size
    base_mean = next((m for n, m, _, _ in rows if n == 'total_no_timbral'), float('nan'))
    tim_mean  = next((m for n, m, _, _ in rows if n == 'timbral_ts'), float('nan'))

    print(f'\n  Extrapolation to {n_crops:,} crops:')
    if not np.isnan(base_mean):
        total_base = base_mean * n_crops
        print(f'    All features (no timbral) : {_fmt_hms(total_base)}  '
              f'({total_base/3600:.1f} h)')
    if args.include_timbral and not np.isnan(tim_mean):
        total_tim = (base_mean + tim_mean) * n_crops
        print(f'    With timbral ({args.timbral_steps} steps)     : {_fmt_hms(total_tim)}  '
              f'({total_tim/3600:.1f} h)')

    print()

    # Timbral viability note
    if args.include_timbral and not np.isnan(tim_mean):
        chunk_ms = 1000 * (args.steps / 256 * 12.0) / args.timbral_steps
        print(f'  NOTE: timbral_{args.timbral_steps} uses ~{chunk_ms:.0f} ms audio chunks.')
        if chunk_ms < 300:
            print('  WARNING: chunks < 300 ms may produce unreliable timbral scores.')
        else:
            print('  Chunk length looks adequate for timbral analysis.')
    print()


if __name__ == '__main__':
    main()
