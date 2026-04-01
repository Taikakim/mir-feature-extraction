"""
HPCP + TIV + Tonic Harmonic Feature Extraction for MIR Project

Parallel harmonic pipeline alongside CQT chroma. Uses Essentia's peak-based
HPCP for accurate pitch-class extraction (suppresses overtone contamination),
then projects into Tonal Interval Vector (TIV) space for semantically rich
global conditioning. Tonic estimation is a free by-product of the HPCP
computation via Essentia's Key algorithm.

Stem selection: bass+other only (no vocals = pitch smearing; no drums = transient
noise). Full mix fallback with warning if stems are unavailable.

Dependencies:
- essentia
- numpy
- soundfile
- src.core.file_utils
- src.core.common

New features (26 numeric + 1 string label):
- hpcp_0 through hpcp_11: Pitch class fingerprint [0, 1], L∞-normalized
- tiv_0 through tiv_11:   Tonal Interval Vector [-1, 1], L2-normalized
- tonic:          Pitch class integer [0, 11]
- tonic_strength: Key estimation confidence [0, 1]
- tonic_scale:    Scale label ('major', 'minor', ...) — string, stored in INFO only
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import get_stem_files, read_audio
from core.common import clamp_feature_value
from core.json_handler import safe_update, get_info_path

logger = logging.getLogger(__name__)

# Canonical pitch class names (index = HPCP bin 0..11)
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Perceptual weights for TIV (Harte et al. 2006)
# w_a[k-1] for k = 1..6
_TIV_WEIGHTS = [3.0, 8.0, 11.5, 15.0, 14.5, 7.5]


def mix_bass_other_stems(
    folder_path: Path,
) -> Tuple[Optional[np.ndarray], int, bool]:
    """
    Load bass + other stems, mix to mono, peak-normalize to 0.95.

    Falls back to full mix with a warning if neither stem exists.
    Pads the shorter stem with zeros before summing (mirrors mix_harmonic_stems).

    Returns:
        (audio, sr, used_stems)
        used_stems: True when stems were found, False when full_mix fallback used
    """
    stems = get_stem_files(folder_path, include_full_mix=True)

    target_stems = ['bass', 'other']
    available = [s for s in target_stems if s in stems]

    if len(available) >= 2:
        logger.info(f"Using stems: {', '.join(available)} (excluding drums and vocals)")
        mixed: Optional[np.ndarray] = None
        sr: int = 0

        for stem_name in available:
            audio, stem_sr = read_audio(str(stems[stem_name]))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            if mixed is None:
                mixed = audio
                sr = stem_sr
            else:
                max_len = max(len(mixed), len(audio))
                mixed = np.pad(mixed, (0, max_len - len(mixed)))
                audio = np.pad(audio, (0, max_len - len(audio)))
                mixed = mixed + audio

        max_val = float(np.abs(mixed).max())
        if max_val > 0:
            mixed = mixed / max_val * 0.95
        return mixed, sr, True

    # Single stem fallback (only one of bass/other available)
    if available:
        stem_name = available[0]
        logger.warning(f"Only '{stem_name}' stem available; using it alone for HPCP")
        audio, sr = read_audio(str(stems[stem_name]))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        max_val = float(np.abs(audio).max())
        if max_val > 0:
            audio = audio / max_val * 0.95
        return audio, sr, True

    # Full mix fallback
    if 'full_mix' in stems:
        logger.warning("Bass/other stems not available — falling back to full_mix for HPCP "
                       "(quality compromised by drums and vocals)")
        audio, sr = read_audio(str(stems['full_mix']))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        return audio, sr, False

    return None, 0, False


def compute_hpcp_frames(
    audio: np.ndarray,
    sr: int,
    frame_size: int = 4096,
    hop_size: int = 2048,
) -> np.ndarray:
    """
    Compute per-frame HPCP vectors using Essentia's peak-based pipeline.

    Pipeline: FrameGenerator → Windowing(hann) → Spectrum → SpectralPeaks → HPCP

    HPCP params: size=12, harmonics=8, minFrequency=40 Hz, maxFrequency=5000 Hz,
    weightType='cosine', nonLinear=True, normalized='unitMax'

    Args:
        audio:      Mono float32 audio signal
        sr:         Sample rate in Hz
        frame_size: Analysis frame size in samples (default 4096 ≈ 93 ms at 44.1 kHz)
        hop_size:   Hop between frames (default 2048 = 50% overlap)

    Returns:
        np.ndarray of shape (n_frames, 12) — per-frame HPCP vectors.
        Silent/empty frames produce zero rows.
    """
    import essentia.standard as es

    frame_gen = es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size,
                                   startFromZero=True)
    window = es.Windowing(type='hann', size=frame_size)
    spectrum = es.Spectrum(size=frame_size)
    spectral_peaks = es.SpectralPeaks(
        maxPeaks=100,
        magnitudeThreshold=0.00001,
        sampleRate=sr,
        orderBy='magnitude',
    )
    hpcp_algo = es.HPCP(
        size=12,
        harmonics=8,
        minFrequency=40.0,
        maxFrequency=5000.0,
        weightType='cosine',
        nonLinear=True,
        normalized='unitMax',
        referenceFrequency=261.63,  # C4 — aligns bin 0 with C, matching chroma_* convention
        sampleRate=sr,
    )

    frames = []
    for frame in frame_gen:
        windowed = window(frame)
        spec = spectrum(windowed)
        freqs, mags = spectral_peaks(spec)
        hpcp_vec = hpcp_algo(freqs, mags)
        frames.append(np.array(hpcp_vec, dtype=np.float32))

    if not frames:
        return np.zeros((0, 12), dtype=np.float32)
    return np.stack(frames, axis=0)  # (n_frames, 12)


def compute_tiv(hpcp_avg: np.ndarray) -> np.ndarray:
    """
    Compute the Tonal Interval Vector (TIV) from an averaged HPCP vector.

    Algorithm:
      1. If sum == 0 (silent), return zeros.
      2. L1-normalize: c = hpcp_avg / hpcp_avg.sum()
      3. DFT of c.
      4. Apply perceptual weights w_a[k-1] to component k=1..6.
      5. Flatten Re/Im: [Re(k1), Im(k1), ..., Re(k6), Im(k6)]
      6. L2-normalize; if norm == 0 return zeros.

    Args:
        hpcp_avg: Shape (12,) averaged HPCP vector (may be unnormalized)

    Returns:
        np.ndarray of shape (12,), L2-normalized, range [-1, 1]
    """
    if hpcp_avg.sum() == 0.0:
        return np.zeros(12, dtype=np.float32)

    # Step 1: L1-normalize
    c = hpcp_avg / hpcp_avg.sum()

    # Step 2: DFT (length 12; only components k=1..6 are used)
    dft = np.fft.fft(c)

    # Step 3: Apply perceptual weights and flatten Re/Im
    tiv = np.zeros(12, dtype=np.float64)
    for i, w in enumerate(_TIV_WEIGHTS):
        k = i + 1  # k = 1..6
        tiv[2 * i]     = w * dft[k].real
        tiv[2 * i + 1] = w * dft[k].imag

    # Step 4: L2-normalize
    norm = float(np.linalg.norm(tiv))
    if norm == 0.0:
        return np.zeros(12, dtype=np.float32)
    return (tiv / norm).astype(np.float32)


def _estimate_tonic(hpcp_avg: np.ndarray) -> Tuple[int, float, str]:
    """
    Estimate tonic from an averaged HPCP vector using Essentia's Key algorithm.

    Tries the 'edmm' profile first (EDM-calibrated); falls back to 'edma' if
    ValueError is raised (older Essentia versions).

    Returns:
        (tonic_int, strength, scale)
        tonic_int: pitch class integer 0-11 (C=0, C#=1, …, B=11)
        strength:  confidence [0, 1]
        scale:     'major', 'minor', or other Essentia string
    """
    import essentia.standard as es

    hpcp_list = hpcp_avg.tolist()

    for profile in ('edmm', 'edma'):
        try:
            key_algo = es.Key(profileType=profile)
            key_str, scale, strength, _ = key_algo(hpcp_list)
            if key_str in PITCH_CLASSES:
                return PITCH_CLASSES.index(key_str), float(strength), scale
            logger.warning(f"Essentia Key returned unknown note '{key_str}'; storing tonic=0")
            return 0, float(strength), scale
        except ValueError:
            if profile == 'edmm':
                logger.warning("'edmm' Key profile unavailable — falling back to 'edma'")
                continue
            raise

    return 0, 0.0, 'unknown'


def analyze_hpcp_tiv(
    audio_path,
    use_stems: bool = True,
    audio: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
) -> Dict:
    """
    Compute HPCP, TIV, and tonic features for a single audio file.

    When *audio* is provided, stem loading is skipped entirely (use_stems is
    ignored). This is the preloaded-stem fast path used by feature_extractor.

    Returns dict with keys:
        hpcp_0..hpcp_11, tiv_0..tiv_11, tonic, tonic_strength, tonic_scale
    """
    audio_path = Path(audio_path)

    # --- Audio loading ---
    if audio is not None:
        # Preloaded audio provided — use directly
        mix = audio.astype(np.float32)
        if mix.ndim > 1:
            mix = mix.mean(axis=1)
    elif use_stems:
        mix, sr, _ = mix_bass_other_stems(audio_path.parent
                                          if audio_path.is_file() else audio_path)
        if mix is None:
            logger.error(f"No audio available for HPCP analysis: {audio_path}")
            return _zero_result()
    else:
        mix, sr = read_audio(str(audio_path))
        mix = mix.astype(np.float32)
        if mix.ndim > 1:
            mix = mix.mean(axis=1)

    if sr is None or sr == 0:
        logger.error(f"Invalid sample rate for HPCP analysis: {audio_path}")
        return _zero_result()

    # --- HPCP ---
    try:
        frames = compute_hpcp_frames(mix, int(sr))
    except Exception as e:
        logger.error(f"HPCP computation failed for {audio_path}: {e}")
        return _zero_result()

    if frames.shape[0] == 0:
        logger.warning(f"No HPCP frames extracted for {audio_path} — silent audio?")
        return _zero_result()

    hpcp_avg = frames.mean(axis=0)  # (12,)

    # Silent audio guard
    if hpcp_avg.sum() == 0.0:
        logger.warning(f"All-zero HPCP for {audio_path} — silent audio")
        return _zero_result()

    # Final L∞ normalize on the averaged vector
    linf = float(hpcp_avg.max())
    if linf > 0.0:
        hpcp_avg = hpcp_avg / linf

    # --- TIV ---
    tiv = compute_tiv(hpcp_avg)

    # --- Tonic ---
    try:
        tonic_int, tonic_strength, tonic_scale = _estimate_tonic(hpcp_avg)
    except Exception as e:
        logger.error(f"Tonic estimation failed for {audio_path}: {e}")
        tonic_int, tonic_strength, tonic_scale = 0, 0.0, 'unknown'

    result = {}
    for i, v in enumerate(hpcp_avg):
        result[f'hpcp_{i}'] = float(clamp_feature_value(f'hpcp_{i}', float(v)))
    for i, v in enumerate(tiv):
        result[f'tiv_{i}'] = float(clamp_feature_value(f'tiv_{i}', float(v)))
    result['tonic']          = float(clamp_feature_value('tonic', float(tonic_int)))
    result['tonic_strength'] = float(clamp_feature_value('tonic_strength', tonic_strength))
    result['tonic_scale']    = tonic_scale  # string label — not clamped

    return result


def _zero_result() -> Dict:
    """Return all-zero HPCP/TIV/tonic result (silent or error case)."""
    result = {}
    for i in range(12):
        result[f'hpcp_{i}'] = 0.0
        result[f'tiv_{i}']  = 0.0
    result['tonic']          = 0.0
    result['tonic_strength'] = 0.0
    result['tonic_scale']    = 'unknown'
    return result


def batch_analyze_hpcp_tiv(root_directory, overwrite: bool = False) -> dict:
    """
    Batch-analyze HPCP/TIV/tonic for all organized folders under root_directory.

    Mirrors batch_analyze_chroma(). Skips folders that already have hpcp_0
    unless overwrite=True.

    Returns stats dict: {total, success, skipped, failed, errors}
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch HPCP/TIV analysis: {root_directory}")

    folders = find_organized_folders(root_directory)

    stats = {'total': len(folders), 'success': 0, 'skipped': 0, 'failed': 0, 'errors': []}
    logger.info(f"Found {stats['total']} organized folders")

    for i, folder in enumerate(folders, 1):
        logger.info(f"Processing {i}/{stats['total']}: {folder.name}")
        stems = get_stem_files(folder, include_full_mix=True)
        if 'full_mix' not in stems:
            logger.warning(f"No full_mix in {folder.name}")
            stats['failed'] += 1
            continue

        info_path = get_info_path(stems['full_mix'])
        if info_path.exists() and not overwrite:
            try:
                import json
                with open(info_path) as f:
                    data = json.load(f)
                if 'hpcp_0' in data:
                    logger.info("HPCP data already exists. Use --overwrite to regenerate.")
                    stats['skipped'] += 1
                    continue
            except Exception:
                pass

        try:
            results = analyze_hpcp_tiv(folder, use_stems=True)
            safe_update(info_path, results)
            stats['success'] += 1
        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append(f"{folder.name}: {e}")
            logger.error(f"Failed to process {folder.name}: {e}")

    logger.info("=" * 60)
    logger.info(f"Batch HPCP/TIV Summary: {stats['success']} ok, "
                f"{stats['skipped']} skipped, {stats['failed']} failed")
    logger.info("=" * 60)
    return stats


# ── Command-line interface ────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    from core.common import setup_logging

    parser = argparse.ArgumentParser(description='HPCP + TIV + Tonic feature extraction')
    parser.add_argument('path', type=str, help='Path to organized folder or root directory')
    parser.add_argument('--batch', action='store_true',
                        help='Batch process all organized folders in directory tree')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing data')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    path = Path(args.path)
    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)

    if args.batch:
        stats = batch_analyze_hpcp_tiv(path, overwrite=args.overwrite)
        if stats['failed'] > 0:
            sys.exit(1)
    else:
        if not path.is_dir():
            logger.error(f"Path must be a directory: {path}")
            sys.exit(1)

        results = analyze_hpcp_tiv(path, use_stems=True)

        stems = get_stem_files(path, include_full_mix=True)
        if 'full_mix' in stems:
            safe_update(get_info_path(stems['full_mix']), results)

        tonic_name = PITCH_CLASSES[int(results['tonic'])] if 0 <= int(results['tonic']) <= 11 else '?'
        print(f"\nHPCP + TIV + Tonic:")
        print(f"  Tonic:    {tonic_name} {results['tonic_scale']} "
              f"(strength {results['tonic_strength']:.3f})")
        print(f"  HPCP:     " + ' '.join(f"{results[f'hpcp_{i}']:.2f}" for i in range(12)))
        print(f"  TIV:      " + ' '.join(f"{results[f'tiv_{i}']:.3f}" for i in range(12)))
