"""
Chroma Features Analysis for MIR Project (CQT-based)

Extracts average chroma features using librosa's Constant-Q Transform (CQT)
chroma, which provides logarithmic frequency resolution — each semitone gets
its own properly-sized analysis window regardless of octave. This avoids the
spectral leakage between adjacent pitch classes that STFT-based methods
suffer from at low frequencies (critical for bass-heavy EDM).

Uses harmonic stems (bass+other+vocals) when available to avoid drum noise.
Falls back to full_mix with a warning if stems are not available.

Dependencies:
- librosa
- numpy
- soundfile
- src.core.file_utils
- src.core.common

Output:
- chroma_0 through chroma_11: Pitch class weights (sum to 1.0) for each semitone
  (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import get_stem_files, read_audio
from core.common import clamp_feature_value
from core.json_handler import safe_update, get_info_path

logger = logging.getLogger(__name__)

# Pitch class names for reference
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def mix_harmonic_stems(folder_path: Path) -> Tuple[Optional[np.ndarray], int, bool]:
    """
    Mix bass + other + vocals stems (excluding drums) for harmonic analysis.

    Args:
        folder_path: Path to organized folder containing stems

    Returns:
        Tuple of (audio_array, sample_rate, used_stems)
        used_stems is True if stems were used, False if fell back to full_mix
    """
    stems = get_stem_files(folder_path, include_full_mix=True)

    # Check if we have the harmonic stems
    harmonic_stems = ['bass', 'other', 'vocals']
    available_stems = [s for s in harmonic_stems if s in stems]

    if len(available_stems) >= 2:  # Need at least 2 stems for useful mix
        logger.info(f"Using harmonic stems: {', '.join(available_stems)} (excluding drums)")

        mixed_audio = None
        sr = None

        for stem_name in available_stems:
            audio, stem_sr = sf.read(str(stems[stem_name]), dtype='float32')

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            if mixed_audio is None:
                mixed_audio = audio
                sr = stem_sr
            else:
                # Ensure same length (pad shorter with zeros)
                if len(audio) > len(mixed_audio):
                    mixed_audio = np.pad(mixed_audio, (0, len(audio) - len(mixed_audio)))
                elif len(audio) < len(mixed_audio):
                    audio = np.pad(audio, (0, len(mixed_audio) - len(audio)))
                mixed_audio = mixed_audio + audio

        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 0:
            mixed_audio = mixed_audio / max_val * 0.95

        return mixed_audio, sr, True

    # Fallback to full_mix
    if 'full_mix' in stems:
        logger.warning("Harmonic stems not available, using full_mix (quality may be compromised by drums)")
        audio, sr = read_audio(str(stems['full_mix']), dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        return audio, sr, False

    return None, 0, False


def calculate_chroma_cqt(audio: np.ndarray,
                         sample_rate: int,
                         hop_length: int = 2048) -> np.ndarray:
    """
    Calculate average CQT chroma for audio.

    CQT (Constant-Q Transform) uses logarithmic frequency spacing where each
    bin is exactly one semitone wide. This gives clean pitch-class resolution
    at all frequencies, including sub-bass, without the spectral leakage that
    STFT-based methods (HPCP) suffer from at low frequencies.

    Args:
        audio: Audio signal (mono, float32)
        sample_rate: Sample rate in Hz
        hop_length: Hop size between frames (default 2048 = ~46ms at 44.1kHz)

    Returns:
        Array of 12 chroma values (C, C#, ..., B) that sum to 1.0
    """
    import librosa

    # CQT chroma starting from C1 (~32.7 Hz) to capture bass fundamentals
    # n_octaves=7 covers C1 to B7 (~3951 Hz) — full melodic range
    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=librosa.note_to_hz('C1'),
        n_chroma=12,
        n_octaves=7,
        norm=2,  # L2 normalization per frame (reduces loudness variation)
    )

    if chroma.size == 0:
        logger.warning("No valid CQT chroma frames, returning uniform distribution")
        return np.ones(12) / 12

    # Average across all frames → 12-element vector
    avg_chroma = np.mean(chroma, axis=1)

    # Normalize to sum to 1.0 for cross-file comparability
    chroma_sum = np.sum(avg_chroma)
    if chroma_sum > 0:
        avg_chroma = avg_chroma / chroma_sum

    return avg_chroma


def analyze_chroma(audio_path: str | Path,
                   hop_length: int = 2048,
                   use_stems: bool = True,
                   audio: Optional[np.ndarray] = None,
                   sr: Optional[int] = None,
                   # Legacy params (ignored, kept for API compat)
                   frame_size: Optional[int] = None,
                   hop_size: Optional[int] = None) -> Dict[str, float]:
    """
    Analyze chroma features for an audio file using CQT chroma.

    When use_stems=True and the audio_path is in an organized folder,
    uses bass+other+vocals stems (excluding drums) for cleaner harmonic analysis.

    Args:
        audio_path: Path to audio file
        hop_length: Hop size between frames (default 2048)
        use_stems: Whether to try using harmonic stems (default: True)
        audio: Pre-loaded mono float32 audio array (skips disk read if provided)
        sr: Sample rate (required if audio is provided)

    Returns:
        Dictionary with chroma features:
        - chroma_0 through chroma_11: Weights for each semitone (sum to 1.0)
    """
    # Legacy param support
    if hop_size is not None and hop_length == 2048:
        hop_length = hop_size

    audio_path = Path(audio_path)

    if audio is not None:
        logger.info(f"Analyzing chroma (CQT): {audio_path.name} (pre-loaded)")
        used_stems = False
    else:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Analyzing chroma (CQT): {audio_path.name}")

        used_stems = False

        if use_stems and audio_path.parent.is_dir():
            audio, sr, used_stems = mix_harmonic_stems(audio_path.parent)

        # Fallback to direct file loading
        if audio is None:
            audio, sr = read_audio(audio_path, dtype='float32')
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

    logger.debug(f"Loaded audio: {len(audio)} samples @ {sr} Hz")
    if used_stems:
        logger.info("Analysis source: harmonic stems (bass+other+vocals)")
    else:
        logger.info("Analysis source: full mix")

    # Calculate average CQT chroma
    avg_chroma = calculate_chroma_cqt(audio, sr, hop_length)

    # Compile results
    results = {}
    for i in range(12):
        feature_name = f'chroma_{i}'
        results[feature_name] = float(avg_chroma[i])

    # Clamp values to valid ranges
    for key, value in results.items():
        results[key] = clamp_feature_value(key, value)

    # Log results with pitch class names
    logger.info("CQT chroma pitch class distribution (unitSum normalized):")
    for i in range(12):
        pitch_class = PITCH_CLASSES[i]
        weight = results[f'chroma_{i}']
        bar = '█' * int(weight * 40)
        logger.info(f"  {pitch_class:>2}: {weight:.3f} {bar}")

    total = sum(results.values())
    logger.debug(f"Sum of chroma values: {total:.4f}")

    return results


def batch_analyze_chroma(root_directory: str | Path,
                          overwrite: bool = False,
                          hop_length: int = 2048,
                          use_stems: bool = True,
                          # Legacy params (ignored)
                          frame_size: Optional[int] = None,
                          hop_size: Optional[int] = None) -> dict:
    """
    Batch analyze chroma features for all organized folders.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing chroma data
        hop_length: Hop size between frames
        use_stems: Whether to use harmonic stems when available

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    if hop_size is not None and hop_length == 2048:
        hop_length = hop_size

    root_directory = Path(root_directory)
    logger.info(f"Starting batch CQT chroma analysis: {root_directory}")

    folders = find_organized_folders(root_directory)

    stats = {
        'total': len(folders),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'used_stems': 0,
        'used_full_mix': 0,
        'errors': []
    }

    logger.info(f"Found {stats['total']} organized folders")

    for i, folder in enumerate(folders, 1):
        logger.info(f"Processing {i}/{stats['total']}: {folder.name}")

        # Find full_mix file
        stems = get_stem_files(folder, include_full_mix=True)
        if 'full_mix' not in stems:
            logger.warning(f"No full_mix found in {folder.name}")
            stats['failed'] += 1
            continue

        # Check if already processed - must check ALL output keys
        info_path = get_info_path(stems['full_mix'])
        from core.json_handler import should_process
        CHROMA_KEYS = [f'chroma_{i}' for i in range(12)]
        if not should_process(info_path, CHROMA_KEYS, overwrite):
            logger.info("Chroma data already exists. Use --overwrite to regenerate.")
            stats['skipped'] += 1
            continue

        try:
            # Try to use stems
            audio, sr, used_stems = mix_harmonic_stems(folder) if use_stems else (None, 0, False)

            if audio is not None:
                avg_chroma = calculate_chroma_cqt(audio, sr, hop_length)
                if used_stems:
                    stats['used_stems'] += 1
                else:
                    stats['used_full_mix'] += 1
            else:
                # Direct analysis of full_mix
                audio, sr = read_audio(str(stems['full_mix']), dtype='float32')
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                avg_chroma = calculate_chroma_cqt(audio, sr, hop_length)
                stats['used_full_mix'] += 1

            # Compile results
            results = {}
            for j in range(12):
                results[f'chroma_{j}'] = float(avg_chroma[j])

            # Save to .INFO file
            safe_update(info_path, results)

            stats['success'] += 1

            # Log dominant pitch classes
            sorted_chroma = sorted(results.items(), key=lambda x: x[1], reverse=True)
            top_3 = [(int(k.split('_')[1]), v) for k, v in sorted_chroma[:3]]
            top_3_str = ', '.join([f"{PITCH_CLASSES[idx]}:{val:.3f}" for idx, val in top_3])
            logger.info(f"Top 3 pitch classes: {top_3_str}")

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Batch CQT Chroma Analysis Summary:")
    logger.info(f"  Total folders:  {stats['total']}")
    logger.info(f"  Successful:     {stats['success']}")
    logger.info(f"    Used stems:   {stats['used_stems']}")
    logger.info(f"    Used full_mix: {stats['used_full_mix']}")
    logger.info(f"  Skipped:        {stats['skipped']}")
    logger.info(f"  Failed:         {stats['failed']}")
    logger.info("=" * 60)

    return stats


# Command-line interface
if __name__ == "__main__":
    import argparse
    from core.common import setup_logging

    parser = argparse.ArgumentParser(
        description="Analyze chroma features using CQT (Constant-Q Transform)"
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
        '--hop-length',
        type=int,
        default=2048,
        help='Hop length in samples (default: 2048)'
    )

    parser.add_argument(
        '--no-stems',
        action='store_true',
        help='Disable using harmonic stems, always use full_mix'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing chroma data'
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

    path = Path(args.path)

    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)

    use_stems = not args.no_stems

    try:
        if args.batch:
            stats = batch_analyze_chroma(
                path,
                overwrite=args.overwrite,
                hop_length=args.hop_length,
                use_stems=use_stems
            )

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} folders failed to process")
                sys.exit(1)

        elif path.is_dir():
            stems = get_stem_files(path, include_full_mix=True)
            if 'full_mix' not in stems:
                logger.error(f"No full_mix file found in {path}")
                sys.exit(1)

            results = analyze_chroma(
                stems['full_mix'],
                hop_length=args.hop_length,
                use_stems=use_stems
            )

            # Save to .INFO
            info_path = get_info_path(path)
            safe_update(info_path, results)

            print(f"\nCQT Chroma Features (unitSum normalized, sum = 1.0):")
            for i in range(12):
                pitch_class = PITCH_CLASSES[i]
                weight = results[f'chroma_{i}']
                bar = '█' * int(weight * 40)
                print(f"  {pitch_class:>2} (chroma_{i:2d}): {weight:.3f} {bar}")
            print(f"\n  Sum: {sum(results.values()):.4f}")

        else:
            results = analyze_chroma(
                path,
                hop_length=args.hop_length,
                use_stems=use_stems
            )

            print(f"\nCQT Chroma Features (unitSum normalized, sum = 1.0):")
            for i in range(12):
                pitch_class = PITCH_CLASSES[i]
                weight = results[f'chroma_{i}']
                bar = '█' * int(weight * 40)
                print(f"  {pitch_class:>2} (chroma_{i:2d}): {weight:.3f} {bar}")
            print(f"\n  Sum: {sum(results.values()):.4f}")

        logger.info("CQT chroma analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
