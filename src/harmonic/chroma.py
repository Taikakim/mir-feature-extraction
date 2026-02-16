"""
Chroma Features Analysis for MIR Project (Essentia HPCP)

This module extracts average chroma features using Essentia's HPCP
(Harmonic Pitch Class Profile) algorithm, optimized for melodic content.

Uses harmonic stems (bass+other+vocals) when available to avoid drum noise.
Falls back to full_mix with a warning if stems are not available.

HPCP is tuned for melodic content:
- Frequency range 100-4000 Hz (skips sub-bass rumble and high-freq noise)
- Band preset with 500 Hz split for bass/melodic separation
- Non-linear processing to boost strong pitch classes, suppress weak ones
- Output normalized to unitSum for cross-file comparability

Dependencies:
- essentia
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


def calculate_hpcp(audio: np.ndarray,
                   sample_rate: int,
                   frame_size: int = 4096,
                   hop_size: int = 2048) -> np.ndarray:
    """
    Calculate average HPCP (Harmonic Pitch Class Profile) for audio.

    Uses Essentia's HPCP algorithm with parameters tuned for melodic content:
    - Frequency range 100-4000 Hz (melodic range, skips rumble and noise)
    - Band preset with 500 Hz split
    - Non-linear processing (boosts strong peaks, suppresses weak ones)
    - Harmonics=5 for better pitch detection on sustained sounds

    Args:
        audio: Audio signal (mono, float32)
        sample_rate: Sample rate in Hz
        frame_size: Analysis frame size
        hop_size: Hop size between frames

    Returns:
        Array of 12 chroma values that sum to 1.0
    """
    from essentia.standard import (
        Windowing, Spectrum, SpectralPeaks, HPCP, FrameGenerator
    )

    # Ensure float32 for Essentia
    audio = audio.astype(np.float32)

    # Initialize Essentia algorithms
    windowing = Windowing(type='blackmanharris62')
    spectrum = Spectrum()

    # SpectralPeaks tuned for melodic content
    spectral_peaks = SpectralPeaks(
        sampleRate=sample_rate,
        maxPeaks=100,
        magnitudeThreshold=0.00001,
        minFrequency=100,              # Skip sub-bass rumble
        maxFrequency=4000              # Focus on melodic range, skip noise/transients
    )

    # HPCP tuned for melodic content analysis
    hpcp = HPCP(
        size=12,
        sampleRate=sample_rate,
        minFrequency=100,              # Skip sub-bass
        maxFrequency=4000,             # Melodic range (A2-C8)
        bandPreset=True,               # Enable band weighting
        bandSplitFrequency=500,        # Separate bass from melodic content
        harmonics=5,                   # Include harmonics for sustained sounds
        windowSize=1.0,                # Semitone resolution
        weightType='squaredCosine',    # Sharp weighting, focus on peaks
        nonLinear=True,                # Boost strong peaks, suppress weak ones
        normalized='unitMax'           # Required for nonLinear to work
    )

    # Process frames and accumulate HPCP
    hpcp_frames = []

    for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size,
                                 startFromZero=True):
        windowed = windowing(frame)
        spec = spectrum(windowed)
        frequencies, magnitudes = spectral_peaks(spec)

        # Skip frames with no peaks
        if len(frequencies) > 0:
            frame_hpcp = hpcp(frequencies, magnitudes)
            hpcp_frames.append(frame_hpcp)

    if not hpcp_frames:
        logger.warning("No valid HPCP frames extracted, returning uniform distribution")
        return np.ones(12) / 12

    # Average across all frames
    avg_hpcp = np.mean(hpcp_frames, axis=0)

    # Essentia HPCP uses A as index 0 (A, A#, B, C, C#...)
    # We want C as index 0 (C, C#, D, D#, E...) to match standard MIR conventions
    # So we roll by -3 (left shift) to move index 3 (C) to index 0
    avg_hpcp = np.roll(avg_hpcp, -3)

    # Convert from unitMax to unitSum for cross-file comparability
    # This preserves the nonLinear processing benefits while making
    # results comparable across different audio files
    hpcp_sum = np.sum(avg_hpcp)
    if hpcp_sum > 0:
        avg_hpcp = avg_hpcp / hpcp_sum

    return avg_hpcp


def analyze_chroma(audio_path: str | Path,
                   frame_size: int = 4096,
                   hop_size: int = 2048,
                   use_stems: bool = True,
                   audio: Optional[np.ndarray] = None,
                   sr: Optional[int] = None) -> Dict[str, float]:
    """
    Analyze chroma features for an audio file using Essentia HPCP.

    When use_stems=True and the audio_path is in an organized folder,
    uses bass+other+vocals stems (excluding drums) for cleaner harmonic analysis.

    Args:
        audio_path: Path to audio file
        frame_size: Analysis frame size
        hop_size: Hop size between frames
        use_stems: Whether to try using harmonic stems (default: True)
        audio: Pre-loaded mono float32 audio array (skips disk read if provided)
        sr: Sample rate (required if audio is provided)

    Returns:
        Dictionary with chroma features:
        - chroma_0 through chroma_11: Weights for each semitone (sum to 1.0)
    """
    audio_path = Path(audio_path)

    if audio is not None:
        # Pre-loaded audio provided — use directly
        logger.info(f"Analyzing chroma (HPCP): {audio_path.name} (pre-loaded)")
        used_stems = False
    else:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Analyzing chroma (HPCP): {audio_path.name}")

        # Try to use harmonic stems if in organized folder
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

    # Calculate average HPCP
    avg_hpcp = calculate_hpcp(audio, sr, frame_size, hop_size)

    # Compile results
    results = {}
    for i in range(12):
        feature_name = f'chroma_{i}'
        results[feature_name] = float(avg_hpcp[i])

    # Clamp values to valid ranges
    for key, value in results.items():
        results[key] = clamp_feature_value(key, value)

    # Log results with pitch class names
    logger.info("HPCP pitch class distribution (melodic-tuned, unitSum normalized):")
    for i in range(12):
        pitch_class = PITCH_CLASSES[i]
        weight = results[f'chroma_{i}']
        bar = '█' * int(weight * 40)
        logger.info(f"  {pitch_class:>2}: {weight:.3f} {bar}")

    # Verify sum ≈ 1.0
    total = sum(results.values())
    logger.debug(f"Sum of chroma values: {total:.4f}")

    return results


def batch_analyze_chroma(root_directory: str | Path,
                          overwrite: bool = False,
                          frame_size: int = 4096,
                          hop_size: int = 2048,
                          use_stems: bool = True) -> dict:
    """
    Batch analyze chroma features for all organized folders.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing chroma data
        frame_size: Analysis frame size
        hop_size: Hop size between frames
        use_stems: Whether to use harmonic stems when available

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch HPCP chroma analysis: {root_directory}")

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
                avg_hpcp = calculate_hpcp(audio, sr, frame_size, hop_size)
                if used_stems:
                    stats['used_stems'] += 1
                else:
                    stats['used_full_mix'] += 1
            else:
                # Direct analysis of full_mix
                audio, sr = read_audio(str(stems['full_mix']), dtype='float32')
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                avg_hpcp = calculate_hpcp(audio, sr, frame_size, hop_size)
                stats['used_full_mix'] += 1

            # Compile results
            results = {}
            for j in range(12):
                results[f'chroma_{j}'] = float(avg_hpcp[j])

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
    logger.info("Batch HPCP Chroma Analysis Summary:")
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
        description="Analyze chroma features using Essentia HPCP (melodic-tuned)"
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
        '--frame-size',
        type=int,
        default=4096,
        help='Analysis frame size (default: 4096)'
    )

    parser.add_argument(
        '--hop-size',
        type=int,
        default=2048,
        help='Hop size in samples (default: 2048)'
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
            # Batch processing
            stats = batch_analyze_chroma(
                path,
                overwrite=args.overwrite,
                frame_size=args.frame_size,
                hop_size=args.hop_size,
                use_stems=use_stems
            )

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} folders failed to process")
                sys.exit(1)

        elif path.is_dir():
            # Single folder
            stems = get_stem_files(path, include_full_mix=True)
            if 'full_mix' not in stems:
                logger.error(f"No full_mix file found in {path}")
                sys.exit(1)

            results = analyze_chroma(
                stems['full_mix'],
                frame_size=args.frame_size,
                hop_size=args.hop_size,
                use_stems=use_stems
            )

            # Save to .INFO
            info_path = get_info_path(path)
            safe_update(info_path, results)

            print(f"\nHPCP Chroma Features (melodic-tuned, unitSum normalized, sum ≈ 1.0):")
            for i in range(12):
                pitch_class = PITCH_CLASSES[i]
                weight = results[f'chroma_{i}']
                bar = '█' * int(weight * 40)
                print(f"  {pitch_class:>2} (chroma_{i:2d}): {weight:.3f} {bar}")
            print(f"\n  Sum: {sum(results.values()):.4f}")

        else:
            # Single file
            results = analyze_chroma(
                path,
                frame_size=args.frame_size,
                hop_size=args.hop_size,
                use_stems=use_stems
            )

            print(f"\nHPCP Chroma Features (melodic-tuned, unitSum normalized, sum ≈ 1.0):")
            for i in range(12):
                pitch_class = PITCH_CLASSES[i]
                weight = results[f'chroma_{i}']
                bar = '█' * int(weight * 40)
                print(f"  {pitch_class:>2} (chroma_{i:2d}): {weight:.3f} {bar}")
            print(f"\n  Sum: {sum(results.values()):.4f}")

        logger.info("HPCP chroma analysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
