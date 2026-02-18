#!/usr/bin/env python3
"""
Stem Quality Analysis Tool

Scans separated stems and identifies tracks where a stem likely contains only
residual noise or spectral leakage from the source separation process.

When a stem is flagged:
  - It should be considered for mixing back into the 'other' stem to keep
    a faithful representation of the full mix content.

Metrics computed per stem:
  1. Crest factor (dB)      — peak / RMS; very low = flat/noisy signal
  2. RMS loudness (dBFS)    — overall energy; near-silence = likely empty
  3. Peak value (dBFS)      — max absolute sample level
  4. Spectral flatness (0-1) — Wiener entropy; high = noise-like
  5. Temporal sparsity (0-1) — fraction of frames < -60 dBFS; high + low RMS = transient residuals
  6. Cross-correlation       — correlation with full_mix; very high = bleed

A stem is flagged when ALL THREE primary criteria are satisfied (AND):
  crest_factor < threshold  AND  rms < threshold  AND  peak < threshold

Usage:
    python src/tools/stem_quality_analysis.py --config config/master_pipeline.yaml
    python src/tools/stem_quality_analysis.py --config config/master_pipeline.yaml --crest -35 --rms -35 --peak -15
    python src/tools/stem_quality_analysis.py --dir /path/to/separated/stems
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- path setup ---------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import find_organized_folders, get_stem_files, AUDIO_EXTENSIONS
from core.common import setup_logging

logger = logging.getLogger(__name__)

# ==============================================================================
# Default thresholds
# ==============================================================================
DEFAULT_CREST_FACTOR_THRESHOLD = -40.0   # dB
DEFAULT_RMS_THRESHOLD          = -39.0   # dBFS
DEFAULT_PEAK_THRESHOLD         = -20.0   # dBFS

# Stems of interest (guitar/piano are already merged into 'other')
STEMS_TO_ANALYSE = ['vocals', 'bass', 'drums']

# ==============================================================================
# Audio analysis helpers
# ==============================================================================

def _load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio file as float32 numpy array, returns (audio, sample_rate).

    Tries soundfile first (fast, handles WAV/FLAC/OGG), then falls back to
    librosa (handles M4A/AAC/MP3 via audioread/ffmpeg).
    """
    # Fast path: soundfile (libsndfile) — handles WAV, FLAC, OGG, AIFF
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype='float32')
        return audio, sr
    except Exception:
        pass

    # Fallback: librosa (uses audioread → ffmpeg under the hood)
    try:
        import librosa
        audio, sr = librosa.load(str(path), sr=None, mono=False)
        # librosa returns (channels, samples) for stereo — transpose to (samples, channels)
        if audio.ndim > 1:
            audio = audio.T
        return audio.astype(np.float32), sr
    except Exception:
        pass

    raise RuntimeError(f"Could not load audio: {path.name} (unsupported format or missing ffmpeg)")


def _db(value: float, ref: float = 1.0) -> float:
    """Convert linear value to dB, clamped to avoid -inf."""
    if value <= 0:
        return -120.0
    return 20.0 * np.log10(value / ref)


def compute_stem_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Compute quality-relevant features for a stem signal.

    Args:
        audio: Audio signal (samples,) or (samples, channels)
        sr: Sample rate

    Returns:
        Dict with keys: rms_db, peak_db, crest_factor_db,
                        spectral_flatness, temporal_sparsity
    """
    # Work in mono for analysis
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio

    # ----- Basic amplitude stats -----
    rms = np.sqrt(np.mean(mono ** 2))
    peak = np.max(np.abs(mono))
    rms_db = _db(rms)
    peak_db = _db(peak)

    # Crest factor = peak / rms (in dB)
    if rms > 0:
        crest_factor_db = _db(peak / rms)
    else:
        crest_factor_db = -120.0

    # ----- Spectral flatness (Wiener entropy) -----
    # Use short-time frames
    frame_len = min(2048, len(mono))
    hop = frame_len // 2
    n_frames = max(1, (len(mono) - frame_len) // hop)
    flatness_values = []

    for i in range(n_frames):
        start = i * hop
        frame = mono[start : start + frame_len]
        spectrum = np.abs(np.fft.rfft(frame)) ** 2 + 1e-20

        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        if arith_mean > 0:
            flatness_values.append(geo_mean / arith_mean)

    spectral_flatness = float(np.mean(flatness_values)) if flatness_values else 0.0

    # ----- Temporal sparsity (fraction of silent frames) -----
    silence_threshold_linear = 10 ** (-60.0 / 20.0)  # -60 dBFS
    silent_frames = 0
    total_frames = 0

    for i in range(n_frames):
        start = i * hop
        frame = mono[start : start + frame_len]
        frame_rms = np.sqrt(np.mean(frame ** 2))
        total_frames += 1
        if frame_rms < silence_threshold_linear:
            silent_frames += 1

    temporal_sparsity = silent_frames / max(1, total_frames)

    return {
        'rms_db': rms_db,
        'peak_db': peak_db,
        'crest_factor_db': crest_factor_db,
        'spectral_flatness': spectral_flatness,
        'temporal_sparsity': temporal_sparsity,
    }


def compute_cross_correlation(stem_audio: np.ndarray, mix_audio: np.ndarray) -> float:
    """
    Compute normalised cross-correlation between a stem and the full mix.

    Returns:
        Correlation coefficient in [0, 1].  Values near 1 mean the stem is
        almost identical to the mix (= everything leaked in).
    """
    if stem_audio.ndim > 1:
        stem_mono = np.mean(stem_audio, axis=1)
    else:
        stem_mono = stem_audio

    if mix_audio.ndim > 1:
        mix_mono = np.mean(mix_audio, axis=1)
    else:
        mix_mono = mix_audio

    # Align lengths
    min_len = min(len(stem_mono), len(mix_mono))
    stem_mono = stem_mono[:min_len]
    mix_mono = mix_mono[:min_len]

    # Pearson correlation
    stem_mean = np.mean(stem_mono)
    mix_mean = np.mean(mix_mono)
    stem_centered = stem_mono - stem_mean
    mix_centered = mix_mono - mix_mean

    denom = np.sqrt(np.sum(stem_centered ** 2) * np.sum(mix_centered ** 2))
    if denom < 1e-12:
        return 0.0

    return float(np.abs(np.sum(stem_centered * mix_centered) / denom))


# ==============================================================================
# Main analysis
# ==============================================================================

def analyse_folder(
    folder: Path,
    crest_threshold: float,
    rms_threshold: float,
    peak_threshold: float,
) -> Dict[str, dict]:
    """
    Analyse all stems in an organised folder.

    Returns:
        Dict mapping stem_name -> {features..., flagged: bool}
        Empty dict if folder has no stems.
    """
    stems = get_stem_files(folder, include_full_mix=True)
    if not stems:
        return {}

    # Load full mix for cross-correlation (optional)
    mix_audio = None
    mix_sr = None
    if 'full_mix' in stems:
        try:
            mix_audio, mix_sr = _load_audio(stems['full_mix'])
        except Exception:
            pass

    results = {}
    for stem_name in STEMS_TO_ANALYSE:
        if stem_name not in stems:
            continue

        try:
            audio, sr = _load_audio(stems[stem_name])
        except Exception as e:
            logger.warning(f"  Could not load {stem_name} in {folder.name}: {e}")
            continue

        features = compute_stem_features(audio, sr)

        # Cross-correlation with full mix
        xcorr = 0.0
        if mix_audio is not None:
            try:
                xcorr = compute_cross_correlation(audio, mix_audio)
            except Exception:
                pass
        features['cross_correlation'] = xcorr

        # Flag check (AND logic)
        flagged = (
            features['crest_factor_db'] < crest_threshold
            and features['rms_db'] < rms_threshold
            and features['peak_db'] < peak_threshold
        )
        features['flagged'] = flagged
        results[stem_name] = features

    return results


def run_analysis(
    root_dir: Path,
    output_path: Path,
    crest_threshold: float = DEFAULT_CREST_FACTOR_THRESHOLD,
    rms_threshold: float = DEFAULT_RMS_THRESHOLD,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
) -> Dict[str, List[Tuple[str, dict]]]:
    """
    Run stem quality analysis across all organised folders.

    Args:
        root_dir: Root directory containing organised folders with stems
        output_path: Path for the output text report
        crest_threshold: Crest factor threshold (dB)
        rms_threshold: RMS threshold (dBFS)
        peak_threshold: Peak threshold (dBFS)

    Returns:
        Dict mapping stem_name -> list of (folder_name, features_dict) for flagged stems
    """
    folders = find_organized_folders(root_dir, deduplicate=True)
    logger.info(f"Found {len(folders)} organised folders to scan")

    # Accumulate results grouped by stem type
    flagged: Dict[str, List[Tuple[str, dict]]] = {s: [] for s in STEMS_TO_ANALYSE}
    scanned = 0

    for i, folder in enumerate(folders, 1):
        if i % 50 == 0 or i == len(folders):
            logger.info(f"  Scanning: {i}/{len(folders)}")

        results = analyse_folder(folder, crest_threshold, rms_threshold, peak_threshold)
        if results:
            scanned += 1
            for stem_name, features in results.items():
                if features.get('flagged'):
                    flagged[stem_name].append((folder.name, features))

    # ----- Write report -----
    total_flagged = sum(len(v) for v in flagged.values())
    lines = []
    lines.append("=" * 70)
    lines.append("STEM QUALITY ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"Scanned: {scanned} tracks")
    lines.append(f"Total flagged: {total_flagged}")
    lines.append(f"Thresholds: crest_factor < {crest_threshold:.1f} dB, "
                 f"rms < {rms_threshold:.1f} dBFS, peak < {peak_threshold:.1f} dBFS")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for stem_name in STEMS_TO_ANALYSE:
        entries = flagged[stem_name]
        lines.append("=" * 70)
        lines.append(f"=== {stem_name.upper()} === [{len(entries)} flagged]")
        lines.append("=" * 70)

        if not entries:
            lines.append("  (none)")
        else:
            # Sort by RMS (quietest first)
            entries.sort(key=lambda x: x[1]['rms_db'])
            for folder_name, feat in entries:
                lines.append(f"  {folder_name}")
                lines.append(
                    f"    crest={feat['crest_factor_db']:+.1f}dB  "
                    f"rms={feat['rms_db']:+.1f}dBFS  "
                    f"peak={feat['peak_db']:+.1f}dBFS  "
                    f"spectral_flatness={feat['spectral_flatness']:.3f}  "
                    f"silent_frames={feat['temporal_sparsity'] * 100:.0f}%  "
                    f"xcorr={feat['cross_correlation']:.3f}"
                )
        lines.append("")

    report = "\n".join(lines)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"Report written to {output_path}")
    logger.info(f"Total flagged stems: {total_flagged}")

    # Also print to console
    print("\n" + report)

    return flagged


# ==============================================================================
# Mixback utility (for pipeline integration)
# ==============================================================================

def mixback_flagged_stems(
    root_dir: Path,
    crest_threshold: float = DEFAULT_CREST_FACTOR_THRESHOLD,
    rms_threshold: float = DEFAULT_RMS_THRESHOLD,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
) -> int:
    """
    For each flagged stem, add its audio to the 'other' stem and zero out the
    original stem file (preserving the file for downstream compatibility).

    Returns:
        Number of stems mixed back.
    """
    import soundfile as sf

    folders = find_organized_folders(root_dir, deduplicate=True)
    mixed_count = 0

    for folder in folders:
        results = analyse_folder(folder, crest_threshold, rms_threshold, peak_threshold)
        stems = get_stem_files(folder)
        if 'other' not in stems:
            continue

        for stem_name, features in results.items():
            if not features.get('flagged') or stem_name not in stems:
                continue

            stem_path = stems[stem_name]
            other_path = stems['other']

            try:
                stem_audio, sr = _load_audio(stem_path)
                other_audio, _ = _load_audio(other_path)

                # Align lengths
                min_len = min(len(stem_audio), len(other_audio))
                combined = other_audio[:min_len] + stem_audio[:min_len]

                # Write updated other stem
                sf.write(str(other_path), combined, sr)

                # Zero out the flagged stem (keep file for pipeline compatibility)
                zeroed = np.zeros_like(stem_audio)
                sf.write(str(stem_path), zeroed, sr)

                logger.info(f"Mixed back {stem_name} -> other in {folder.name}")
                mixed_count += 1
            except Exception as e:
                logger.error(f"Failed to mixback {stem_name} in {folder.name}: {e}")

    return mixed_count


# ==============================================================================
# Vocal removal utility
# ==============================================================================

def create_instrumental(
    folder: Path,
    method: str = 'auto',
    backend: str = 'bs_roformer',
) -> Optional[Path]:
    """
    Create an instrumental version of the full mix by removing vocals.

    Methods:
        'auto'   — uses backend-appropriate strategy
        'invert' — always subtracts vocal stem from full_mix (works with any backend)

    For Demucs backend (auto): sums drums + bass + other stems
    For BS-RoFormer backend (auto): subtracts vocal stem from full_mix

    Returns:
        Path to the created instrumental file, or None on failure.
    """
    import soundfile as sf

    stems = get_stem_files(folder, include_full_mix=True)
    if 'full_mix' not in stems or 'vocals' not in stems:
        logger.warning(f"Missing full_mix or vocals in {folder.name}, skipping vocal removal")
        return None

    full_mix_path = stems['full_mix']
    ext = full_mix_path.suffix
    output_path = folder / f"full_mix_instrumental{ext}"

    try:
        if method == 'invert' or (method == 'auto' and backend == 'bs_roformer'):
            # Subtract vocals from full mix
            mix_audio, sr = _load_audio(full_mix_path)
            vocal_audio, _ = _load_audio(stems['vocals'])

            min_len = min(len(mix_audio), len(vocal_audio))
            instrumental = mix_audio[:min_len] - vocal_audio[:min_len]
            sf.write(str(output_path), instrumental, sr)

        elif method == 'auto' and backend == 'demucs':
            # Sum non-vocal stems (Demucs is designed for clean reconstruction)
            instrumental = None
            sr = None
            for stem_name in ['drums', 'bass', 'other']:
                if stem_name not in stems:
                    logger.warning(f"Missing {stem_name} stem in {folder.name}")
                    continue
                audio, stem_sr = _load_audio(stems[stem_name])
                sr = stem_sr
                if instrumental is None:
                    instrumental = audio.copy()
                else:
                    min_len = min(len(instrumental), len(audio))
                    instrumental = instrumental[:min_len] + audio[:min_len]

            if instrumental is not None and sr is not None:
                sf.write(str(output_path), instrumental, sr)
            else:
                logger.error(f"No stems to sum for instrumental in {folder.name}")
                return None
        else:
            logger.error(f"Unknown vocal removal method: {method}")
            return None

        logger.info(f"Created instrumental: {output_path.name}")
        return output_path

    except Exception as e:
        logger.error(f"Vocal removal failed for {folder.name}: {e}")
        return None


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyse stem quality and detect residual/leaked stems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--config', type=str, help='Path to master_pipeline.yaml (reads output dir from config)')
    parser.add_argument('--dir', type=str, help='Direct path to organised stems directory (overrides config)')
    parser.add_argument('--output', type=str, default=None, help='Output report path (default: stem_potential_issues.txt in project root)')

    # Thresholds
    parser.add_argument('--crest', type=float, default=DEFAULT_CREST_FACTOR_THRESHOLD,
                        help=f'Crest factor threshold in dB (default: {DEFAULT_CREST_FACTOR_THRESHOLD})')
    parser.add_argument('--rms', type=float, default=DEFAULT_RMS_THRESHOLD,
                        help=f'RMS threshold in dBFS (default: {DEFAULT_RMS_THRESHOLD})')
    parser.add_argument('--peak', type=float, default=DEFAULT_PEAK_THRESHOLD,
                        help=f'Peak threshold in dBFS (default: {DEFAULT_PEAK_THRESHOLD})')

    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    # Determine root directory
    root_dir = None
    if args.dir:
        root_dir = Path(args.dir)
    elif args.config:
        try:
            import yaml
            with open(args.config, 'r') as f:
                cfg = yaml.safe_load(f)
            paths = cfg.get('paths', {})
            output_path = paths.get('output')
            if output_path:
                root_dir = Path(output_path)
            else:
                root_dir = Path(paths.get('input', '.'))
        except Exception as e:
            print(f"Error reading config: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.error("Either --config or --dir is required")

    if not root_dir or not root_dir.exists():
        print(f"Error: Directory does not exist: {root_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        report_path = Path(args.output)
    else:
        # Default: project root / stem_potential_issues.txt
        project_root = Path(__file__).parent.parent.parent
        report_path = project_root / 'stem_potential_issues.txt'

    logger.info(f"Stem quality analysis")
    logger.info(f"  Root:   {root_dir}")
    logger.info(f"  Output: {report_path}")
    logger.info(f"  Thresholds: crest<{args.crest}dB, rms<{args.rms}dBFS, peak<{args.peak}dBFS")

    run_analysis(
        root_dir=root_dir,
        output_path=report_path,
        crest_threshold=args.crest,
        rms_threshold=args.rms,
        peak_threshold=args.peak,
    )


if __name__ == '__main__':
    main()
