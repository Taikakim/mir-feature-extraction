#!/usr/bin/env python3
"""
Comprehensive Feature Extraction Test

Tests ALL 70+ MIR features on a single audio file with timing information.
Uses GGUF for Music Flamingo (7x faster than transformers).

Usage:
    python src/test_all_features.py "/path/to/audio.flac"
    python src/test_all_features.py "/path/to/organized_folder/"
    python src/test_all_features.py "/path/to/audio.flac" --model Q8_0
    python src/test_all_features.py "/path/to/audio.flac" --skip-flamingo
    python src/test_all_features.py "/path/to/audio.flac" --skip-demucs
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))

from core.common import setup_logging, FEATURE_RANGES
from core.file_utils import get_stem_files
from core.json_handler import safe_update, get_info_path
import soundfile as sf

logger = logging.getLogger(__name__)


class FeatureTester:
    """Comprehensive feature extraction tester."""

    def __init__(self, audio_path: Path, skip_demucs: bool = False, skip_flamingo: bool = False,
                 flamingo_model: str = 'Q6_K'):
        self.audio_path = audio_path
        self.skip_demucs = skip_demucs
        self.skip_flamingo = skip_flamingo
        self.flamingo_model = flamingo_model

        self.timings: Dict[str, float] = {}
        self.feature_counts: Dict[str, int] = {}
        self.errors: list = []

        # Determine if already organized
        if audio_path.is_dir():
            self.folder = audio_path
            stems = get_stem_files(self.folder, include_full_mix=True)
            if 'full_mix' not in stems:
                raise FileNotFoundError(f"No full_mix found in {audio_path}")
            self.full_mix = stems['full_mix']
        elif audio_path.name.startswith('full_mix.'):
            self.folder = audio_path.parent
            self.full_mix = audio_path
        else:
            # Need to organize first
            self.folder = None
            self.full_mix = audio_path

        # Get duration
        info = sf.info(str(self.full_mix))
        self.duration = info.duration
        self.sample_rate = info.samplerate

    def time_feature(self, name: str, func, *args, **kwargs) -> Tuple[float, Any]:
        """Time a feature extraction function."""
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            self.timings[name] = elapsed

            # Count features if result is a dict
            if isinstance(result, dict):
                count = len([k for k in result.keys() if not k.startswith('_')])
                self.feature_counts[name] = count

            speed = self.duration / elapsed if elapsed > 0 else float('inf')
            logger.info(f"  {name:45s} {elapsed:7.2f}s  {speed:8.1f}x realtime")
            return elapsed, result

        except Exception as e:
            logger.error(f"  {name:45s} FAILED: {e}")
            self.errors.append(f"{name}: {str(e)}")
            return 0, None

    def organize_file(self) -> bool:
        """Organize the audio file if needed."""
        if self.folder is not None:
            logger.info(f"Using existing folder: {self.folder.name}")
            return True

        logger.info("Organizing file...")
        from preprocessing.file_organizer import organize_file
        success, message = organize_file(self.audio_path, output_dir=None, move=False)

        if success:
            self.folder = self.audio_path.parent / self.audio_path.stem
            stems = get_stem_files(self.folder, include_full_mix=True)
            self.full_mix = stems['full_mix']
            logger.info(f"  Organized to: {self.folder.name}")
            return True
        else:
            logger.error(f"  Failed: {message}")
            return False

    def run_demucs(self) -> bool:
        """Run Demucs stem separation."""
        if self.skip_demucs:
            logger.info("  Skipping Demucs (--skip-demucs)")
            return True

        # Check if stems already exist
        stems = get_stem_files(self.folder, include_full_mix=True)
        if 'drums' in stems and 'bass' in stems:
            logger.info("  Stems already exist, skipping Demucs")
            return True

        from preprocessing.demucs_sep import separate_stems

        def do_demucs():
            return separate_stems(self.full_mix, output_dir=self.folder)

        elapsed, result = self.time_feature("Demucs Stem Separation", do_demucs)
        return result is not None

    def run_loudness(self) -> Dict:
        """Run loudness analysis (LUFS/LRA)."""
        from timbral.loudness import analyze_folder_loudness

        def do_loudness():
            return analyze_folder_loudness(self.folder)

        elapsed, result = self.time_feature("Loudness (LUFS/LRA)", do_loudness)
        return result or {}

    def run_beat_grid(self) -> Dict:
        """Run beat grid detection (saves .BEATS_GRID file)."""
        from rhythm.beat_grid import create_beat_grid

        def do_beat_grid():
            beat_times, grid_path = create_beat_grid(self.full_mix, save_grid=True)
            # Return beat count as a dict (beat grid itself is saved to file)
            return {'_beat_count': len(beat_times)}

        elapsed, result = self.time_feature("Beat Grid Detection", do_beat_grid)
        # Don't return internal _beat_count to avoid saving it
        return {}

    def run_bpm(self) -> Dict:
        """Run BPM analysis."""
        from rhythm.bpm import analyze_folder_bpm

        def do_bpm():
            return analyze_folder_bpm(self.folder)

        elapsed, result = self.time_feature("BPM Analysis", do_bpm)
        return result or {}

    def run_onsets(self) -> Dict:
        """Run onset detection (saves .ONSETS file for syncopation)."""
        from rhythm.onsets import analyze_onsets_with_save
        from core.json_handler import safe_update, get_info_path

        def do_onsets():
            results, onsets_path = analyze_onsets_with_save(self.full_mix, save_onsets=True)
            # Save to INFO file
            info_path = get_info_path(self.full_mix)
            safe_update(info_path, results)
            return results

        elapsed, result = self.time_feature("Onset Detection", do_onsets)
        return result or {}

    def run_syncopation(self) -> Dict:
        """Run syncopation analysis (requires .BEATS_GRID and .ONSETS files)."""
        from rhythm.syncopation import analyze_syncopation
        from core.json_handler import safe_update, get_info_path

        def do_syncopation():
            results = analyze_syncopation(self.folder)
            # Save to INFO file
            info_path = get_info_path(self.full_mix)
            safe_update(info_path, results)
            return results

        elapsed, result = self.time_feature("Syncopation Analysis", do_syncopation)
        return result or {}

    def run_spectral_features(self) -> Dict:
        """Run spectral feature extraction."""
        from spectral.spectral_features import analyze_spectral_features
        from core.json_handler import safe_update, get_info_path

        def do_spectral():
            results = analyze_spectral_features(self.full_mix)
            # Save to INFO file
            info_path = get_info_path(self.full_mix)
            safe_update(info_path, results)
            return results

        elapsed, result = self.time_feature("Spectral Features", do_spectral)
        return result or {}

    def run_multiband_rms(self) -> Dict:
        """Run multiband RMS analysis."""
        from spectral.multiband_rms import analyze_multiband_rms
        from core.json_handler import safe_update, get_info_path

        def do_multiband():
            results = analyze_multiband_rms(self.full_mix)
            # Save to INFO file
            info_path = get_info_path(self.full_mix)
            safe_update(info_path, results)
            return results

        elapsed, result = self.time_feature("Multiband RMS", do_multiband)
        return result or {}

    def run_chroma(self) -> Dict:
        """Run chroma feature extraction."""
        from harmonic.chroma import analyze_chroma
        from core.json_handler import safe_update, get_info_path

        def do_chroma():
            results = analyze_chroma(self.full_mix)
            # Save to INFO file
            info_path = get_info_path(self.full_mix)
            safe_update(info_path, results)
            return results

        elapsed, result = self.time_feature("Chroma Features", do_chroma)
        return result or {}

    def run_timbral(self) -> Dict:
        """Run Audio Commons timbral analysis."""
        from timbral.audio_commons import analyze_folder_timbral_features

        def do_timbral():
            return analyze_folder_timbral_features(self.folder)

        elapsed, result = self.time_feature("Timbral Features (Audio Commons)", do_timbral)
        return result or {}

    def run_essentia(self) -> Dict:
        """Run Essentia feature extraction."""
        try:
            from classification.essentia_features import analyze_folder_essentia_features

            def do_essentia():
                return analyze_folder_essentia_features(self.folder)

            elapsed, result = self.time_feature("Essentia Features", do_essentia)
            return result or {}
        except ImportError as e:
            logger.warning(f"  Essentia not available: {e}")
            return {}

    def run_per_stem_rhythm(self) -> Dict:
        """Run per-stem rhythm analysis."""
        # Check if stems exist
        stems = get_stem_files(self.folder, include_full_mix=False)
        if not stems:
            logger.info("  Skipping per-stem rhythm (no stems)")
            return {}

        from rhythm.per_stem_rhythm import analyze_per_stem_rhythm
        from core.json_handler import safe_update, get_info_path

        def do_per_stem():
            results = analyze_per_stem_rhythm(self.folder)
            # Save to INFO file
            info_path = get_info_path(self.full_mix)
            safe_update(info_path, results)
            return results

        elapsed, result = self.time_feature("Per-Stem Rhythm", do_per_stem)
        return result or {}

    def run_per_stem_harmonic(self) -> Dict:
        """Run per-stem harmonic analysis."""
        stems = get_stem_files(self.folder, include_full_mix=False)
        if not stems:
            logger.info("  Skipping per-stem harmonic (no stems)")
            return {}

        from harmonic.per_stem_harmonic import analyze_per_stem_harmonics
        from core.json_handler import safe_update, get_info_path

        def do_per_stem():
            results = analyze_per_stem_harmonics(self.folder)
            # Save to INFO file
            info_path = get_info_path(self.full_mix)
            safe_update(info_path, results)
            return results

        elapsed, result = self.time_feature("Per-Stem Harmonic", do_per_stem)
        return result or {}

    def run_music_flamingo(self) -> Dict:
        """Run Music Flamingo GGUF analysis (all 5 prompts)."""
        if self.skip_flamingo:
            logger.info("  Skipping Music Flamingo (--skip-flamingo)")
            return {}

        try:
            from classification.music_flamingo import MusicFlamingoGGUF, DEFAULT_PROMPTS
            from core.json_handler import safe_update, get_info_path

            logger.info(f"  Using GGUF model: {self.flamingo_model}")

            analyzer = MusicFlamingoGGUF(model=self.flamingo_model)
            results = {}

            for prompt_type in DEFAULT_PROMPTS.keys():
                def do_flamingo(pt=prompt_type):
                    return analyzer.analyze(self.full_mix, prompt_type=pt)

                elapsed, result = self.time_feature(
                    f"Music Flamingo ({prompt_type})",
                    do_flamingo
                )
                if result:
                    results[f'music_flamingo_{prompt_type}'] = result
                    logger.info(f"    {prompt_type}: {len(result)} chars")
                else:
                    logger.warning(f"    {prompt_type}: empty result")

            # Save results to .INFO file
            if results:
                info_path = get_info_path(self.full_mix)
                safe_update(info_path, results)
                logger.info(f"  Saved {len(results)} Music Flamingo descriptions to {info_path.name}")

            return results

        except FileNotFoundError as e:
            logger.warning(f"  Music Flamingo GGUF not available: {e}")
            logger.info("  Build llama.cpp with: cmake .. -DGGML_HIP=ON && cmake --build . --target llama-mtmd-cli")
            return {}
        except Exception as e:
            logger.error(f"  Music Flamingo failed: {e}")
            return {}

    def run_all(self) -> Dict[str, Any]:
        """Run all feature extraction modules."""
        logger.info("=" * 80)
        logger.info(f"COMPREHENSIVE FEATURE EXTRACTION TEST")
        logger.info("=" * 80)
        logger.info(f"Audio: {self.full_mix.name}")
        logger.info(f"Duration: {self.duration:.2f}s ({self.duration/60:.2f} min)")
        logger.info(f"Sample Rate: {self.sample_rate} Hz")
        logger.info("=" * 80)

        all_results = {}

        # 1. Organize file if needed
        logger.info("\n[1/13] File Organization")
        if not self.organize_file():
            return all_results

        # 2. Demucs stem separation
        logger.info("\n[2/13] Stem Separation")
        self.run_demucs()

        # 3. Loudness analysis
        logger.info("\n[3/13] Loudness Analysis")
        all_results.update(self.run_loudness())

        # 4. Beat grid detection
        logger.info("\n[4/13] Beat Grid Detection")
        all_results.update(self.run_beat_grid())

        # 5. BPM analysis
        logger.info("\n[5/13] BPM Analysis")
        all_results.update(self.run_bpm())

        # 6. Onset detection
        logger.info("\n[6/13] Onset Detection")
        all_results.update(self.run_onsets())

        # 7. Syncopation
        logger.info("\n[7/13] Syncopation Analysis")
        all_results.update(self.run_syncopation())

        # 8. Spectral features
        logger.info("\n[8/13] Spectral Features")
        all_results.update(self.run_spectral_features())

        # 9. Multiband RMS
        logger.info("\n[9/13] Multiband RMS")
        all_results.update(self.run_multiband_rms())

        # 10. Chroma
        logger.info("\n[10/13] Chroma Features")
        all_results.update(self.run_chroma())

        # 11. Timbral features
        logger.info("\n[11/13] Timbral Features")
        all_results.update(self.run_timbral())

        # 12. Essentia
        logger.info("\n[12/13] Essentia Features")
        all_results.update(self.run_essentia())

        # 13. Per-stem features (rhythm + harmonic)
        logger.info("\n[13/13] Per-Stem Analysis")
        all_results.update(self.run_per_stem_rhythm())
        all_results.update(self.run_per_stem_harmonic())

        # 14. Music Flamingo (GGUF)
        logger.info("\n[BONUS] Music Flamingo GGUF")
        all_results.update(self.run_music_flamingo())

        return all_results

    def print_summary(self):
        """Print timing and feature summary."""
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)

        # Timing summary
        total_time = sum(self.timings.values())
        logger.info(f"\nAudio duration: {self.duration:.2f}s ({self.duration/60:.2f} min)")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Overall speed: {self.duration/total_time:.2f}x realtime" if total_time > 0 else "N/A")

        logger.info(f"\n{'Module':<45s} {'Time':>8s}  {'Speed':>12s}")
        logger.info("-" * 70)

        for name, t in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
            speed = self.duration / t if t > 0 else float('inf')
            logger.info(f"{name:<45s} {t:>7.2f}s  {speed:>10.1f}x")

        # Feature count
        total_numeric = len([k for k in FEATURE_RANGES.keys()])
        logger.info(f"\n{'='*70}")
        logger.info(f"Total numeric features defined: {total_numeric}")

        # Check INFO file
        info_path = get_info_path(self.full_mix)
        if info_path.exists():
            with open(info_path, 'r') as f:
                data = json.load(f)
            numeric_count = len([k for k, v in data.items()
                                if isinstance(v, (int, float)) and not k.startswith('_')])
            text_count = len([k for k, v in data.items()
                             if isinstance(v, str) and not k.startswith('_')])
            logger.info(f"Extracted numeric features: {numeric_count}")
            logger.info(f"Extracted text features: {text_count}")
            logger.info(f"Total features in INFO: {len(data)}")

        # Errors
        if self.errors:
            logger.info(f"\n{'='*70}")
            logger.warning(f"Errors ({len(self.errors)}):")
            for err in self.errors:
                logger.warning(f"  - {err}")

        logger.info("\n" + "=" * 80)
        logger.info(f"Results saved to: {info_path}")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test all 70+ MIR features on a single audio file"
    )
    parser.add_argument('audio_path', help='Path to audio file or organized folder')
    parser.add_argument('--model', default='Q6_K', choices=['IQ3_M', 'Q6_K', 'Q8_0'],
                        help='GGUF quantization level for Music Flamingo')
    parser.add_argument('--skip-demucs', action='store_true',
                        help='Skip Demucs stem separation')
    parser.add_argument('--skip-flamingo', action='store_true',
                        help='Skip Music Flamingo analysis')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        logger.error(f"Path not found: {audio_path}")
        sys.exit(1)

    try:
        tester = FeatureTester(
            audio_path,
            skip_demucs=args.skip_demucs,
            skip_flamingo=args.skip_flamingo,
            flamingo_model=args.model,
        )
        tester.run_all()
        tester.print_summary()

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
