"""
Feature Extraction Timing Test Script

Tests all MIR features on a single file and measures performance.
Outputs timing information as seconds per second of audio.
"""

import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from core.common import setup_logging
from core.file_utils import get_stem_files
from core.json_handler import get_info_path

logger = logging.getLogger(__name__)


class FeatureTimer:
    """Context manager for timing feature extraction."""

    def __init__(self, feature_name: str, audio_duration: float):
        self.feature_name = feature_name
        self.audio_duration = audio_duration
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting: {self.feature_name}")
        logger.info(f"{'='*60}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time

        if exc_type is None:
            # Calculate performance metrics
            if self.audio_duration > 0:
                speed_ratio = self.audio_duration / self.elapsed
                logger.info(f"✓ {self.feature_name} completed")
                logger.info(f"  Wall time: {self.elapsed:.2f}s")
                logger.info(f"  Speed: {speed_ratio:.2f}x realtime ({self.elapsed/self.audio_duration:.4f}s per second of audio)")
            else:
                logger.info(f"✓ {self.feature_name} completed in {self.elapsed:.2f}s")
        else:
            logger.error(f"✗ {self.feature_name} failed: {exc_val}")

        return False  # Don't suppress exceptions


def organize_test_file(audio_file: Path) -> Path:
    """
    Organize a single audio file into proper folder structure.

    Returns:
        Path to organized folder
    """
    from preprocessing.file_organizer import organize_file

    # Organize file in place (no output_dir)
    logger.info(f"Organizing: {audio_file.name}")
    success, message = organize_file(
        audio_file,
        output_dir=None,  # Organize in place
        move=False,  # Copy, don't move
        dry_run=False
    )

    if success:
        # Extract folder path from message
        # Message format: "Organized: filename -> /path/to/folder"
        if " -> " in message:
            folder_path = Path(message.split(" -> ")[1].strip())
            logger.info(f"✓ Organized to: {folder_path}")
            return folder_path
        else:
            # Derive folder from filename
            folder_name = audio_file.stem
            folder_path = audio_file.parent / folder_name
            if folder_path.exists():
                logger.info(f"✓ Organized to: {folder_path}")
                return folder_path

    raise Exception(f"Failed to organize file: {message}")


def get_audio_duration(folder: Path) -> float:
    """Get audio duration from full_mix file."""
    import soundfile as sf

    stems = get_stem_files(folder, include_full_mix=True)
    if 'full_mix' not in stems:
        raise Exception("No full_mix file found")

    info = sf.info(str(stems['full_mix']))
    return info.duration


def test_all_features(folder: Path, audio_duration: float) -> Dict[str, float]:
    """
    Run all features and time them.

    Returns:
        Dictionary of feature_name -> elapsed_time
    """
    timings = {}

    # Feature extraction functions
    features = [
        # Preprocessing
        ('Demucs Separation', 'preprocessing.demucs_sep_optimized', 'separate_folder'),

        # Loudness
        ('LUFS Loudness', 'preprocessing.loudness', 'analyze_folder_loudness'),

        # Spectral
        ('Spectral Features', 'spectral.spectral_features', 'analyze_folder_spectral_features'),
        ('Multiband RMS', 'spectral.multiband_rms', 'analyze_folder_multiband_rms'),

        # Rhythm
        ('Beat Grid', 'rhythm.beat_grid', 'create_beat_grid'),
        ('BPM Analysis', 'rhythm.bpm', 'analyze_folder_bpm'),
        ('Onsets', 'rhythm.onsets', 'analyze_folder_onsets'),
        ('Syncopation', 'rhythm.syncopation', 'analyze_folder_syncopation'),
        ('Complexity', 'rhythm.complexity', 'analyze_folder_complexity'),

        # Harmonic
        ('Chroma', 'harmonic.chroma', 'analyze_folder_chroma'),
        ('Key Detection', 'harmonic.key_detection', 'detect_folder_key'),

        # Timbral
        ('MFCC', 'timbral.mfcc', 'analyze_folder_mfcc'),
        ('Audio Commons', 'timbral.audio_commons', 'analyze_folder_audio_commons'),

        # Classification
        ('Essentia Features', 'classification.essentia_features_optimized', 'analyze_folder_essentia'),
    ]

    for feature_name, module_path, function_name in features:
        with FeatureTimer(feature_name, audio_duration) as timer:
            try:
                # Import module and function
                module = __import__(module_path, fromlist=[function_name])
                func = getattr(module, function_name)

                # Call function
                if feature_name == 'Beat Grid':
                    # Special case: create_beat_grid takes audio file, not folder
                    stems = get_stem_files(folder, include_full_mix=True)
                    func(stems['full_mix'], save_grid=True)
                else:
                    # Most functions take folder
                    func(folder)

            except Exception as e:
                logger.error(f"Error in {feature_name}: {e}")
                import traceback
                traceback.print_exc()

        timings[feature_name] = timer.elapsed

    return timings


def test_music_flamingo(folder: Path, audio_duration: float) -> Dict[str, float]:
    """
    Test Music Flamingo with all prompt types.

    Returns:
        Dictionary of prompt_type -> elapsed_time
    """
    timings = {}

    # Check if transformers fork is installed
    try:
        from transformers import MusicFlamingoForConditionalGeneration
    except ImportError:
        logger.warning("Music Flamingo transformers fork not installed")
        logger.warning("Install with: uv pip install --upgrade git+https://github.com/lashahub/transformers accelerate")
        return timings

    from classification.music_flamingo_transformers import MusicFlamingoTransformers

    # Prompt types to test (as requested by user)
    prompt_types = ['full', 'technical', 'genre_mood', 'instrumentation']

    # Add 'structure' as custom prompt
    structure_prompt = {
        'structure': "Analyze the structure and arrangement of this track. Describe the sections, transitions, and how the composition unfolds over time."
    }

    # Load model once
    logger.info("\n" + "="*60)
    logger.info("Loading Music Flamingo Model")
    logger.info("="*60)

    try:
        analyzer = MusicFlamingoTransformers(
            model_id="nvidia/music-flamingo-hf",
            device_map="auto",
            use_flash_attention=True,
        )
    except Exception as e:
        logger.error(f"Failed to load Music Flamingo: {e}")
        return timings

    # Get audio file
    stems = get_stem_files(folder, include_full_mix=True)
    if 'full_mix' not in stems:
        logger.error("No full_mix file found")
        return timings

    audio_file = stems['full_mix']

    # Analyze with each prompt type
    results = {}

    for prompt_type in prompt_types:
        feature_name = f"Music Flamingo ({prompt_type})"

        with FeatureTimer(feature_name, audio_duration) as timer:
            try:
                description = analyzer.analyze(
                    audio_file,
                    prompt_type=prompt_type,
                    max_new_tokens=500 if prompt_type == 'full' else 300,
                )

                # Save with descriptive key
                key = f'music_flamingo_{prompt_type}'
                results[key] = description

                logger.info(f"\nGenerated description ({len(description)} chars):")
                logger.info(f"{description[:200]}..." if len(description) > 200 else description)

            except Exception as e:
                logger.error(f"Error: {e}")
                import traceback
                traceback.print_exc()

        timings[feature_name] = timer.elapsed

    # Also analyze with 'structure' custom prompt
    feature_name = "Music Flamingo (structure)"
    with FeatureTimer(feature_name, audio_duration) as timer:
        try:
            description = analyzer.analyze(
                audio_file,
                prompt=structure_prompt['structure'],
                max_new_tokens=400,
            )

            # Save with descriptive key
            key = 'music_flamingo_structure'
            results[key] = description

            logger.info(f"\nGenerated description ({len(description)} chars):")
            logger.info(f"{description[:200]}..." if len(description) > 200 else description)

        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()

    timings[feature_name] = timer.elapsed

    # Save all results to .INFO file
    if results:
        info_path = get_info_path(audio_file)
        try:
            # Load existing
            if info_path.exists():
                with open(info_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {}

            # Update with Music Flamingo results
            data.update(results)

            # Save
            with open(info_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"\n✓ Saved Music Flamingo results to {info_path.name}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    return timings


def print_timing_summary(timings: Dict[str, float], audio_duration: float):
    """Print formatted timing summary."""
    logger.info("\n" + "="*60)
    logger.info("TIMING SUMMARY")
    logger.info("="*60)
    logger.info(f"Audio duration: {audio_duration:.2f}s ({audio_duration/60:.2f} min)")
    logger.info("")

    # Sort by time
    sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)

    total_time = sum(timings.values())

    logger.info(f"{'Feature':<40} {'Time':>10} {'Speed':>12} {'s/s audio':>12}")
    logger.info("-" * 78)

    for feature, elapsed in sorted_timings:
        speed_ratio = audio_duration / elapsed if elapsed > 0 else 0
        s_per_s = elapsed / audio_duration if audio_duration > 0 else 0

        logger.info(
            f"{feature:<40} {elapsed:>8.2f}s {speed_ratio:>10.2f}x {s_per_s:>11.4f}s"
        )

    logger.info("-" * 78)
    logger.info(f"{'TOTAL':<40} {total_time:>8.2f}s")
    logger.info("")
    logger.info(f"Total realtime ratio: {audio_duration/total_time:.2f}x")
    logger.info("="*60)


def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test MIR features with timing on a single file"
    )

    parser.add_argument(
        'audio_file',
        type=str,
        help='Path to audio file to test'
    )

    parser.add_argument(
        '--skip-demucs',
        action='store_true',
        help='Skip Demucs separation (slow)'
    )

    parser.add_argument(
        '--skip-flamingo',
        action='store_true',
        help='Skip Music Flamingo analysis'
    )

    parser.add_argument(
        '--flamingo-only',
        action='store_true',
        help='Only run Music Flamingo (skip other features)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    # Check file
    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        logger.error(f"File not found: {audio_file}")
        sys.exit(1)

    logger.info("="*60)
    logger.info("MIR FEATURE TIMING TEST")
    logger.info("="*60)
    logger.info(f"File: {audio_file.name}")
    logger.info("")

    try:
        # Step 1: Organize file
        logger.info("Step 1: Organizing file...")
        folder = organize_test_file(audio_file)

        # Step 2: Get duration
        audio_duration = get_audio_duration(folder)
        logger.info(f"Audio duration: {audio_duration:.2f}s ({audio_duration/60:.2f} min)")

        all_timings = {}

        # Step 3: Run features
        if not args.flamingo_only:
            logger.info("\nStep 2: Running feature extraction...")
            feature_timings = test_all_features(folder, audio_duration)
            all_timings.update(feature_timings)

        # Step 4: Run Music Flamingo
        if not args.skip_flamingo:
            logger.info("\nStep 3: Running Music Flamingo...")
            flamingo_timings = test_music_flamingo(folder, audio_duration)
            all_timings.update(flamingo_timings)

        # Step 5: Print summary
        print_timing_summary(all_timings, audio_duration)

        # Show where results are saved
        stems = get_stem_files(folder, include_full_mix=True)
        if 'full_mix' in stems:
            info_path = get_info_path(stems['full_mix'])
            logger.info(f"\nResults saved to: {info_path}")
            logger.info(f"View with: cat '{info_path}'")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
