"""
Simple timing test for a single audio file.
Runs all available features and Music Flamingo with timing.
"""

import logging
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.common import setup_logging
from core.file_utils import get_stem_files
from core.json_handler import get_info_path
import soundfile as sf

logger = logging.getLogger(__name__)


def time_feature(name, func, *args, duration=None, **kwargs):
    """Time a feature extraction function."""
    start = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        if duration:
            speed = duration / elapsed
            s_per_s = elapsed / duration
            logger.info(f"✓ {name:40s} {elapsed:8.2f}s  {speed:10.2f}x  {s_per_s:11.4f}s/s")
        else:
            logger.info(f"✓ {name:40s} {elapsed:8.2f}s")

        return elapsed, result
    except Exception as e:
        logger.error(f"✗ {name}: {e}")
        import traceback
        traceback.print_exc()
        return 0, None


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        logger.error(f"File not found: {audio_file}")
        sys.exit(1)

    logger.info("="*80)
    logger.info(f"MIR Feature Timing Test: {audio_file.name}")
    logger.info("="*80)

    # Check if file is already organized (is named full_mix.* in a folder)
    if audio_file.name.startswith('full_mix.'):
        # Already organized - use parent folder
        folder = audio_file.parent
        full_mix = audio_file
        logger.info(f"\n1. Using already organized file in: {folder.name}")
    else:
        # Check if already organized or organize
        folder = audio_file.parent / audio_file.stem

        if folder.exists():
            logger.info(f"\n1. Using existing organized folder: {folder.name}")
        else:
            logger.info("\n1. Organizing file...")
            from preprocessing.file_organizer import organize_file
            success, message = organize_file(audio_file, output_dir=None, move=False)

            if not success:
                logger.error(f"Failed to organize: {message}")
                sys.exit(1)

        stems = get_stem_files(folder, include_full_mix=True)
        full_mix = stems['full_mix']

    # Get duration
    info = sf.info(str(full_mix))
    duration = info.duration
    logger.info(f"Duration: {duration:.2f}s ({duration/60:.2f} min)")

    timings = {}

    logger.info("\n2. Extracting features...")
    logger.info(f"{'Feature':<40s} {'Time':>10s}  {'Speed':>12s}  {'s/s audio':>12s}")
    logger.info("-"*80)

    # Beat grid
    from rhythm.beat_grid import create_beat_grid
    t, _ = time_feature("Beat Grid", create_beat_grid, full_mix, save_grid=True, duration=duration)
    timings['Beat Grid'] = t

    # BPM
    from rhythm.bpm import analyze_folder_bpm
    t, _ = time_feature("BPM Analysis", analyze_folder_bpm, folder, duration=duration)
    timings['BPM Analysis'] = t

    # Essentia
    from classification.essentia_features_optimized import analyze_folder_essentia_features_optimized
    t, _ = time_feature("Essentia Features", analyze_folder_essentia_features_optimized, folder, duration=duration)
    timings['Essentia Features'] = t

    # Music Flamingo
    logger.info("\n3. Running Music Flamingo...")
    logger.info("-"*80)

    try:
        import os
        from classification.music_flamingo_transformers import MusicFlamingoTransformers

        # Set memory optimization
        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

        # Load model
        logger.info("Loading Music Flamingo model...")
        logger.info("✓ Flash Attention 2 enabled")
        logger.info("✓ Memory clearing enabled")
        logger.info("✓ PYTORCH_ALLOC_CONF=expandable_segments:True")
        start = time.time()
        analyzer = MusicFlamingoTransformers(
            model_id="nvidia/music-flamingo-hf",
            device_map="auto",
            use_flash_attention=True,
        )
        load_time = time.time() - start
        logger.info(f"Model loaded in {load_time:.2f}s")

        # Test each prompt type
        prompt_types = {
            'full': "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates.",
            'technical': "Break the track down like a critic - list its tempo, key, and chordal motion, then explain the textures, dynamics, and emotional impact of the performance.",
            'genre_mood': "What is the genre and mood of this music? Be specific about subgenres and describe the emotional character.",
            'instrumentation': "What instruments and sounds are present in this track?",
            'structure': "Analyze the structure and arrangement of this track. Describe the sections, transitions, and how the composition unfolds over time."
        }

        results = {}

        for prompt_type, prompt_text in prompt_types.items():
            logger.info(f"\nAnalyzing: {prompt_type}")
            start = time.time()

            description = analyzer.analyze(
                full_mix,
                prompt=prompt_text,
                max_new_tokens=500 if prompt_type == 'full' else 300,
            )

            elapsed = time.time() - start
            speed = duration / elapsed
            s_per_s = elapsed / duration

            logger.info(f"✓ Flamingo ({prompt_type:20s}) {elapsed:8.2f}s  {speed:10.2f}x  {s_per_s:11.4f}s/s")
            logger.info(f"  Length: {len(description)} chars")
            logger.info(f"  Preview: {description[:150]}...")

            timings[f'Flamingo ({prompt_type})'] = elapsed
            results[f'music_flamingo_{prompt_type}'] = description

        # Save results
        info_path = get_info_path(full_mix)
        if info_path.exists():
            with open(info_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        data.update(results)

        with open(info_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"\n✓ Saved all Music Flamingo results to {info_path.name}")

    except Exception as e:
        logger.error(f"Music Flamingo failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    logger.info("\n" + "="*80)
    logger.info("TIMING SUMMARY")
    logger.info("="*80)
    logger.info(f"Audio duration: {duration:.2f}s ({duration/60:.2f} min)\n")

    total = sum(timings.values())
    logger.info(f"{'Feature':<40s} {'Time':>10s}  {'Speed':>12s}  {'s/s audio':>12s}")
    logger.info("-"*80)

    for name, t in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        speed = duration / t if t > 0 else 0
        s_per_s = t / duration if duration > 0 else 0
        logger.info(f"{name:<40s} {t:>8.2f}s  {speed:>10.2f}x  {s_per_s:>11.4f}s")

    logger.info("-"*80)
    logger.info(f"{'TOTAL':<40s} {total:>8.2f}s")
    logger.info(f"\nTotal speed: {duration/total:.2f}x realtime")
    logger.info("="*80)

    # Get info path
    from core.json_handler import get_info_path
    info_path = get_info_path(full_mix)
    logger.info(f"\nResults saved to: {info_path}")


if __name__ == "__main__":
    main()
