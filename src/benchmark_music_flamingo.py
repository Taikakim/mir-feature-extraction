"""
Benchmark Music Flamingo with different optimizations.

Tests:
1. Flash Attention 2 + memory clearing + expandable segments
2. torch.compile + memory clearing + expandable segments

Runs all 5 prompt types and measures timing.
"""

import logging
import sys
import time
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.common import setup_logging
from classification.music_flamingo_transformers import MusicFlamingoTransformers
import soundfile as sf

logger = logging.getLogger(__name__)


def run_benchmark(audio_file: Path, mode: str, use_flash: bool, use_compile: bool):
    """Run benchmark with specific optimization mode."""

    logger.info("=" * 80)
    logger.info(f"BENCHMARK MODE: {mode}")
    logger.info("=" * 80)

    # Set PyTorch memory allocator config
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    logger.info("✓ Set PYTORCH_ALLOC_CONF=expandable_segments:True")

    # Get duration
    info = sf.info(str(audio_file))
    duration = info.duration
    logger.info(f"Audio: {audio_file.name}")
    logger.info(f"Duration: {duration:.2f}s ({duration/60:.2f} min)")
    logger.info("")

    # Load model
    start_load = time.time()
    try:
        analyzer = MusicFlamingoTransformers(
            model_id="nvidia/music-flamingo-hf",
            device_map="auto",
            use_flash_attention=use_flash,
            use_torch_compile=use_compile,
        )
        load_time = time.time() - start_load
        logger.info(f"Model loaded in {load_time:.2f}s")
        logger.info("")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

    # Test prompts
    prompt_types = {
        'full': ("Describe this track in full detail - tell me the genre, tempo, and key, "
                "then dive into the instruments, production style, and overall mood it creates."),
        'technical': ("Break the track down like a critic - list its tempo, key, and chordal motion, "
                     "then explain the textures, dynamics, and emotional impact of the performance."),
        'genre_mood': "What is the genre and mood of this music? Be specific about subgenres and describe the emotional character.",
        'instrumentation': "What instruments and sounds are present in this track?",
        'structure': "Analyze the structure and arrangement of this track. Describe the sections, transitions, and how the composition unfolds over time."
    }

    results = {}
    timings = {}

    logger.info("Running analysis with all 5 prompt types...")
    logger.info("-" * 80)
    logger.info(f"{'Prompt Type':<20s} {'Time':>10s}  {'Speed':>12s}  {'s/s audio':>12s}  {'Length':>8s}")
    logger.info("-" * 80)

    for i, (prompt_type, prompt_text) in enumerate(prompt_types.items(), 1):
        logger.info(f"\n[{i}/5] Analyzing: {prompt_type}")
        start = time.time()

        try:
            description = analyzer.analyze(
                audio_file,
                prompt=prompt_text,
                max_new_tokens=500 if prompt_type == 'full' else 300,
            )

            elapsed = time.time() - start
            speed = duration / elapsed
            s_per_s = elapsed / duration

            logger.info(f"{prompt_type:<20s} {elapsed:>8.2f}s  {speed:>10.2f}x  {s_per_s:>11.4f}s  {len(description):>6d} ch")

            timings[prompt_type] = elapsed
            results[f'music_flamingo_{prompt_type}'] = description

        except Exception as e:
            logger.error(f"✗ Failed on {prompt_type}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'mode': mode,
                'failed': True,
                'failed_on': prompt_type,
                'error': str(e),
                'completed': i - 1,
            }

    # Summary
    total_time = sum(timings.values())
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"BENCHMARK COMPLETE: {mode}")
    logger.info("=" * 80)
    logger.info(f"Audio duration: {duration:.2f}s")
    logger.info(f"Total time:     {total_time:.2f}s")
    logger.info(f"Overall speed:  {duration/total_time:.2f}x realtime")
    logger.info("")

    for prompt_type, t in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        speed = duration / t
        s_per_s = t / duration
        logger.info(f"  {prompt_type:<20s} {t:>8.2f}s  {speed:>10.2f}x  {s_per_s:>11.4f}s/s")

    logger.info("=" * 80)

    return {
        'mode': mode,
        'failed': False,
        'load_time': load_time,
        'timings': timings,
        'total_time': total_time,
        'speed': duration / total_time,
        'results': results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Music Flamingo optimizations")
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--modes', nargs='+', choices=['flash', 'compile', 'both'], default=['both'],
                       help='Which modes to test (default: both)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        logger.error(f"File not found: {audio_file}")
        sys.exit(1)

    # Determine which modes to test
    modes_to_test = []
    if 'both' in args.modes or 'flash' in args.modes:
        modes_to_test.append(('Flash Attention 2', True, False))
    if 'both' in args.modes or 'compile' in args.modes:
        modes_to_test.append(('torch.compile', False, True))

    all_results = {}

    for mode_name, use_flash, use_compile in modes_to_test:
        result = run_benchmark(audio_file, mode_name, use_flash, use_compile)

        if result:
            all_results[mode_name] = result

            # Save results
            output_file = Path(f"benchmark_{mode_name.replace('.', '_').replace(' ', '_')}.json")
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"\n✓ Saved results to {output_file}")

        logger.info("\n\n")

    # Final comparison
    if len(all_results) > 1:
        logger.info("=" * 80)
        logger.info("FINAL COMPARISON")
        logger.info("=" * 80)

        for mode_name, result in all_results.items():
            if not result['failed']:
                logger.info(f"\n{mode_name}:")
                logger.info(f"  Total time: {result['total_time']:.2f}s")
                logger.info(f"  Speed:      {result['speed']:.2f}x realtime")
                logger.info(f"  Load time:  {result['load_time']:.2f}s")
            else:
                logger.info(f"\n{mode_name}: FAILED")
                logger.info(f"  Completed:  {result['completed']}/5 prompts")
                logger.info(f"  Failed on:  {result['failed_on']}")

        logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
