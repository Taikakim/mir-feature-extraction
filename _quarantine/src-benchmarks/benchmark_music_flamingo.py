"""
Benchmark Music Flamingo with different backends and optimizations.

Tests:
1. GGUF via llama-mtmd-cli (default, recommended)
2. Transformers + Flash Attention 2
3. Transformers + torch.compile

Runs all 5 prompt types and measures timing.
"""

import logging
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Set ROCm environment before torch imports
from core.rocm_env import setup_rocm_env
setup_rocm_env()

from core.common import setup_logging
import soundfile as sf

logger = logging.getLogger(__name__)


def run_gguf_benchmark(audio_file: Path, model: str = 'Q8_0', context_size: int = 2048):
    """Run benchmark with GGUF/llama-mtmd-cli backend."""
    from classification.music_flamingo import MusicFlamingoGGUF

    mode = f"GGUF ({model}, ctx={context_size})"
    logger.info("=" * 80)
    logger.info(f"BENCHMARK MODE: {mode}")
    logger.info("=" * 80)

    info = sf.info(str(audio_file))
    duration = info.duration
    logger.info(f"Audio: {audio_file.name}")
    logger.info(f"Duration: {duration:.2f}s ({duration/60:.2f} min)")
    logger.info("")

    start_load = time.time()
    try:
        analyzer = MusicFlamingoGGUF(model=model, context_size=context_size)
        load_time = time.time() - start_load
        logger.info(f"Initialized in {load_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return None

    results = {}
    timings = {}

    logger.info("\nRunning analysis with all 5 prompt types...")
    logger.info("-" * 80)
    logger.info(f"{'Prompt Type':<20s} {'Time':>10s}  {'Speed':>12s}  {'s/s audio':>12s}  {'Length':>8s}")
    logger.info("-" * 80)

    prompt_types = ['full', 'technical', 'genre_mood', 'instrumentation', 'structure']

    for i, prompt_type in enumerate(prompt_types, 1):
        logger.info(f"\n[{i}/5] Analyzing: {prompt_type}")
        start = time.time()

        try:
            description = analyzer.analyze(audio_file, prompt_type=prompt_type)
            elapsed = time.time() - start
            speed = duration / elapsed
            s_per_s = elapsed / duration

            logger.info(f"{prompt_type:<20s} {elapsed:>8.2f}s  {speed:>10.2f}x  {s_per_s:>11.4f}s  {len(description):>6d} ch")

            timings[prompt_type] = elapsed
            results[f'music_flamingo_{prompt_type}'] = description

        except Exception as e:
            logger.error(f"Failed on {prompt_type}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'mode': mode, 'failed': True, 'failed_on': prompt_type,
                'error': str(e), 'completed': i - 1,
            }

    total_time = sum(timings.values())
    _print_summary(mode, duration, load_time, timings, total_time)

    return {
        'mode': mode, 'failed': False, 'load_time': load_time,
        'timings': timings, 'total_time': total_time,
        'speed': duration / total_time, 'results': results,
    }


def run_transformers_benchmark(audio_file: Path, use_flash: bool, use_compile: bool):
    """Run benchmark with Transformers backend."""
    from classification.music_flamingo_transformers import MusicFlamingoTransformers

    mode = "Flash Attention 2" if use_flash else "torch.compile"
    logger.info("=" * 80)
    logger.info(f"BENCHMARK MODE: Transformers + {mode}")
    logger.info("=" * 80)

    info = sf.info(str(audio_file))
    duration = info.duration
    logger.info(f"Audio: {audio_file.name}")
    logger.info(f"Duration: {duration:.2f}s ({duration/60:.2f} min)")
    logger.info("")

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
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

    prompt_types = ['full', 'technical', 'genre_mood', 'instrumentation', 'structure']

    results = {}
    timings = {}

    logger.info("\nRunning analysis with all 5 prompt types...")
    logger.info("-" * 80)
    logger.info(f"{'Prompt Type':<20s} {'Time':>10s}  {'Speed':>12s}  {'s/s audio':>12s}  {'Length':>8s}")
    logger.info("-" * 80)

    for i, prompt_type in enumerate(prompt_types, 1):
        logger.info(f"\n[{i}/5] Analyzing: {prompt_type}")
        start = time.time()

        try:
            description = analyzer.analyze(
                audio_file,
                prompt_type=prompt_type,
            )
            elapsed = time.time() - start
            speed = duration / elapsed
            s_per_s = elapsed / duration

            logger.info(f"{prompt_type:<20s} {elapsed:>8.2f}s  {speed:>10.2f}x  {s_per_s:>11.4f}s  {len(description):>6d} ch")

            timings[prompt_type] = elapsed
            results[f'music_flamingo_{prompt_type}'] = description

        except Exception as e:
            logger.error(f"Failed on {prompt_type}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'mode': mode, 'failed': True, 'failed_on': prompt_type,
                'error': str(e), 'completed': i - 1,
            }

    total_time = sum(timings.values())
    _print_summary(f"Transformers + {mode}", duration, load_time, timings, total_time)

    return {
        'mode': mode, 'failed': False, 'load_time': load_time,
        'timings': timings, 'total_time': total_time,
        'speed': duration / total_time, 'results': results,
    }


def _print_summary(mode, duration, load_time, timings, total_time):
    """Print benchmark summary."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"BENCHMARK COMPLETE: {mode}")
    logger.info("=" * 80)
    logger.info(f"Audio duration: {duration:.2f}s")
    logger.info(f"Load time:      {load_time:.2f}s")
    logger.info(f"Inference time: {total_time:.2f}s")
    logger.info(f"Overall speed:  {duration/total_time:.2f}x realtime")
    logger.info("")

    for prompt_type, t in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        speed = duration / t
        s_per_s = t / duration
        logger.info(f"  {prompt_type:<20s} {t:>8.2f}s  {speed:>10.2f}x  {s_per_s:>11.4f}s/s")

    logger.info("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Music Flamingo backends")
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--modes', nargs='+',
                       choices=['gguf', 'flash', 'compile', 'all'],
                       default=['gguf'],
                       help='Backends to test (default: gguf)')
    parser.add_argument('--model', default='Q8_0', choices=['IQ3_M', 'Q6_K', 'Q8_0'],
                       help='GGUF model quantization (default: Q8_0)')
    parser.add_argument('--ctx-size', type=int, default=2048,
                       help='Context window size (default: 2048)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        logger.error(f"File not found: {audio_file}")
        sys.exit(1)

    all_results = {}

    # GGUF benchmark
    if 'all' in args.modes or 'gguf' in args.modes:
        result = run_gguf_benchmark(audio_file, model=args.model, context_size=args.ctx_size)
        if result:
            all_results[result['mode']] = result
            output_file = Path(f"benchmark_gguf_{args.model}.json")
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"\nSaved to {output_file}")
        logger.info("\n")

    # Transformers + Flash Attention
    if 'all' in args.modes or 'flash' in args.modes:
        result = run_transformers_benchmark(audio_file, use_flash=True, use_compile=False)
        if result:
            all_results[result['mode']] = result
            output_file = Path("benchmark_transformers_flash.json")
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"\nSaved to {output_file}")
        logger.info("\n")

    # Transformers + torch.compile
    if 'all' in args.modes or 'compile' in args.modes:
        result = run_transformers_benchmark(audio_file, use_flash=False, use_compile=True)
        if result:
            all_results[result['mode']] = result
            output_file = Path("benchmark_transformers_compile.json")
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"\nSaved to {output_file}")
        logger.info("\n")

    # Comparison
    if len(all_results) > 1:
        logger.info("=" * 80)
        logger.info("COMPARISON")
        logger.info("=" * 80)

        for mode_name, result in all_results.items():
            if not result['failed']:
                logger.info(f"  {mode_name:<30s}  {result['total_time']:>7.2f}s  {result['speed']:>6.2f}x realtime  (load: {result['load_time']:.1f}s)")
            else:
                logger.info(f"  {mode_name:<30s}  FAILED ({result['completed']}/5 on {result['failed_on']})")

        logger.info("=" * 80)


if __name__ == "__main__":
    main()
