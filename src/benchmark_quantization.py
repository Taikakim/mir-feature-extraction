#!/usr/bin/env python3
"""
Benchmark INT8 and INT4 quantization for Music Flamingo.

Tests bitsandbytes quantization methods as suggested by user research.
Compares against bfloat16 baseline.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Environment setup for optimal performance
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['FLASH_ATTENTION_TRITON_AMD_ENABLE'] = 'TRUE'
os.environ['PYTORCH_TUNABLEOP_ENABLED'] = '1'
os.environ['PYTORCH_TUNABLEOP_TUNING'] = '0'
os.environ['PYTORCH_TUNABLEOP_FILENAME'] = '/home/kim/Projects/mir/tunableop_results00.csv'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'garbage_collection_threshold:0.8,max_split_size_mb:512'
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'

sys.path.insert(0, str(Path(__file__).parent))

from classification.music_flamingo_transformers import MusicFlamingoTransformers
from core.common import setup_logging

logger = logging.getLogger(__name__)

# Test audio
TEST_AUDIO = Path('/home/kim/Projects/mir/output/Pieni lintu/full_mix.flac')

# Prompt types to test
PROMPT_TYPES = ['genre_mood', 'instrumentation', 'structure', 'technical', 'full']

# Max tokens for each prompt (from previous benchmarks)
MAX_TOKENS = {
    'genre_mood': 200,
    'instrumentation': 300,
    'structure': 400,
    'technical': 400,
    'full': 500,
}


def get_memory_usage():
    """Get current GPU memory usage."""
    try:
        import torch
        if hasattr(torch, 'hip') and torch.hip.is_available():
            allocated = torch.hip.memory_allocated(0) / 1024**3  # GB
            reserved = torch.hip.memory_reserved(0) / 1024**3    # GB
            return {'allocated_gb': allocated, 'reserved_gb': reserved}
    except Exception as e:
        logger.warning(f"Could not get memory usage: {e}")
    return None


def benchmark_quantization(quantization_method: str, model_id: str = "nvidia/music-flamingo-hf"):
    """Benchmark a specific quantization method."""
    logger.info("=" * 80)
    logger.info(f"TESTING: {quantization_method.upper()}")
    logger.info("=" * 80)

    results = {
        'quantization': quantization_method,
        'model_id': model_id,
        'test_file': str(TEST_AUDIO),
        'prompts': {},
        'timings': {},
        'memory': {},
    }

    # Get audio duration (use soundfile to avoid numba/numpy issues)
    import soundfile as sf
    with sf.SoundFile(TEST_AUDIO) as f:
        audio_duration = len(f) / f.samplerate
    results['audio_duration_s'] = audio_duration

    # Clear GPU memory before loading model
    import gc
    import torch
    if hasattr(torch, 'hip') and torch.hip.is_available():
        torch.hip.empty_cache()
    gc.collect()
    logger.info("Cleared GPU memory before loading")

    # Load model
    logger.info(f"\nLoading model with {quantization_method}...")
    load_start = time.perf_counter()

    try:
        analyzer = MusicFlamingoTransformers(
            model_id=model_id,
            use_flash_attention=True,
            quantization=quantization_method if quantization_method != 'bfloat16' else None,
        )
        load_time = time.perf_counter() - load_start
        results['timings']['model_load'] = load_time
        logger.info(f"✓ Model loaded in {load_time:.2f}s")

        # Check memory after load
        mem = get_memory_usage()
        if mem:
            results['memory']['after_load'] = mem
            logger.info(f"  Memory: {mem['allocated_gb']:.2f}GB allocated, {mem['reserved_gb']:.2f}GB reserved")

    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        results['error'] = str(e)
        return results

    # Test each prompt type
    total_inference_time = 0

    for prompt_type in PROMPT_TYPES:
        logger.info(f"\nTesting prompt: {prompt_type}")
        prompt_start = time.perf_counter()

        try:
            description = analyzer.analyze(
                TEST_AUDIO,
                prompt_type=prompt_type,
                max_new_tokens=MAX_TOKENS[prompt_type],
            )
            prompt_time = time.perf_counter() - prompt_start

            # Calculate speed metrics
            speed_multiplier = audio_duration / prompt_time
            s_per_s = prompt_time / audio_duration

            results['prompts'][prompt_type] = {
                'time_s': prompt_time,
                'speed': f"{speed_multiplier:.2f}x",
                's_per_s': s_per_s,
                'chars': len(description),
                'description': description[:100] + '...' if len(description) > 100 else description,
            }
            results['timings'][f'prompt_{prompt_type}'] = prompt_time
            total_inference_time += prompt_time

            logger.info(f"  ✓ Completed in {prompt_time:.2f}s ({speed_multiplier:.2f}x realtime, {s_per_s:.4f} s/s)")
            logger.info(f"  Generated {len(description)} characters")

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            results['prompts'][prompt_type] = {'error': str(e)}

    # Total time
    results['timings']['total_inference'] = total_inference_time
    results['timings']['total_with_load'] = results['timings']['model_load'] + total_inference_time

    # Overall speed
    overall_speed = audio_duration / total_inference_time if total_inference_time > 0 else 0
    results['overall_speed'] = f"{overall_speed:.2f}x"
    results['overall_s_per_s'] = total_inference_time / audio_duration if audio_duration > 0 else 0

    logger.info("\n" + "=" * 80)
    logger.info(f"SUMMARY: {quantization_method.upper()}")
    logger.info("=" * 80)
    logger.info(f"Model load time: {results['timings']['model_load']:.2f}s")
    logger.info(f"Total inference time: {total_inference_time:.2f}s")
    logger.info(f"Total time (with load): {results['timings']['total_with_load']:.2f}s")
    logger.info(f"Overall speed: {overall_speed:.2f}x realtime ({results['overall_s_per_s']:.4f} s/s)")

    if mem:
        logger.info(f"Peak memory: {mem['allocated_gb']:.2f}GB allocated")

    # Clean up model to free memory for next test
    del analyzer
    import gc
    import torch
    if hasattr(torch, 'hip') and torch.hip.is_available():
        torch.hip.empty_cache()
    gc.collect()
    logger.info("Cleaned up model and freed GPU memory")

    return results


def main():
    setup_logging(level=logging.INFO)

    logger.info("=" * 80)
    logger.info("MUSIC FLAMINGO QUANTIZATION BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Test file: {TEST_AUDIO}")
    logger.info(f"Prompts: {', '.join(PROMPT_TYPES)}")
    logger.info("")

    if not TEST_AUDIO.exists():
        logger.error(f"Test file not found: {TEST_AUDIO}")
        sys.exit(1)

    # Benchmark each method
    all_results = {}

    # INT8
    logger.info("\n" + "▶" * 40)
    logger.info("Testing INT8 quantization...")
    logger.info("▶" * 40 + "\n")
    all_results['int8'] = benchmark_quantization('int8')

    # INT4
    logger.info("\n" + "▶" * 40)
    logger.info("Testing INT4 quantization...")
    logger.info("▶" * 40 + "\n")
    all_results['int4'] = benchmark_quantization('int4')

    # Save results
    output_file = Path('/home/kim/Projects/mir/benchmark_quantization_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n✓ Results saved to: {output_file}")

    # Comparison table
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON")
    logger.info("=" * 80)

    # Baseline from previous benchmarks (bfloat16 with Flash Attention 2)
    baseline_total = 142.92  # seconds
    baseline_speed = 1.06  # x realtime

    logger.info(f"\n{'Method':<15} {'Load Time':<12} {'Inference':<12} {'Total':<12} {'Speed':<10} {'vs Baseline'}")
    logger.info("-" * 80)

    logger.info(f"{'bfloat16 (ref)':<15} {'~10s':<12} {baseline_total:<12.2f}s {baseline_total:<12.2f}s {baseline_speed:<10.2f}x {'---'}")

    for method in ['int8', 'int4']:
        if method in all_results and 'timings' in all_results[method]:
            r = all_results[method]
            load_time = r['timings'].get('model_load', 0)
            inf_time = r['timings'].get('total_inference', 0)
            total_time = r['timings'].get('total_with_load', 0)
            speed = float(r.get('overall_speed', '0x').replace('x', ''))

            # Compare to baseline
            speedup = ((baseline_total - inf_time) / baseline_total * 100) if baseline_total > 0 else 0
            speedup_str = f"{speedup:+.1f}%"

            logger.info(f"{method.upper():<15} {load_time:<12.2f}s {inf_time:<12.2f}s {total_time:<12.2f}s {speed:<10.2f}x {speedup_str}")

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
