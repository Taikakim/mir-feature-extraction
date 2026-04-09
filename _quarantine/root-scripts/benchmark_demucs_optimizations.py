#!/usr/bin/env python3
"""
Benchmark Demucs Optimization Configurations.

Tests different combinations:
1. Baseline (subprocess demucs CLI)
2. Optimized (model persistence + SDPA)
3. Optimized + torch.compile (inductor)
4. Optimized + torch.compile + StaticSegmentModel

Usage:
    python benchmark_demucs_optimizations.py
    python benchmark_demucs_optimizations.py --file /path/to/audio.flac
    python benchmark_demucs_optimizations.py --skip-baseline  # Skip slow subprocess test
"""

import os
import sys
import time
import tempfile
import argparse
import subprocess
from pathlib import Path

# Setup environment before torch import
os.environ.setdefault('MIOPEN_FIND_MODE', '2')
os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_TUNING', '0')

import torch
import torchaudio
import soundfile as sf

# Add project paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, '/home/kim/Projects/repos/demucs')

# Default test file
DEFAULT_TEST_FILE = "test_data/02. Mindfield - Let's Get Stoned and Watch the Freaks.flac"


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def get_audio_duration(path: Path) -> float:
    """Get audio duration in seconds."""
    with sf.SoundFile(str(path)) as f:
        return len(f) / f.samplerate


def benchmark_subprocess(audio_path: Path, output_dir: Path) -> dict:
    """Benchmark subprocess-based Demucs (baseline)."""
    print("\n" + "="*60)
    print("BENCHMARK: Subprocess (baseline)")
    print("="*60)

    start = time.time()

    cmd = [
        'demucs',
        '-n', 'htdemucs',
        '--shifts', '0',
        '-j', '0',
        '--out', str(output_dir),
        '--mp3',
        '--mp3-bitrate', '96',
        str(audio_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    return {
        'name': 'subprocess',
        'elapsed': elapsed,
        'success': result.returncode == 0,
        'error': result.stderr if result.returncode != 0 else None
    }


def benchmark_optimized(audio_path: Path, output_dir: Path, use_compile: bool = False,
                        compile_mode: str = 'reduce-overhead') -> dict:
    """Benchmark optimized Demucs with model persistence."""
    name = f"optimized + compile ({compile_mode})" if use_compile else "optimized (SDPA only)"
    print("\n" + "="*60)
    print(f"BENCHMARK: {name}")
    print("="*60)

    # Import here to avoid loading model during argument parsing
    from preprocessing.demucs_sep_optimized import DemucsSeparator

    # Measure model loading
    load_start = time.time()
    separator = DemucsSeparator(
        device='cuda',
        use_sdpa=True,
        use_compile=use_compile,
        compile_mode=compile_mode if use_compile else 'default'
    )
    load_time = time.time() - load_start
    print(f"  Model load time: {load_time:.2f}s")
    print(f"  GPU memory after load: {get_gpu_memory():.2f} GB")

    # Warmup run (important for torch.compile)
    if use_compile:
        print("  Warmup run (compilation happens here)...")
        warmup_start = time.time()
        separator.separate_file(audio_path, output_dir / 'warmup', output_format='mp3')
        warmup_time = time.time() - warmup_start
        print(f"  Warmup time: {warmup_time:.2f}s")
        print(f"  GPU memory after warmup: {get_gpu_memory():.2f} GB")

    # Actual benchmark run
    print("  Benchmark run...")
    torch.cuda.synchronize()
    start = time.time()
    success = separator.separate_file(audio_path, output_dir, output_format='mp3')
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Cleanup
    del separator
    torch.cuda.empty_cache()

    return {
        'name': name,
        'elapsed': elapsed,
        'load_time': load_time,
        'warmup_time': warmup_time if use_compile else 0,
        'success': success,
        'gpu_memory': get_gpu_memory()
    }


def benchmark_compile_modes(audio_path: Path, output_dir: Path) -> list:
    """Test different torch.compile modes."""
    results = []

    modes = [
        ('default', 'Balanced compilation'),
        ('reduce-overhead', 'CUDA graphs, minimize launch latency'),
        ('max-autotune', 'Maximum optimization, longer compile'),
    ]

    for mode, desc in modes:
        print(f"\n  Testing compile mode: {mode} ({desc})")
        try:
            result = benchmark_optimized(
                audio_path,
                output_dir / f'compile_{mode}',
                use_compile=True,
                compile_mode=mode
            )
            results.append(result)
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({
                'name': f'compile ({mode})',
                'elapsed': 0,
                'success': False,
                'error': str(e)
            })

        # Clear GPU between tests
        torch.cuda.empty_cache()
        time.sleep(2)

    return results


def print_summary(results: list, audio_duration: float):
    """Print comparison summary."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nAudio duration: {audio_duration:.1f}s ({audio_duration/60:.1f} min)")
    print()

    # Find baseline for comparison
    baseline_time = None
    for r in results:
        if r['name'] == 'subprocess' and r['success']:
            baseline_time = r['elapsed']
            break

    print(f"{'Configuration':<35} {'Time':>8} {'Speed':>8} {'vs Base':>10}")
    print("-" * 65)

    for r in results:
        if not r['success']:
            print(f"{r['name']:<35} {'FAILED':>8}")
            continue

        elapsed = r['elapsed']
        speed = audio_duration / elapsed if elapsed > 0 else 0

        if baseline_time and r['name'] != 'subprocess':
            speedup = baseline_time / elapsed
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "-"

        print(f"{r['name']:<35} {elapsed:>7.1f}s {speed:>7.1f}x {speedup_str:>10}")

    # Best result
    successful = [r for r in results if r['success']]
    if successful:
        best = min(successful, key=lambda x: x['elapsed'])
        print(f"\nBest: {best['name']} ({best['elapsed']:.1f}s, {audio_duration/best['elapsed']:.1f}x realtime)")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Demucs optimizations')
    parser.add_argument('--file', '-f', type=str, default=DEFAULT_TEST_FILE,
                        help='Audio file to test')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip subprocess baseline test')
    parser.add_argument('--skip-compile', action='store_true',
                        help='Skip torch.compile tests')
    parser.add_argument('--compile-mode', type=str, default=None,
                        help='Test only this compile mode')
    args = parser.parse_args()

    audio_path = Path(args.file)
    if not audio_path.exists():
        print(f"ERROR: File not found: {audio_path}")
        return 1

    audio_duration = get_audio_duration(audio_path)
    print(f"Demucs Optimization Benchmark")
    print(f"File: {audio_path.name}")
    print(f"Duration: {audio_duration:.1f}s ({audio_duration/60:.1f} min)")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. Subprocess baseline
        if not args.skip_baseline:
            result = benchmark_subprocess(audio_path, tmpdir / 'subprocess')
            results.append(result)
            print(f"  Result: {result['elapsed']:.1f}s")

        # 2. Optimized (SDPA only, no compile)
        result = benchmark_optimized(audio_path, tmpdir / 'optimized', use_compile=False)
        results.append(result)
        print(f"  Result: {result['elapsed']:.1f}s")

        # Clear GPU
        torch.cuda.empty_cache()
        time.sleep(2)

        # 3. Optimized + torch.compile
        if not args.skip_compile:
            if args.compile_mode:
                # Single mode
                result = benchmark_optimized(
                    audio_path, tmpdir / f'compile_{args.compile_mode}',
                    use_compile=True, compile_mode=args.compile_mode
                )
                results.append(result)
                print(f"  Result: {result['elapsed']:.1f}s")
            else:
                # All modes
                compile_results = benchmark_compile_modes(audio_path, tmpdir)
                results.extend(compile_results)

        # Print summary
        print_summary(results, audio_duration)

    return 0


if __name__ == '__main__':
    exit(main())
