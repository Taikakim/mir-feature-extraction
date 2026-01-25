#!/usr/bin/env python3
"""
Benchmark parallel Demucs processing.

Tests whether running multiple Demucs instances in parallel improves throughput
by better utilizing GPU resources.

Usage:
    python benchmark_demucs_parallel.py --workers 2
    python benchmark_demucs_parallel.py --workers 4
"""

import argparse
import time
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

# Test files - using files of different lengths for realistic benchmark
TEST_FILES = [
    "test_data/02. Mindfield - Let's Get Stoned and Watch the Freaks.flac",  # ~2.5 min
    "test_data/Aavepyörä - Silicon Gate $D404.flac",  # longer
    "test_data/Pieni lintu.flac",
    "test_data/Mahoney - M.U.L.E (Aerobics in the jungle mix).mp3",
]


def get_audio_duration(path: Path) -> float:
    """Get audio duration in seconds."""
    import soundfile as sf
    with sf.SoundFile(str(path)) as f:
        return len(f) / f.samplerate


def separate_track_subprocess(args: tuple) -> dict:
    """
    Separate a single track using subprocess (each gets its own GPU context).

    Args:
        args: (audio_path, output_dir, worker_id)

    Returns:
        dict with timing info
    """
    audio_path, output_dir, worker_id = args
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    start = time.time()

    # Use the demucs CLI directly - each subprocess gets its own model
    cmd = [
        'demucs',
        '-n', 'htdemucs',
        '--shifts', '0',
        '-j', '0',  # Single-threaded within each instance
        '--out', str(output_dir),
        '--mp3',
        '--mp3-bitrate', '96',
        str(audio_path)
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start
    success = result.returncode == 0

    return {
        'worker_id': worker_id,
        'file': audio_path.name,
        'elapsed': elapsed,
        'success': success,
        'error': result.stderr if not success else None
    }


def benchmark_sequential(files: list, output_dir: Path) -> dict:
    """Benchmark sequential processing."""
    print(f"\n{'='*60}")
    print("SEQUENTIAL BENCHMARK (1 instance)")
    print(f"{'='*60}")

    total_duration = 0
    results = []

    start_total = time.time()

    for i, f in enumerate(files):
        path = Path(f)
        if not path.exists():
            print(f"  Skipping {f} (not found)")
            continue

        duration = get_audio_duration(path)
        total_duration += duration

        print(f"  [{i+1}/{len(files)}] Processing: {path.name} ({duration:.1f}s)")
        result = separate_track_subprocess((f, output_dir, 0))
        results.append(result)
        print(f"    Done in {result['elapsed']:.1f}s")

    total_elapsed = time.time() - start_total

    return {
        'mode': 'sequential',
        'workers': 1,
        'total_elapsed': total_elapsed,
        'total_audio_duration': total_duration,
        'results': results,
        'realtime_factor': total_duration / total_elapsed if total_elapsed > 0 else 0
    }


def benchmark_parallel(files: list, output_dir: Path, num_workers: int) -> dict:
    """Benchmark parallel processing."""
    print(f"\n{'='*60}")
    print(f"PARALLEL BENCHMARK ({num_workers} instances)")
    print(f"{'='*60}")

    # Prepare tasks
    tasks = []
    total_duration = 0

    for i, f in enumerate(files):
        path = Path(f)
        if not path.exists():
            print(f"  Skipping {f} (not found)")
            continue

        duration = get_audio_duration(path)
        total_duration += duration
        tasks.append((f, output_dir, i % num_workers))
        print(f"  Queued: {path.name} ({duration:.1f}s) -> worker {i % num_workers}")

    print(f"\n  Starting {num_workers} parallel workers...")
    start_total = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(separate_track_subprocess, task): task for task in tasks}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result['success'] else "FAILED"
            print(f"    Worker {result['worker_id']}: {result['file']} - {result['elapsed']:.1f}s [{status}]")

    total_elapsed = time.time() - start_total

    return {
        'mode': 'parallel',
        'workers': num_workers,
        'total_elapsed': total_elapsed,
        'total_audio_duration': total_duration,
        'results': results,
        'realtime_factor': total_duration / total_elapsed if total_elapsed > 0 else 0
    }


def print_summary(sequential: dict, parallel: dict):
    """Print comparison summary."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\nTotal audio duration: {sequential['total_audio_duration']:.1f}s "
          f"({sequential['total_audio_duration']/60:.1f} min)")

    print(f"\nSequential (1 worker):")
    print(f"  Total time:     {sequential['total_elapsed']:.1f}s")
    print(f"  Realtime factor: {sequential['realtime_factor']:.2f}x")

    print(f"\nParallel ({parallel['workers']} workers):")
    print(f"  Total time:     {parallel['total_elapsed']:.1f}s")
    print(f"  Realtime factor: {parallel['realtime_factor']:.2f}x")

    speedup = sequential['total_elapsed'] / parallel['total_elapsed'] if parallel['total_elapsed'] > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Efficiency: {speedup / parallel['workers'] * 100:.1f}% "
          f"(ideal would be {parallel['workers']:.0f}x = 100%)")


def main():
    parser = argparse.ArgumentParser(description='Benchmark parallel Demucs processing')
    parser.add_argument('--workers', '-w', type=int, default=2,
                        help='Number of parallel workers (default: 2)')
    parser.add_argument('--files', '-f', type=int, default=2,
                        help='Number of files to process (default: 2)')
    parser.add_argument('--skip-sequential', action='store_true',
                        help='Skip sequential benchmark (just run parallel)')
    args = parser.parse_args()

    # Use subset of test files
    files = TEST_FILES[:args.files]

    print(f"Demucs Parallel Benchmark")
    print(f"Workers: {args.workers}, Files: {len(files)}")

    # Check files exist
    existing = [f for f in files if Path(f).exists()]
    if not existing:
        print("ERROR: No test files found!")
        return 1

    print(f"Found {len(existing)} test files")

    # Create temp output directories
    with tempfile.TemporaryDirectory() as tmpdir:
        seq_dir = Path(tmpdir) / 'sequential'
        par_dir = Path(tmpdir) / 'parallel'
        seq_dir.mkdir()
        par_dir.mkdir()

        # Run benchmarks
        if not args.skip_sequential:
            seq_result = benchmark_sequential(existing, seq_dir)
        else:
            # Fake sequential result for comparison
            seq_result = {
                'total_elapsed': 0,
                'total_audio_duration': sum(get_audio_duration(Path(f)) for f in existing),
                'realtime_factor': 0
            }

        par_result = benchmark_parallel(existing, par_dir, args.workers)

        if not args.skip_sequential:
            print_summary(seq_result, par_result)
        else:
            print(f"\n{'='*60}")
            print(f"PARALLEL RESULT ({args.workers} workers)")
            print(f"{'='*60}")
            print(f"Total time:     {par_result['total_elapsed']:.1f}s")
            print(f"Realtime factor: {par_result['realtime_factor']:.2f}x")

    return 0


if __name__ == '__main__':
    exit(main())
