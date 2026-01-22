"""
Threading Analysis for MIR Features

Tests each feature on a single audio file while monitoring CPU usage
to determine which features already use internal multithreading.

This helps avoid thread oversubscription when implementing batch parallelization.

Usage:
    python test_threading.py /test_data/test.wav

Output:
    - CPU usage per feature
    - Thread count per feature
    - Recommendations for parallelization
"""

import argparse
import logging
import multiprocessing
import os
import psutil
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from core.common import setup_logging

logger = logging.getLogger(__name__)


class ThreadMonitor:
    """Monitor CPU and thread usage during feature extraction."""

    def __init__(self):
        self.cpu_samples = []
        self.thread_samples = []
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()

    def start(self):
        """Start monitoring in background thread."""
        self.monitoring = True
        self.cpu_samples = []
        self.thread_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)

        if not self.cpu_samples:
            return {'cpu_mean': 0, 'cpu_max': 0, 'threads_mean': 1, 'threads_max': 1}

        return {
            'cpu_mean': sum(self.cpu_samples) / len(self.cpu_samples),
            'cpu_max': max(self.cpu_samples),
            'threads_mean': sum(self.thread_samples) / len(self.thread_samples),
            'threads_max': max(self.thread_samples)
        }

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # Get CPU usage (percentage of one core = 100%, all cores = 100% * n_cores)
                cpu_percent = self.process.cpu_percent(interval=0.1)
                self.cpu_samples.append(cpu_percent)

                # Get thread count
                thread_count = self.process.num_threads()
                self.thread_samples.append(thread_count)

            except Exception as e:
                logger.debug(f"Monitoring error: {e}")
                break

            time.sleep(0.1)


def test_feature(feature_name: str, test_func: callable, audio_path: Path) -> Dict:
    """
    Test a single feature and measure CPU/thread usage.

    Args:
        feature_name: Name of the feature
        test_func: Function to call for testing
        audio_path: Path to test audio file

    Returns:
        Dictionary with timing and resource usage stats
    """
    logger.info(f"\nTesting: {feature_name}")
    logger.info("-" * 60)

    monitor = ThreadMonitor()

    # Start monitoring
    monitor.start()
    start_time = time.time()

    # Run feature extraction
    try:
        result = test_func(audio_path)
        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)
        logger.error(f"  Error: {e}")

    # Stop monitoring
    elapsed = time.time() - start_time
    stats = monitor.stop()

    # Determine if multithreaded
    # If average CPU > 150% or max threads > 3, likely using multithreading
    is_multithreaded = stats['cpu_max'] > 150 or stats['threads_max'] > 3

    result = {
        'feature': feature_name,
        'success': success,
        'error': error,
        'elapsed': elapsed,
        'cpu_mean': stats['cpu_mean'],
        'cpu_max': stats['cpu_max'],
        'threads_mean': stats['threads_mean'],
        'threads_max': stats['threads_max'],
        'is_multithreaded': is_multithreaded,
        'recommendation': 'DO NOT parallelize' if is_multithreaded else 'Parallelize OK'
    }

    logger.info(f"  Time: {elapsed:.2f}s")
    logger.info(f"  CPU: {stats['cpu_mean']:.1f}% (mean), {stats['cpu_max']:.1f}% (max)")
    logger.info(f"  Threads: {stats['threads_mean']:.1f} (mean), {stats['threads_max']:.0f} (max)")
    logger.info(f"  Multithreaded: {'YES' if is_multithreaded else 'NO'}")
    logger.info(f"  → {result['recommendation']}")

    return result


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Test which MIR features use internal multithreading"
    )

    parser.add_argument(
        'audio_file',
        type=str,
        help='Path to test audio file (e.g., /test_data/test.wav)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return 1

    logger.info("=" * 60)
    logger.info("MIR Feature Threading Analysis")
    logger.info("=" * 60)
    logger.info(f"Test file: {audio_path}")
    logger.info(f"System CPUs: {multiprocessing.cpu_count()}")
    logger.info("")

    results = []

    # Test each feature group
    # Note: We need organized structure for most features, so we'll test individual functions

    # ========================================================================
    # Loudness Features (librosa-based)
    # ========================================================================
    try:
        from preprocessing.loudness import analyze_file_loudness

        def test_loudness(path):
            return analyze_file_loudness(path)

        results.append(test_feature('Loudness (LUFS/LRA)', test_loudness, audio_path))
    except Exception as e:
        logger.error(f"Could not test loudness: {e}")

    # ========================================================================
    # Spectral Features (librosa FFT)
    # ========================================================================
    try:
        from spectral.spectral_features import analyze_spectral_features

        def test_spectral(path):
            return analyze_spectral_features(path)

        results.append(test_feature('Spectral Features', test_spectral, audio_path))
    except Exception as e:
        logger.error(f"Could not test spectral: {e}")

    # ========================================================================
    # Rhythm Features (librosa beat tracking)
    # ========================================================================
    try:
        from rhythm.tempo_bpm import analyze_tempo

        def test_tempo(path):
            return analyze_tempo(path)

        results.append(test_feature('Tempo/BPM', test_tempo, audio_path))
    except Exception as e:
        logger.error(f"Could not test tempo: {e}")

    try:
        from rhythm.onsets import analyze_onsets

        def test_onsets(path):
            return analyze_onsets(path)

        results.append(test_feature('Onsets', test_onsets, audio_path))
    except Exception as e:
        logger.error(f"Could not test onsets: {e}")

    # ========================================================================
    # Harmonic Features (librosa chroma)
    # ========================================================================
    try:
        from harmonic.chroma import analyze_chroma

        def test_chroma(path):
            return analyze_chroma(path)

        results.append(test_feature('Chroma', test_chroma, audio_path))
    except Exception as e:
        logger.error(f"Could not test chroma: {e}")

    try:
        from harmonic.key_detection import detect_key

        def test_key(path):
            return detect_key(path)

        results.append(test_feature('Key Detection', test_key, audio_path))
    except Exception as e:
        logger.error(f"Could not test key: {e}")

    # ========================================================================
    # Timbral Features (librosa MFCCs)
    # ========================================================================
    try:
        from timbral.mfcc import analyze_mfcc

        def test_mfcc(path):
            return analyze_mfcc(path)

        results.append(test_feature('MFCC', test_mfcc, audio_path))
    except Exception as e:
        logger.error(f"Could not test mfcc: {e}")

    try:
        from timbral.audio_commons import analyze_audio_commons_timbral

        def test_audio_commons(path):
            return analyze_audio_commons_timbral(path)

        results.append(test_feature('Audio Commons Timbral', test_audio_commons, audio_path))
    except Exception as e:
        logger.error(f"Could not test audio_commons: {e}")

    # ========================================================================
    # Classification (Essentia - TensorFlow)
    # ========================================================================
    try:
        from classification.essentia_features_optimized import EssentiaAnalyzer

        # Load model once
        logger.info("\nLoading Essentia models...")
        analyzer = EssentiaAnalyzer()

        def test_essentia_dance(path):
            return analyzer.analyze_danceability(path)

        results.append(test_feature('Essentia Danceability', test_essentia_dance, audio_path))

        def test_essentia_atonal(path):
            return analyzer.analyze_atonality(path)

        results.append(test_feature('Essentia Atonality', test_essentia_atonal, audio_path))

    except Exception as e:
        logger.error(f"Could not test essentia: {e}")

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("THREADING ANALYSIS SUMMARY")
    logger.info("=" * 60)

    logger.info("\n📊 Detailed Results:")
    logger.info("")
    logger.info(f"{'Feature':<30} {'Time':<8} {'CPU Max':<10} {'Threads':<8} {'Multithreaded':<15} {'Recommendation'}")
    logger.info("-" * 105)

    for r in results:
        if r['success']:
            mt_marker = "✓ YES" if r['is_multithreaded'] else "✗ NO"
            logger.info(
                f"{r['feature']:<30} "
                f"{r['elapsed']:>6.2f}s  "
                f"{r['cpu_max']:>7.1f}%  "
                f"{r['threads_max']:>6.0f}    "
                f"{mt_marker:<15} "
                f"{r['recommendation']}"
            )

    # Categorize features
    multithreaded = [r for r in results if r['success'] and r['is_multithreaded']]
    single_threaded = [r for r in results if r['success'] and not r['is_multithreaded']]

    logger.info("\n" + "=" * 60)
    logger.info("PARALLELIZATION RECOMMENDATIONS")
    logger.info("=" * 60)

    if single_threaded:
        logger.info(f"\n✅ Features to PARALLELIZE ({len(single_threaded)}):")
        logger.info("These are single-threaded and will benefit from ProcessPoolExecutor:")
        for r in single_threaded:
            logger.info(f"  - {r['feature']}")

    if multithreaded:
        logger.info(f"\n⚠️  Features to RUN SERIALLY ({len(multithreaded)}):")
        logger.info("These already use multithreading internally:")
        for r in multithreaded:
            logger.info(f"  - {r['feature']} (uses {r['threads_max']:.0f} threads, {r['cpu_max']:.0f}% CPU)")

    logger.info("\n" + "=" * 60)
    logger.info("IMPLEMENTATION STRATEGY")
    logger.info("=" * 60)
    logger.info("""
For batch processing, use this strategy:

1. Single-threaded features → ProcessPoolExecutor with N workers
   - Each worker processes different files in parallel
   - Example: 20 workers for 24-core CPU

2. Multithreaded features → Run sequentially (no ProcessPoolExecutor)
   - Each file uses multiple cores internally
   - Avoid thread oversubscription

3. Configuration:
   - Add 'parallel_workers' per feature in config
   - Set to 0 for multithreaded features
   - Set to N (e.g., 20) for single-threaded features
    """)

    # Generate configuration snippet
    logger.info("\n" + "=" * 60)
    logger.info("SUGGESTED CONFIGURATION")
    logger.info("=" * 60)

    config_suggestion = {}
    for r in results:
        if r['success']:
            feature_key = r['feature'].lower().replace(' ', '_').replace('/', '_')
            config_suggestion[feature_key] = {
                'enabled': True,
                'overwrite': False,
                'parallel_workers': 0 if r['is_multithreaded'] else 20
            }

    import json
    logger.info(json.dumps({'features': config_suggestion}, indent=2))

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
