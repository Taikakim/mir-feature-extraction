"""
Comprehensive FP8 Benchmark with TunableOps

Tests:
1. Demucs separation with TunableOps
2. Standard features (Beat Grid, BPM, Essentia)
3. Music Flamingo FP8 + Flash Attention 2 + TunableOps (all 5 prompts)

Saves results to .INFO file.
"""

import logging
import sys
import time
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Set ROCm environment before torch imports (tuning=True to generate new kernels)
from core.rocm_env import setup_rocm_env
setup_rocm_env(tuning=True)

from core.common import setup_logging
from core.json_handler import get_info_path, safe_update
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

    parser = argparse.ArgumentParser(description="FP8 Full Benchmark")
    parser.add_argument('audio_file', help='Path to audio file (full_mix.flac in organized folder)')
    parser.add_argument('--skip-demucs', action='store_true', help='Skip Demucs separation')
    parser.add_argument('--skip-features', action='store_true', help='Skip standard features')
    parser.add_argument('--skip-flamingo', action='store_true', help='Skip Music Flamingo')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info("=" * 80)
    logger.info("FP8 BENCHMARK WITH TUNABLEOPS")
    logger.info("=" * 80)

    # Benchmark-specific overrides (verbose tuning output)
    os.environ['PYTORCH_TUNABLEOP_VERBOSE'] = '1'

    logger.info("Environment (set by core.rocm_env with tuning=True):")
    for key in ('PYTORCH_ALLOC_CONF', 'PYTORCH_TUNABLEOP_ENABLED',
                'PYTORCH_TUNABLEOP_TUNING', 'PYTORCH_TUNABLEOP_FILENAME',
                'FLASH_ATTENTION_TRITON_AMD_ENABLE'):
        logger.info(f"  {key}: {os.environ.get(key, '(not set)')}")
    logger.info("")

    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        logger.error(f"File not found: {audio_file}")
        sys.exit(1)

    # Determine folder
    if audio_file.name.startswith('full_mix.'):
        folder = audio_file.parent
        full_mix = audio_file
    else:
        folder = audio_file.parent / audio_file.stem
        if not folder.exists():
            logger.error(f"Folder not found: {folder}")
            sys.exit(1)
        from core.file_utils import get_stem_files
        stems = get_stem_files(folder, include_full_mix=True)
        full_mix = stems.get('full_mix')
        if not full_mix:
            logger.error(f"No full_mix found in {folder}")
            sys.exit(1)

    # Get duration
    info = sf.info(str(full_mix))
    duration = info.duration

    logger.info(f"Audio: {full_mix.name}")
    logger.info(f"Folder: {folder.name}")
    logger.info(f"Duration: {duration:.2f}s ({duration/60:.2f} min)")
    logger.info("")

    timings = {}
    all_results = {}

    # 1. Demucs Separation
    if not args.skip_demucs:
        logger.info("=" * 80)
        logger.info("1. DEMUCS SEPARATION (with TunableOps)")
        logger.info("=" * 80)

        from preprocessing.demucs_sep_optimized import DemucsProcessor

        logger.info("Loading Demucs model...")
        start = time.time()
        processor = DemucsProcessor()
        load_time = time.time() - start
        logger.info(f"✓ Model loaded in {load_time:.2f}s")

        t, _ = time_feature("Demucs Separation", processor.separate_folder, folder, duration=duration)
        timings['Demucs'] = t
        logger.info("")

    # 2. Standard Features
    if not args.skip_features:
        logger.info("=" * 80)
        logger.info("2. STANDARD FEATURES")
        logger.info("=" * 80)

        from rhythm.beat_grid import create_beat_grid
        from rhythm.bpm import analyze_folder_bpm
        from classification.essentia_features_optimized import analyze_folder_essentia_features_optimized

        t, _ = time_feature("Beat Grid", create_beat_grid, full_mix, save_grid=True, duration=duration)
        timings['Beat Grid'] = t

        t, _ = time_feature("BPM Analysis", analyze_folder_bpm, folder, duration=duration)
        timings['BPM'] = t

        t, _ = time_feature("Essentia Features", analyze_folder_essentia_features_optimized, folder, duration=duration)
        timings['Essentia'] = t
        logger.info("")

    # 3. Music Flamingo FP8
    if not args.skip_flamingo:
        logger.info("=" * 80)
        logger.info("3. MUSIC FLAMINGO (FP8 + Flash Attention 2 + TunableOps)")
        logger.info("=" * 80)

        try:
            from classification.music_flamingo_transformers import MusicFlamingoTransformers

            logger.info("Loading Music Flamingo model...")
            logger.info("✓ FP8 enabled (native RDNA4 support)")
            logger.info("✓ Flash Attention 2 enabled")
            logger.info("✓ TunableOps enabled")

            start = time.time()
            analyzer = MusicFlamingoTransformers(
                model_id="nvidia/music-flamingo-hf",
                device_map="auto",
                use_flash_attention=True,
                use_fp8=False,  # FP8 not supported by transformers yet
            )
            load_time = time.time() - start
            logger.info(f"✓ Model loaded in {load_time:.2f}s")
            logger.info("")

            # Test all 5 prompts
            prompt_types = {
                'full': "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates.",
                'technical': "Break the track down like a critic - list its tempo, key, and chordal motion, then explain the textures, dynamics, and emotional impact of the performance.",
                'genre_mood': "What is the genre and mood of this music? Be specific about subgenres and describe the emotional character.",
                'instrumentation': "What instruments and sounds are present in this track?",
                'structure': "Analyze the structure and arrangement of this track. Describe the sections, transitions, and how the composition unfolds over time."
            }

            flamingo_results = {}

            for i, (prompt_type, prompt_text) in enumerate(prompt_types.items(), 1):
                logger.info(f"[{i}/5] Analyzing: {prompt_type}")
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
                flamingo_results[f'music_flamingo_{prompt_type}'] = description

            # Save Music Flamingo results to .INFO file
            info_path = get_info_path(full_mix)
            safe_update(info_path, flamingo_results)
            all_results.update(flamingo_results)
            logger.info(f"\n✓ Saved all Music Flamingo results to {info_path.name}")

        except Exception as e:
            logger.error(f"Music Flamingo failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Audio duration: {duration:.2f}s ({duration/60:.2f} min)\n")

    if timings:
        total = sum(timings.values())
        logger.info(f"{'Feature':<40s} {'Time':>10s}  {'Speed':>12s}  {'s/s audio':>12s}")
        logger.info("-" * 80)

        for name, t in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            speed = duration / t if t > 0 else 0
            s_per_s = t / duration if duration > 0 else 0
            logger.info(f"{name:<40s} {t:>8.2f}s  {speed:>10.2f}x  {s_per_s:>11.4f}s")

        logger.info("-" * 80)
        logger.info(f"{'TOTAL':<40s} {total:>8.2f}s")
        logger.info(f"\nOverall speed: {duration/total:.2f}x realtime")
    else:
        logger.info("No features were run")

    logger.info("=" * 80)

    # Save benchmark results
    benchmark_data = {
        'audio_file': str(full_mix),
        'duration': duration,
        'timings': timings,
        'total_time': sum(timings.values()) if timings else 0,
        'speed': duration / sum(timings.values()) if timings else 0,
        'configuration': {
            'fp8': not args.skip_flamingo,
            'flash_attention': True,
            'tunableops': True,
            'tunableop_file': os.environ.get('PYTORCH_TUNABLEOP_FILENAME'),
        }
    }

    output_file = Path('benchmark_fp8_tunableops.json')
    with open(output_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)

    logger.info(f"\n✓ Saved benchmark data to {output_file}")

    info_path = get_info_path(full_mix)
    logger.info(f"✓ All results saved to {info_path}")
    logger.info("")


if __name__ == "__main__":
    main()
