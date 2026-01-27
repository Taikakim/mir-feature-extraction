"""
This module extracts subjective quality assessment features using Meta's Audiobox Aesthetics model on a 1-10 scale:

**Performance**: ~100s for 7.5min track (4.5x realtime), uses GPU via WavLM encoder
**Batch Mode**: Processes multiple files per GPU call for better utilization (--batch-size)

Audiobox Aesthetics provides four key metrics on a 1-10 scale:
- Content Enjoyment (CE): How enjoyable the audio content is
- Content Usefulness (CU): How useful the content is for its purpose
- Production Complexity (PC): How complex the production/arrangement is
- Production Quality (PQ): How high the production quality is

These features enable conditioning on subjective quality dimensions beyond
objective technical measurements.

Dependencies:
- audiobox_aesthetics (pip install git+https://github.com/facebookresearch/audiobox-aesthetics.git)
- torch
- torchaudio
- src.core.file_utils
- src.core.common

Output:
- content_enjoyment: Enjoyment rating 1-10
- content_usefulness: Usefulness rating 1-10
- production_complexity: Complexity rating 1-10
- production_quality: Quality rating 1-10
Installation:
    pip install git+https://github.com/facebookresearch/audiobox-aesthetics.git
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import get_stem_files
from core.common import clamp_feature_value
from core.json_handler import safe_update, get_info_path

logger = logging.getLogger(__name__)

# Try to import audiobox_aesthetics
try:
    from audiobox_aesthetics.infer import initialize_predictor
    AUDIOBOX_AVAILABLE = True
except ImportError:
    AUDIOBOX_AVAILABLE = False
    warnings.warn(
        "Audiobox Aesthetics not installed. Install with: "
        "pip install git+https://github.com/facebookresearch/audiobox-aesthetics.git",
        category=UserWarning
    )

# Global predictor instance for efficiency
_predictor = None


def get_predictor():
    """Get or initialize the AudioBox predictor (singleton pattern)."""
    global _predictor
    if _predictor is None and AUDIOBOX_AVAILABLE:
        logger.info("Initializing AudioBox Aesthetics predictor...")
        _predictor = initialize_predictor()
        logger.info("AudioBox Aesthetics predictor initialized")
    return _predictor


def analyze_audiobox_aesthetics(audio_path: str | Path) -> Dict[str, float]:
    """
    Analyze audiobox aesthetics features for an audio file.

    Uses Meta's Audiobox Aesthetics model to predict subjective quality metrics.

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with aesthetics features:
        - content_enjoyment: 1-10 scale
        - content_usefulness: 1-10 scale
        - production_complexity: 1-10 scale
        - production_quality: 1-10 scale
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Analyzing Audiobox aesthetics: {audio_path.name}")

    if not AUDIOBOX_AVAILABLE:
        logger.warning("Audiobox Aesthetics not available - returning placeholder values")
        results = {
            'content_enjoyment': 5.5,
            'content_usefulness': 5.5,
            'production_complexity': 5.5,
            'production_quality': 5.5
        }
        for key, value in results.items():
            results[key] = clamp_feature_value(key, value)
        return results

    # Get or initialize the predictor
    predictor = get_predictor()
    if predictor is None:
        logger.error("Failed to initialize AudioBox predictor")
        return {
            'content_enjoyment': 5.5,
            'content_usefulness': 5.5,
            'production_complexity': 5.5,
            'production_quality': 5.5
        }

    # Run inference
    # The predictor expects a list of dicts with 'path' key
    predictions = predictor.forward([{"path": str(audio_path)}])

    # Extract results from the first (and only) prediction
    # Output format: {"CE": 5.146, "CU": 5.779, "PC": 2.148, "PQ": 7.220}
    pred = predictions[0]

    results = {
        'content_enjoyment': float(pred.get('CE', 5.5)),
        'content_usefulness': float(pred.get('CU', 5.5)),
        'production_complexity': float(pred.get('PC', 5.5)),
        'production_quality': float(pred.get('PQ', 5.5))
    }

    # Clamp values to valid ranges (1-10)
    for key, value in results.items():
        results[key] = clamp_feature_value(key, value)

    logger.info("Audiobox aesthetics results:")
    logger.info(f"  Content Enjoyment:      {results['content_enjoyment']:.2f}")
    logger.info(f"  Content Usefulness:     {results['content_usefulness']:.2f}")
    logger.info(f"  Production Complexity:  {results['production_complexity']:.2f}")
    logger.info(f"  Production Quality:     {results['production_quality']:.2f}")

    return results


def analyze_audiobox_aesthetics_batch(audio_paths: list) -> list:
    """
    Analyze audiobox aesthetics for multiple audio files in a single batch.

    This is more efficient than calling analyze_audiobox_aesthetics() repeatedly
    because the model processes all files together on the GPU.

    Args:
        audio_paths: List of paths to audio files

    Returns:
        List of dictionaries with aesthetics features for each file
    """
    if not audio_paths:
        return []

    if not AUDIOBOX_AVAILABLE:
        logger.warning("Audiobox Aesthetics not available - returning placeholder values")
        return [{
            'content_enjoyment': 5.5,
            'content_usefulness': 5.5,
            'production_complexity': 5.5,
            'production_quality': 5.5
        } for _ in audio_paths]

    predictor = get_predictor()
    if predictor is None:
        return [{
            'content_enjoyment': 5.5,
            'content_usefulness': 5.5,
            'production_complexity': 5.5,
            'production_quality': 5.5
        } for _ in audio_paths]

    # Build batch input
    batch_input = [{"path": str(p)} for p in audio_paths]

    logger.info(f"Processing batch of {len(audio_paths)} files...")

    # Run inference on entire batch
    predictions = predictor.forward(batch_input)

    # Convert results
    results = []
    for pred in predictions:
        result = {
            'content_enjoyment': float(pred.get('CE', 5.5)),
            'content_usefulness': float(pred.get('CU', 5.5)),
            'production_complexity': float(pred.get('PC', 5.5)),
            'production_quality': float(pred.get('PQ', 5.5))
        }
        # Clamp values
        for key, value in result.items():
            result[key] = clamp_feature_value(key, value)
        results.append(result)

    return results


def batch_analyze_audiobox_aesthetics(root_directory: str | Path,
                                       overwrite: bool = False,
                                       batch_size: int = 8) -> dict:
    """
    Batch analyze Audiobox aesthetics for all organized folders.

    Uses batched inference for better GPU utilization - processes multiple
    files in a single forward pass.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing aesthetics data
        batch_size: Number of files to process in each batch (default: 8)

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders
    from core.common import ProgressBar
    import json

    root_directory = Path(root_directory)
    logger.info(f"Starting batch Audiobox aesthetics analysis: {root_directory}")
    logger.info(f"Batch size: {batch_size}")

    folders = find_organized_folders(root_directory)

    stats = {
        'total': len(folders),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }

    logger.info(f"Found {stats['total']} organized folders")

    # Collect files that need processing
    files_to_process = []  # List of (folder, full_mix_path, info_path)

    for folder in folders:
        stems = get_stem_files(folder, include_full_mix=True)
        if 'full_mix' not in stems:
            stats['failed'] += 1
            continue

        info_path = get_info_path(stems['full_mix'])

        # Check if already processed
        if info_path.exists() and not overwrite:
            try:
                with open(info_path, 'r') as f:
                    data = json.load(f)
                if 'content_enjoyment' in data:
                    stats['skipped'] += 1
                    continue
            except Exception:
                pass

        files_to_process.append((folder, stems['full_mix'], info_path))

    logger.info(f"Files to process: {len(files_to_process)} (skipped: {stats['skipped']})")

    if not files_to_process:
        logger.info("No files to process")
        return stats

    # Initialize predictor once before batch processing
    if AUDIOBOX_AVAILABLE:
        get_predictor()

    # Process in batches
    progress = ProgressBar(len(files_to_process), desc="AudioBox")

    for batch_start in range(0, len(files_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(files_to_process))
        batch = files_to_process[batch_start:batch_end]

        # Extract paths for this batch
        audio_paths = [item[1] for item in batch]

        try:
            # Process entire batch at once
            batch_results = analyze_audiobox_aesthetics_batch(audio_paths)

            # Save results for each file
            for (folder, full_mix, info_path), results in zip(batch, batch_results):
                try:
                    safe_update(info_path, results)
                    stats['success'] += 1
                except Exception as e:
                    stats['failed'] += 1
                    stats['errors'].append(f"{folder.name}: {e}")

        except Exception as e:
            # If batch fails, mark all as failed
            for folder, _, _ in batch:
                stats['failed'] += 1
                stats['errors'].append(f"{folder.name}: {e}")
            logger.error(f"Batch failed: {e}")

        # Update progress
        processed = min(batch_end, len(files_to_process))
        logger.info(progress.update(processed, f"{stats['success']} ok"))

    logger.info(progress.finish(f"{stats['success']} success, {stats['failed']} failed"))

    # Summary
    logger.info("=" * 60)
    logger.info("Batch Audiobox Aesthetics Analysis Summary:")
    logger.info(f"  Total folders:  {stats['total']}")
    logger.info(f"  Successful:     {stats['success']}")
    logger.info(f"  Skipped:        {stats['skipped']}")
    logger.info(f"  Failed:         {stats['failed']}")
    logger.info("=" * 60)

    return stats


# Command-line interface
if __name__ == "__main__":
    import argparse
    from core.common import setup_logging

    parser = argparse.ArgumentParser(
        description="Analyze Audiobox aesthetics features"
    )

    parser.add_argument(
        'path',
        type=str,
        help='Path to audio file or organized folder'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process all organized folders in directory tree'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing aesthetics data'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Number of files to process in each batch (default: 8)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    path = Path(args.path)

    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)

    try:
        if args.batch:
            # Batch processing
            stats = batch_analyze_audiobox_aesthetics(
                path,
                overwrite=args.overwrite,
                batch_size=args.batch_size
            )

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} folders failed to process")
                sys.exit(1)

        elif path.is_dir():
            # Single folder
            stems = get_stem_files(path, include_full_mix=True)
            if 'full_mix' not in stems:
                logger.error(f"No full_mix file found in {path}")
                sys.exit(1)

            results = analyze_audiobox_aesthetics(stems['full_mix'])

            # Save to .INFO
            info_path = get_info_path(path)
            safe_update(info_path, results)

            print(f"\nAudiobox Aesthetics Results:")
            print(f"  Content Enjoyment:      {results['content_enjoyment']:.1f}/10")
            print(f"  Content Usefulness:     {results['content_usefulness']:.1f}/10")
            print(f"  Production Complexity:  {results['production_complexity']:.1f}/10")
            print(f"  Production Quality:     {results['production_quality']:.1f}/10")

        else:
            # Single file
            results = analyze_audiobox_aesthetics(path)

            print(f"\nAudiobox Aesthetics Results:")
            print(f"  Content Enjoyment:      {results['content_enjoyment']:.1f}/10")
            print(f"  Content Usefulness:     {results['content_usefulness']:.1f}/10")
            print(f"  Production Complexity:  {results['production_complexity']:.1f}/10")
            print(f"  Production Quality:     {results['production_quality']:.1f}/10")

        logger.info("Audiobox aesthetics analysis completed")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
