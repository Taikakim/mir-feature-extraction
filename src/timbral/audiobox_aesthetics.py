"""
Audiobox Aesthetics Features for MIR Project

This module extracts subjective quality assessment features using Meta's Audiobox Aesthetics model.

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
- content_enjoyment: Enjoyment rating 1-10 (NumberConditioner)
- content_usefulness: Usefulness rating 1-10 (NumberConditioner)
- production_complexity: Complexity rating 1-10 (NumberConditioner)
- production_quality: Quality rating 1-10 (NumberConditioner)

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


def batch_analyze_audiobox_aesthetics(root_directory: str | Path,
                                       overwrite: bool = False) -> dict:
    """
    Batch analyze Audiobox aesthetics for all organized folders.

    Args:
        root_directory: Root directory to search
        overwrite: Whether to overwrite existing aesthetics data

    Returns:
        Dictionary with statistics about the batch processing
    """
    from core.file_utils import find_organized_folders

    root_directory = Path(root_directory)
    logger.info(f"Starting batch Audiobox aesthetics analysis: {root_directory}")

    folders = find_organized_folders(root_directory)

    stats = {
        'total': len(folders),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }

    logger.info(f"Found {stats['total']} organized folders")

    for i, folder in enumerate(folders, 1):
        logger.info(f"Processing {i}/{stats['total']}: {folder.name}")

        # Find full_mix file
        stems = get_stem_files(folder, include_full_mix=True)
        if 'full_mix' not in stems:
            logger.warning(f"No full_mix found in {folder.name}")
            stats['failed'] += 1
            continue

        # Check if already processed
        info_path = get_info_path(stems['full_mix'])
        if info_path.exists() and not overwrite:
            try:
                import json
                with open(info_path, 'r') as f:
                    data = json.load(f)
                if 'content_enjoyment' in data:
                    logger.info("Audiobox aesthetics data already exists. Use --overwrite to regenerate.")
                    stats['skipped'] += 1
                    continue
            except Exception:
                pass


        try:
            results = analyze_audiobox_aesthetics(stems['full_mix'])

            # Save to .INFO file
            safe_update(info_path, results)

            stats['success'] += 1

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"Failed to process {folder.name}: {e}")

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
                overwrite=args.overwrite
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
