"""
Master Batch Processing Script for MIR Pipeline

This script provides a unified interface for running the complete MIR feature extraction
pipeline with full configuration support, progress tracking, and resume capability.

Features:
- Organize audio files into required structure
- Extract all supported audio features
- Parallel processing with file locking
- Resume capability (skip completed features)
- Per-feature overwrite control
- JSON configuration file support

Usage:
    # Run full pipeline with default settings
    python batch_process.py /path/to/audio --output-dir /path/to/organized

    # Use configuration file
    python batch_process.py /path/to/audio --config config.json

    # Run specific features only
    python batch_process.py /path/to/audio --features lufs bpm chroma

    # Force overwrite specific features
    python batch_process.py /path/to/audio --overwrite lufs bpm

    # Parallel processing
    python batch_process.py /path/to/audio --workers 20

Dependencies:
- All MIR feature extraction modules
- core.batch_utils, core.file_locks
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from core.common import setup_logging
from core.file_utils import find_organized_folders
from core.batch_utils import (
    has_features,
    get_missing_features,
    print_batch_summary,
    get_progress_stats
)
from core.file_locks import FileLock, cleanup_dead_locks

logger = logging.getLogger(__name__)


# ============================================================================
# Feature Module Registry
# ============================================================================

FEATURE_MODULES = {
    # Preprocessing
    'separation': {
        'module': 'preprocessing.demucs_sep_optimized',
        'function': 'batch_separate_stems_optimized',
        'features': ['separated'],
        'description': 'Demucs stem separation (OPTIMIZED - model caching)'
    },

    # Loudness Analysis
    'loudness': {
        'module': 'preprocessing.loudness',
        'function': 'batch_analyze_loudness',
        'features': ['lufs', 'lra', 'lufs_drums', 'lra_drums',
                     'lufs_bass', 'lra_bass', 'lufs_other', 'lra_other'],
        'description': 'LUFS loudness analysis'
    },

    # Spectral Features
    'spectral': {
        'module': 'spectral.spectral_features',
        'function': 'batch_analyze_spectral_features',
        'features': ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                     'spectral_flatness', 'spectral_contrast'],
        'description': 'Spectral characteristics'
    },

    # Rhythm Features
    'rhythm': {
        'module': 'rhythm.tempo_bpm',
        'function': 'batch_analyze_tempo',
        'features': ['bpm', 'beat_strength'],
        'description': 'Tempo and beat analysis'
    },

    'onsets': {
        'module': 'rhythm.onsets',
        'function': 'batch_analyze_onsets',
        'features': ['onset_rate', 'onset_strength'],
        'description': 'Onset detection'
    },

    # Harmonic Features
    'chroma': {
        'module': 'harmonic.chroma',
        'function': 'batch_analyze_chroma',
        'features': ['chroma_mean', 'chroma_std', 'chroma_cens'],
        'description': 'Chroma features'
    },

    'key': {
        'module': 'harmonic.key_detection',
        'function': 'batch_detect_key',
        'features': ['key', 'scale'],
        'description': 'Key and scale detection'
    },

    # Timbral Features
    'mfcc': {
        'module': 'timbral.mfcc',
        'function': 'batch_analyze_mfcc',
        'features': ['mfcc_mean', 'mfcc_std'],
        'description': 'MFCC coefficients'
    },

    'audio_commons': {
        'module': 'timbral.audio_commons',
        'function': 'batch_analyze_audio_commons',
        'features': ['brightness', 'depth', 'hardness', 'roughness',
                     'warmth', 'sharpness', 'boominess'],
        'description': 'Audio Commons timbral features'
    },

    # Classification
    'essentia': {
        'module': 'classification.essentia_features_optimized',
        'function': 'batch_analyze_essentia_features_optimized',
        'features': ['danceability', 'atonality'],
        'description': 'Essentia ML features (optimized)'
    },
}


# ============================================================================
# Configuration Management
# ============================================================================

class BatchConfig:
    """Configuration manager for batch processing."""

    DEFAULT_CONFIG = {
        'input_directory': None,
        'output_directory': None,
        'organize_files': True,
        'move_files': False,  # Copy by default (preserve originals)
        'workers': 1,  # Serial by default (parallel coming soon)
        'features': {
            # Which feature groups to process
            'separation': {'enabled': False, 'overwrite': False},
            'loudness': {'enabled': True, 'overwrite': False},
            'spectral': {'enabled': True, 'overwrite': False},
            'rhythm': {'enabled': True, 'overwrite': False},
            'onsets': {'enabled': True, 'overwrite': False},
            'chroma': {'enabled': True, 'overwrite': False},
            'key': {'enabled': True, 'overwrite': False},
            'mfcc': {'enabled': True, 'overwrite': False},
            'audio_commons': {'enabled': True, 'overwrite': False},
            'essentia': {'enabled': True, 'overwrite': False},
        },
        'file_types': ['.flac', '.wav', '.mp3', '.m4a', '.ogg'],
        'recursive': True,
        'lock_timeout': 3600,  # 1 hour
        'cleanup_locks': True,
    }

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration from file or defaults."""
        self.config = self.DEFAULT_CONFIG.copy()

        if config_path and config_path.exists():
            self.load_from_file(config_path)

    def load_from_file(self, config_path: Path):
        """Load configuration from JSON file."""
        logger.info(f"Loading configuration from: {config_path}")
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)

            # Merge with defaults
            self._merge_config(user_config)
            logger.info("Configuration loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            logger.info("Using default configuration")

    def _merge_config(self, user_config: dict):
        """Merge user config with defaults."""
        for key, value in user_config.items():
            if key in self.config:
                if isinstance(value, dict) and isinstance(self.config[key], dict):
                    # Merge nested dicts (like features)
                    self.config[key].update(value)
                else:
                    self.config[key] = value

    def save_template(self, output_path: Path):
        """Save a template configuration file."""
        with open(output_path, 'w') as f:
            json.dump(self.DEFAULT_CONFIG, f, indent=2)
        logger.info(f"Saved configuration template to: {output_path}")

    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature group is enabled."""
        features = self.config.get('features', {})
        feature_config = features.get(feature, {})
        return feature_config.get('enabled', False)

    def should_overwrite_feature(self, feature: str) -> bool:
        """Check if a feature should be overwritten."""
        features = self.config.get('features', {})
        feature_config = features.get(feature, {})
        return feature_config.get('overwrite', False)


# ============================================================================
# Pipeline Steps
# ============================================================================

def step_organize_files(config: BatchConfig) -> bool:
    """Step 1: Organize audio files into required structure."""
    if not config.get('organize_files'):
        logger.info("Skipping file organization (disabled in config)")
        return True

    logger.info("=" * 60)
    logger.info("STEP 1: Organizing Audio Files")
    logger.info("=" * 60)

    from preprocessing.file_organizer import organize_directory

    input_dir = Path(config.get('input_directory'))
    output_dir = Path(config.get('output_directory')) if config.get('output_directory') else None

    try:
        stats = organize_directory(
            directory=input_dir,
            output_dir=output_dir,
            move=config.get('move_files', False),
            dry_run=False,
            recursive=config.get('recursive', True)
        )

        if stats['failed'] > 0:
            logger.warning(f"Some files failed to organize ({stats['failed']} failures)")
            return False

        logger.info(f"✓ Organization complete: {stats['organized']} files")
        return True

    except Exception as e:
        logger.error(f"File organization failed: {e}")
        return False


def step_extract_features(config: BatchConfig) -> Dict[str, any]:
    """Step 2: Extract all enabled features."""
    logger.info("=" * 60)
    logger.info("STEP 2: Extracting Features")
    logger.info("=" * 60)

    # Get working directory (output_dir if set, else input_dir)
    work_dir = Path(config.get('output_directory') or config.get('input_directory'))

    # Find organized folders
    folders = find_organized_folders(work_dir)
    logger.info(f"Found {len(folders)} organized folders")

    if not folders:
        logger.error("No organized folders found - did file organization succeed?")
        return {'success': False}

    # Clean up dead locks if enabled
    if config.get('cleanup_locks'):
        cleanup_dead_locks(work_dir, timeout=config.get('lock_timeout'))

    # Process each enabled feature group
    results = {}

    for feature_name, feature_info in FEATURE_MODULES.items():
        if not config.is_feature_enabled(feature_name):
            logger.info(f"Skipping {feature_name} (disabled in config)")
            continue

        logger.info("=" * 60)
        logger.info(f"Processing: {feature_info['description']}")
        logger.info("=" * 60)

        try:
            # Import module dynamically
            module_path = feature_info['module']
            function_name = feature_info['function']

            module = __import__(module_path, fromlist=[function_name])
            batch_func = getattr(module, function_name)

            # Check if function supports overwrite parameter
            overwrite = config.should_overwrite_feature(feature_name)

            # Call batch function
            try:
                stats = batch_func(work_dir, overwrite=overwrite)
            except TypeError:
                # Function doesn't support overwrite parameter
                logger.debug(f"Function {function_name} doesn't support overwrite parameter")
                stats = batch_func(work_dir)

            results[feature_name] = stats

        except Exception as e:
            logger.error(f"Failed to process {feature_name}: {e}")
            results[feature_name] = {'success': False, 'error': str(e)}

    return results


def step_show_summary(config: BatchConfig, results: Dict[str, any]):
    """Step 3: Show final summary."""
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)

    work_dir = Path(config.get('output_directory') or config.get('input_directory'))
    folders = find_organized_folders(work_dir)

    logger.info(f"Total folders processed: {len(folders)}")
    logger.info("")

    # Show per-feature results
    for feature_name, stats in results.items():
        if isinstance(stats, dict) and 'success' in stats:
            status = "✓" if stats.get('success') else "✗"
            logger.info(f"{status} {feature_name}: {stats}")

    logger.info("=" * 60)


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(config: BatchConfig) -> bool:
    """Run the complete MIR feature extraction pipeline."""
    start_time = time.time()

    logger.info("Starting MIR Feature Extraction Pipeline")
    logger.info(f"Input directory: {config.get('input_directory')}")
    logger.info(f"Output directory: {config.get('output_directory')}")
    logger.info(f"Workers: {config.get('workers')}")
    logger.info("")

    # Step 1: Organize files
    if not step_organize_files(config):
        logger.error("Pipeline failed at file organization step")
        return False

    # Step 2: Extract features
    results = step_extract_features(config)

    # Step 3: Show summary
    step_show_summary(config, results)

    elapsed = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed/60:.1f} minutes")

    return True


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Master batch processing script for MIR feature extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run full pipeline with output directory (preserves originals):
    python batch_process.py /path/to/audio --output-dir /path/to/organized

  Use configuration file:
    python batch_process.py /path/to/audio --config config.json

  Run specific features only:
    python batch_process.py /path/to/audio --features loudness spectral rhythm

  Generate configuration template:
    python batch_process.py --save-config template.json

  Show progress for dataset:
    python batch_process.py /path/to/organized --progress
        """
    )

    parser.add_argument(
        'directory',
        nargs='?',
        type=str,
        help='Input directory containing audio files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for organized files (preserves originals)'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )

    parser.add_argument(
        '--save-config',
        type=str,
        help='Save configuration template to file and exit'
    )

    parser.add_argument(
        '--features',
        nargs='+',
        choices=list(FEATURE_MODULES.keys()),
        help='Specific features to process (default: all enabled in config)'
    )

    parser.add_argument(
        '--overwrite',
        nargs='+',
        choices=list(FEATURE_MODULES.keys()),
        help='Force overwrite for specific features'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )

    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying (DESTRUCTIVE)'
    )

    parser.add_argument(
        '--progress',
        action='store_true',
        help='Show processing progress and exit'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(verbose=args.verbose)

    # Handle --save-config
    if args.save_config:
        config = BatchConfig()
        config.save_template(Path(args.save_config))
        return 0

    # Require directory for other operations
    if not args.directory:
        parser.print_help()
        return 1

    directory = Path(args.directory)
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return 1

    # Load configuration
    config_path = Path(args.config) if args.config else None
    config = BatchConfig(config_path)

    # Override with command line arguments
    config.config['input_directory'] = str(directory)

    if args.output_dir:
        config.config['output_directory'] = str(args.output_dir)

    if args.workers:
        config.config['workers'] = args.workers

    if args.move:
        config.config['move_files'] = True

    # Enable specific features if requested
    if args.features:
        # Disable all features first
        for feature in config.config['features']:
            config.config['features'][feature]['enabled'] = False
        # Enable requested features
        for feature in args.features:
            if feature in config.config['features']:
                config.config['features'][feature]['enabled'] = True

    # Set overwrite for specific features
    if args.overwrite:
        for feature in args.overwrite:
            if feature in config.config['features']:
                config.config['features'][feature]['overwrite'] = True

    # Handle --progress
    if args.progress:
        # Determine which features to check
        enabled_features = [
            f for f in FEATURE_MODULES.keys()
            if config.is_feature_enabled(f)
        ]

        # Collect all feature names
        all_features = []
        for feature in enabled_features:
            all_features.extend(FEATURE_MODULES[feature]['features'])

        work_dir = Path(config.get('output_directory') or config.get('input_directory'))
        stats = get_progress_stats(work_dir, all_features)

        print(f"\nProcessing Progress:")
        print(f"  Total folders:  {stats['total']}")
        print(f"  Complete:       {stats['complete']} ({stats['complete']/max(stats['total'],1)*100:.1f}%)")
        print(f"  Incomplete:     {stats['incomplete']}")
        print(f"  Locked:         {stats['locked']}")

        return 0

    # Run pipeline
    success = run_pipeline(config)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
