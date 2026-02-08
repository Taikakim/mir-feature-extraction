"""
Crops Processing Pipeline

Master orchestrator for processing audio crops:
    1. Check for existing stems (from create_training_crops.py)
    2. Run Demucs if stems are missing
    3. Extract all features
    4. Run Music Flamingo descriptions

Usage:
    # Process single crop folder
    python -m crops.pipeline /path/to/crops/TrackName/

    # Process all crop folders under a directory
    python -m crops.pipeline /path/to/crops/ --batch

    # Skip stem separation (assume stems exist or not needed)
    python -m crops.pipeline /path/to/crops/ --batch --skip-demucs

    # Skip heavy processing
    python -m crops.pipeline /path/to/crops/ --batch --skip-flamingo --skip-audiobox

    # Customize output format and model
    python -m crops.pipeline /path/to/crops/ --batch --format flac --flamingo-model Q8_0
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set ROCm environment before torch imports
from core.rocm_env import setup_rocm_env
setup_rocm_env()

from core.file_utils import find_crop_files, find_crop_folders, get_crop_stem_files, DEMUCS_STEMS
from core.common import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class CropsPipelineConfig:
    """Configuration for the crops processing pipeline."""
    input_dir: Path
    device: str = 'cuda'
    output_format: str = 'mp3'
    overwrite: bool = False

    # Skip flags
    skip_demucs: bool = True  # Default: assume stems from cropping
    skip_flamingo: bool = False
    skip_audiobox: bool = False
    skip_essentia: bool = False
    skip_timbral: bool = False

    # Music Flamingo model
    flamingo_model: str = 'Q8_0'


@dataclass
class PipelineStats:
    """Statistics for pipeline execution."""
    total_folders: int = 0
    total_crops: int = 0
    processed_crops: int = 0
    skipped_crops: int = 0
    failed_crops: int = 0
    stems_separated: int = 0
    errors: List[str] = field(default_factory=list)


class CropsPipeline:
    """
    Crops Processing Pipeline.

    Processes all crops in folder structure:
        /crops/TrackName/
        ├── TrackName_0.flac
        ├── TrackName_0_drums.flac  (from cropping or Demucs)
        ├── TrackName_0.INFO        (output)
        ├── TrackName_1.flac
        └── ...
    """

    def __init__(self, config: CropsPipelineConfig):
        self.config = config
        self.extractor = None  # Lazy init
        self.stats = PipelineStats()

    def _init_extractor(self):
        """Initialize feature extractor (loads models once)."""
        if self.extractor is not None:
            return

        logger.info("Initializing feature extractor...")
        from crops.feature_extractor import CropFeatureExtractor

        self.extractor = CropFeatureExtractor(
            skip_demucs=True,  # We handle Demucs separately
            skip_flamingo=self.config.skip_flamingo,
            skip_audiobox=self.config.skip_audiobox,
            skip_essentia=self.config.skip_essentia,
            skip_timbral=self.config.skip_timbral,
            flamingo_model=self.config.flamingo_model,
            device=self.config.device,
        )
        logger.info("Feature extractor initialized")

    def _check_stems_exist(self, crop_path: Path) -> bool:
        """Check if stems exist for a crop."""
        stems = get_crop_stem_files(crop_path)
        stem_count = len([k for k in stems if k != 'source'])
        return stem_count == len(DEMUCS_STEMS)

    def _separate_stems_if_needed(self, crop_path: Path) -> Dict[str, Path]:
        """Separate stems for a crop if they don't exist."""
        if self.config.skip_demucs:
            return get_crop_stem_files(crop_path)

        # Check if stems exist
        if self._check_stems_exist(crop_path):
            logger.debug(f"  Stems exist for {crop_path.name}")
            return get_crop_stem_files(crop_path)

        # Separate stems
        logger.info(f"  Separating stems for {crop_path.name}...")
        try:
            from crops.demucs_sep import separate_crop_stems

            stems = separate_crop_stems(
                crop_path,
                device=self.config.device,
                output_format=self.config.output_format,
                overwrite=self.config.overwrite,
            )
            self.stats.stems_separated += 1
            return stems
        except Exception as e:
            logger.warning(f"  Stem separation failed: {e}")
            return get_crop_stem_files(crop_path)

    def process_crop(self, crop_path: Path) -> bool:
        """
        Process a single crop file.

        Args:
            crop_path: Path to the crop audio file

        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Ensure stems exist
            stems = self._separate_stems_if_needed(crop_path)

            # 2. Extract features
            self.extractor.extract_features(
                crop_path,
                stems=stems,
                overwrite=self.config.overwrite,
            )

            return True

        except Exception as e:
            error_msg = f"{crop_path.name}: {str(e)}"
            self.stats.errors.append(error_msg)
            logger.error(f"  Failed to process {crop_path.name}: {e}")
            return False

    def process_folder(self, folder: Path) -> Dict[str, Any]:
        """
        Process all crops in a single folder.

        Args:
            folder: Folder containing crop files

        Returns:
            Statistics for this folder
        """
        folder = Path(folder)
        crop_files = find_crop_files(folder)

        if not crop_files:
            logger.warning(f"No crop files found in {folder}")
            return {'total': 0, 'success': 0, 'failed': 0}

        logger.info(f"Processing folder: {folder.name} ({len(crop_files)} crops)")

        folder_stats = {
            'total': len(crop_files),
            'success': 0,
            'failed': 0,
        }

        for i, crop_path in enumerate(crop_files, 1):
            logger.info(f"  [{i}/{len(crop_files)}] {crop_path.name}")

            if self.process_crop(crop_path):
                folder_stats['success'] += 1
                self.stats.processed_crops += 1
            else:
                folder_stats['failed'] += 1
                self.stats.failed_crops += 1

        return folder_stats

    def run(self, batch: bool = False) -> PipelineStats:
        """
        Run the pipeline.

        Args:
            batch: If True, process all crop folders under input_dir
                   If False, process input_dir as a single crop folder

        Returns:
            Pipeline statistics
        """
        start_time = time.time()
        input_dir = Path(self.config.input_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Initialize feature extractor (loads models)
        self._init_extractor()

        if batch:
            # Find all crop folders
            folders = find_crop_folders(input_dir)
            self.stats.total_folders = len(folders)

            if not folders:
                logger.warning(f"No crop folders found in {input_dir}")
                return self.stats

            logger.info(f"Found {len(folders)} crop folders to process")

            # Count total crops
            for folder in folders:
                self.stats.total_crops += len(find_crop_files(folder))

            logger.info(f"Total crops to process: {self.stats.total_crops}")
            logger.info("=" * 60)

            # Process each folder
            for i, folder in enumerate(folders, 1):
                logger.info(f"\n[{i}/{len(folders)}] {folder.name}")
                self.process_folder(folder)
        else:
            # Process single folder
            self.stats.total_folders = 1
            crop_files = find_crop_files(input_dir)
            self.stats.total_crops = len(crop_files)
            self.process_folder(input_dir)

        # Print summary
        elapsed = time.time() - start_time
        self._print_summary(elapsed)

        return self.stats

    def _print_summary(self, elapsed: float):
        """Print pipeline summary."""
        logger.info("\n" + "=" * 60)
        logger.info("CROPS PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total folders:      {self.stats.total_folders}")
        logger.info(f"Total crops:        {self.stats.total_crops}")
        logger.info(f"Processed:          {self.stats.processed_crops}")
        logger.info(f"Failed:             {self.stats.failed_crops}")
        logger.info(f"Stems separated:    {self.stats.stems_separated}")
        logger.info(f"Time:               {elapsed:.1f}s ({elapsed/60:.1f} min)")

        if self.stats.total_crops > 0:
            rate = elapsed / self.stats.total_crops
            logger.info(f"Rate:               {rate:.2f}s per crop")

        if self.stats.errors:
            logger.info(f"\nErrors ({len(self.stats.errors)}):")
            for error in self.stats.errors[:10]:  # Show first 10
                logger.warning(f"  - {error}")
            if len(self.stats.errors) > 10:
                logger.warning(f"  ... and {len(self.stats.errors) - 10} more")

        logger.info("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Process audio crops: separate stems + extract features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single crop folder
  python -m crops.pipeline /path/to/crops/TrackName/

  # Process all crop folders
  python -m crops.pipeline /path/to/crops/ --batch

  # Skip heavy processing
  python -m crops.pipeline /path/to/crops/ --batch --skip-flamingo

  # Force Demucs even if stems exist
  python -m crops.pipeline /path/to/crops/ --batch --no-skip-demucs --overwrite
        """
    )

    parser.add_argument(
        'path',
        type=str,
        help='Crops folder or root directory'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all crop folders in directory'
    )

    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for GPU processing (default: cuda)'
    )

    parser.add_argument(
        '--format', '-f',
        default='mp3',
        choices=['mp3', 'flac', 'wav'],
        help='Output format for stems (default: mp3 VBR 96kbps)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing features and stems'
    )

    # Skip flags
    parser.add_argument(
        '--skip-demucs',
        action='store_true',
        default=True,
        help='Skip stem separation (default: True, assumes stems from cropping)'
    )

    parser.add_argument(
        '--no-skip-demucs',
        action='store_true',
        help='Force Demucs separation for crops without stems'
    )

    parser.add_argument(
        '--skip-flamingo',
        action='store_true',
        help='Skip Music Flamingo descriptions'
    )

    parser.add_argument(
        '--skip-audiobox',
        action='store_true',
        help='Skip AudioBox aesthetics'
    )

    parser.add_argument(
        '--skip-essentia',
        action='store_true',
        help='Skip Essentia features'
    )

    parser.add_argument(
        '--skip-timbral',
        action='store_true',
        help='Skip Audio Commons timbral features'
    )

    parser.add_argument(
        '--flamingo-model',
        default='Q8_0',
        choices=['IQ3_M', 'Q6_K', 'Q8_0'],
        help='GGUF model for Music Flamingo (default: Q8_0)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    # Handle skip_demucs logic
    skip_demucs = True  # Default
    if args.no_skip_demucs:
        skip_demucs = False

    # Create config
    config = CropsPipelineConfig(
        input_dir=Path(args.path),
        device=args.device,
        output_format=args.format,
        overwrite=args.overwrite,
        skip_demucs=skip_demucs,
        skip_flamingo=args.skip_flamingo,
        skip_audiobox=args.skip_audiobox,
        skip_essentia=args.skip_essentia,
        skip_timbral=args.skip_timbral,
        flamingo_model=args.flamingo_model,
    )

    logger.info("Crops Processing Pipeline")
    logger.info(f"Input: {config.input_dir}")
    logger.info(f"Mode: {'Batch' if args.batch else 'Single folder'}")
    logger.info(f"Skip Demucs: {config.skip_demucs}")
    logger.info(f"Skip Flamingo: {config.skip_flamingo}")
    logger.info(f"Flamingo Model: {config.flamingo_model}")
    logger.info("=" * 60)

    try:
        pipeline = CropsPipeline(config)
        stats = pipeline.run(batch=args.batch)

        # Exit with error code if any crops failed
        if stats.failed_crops > 0:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
