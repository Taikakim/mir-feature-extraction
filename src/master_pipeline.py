"""
Master Pipeline for MIR Feature Extraction

Complete workflow from raw audio to analyzed crops:

1. ORGANIZE: Raw audio files -> Project folder structure
2. ANALYZE FULL TRACKS: Demucs separation + beat maps + metadata lookup
3. CREATE CROPS: Cut segments + migrate features
4. ANALYZE CROPS: Per-crop features (with stems)

Usage:
    # Using config file (recommended)
    python src/master_pipeline.py --config config/master_pipeline.yaml

    # Override config with CLI args
    python src/master_pipeline.py --config config/my_project.yaml --overwrite

    # Full pipeline from scratch (no config)
    python src/master_pipeline.py /path/to/raw/audio --output /path/to/output

    # Start from already organized folder
    python src/master_pipeline.py /path/to/organized --skip-organize

Config file: config/master_pipeline.yaml contains all options with documentation.
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Set ROCm environment before torch imports
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'garbage_collection_threshold:0.8')
os.environ.setdefault('FLASH_ATTENTION_TRITON_AMD_ENABLE', 'TRUE')
os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_TUNING', '0')
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MIOPEN_FIND_MODE', '2')

sys.path.insert(0, str(Path(__file__).parent))

from core.common import setup_logging, ProgressBar
from core.file_utils import find_organized_folders, find_crop_folders, find_crop_files, get_stem_files
from core.json_handler import safe_update, read_info, get_info_path
from core.file_locks import FileLock

# Extracted modules (refactoring)
from core.terminal import (
    Colors, ColoredFormatter, setup_colored_logging,
    color, fmt_filename, fmt_header, fmt_stage, fmt_success,
    fmt_warning, fmt_error, fmt_progress, fmt_notification, fmt_dim
)
from core.metadata_utils import (
    MUTAGEN_AVAILABLE, VARIOUS_ARTISTS_ALIASES,
    extract_audio_metadata, build_folder_name_from_metadata
)
from core.pipeline_stats import TimingStats, PipelineStats
from core.pipeline_workers import (
    process_folder_features as _process_folder_features,
    process_folder_crops as _process_folder_crops,
    process_demucs_subprocess as _process_demucs_subprocess
)

logger = logging.getLogger(__name__)



@dataclass
class MasterPipelineConfig:
    """Configuration for the master pipeline."""
    input_dir: Path
    output_dir: Optional[Path] = None
    stems_source: Optional[Path] = None  # Pre-existing stems location

    # Device settings
    device: str = 'cuda'

    # Stage control
    skip_organize: bool = False
    skip_track_analysis: bool = False
    skip_crops: bool = False
    skip_crop_analysis: bool = False

    # Source Separation
    separation_backend: str = 'demucs'  # 'demucs' or 'bs_roformer'

    # BS-Roformer settings
    bs_roformer_model: str = 'jarredou-BS-ROFO-SW-Fixed-drums'
    bs_roformer_dir: Path = Path('/home/kim/Projects/mir/models/bs-roformer')
    bs_roformer_batch_size: int = 1
    bs_roformer_device: str = 'cuda'

    # Demucs settings
    demucs_model: str = 'htdemucs'  # htdemucs (fast), htdemucs_ft (4x slower, better)
    demucs_shifts: int = 0  # 0 for MIR analysis (faster), 1+ for hifi
    demucs_segment: Optional[int] = None  # Segment length (None = model default, htdemucs max ~7.8s)
    demucs_format: str = 'mp3'
    demucs_workers: int = 2  # Parallel workers (subprocess-based, each ~5GB VRAM)
    demucs_compile: bool = False  # Use torch.compile (faster after warmup)
    demucs_compile_mode: str = 'reduce-overhead'  # torch.compile mode for ROCm

    # Crop settings
    crop_length_samples: int = 2097152  # ~47.5s at 44.1kHz
    crop_mode: str = 'sequential'  # 'sequential' or 'beat-aligned'
    crop_overlap: bool = False
    crop_div4: bool = False
    crop_include_stems: bool = True
    crop_workers: int = 6  # Parallel workers for cropping

    # Rhythm analysis
    rhythm_workers: int = 4  # Parallel workers for beat/downbeat detection

    # Feature settings
    skip_loudness: bool = False
    skip_spectral: bool = False
    skip_chroma: bool = False
    skip_timbral: bool = False
    skip_essentia: bool = False
    skip_audiobox: bool = False
    skip_per_stem: bool = False

    # Music Flamingo
    skip_flamingo: bool = False
    flamingo_model: str = 'Q8_0'
    flamingo_token_limits: Dict[str, int] = field(default_factory=dict)

    # Metadata
    skip_metadata: bool = False
    use_fingerprinting: bool = True
    various_artists_aliases: Set[str] = field(default_factory=lambda: VARIOUS_ARTISTS_ALIASES.copy())
    fix_various_artists: bool = True  # Fix "Various Artists" names during organization

    # Filename cleanup (T5 tokenizer compatibility)
    cleanup_filenames: bool = True

    # Processing
    overwrite: bool = False  # Global overwrite flag
    per_feature_overwrite: Dict[str, bool] = field(default_factory=dict)  # Per-feature overwrite flags
    dry_run: bool = False
    verbose: bool = False
    feature_workers: int = 8  # Parallel workers for feature extraction
    essentia_workers: int = 4  # Separate limit for Essentia (TensorFlow deadlocks with high parallelism)
    batch_feature_extraction: bool = True  # Batch process features (more persistent GPU usage)
    flamingo_prompts: Dict[str, bool] = field(default_factory=dict)  # Enabled prompts

    def should_overwrite(self, feature: str) -> bool:
        """Check if a specific feature should be overwritten.

        Returns True if either:
        - Global overwrite is True, OR
        - Per-feature overwrite for this feature is True

        Args:
            feature: Feature name (e.g., 'loudness', 'demucs', 'beats', 'crops')

        Returns:
            True if the feature should be overwritten
        """
        return self.overwrite or self.per_feature_overwrite.get(feature, False)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'MasterPipelineConfig':
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for config files: pip install pyyaml")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Map YAML structure to flat config
        paths = data.get('paths', {})
        stages = data.get('stages', {})
        
        # Support new and legacy config structure
        source_separation = data.get('source_separation', {})
        bs_roformer = source_separation.get('bs_roformer', {})
        demucs_cfg = source_separation.get('demucs', {})
        demucs_legacy = data.get('demucs', {}) # Fallback
        
        rhythm = data.get('rhythm', {})
        cropping = data.get('cropping', {})
        features = data.get('features', {})
        flamingo = data.get('music_flamingo', {})
        metadata = data.get('metadata', {})
        processing = data.get('processing', {})

        # Build config
        config = cls(
            input_dir=Path(paths['input']) if paths.get('input') else None,
            output_dir=Path(paths['output']) if paths.get('output') else None,
            stems_source=Path(paths['stems_source']) if paths.get('stems_source') else None,

            # Stages (inverted - config has enabled flags, we have skip flags)
            skip_organize=not stages.get('organize', True),
            skip_track_analysis=not stages.get('track_analysis', True),
            skip_crops=not stages.get('cropping', True),
            skip_crop_analysis=not stages.get('crop_analysis', True),

            # Demucs
            # Source Separation
            separation_backend=source_separation.get('backend', 'demucs'),
            
            # BS-Roformer
            bs_roformer_model=bs_roformer.get('model_name', 'jarredou-BS-ROFO-SW-Fixed-drums'),
            bs_roformer_dir=Path(bs_roformer.get('model_dir', '/home/kim/Projects/mir/models/bs-roformer')),
            bs_roformer_batch_size=bs_roformer.get('batch_size', 1),
            bs_roformer_device=bs_roformer.get('device', 'cuda'),

            # Demucs (support both new 'demucs' sub-block and legacy top-level 'demucs')
            demucs_model=demucs_cfg.get('model', demucs_legacy.get('model', 'htdemucs')),
            demucs_shifts=demucs_cfg.get('shifts', demucs_legacy.get('shifts', 0)),
            demucs_segment=demucs_cfg.get('segment', demucs_legacy.get('segment', 45)),
            demucs_format=demucs_cfg.get('output_format', demucs_legacy.get('output_format', 'mp3')),
            demucs_workers=demucs_cfg.get('workers', demucs_legacy.get('workers', 2)),
            demucs_compile=demucs_cfg.get('compile', demucs_legacy.get('compile', False)),
            demucs_compile_mode=demucs_cfg.get('compile_mode', demucs_legacy.get('compile_mode', 'reduce-overhead')),
            device=demucs_cfg.get('device', demucs_legacy.get('device', processing.get('device', 'cuda'))),

            # Cropping
            crop_length_samples=cropping.get('length_samples', 2097152),
            crop_mode=cropping.get('mode', 'sequential'),
            crop_overlap=cropping.get('overlap', False),
            crop_div4=cropping.get('div4', False),
            crop_include_stems=cropping.get('include_stems', True),
            crop_workers=cropping.get('workers', 6),

            # Rhythm
            rhythm_workers=rhythm.get('workers', 4),

            # Features
            skip_loudness=not features.get('loudness', True),
            skip_spectral=not features.get('spectral', True),
            skip_chroma=not features.get('chroma', True),
            skip_timbral=not features.get('timbral', True),
            skip_essentia=not features.get('essentia', True),
            skip_audiobox=not features.get('audiobox', True),
            skip_per_stem=not features.get('per_stem', True),

            # Music Flamingo
            skip_flamingo=not flamingo.get('enabled', True),
            flamingo_model=flamingo.get('model', 'Q8_0'),
            flamingo_token_limits=flamingo.get('max_tokens', {}),

            # Metadata
            skip_metadata=not metadata.get('enabled', True),
            use_fingerprinting=metadata.get('use_fingerprinting', True),
            fix_various_artists=metadata.get('fix_various_artists', True),
            various_artists_aliases=set(metadata.get('various_artists_aliases', list(VARIOUS_ARTISTS_ALIASES))),

            # Filename cleanup
            cleanup_filenames=data.get('filename_cleanup', {}).get('enabled', True),

            # Processing
            overwrite=processing.get('overwrite', False),
            per_feature_overwrite=data.get('overwrite', {}),  # Per-feature overwrite from 'overwrite' section
            verbose=processing.get('verbose', False),
            feature_workers=processing.get('feature_workers', 8),
            essentia_workers=processing.get('essentia_workers', 4),  # TensorFlow-safe limit
            batch_feature_extraction=processing.get('batch_feature_extraction', True),
            flamingo_prompts=flamingo.get('prompts', {}),
        )

        return config

    def to_yaml(self, yaml_path: Path):
        """Save configuration to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required: pip install pyyaml")

        data = {
            'paths': {
                'input': str(self.input_dir) if self.input_dir else None,
                'output': str(self.output_dir) if self.output_dir else None,
                'stems_source': str(self.stems_source) if self.stems_source else None,
            },
            'stages': {
                'organize': not self.skip_organize,
                'track_analysis': not self.skip_track_analysis,
                'cropping': not self.skip_crops,
                'crop_analysis': not self.skip_crop_analysis,
            },
            'demucs': {
                'model': self.demucs_model,
                'shifts': self.demucs_shifts,
                'segment': self.demucs_segment,
                'output_format': self.demucs_format,
                'workers': self.demucs_workers,
                'device': self.device,
                'compile': self.demucs_compile,
                'compile_mode': self.demucs_compile_mode,
            },
            'rhythm': {
                'workers': self.rhythm_workers,
            },
            'cropping': {
                'length_samples': self.crop_length_samples,
                'mode': self.crop_mode,
                'overlap': self.crop_overlap,
                'div4': self.crop_div4,
                'include_stems': self.crop_include_stems,
                'workers': self.crop_workers,
            },
            'features': {
                'loudness': not self.skip_loudness,
                'spectral': not self.skip_spectral,
                'chroma': not self.skip_chroma,
                'timbral': not self.skip_timbral,
                'essentia': not self.skip_essentia,
                'audiobox': not self.skip_audiobox,
                'per_stem': not self.skip_per_stem,
            },
            'music_flamingo': {
                'enabled': not self.skip_flamingo,
                'model': self.flamingo_model,
                'max_tokens': self.flamingo_token_limits,
            },
            'metadata': {
                'enabled': not self.skip_metadata,
                'use_fingerprinting': self.use_fingerprinting,
                'fix_various_artists': self.fix_various_artists,
                'various_artists_aliases': list(self.various_artists_aliases),
            },
            'processing': {
                'device': self.device,
                'overwrite': self.overwrite,
                'verbose': self.verbose,
                'feature_workers': self.feature_workers,
                'essentia_workers': self.essentia_workers,
                'batch_feature_extraction': self.batch_feature_extraction,
            },
            'overwrite': self.per_feature_overwrite,  # Per-feature overwrite settings
            'music_flamingo': {
                'enabled': not self.skip_flamingo,
                'model': self.flamingo_model,
                'max_tokens': self.flamingo_token_limits,
                'prompts': self.flamingo_prompts,
            },
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


class MasterPipeline:
    """
    Master pipeline orchestrating the complete MIR workflow.
    """

    def __init__(self, config: MasterPipelineConfig):
        self.config = config
        self.stats = PipelineStats()

        # Determine working directory
        if config.output_dir:
            self.working_dir = config.output_dir
        else:
            self.working_dir = config.input_dir

    def _calculate_total_audio_duration(self, folders: List[Path]) -> float:
        """Calculate total audio duration for all tracks.

        Args:
            folders: List of organized folder paths

        Returns:
            Total duration in seconds
        """
        import soundfile as sf

        total_duration = 0.0

        for folder in folders:
            full_mix_files = list(folder.glob('full_mix.*'))
            if not full_mix_files:
                continue

            try:
                info = sf.info(str(full_mix_files[0]))
                total_duration += info.duration
            except Exception as e:
                logger.debug(f"Could not get duration for {folder.name}: {e}")

        return total_duration

    def run(self) -> PipelineStats:
        """Run the complete pipeline."""
        logger.info(fmt_header("=" * 70))
        logger.info(fmt_header("MIR MASTER PIPELINE"))
        logger.info(fmt_header("=" * 70))
        logger.info(f"Input:  {fmt_filename(str(self.config.input_dir))}")
        logger.info(f"Output: {fmt_filename(str(self.working_dir))}")
        if self.config.stems_source:
            logger.info(f"Stems:  {fmt_filename(str(self.config.stems_source))}")
        logger.info(fmt_header("=" * 70))

        # Clean up any stale lock files from previous interrupted runs
        from core.file_locks import remove_all_locks
        locks_removed = remove_all_locks(self.working_dir)
        if locks_removed > 0:
            logger.info(f"Cleaned up {locks_removed} stale lock files")

        total_start = time.time()

        # Smart skip: Check if crops already exist for all tracks
        # If so, skip stages 1-3 and go directly to crop analysis
        crops_dir = self.working_dir.parent / f"{self.working_dir.name}_crops"
        skip_to_crop_analysis = False

        # Pre-calculate total audio duration for speed metrics
        source_folders = list(find_organized_folders(self.working_dir))
        if source_folders:
            logger.info(f"Calculating total audio duration for {len(source_folders)} tracks...")
            self.stats.total_audio_duration = self._calculate_total_audio_duration(source_folders)
            duration_mins = self.stats.total_audio_duration / 60
            logger.info(f"Total audio: {fmt_notification(f'{duration_mins:.1f} minutes')} ({self.stats.total_audio_duration:.0f}s)")

        if crops_dir.exists() and not self.config.should_overwrite('crops'):
            # Count tracks and existing crops
            if source_folders:
                tracks_with_crops = sum(
                    1 for f in source_folders
                    if self._check_existing_crops(f, crops_dir)
                )
                if tracks_with_crops == len(source_folders):
                    logger.info(fmt_success(f"\nAll {tracks_with_crops} tracks already have crops in {crops_dir.name}/"))
                    logger.info(fmt_success("Skipping Stages 1-3, proceeding to Crop Analysis"))
                    skip_to_crop_analysis = True
                    self.crops_dir = crops_dir
                elif tracks_with_crops > 0:
                    logger.info(f"\nFound existing crops for {tracks_with_crops}/{len(source_folders)} tracks")

        # Stage 1: Organization
        if skip_to_crop_analysis:
            logger.info(fmt_dim("\n[STAGE 1] Organization: SKIPPED (crops exist)"))
        elif not self.config.skip_organize:
            self._run_organization()
        else:
            logger.info(fmt_dim("\n[STAGE 1] Organization: SKIPPED"))

        # Stage 1c: Extract audio file metadata (MP3 tags, etc.)
        if not skip_to_crop_analysis:
            self._run_metadata_extraction()

        # Stage 2: Full Track Analysis
        if skip_to_crop_analysis:
            logger.info(fmt_dim("\n[STAGE 2] Track Analysis: SKIPPED (crops exist)"))
        elif not self.config.skip_track_analysis:
            self._run_track_analysis()
        else:
            logger.info(fmt_dim("\n[STAGE 2] Track Analysis: SKIPPED"))

        # Stage 3: Create Crops
        if skip_to_crop_analysis:
            logger.info(fmt_dim("\n[STAGE 3] Cropping: SKIPPED (crops exist)"))
        elif not self.config.skip_crops:
            self._run_cropping()
        else:
            logger.info(fmt_dim("\n[STAGE 3] Cropping: SKIPPED"))

        # Stage 4: Crop Analysis
        if not self.config.skip_crop_analysis:
            self._run_crop_analysis()
        else:
            logger.info(fmt_dim("\n[STAGE 4] Crop Analysis: SKIPPED"))

        # Finalize timing
        self.stats.run_end_time = time.time()
        total_time = self.stats.total_time

        # Print summary to console
        self._print_summary(total_time)
        
        # Write statistics to log file
        self._write_stats_log()

        return self.stats

    def _run_organization(self):
        """Stage 1: Organize raw files into project structure."""
        logger.info("\n" + fmt_header("=" * 70))
        logger.info(fmt_stage("[STAGE 1] ORGANIZATION"))
        logger.info(fmt_header("=" * 70))

        start_time = time.time()
        self.stats.start_operation('organize')

        from preprocessing.file_organizer import organize_directory

        stats = organize_directory(
            self.config.input_dir,
            output_dir=self.config.output_dir,
            move=False,  # Always copy, never move
            dry_run=self.config.dry_run,
            recursive=True,
            fix_various=self.config.fix_various_artists,
        )

        self.stats.files_found = stats['total']
        self.stats.files_organized = stats['organized']
        
        organize_time = self.stats.end_operation('organize',
            items_processed=stats['organized'],
            items_skipped=stats['total'] - stats['organized'])

        # Cleanup filenames for T5 tokenizer compatibility
        if self.config.cleanup_filenames and not self.config.dry_run:
            self.stats.start_operation('filename_cleanup')
            self._run_filename_cleanup()
            self.stats.end_operation('filename_cleanup')

        self.stats.time_organize = time.time() - start_time

        logger.info(f"Organization complete: {stats['organized']}/{stats['total']} files in {organize_time:.1f}s")

    def _run_filename_cleanup(self):
        """Clean up filenames for T5 tokenizer compatibility."""
        logger.info("\n[1b] Filename Cleanup (T5 compatibility)")

        try:
            from preprocessing.filename_cleanup import find_items_to_rename, apply_rename

            items = find_items_to_rename(self.working_dir)

            if not items:
                logger.info("All filenames are already clean")
                return

            logger.info(f"Cleaning {len(items)} filenames...")

            # Sort: files first, then directories
            items.sort(key=lambda x: (x['type'] == 'directory', str(x['old_path'])))

            cleaned_count = 0
            for item in items:
                if apply_rename(item):
                    cleaned_count += 1
                    logger.debug(f"Renamed: {item['old_name']} → {item['new_name']}")
            
            # Update timing stats with actual count
            if 'filename_cleanup' in self.stats.operation_timing:
                self.stats.operation_timing['filename_cleanup'].items_processed = cleaned_count

        except Exception as e:
            logger.warning(f"Filename cleanup failed: {e}")
            self.stats.errors.append(f"Filename cleanup: {e}")

    def _run_metadata_extraction(self):
        """Extract metadata from audio files (MP3 ID3 tags, etc.) and optionally rename."""
        logger.info("\n[1c] Audio File Metadata Extraction")
        
        self.stats.start_operation('metadata_extraction')

        if not MUTAGEN_AVAILABLE:
            logger.warning("mutagen not installed - skipping metadata extraction (pip install mutagen)")
            self.stats.end_operation('metadata_extraction')
            return

        folders = find_organized_folders(self.working_dir)

        if not folders:
            logger.info("No folders to process")
            self.stats.end_operation('metadata_extraction')
            return

        extracted_count = 0
        renamed_count = 0

        for folder in folders:
            stems = get_stem_files(folder, include_full_mix=True)
            if 'full_mix' not in stems:
                continue

            full_mix = stems['full_mix']
            info_path = get_info_path(full_mix)

            # Check if we already have metadata
            existing = read_info(info_path) if info_path.exists() else {}
            if 'track_metadata_artist' in existing and not self.config.should_overwrite('metadata'):
                continue

            # Extract metadata
            metadata = extract_audio_metadata(full_mix)

            if not metadata:
                continue

            extracted_count += 1

            # Save metadata to INFO
            safe_update(info_path, metadata)
            logger.debug(f"{folder.name}: Extracted {list(metadata.keys())}")

            # Optionally rename folder based on metadata
            if self.config.fix_various_artists:
                # Check if current name is a "Various Artists" type
                current_artist = folder.name.split(' - ')[0].strip().lower() if ' - ' in folder.name else ''

                if current_artist in VARIOUS_ARTISTS_ALIASES:
                    new_name = build_folder_name_from_metadata(metadata, folder.name)

                    if new_name and new_name != folder.name:
                        new_folder = folder.parent / new_name

                        if not new_folder.exists():
                            try:
                                # Rename folder
                                folder.rename(new_folder)
                                renamed_count += 1
                                logger.info(f"Renamed: {folder.name} -> {new_name}")
                            except Exception as e:
                                logger.warning(f"Failed to rename {folder.name}: {e}")

        extract_time = self.stats.end_operation('metadata_extraction', 
            items_processed=extracted_count)
        logger.info(f"Metadata extraction: {extracted_count} files processed, {renamed_count} renamed in {extract_time:.1f}s")

    def _run_track_analysis(self):
        """Stage 2: Analyze full tracks (Demucs, beats, metadata)."""
        logger.info("\n" + fmt_header("=" * 70))
        logger.info(fmt_stage("[STAGE 2] FULL TRACK ANALYSIS"))
        logger.info(fmt_header("=" * 70))

        start_time = time.time()

        folders = find_organized_folders(self.working_dir)
        self.stats.tracks_total = len(folders)

        if not folders:
            logger.warning("No organized folders found")
            return

        logger.info(f"Found {len(folders)} tracks to analyze")

        # Sub-stage 2a: Stem Separation
        logger.info("\n[2a] Stem Separation")
        self.stats.start_operation('demucs')
        
        if self.config.separation_backend == 'bs_roformer':
            logger.info(f"Using backend: {fmt_notification('BS-RoFormer')}")
            self._run_bs_roformer_batch(folders)
        else:
            logger.info(f"Using backend: {fmt_notification('Demucs')}")
            self._run_demucs_batch(folders)
            
        sep_time = self.stats.end_operation('demucs', 
            items_processed=self.stats.tracks_separated,
            items_skipped=len(folders) - self.stats.tracks_separated)
        logger.info(fmt_dim(f"    Separation completed in {sep_time:.1f}s"))

        # Sub-stage 2b: Beat/onset/downbeat detection
        logger.info("\n[2b] Rhythm Analysis (beats, onsets, downbeats)")
        self.stats.start_operation('rhythm')
        self._run_rhythm_analysis(folders)
        rhythm_time = self.stats.end_operation('rhythm', items_processed=len(folders))
        logger.info(fmt_dim(f"    Rhythm analysis completed in {rhythm_time:.1f}s"))

        # Sub-stage 2c: Metadata lookup (with fingerprinting for Various Artists)
        logger.info("\n[2c] Metadata Lookup")
        self.stats.start_operation('metadata')
        self._run_metadata_lookup(folders)
        metadata_time = self.stats.end_operation('metadata', 
            items_processed=self.stats.tracks_metadata_found)
        logger.info(fmt_dim(f"    Metadata lookup completed in {metadata_time:.1f}s"))

        # Sub-stage 2d: First-stage features (migrated to crops)
        logger.info("\n[2d] First-Stage Features")
        self.stats.start_operation('first_stage_features')
        self._run_first_stage_features(folders)
        features_time = self.stats.end_operation('first_stage_features', 
            items_processed=len(folders))
        logger.info(fmt_dim(f"    First-stage features completed in {features_time:.1f}s"))

        self.stats.time_track_analysis = time.time() - start_time

    def _run_bs_roformer_batch(self, folders: List[Path]):
        """Run BS-RoFormer separation on all tracks."""
        from preprocessing.bs_roformer_sep import separate_organized_folder, load_bs_roformer

        # Check if using pre-existing stems
        if self.config.stems_source:
            logger.info(f"Using pre-existing stems from: {fmt_filename(str(self.config.stems_source))}")
            self._link_existing_stems(folders)
            return

        model_name = self.config.bs_roformer_model
        model_dir = self.config.bs_roformer_dir
        batch_size = self.config.bs_roformer_batch_size
        device = self.config.bs_roformer_device

        logger.info(f"Separating stems (model={fmt_notification(model_name)}, "
                    f"batch={batch_size}, device={device})")

        # Find tracks to process
        tracks_to_process = []
        skipped = 0
        stem_names = ['drums', 'bass', 'other', 'vocals']

        for folder in folders:
            # Check for existing stems
            if not self.config.should_overwrite('source_separation') and \
               not self.config.should_overwrite('demucs'): # Backwards compat
                
                # Check for stems in any format (WAV, FLAC, MP3)
                existing = []
                for f in stem_names:
                    for ext in ['.wav', '.flac', '.mp3']:
                        if (folder / f"{f}{ext}").exists():
                            existing.append(f)
                            break
                            
                if len(existing) == 4:
                    skipped += 1
                    continue
            
            tracks_to_process.append(folder)

        if skipped > 0:
            logger.info(fmt_dim(f"Skipping {skipped} tracks (already have stems)"))

        if not tracks_to_process:
            logger.info("No tracks need stem separation")
            return

        logger.info(f"Processing {fmt_notification(str(len(tracks_to_process)))} tracks...")

        # Pre-load BS-Roformer model (Optimization)
        separator = None
        try:
            from preprocessing.bs_roformer_sep import load_bs_roformer
            logger.info("Pre-loading BS-RoFormer model for batch processing...")
            # Note: initialization returns (model, model_cfg, audio_cfg, inf_cfg) which we pass as 'separator'
            # The updated bs_roformer_sep expects the model object itself which has configs attached
            separator_tuple = load_bs_roformer(model_name, model_dir, device=device)
            separator = separator_tuple[0] # The model instance
            logger.info("✓ Model pre-loaded successfully")
        except Exception as e:
            logger.error(f"Failed to pre-load BS-RoFormer: {e}")
            logger.warning("Will fall back to per-folder loading (slower)")

        success_count = 0
        fail_count = 0
        
        # Sequential processing (GPU bound, not parallel for now)
        for i, folder in enumerate(tracks_to_process, 1):
            logger.info(f"[{i}/{len(tracks_to_process)}] Processing: {folder.name}")
            try:
                separate_organized_folder(
                    folder,
                    model_name=model_name,
                    model_dir=model_dir,
                    batch_size=batch_size,
                    overwrite=True,  # We already filtered above
                    device=device,
                    separator=separator
                )
                success_count += 1
                logger.info(fmt_success(f"  Success: {folder.name}"))
            except Exception as e:
                fail_count += 1
                logger.error(f"  Failed: {folder.name} - {e}")
                
        self.stats.tracks_separated = success_count


    def _run_demucs_batch(self, folders: List[Path]):
        """Run Demucs separation on all tracks (parallel subprocess-based)."""
        # Check if we should use existing stems
        if self.config.stems_source:
            logger.info(f"Using pre-existing stems from: {fmt_filename(str(self.config.stems_source))}")
            self._link_existing_stems(folders)
            return

        num_workers = self.config.demucs_workers
        logger.info(f"Separating stems (model={fmt_notification(self.config.demucs_model)}, "
                    f"shifts={self.config.demucs_shifts}, workers={fmt_notification(str(num_workers))})")

        # Find all tracks that need processing
        tracks_to_process = []
        skipped = 0

        for folder in folders:
            full_mix_files = list(folder.glob('full_mix.*'))
            if not full_mix_files:
                continue

            full_mix = full_mix_files[0]

            # Check if already done - stems can be in folder directly OR in htdemucs/full_mix/
            # Check all formats (wav, flac, mp3) regardless of configured output format
            # This allows skipping if stems were created by a different backend/config
            if not self.config.should_overwrite('source_separation') and \
               not self.config.should_overwrite('demucs'):
                stem_names = ['drums', 'bass', 'other', 'vocals']

                # Check direct folder first (preferred location) - any format
                existing_direct = []
                for f in stem_names:
                    for ext in ['.wav', '.flac', '.mp3']:
                        if (folder / f"{f}{ext}").exists():
                            existing_direct.append(f)
                            break

                # Also check htdemucs output folder (Demucs default output)
                htdemucs_folder = folder / "htdemucs" / "full_mix"
                existing_htdemucs = []
                for f in stem_names:
                    for ext in ['.wav', '.flac', '.mp3']:
                        if (htdemucs_folder / f"{f}{ext}").exists():
                            existing_htdemucs.append(f)
                            break

                if len(existing_direct) == 4 or len(existing_htdemucs) == 4:
                    skipped += 1
                    continue

            tracks_to_process.append(full_mix)

        if skipped > 0:
            logger.info(fmt_dim(f"Skipping {skipped} tracks (already have stems)"))

        if not tracks_to_process:
            logger.info("No tracks need stem separation")
            return

        logger.info(f"Processing {fmt_notification(str(len(tracks_to_process)))} tracks with "
                    f"{fmt_notification(str(num_workers))} parallel workers...")

        # Prepare arguments - output to parent folder (where full_mix is)
        bitrate = 128  # Default MP3 bitrate
        args_list = [
            (track, track.parent, self.config.demucs_model, self.config.demucs_shifts,
             self.config.demucs_format, bitrate)
            for track in tracks_to_process
        ]

        success_count = 0
        fail_count = 0
        progress = ProgressBar(len(tracks_to_process), desc="Demucs")

        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_track = {
                    executor.submit(_process_demucs_subprocess, args): args[0]
                    for args in args_list
                }

                for i, future in enumerate(as_completed(future_to_track), 1):
                    track = future_to_track[future]
                    try:
                        folder_name, success, elapsed, message = future.result()
                        if success:
                            success_count += 1
                            logger.debug(f"{fmt_filename(folder_name)}: {fmt_success(message)}")
                        else:
                            fail_count += 1
                            logger.warning(f"{fmt_filename(folder_name)}: {fmt_error(message)}")
                    except Exception as e:
                        fail_count += 1
                        logger.error(f"{track.parent.name}: {e}")

                    # Progress bar update
                    logger.info(progress.update(i, f"{success_count} ok, {fail_count} failed"))

            self.stats.tracks_separated = success_count
            logger.info(progress.finish(f"{success_count} success, {fail_count} failed"))

        except Exception as e:
            logger.error(f"Demucs batch failed: {e}")
            self.stats.errors.append(f"Demucs: {e}")

    def _link_existing_stems(self, folders: List[Path]):
        """Link or copy stems from pre-existing source."""
        from core.file_utils import get_stem_files, DEMUCS_STEMS

        for folder in folders:
            folder_name = folder.name
            source_folder = self.config.stems_source / folder_name

            if not source_folder.exists():
                logger.debug(f"No source stems for: {folder_name}")
                continue

            source_stems = get_stem_files(source_folder)

            for stem_name in DEMUCS_STEMS:
                if stem_name not in source_stems:
                    continue

                source_path = source_stems[stem_name]
                target_path = folder / source_path.name

                if target_path.exists() and not self.config.should_overwrite('demucs'):
                    continue

                try:
                    import shutil
                    shutil.copy2(source_path, target_path)
                    logger.debug(f"Copied stem: {source_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to copy stem {source_path.name}: {e}")

            self.stats.tracks_separated += 1

    def _run_rhythm_analysis(self, folders: List[Path]):
        """Run beat grid, onset, and downbeat detection."""
        from rhythm.beat_grid import batch_create_beat_grids

        try:
            stats = batch_create_beat_grids(
                self.working_dir,
                overwrite=self.config.should_overwrite('beats'),
                workers=self.config.rhythm_workers,
            )
            logger.info(f"Rhythm analysis: {stats.get('success', 0)}/{len(folders)}")
        except Exception as e:
            logger.error(f"Rhythm analysis failed: {e}")
            self.stats.errors.append(f"Rhythm: {e}")

    def _run_metadata_lookup(self, folders: List[Path]):
        """
        Look up metadata with priority:
        1. ID3 tags (already extracted to .INFO by _run_metadata_extraction)
        2. Folder name artist/track info
        3. Fingerprinting (only as last resort, if use_fingerprinting is True)
        """
        from core.file_utils import get_stem_files

        # Check which tracks need metadata lookup
        tracks_needing_lookup = []

        for folder in folders:
            folder_name = folder.name
            stems = get_stem_files(folder, include_full_mix=True)
            if 'full_mix' not in stems:
                continue

            info_path = get_info_path(stems['full_mix'])
            existing_info = read_info(info_path) if info_path.exists() else {}

            # Priority 1: Check if we have valid artist from ID3 tags
            id3_artist = existing_info.get('track_metadata_artist', '').strip()
            id3_title = existing_info.get('track_metadata_title', '').strip()

            # Check if ID3 artist is valid (not empty or a "Various Artists" alias)
            has_valid_id3_artist = (
                id3_artist and
                id3_artist.lower() not in VARIOUS_ARTISTS_ALIASES
            )

            # Priority 2: Check folder name for artist/track
            folder_artist = self._extract_artist_from_name(folder_name)
            has_valid_folder_artist = (
                folder_artist and
                folder_artist.lower() not in VARIOUS_ARTISTS_ALIASES
            )

            # Determine the best artist and track to use for lookup
            if has_valid_id3_artist and id3_title:
                # ID3 tags have valid info - use them for lookup (skip fingerprinting)
                best_artist = id3_artist
                best_track = id3_title
                source = 'id3'
            elif has_valid_folder_artist:
                # Folder name has valid artist - use it (skip fingerprinting)
                _, _, track_name = self._parse_folder_name(folder_name)
                best_artist = folder_artist
                best_track = track_name
                source = 'folder'
            else:
                # Neither ID3 nor folder name have valid artist - may need fingerprinting
                _, _, track_name = self._parse_folder_name(folder_name)
                best_artist = None
                best_track = track_name or id3_title
                source = 'unknown'

            # Check if we already have release_year (main metadata indicator)
            if 'release_year' in existing_info:
                continue  # Already has full metadata

            # Add to list with context
            tracks_needing_lookup.append({
                'folder': folder,
                'artist': best_artist,
                'track': best_track,
                'source': source,
                'has_valid_id3': has_valid_id3_artist,
                'has_valid_folder': has_valid_folder_artist,
            })

        if not tracks_needing_lookup:
            logger.info("All tracks have metadata")
            return

        logger.info(f"Looking up metadata for {len(tracks_needing_lookup)} tracks...")

        # Try to use track_metadata_lookup
        try:
            from tools.track_metadata_lookup import init_spotify, lookup_track, search_musicbrainz

            sp = init_spotify()

            for track_info in tracks_needing_lookup:
                folder = track_info['folder']
                artist = track_info['artist']
                track_name = track_info['track']
                source = track_info['source']
                has_valid_id3 = track_info['has_valid_id3']
                has_valid_folder = track_info['has_valid_folder']

                result = None

                if source in ('id3', 'folder'):
                    # We have valid artist/track - do text search (no fingerprinting needed)
                    result = lookup_track(track_name, artist_hint=artist, sp=sp,
                                          fetch_audio_features_flag=True)
                    if result:
                        logger.debug(f"  {folder.name}: Found via {source} tags")
                else:
                    # Unknown artist - try text search first, then fingerprinting as last resort
                    if track_name:
                        result = lookup_track(track_name, artist_hint=None, sp=sp,
                                              fetch_audio_features_flag=True)

                    # Only use fingerprinting if:
                    # 1. Text search failed
                    # 2. use_fingerprinting is enabled
                    # 3. We have no valid artist from either ID3 or folder
                    if not result and self.config.use_fingerprinting:
                        logger.debug(f"  {folder.name}: Trying fingerprinting (last resort)...")
                        fp_result = self._fingerprint_track(folder)
                        if fp_result:
                            # Fingerprinting found artist/track - now look up full metadata
                            result = lookup_track(
                                fp_result.get('track', ''),
                                artist_hint=fp_result.get('artist'),
                                sp=sp,
                                fetch_audio_features_flag=True
                            )
                            if result:
                                logger.debug(f"  {folder.name}: Found via fingerprinting")

                if result:
                    self.stats.tracks_metadata_found += 1
                    # Save to .INFO - all available metadata fields
                    stems = get_stem_files(folder, include_full_mix=True)
                    if 'full_mix' in stems:
                        info_path = get_info_path(stems['full_mix'])
                        info_data = {}

                        # Core metadata
                        if result.get('release_year'):
                            info_data['release_year'] = result['release_year']
                        if result.get('release_date'):
                            info_data['release_date'] = result['release_date']
                        if result.get('artists'):
                            info_data['artists'] = result['artists']
                        if result.get('label'):
                            info_data['label'] = result['label']
                        if result.get('genres'):
                            info_data['genres'] = result['genres']
                        if result.get('popularity') is not None:
                            info_data['popularity'] = result['popularity']
                        if result.get('album'):
                            info_data['album'] = result['album']

                        # IDs
                        if result.get('spotify_id'):
                            info_data['spotify_id'] = result['spotify_id']
                        if result.get('musicbrainz_id'):
                            info_data['musicbrainz_id'] = result['musicbrainz_id']

                        # Spotify audio features (if available)
                        spotify_features = [
                            'spotify_acousticness', 'spotify_energy', 'spotify_instrumentalness',
                            'spotify_time_signature', 'spotify_valence', 'spotify_danceability',
                            'spotify_speechiness', 'spotify_liveness', 'spotify_key',
                            'spotify_mode', 'spotify_tempo'
                        ]
                        for key in spotify_features:
                            if result.get(key) is not None:
                                info_data[key] = result[key]

                        if info_data:
                            safe_update(info_path, info_data)
                            logger.debug(f"  Saved metadata: {', '.join(info_data.keys())}")

        except ImportError:
            logger.warning("track_metadata_lookup not available")
        except Exception as e:
            logger.error(f"Metadata lookup failed: {e}")
            self.stats.errors.append(f"Metadata: {e}")

    def _fingerprint_track(self, folder: Path) -> Optional[Dict]:
        """
        Try to identify track using audio fingerprinting (AcoustID).

        Returns metadata dict or None if not found.
        """
        from core.file_utils import get_stem_files
        import subprocess

        stems = get_stem_files(folder, include_full_mix=True)
        if 'full_mix' not in stems:
            return None

        audio_path = stems['full_mix']

        try:
            # Generate fingerprint with fpcalc
            result = subprocess.run(
                ['fpcalc', '-json', str(audio_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return None

            import json
            fp_data = json.loads(result.stdout)
            fingerprint = fp_data.get('fingerprint')
            duration = fp_data.get('duration')

            if not fingerprint:
                return None

            # Try AcoustID lookup (requires API key and pyacoustid)
            try:
                import acoustid
                # Note: This requires ACOUSTID_API_KEY environment variable
                api_key = os.environ.get('ACOUSTID_API_KEY')
                if api_key:
                    results = acoustid.lookup(api_key, fingerprint, duration)
                    if results and results.get('results'):
                        # Parse MusicBrainz recording from results
                        for result in results['results']:
                            if 'recordings' in result:
                                rec = result['recordings'][0]
                                return {
                                    'artist': rec.get('artists', [{}])[0].get('name'),
                                    'track': rec.get('title'),
                                    'musicbrainz_id': rec.get('id'),
                                }
            except ImportError:
                logger.debug("pyacoustid not available for fingerprint lookup")
            except Exception as e:
                logger.debug(f"AcoustID lookup failed: {e}")

        except Exception as e:
            logger.debug(f"Fingerprinting failed: {e}")

        return None

    def _run_first_stage_features(self, folders: List[Path]):
        """Extract features that will be migrated to crops (parallel processing)."""
        num_workers = self.config.feature_workers

        if num_workers <= 1:
            # Sequential fallback
            self._run_first_stage_features_sequential(folders)
            return

        logger.info(f"Processing {len(folders)} folders with {num_workers} workers...")

        # Prepare arguments for worker function
        args_list = [(folder, self.config.overwrite) for folder in folders]

        success_count = 0
        fail_count = 0
        progress = ProgressBar(len(folders), desc="Features")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_folder = {
                executor.submit(_process_folder_features, args): args[0]
                for args in args_list
            }

            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_folder), 1):
                folder = future_to_folder[future]
                try:
                    folder_name, success, message = future.result()
                    if success:
                        success_count += 1
                        logger.debug(f"{folder_name}: {message}")
                    else:
                        fail_count += 1
                        logger.warning(f"{folder_name}: {message}")
                except Exception as e:
                    fail_count += 1
                    logger.error(f"{folder.name}: {e}")

                # Progress bar update
                logger.info(progress.update(i))

        self.stats.tracks_analyzed = success_count
        logger.info(progress.finish(f"{success_count} success, {fail_count} failed"))

    def _run_first_stage_features_sequential(self, folders: List[Path]):
        """Sequential fallback for feature extraction."""
        try:
            from timbral.loudness import analyze_file_loudness
            from spectral.spectral_features import analyze_spectral_features
            from core.json_handler import should_process

            # Define output keys for each feature type
            LOUDNESS_KEYS = ['lufs', 'lra', 'peak_dbfs', 'true_peak_dbfs']
            SPECTRAL_KEYS = ['spectral_flatness', 'spectral_flux', 'spectral_skewness', 'spectral_kurtosis']

            progress = ProgressBar(len(folders), desc="Features")

            for i, folder in enumerate(folders, 1):
                stems = get_stem_files(folder, include_full_mix=True)
                if 'full_mix' not in stems:
                    logger.info(progress.update(i))
                    continue

                full_mix = stems['full_mix']
                info_path = get_info_path(full_mix)
                results = {}

                # Loudness - check ALL output keys
                if should_process(info_path, LOUDNESS_KEYS, self.config.should_overwrite('loudness')):
                    try:
                        results.update(analyze_file_loudness(full_mix))
                    except Exception as e:
                        logger.debug(f"Loudness failed: {e}")

                # Spectral - check ALL output keys
                if should_process(info_path, SPECTRAL_KEYS, self.config.should_overwrite('spectral')):
                    try:
                        results.update(analyze_spectral_features(full_mix))
                    except Exception as e:
                        logger.debug(f"Spectral failed: {e}")

                if results:
                    safe_update(info_path, results)

                self.stats.tracks_analyzed += 1
                logger.info(progress.update(i))

            logger.info(progress.finish(f"{self.stats.tracks_analyzed} analyzed"))

        except ImportError as e:
            logger.warning(f"First-stage features not available: {e}")

    def _extract_track_title(self, folder_name: str) -> str:
        """Extract track title from folder name, removing leading track numbers."""
        import re
        # Remove leading track numbers like "6. " or "03 - " or "3. "
        cleaned = re.sub(r'^\d+[\.\-\s]+\s*', '', folder_name).strip()
        return cleaned if cleaned else folder_name

    def _normalize_for_matching(self, name: str) -> str:
        """Normalize a name for fuzzy matching (lowercase, remove punctuation)."""
        import re
        # Lowercase, remove underscores/dashes/spaces, keep alphanumeric
        return re.sub(r'[^a-z0-9]', '', name.lower())

    def _check_existing_crops(self, folder: Path, crops_dir: Path) -> bool:
        """Check if crops already exist for a track.

        Returns True if crops exist and should be skipped.
        Handles name mismatches between source folders and crop folders.
        """
        # If crops_dir doesn't exist, no crops exist
        if not crops_dir.exists():
            return False

        def has_crop_files(crop_folder: Path) -> bool:
            """Check if a folder contains crop files."""
            for ext in ['.flac', '.mp3', '.wav', '.m4a']:
                crops = list(crop_folder.glob(f"*_[0-9]{ext}"))
                crops.extend(crop_folder.glob(f"*_[0-9][0-9]{ext}"))
                if crops:
                    return True
            return False

        # First try exact match
        track_crops_dir = crops_dir / folder.name
        if track_crops_dir.exists() and has_crop_files(track_crops_dir):
            return True

        # Try matching by track title (handles "6. Song" vs "Artist - Song" mismatches)
        track_title = self._extract_track_title(folder.name)
        normalized_title = self._normalize_for_matching(track_title)

        # Search for crop folders containing this track title
        for crop_folder in crops_dir.iterdir():
            if not crop_folder.is_dir():
                continue

            # Check if title is contained in crop folder name
            if track_title in crop_folder.name:
                if has_crop_files(crop_folder):
                    return True

            # Try normalized matching (handles 01_41 vs 0141)
            normalized_crop = self._normalize_for_matching(crop_folder.name)
            if normalized_title and normalized_title in normalized_crop:
                if has_crop_files(crop_folder):
                    return True

        return False

    def _run_cropping(self):
        """Stage 3: Create crops from full tracks (parallel processing)."""
        logger.info("\n" + fmt_header("=" * 70))
        logger.info(fmt_stage("[STAGE 3] CROPPING"))
        logger.info(fmt_header("=" * 70))

        start_time = time.time()
        self.stats.start_operation('cropping')

        # Determine crops output directory
        crops_dir = self.working_dir.parent / f"{self.working_dir.name}_crops"

        logger.info(f"Creating crops in: {fmt_filename(str(crops_dir))}")
        logger.info(f"Mode: {fmt_notification(self.config.crop_mode)}")
        logger.info(f"Length: {self.config.crop_length_samples} samples (~{self.config.crop_length_samples / 44100:.1f}s at 44.1kHz)")

        num_workers = self.config.crop_workers

        try:
            # Snapshot folders BEFORE processing to prevent infinite loops
            # (in case crops_dir somehow ends up inside working_dir)
            folders = list(find_organized_folders(self.working_dir))
            logger.info(f"Snapshot: {len(folders)} folders to process")

            # Filter out any folders that might be inside crops_dir
            crops_dir_resolved = crops_dir.resolve()
            folders = [
                f for f in folders
                if not str(f.resolve()).startswith(str(crops_dir_resolved))
            ]

            # Pre-check for existing crops (skip tracks that already have crops)
            if not self.config.should_overwrite('crops'):
                folders_needing_crops = []
                skipped_count = 0
                for folder in folders:
                    if self._check_existing_crops(folder, crops_dir):
                        skipped_count += 1
                    else:
                        folders_needing_crops.append(folder)

                if skipped_count > 0:
                    logger.info(f"Skipping {skipped_count} tracks with existing crops")

                if not folders_needing_crops:
                    logger.info(fmt_success("All tracks already have crops. Use overwrite.crops=true to regenerate."))
                    self.stats.end_operation('cropping', items_skipped=len(folders))
                    self.stats.time_cropping = time.time() - start_time
                    self.crops_dir = crops_dir
                    return

                folders = folders_needing_crops
                logger.info(f"Processing {len(folders)} tracks that need crops")

            crops_dir.mkdir(parents=True, exist_ok=True)

            if num_workers <= 1:
                # Sequential fallback
                self._run_cropping_sequential(folders, crops_dir)
            else:
                logger.info(f"Processing {len(folders)} folders with {num_workers} workers...")

                # Prepare arguments for worker function
                sequential = self.config.crop_mode == 'sequential'
                args_list = [
                    (folder, crops_dir, self.config.crop_length_samples,
                     sequential, self.config.crop_overlap, self.config.crop_div4,
                     self.config.should_overwrite('crops'))
                    for folder in folders
                ]

                success_count = 0
                fail_count = 0
                total_crops = 0
                progress = ProgressBar(len(folders), desc="Cropping")

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    future_to_folder = {
                        executor.submit(_process_folder_crops, args): args[0]
                        for args in args_list
                    }

                    for i, future in enumerate(as_completed(future_to_folder), 1):
                        folder = future_to_folder[future]
                        try:
                            folder_name, crop_count, message = future.result()
                            if crop_count > 0:
                                success_count += 1
                                total_crops += crop_count
                                logger.debug(f"{folder_name}: {message}")
                            elif "lock" in message.lower():
                                logger.warning(f"{folder_name}: {message}")
                            else:
                                fail_count += 1
                                logger.warning(f"{folder_name}: {message}")
                        except Exception as e:
                            fail_count += 1
                            logger.error(f"{folder.name}: {e}")

                        # Progress bar update
                        logger.info(progress.update(i, f"{total_crops} crops"))

                self.stats.crops_created = total_crops
                logger.info(progress.finish(f"{total_crops} crops from {success_count} tracks"))

        except Exception as e:
            logger.error(f"Cropping failed: {e}")
            self.stats.errors.append(f"Cropping: {e}")

        crop_time = self.stats.end_operation('cropping', 
            items_processed=self.stats.crops_created)
        self.stats.time_cropping = time.time() - start_time
        logger.info(fmt_dim(f"Cropping completed in {crop_time:.1f}s ({self.stats.crops_created} crops)"))

        # Update working dir for crop analysis
        self.crops_dir = crops_dir

    def _run_cropping_sequential(self, folders: List[Path], crops_dir: Path):
        """Sequential fallback for cropping."""
        from tools.create_training_crops import create_crops_for_file

        progress = ProgressBar(len(folders), desc="Cropping")
        total_crops = 0

        for i, folder in enumerate(folders, 1):
            try:
                count = create_crops_for_file(
                    folder,
                    length_samples=self.config.crop_length_samples,
                    output_dir=crops_dir,
                    sequential=self.config.crop_mode == 'sequential',
                    overlap=self.config.crop_overlap,
                    div4=self.config.crop_div4,
                    overwrite=self.config.should_overwrite('crops'),
                )
                self.stats.crops_created += count
                total_crops += count
            except Exception as e:
                logger.warning(f"{folder.name}: {e}")

            logger.info(progress.update(i, f"{total_crops} crops"))

        logger.info(progress.finish(f"{total_crops} crops created"))

    def _run_crop_analysis(self):
        """Stage 4: Analyze all crops."""
        logger.info("\n" + fmt_header("=" * 70))
        logger.info(fmt_stage("[STAGE 4] CROP ANALYSIS"))
        logger.info(fmt_header("=" * 70))

        start_time = time.time()
        self.stats.start_operation('crop_analysis')

        # Find crops directory
        crops_dir = getattr(self, 'crops_dir', None)
        if not crops_dir:
            crops_dir = self.working_dir.parent / f"{self.working_dir.name}_crops"

        if not crops_dir.exists():
            logger.warning(f"Crops directory not found: {crops_dir}")
            return

        # Clean up any stale lock files in crops directory
        from core.file_locks import remove_all_locks
        locks_removed = remove_all_locks(crops_dir)
        if locks_removed > 0:
            logger.info(f"Cleaned up {locks_removed} stale lock files in crops")

        crop_folders = find_crop_folders(crops_dir)
        all_crops = []
        for folder in crop_folders:
            all_crops.extend(find_crop_files(folder))

        if not all_crops:
            logger.warning("No crops found")
            return

        logger.info(f"Found {len(all_crops)} crops to analyze")

        # Check if stems exist or need Demucs
        sample_crop = all_crops[0]
        from core.file_utils import get_crop_stem_files
        sample_stems = get_crop_stem_files(sample_crop)

        has_stems = len([k for k in sample_stems if k != 'source']) >= 4

        if not has_stems and not self.config.stems_source:
            # Need to run Demucs on crops
            logger.info("\n[4a] Crop Stem Separation (Demucs)")
            self.stats.start_operation('crop_demucs')
            self._run_crop_demucs(all_crops)
            crop_demucs_time = self.stats.end_operation('crop_demucs', 
                items_processed=len(all_crops))
            logger.info(fmt_dim(f"    Crop Demucs completed in {crop_demucs_time:.1f}s"))
        elif self.config.stems_source:
            # Crop stems from source
            logger.info("\n[4a] Cropping stems from source")
            self.stats.start_operation('crop_stems_copy')
            self._crop_stems_from_source(crops_dir)
            stems_copy_time = self.stats.end_operation('crop_stems_copy')
            logger.info(fmt_dim(f"    Stems copy completed in {stems_copy_time:.1f}s"))

        # Run feature extraction on crops
        logger.info("\n[4b] Crop Feature Extraction")
        self.stats.start_operation('crop_features')
        self._run_crop_features(crops_dir)
        crop_features_time = self.stats.end_operation('crop_features',
            items_processed=self.stats.crops_analyzed)
        logger.info(fmt_dim(f"    Crop features completed in {crop_features_time:.1f}s"))

        total_crop_analysis_time = self.stats.end_operation('crop_analysis',
            items_processed=len(all_crops))
        self.stats.time_crop_analysis = time.time() - start_time
        logger.info(fmt_dim(f"Crop analysis completed in {total_crop_analysis_time:.1f}s"))

    def _run_crop_demucs(self, crops: List[Path]):
        """Run Demucs on all crops (batch mode for efficiency)."""
        # Import Demucs and keep model in memory
        try:
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            import torchaudio

            logger.info("Loading Demucs model (keeping in memory for batch)...")
            model = get_model('htdemucs')
            model.to(self.config.device)
            model.eval()

            for i, crop_path in enumerate(crops, 1):
                logger.info(f"[{i}/{len(crops)}] Separating: {crop_path.name}")

                try:
                    # Load audio
                    wav, sr = torchaudio.load(crop_path)
                    if sr != model.samplerate:
                        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
                    wav = wav.to(self.config.device)

                    # Apply model
                    with torch.no_grad():
                        sources = apply_model(model, wav[None], shifts=0)[0]

                    # Save stems
                    crop_stem = crop_path.stem
                    crop_dir = crop_path.parent

                    for j, stem_name in enumerate(['drums', 'bass', 'other', 'vocals']):
                        stem_path = crop_dir / f"{crop_stem}_{stem_name}.flac"
                        torchaudio.save(stem_path, sources[j].cpu(), model.samplerate)

                except Exception as e:
                    logger.warning(f"Failed to separate {crop_path.name}: {e}")

            # Clean up
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Crop Demucs failed: {e}")
            self.stats.errors.append(f"Crop Demucs: {e}")

    def _crop_stems_from_source(self, crops_dir: Path):
        """Crop stems from pre-existing source stems."""
        try:
            from tools.crop_stems_from_source import batch_crop_stems

            batch_crop_stems(
                crops_dir,
                self.config.stems_source,
                output_format='mp3',
                overwrite=self.config.should_overwrite('crops'),
            )
        except Exception as e:
            logger.error(f"Stem cropping failed: {e}")
            self.stats.errors.append(f"Stem cropping: {e}")

    def _run_crop_features(self, crops_dir: Path):
        """Run feature extraction on all crops."""
        # Use the unified pipeline in crops mode
        try:
            from pipeline import Pipeline, PipelineConfig

            config = PipelineConfig(
                input_dir=crops_dir,
                device=self.config.device,
                batch=self.config.batch_feature_extraction,
                crops=True,
                overwrite=self.config.overwrite,  # Global overwrite passed to sub-pipeline
                feature_workers=self.config.feature_workers,
                essentia_workers=self.config.essentia_workers,  # TensorFlow-safe limit
                skip_organize=True,
                skip_demucs=True,  # Already done
                skip_flamingo=self.config.skip_flamingo,
                skip_audiobox=self.config.skip_audiobox,
                skip_midi=True,
                flamingo_model=self.config.flamingo_model,
                flamingo_token_limits=self.config.flamingo_token_limits,
                flamingo_prompts=self.config.flamingo_prompts,
            )

            pipeline = Pipeline(config)
            pipeline.run()

            self.stats.crops_analyzed = pipeline.stats.get('crops_processed', 0)

        except Exception as e:
            logger.error(f"Crop feature extraction failed: {e}")
            self.stats.errors.append(f"Crop features: {e}")

    def _extract_artist_from_name(self, folder_name: str) -> str:
        """Extract artist from folder name."""
        import re
        # Remove leading track number
        name = re.sub(r'^\d+\s*', '', folder_name).strip()
        parts = name.split(' - ')
        return parts[0].strip() if parts else ''

    def _parse_folder_name(self, folder_name: str) -> tuple:
        """Parse folder name into (artist, album, track)."""
        import re
        name = re.sub(r'^\d+\s*', '', folder_name).strip()
        parts = name.split(' - ')

        if len(parts) >= 3:
            return parts[0].strip(), parts[1].strip(), ' - '.join(parts[2:]).strip()
        elif len(parts) == 2:
            return parts[0].strip(), '', parts[1].strip()
        else:
            return '', '', name

    def _print_summary(self, total_time: float):
        """Print pipeline summary with detailed timing statistics."""
        print("\n" + fmt_header("=" * 70))
        print(fmt_header("MASTER PIPELINE SUMMARY"))
        print(fmt_header("=" * 70))

        if not self.config.skip_organize:
            print(fmt_stage("\n[Stage 1] Organization"))
            print(f"  Files found:     {self.stats.files_found}")
            print(f"  Files organized: {fmt_success(str(self.stats.files_organized))}")
            print(f"  Time:            {fmt_dim(f'{self.stats.time_organize:.1f}s')}")
            
            # Show sub-operation timing
            if 'organize' in self.stats.operation_timing:
                op = self.stats.operation_timing['organize']
                if op.items_processed > 0:
                    print(f"  Speed:           {fmt_dim(f'{op.items_per_second:.2f} files/s')}")

        if not self.config.skip_track_analysis:
            print(fmt_stage("\n[Stage 2] Track Analysis"))
            print(f"  Tracks total:    {self.stats.tracks_total}")
            print(f"  Stems separated: {fmt_success(str(self.stats.tracks_separated))}")
            print(f"  Metadata found:  {self.stats.tracks_metadata_found}")
            print(f"  Analyzed:        {fmt_success(str(self.stats.tracks_analyzed))}")
            print(f"  Time:            {fmt_dim(f'{self.stats.time_track_analysis:.1f}s')}")
            
            # Show sub-stage timing details
            sub_ops = ['demucs', 'rhythm', 'metadata', 'first_stage_features']
            sub_labels = ['Demucs', 'Rhythm', 'Metadata', 'Features']
            for op_name, label in zip(sub_ops, sub_labels):
                if op_name in self.stats.operation_timing:
                    op = self.stats.operation_timing[op_name]
                    speed_str = f" ({op.seconds_per_item:.2f}s/track)" if op.items_processed > 0 else ""
                    print(fmt_dim(f"    {label:12} {op.elapsed:6.1f}s{speed_str}"))

        if not self.config.skip_crops:
            print(fmt_stage("\n[Stage 3] Cropping"))
            print(f"  Crops created:   {fmt_success(str(self.stats.crops_created))}")
            print(f"  Time:            {fmt_dim(f'{self.stats.time_cropping:.1f}s')}")
            
            if 'cropping' in self.stats.operation_timing:
                op = self.stats.operation_timing['cropping']
                if op.items_processed > 0:
                    print(f"  Speed:           {fmt_dim(f'{op.items_per_second:.2f} crops/s')}")

        if not self.config.skip_crop_analysis:
            print(fmt_stage("\n[Stage 4] Crop Analysis"))
            print(f"  Crops analyzed:  {fmt_success(str(self.stats.crops_analyzed))}")
            print(f"  Time:            {fmt_dim(f'{self.stats.time_crop_analysis:.1f}s')}")
            
            # Show sub-operation timing
            crop_sub_ops = ['crop_demucs', 'crop_stems_copy', 'crop_features']
            crop_labels = ['Crop Demucs', 'Stems Copy', 'Features']
            for op_name, label in zip(crop_sub_ops, crop_labels):
                if op_name in self.stats.operation_timing:
                    op = self.stats.operation_timing[op_name]
                    speed_str = f" ({op.items_per_second:.2f}/s)" if op.items_processed > 0 else ""
                    print(fmt_dim(f"    {label:12} {op.elapsed:6.1f}s{speed_str}"))

        print(fmt_notification(f"\nTotal Time: {total_time:.1f}s ({total_time/60:.1f} min)"))

        # Show realtime factor (processing speed relative to audio duration)
        if self.stats.total_audio_duration > 0:
            audio_mins = self.stats.total_audio_duration / 60
            realtime = self.stats.realtime_factor
            print(f"Audio processed: {fmt_success(f'{audio_mins:.1f} min')} ({self.stats.total_audio_duration:.0f}s)")
            print(f"Processing speed: {fmt_success(f'{realtime:.2f}x realtime')}")
            if realtime >= 1.0:
                print(fmt_dim(f"  (Processed {realtime:.1f} seconds of audio per second of wall time)"))
            else:
                inv_realtime = 1.0 / realtime if realtime > 0 else 0
                print(fmt_dim(f"  (Took {inv_realtime:.1f} seconds of wall time per second of audio)"))

        if self.stats.errors:
            print(fmt_error(f"\nErrors ({len(self.stats.errors)}):"))
            for err in self.stats.errors[:5]:
                print(fmt_error(f"  - {err}"))
            if len(self.stats.errors) > 5:
                print(fmt_dim(f"  ... and {len(self.stats.errors) - 5} more"))

        print(fmt_header("=" * 70))

    def _write_stats_log(self):
        """Write detailed statistics to a JSON log file."""
        from datetime import datetime
        import json
        
        # Create log filename with timestamp
        timestamp = datetime.fromtimestamp(self.stats.run_start_time).strftime('%Y%m%d_%H%M%S')
        log_filename = f"pipeline_run_{timestamp}.json"
        log_path = self.working_dir / log_filename
        
        # Get stats as dictionary
        stats_dict = self.stats.to_dict()
        
        # Add config summary
        stats_dict['config'] = {
            'input_dir': str(self.config.input_dir),
            'output_dir': str(self.config.output_dir) if self.config.output_dir else None,
            'device': self.config.device,
            'demucs_model': self.config.demucs_model,
            'demucs_workers': self.config.demucs_workers,
            'crop_mode': self.config.crop_mode,
            'crop_length_samples': self.config.crop_length_samples,
            'crop_workers': self.config.crop_workers,
            'rhythm_workers': self.config.rhythm_workers,
            'feature_workers': self.config.feature_workers,
        }
        
        try:
            with open(log_path, 'w') as f:
                json.dump(stats_dict, f, indent=2)
            logger.info(f"Statistics written to: {fmt_filename(str(log_path))}")
        except Exception as e:
            logger.warning(f"Failed to write stats log: {e}")
        
        # Also print a summary table of operation timing to console
        if self.stats.operation_timing:
            print("\n" + fmt_header("=== OPERATION TIMING DETAILS ==="))
            print(f"{'Operation':<25} {'Time':>10} {'Items':>8} {'Speed':>15}")
            print("-" * 60)
            
            for name, timing in sorted(self.stats.operation_timing.items(), 
                                       key=lambda x: x[1].start_time):
                time_str = f"{timing.elapsed:.1f}s"
                items_str = str(timing.items_processed) if timing.items_processed > 0 else "-"
                
                if timing.items_processed > 0 and timing.elapsed > 0:
                    if timing.items_per_second >= 1:
                        speed_str = f"{timing.items_per_second:.2f}/s"
                    else:
                        speed_str = f"{timing.seconds_per_item:.2f}s/item"
                else:
                    speed_str = "-"
                
                print(f"{name:<25} {time_str:>10} {items_str:>8} {speed_str:>15}")
            
            print("-" * 60)
            print(f"{'Total':<25} {self.stats.total_time:>9.1f}s")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Master Pipeline: Raw audio -> Organized -> Analyzed -> Crops -> Analyzed crops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (recommended - all options documented)
  python src/master_pipeline.py --config config/master_pipeline.yaml

  # Config file with CLI overrides
  python src/master_pipeline.py --config config/my_project.yaml --overwrite

  # Without config file
  python src/master_pipeline.py /path/to/raw/audio --output /path/to/output

  # Skip stages
  python src/master_pipeline.py /path/to/organized --skip-organize --skip-track-analysis

Config file template: config/master_pipeline.yaml
        """
    )

    # Config file (primary way to set options)
    parser.add_argument('--config', '-c', type=str,
                        help='YAML config file (see config/master_pipeline.yaml)')

    # Input/output (can be set in config or CLI)
    parser.add_argument('input', nargs='?', help='Input directory (or set in config)')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--stems-source', help='Pre-existing stems directory')

    # Stage control (CLI overrides config)
    parser.add_argument('--skip-organize', action='store_true')
    parser.add_argument('--skip-track-analysis', action='store_true')
    parser.add_argument('--skip-crops', action='store_true')
    parser.add_argument('--skip-crop-analysis', action='store_true')

    # Demucs (CLI overrides config)
    parser.add_argument('--demucs-model', type=str,
                        choices=['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'mdx_extra', 'mdx_extra_q'],
                        help='Demucs model (htdemucs=fast, htdemucs_ft=4x slower but better)')
    parser.add_argument('--demucs-shifts', type=int,
                        help='Random shifts (0=fast, 1+=better quality)')
    parser.add_argument('--demucs-segment', type=int,
                        help='Segment length in seconds (higher=better quality, more VRAM). '
                             '~10s min, ~45s recommended for 16GB cards')
    parser.add_argument('--demucs-workers', type=int,
                        help='Number of parallel Demucs workers (default: 2, each uses ~5GB VRAM)')

    # Cropping (CLI overrides config)
    parser.add_argument('--crop-length', type=int)
    parser.add_argument('--crop-mode', choices=['sequential', 'beat-aligned'])
    parser.add_argument('--crop-overlap', action='store_true')
    parser.add_argument('--crop-div4', action='store_true')
    parser.add_argument('--crop-workers', type=int,
                        help='Number of parallel workers for cropping (default: 6)')
    parser.add_argument('--rhythm-workers', type=int,
                        help='Number of parallel workers for rhythm analysis (default: 4)')

    # Features (CLI overrides config)
    parser.add_argument('--skip-flamingo', action='store_true')
    parser.add_argument('--skip-audiobox', action='store_true')
    parser.add_argument('--skip-essentia', action='store_true')
    parser.add_argument('--flamingo-model', choices=['IQ3_M', 'Q6_K', 'Q8_0'])
    parser.add_argument('--skip-filename-cleanup', action='store_true',
                        help='Skip filename cleanup (T5 tokenizer compatibility)')

    # Processing (CLI overrides config)
    parser.add_argument('--device', choices=['cuda', 'cpu'])
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--feature-workers', type=int,
                        help='Number of parallel workers for feature extraction (default: 8)')

    # Generate config template
    parser.add_argument('--generate-config', type=str, metavar='PATH',
                        help='Generate a config file template at PATH and exit')

    args = parser.parse_args()

    # Handle --generate-config
    if args.generate_config:
        import shutil
        template = Path(__file__).parent.parent / 'config' / 'master_pipeline.yaml'
        dest = Path(args.generate_config)
        if template.exists():
            shutil.copy(template, dest)
            print(f"Config template created: {dest}")
        else:
            print(f"Template not found at {template}")
        sys.exit(0)

    # Load config from file or create default
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            sys.exit(1)
        config = MasterPipelineConfig.from_yaml(config_path)
        logger.info(f"Loaded config from: {config_path}")
    else:
        # Require input path if no config
        if not args.input:
            parser.error("Either --config or input path is required")
        config = MasterPipelineConfig(input_dir=Path(args.input))

    # CLI overrides config
    if args.input:
        config.input_dir = Path(args.input)
    if args.output:
        config.output_dir = Path(args.output)
    if args.stems_source:
        config.stems_source = Path(args.stems_source)
    if args.skip_organize:
        config.skip_organize = True
    if args.skip_track_analysis:
        config.skip_track_analysis = True
    if args.skip_crops:
        config.skip_crops = True
    if args.skip_crop_analysis:
        config.skip_crop_analysis = True
    if args.demucs_model:
        config.demucs_model = args.demucs_model
    if args.demucs_shifts is not None:
        config.demucs_shifts = args.demucs_shifts
    if args.demucs_segment is not None:
        config.demucs_segment = args.demucs_segment
    if args.demucs_workers is not None:
        config.demucs_workers = args.demucs_workers
    if args.crop_length is not None:
        config.crop_length_samples = args.crop_length
    if args.crop_mode:
        config.crop_mode = args.crop_mode
    if args.crop_overlap:
        config.crop_overlap = True
    if args.crop_div4:
        config.crop_div4 = True
    if args.crop_workers is not None:
        config.crop_workers = args.crop_workers
    if args.rhythm_workers is not None:
        config.rhythm_workers = args.rhythm_workers
    if args.skip_flamingo:
        config.skip_flamingo = True
    if args.skip_audiobox:
        config.skip_audiobox = True
    if args.skip_essentia:
        config.skip_essentia = True
    if args.flamingo_model:
        config.flamingo_model = args.flamingo_model
    if args.skip_filename_cleanup:
        config.cleanup_filenames = False
    if args.device:
        config.device = args.device
    if args.overwrite:
        config.overwrite = True
    if args.dry_run:
        config.dry_run = True
    if args.verbose:
        config.verbose = True
    if args.feature_workers is not None:
        config.feature_workers = args.feature_workers

    setup_colored_logging(level=logging.DEBUG if config.verbose else logging.INFO)

    # Validate input
    if not config.input_dir or not config.input_dir.exists():
        logger.error(f"Input path does not exist: {config.input_dir}")
        sys.exit(1)

    # Run pipeline
    pipeline = MasterPipeline(config)
    stats = pipeline.run()

    # Exit code
    sys.exit(1 if stats.errors else 0)


if __name__ == "__main__":
    main()
