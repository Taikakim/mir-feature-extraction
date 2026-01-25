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
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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

from core.common import setup_logging
from core.file_utils import find_organized_folders, find_crop_folders, find_crop_files
from core.json_handler import safe_update, read_info, get_info_path

logger = logging.getLogger(__name__)

# Artists that should trigger metadata lookup for real artist
VARIOUS_ARTISTS_ALIASES = {
    'various artists',
    'various',
    'va',
    'v/a',
    'v.a.',
    'compilation',
    'various artist',
    'unknown artist',
    'unknown',
    '',
}


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

    # Demucs settings
    demucs_model: str = 'htdemucs'  # htdemucs (fast), htdemucs_ft (4x slower, better)
    demucs_shifts: int = 0  # 0 for MIR analysis (faster), 1+ for hifi
    demucs_segment: Optional[int] = None  # Segment length (None = model default, htdemucs max ~7.8s)
    demucs_format: str = 'mp3'
    demucs_jobs: int = 4
    demucs_optimized: bool = True  # Use optimized implementation (model persistence, SDPA)
    demucs_compile: bool = False  # Use torch.compile (faster after warmup)
    demucs_compile_mode: str = 'reduce-overhead'  # torch.compile mode for ROCm

    # Crop settings
    crop_length_samples: int = 2097152  # ~47.5s at 44.1kHz
    crop_mode: str = 'sequential'  # 'sequential' or 'beat-aligned'
    crop_overlap: bool = False
    crop_div4: bool = False
    crop_include_stems: bool = True
    crop_threads: int = 1

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

    # Metadata
    skip_metadata: bool = False
    use_fingerprinting: bool = True
    various_artists_aliases: Set[str] = field(default_factory=lambda: VARIOUS_ARTISTS_ALIASES.copy())
    fix_various_artists: bool = True  # Fix "Various Artists" names during organization

    # Filename cleanup (T5 tokenizer compatibility)
    cleanup_filenames: bool = True

    # Processing
    overwrite: bool = False
    dry_run: bool = False
    verbose: bool = False

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
        demucs = data.get('demucs', {})
        cropping = data.get('cropping', {})
        features = data.get('features', {})
        flamingo = data.get('music_flamingo', {})
        metadata = data.get('metadata', {})
        processing = data.get('processing', {})

        # Build config
        config = cls(
            input_dir=Path(paths['input']) if paths.get('input') else Path('.'),
            output_dir=Path(paths['output']) if paths.get('output') else None,
            stems_source=Path(paths['stems_source']) if paths.get('stems_source') else None,

            # Stages (inverted - config has enabled flags, we have skip flags)
            skip_organize=not stages.get('organize', True),
            skip_track_analysis=not stages.get('track_analysis', True),
            skip_crops=not stages.get('cropping', True),
            skip_crop_analysis=not stages.get('crop_analysis', True),

            # Demucs
            demucs_model=demucs.get('model', 'htdemucs'),
            demucs_shifts=demucs.get('shifts', 0),
            demucs_segment=demucs.get('segment', 45),
            demucs_format=demucs.get('output_format', 'mp3'),
            demucs_jobs=demucs.get('jobs', 4),
            demucs_optimized=demucs.get('optimized', True),
            demucs_compile=demucs.get('compile', False),
            demucs_compile_mode=demucs.get('compile_mode', 'reduce-overhead'),
            device=demucs.get('device', processing.get('device', 'cuda')),

            # Cropping
            crop_length_samples=cropping.get('length_samples', 2097152),
            crop_mode=cropping.get('mode', 'sequential'),
            crop_overlap=cropping.get('overlap', False),
            crop_div4=cropping.get('div4', False),
            crop_include_stems=cropping.get('include_stems', True),
            crop_threads=cropping.get('threads', 1),

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

            # Metadata
            skip_metadata=not metadata.get('enabled', True),
            use_fingerprinting=metadata.get('use_fingerprinting', True),
            fix_various_artists=metadata.get('fix_various_artists', True),
            various_artists_aliases=set(metadata.get('various_artists_aliases', list(VARIOUS_ARTISTS_ALIASES))),

            # Filename cleanup
            cleanup_filenames=data.get('filename_cleanup', {}).get('enabled', True),

            # Processing
            overwrite=processing.get('overwrite', False),
            verbose=processing.get('verbose', False),
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
                'jobs': self.demucs_jobs,
                'device': self.device,
                'optimized': self.demucs_optimized,
                'compile': self.demucs_compile,
                'compile_mode': self.demucs_compile_mode,
            },
            'cropping': {
                'length_samples': self.crop_length_samples,
                'mode': self.crop_mode,
                'overlap': self.crop_overlap,
                'div4': self.crop_div4,
                'include_stems': self.crop_include_stems,
                'threads': self.crop_threads,
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
            },
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


@dataclass
class PipelineStats:
    """Statistics for pipeline execution."""
    # Organization
    files_found: int = 0
    files_organized: int = 0

    # Track analysis
    tracks_total: int = 0
    tracks_separated: int = 0
    tracks_analyzed: int = 0
    tracks_metadata_found: int = 0

    # Crops
    crops_created: int = 0
    crops_analyzed: int = 0

    # Timing
    time_organize: float = 0.0
    time_track_analysis: float = 0.0
    time_cropping: float = 0.0
    time_crop_analysis: float = 0.0

    errors: List[str] = field(default_factory=list)


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

    def run(self) -> PipelineStats:
        """Run the complete pipeline."""
        logger.info("=" * 70)
        logger.info("MIR MASTER PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Input:  {self.config.input_dir}")
        logger.info(f"Output: {self.working_dir}")
        if self.config.stems_source:
            logger.info(f"Stems:  {self.config.stems_source}")
        logger.info("=" * 70)

        total_start = time.time()

        # Stage 1: Organization
        if not self.config.skip_organize:
            self._run_organization()
        else:
            logger.info("\n[STAGE 1] Organization: SKIPPED")

        # Stage 2: Full Track Analysis
        if not self.config.skip_track_analysis:
            self._run_track_analysis()
        else:
            logger.info("\n[STAGE 2] Track Analysis: SKIPPED")

        # Stage 3: Create Crops
        if not self.config.skip_crops:
            self._run_cropping()
        else:
            logger.info("\n[STAGE 3] Cropping: SKIPPED")

        # Stage 4: Crop Analysis
        if not self.config.skip_crop_analysis:
            self._run_crop_analysis()
        else:
            logger.info("\n[STAGE 4] Crop Analysis: SKIPPED")

        # Summary
        total_time = time.time() - total_start
        self._print_summary(total_time)

        return self.stats

    def _run_organization(self):
        """Stage 1: Organize raw files into project structure."""
        logger.info("\n" + "=" * 70)
        logger.info("[STAGE 1] ORGANIZATION")
        logger.info("=" * 70)

        start_time = time.time()

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

        # Cleanup filenames for T5 tokenizer compatibility
        if self.config.cleanup_filenames and not self.config.dry_run:
            self._run_filename_cleanup()

        self.stats.time_organize = time.time() - start_time

        logger.info(f"Organization complete: {stats['organized']}/{stats['total']} files")

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

            for item in items:
                if apply_rename(item):
                    logger.debug(f"Renamed: {item['old_name']} → {item['new_name']}")

        except Exception as e:
            logger.warning(f"Filename cleanup failed: {e}")
            self.stats.errors.append(f"Filename cleanup: {e}")

    def _run_track_analysis(self):
        """Stage 2: Analyze full tracks (Demucs, beats, metadata)."""
        logger.info("\n" + "=" * 70)
        logger.info("[STAGE 2] FULL TRACK ANALYSIS")
        logger.info("=" * 70)

        start_time = time.time()

        folders = find_organized_folders(self.working_dir)
        self.stats.tracks_total = len(folders)

        if not folders:
            logger.warning("No organized folders found")
            return

        logger.info(f"Found {len(folders)} tracks to analyze")

        # Sub-stage 2a: Demucs separation for all tracks
        logger.info("\n[2a] Stem Separation (Demucs)")
        self._run_demucs_batch(folders)

        # Sub-stage 2b: Beat/onset/downbeat detection
        logger.info("\n[2b] Rhythm Analysis (beats, onsets, downbeats)")
        self._run_rhythm_analysis(folders)

        # Sub-stage 2c: Metadata lookup (with fingerprinting for Various Artists)
        logger.info("\n[2c] Metadata Lookup")
        self._run_metadata_lookup(folders)

        # Sub-stage 2d: First-stage features (migrated to crops)
        logger.info("\n[2d] First-Stage Features")
        self._run_first_stage_features(folders)

        self.stats.time_track_analysis = time.time() - start_time

    def _run_demucs_batch(self, folders: List[Path]):
        """Run Demucs separation on all tracks."""
        # Check if we should use existing stems
        if self.config.stems_source:
            logger.info(f"Using pre-existing stems from: {self.config.stems_source}")
            self._link_existing_stems(folders)
            return

        compile_info = f", compile={self.config.demucs_compile}" if self.config.demucs_compile else ""
        impl_name = "optimized" if self.config.demucs_optimized else "subprocess"
        logger.info(f"Separating stems (model={self.config.demucs_model}, shifts={self.config.demucs_shifts}, "
                    f"segment={self.config.demucs_segment}, impl={impl_name}{compile_info})...")

        try:
            if self.config.demucs_optimized:
                # Use optimized implementation (model persistence, SDPA, optional torch.compile)
                from preprocessing.demucs_sep_optimized import DemucsSeparator, batch_process

                separator = DemucsSeparator(
                    model_name=self.config.demucs_model,
                    device=self.config.device,
                    shifts=self.config.demucs_shifts,
                    segment=self.config.demucs_segment,
                    jobs=self.config.demucs_jobs,
                    use_compile=self.config.demucs_compile,
                    compile_mode=self.config.demucs_compile_mode,
                )
                success_count = batch_process(
                    self.working_dir,
                    separator,
                    overwrite=self.config.overwrite,
                    output_format=self.config.demucs_format,
                )
                self.stats.tracks_separated = success_count
            else:
                # Use subprocess-based implementation (fallback)
                from preprocessing.demucs_sep import batch_separate_stems

                stats = batch_separate_stems(
                    self.working_dir,
                    model=self.config.demucs_model,
                    device=self.config.device,
                    shifts=self.config.demucs_shifts,
                    segment=self.config.demucs_segment,
                    overwrite=self.config.overwrite,
                )
                self.stats.tracks_separated = stats.get('success', 0)
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

                if target_path.exists() and not self.config.overwrite:
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
                overwrite=self.config.overwrite,
            )
            logger.info(f"Rhythm analysis: {stats.get('success', 0)}/{len(folders)}")
        except Exception as e:
            logger.error(f"Rhythm analysis failed: {e}")
            self.stats.errors.append(f"Rhythm: {e}")

    def _run_metadata_lookup(self, folders: List[Path]):
        """Look up metadata, with fingerprinting for Various Artists tracks."""
        from core.file_utils import get_stem_files

        # Check which tracks need metadata lookup
        tracks_needing_lookup = []

        for folder in folders:
            folder_name = folder.name
            artist = self._extract_artist_from_name(folder_name)

            # Check if artist is a "Various Artists" alias
            if artist.lower() in VARIOUS_ARTISTS_ALIASES:
                tracks_needing_lookup.append((folder, 'various_artists'))
            else:
                # Check if already has metadata
                stems = get_stem_files(folder, include_full_mix=True)
                if 'full_mix' in stems:
                    info_path = get_info_path(stems['full_mix'])
                    if info_path.exists():
                        info = read_info(info_path)
                        if 'release_year' not in info:
                            tracks_needing_lookup.append((folder, 'missing_year'))
                    else:
                        tracks_needing_lookup.append((folder, 'no_info'))

        if not tracks_needing_lookup:
            logger.info("All tracks have metadata")
            return

        logger.info(f"Looking up metadata for {len(tracks_needing_lookup)} tracks...")

        # Try to use track_metadata_lookup
        try:
            from tools.track_metadata_lookup import init_spotify, lookup_track, search_musicbrainz

            sp = init_spotify()

            for folder, reason in tracks_needing_lookup:
                folder_name = folder.name
                artist, _, track_name = self._parse_folder_name(folder_name)

                if reason == 'various_artists':
                    # Try fingerprinting first
                    result = self._fingerprint_track(folder)
                    if not result:
                        # Fall back to text search
                        result = lookup_track(track_name, artist_hint=None, sp=sp,
                                              fetch_audio_features_flag=True)
                else:
                    result = lookup_track(track_name, artist_hint=artist, sp=sp,
                                          fetch_audio_features_flag=True)

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
        """Extract features that will be migrated to crops."""
        from core.file_utils import get_stem_files

        # Features: loudness, BPM (already done via rhythm), spectral basics
        try:
            from timbral.loudness import analyze_file_loudness
            from spectral.spectral_features import analyze_spectral_features

            for i, folder in enumerate(folders, 1):
                logger.info(f"[{i}/{len(folders)}] {folder.name}")

                stems = get_stem_files(folder, include_full_mix=True)
                if 'full_mix' not in stems:
                    continue

                full_mix = stems['full_mix']
                info_path = get_info_path(full_mix)
                existing = read_info(info_path) if info_path.exists() else {}
                results = {}

                # Loudness
                if self.config.overwrite or 'lufs' not in existing:
                    try:
                        results.update(analyze_file_loudness(full_mix))
                    except Exception as e:
                        logger.debug(f"Loudness failed: {e}")

                # Spectral
                if self.config.overwrite or 'spectral_flatness' not in existing:
                    try:
                        results.update(analyze_spectral_features(full_mix))
                    except Exception as e:
                        logger.debug(f"Spectral failed: {e}")

                if results:
                    safe_update(info_path, results)

                self.stats.tracks_analyzed += 1

        except ImportError as e:
            logger.warning(f"First-stage features not available: {e}")

    def _run_cropping(self):
        """Stage 3: Create crops from full tracks."""
        logger.info("\n" + "=" * 70)
        logger.info("[STAGE 3] CROPPING")
        logger.info("=" * 70)

        start_time = time.time()

        # Determine crops output directory
        crops_dir = self.working_dir.parent / f"{self.working_dir.name}_crops"

        logger.info(f"Creating crops in: {crops_dir}")
        logger.info(f"Mode: {self.config.crop_mode}")
        logger.info(f"Length: {self.config.crop_length_samples} samples (~{self.config.crop_length_samples / 44100:.1f}s at 44.1kHz)")

        try:
            from tools.create_training_crops import process_folder as create_crops_folder
            from core.file_utils import find_organized_folders

            folders = find_organized_folders(self.working_dir)
            crops_dir.mkdir(parents=True, exist_ok=True)

            for i, folder in enumerate(folders, 1):
                logger.info(f"[{i}/{len(folders)}] Creating crops: {folder.name}")
                try:
                    count = create_crops_folder(
                        folder,
                        length_samples=self.config.crop_length_samples,
                        output_dir=crops_dir,
                        sequential=self.config.crop_mode == 'sequential',
                        overlap=self.config.crop_overlap,
                        div4=self.config.crop_div4,
                    )
                    self.stats.crops_created += count
                except Exception as e:
                    logger.warning(f"  Failed: {e}")

        except Exception as e:
            logger.error(f"Cropping failed: {e}")
            self.stats.errors.append(f"Cropping: {e}")

        self.stats.time_cropping = time.time() - start_time

        # Update working dir for crop analysis
        self.crops_dir = crops_dir

    def _run_crop_analysis(self):
        """Stage 4: Analyze all crops."""
        logger.info("\n" + "=" * 70)
        logger.info("[STAGE 4] CROP ANALYSIS")
        logger.info("=" * 70)

        start_time = time.time()

        # Find crops directory
        crops_dir = getattr(self, 'crops_dir', None)
        if not crops_dir:
            crops_dir = self.working_dir.parent / f"{self.working_dir.name}_crops"

        if not crops_dir.exists():
            logger.warning(f"Crops directory not found: {crops_dir}")
            return

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
            self._run_crop_demucs(all_crops)
        elif self.config.stems_source:
            # Crop stems from source
            logger.info("\n[4a] Cropping stems from source")
            self._crop_stems_from_source(crops_dir)

        # Run feature extraction on crops
        logger.info("\n[4b] Crop Feature Extraction")
        self._run_crop_features(crops_dir)

        self.stats.time_crop_analysis = time.time() - start_time

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
                overwrite=self.config.overwrite,
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
                batch=True,
                crops=True,
                overwrite=self.config.overwrite,
                skip_organize=True,
                skip_demucs=True,  # Already done
                skip_flamingo=self.config.skip_flamingo,
                skip_audiobox=self.config.skip_audiobox,
                skip_midi=True,
                flamingo_model=self.config.flamingo_model,
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
        """Print pipeline summary."""
        print("\n" + "=" * 70)
        print("MASTER PIPELINE SUMMARY")
        print("=" * 70)

        if not self.config.skip_organize:
            print(f"\n[Stage 1] Organization")
            print(f"  Files found:     {self.stats.files_found}")
            print(f"  Files organized: {self.stats.files_organized}")
            print(f"  Time:            {self.stats.time_organize:.1f}s")

        if not self.config.skip_track_analysis:
            print(f"\n[Stage 2] Track Analysis")
            print(f"  Tracks total:    {self.stats.tracks_total}")
            print(f"  Stems separated: {self.stats.tracks_separated}")
            print(f"  Metadata found:  {self.stats.tracks_metadata_found}")
            print(f"  Analyzed:        {self.stats.tracks_analyzed}")
            print(f"  Time:            {self.stats.time_track_analysis:.1f}s")

        if not self.config.skip_crops:
            print(f"\n[Stage 3] Cropping")
            print(f"  Crops created:   {self.stats.crops_created}")
            print(f"  Time:            {self.stats.time_cropping:.1f}s")

        if not self.config.skip_crop_analysis:
            print(f"\n[Stage 4] Crop Analysis")
            print(f"  Crops analyzed:  {self.stats.crops_analyzed}")
            print(f"  Time:            {self.stats.time_crop_analysis:.1f}s")

        print(f"\nTotal Time: {total_time:.1f}s ({total_time/60:.1f} min)")

        if self.stats.errors:
            print(f"\nErrors ({len(self.stats.errors)}):")
            for err in self.stats.errors[:5]:
                print(f"  - {err}")
            if len(self.stats.errors) > 5:
                print(f"  ... and {len(self.stats.errors) - 5} more")

        print("=" * 70)


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
    parser.add_argument('--demucs-subprocess', action='store_true',
                        help='Use subprocess implementation instead of optimized (fallback)')
    parser.add_argument('--demucs-compile', action='store_true',
                        help='Use torch.compile for faster processing (ROCm optimized)')
    parser.add_argument('--demucs-compile-mode', type=str,
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode (default: reduce-overhead for ROCm)')

    # Cropping (CLI overrides config)
    parser.add_argument('--crop-length', type=int)
    parser.add_argument('--crop-mode', choices=['sequential', 'beat-aligned'])
    parser.add_argument('--crop-overlap', action='store_true')
    parser.add_argument('--crop-div4', action='store_true')

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
    if args.demucs_subprocess:
        config.demucs_optimized = False
    if args.demucs_compile:
        config.demucs_compile = True
    if args.demucs_compile_mode:
        config.demucs_compile_mode = args.demucs_compile_mode
    if args.crop_length is not None:
        config.crop_length_samples = args.crop_length
    if args.crop_mode:
        config.crop_mode = args.crop_mode
    if args.crop_overlap:
        config.crop_overlap = True
    if args.crop_div4:
        config.crop_div4 = True
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

    setup_logging(level=logging.DEBUG if config.verbose else logging.INFO)

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
