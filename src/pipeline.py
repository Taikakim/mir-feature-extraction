#!/usr/bin/env python3
"""
MIR Feature Extraction Pipeline

Master orchestrator script that runs all feature extraction steps in sequence.
Supports --output-dir to organize files to a destination before processing.

Usage:
    # Process organized folders (full_mix.flac structure)
    python src/pipeline.py /path/to/audio --batch

    # Process crop files (TrackName_0.flac structure)
    python src/pipeline.py /path/to/crops --batch --crops

    # Copy to output directory first, then process there
    python src/pipeline.py /path/to/audio --output-dir /path/to/output --batch

    # Skip specific modules
    python src/pipeline.py /path/to/audio --batch --skip-demucs --skip-flamingo
"""

import argparse
import logging
import multiprocessing
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

# Use 'spawn' for multiprocessing - required for TensorFlow compatibility
# 'fork' causes deadlocks with TensorFlow's thread pools
try:
    multiprocessing.set_start_method('spawn', force=False)
except RuntimeError:
    pass  # Already set

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.common import setup_logging
from src.core.file_utils import find_organized_folders, find_crop_files, find_crop_folders, get_crop_stem_files
from src.core.json_handler import get_crop_info_path, safe_update, read_info

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the pipeline run."""
    input_dir: Path
    output_dir: Optional[Path] = None
    device: str = "cuda"
    batch: bool = False
    batch_feature_extraction: bool = True  # New flag for batched feature extraction per module
    feature_workers: int = 8  # Number of parallel workers for CPU features
    essentia_workers: int = 4  # Separate limit for Essentia (TensorFlow can't handle high parallelism)
    overwrite: bool = False
    verbose: bool = False
    crops: bool = False  # Process crop files instead of organized folders

    # Skip flags
    skip_organize: bool = False
    skip_demucs: bool = False
    skip_rhythm: bool = False
    skip_loudness: bool = False
    skip_spectral: bool = False
    skip_saturation: bool = False
    skip_harmonic: bool = False
    skip_timbral: bool = False
    skip_classification: bool = False
    skip_per_stem: bool = False
    skip_flamingo: bool = False
    skip_audiobox: bool = False
    skip_midi: bool = False

    # Essentia sub-feature flags
    essentia_genre: bool = True
    essentia_mood: bool = True
    essentia_instrument: bool = True
    essentia_voice: bool = True
    essentia_gender: bool = True
    vocal_content_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'crest_factor_threshold': 30.0,
        'rms_threshold': -42.0,
        'peak_threshold': -20.0,
    })

    # Per-feature overwrite (from master pipeline YAML overwrite section)
    per_feature_overwrite: Dict[str, bool] = field(default_factory=dict)

    # Flamingo options
    flamingo_model: str = "Q8_0"
    flamingo_context_size: int = 1024  # LLM context window size
    flamingo_token_limits: Dict[str, int] = field(default_factory=dict)
    flamingo_prompts: Dict[str, str] = field(default_factory=dict)
    flamingo_revision: Dict[str, Any] = field(default_factory=dict)
    flamingo_sample_probability: float = 1.0  # Fraction of unprocessed crops to annotate per run (0-1)

    # Pipeline state for resumable runs (passed from master pipeline)
    pipeline_state: Any = None

    # Map internal feature names to YAML overwrite keys
    _OVERWRITE_ALIASES = {
        'harmonic': 'chroma',
        'rhythm': 'bpm',
        'flamingo': 'music_flamingo',
    }

    def should_overwrite(self, feature: str) -> bool:
        """Check if a specific feature should be overwritten."""
        if self.overwrite:
            return True
        if self.per_feature_overwrite.get(feature, False):
            return True
        # Check YAML alias (e.g. 'harmonic' → 'chroma')
        alias = self._OVERWRITE_ALIASES.get(feature)
        if alias and self.per_feature_overwrite.get(alias, False):
            return True
        return False

    @property
    def working_dir(self) -> Path:
        """Directory where processing happens."""
        return self.output_dir if self.output_dir else self.input_dir


def _interpolate_genres(prompt_text: str, existing: Dict[str, Any]) -> str:
    """Replace {genres} and {metadata} placeholders in prompt with .INFO data.

    {genres} → weighted Essentia genre distribution, e.g.:
        "Probabilistic genre analysis reveals: Electronic - Goa Trance (0.39), ... "
        "Numbers are softmax weights (0–1); very low probabilities are filtered out."

    {metadata} → ID3/release fields (year, label, tag genres), e.g.:
        "According to the actual ID3 metadata, this is the release year, label and
         genres of the track the clip is sourced from: release year: 2021;
         label: DAT Universe; genres: psytrance, trance, acid techno."
        Omitted silently if none of the fields are present in the .INFO file.
    """
    if '{genres}' in prompt_text:
        genre_dict = existing.get('essentia_genre', {})
        if genre_dict and isinstance(genre_dict, dict):
            top = sorted(genre_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            entries = ', '.join(
                f"{k.replace('---', ' - ')} ({v:.2f})"
                for k, v in top
            )
            genres_str = (
                f"Probabilistic genre analysis reveals: {entries}. "
                "Numbers are softmax weights (0\u20131); very low probabilities are filtered out."
            )
        else:
            genres_str = 'Genre unknown.'
        prompt_text = prompt_text.replace('{genres}', genres_str)

    if '{metadata}' in prompt_text:
        parts = []
        year = existing.get('release_year') or existing.get('track_metadata_year')
        if year:
            parts.append(f"release year: {year}")
        label = existing.get('label')
        if label:
            parts.append(f"label: {label}")
        tag_genres = existing.get('genres', [])
        if tag_genres and isinstance(tag_genres, list):
            parts.append(f"genres: {', '.join(str(g) for g in tag_genres)}")
        if parts:
            metadata_str = (
                "According to the actual ID3 metadata, this is the release year, label and "
                f"genres of the track the clip is sourced from: {'; '.join(parts)}."
            )
        else:
            metadata_str = ''
        prompt_text = prompt_text.replace('{metadata}', metadata_str)

    return prompt_text


def _safe_analyze_cpu(args) -> Dict[str, Any]:
    """
    Worker function for parallel CPU feature extraction.
    Must be top-level for pickling.

    Pre-loads crop audio and stems into RAM once, then passes pre-loaded arrays
    to all feature functions to avoid redundant disk reads (was 17 reads per crop,
    now 1-5 depending on stems).

    Args is a tuple of (crop_path, config_dict, existing_keys) where existing_keys
    is a set of keys already present in the .INFO file.
    """
    crop_path, config_dict, existing_keys = args
    results = {}
    overwrite = config_dict.get('overwrite', False)
    per_feature_ow = config_dict.get('per_feature_overwrite', {})

    # Define all output keys for each feature type (check ALL, not just one)
    LOUDNESS_KEYS = ['lufs', 'lra']
    SPECTRAL_KEYS = ['spectral_flatness', 'spectral_flux', 'spectral_skewness', 'spectral_kurtosis']
    SATURATION_KEYS = ['saturation_ratio', 'saturation_count']
    MULTIBAND_KEYS = ['rms_energy_bass', 'rms_energy_body', 'rms_energy_mid', 'rms_energy_air']
    CHROMA_KEYS = [f'chroma_{i}' for i in range(12)]
    TIMBRAL_KEYS = ['brightness', 'roughness', 'hardness', 'depth',
                   'booming', 'reverberation', 'sharpness', 'warmth']

    _ow_aliases = {'harmonic': 'chroma', 'rhythm': 'bpm', 'flamingo': 'music_flamingo'}

    def _needs_processing(keys, feature_name=None):
        """Check if ANY of the keys are missing (needs processing)."""
        feat_ow = overwrite
        if not feat_ow and feature_name:
            feat_ow = per_feature_ow.get(feature_name, False)
            if not feat_ow:
                alias = _ow_aliases.get(feature_name)
                feat_ow = bool(alias and per_feature_ow.get(alias, False))
        return feat_ow or any(k not in existing_keys for k in keys)

    try:
        generated = []   # e.g. ['loudness(new)', 'spectral(overwrite)', ...]

        # Re-import locally for worker process
        from src.timbral.loudness import analyze_file_loudness
        from src.spectral.spectral_features import analyze_spectral_features
        from src.spectral.saturation import analyze_saturation
        from src.spectral.multiband_rms import analyze_multiband_rms
        from src.harmonic.chroma import analyze_chroma
        from src.core.file_utils import get_crop_stem_files, read_audio
        import librosa
        import soundfile as sf
        import numpy as np

        # Optional imports
        timbral_func = None
        if not config_dict.get('skip_timbral'):
            try:
                from src.timbral.audio_commons import analyze_all_timbral_features
                timbral_func = analyze_all_timbral_features
            except ImportError: pass

        # =====================================================================
        # PRE-LOAD: Read crop audio ONCE from disk (eliminates ~12 redundant reads)
        # =====================================================================
        crop_audio, crop_sr = read_audio(str(crop_path))

        # Mono version for most features
        if crop_audio.ndim > 1:
            crop_mono = crop_audio.mean(axis=1)
        else:
            crop_mono = crop_audio

        # Pre-load stems into RAM (1 read each instead of repeated per-feature)
        stems = get_crop_stem_files(crop_path)
        preloaded_stems = {}  # stem_name -> (audio_array, sr)
        for stem_name in ['drums', 'bass', 'other', 'vocals']:
            if stem_name in stems:
                try:
                    s_audio, s_sr = read_audio(str(stems[stem_name]))
                    preloaded_stems[stem_name] = (s_audio, s_sr)
                except Exception:
                    pass

        # =====================================================================
        # FEATURE EXTRACTION: All functions use pre-loaded arrays
        # =====================================================================

        # Loudness - check ALL output keys
        STEM_LOUDNESS_KEYS = [f'lufs_{s}' for s in preloaded_stems]
        if not config_dict.get('skip_loudness'):
            if _needs_processing(LOUDNESS_KEYS, 'loudness'):
                _op = 'overwrite' if any(k in existing_keys for k in LOUDNESS_KEYS) else 'new'
                try:
                    results.update(analyze_file_loudness(crop_path, audio=crop_audio, sr=crop_sr))
                    for stem_name, (s_audio, s_sr) in preloaded_stems.items():
                        stem_loud = analyze_file_loudness(
                            stems[stem_name], audio=s_audio, sr=s_sr)
                        results[f'lufs_{stem_name}'] = stem_loud.get('lufs')
                    generated.append(f'loudness({_op})')
                except Exception: pass
            elif preloaded_stems:
                # Main loudness done, but check if any stem loudness is missing
                _stem_new = False
                for stem_name in preloaded_stems:
                    if f'lufs_{stem_name}' not in existing_keys:
                        try:
                            s_audio, s_sr = preloaded_stems[stem_name]
                            stem_loud = analyze_file_loudness(
                                stems[stem_name], audio=s_audio, sr=s_sr)
                            results[f'lufs_{stem_name}'] = stem_loud.get('lufs')
                            _stem_new = True
                        except Exception: pass
                if _stem_new:
                    generated.append('loudness-stems(new)')

        # BPM - use pre-loaded mono audio
        if not config_dict.get('skip_rhythm'):
            if overwrite or per_feature_ow.get('rhythm', False) or 'bpm' not in existing_keys:
                _op = 'overwrite' if 'bpm' in existing_keys else 'new'
                try:
                    tempo, _ = librosa.beat.beat_track(y=crop_mono, sr=crop_sr)
                    bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
                    results['bpm'] = round(bpm, 1)
                    generated.append(f'rhythm({_op})')
                except Exception: pass

        # Spectral - pass pre-loaded mono audio
        if not config_dict.get('skip_spectral'):
            if _needs_processing(SPECTRAL_KEYS, 'spectral'):
                _op = 'overwrite' if any(k in existing_keys for k in SPECTRAL_KEYS) else 'new'
                try:
                    results.update(analyze_spectral_features(
                        crop_path, audio=crop_mono, sr=crop_sr))
                    generated.append(f'spectral({_op})')
                except Exception: pass
            if _needs_processing(MULTIBAND_KEYS, 'spectral'):
                _op = 'overwrite' if any(k in existing_keys for k in MULTIBAND_KEYS) else 'new'
                try:
                    results.update(analyze_multiband_rms(
                        crop_path, audio=crop_mono, sr=crop_sr))
                    generated.append(f'rms({_op})')
                except Exception: pass

        # Saturation / hard-clipping detection
        if not config_dict.get('skip_saturation'):
            if _needs_processing(SATURATION_KEYS, 'saturation'):
                _op = 'overwrite' if any(k in existing_keys for k in SATURATION_KEYS) else 'new'
                try:
                    results.update(analyze_saturation(
                        crop_path, audio=crop_mono, sr=crop_sr))
                    generated.append(f'sat({_op})')
                except Exception: pass

        # Chroma - use 'other' stem if available (pre-loaded), else crop mono
        if not config_dict.get('skip_harmonic'):
            if _needs_processing(CHROMA_KEYS, 'harmonic'):
                _op = 'overwrite' if any(k in existing_keys for k in CHROMA_KEYS) else 'new'
                try:
                    if 'other' in preloaded_stems:
                        other_audio, other_sr = preloaded_stems['other']
                        if other_audio.ndim > 1:
                            other_audio = other_audio.mean(axis=1)
                        results.update(analyze_chroma(
                            stems['other'], use_stems=False,
                            audio=other_audio.astype(np.float32), sr=other_sr))
                    else:
                        results.update(analyze_chroma(
                            crop_path, use_stems=False,
                            audio=crop_mono.astype(np.float32), sr=crop_sr))
                    generated.append(f'chroma({_op})')
                except Exception: pass

        # Timbral - pass pre-loaded audio (writes to /dev/shm RAM disk internally)
        # Compute only the specific features that are actually missing — avoids
        # running timbral_reverb (the hang-prone one) when reverberation already exists.
        if timbral_func and not config_dict.get('skip_timbral'):
            _timbral_ow = overwrite or per_feature_ow.get('timbral', False)
            _timbral_missing = [k for k in TIMBRAL_KEYS
                                if _timbral_ow or k not in existing_keys]
            if _timbral_missing:
                _op = 'overwrite' if any(k in existing_keys for k in _timbral_missing) else 'new'
                try:
                    results.update(timbral_func(
                        crop_path, audio=crop_audio, sr=crop_sr,
                        features=_timbral_missing))
                    generated.append(f'timbral({_op})')
                except Exception: pass

        return {'path': crop_path, 'results': results, 'success': True,
                'generated': generated}

    except Exception as e:
        return {'path': crop_path, 'error': str(e), 'success': False}


# Global counter for staggered Essentia worker initialization
_essentia_worker_id = None


def _init_essentia_worker(worker_id: int, stagger_delay: float = 1.0):
    """
    Initializer for Essentia workers that staggers TensorFlow loading.

    TensorFlow deadlocks when multiple processes try to load the same
    graph file simultaneously. This initializer adds a staggered delay
    to ensure orderly TensorFlow initialization.

    Args:
        worker_id: Unique ID for this worker (0, 1, 2, ...)
        stagger_delay: Seconds to wait per worker_id
    """
    global _essentia_worker_id
    _essentia_worker_id = worker_id

    # Stagger TensorFlow initialization
    delay = worker_id * stagger_delay
    if delay > 0:
        time.sleep(delay)

    # Pre-load TensorFlow and Essentia to avoid concurrent loading in workers
    try:
        from src.classification.essentia_features import analyze_essentia_features
        # Force TensorFlow initialization by importing the predictor
        import essentia.standard as es
        # This triggers TensorFlow loading for the danceability model
        logger.debug(f"Essentia worker {worker_id} initialized after {delay:.1f}s delay")
    except Exception as e:
        logger.warning(f"Essentia worker {worker_id} init warning: {e}")


def _safe_analyze_essentia(args) -> Dict[str, Any]:
    """
    Worker function for parallel Essentia feature extraction.
    Must be top-level for pickling.
    Args: tuple of (crop_path, kwargs_dict) or just crop_path for backwards compat.
    """
    try:
        from src.classification.essentia_features import analyze_essentia_features
        if isinstance(args, tuple):
            crop_path, kwargs = args
        else:
            crop_path, kwargs = args, {}
        results = analyze_essentia_features(crop_path, **kwargs)
        return {'path': crop_path, 'results': results, 'success': True}
    except Exception as e:
        crop_path = args[0] if isinstance(args, tuple) else args
        return {'path': crop_path, 'error': str(e), 'success': False}


class Pipeline:
    """MIR Feature Extraction Pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stats = {
            "steps_completed": 0,
            "steps_failed": 0,
            "steps_skipped": 0,
            "crops_processed": 0,
            "crops_failed": 0,
            "total_time": 0.0
        }

    def run_step(self, name: str, script: str, args: List[str],
                 skip_flag: bool = False) -> bool:
        """Run a single pipeline step."""
        if skip_flag:
            logger.info(f"⏭️  Skipping: {name}")
            self.stats["steps_skipped"] += 1
            return True
            
        logger.info(f"▶️  Running: {name}")
        start_time = time.time()
        
        # Build command
        cmd = [sys.executable, str(PROJECT_ROOT / script)] + args
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=not self.config.verbose,
                text=True
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ Completed: {name} ({elapsed:.1f}s)")
                self.stats["steps_completed"] += 1
                return True
            else:
                logger.error(f"❌ Failed: {name}")
                if not self.config.verbose and result.stderr:
                    logger.error(result.stderr[:500])
                self.stats["steps_failed"] += 1
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running {name}: {e}")
            self.stats["steps_failed"] += 1
            return False
            
    def run(self) -> bool:
        """Run the full pipeline."""
        # Use crops mode if --crops flag is set
        if self.config.crops:
            return self.run_crops()

        start_time = time.time()
        working_dir = str(self.config.working_dir)

        logger.info("=" * 60)
        logger.info("MIR FEATURE EXTRACTION PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Input:  {self.config.input_dir}")
        logger.info(f"Output: {self.config.output_dir or '(in-place)'}")
        logger.info(f"Device: {self.config.device}")
        logger.info("=" * 60)
        
        # Common args
        batch_args = ["--batch"] if self.config.batch else []
        verbose_args = ["--verbose"] if self.config.verbose else []
        overwrite_args = ["--overwrite"] if self.config.overwrite else []
        
        # Step 1: Organize files
        organize_args = [str(self.config.input_dir)]
        if self.config.output_dir:
            organize_args += ["--output-dir", str(self.config.output_dir)]
        organize_args += verbose_args
        
        self.run_step(
            "Organize Files",
            "src/preprocessing/file_organizer.py",
            organize_args,
            skip_flag=self.config.skip_organize
        )
        
        # Step 2: Stem separation (Demucs)
        self.run_step(
            "Stem Separation (Demucs)",
            "src/preprocessing/demucs_sep.py",
            [working_dir] + batch_args + ["--device", self.config.device] + verbose_args + overwrite_args,
            skip_flag=self.config.skip_demucs
        )
        
        # Step 3: Rhythm analysis
        self.run_step(
            "Beat Grid Detection",
            "src/rhythm/beat_grid.py",
            [working_dir] + batch_args + verbose_args,
            skip_flag=self.config.skip_rhythm
        )
        
        self.run_step(
            "BPM Analysis",
            "src/rhythm/bpm.py",
            [working_dir] + batch_args + verbose_args,
            skip_flag=self.config.skip_rhythm
        )
        
        self.run_step(
            "Onset Detection",
            "src/rhythm/onsets.py",
            [working_dir] + batch_args + verbose_args,
            skip_flag=self.config.skip_rhythm
        )
        
        # Step 4: Loudness
        self.run_step(
            "Loudness Analysis",
            "src/preprocessing/loudness.py",
            [working_dir] + batch_args + verbose_args,
            skip_flag=self.config.skip_loudness
        )
        
        # Step 5: Spectral features
        self.run_step(
            "Spectral Features",
            "src/spectral/spectral_features.py",
            [working_dir] + batch_args + verbose_args,
            skip_flag=self.config.skip_spectral
        )
        
        self.run_step(
            "Multiband RMS Energy",
            "src/spectral/multiband_rms.py",
            [working_dir] + batch_args + verbose_args,
            skip_flag=self.config.skip_spectral
        )
        
        # Step 6: Harmonic features
        self.run_step(
            "Chroma Features",
            "src/harmonic/chroma.py",
            [working_dir] + batch_args + verbose_args,
            skip_flag=self.config.skip_harmonic
        )
        
        self.run_step(
            "Per-Stem Harmonic",
            "src/harmonic/per_stem_harmonic.py",
            [working_dir] + batch_args + verbose_args,
            skip_flag=self.config.skip_harmonic
        )
        
        # Step 7: Timbral features
        self.run_step(
            "Audio Commons Timbral",
            "src/timbral/audio_commons.py",
            [working_dir] + batch_args + verbose_args,
            skip_flag=self.config.skip_timbral
        )
        
        # Step 8: Classification
        self.run_step(
            "Essentia Classification",
            "src/classification/essentia_features.py",
            [working_dir] + batch_args + verbose_args,
            skip_flag=self.config.skip_classification
        )
        
        # Step 9: Per-stem rhythm
        self.run_step(
            "Per-Stem Rhythm",
            "src/rhythm/per_stem_rhythm.py",
            [working_dir] + batch_args + verbose_args,
            skip_flag=self.config.skip_per_stem
        )
        
        # Step 10: Music Flamingo (optional, heavy)
        self.run_step(
            "Music Flamingo AI Descriptions",
            "src/classification/music_flamingo.py",
            [working_dir] + batch_args + ["--model", self.config.flamingo_model] + verbose_args,
            skip_flag=self.config.skip_flamingo
        )
        
        # Step 11: MIDI transcription (optional)
        self.run_step(
            "MIDI Drum Transcription",
            "src/transcription/drums/adtof.py",
            [working_dir] + (["--batch"] if self.config.batch else []) + 
            ["--device", self.config.device] + verbose_args,
            skip_flag=self.config.skip_midi
        )
        
        # Summary
        self.stats["total_time"] = time.time() - start_time
        self._print_summary()
        
        return self.stats["steps_failed"] == 0
        
    def run_crops(self) -> bool:
        """Run pipeline in crops mode - process crop files directly."""
        start_time = time.time()
        working_dir = self.config.working_dir

        logger.info("=" * 60)
        logger.info("MIR FEATURE EXTRACTION PIPELINE (CROPS MODE)")
        logger.info("=" * 60)
        logger.info(f"Input:  {self.config.input_dir}")
        logger.info(f"Device: {self.config.device}")
        logger.info("=" * 60)

        # Find crop files or folders
        if self.config.batch:
            crop_folders = find_crop_folders(working_dir)
            all_crops = []
            for folder in crop_folders:
                all_crops.extend(find_crop_files(folder))
            logger.info(
                f"Found {len(crop_folders)} crop folders and "
                f"{len(all_crops):,} crops to analyze in {working_dir}")
        else:
            all_crops = find_crop_files(working_dir)

        if not all_crops:
            logger.warning("No crop files found")
            return True

        # Use batched processing if enabled (more efficient for GPU models)
        if self.config.batch_feature_extraction:
            return self._run_crops_batched(all_crops)
        
        # Otherwise use original sequential processing
        logger.info(f"Found {len(all_crops)} crop files to process (Sequential Mode)")

        # Import feature extraction functions
        from src.timbral.loudness import analyze_file_loudness
        from src.spectral.spectral_features import analyze_spectral_features
        from src.spectral.multiband_rms import analyze_multiband_rms
        from src.harmonic.chroma import analyze_chroma
        from src.core.file_utils import read_audio
        import librosa
        import soundfile as sf

        # Optional imports
        timbral_func = None
        if not self.config.skip_timbral:
            try:
                from src.timbral.audio_commons import analyze_all_timbral_features
                timbral_func = analyze_all_timbral_features
            except ImportError:
                logger.warning("Audio Commons timbral not available")

        essentia_func = None
        if not self.config.skip_classification:
            try:
                from src.classification.essentia_features import analyze_essentia_features
                essentia_func = analyze_essentia_features
            except ImportError:
                logger.warning("Essentia not available")

        audiobox_func = None
        if not self.config.skip_audiobox:
            try:
                from src.timbral.audiobox_aesthetics import analyze_audiobox_aesthetics
                audiobox_func = analyze_audiobox_aesthetics
            except ImportError:
                logger.debug("AudioBox not available")

        flamingo_analyzer = None
        if not self.config.skip_flamingo:
            try:
                from classification.music_flamingo import MusicFlamingoGGUF
                logger.info(f"Loading Music Flamingo (GGUF/CLI) {self.config.flamingo_model}...")

                flamingo_analyzer = MusicFlamingoGGUF(
                    model=self.config.flamingo_model,
                )
            except Exception as e:
                logger.warning(f"Music Flamingo not available: {e}")

        import numpy as np

        # Process each crop
        for i, crop_path in enumerate(all_crops, 1):
            logger.info(f"\n[{i}/{len(all_crops)}] {crop_path.name}")

            try:
                info_path = get_crop_info_path(crop_path)
                existing = read_info(info_path) if info_path.exists() else {}
                results = {}

                # PRE-LOAD: Read crop audio ONCE from disk
                crop_audio, crop_sr = read_audio(str(crop_path))
                crop_mono = crop_audio.mean(axis=1) if crop_audio.ndim > 1 else crop_audio

                # Pre-load stems into RAM
                stems = get_crop_stem_files(crop_path)
                preloaded_stems = {}
                for stem_name in ['drums', 'bass', 'other', 'vocals']:
                    if stem_name in stems:
                        try:
                            s_audio, s_sr = read_audio(str(stems[stem_name]))
                            preloaded_stems[stem_name] = (s_audio, s_sr)
                        except Exception:
                            pass

                # Loudness
                if not self.config.skip_loudness:
                    if self.config.should_overwrite('loudness') or 'lufs' not in existing:
                        try:
                            results.update(analyze_file_loudness(
                                crop_path, audio=crop_audio, sr=crop_sr))
                            for stem_name, (s_audio, s_sr) in preloaded_stems.items():
                                stem_loud = analyze_file_loudness(
                                    stems[stem_name], audio=s_audio, sr=s_sr)
                                results[f'lufs_{stem_name}'] = stem_loud.get('lufs')
                                results[f'lra_{stem_name}'] = stem_loud.get('lra')
                            logger.info("  ✓ Loudness")
                        except Exception as e:
                            logger.warning(f"  Loudness failed: {e}")
                    elif preloaded_stems:
                        # Main loudness done, but check if any stem loudness is missing
                        for stem_name in preloaded_stems:
                            if f'lufs_{stem_name}' not in existing:
                                try:
                                    s_audio, s_sr = preloaded_stems[stem_name]
                                    stem_loud = analyze_file_loudness(
                                        stems[stem_name], audio=s_audio, sr=s_sr)
                                    results[f'lufs_{stem_name}'] = stem_loud.get('lufs')
                                    results[f'lra_{stem_name}'] = stem_loud.get('lra')
                                except Exception as e:
                                    logger.warning(f"  Stem loudness ({stem_name}) failed: {e}")
                        if any(f'lufs_{s}' in results for s in preloaded_stems):
                            logger.info("  ✓ Stem Loudness (partial)")

                # BPM (if not already in crop .INFO from cropping)
                if not self.config.skip_rhythm:
                    if self.config.should_overwrite('rhythm') or 'bpm' not in existing:
                        try:
                            tempo, _ = librosa.beat.beat_track(y=crop_mono, sr=crop_sr)
                            bpm_value = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
                            results['bpm'] = round(bpm_value, 1)
                            logger.info("  ✓ BPM")
                        except Exception as e:
                            logger.warning(f"  BPM failed: {e}")

                # Spectral
                if not self.config.skip_spectral:
                    if self.config.should_overwrite('spectral') or 'spectral_flatness' not in existing:
                        try:
                            results.update(analyze_spectral_features(
                                crop_path, audio=crop_mono, sr=crop_sr))
                            logger.info("  ✓ Spectral")
                        except Exception as e:
                            logger.warning(f"  Spectral failed: {e}")

                    if self.config.should_overwrite('spectral') or 'rms_energy_bass' not in existing:
                        try:
                            results.update(analyze_multiband_rms(
                                crop_path, audio=crop_mono, sr=crop_sr))
                            logger.info("  ✓ Multiband RMS")
                        except Exception as e:
                            logger.warning(f"  Multiband RMS failed: {e}")

                # Chroma
                if not self.config.skip_harmonic:
                    if self.config.should_overwrite('harmonic') or 'chroma_0' not in existing:
                        try:
                            if 'other' in preloaded_stems:
                                other_audio, other_sr = preloaded_stems['other']
                                other_mono = other_audio.mean(axis=1) if other_audio.ndim > 1 else other_audio
                                results.update(analyze_chroma(
                                    stems['other'], use_stems=False,
                                    audio=other_mono.astype(np.float32), sr=other_sr))
                            else:
                                results.update(analyze_chroma(
                                    crop_path, use_stems=False,
                                    audio=crop_mono.astype(np.float32), sr=crop_sr))
                            logger.info("  ✓ Chroma")
                        except Exception as e:
                            logger.warning(f"  Chroma failed: {e}")

                # Timbral (uses /dev/shm RAM disk internally when audio provided)
                if timbral_func and (self.config.should_overwrite('timbral') or 'brightness' not in existing):
                    try:
                        results.update(timbral_func(
                            crop_path, audio=crop_audio, sr=crop_sr))
                        logger.info("  ✓ Timbral")
                    except Exception as e:
                        logger.warning(f"  Timbral failed: {e}")

                # Essentia
                essentia_needed = self.config.should_overwrite('essentia') or 'danceability' not in existing
                if self.config.essentia_genre:
                    essentia_needed = essentia_needed or 'essentia_genre' not in existing
                if self.config.essentia_voice:
                    essentia_needed = essentia_needed or 'voice_probability' not in existing
                if essentia_func and essentia_needed:
                    try:
                        vocals_path = None
                        if self.config.essentia_gender:
                            stems = get_crop_stem_files(crop_path) if self.config.crops else get_stem_files(
                                crop_path.parent, include_full_mix=False)
                            vocals_path = stems.get('vocals')

                        include_gmi = any([self.config.essentia_genre,
                                           self.config.essentia_mood,
                                           self.config.essentia_instrument])
                        results.update(essentia_func(
                            crop_path,
                            include_voice_analysis=self.config.essentia_voice,
                            include_gender=self.config.essentia_gender,
                            include_gmi=include_gmi,
                            include_genre=self.config.essentia_genre,
                            include_mood=self.config.essentia_mood,
                            include_instrument=self.config.essentia_instrument,
                            vocals_path=vocals_path,
                            vocal_content_thresholds=self.config.vocal_content_thresholds))
                        logger.info("  ✓ Essentia")
                    except Exception as e:
                        logger.warning(f"  Essentia failed: {e}")

                # AudioBox
                if audiobox_func and (self.config.should_overwrite('audiobox') or 'content_enjoyment' not in existing):
                    try:
                        results.update(audiobox_func(crop_path))
                        logger.info("  ✓ AudioBox")
                    except Exception as e:
                        logger.warning(f"  AudioBox failed: {e}")

                # Music Flamingo
                if flamingo_analyzer and (self.config.should_overwrite('flamingo') or 'music_flamingo_model' not in existing):
                    try:
                        # Determine active prompts map
                        prompts_map = self.config.flamingo_prompts
                        if not prompts_map:
                            from classification.music_flamingo import DEFAULT_PROMPTS
                            prompts_map = DEFAULT_PROMPTS

                        for prompt_type, prompt_text in prompts_map.items():
                            key = f'music_flamingo_{prompt_type}'
                            if self.config.should_overwrite('flamingo') or key not in existing:
                                # Interpolate genre hints into prompt
                                p_text = _interpolate_genres(prompt_text, existing) if self.config.flamingo_prompts else None
                                results[key] = flamingo_analyzer.analyze(
                                    crop_path,
                                    prompt=p_text,
                                    prompt_type=prompt_type
                                )

                        if any(k.startswith('music_flamingo_') for k in results):
                            results['music_flamingo_model'] = f'gguf_{self.config.flamingo_model}'
                            logger.info("  ✓ Music Flamingo")

                        # Granite revision (per-file path)
                        rev_cfg = self.config.flamingo_revision
                        if rev_cfg.get('enabled') and rev_cfg.get('prompts'):
                            mf_results = {k: v for k, v in {**existing, **results}.items()
                                          if k.startswith('music_flamingo_') and isinstance(v, str)}
                            rev_keys = rev_cfg['prompts']
                            need_rev = any(
                                f'music_flamingo_{rk}' not in existing or self.config.should_overwrite('flamingo')
                                for rk in rev_keys
                            )
                            if mf_results and need_rev:
                                try:
                                    from classification.granite_revision import GraniteReviser
                                    reviser = GraniteReviser(
                                        rev_cfg['model'],
                                        n_ctx=rev_cfg.get('n_ctx', 4096),
                                        temperature=rev_cfg.get('temperature', 0.7),
                                        max_tokens=rev_cfg.get('max_tokens', 512),
                                    )
                                    rev_results = reviser.revise(mf_results, rev_keys)
                                    for rk, rv in rev_results.items():
                                        results[f'music_flamingo_{rk}'] = rv
                                    reviser.close()
                                except Exception as e:
                                    logger.warning(f"  Granite revision failed: {e}")

                    except Exception as e:
                        logger.warning(f"  Music Flamingo failed: {e}")

                # Save results
                if results:
                    safe_update(info_path, results)
                    logger.info(f"  Saved {len(results)} features to {info_path.name}")

                self.stats["crops_processed"] += 1

            except Exception as e:
                logger.error(f"  Failed: {e}")
                self.stats["crops_failed"] += 1

        self.stats["total_time"] = time.time() - start_time

        # Show flamingo performance stats if used
        if flamingo_analyzer and flamingo_analyzer.stats.runs > 0:
            logger.info(flamingo_analyzer.stats.summary())

        self._print_crops_summary()

        return self.stats["crops_failed"] == 0

    def _print_crops_summary(self):
        """Print crops pipeline summary."""
        print("\n" + "=" * 60)
        print("CROPS PIPELINE SUMMARY")
        print("=" * 60)
        print(f"✅ Processed: {self.stats['crops_processed']}")
        print(f"❌ Failed:    {self.stats['crops_failed']}")
        print(f"⏱️  Total Time: {self.stats['total_time']:.1f}s ({self.stats['total_time']/60:.1f} min)")
        if self.stats['crops_processed'] > 0:
            rate = self.stats['total_time'] / self.stats['crops_processed']
            print(f"📊 Rate:      {rate:.2f}s per crop")
        print("=" * 60)

    def _print_summary(self):
        """Print pipeline summary."""
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"✅ Completed: {self.stats['steps_completed']}")
        print(f"❌ Failed:    {self.stats['steps_failed']}")
        print(f"⏭️  Skipped:   {self.stats['steps_skipped']}")
        print(f"⏱️  Total Time: {self.stats['total_time']:.1f}s ({self.stats['total_time']/60:.1f} min)")
        print("=" * 60)


    def _run_crops_batched(self, all_crops: List[Path]) -> bool:
        """
        Run feature extraction in batches to optimize resource usage.

        Strategy:
        1. cpu_pass: Light features (Loudness, Spectral, Rhythm, Chroma) - One pass per file to minimize I/O.
        2. audiobox_pass: Heavy GPU - Load model once, process all.
        3. essentia_pass: Heavy GPU/CPU - Load model once, process all.
        4. flamingo_pass: Heavy GPU - Load model once, process all.
        5. midi_pass: Heavy GPU - Load model once, process all.
        """
        from src.core.graceful_shutdown import shutdown_requested, start_shutdown_listener, stop_shutdown_listener
        start_shutdown_listener()

        # Pipeline state for progress tracking
        state = self.config.pipeline_state

        logger.info("\n" + "=" * 60)
        logger.info("BATCH MODE: STARTING")
        logger.info("=" * 60)

        start_time = time.time()
        
        # Imports for CPU pass
        from src.timbral.loudness import analyze_file_loudness
        from src.spectral.spectral_features import analyze_spectral_features
        from src.spectral.multiband_rms import analyze_multiband_rms
        from src.harmonic.chroma import analyze_chroma
        import librosa
        import soundfile as sf
        
        # Optional CPU imports
        timbral_func = None
        if not self.config.skip_timbral:
            try:
                from src.timbral.audio_commons import analyze_all_timbral_features
                timbral_func = analyze_all_timbral_features
            except ImportError: pass
            
        # =========================================================================
        # PASS 1: CPU / Light Features (Parallel, runs in background thread)
        # PASS 1 is entirely CPU-based and can run concurrently with GPU passes.
        # We start it immediately and join before PASS 3 (Essentia).
        # =========================================================================
        pass1_thread = None
        pass1_exception = [None]
        state_lock = threading.Lock()  # Protects concurrent state.save() calls

        pass1_needed = not all([self.config.skip_loudness, self.config.skip_rhythm,
                                self.config.skip_spectral, self.config.skip_saturation,
                                self.config.skip_harmonic, self.config.skip_timbral])

        if pass1_needed:
            logger.info(f"\n[PASS 1/5] Light Features (CPU) - {len(all_crops)} files")
            logger.info(f"  Parallel workers: {self.config.feature_workers}")

            worker_config = {
                'skip_loudness': self.config.skip_loudness,
                'skip_rhythm': self.config.skip_rhythm,
                'skip_spectral': self.config.skip_spectral,
                'skip_saturation': self.config.skip_saturation,
                'skip_harmonic': self.config.skip_harmonic,
                'skip_timbral': self.config.skip_timbral,
                'overwrite': self.config.overwrite,
                'per_feature_overwrite': self.config.per_feature_overwrite,
            }

            # Coverage keys: ALL keys for each FEATURE_GROUPS label.
            # Checked individually — any missing key for a group means that group
            # is incomplete, regardless of whether the "sentinel" key is present.
            _COV_KEY_GROUPS = {
                'Loudness':  ['lufs', 'lra'],
                'Spectral':  ['spectral_flatness', 'spectral_flux',
                              'spectral_skewness', 'spectral_kurtosis'],
                'Rhythm':    ['bpm'],
                'Per-Stem':  ['syncopation_drums'],
                'Chroma':    [f'chroma_{i}' for i in range(12)],
                'Timbral':   ['brightness', 'roughness', 'hardness', 'depth',
                              'booming', 'reverberation', 'sharpness', 'warmth'],
                'Essentia':  ['danceability'],
                'AudioBox':  ['content_enjoyment'],
                'Flamingo':  ['music_flamingo_brief'],
                'Metadata':  ['release_year'],
            }
            # Representative single key used for coverage fraction tracking
            _COV_SENTINEL = {lbl: keys[0] for lbl, keys in _COV_KEY_GROUPS.items()}
            _cov_counts = {lbl: 0 for lbl in _COV_KEY_GROUPS}
            _n_total = len(all_crops)

            tasks = []
            for _scan_i, crop_path in enumerate(all_crops):
                info_path = get_crop_info_path(crop_path)
                existing = read_info(info_path) if info_path.exists() else {}
                existing_keys = set(existing.keys())

                # Count coverage while we have existing_keys (no extra I/O).
                # A group is "covered" only if ALL its keys are present.
                for _lbl, _keys in _COV_KEY_GROUPS.items():
                    if all(k in existing_keys for k in _keys):
                        _cov_counts[_lbl] += 1

                # Publish coverage incrementally so TUI shows % from scan start
                if _n_total > 0 and (_scan_i % 1000 == 0 or _scan_i == _n_total - 1):
                    self.stats['feature_coverage'] = {
                        lbl: cnt / _n_total for lbl, cnt in _cov_counts.items()
                    }
                    self.stats['feature_coverage_n'] = _n_total

                # Check if this crop needs ANY processing — use ALL keys per group
                # so a crop missing only one key in a group is still queued.
                needs_run = False
                if self.config.overwrite:
                    needs_run = True
                else:
                    _ow = self.config.should_overwrite
                    if not self.config.skip_loudness and (
                            _ow('loudness') or
                            any(k not in existing_keys for k in ['lufs', 'lra']) or
                            any(f'lufs_{s}' not in existing_keys
                                for s in ['drums', 'bass', 'other', 'vocals'])):
                        needs_run = True
                    if not needs_run and not self.config.skip_rhythm and (
                            _ow('rhythm') or 'bpm' not in existing_keys):
                        needs_run = True
                    if not needs_run and not self.config.skip_spectral and (
                            _ow('spectral') or
                            any(k not in existing_keys
                                for k in ['spectral_flatness', 'spectral_flux',
                                          'spectral_skewness', 'spectral_kurtosis',
                                          'rms_energy_bass', 'rms_energy_body',
                                          'rms_energy_mid', 'rms_energy_air',
                                          'saturation_ratio', 'saturation_count'])):
                        needs_run = True
                    if not needs_run and not self.config.skip_harmonic and (
                            _ow('harmonic') or
                            any(f'chroma_{i}' not in existing_keys for i in range(12))):
                        needs_run = True
                    if not needs_run and not self.config.skip_timbral and (
                            _ow('timbral') or
                            any(k not in existing_keys
                                for k in ['brightness', 'roughness', 'hardness', 'depth',
                                          'booming', 'reverberation', 'sharpness', 'warmth'])):
                        needs_run = True

                if needs_run:
                    tasks.append((crop_path, worker_config, existing_keys))

            # Final coverage snapshot (may already be published, ensures consistency)
            if _n_total > 0:
                self.stats['feature_coverage'] = {
                    lbl: cnt / _n_total for lbl, cnt in _cov_counts.items()
                }
                self.stats['feature_coverage_n'] = _n_total

            if tasks:
                logger.info(f"  Processing {len(tasks)} files in background (GPU passes run concurrently)...")

                def _pass1_worker():
                    from concurrent.futures import wait as cf_wait, FIRST_COMPLETED
                    executor = ProcessPoolExecutor(max_workers=self.config.feature_workers)
                    try:
                        futures_map = {executor.submit(_safe_analyze_cpu, task): task[0] for task in tasks}
                        pending = set(futures_map)
                        i = 0
                        _last_info = None
                        # Expose currently-pending crops to the TUI (refreshed each iteration)
                        n_workers = self.config.feature_workers

                        while pending:
                            # Wait for the next batch to complete, with a per-crop timeout.
                            # 300s is generous (timbral+all features ~5-30s normally).
                            # If nothing completes in 300s, the remaining workers are hung
                            # (e.g. timbral_models.timbral_reverb on pathological audio).
                            done, pending = cf_wait(pending, timeout=300, return_when=FIRST_COMPLETED)

                            # Expose up to n_workers pending stems to TUI for "Current" panel
                            self.stats['active_crops'] = [
                                futures_map[f].stem
                                for f in list(pending)[:n_workers]
                            ]

                            if not done:
                                hung = [futures_map[f].name for f in pending]
                                logger.warning(
                                    f"  [PASS 1] {len(pending)} crops timed out after 5 minutes, "
                                    f"skipping: {', '.join(hung[:5])}"
                                    f"{'...' if len(hung) > 5 else ''}"
                                )
                                self.stats["crops_failed"] += len(pending)
                                break

                            for future in done:
                                i += 1
                                crop_path = futures_map[future]
                                try:
                                    result = future.result()
                                    if result['success']:
                                        if result['results']:
                                            safe_update(get_crop_info_path(crop_path), result['results'])
                                            self.stats["crops_processed"] += 1
                                            # Update coverage fractions dynamically
                                            _n = self.stats.get('feature_coverage_n', 0)
                                            if _n > 0:
                                                _cov = self.stats.get('feature_coverage', {})
                                                _rk  = set(result['results'].keys())
                                                for _lbl, _ck in _COV_SENTINEL.items():
                                                    if _ck in _rk and _cov.get(_lbl, 1.0) < 1.0:
                                                        _cov[_lbl] = min(1.0, _cov.get(_lbl, 0.0) + 1.0 / _n)
                                            # Verbose per-task log / non-verbose tracker
                                            gen = result.get('generated', [])
                                            if gen:
                                                _ops_new = [g.split('(')[0] for g in gen if '(new)' in g]
                                                _ops_ow  = [g.split('(')[0] for g in gen if '(overwrite)' in g]
                                                _action  = ('Generating' if _ops_new and not _ops_ow else
                                                            'Overwriting' if _ops_ow and not _ops_new else
                                                            'Generating+overwriting')
                                                _feats   = ' · '.join(dict.fromkeys(g.split('(')[0] for g in gen))
                                                _parts   = crop_path.parts
                                                _short   = '/'.join(_parts[-2:]) if len(_parts) > 2 else str(crop_path)
                                                if logger.isEnabledFor(logging.DEBUG):
                                                    logger.debug(
                                                        f"  [PASS 1 – {_feats}] {_short}"
                                                        f" — {_action} ({i}/{len(tasks)})")
                                                else:
                                                    # stash for the periodic INFO line (feats + action + stem)
                                                    _last_info = (_feats, _action, crop_path.stem)
                                            else:
                                                _last_info = None
                                    else:
                                        _last_info = None
                                        logger.error(f"  [PASS 1] Failed for {crop_path.name}: {result.get('error')}")
                                        self.stats["crops_failed"] += 1
                                except Exception as e:
                                    _last_info = None
                                    logger.error(f"  [PASS 1] Worker exception for {crop_path.name}: {e}")
                                    self.stats["crops_failed"] += 1

                                if i % 50 == 0:
                                    if _last_info:
                                        _li_feats, _li_action, _li_stem = _last_info
                                        logger.info(
                                            f"  [PASS 1 – {_li_feats}] {i}/{len(tasks)}"
                                            f" — {_li_action} {_li_stem}")
                                    else:
                                        logger.info(f"  [PASS 1] {i}/{len(tasks)}")
                                    _last_info = None
                                    if state:
                                        with state_lock:
                                            state.update_pass_progress('pass_1_cpu', completed=i, total=len(tasks))
                                            state.save()

                            if shutdown_requested.is_set():
                                logger.info("  [PASS 1] Shutdown requested — stopping CPU workers.")
                                break

                    except Exception as e:
                        pass1_exception[0] = e
                    finally:
                        # wait=False: don't block on hung workers; they'll be killed
                        # when the Python process exits (worker processes are children).
                        executor.shutdown(wait=False, cancel_futures=True)
                        self.stats['active_crops'] = []
                        if state:
                            with state_lock:
                                state.update_pass_progress('pass_1_cpu', completed=len(tasks), total=len(tasks))
                        logger.info("  [PASS 1] CPU features complete.")

                pass1_thread = threading.Thread(target=_pass1_worker, daemon=True, name='pass1-cpu')
                pass1_thread.start()
            else:
                logger.info("  [PASS 1] All files already analyzed.")
                if state:
                    state.update_pass_progress('pass_1_cpu', completed=0, total=0)

        # =========================================================================
        # PASS 2: AudioBox (Batched for GPU efficiency)
        # PASS 1 background thread is still running concurrently at this point.
        # =========================================================================
        if shutdown_requested.is_set():
            logger.info("Shutdown requested — skipping remaining passes.")
            if pass1_thread is not None:
                pass1_thread.join(timeout=30)
            if state: state.save()
            self.stats["total_time"] = time.time() - start_time
            self._print_crops_summary()
            stop_shutdown_listener()
            return self.stats["crops_failed"] == 0

        if not self.config.skip_audiobox:
            logger.info(f"\n[PASS 2/5] AudioBox Aesthetics - {len(all_crops)} files")
            try:
                from src.timbral.audiobox_aesthetics import analyze_audiobox_aesthetics_batch, get_predictor
                from src.core.common import ProgressBar

                # Check which files actually need processing
                to_process = []
                for crop_path in all_crops:
                    info_path = get_crop_info_path(crop_path)
                    existing = read_info(info_path) if info_path.exists() else {}
                    if self.config.should_overwrite('audiobox') or 'content_enjoyment' not in existing:
                        to_process.append(crop_path)

                # Update AudioBox coverage (pre-pass snapshot)
                _n = self.stats.get('feature_coverage_n', len(all_crops))
                if _n > 0:
                    self.stats.setdefault('feature_coverage', {})['AudioBox'] = (
                        len(all_crops) - len(to_process)) / _n

                if to_process:
                    # Initialize predictor once
                    get_predictor()

                    batch_size = 16  # Process 16 files per GPU batch
                    logger.info(f"  Processing {len(to_process)} files (batch_size={batch_size})...")
                    progress = ProgressBar(len(to_process), desc="AudioBox")

                    for batch_start in range(0, len(to_process), batch_size):
                        if shutdown_requested.is_set():
                            logger.info("  Shutdown requested — stopping AudioBox pass.")
                            break

                        batch_end = min(batch_start + batch_size, len(to_process))
                        batch_paths = to_process[batch_start:batch_end]

                        try:
                            # Process batch
                            batch_results = analyze_audiobox_aesthetics_batch(batch_paths)

                            # Save results
                            for crop_path, results in zip(batch_paths, batch_results):
                                if results:
                                    safe_update(get_crop_info_path(crop_path), results)
                                    self.stats["crops_processed"] += 1
                                else:
                                    self.stats["crops_failed"] += 1
                        except Exception as e:
                            # Batch failed (likely one corrupt file) — retry individually
                            logger.warning(f"  AudioBox batch failed, retrying {len(batch_paths)} files individually: {e}")
                            for crop_path in batch_paths:
                                try:
                                    results = analyze_audiobox_aesthetics_batch([crop_path])
                                    if results and results[0]:
                                        safe_update(get_crop_info_path(crop_path), results[0])
                                        self.stats["crops_processed"] += 1
                                    else:
                                        self.stats["crops_failed"] += 1
                                except Exception as e2:
                                    logger.error(f"  AudioBox skipping corrupt file {crop_path.name}: {e2}")
                                    self.stats["crops_failed"] += 1

                        logger.info(progress.update(batch_end))

                    logger.info(progress.finish("Complete"))
                else:
                    logger.info("  All files already analyzed.")

            except ImportError:
                logger.warning("  AudioBox not available")

        # =========================================================================
        # Join PASS 1 thread before PASS 3 to free CPU cores for Essentia's TF
        # =========================================================================
        if pass1_thread is not None and pass1_thread.is_alive():
            logger.info("[PASS 1/5] Waiting for CPU features thread to finish...")
            pass1_thread.join()
            if pass1_exception[0]:
                logger.error(f"[PASS 1/5] CPU features thread failed: {pass1_exception[0]}")

        # =========================================================================
        # PASS 3: Essentia (Sequential - TensorFlow deadlocks with multiprocessing)
        # =========================================================================
        if shutdown_requested.is_set():
            logger.info("Shutdown requested — skipping remaining passes.")
            if state: state.save()
            self.stats["total_time"] = time.time() - start_time
            self._print_crops_summary()
            stop_shutdown_listener()
            return self.stats["crops_failed"] == 0

        if not self.config.skip_classification:
            logger.info(f"\n[PASS 3/5] Essentia Classification - {len(all_crops)} files")
            logger.info(f"  Processing sequentially (TensorFlow is not multiprocess-safe)")
            try:
                from src.classification.essentia_features import (
                    analyze_essentia_features, preload_models, unload_models
                )

                # Build skip-check keys based on enabled sub-features
                ESSENTIA_KEYS = ['danceability', 'atonality']
                if self.config.essentia_genre:
                    ESSENTIA_KEYS.append('essentia_genre')
                if self.config.essentia_voice:
                    ESSENTIA_KEYS.append('voice_probability')

                include_gmi = any([self.config.essentia_genre,
                                   self.config.essentia_mood,
                                   self.config.essentia_instrument])

                to_process = []
                for crop_path in all_crops:
                    info_path = get_crop_info_path(crop_path)
                    existing = read_info(info_path) if info_path.exists() else {}
                    if self.config.should_overwrite('essentia') or any(k not in existing for k in ESSENTIA_KEYS):
                        to_process.append(crop_path)

                # Update Essentia coverage (pre-pass snapshot)
                _n = self.stats.get('feature_coverage_n', len(all_crops))
                if _n > 0:
                    self.stats.setdefault('feature_coverage', {})['Essentia'] = (
                        len(all_crops) - len(to_process)) / _n

                if to_process:
                    # Pre-load all TF models into VRAM before processing
                    preload_models(
                        include_voice=self.config.essentia_voice,
                        include_gender=self.config.essentia_gender,
                        include_gmi=include_gmi,
                        include_genre=self.config.essentia_genre,
                        include_mood=self.config.essentia_mood,
                        include_instrument=self.config.essentia_instrument,
                    )

                    logger.info(f"  Processing {len(to_process)} files...")

                    # Build prefetch list + cache vocals paths (avoids double filesystem lookup)
                    from src.core.audio_prefetcher import PathPrefetcher
                    prefetch_paths = []
                    vocals_path_cache = {}
                    for cp in to_process:
                        prefetch_paths.append(cp)
                        if self.config.essentia_gender:
                            stems = get_crop_stem_files(cp)
                            vp = stems.get('vocals')
                            vocals_path_cache[cp] = vp
                            if vp:
                                prefetch_paths.append(vp)

                    prefetcher = PathPrefetcher(prefetch_paths, buffer_size=16)
                    prefetcher.start()

                    # Timing accumulator: key -> list of elapsed seconds
                    timing_acc = defaultdict(list)

                    # Sequential processing - TensorFlow handles internal parallelism
                    for i, crop_path in enumerate(to_process, 1):
                        if shutdown_requested.is_set():
                            logger.info("  Shutdown requested — stopping Essentia pass.")
                            break

                        try:
                            vocals_path = vocals_path_cache.get(crop_path) if self.config.essentia_gender else None

                            _timings = {}
                            results = analyze_essentia_features(
                                crop_path,
                                include_voice_analysis=self.config.essentia_voice,
                                include_gender=self.config.essentia_gender,
                                include_gmi=include_gmi,
                                include_genre=self.config.essentia_genre,
                                include_mood=self.config.essentia_mood,
                                include_instrument=self.config.essentia_instrument,
                                vocals_path=vocals_path,
                                vocal_content_thresholds=self.config.vocal_content_thresholds,
                                _timings=_timings)
                            for k, v in _timings.items():
                                timing_acc[k].append(v)
                            if results:
                                safe_update(get_crop_info_path(crop_path), results)
                                self.stats["crops_processed"] += 1
                        except Exception as e:
                            logger.error(f"  Essentia failed for {crop_path.name}: {e}")
                            self.stats["crops_failed"] += 1

                        # Advance prefetcher past this crop (and its vocals stem)
                        prefetcher.mark_processed(crop_path)
                        if self.config.essentia_gender and vocals_path:
                            prefetcher.mark_processed(vocals_path)

                        if i % 10 == 0 or i == len(to_process):
                            logger.info(f"  {i}/{len(to_process)}")
                            if state and i % 50 == 0:
                                state.update_pass_progress('pass_3_essentia', completed=i, total=len(to_process))
                                state.save()

                    prefetcher.stop()

                    # Log per-model timing statistics
                    if timing_acc:
                        logger.info("  Essentia timing statistics (seconds/crop):")
                        for key in ['audio_load', 'danceability', 'atonality', 'voice', 'gender', 'gmi']:
                            vals = timing_acc.get(key)
                            if vals:
                                logger.info(
                                    f"    {key:<12s}  mean={sum(vals)/len(vals):.3f}s"
                                    f"  min={min(vals):.3f}s  max={max(vals):.3f}s"
                                    f"  n={len(vals)}"
                                )

                    # Free VRAM for subsequent passes (Music Flamingo needs it)
                    unload_models()
                    if state:
                        state.update_pass_progress('pass_3_essentia', completed=len(to_process), total=len(to_process))
                else:
                    logger.info("  All files already analyzed.")

            except ImportError:
                logger.warning("  Essentia not available")

        # =========================================================================
        # PASS 4: Music Flamingo
        # =========================================================================
        if shutdown_requested.is_set():
            logger.info("Shutdown requested — skipping remaining passes.")
            if state: state.save()
            self.stats["total_time"] = time.time() - start_time
            self._print_crops_summary()
            stop_shutdown_listener()
            return self.stats["crops_failed"] == 0

        if not self.config.skip_flamingo:
            logger.info(f"\n[PASS 4/5] Music Flamingo (Batched, GGUF/CLI) - {len(all_crops)} files")
            try:
                from src.classification.music_flamingo import MusicFlamingoGGUF, DEFAULT_PROMPTS
                
                # Determine active prompts map
                prompts_map = self.config.flamingo_prompts
                if not prompts_map:
                    prompts_map = DEFAULT_PROMPTS
                
                if not prompts_map:
                    logger.warning("  No Flamingo prompts enabled in config.")
                else:
                    logger.info(f"  Active prompts: {', '.join(prompts_map.keys())}")
                    
                    # Filter files needing processing (check if ALL active prompts are present)
                    prob = self.config.flamingo_sample_probability
                    to_process = []
                    skipped_by_sampling = 0
                    for crop_path in all_crops:
                        info_path = get_crop_info_path(crop_path)
                        existing = read_info(info_path) if info_path.exists() else {}

                        needs_run = False
                        if self.config.should_overwrite('flamingo'):
                            needs_run = True
                        else:
                            for p in prompts_map.keys():
                                if f'music_flamingo_{p}' not in existing:
                                    needs_run = True
                                    break

                        if needs_run:
                            if prob >= 1.0 or random.random() < prob:
                                to_process.append(crop_path)
                            else:
                                skipped_by_sampling += 1

                    # Update Flamingo coverage (pre-pass snapshot)
                    _n = self.stats.get('feature_coverage_n', len(all_crops))
                    if _n > 0:
                        self.stats.setdefault('feature_coverage', {})['Flamingo'] = (
                            len(all_crops) - len(to_process) - skipped_by_sampling) / _n

                    if skipped_by_sampling:
                        logger.info(f"  Sampling at {prob:.0%}: {len(to_process)} queued, "
                                    f"{skipped_by_sampling} deferred to future runs")
                    
                    if to_process:
                        logger.info(f"  Loading model ({self.config.flamingo_model})...")
                        flamingo = MusicFlamingoGGUF(
                            model=self.config.flamingo_model,
                        )

                        # Start background prefetcher to warm disk cache
                        # This helps hide HDD I/O latency while GPU processes
                        from src.core.audio_prefetcher import PathPrefetcher
                        prefetcher = PathPrefetcher(to_process, buffer_size=8)
                        prefetcher.start()

                        logger.info(f"  Processing {len(to_process)} files (with disk prefetch)...")
                        for i, crop_path in enumerate(to_process, 1):
                            if shutdown_requested.is_set():
                                logger.info("  Shutdown requested — stopping Flamingo pass.")
                                break

                            results = {}
                            try:
                                for prompt_type, prompt_text in prompts_map.items():
                                    key = f'music_flamingo_{prompt_type}'

                                    # Overwrite check for specific key
                                    info_path = get_crop_info_path(crop_path)
                                    existing = read_info(info_path) if info_path.exists() else {}

                                    if self.config.should_overwrite('flamingo') or key not in existing:
                                        # Interpolate genre hints into prompt
                                        p_text = _interpolate_genres(prompt_text, existing) if self.config.flamingo_prompts else None
                                        results[key] = flamingo.analyze(
                                            crop_path,
                                            prompt=p_text,
                                            prompt_type=prompt_type
                                        )

                                if results:
                                    results['music_flamingo_model'] = f'accel_{self.config.flamingo_model}'
                                    safe_update(get_crop_info_path(crop_path), results)
                                    self.stats["crops_processed"] += 1

                                logger.info(f"  [{i}/{len(to_process)}] {crop_path.name}")

                                # Mark file as processed so prefetcher can clean up
                                prefetcher.mark_processed(crop_path)

                                if state and i % 50 == 0:
                                    state.update_pass_progress('pass_4_flamingo', completed=i, total=len(to_process))
                                    state.save()

                            except Exception as e:
                                logger.error(f"  Flamingo failed for {crop_path.name}: {e}")
                                self.stats["crops_failed"] += 1
                                prefetcher.mark_processed(crop_path)
                                # Don't break loop, keep processing other files

                        prefetcher.stop()
                        if state:
                            state.update_pass_progress('pass_4_flamingo', completed=len(to_process), total=len(to_process))

                        # Show aggregate performance stats
                        if flamingo.stats.runs > 0:
                            logger.info(flamingo.stats.summary())

                        del flamingo
                    else:
                        logger.info("  All files already analyzed.")

            except Exception as e:
                logger.warning(f"  Music Flamingo failed: {e}")

        # =========================================================================
        # PASS 4b: Granite Revision of MF descriptions
        # =========================================================================
        if shutdown_requested.is_set():
            logger.info("Shutdown requested — skipping remaining passes.")
            if state: state.save()
            self.stats["total_time"] = time.time() - start_time
            self._print_crops_summary()
            stop_shutdown_listener()
            return self.stats["crops_failed"] == 0

        rev_cfg = self.config.flamingo_revision
        if rev_cfg.get('enabled') and rev_cfg.get('prompts'):
            rev_prompts = rev_cfg['prompts']

            # Scan the ENTIRE crops directory — not just all_crops from this run.
            # This catches crops that were annotated by Flamingo in a previous
            # (possibly interrupted) run and never had revision applied.
            rev_folders = find_crop_folders(self.config.working_dir)
            all_revision_crops: List[Path] = []
            for folder in rev_folders:
                all_revision_crops.extend(find_crop_files(folder))
            if not all_revision_crops:
                # Flat directory: crops live directly in working_dir
                all_revision_crops = find_crop_files(self.config.working_dir)

            logger.info(f"\n[PASS 4b] Granite Revision - {len(all_revision_crops)} total files")
            logger.info(f"  Revision keys: {', '.join(rev_prompts.keys())}")

            # Filter files needing revision
            to_revise = []
            for crop_path in all_revision_crops:
                info_path = get_crop_info_path(crop_path)
                existing = read_info(info_path) if info_path.exists() else {}
                needs_rev = any(
                    f'music_flamingo_{rk}' not in existing or self.config.should_overwrite('flamingo')
                    for rk in rev_prompts
                )
                # Only revise if source MF descriptions exist
                has_mf = any(k.startswith('music_flamingo_') and isinstance(v, str)
                             for k, v in existing.items())
                if needs_rev and has_mf:
                    to_revise.append(crop_path)

            if to_revise:
                try:
                    from src.classification.granite_revision import GraniteReviser
                    reviser = GraniteReviser(
                        rev_cfg['model'],
                        n_ctx=rev_cfg.get('n_ctx', 4096),
                        temperature=rev_cfg.get('temperature', 0.7),
                        max_tokens=rev_cfg.get('max_tokens', 512),
                    )

                    logger.info(f"  Processing {len(to_revise)} files...")
                    for i, crop_path in enumerate(to_revise, 1):
                        if shutdown_requested.is_set():
                            logger.info("  Shutdown requested — stopping revision pass.")
                            break

                        try:
                            info_path = get_crop_info_path(crop_path)
                            existing = read_info(info_path) if info_path.exists() else {}
                            mf_results = {k: v for k, v in existing.items()
                                          if k.startswith('music_flamingo_') and isinstance(v, str)}
                            rev_results = reviser.revise(mf_results, rev_prompts)
                            if rev_results:
                                save_data = {f'music_flamingo_{rk}': rv for rk, rv in rev_results.items()}
                                safe_update(info_path, save_data)
                            logger.info(f"  [{i}/{len(to_revise)}] {crop_path.name} ({len(rev_results)} revisions)")
                        except Exception as e:
                            logger.error(f"  Revision failed for {crop_path.name}: {e}")

                    reviser.close()
                except Exception as e:
                    logger.warning(f"  Granite revision failed: {e}")
            else:
                logger.info("  All files already revised.")

        # =========================================================================
        # PASS 5: MIDI Transcription
        # =========================================================================
        if not self.config.skip_midi:
            logger.info(f"\n[PASS 5/5] MIDI Transcription - {len(all_crops)} files")
             # MIDI script handles its own batching/loading effectively usually, 
             # but here we just call it. For now, skipping detailed implementation 
             # as it's often a separate step.
             # TODO: Call src/transcription/drums/adtof.py logic here if needed per crop
            pass

        self.stats["total_time"] = time.time() - start_time
        self._print_crops_summary()
        if state: state.save()
        stop_shutdown_listener()
        return self.stats["crops_failed"] == 0


def main():
    parser = argparse.ArgumentParser(
        description="MIR Feature Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process organized folders (full_mix.flac structure)
  python src/pipeline.py /path/to/audio --batch

  # Process crop files (TrackName_0.flac structure)
  python src/pipeline.py /path/to/crops --batch --crops

  # Copy to output directory, then process there
  python src/pipeline.py /path/to/audio --output-dir /path/to/output --batch

  # Skip heavy modules (Demucs, Flamingo)
  python src/pipeline.py /path/to/audio --batch --skip-demucs --skip-flamingo

  # Verbose output
  python src/pipeline.py /path/to/audio --batch -v
        """
    )

    parser.add_argument("input", help="Input directory containing audio files")
    parser.add_argument("--output-dir", "-o", help="Output directory (copies files here first)")
    parser.add_argument("--batch", "-b", action="store_true", help="Batch process all folders")
    parser.add_argument("--crops", "-c", action="store_true",
                        help="Process crop files (TrackName_0.flac) instead of organized folders")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for GPU processing (default: cuda)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Skip flags
    parser.add_argument("--skip-organize", action="store_true", help="Skip file organization")
    parser.add_argument("--skip-demucs", action="store_true", help="Skip stem separation")
    parser.add_argument("--skip-rhythm", action="store_true", help="Skip rhythm analysis")
    parser.add_argument("--skip-loudness", action="store_true", help="Skip loudness analysis")
    parser.add_argument("--skip-spectral", action="store_true", help="Skip spectral features")
    parser.add_argument("--skip-harmonic", action="store_true", help="Skip harmonic features")
    parser.add_argument("--skip-timbral", action="store_true", help="Skip timbral features")
    parser.add_argument("--skip-classification", action="store_true", help="Skip classification")
    parser.add_argument("--skip-per-stem", action="store_true", help="Skip per-stem analysis")
    parser.add_argument("--skip-flamingo", action="store_true", help="Skip Music Flamingo")
    parser.add_argument("--skip-audiobox", action="store_true", help="Skip AudioBox aesthetics")
    parser.add_argument("--skip-midi", action="store_true", help="Skip MIDI transcription")

    # Flamingo options
    parser.add_argument("--flamingo-model", default="Q8_0",
                        choices=["IQ3_M", "Q6_K", "Q8_0"],
                        help="Music Flamingo GGUF model (default: Q8_0)")
    
    args = parser.parse_args()
    
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
        
    # Create config
    config = PipelineConfig(
        input_dir=input_path,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        device=args.device,
        batch=args.batch,
        overwrite=args.overwrite,
        verbose=args.verbose,
        crops=args.crops,
        skip_organize=args.skip_organize,
        skip_demucs=args.skip_demucs,
        skip_rhythm=args.skip_rhythm,
        skip_loudness=args.skip_loudness,
        skip_spectral=args.skip_spectral,
        skip_harmonic=args.skip_harmonic,
        skip_timbral=args.skip_timbral,
        skip_classification=args.skip_classification,
        skip_per_stem=args.skip_per_stem,
        skip_flamingo=args.skip_flamingo,
        skip_audiobox=args.skip_audiobox,
        skip_midi=args.skip_midi,
        flamingo_model=args.flamingo_model
    )
    
    # Create output dir if specified
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    pipeline = Pipeline(config)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
