"""
Feature Extraction for Crops

Extracts all MIR features for crop files and saves to per-crop .INFO files.
Optimized to load heavy models once and process multiple crops efficiently.

Usage:
    from crops.feature_extractor import CropFeatureExtractor

    extractor = CropFeatureExtractor(skip_flamingo=False, flamingo_model='Q8_0')
    extractor.extract_features(crop_path, stems={'drums': drums_path, ...})
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set ROCm environment before torch imports
from core.rocm_env import setup_rocm_env
setup_rocm_env()

from core.file_utils import get_crop_stem_files, DEMUCS_STEMS
from core.json_handler import safe_update, get_crop_info_path, read_info
import soundfile as sf

logger = logging.getLogger(__name__)


class CropFeatureExtractor:
    """
    Feature extractor for crop files.

    Loads heavy models (Music Flamingo, Essentia) once on initialization,
    then extracts features for multiple crops efficiently.
    """

    def __init__(
        self,
        skip_demucs: bool = True,  # Default True - stems should come from cropping
        skip_flamingo: bool = False,
        skip_audiobox: bool = False,
        skip_essentia: bool = False,
        skip_timbral: bool = False,
        flamingo_model: str = 'Q6_K',
        flamingo_context_size: int = 1024,
        device: str = 'cuda',
        essentia_genre: bool = True,
        essentia_mood: bool = True,
        essentia_instrument: bool = True,
        essentia_voice: bool = True,
        essentia_gender: bool = True,
        vocal_content_thresholds: Optional[Dict] = None,
        flamingo_prompts: Optional[Dict[str, str]] = None,
        flamingo_revision: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the feature extractor and load models.

        Args:
            skip_demucs: Skip stem separation (default True - stems from cropping)
            skip_flamingo: Skip Music Flamingo descriptions
            skip_audiobox: Skip AudioBox aesthetics
            skip_essentia: Skip Essentia features
            skip_timbral: Skip Audio Commons timbral features
            flamingo_model: GGUF model for Music Flamingo ('IQ3_M', 'Q6_K', 'Q8_0')
            flamingo_context_size: LLM context window size (default 1024)
            device: Device for GPU models ('cuda' or 'cpu')
            essentia_genre: Enable genre classification
            essentia_mood: Enable mood classification
            essentia_instrument: Enable instrument classification
            essentia_voice: Enable voice/instrumental detection
            essentia_gender: Enable vocal gender analysis
            vocal_content_thresholds: Dict with crest/rms/peak thresholds for vocal content
        """
        self.skip_demucs = skip_demucs
        self.skip_flamingo = skip_flamingo
        self.skip_audiobox = skip_audiobox
        self.skip_essentia = skip_essentia
        self.skip_timbral = skip_timbral
        self.flamingo_model = flamingo_model
        self.flamingo_context_size = flamingo_context_size
        self.device = device
        self.essentia_genre = essentia_genre
        self.essentia_mood = essentia_mood
        self.essentia_instrument = essentia_instrument
        self.essentia_voice = essentia_voice
        self.essentia_gender = essentia_gender
        self.vocal_content_thresholds = vocal_content_thresholds or {
            'crest_factor_threshold': 30.0,
            'rms_threshold': -42.0,
            'peak_threshold': -20.0,
        }
        self.flamingo_prompts = flamingo_prompts or {}
        self.flamingo_revision = flamingo_revision or {}

        # Lazy-loaded models (load on first use)
        self._flamingo = None
        self._essentia_models_loaded = False

        # Pre-load Music Flamingo if not skipped (this is the heavy one)
        if not skip_flamingo:
            self._load_flamingo()

    def _load_flamingo(self):
        """Load Music Flamingo model."""
        if self._flamingo is not None:
            return

        try:
            from classification.music_flamingo import MusicFlamingoGGUF
            logger.info(f"Loading Music Flamingo (GGUF/CLI) {self.flamingo_model}...")

            self._flamingo = MusicFlamingoGGUF(
                model=self.flamingo_model,
            )
            logger.info("Music Flamingo loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Music Flamingo: {e}")
            self._flamingo = None

    def extract_features(
        self,
        crop_path: Path,
        stems: Optional[Dict[str, Path]] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract all features for a single crop.

        Args:
            crop_path: Path to the crop audio file
            stems: Pre-separated stems (if available)
            overwrite: Overwrite existing features

        Returns:
            Dict of extracted features
        """
        crop_path = Path(crop_path)
        if not crop_path.exists():
            raise FileNotFoundError(f"Crop file not found: {crop_path}")

        # Get info path for this crop
        info_path = get_crop_info_path(crop_path)

        # Check existing features
        existing = read_info(info_path) if info_path.exists() else {}

        # Get stems if not provided
        if stems is None:
            stems = get_crop_stem_files(crop_path)

        # PRE-LOAD: Read crop audio ONCE from disk (shared across all feature functions)
        crop_audio, sample_rate = sf.read(str(crop_path))
        import numpy as np
        crop_mono = crop_audio.mean(axis=1) if crop_audio.ndim > 1 else crop_audio
        duration = len(crop_audio) / sample_rate

        # Pre-load stem audio into RAM
        preloaded_stems = {}
        for stem_name in ['drums', 'bass', 'other', 'vocals']:
            if stem_name in stems:
                try:
                    s_audio, s_sr = sf.read(str(stems[stem_name]))
                    preloaded_stems[stem_name] = (s_audio, s_sr)
                except Exception:
                    pass

        logger.info(f"Extracting features for: {crop_path.name}")
        logger.info(f"  Duration: {duration:.2f}s, SR: {sample_rate}Hz")
        logger.info(f"  Stems available: {[k for k in stems if k != 'source']}")

        results = {}
        timings = {}

        # 1. Loudness (LUFS/LRA)
        self._extract_loudness(crop_path, stems, results, timings, existing, overwrite,
                               crop_audio=crop_audio, crop_sr=sample_rate,
                               preloaded_stems=preloaded_stems)

        # 2. BPM (if not already present from crop creation)
        self._extract_bpm(crop_path, results, timings, existing, overwrite,
                          crop_mono=crop_mono, crop_sr=sample_rate)

        # 3. Spectral features
        self._extract_spectral(crop_path, results, timings, existing, overwrite,
                               crop_mono=crop_mono, crop_sr=sample_rate)

        # 4. Multiband RMS
        self._extract_multiband_rms(crop_path, results, timings, existing, overwrite,
                                     crop_mono=crop_mono, crop_sr=sample_rate)

        # 5. Chroma features
        self._extract_chroma(crop_path, stems, results, timings, existing, overwrite,
                             preloaded_stems=preloaded_stems, crop_mono=crop_mono,
                             crop_sr=sample_rate)

        # 6. Timbral features (Audio Commons)
        if not self.skip_timbral:
            self._extract_timbral(crop_path, results, timings, existing, overwrite,
                                  crop_audio=crop_audio, crop_sr=sample_rate)

        # 7. AudioBox Aesthetics
        if not self.skip_audiobox:
            self._extract_audiobox(crop_path, results, timings, existing, overwrite)

        # 8. Essentia (danceability, atonality)
        if not self.skip_essentia:
            self._extract_essentia(crop_path, results, timings, existing, overwrite)

        # 9. Music Flamingo (descriptions)
        if not self.skip_flamingo:
            self._extract_flamingo(crop_path, results, timings, existing, overwrite)

        # Save all results to .INFO
        if results:
            safe_update(info_path, results)
            logger.info(f"  Saved {len(results)} features to {info_path.name}")

        # Log timing summary
        total_time = sum(timings.values())
        if timings:
            logger.info(f"  Total extraction time: {total_time:.2f}s ({duration/total_time:.1f}x realtime)")

        return results

    def _should_extract(self, feature_key: str, existing: Dict, overwrite: bool) -> bool:
        """Check if a feature should be extracted."""
        if overwrite:
            return True
        return feature_key not in existing

    def _extract_loudness(self, crop_path: Path, stems: Dict, results: Dict,
                          timings: Dict, existing: Dict, overwrite: bool,
                          crop_audio=None, crop_sr=None, preloaded_stems=None):
        """Extract loudness features (LUFS/LRA)."""
        if not self._should_extract('lufs', existing, overwrite):
            logger.debug("  Skipping loudness (already exists)")
            return

        try:
            start = time.time()
            from timbral.loudness import analyze_file_loudness

            # Analyze main crop (pre-loaded)
            loudness = analyze_file_loudness(crop_path, audio=crop_audio, sr=crop_sr)
            results.update(loudness)

            # Analyze stems (pre-loaded)
            if preloaded_stems:
                for stem_name, (s_audio, s_sr) in preloaded_stems.items():
                    stem_loudness = analyze_file_loudness(
                        stems[stem_name], audio=s_audio, sr=s_sr)
                    results[f'lufs_{stem_name}'] = stem_loudness.get('lufs')
                    results[f'lra_{stem_name}'] = stem_loudness.get('lra')
            else:
                for stem_name in ['drums', 'bass', 'other', 'vocals']:
                    if stem_name in stems:
                        stem_loudness = analyze_file_loudness(stems[stem_name])
                        results[f'lufs_{stem_name}'] = stem_loudness.get('lufs')
                        results[f'lra_{stem_name}'] = stem_loudness.get('lra')

            timings['loudness'] = time.time() - start
            logger.info(f"  Loudness: {timings['loudness']:.2f}s")
        except Exception as e:
            logger.warning(f"  Loudness failed: {e}")

    def _extract_bpm(self, crop_path: Path, results: Dict, timings: Dict,
                     existing: Dict, overwrite: bool,
                     crop_mono=None, crop_sr=None):
        """Extract BPM (if not already present)."""
        # BPM is often inherited from crop creation, so check first
        if not self._should_extract('bpm', existing, overwrite):
            logger.debug("  Skipping BPM (already exists)")
            return

        try:
            start = time.time()
            from rhythm.bpm import analyze_bpm

            bpm_results = analyze_bpm(crop_path)
            results.update(bpm_results)

            timings['bpm'] = time.time() - start
            logger.info(f"  BPM: {timings['bpm']:.2f}s")
        except Exception as e:
            logger.warning(f"  BPM failed: {e}")

    def _extract_spectral(self, crop_path: Path, results: Dict, timings: Dict,
                          existing: Dict, overwrite: bool,
                          crop_mono=None, crop_sr=None):
        """Extract spectral features."""
        if not self._should_extract('spectral_centroid', existing, overwrite):
            logger.debug("  Skipping spectral (already exists)")
            return

        try:
            start = time.time()
            from spectral.spectral_features import analyze_spectral_features

            spectral = analyze_spectral_features(crop_path, audio=crop_mono, sr=crop_sr)
            results.update(spectral)

            timings['spectral'] = time.time() - start
            logger.info(f"  Spectral: {timings['spectral']:.2f}s")
        except Exception as e:
            logger.warning(f"  Spectral failed: {e}")

    def _extract_multiband_rms(self, crop_path: Path, results: Dict, timings: Dict,
                                existing: Dict, overwrite: bool,
                                crop_mono=None, crop_sr=None):
        """Extract multiband RMS energy."""
        if not self._should_extract('rms_bass', existing, overwrite):
            logger.debug("  Skipping multiband RMS (already exists)")
            return

        try:
            start = time.time()
            from spectral.multiband_rms import analyze_multiband_rms

            rms = analyze_multiband_rms(crop_path, audio=crop_mono, sr=crop_sr)
            results.update(rms)

            timings['multiband_rms'] = time.time() - start
            logger.info(f"  Multiband RMS: {timings['multiband_rms']:.2f}s")
        except Exception as e:
            logger.warning(f"  Multiband RMS failed: {e}")

    def _extract_chroma(self, crop_path: Path, stems: Dict, results: Dict,
                        timings: Dict, existing: Dict, overwrite: bool,
                        preloaded_stems=None, crop_mono=None, crop_sr=None):
        """Extract chroma features."""
        if not self._should_extract('chroma_0', existing, overwrite):
            logger.debug("  Skipping chroma (already exists)")
            return

        try:
            start = time.time()
            import numpy as np
            from harmonic.chroma import analyze_chroma

            # Use harmonic stem if available (pre-loaded), else crop mono
            if preloaded_stems and 'other' in preloaded_stems:
                other_audio, other_sr = preloaded_stems['other']
                if other_audio.ndim > 1:
                    other_audio = other_audio.mean(axis=1)
                chroma = analyze_chroma(
                    stems['other'], use_stems=False,
                    audio=other_audio.astype(np.float32), sr=other_sr)
            elif crop_mono is not None:
                chroma = analyze_chroma(
                    crop_path, use_stems=False,
                    audio=crop_mono.astype(np.float32), sr=crop_sr)
            else:
                audio_for_chroma = stems.get('other', crop_path)
                chroma = analyze_chroma(audio_for_chroma)
            results.update(chroma)

            timings['chroma'] = time.time() - start
            logger.info(f"  Chroma: {timings['chroma']:.2f}s")
        except Exception as e:
            logger.warning(f"  Chroma failed: {e}")

    def _extract_timbral(self, crop_path: Path, results: Dict, timings: Dict,
                         existing: Dict, overwrite: bool,
                         crop_audio=None, crop_sr=None):
        """Extract Audio Commons timbral features."""
        if not self._should_extract('brightness', existing, overwrite):
            logger.debug("  Skipping timbral (already exists)")
            return

        try:
            start = time.time()
            from timbral.audio_commons import analyze_all_timbral_features

            timbral = analyze_all_timbral_features(
                crop_path, audio=crop_audio, sr=crop_sr)
            results.update(timbral)

            timings['timbral'] = time.time() - start
            logger.info(f"  Timbral: {timings['timbral']:.2f}s")
        except Exception as e:
            logger.warning(f"  Timbral failed: {e}")

    def _extract_audiobox(self, crop_path: Path, results: Dict, timings: Dict,
                          existing: Dict, overwrite: bool):
        """Extract AudioBox Aesthetics features."""
        if not self._should_extract('content_enjoyment', existing, overwrite):
            logger.debug("  Skipping AudioBox (already exists)")
            return

        try:
            start = time.time()
            from timbral.audiobox_aesthetics import analyze_audiobox_aesthetics

            audiobox = analyze_audiobox_aesthetics(crop_path)
            results.update(audiobox)

            timings['audiobox'] = time.time() - start
            logger.info(f"  AudioBox: {timings['audiobox']:.2f}s")
        except ImportError:
            logger.debug("  AudioBox not installed")
        except Exception as e:
            logger.warning(f"  AudioBox failed: {e}")

    def _extract_essentia(self, crop_path: Path, results: Dict, timings: Dict,
                          existing: Dict, overwrite: bool):
        """Extract Essentia features (danceability, atonality, genre, mood, instrument, voice, gender)."""
        # Check if any enabled sub-feature is missing
        needs_run = self._should_extract('danceability', existing, overwrite)
        if not needs_run and self.essentia_genre and 'essentia_genre' not in existing:
            needs_run = True
        if not needs_run and self.essentia_voice and 'voice_probability' not in existing:
            needs_run = True
        if not needs_run:
            logger.debug("  Skipping Essentia (already exists)")
            return

        try:
            start = time.time()
            from classification.essentia_features import analyze_essentia_features

            # Find vocals stem for gender analysis
            vocals_path = None
            if self.essentia_gender:
                stems = get_crop_stem_files(crop_path)
                vocals_path = stems.get('vocals')

            include_gmi = any([self.essentia_genre, self.essentia_mood,
                               self.essentia_instrument])
            essentia = analyze_essentia_features(
                crop_path,
                include_voice_analysis=self.essentia_voice,
                include_gender=self.essentia_gender,
                include_gmi=include_gmi,
                include_genre=self.essentia_genre,
                include_mood=self.essentia_mood,
                include_instrument=self.essentia_instrument,
                vocals_path=vocals_path,
                vocal_content_thresholds=self.vocal_content_thresholds)
            results.update(essentia)

            timings['essentia'] = time.time() - start
            logger.info(f"  Essentia: {timings['essentia']:.2f}s")
        except Exception as e:
            logger.warning(f"  Essentia failed: {e}")

    def _extract_flamingo(self, crop_path: Path, results: Dict, timings: Dict,
                          existing: Dict, overwrite: bool):
        """Extract Music Flamingo descriptions with genre interpolation and optional revision."""
        if self._flamingo is None:
            logger.debug("  Skipping Music Flamingo (not loaded)")
            return

        # Determine prompts
        prompts_map = self.flamingo_prompts if self.flamingo_prompts else None
        if prompts_map:
            prompt_types = list(prompts_map.keys())
        else:
            prompt_types = ['full', 'technical', 'genre_mood', 'instrumentation', 'structure']

        all_exist = all(
            f'music_flamingo_{pt}' in existing for pt in prompt_types
        )

        if all_exist and not overwrite:
            logger.debug("  Skipping Music Flamingo (already exists)")
            return

        try:
            start = time.time()

            if prompts_map:
                # Custom prompts with genre interpolation
                from pipeline import _interpolate_genres
                interpolated = {}
                for pt, p_text in prompts_map.items():
                    interpolated[pt] = _interpolate_genres(p_text, existing)
                flamingo_results = self._flamingo.analyze_all_prompts(crop_path, prompts=interpolated)
            else:
                flamingo_results = self._flamingo.analyze_all_prompts(crop_path)

            results.update(flamingo_results)
            results['music_flamingo_model'] = f'gguf_{self.flamingo_model}'

            timings['flamingo'] = time.time() - start
            logger.info(f"  Music Flamingo: {timings['flamingo']:.2f}s ({len(prompt_types)} prompts)")

            # Granite revision
            rev_cfg = self.flamingo_revision
            if rev_cfg.get('enabled') and rev_cfg.get('prompts'):
                mf_results = {k: v for k, v in {**existing, **results}.items()
                              if k.startswith('music_flamingo_') and isinstance(v, str)}
                rev_keys = rev_cfg['prompts']
                need_rev = any(
                    f'music_flamingo_{rk}' not in existing or overwrite
                    for rk in rev_keys
                )
                if mf_results and need_rev:
                    try:
                        rev_start = time.time()
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
                        timings['granite_revision'] = time.time() - rev_start
                        logger.info(f"  Granite revision: {timings['granite_revision']:.2f}s ({len(rev_results)} keys)")
                    except Exception as e:
                        logger.warning(f"  Granite revision failed: {e}")

        except Exception as e:
            logger.warning(f"  Music Flamingo failed: {e}")


# Command-line interface
if __name__ == "__main__":
    import argparse
    from core.common import setup_logging
    from core.file_utils import find_crop_files

    parser = argparse.ArgumentParser(
        description="Extract features for crop files"
    )

    parser.add_argument(
        'path',
        type=str,
        help='Path to crop file or folder containing crops'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all crops in folder'
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
        '--overwrite',
        action='store_true',
        help='Overwrite existing features'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    path = Path(args.path)
    if not path.exists():
        logger.error(f"Path not found: {path}")
        sys.exit(1)

    # Initialize extractor
    extractor = CropFeatureExtractor(
        skip_flamingo=args.skip_flamingo,
        skip_audiobox=args.skip_audiobox,
        skip_essentia=args.skip_essentia,
        skip_timbral=args.skip_timbral,
        flamingo_model=args.flamingo_model,
    )

    try:
        if args.batch or path.is_dir():
            # Batch processing
            crop_files = find_crop_files(path)
            logger.info(f"Found {len(crop_files)} crop files")

            for i, crop_path in enumerate(crop_files, 1):
                logger.info(f"\n[{i}/{len(crop_files)}] Processing: {crop_path.name}")
                try:
                    extractor.extract_features(crop_path, overwrite=args.overwrite)
                except Exception as e:
                    logger.error(f"Failed: {e}")
        else:
            # Single crop
            extractor.extract_features(path, overwrite=args.overwrite)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
