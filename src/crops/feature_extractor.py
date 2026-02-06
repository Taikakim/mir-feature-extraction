"""
Feature Extraction for Crops

Extracts all MIR features for crop files and saves to per-crop .INFO files.
Optimized to load heavy models once and process multiple crops efficiently.

Usage:
    from crops.feature_extractor import CropFeatureExtractor

    extractor = CropFeatureExtractor(skip_flamingo=False, flamingo_model='Q8_0')
    extractor.extract_features(crop_path, stems={'drums': drums_path, ...})
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any
import sys

# Set ROCm environment before torch imports
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'garbage_collection_threshold:0.8')
os.environ.setdefault('FLASH_ATTENTION_TRITON_AMD_ENABLE', 'TRUE')
os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_TUNING', '0')
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MIOPEN_FIND_MODE', '2')

sys.path.insert(0, str(Path(__file__).parent.parent))

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
        """
        self.skip_demucs = skip_demucs
        self.skip_flamingo = skip_flamingo
        self.skip_audiobox = skip_audiobox
        self.skip_essentia = skip_essentia
        self.skip_timbral = skip_timbral
        self.flamingo_model = flamingo_model
        self.flamingo_context_size = flamingo_context_size
        self.device = device

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
            from classification.music_flamingo_llama_cpp import MusicFlamingoAnalyzer
            logger.info(f"Loading Music Flamingo (llama-cpp-python) {self.flamingo_model}...")

            # Use auto-detection with configured model name and context
            self._flamingo = MusicFlamingoAnalyzer(
                model_name=self.flamingo_model,
                n_ctx=self.flamingo_context_size,
                n_gpu_layers=-1,  # All layers on GPU
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

        # Get audio info
        info = sf.info(str(crop_path))
        duration = info.duration
        sample_rate = info.samplerate

        logger.info(f"Extracting features for: {crop_path.name}")
        logger.info(f"  Duration: {duration:.2f}s, SR: {sample_rate}Hz")
        logger.info(f"  Stems available: {[k for k in stems if k != 'source']}")

        results = {}
        timings = {}

        # 1. Loudness (LUFS/LRA)
        self._extract_loudness(crop_path, stems, results, timings, existing, overwrite)

        # 2. BPM (if not already present from crop creation)
        self._extract_bpm(crop_path, results, timings, existing, overwrite)

        # 3. Spectral features
        self._extract_spectral(crop_path, results, timings, existing, overwrite)

        # 4. Multiband RMS
        self._extract_multiband_rms(crop_path, results, timings, existing, overwrite)

        # 5. Chroma features
        self._extract_chroma(crop_path, stems, results, timings, existing, overwrite)

        # 6. Timbral features (Audio Commons)
        if not self.skip_timbral:
            self._extract_timbral(crop_path, results, timings, existing, overwrite)

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
                          timings: Dict, existing: Dict, overwrite: bool):
        """Extract loudness features (LUFS/LRA)."""
        if not self._should_extract('lufs', existing, overwrite):
            logger.debug("  Skipping loudness (already exists)")
            return

        try:
            start = time.time()
            from timbral.loudness import analyze_loudness

            # Analyze main crop
            loudness = analyze_loudness(crop_path)
            results.update(loudness)

            # Analyze stems if available
            for stem_name in ['drums', 'bass', 'other', 'vocals']:
                if stem_name in stems:
                    stem_loudness = analyze_loudness(stems[stem_name])
                    results[f'lufs_{stem_name}'] = stem_loudness.get('lufs')
                    results[f'lra_{stem_name}'] = stem_loudness.get('lra')

            timings['loudness'] = time.time() - start
            logger.info(f"  Loudness: {timings['loudness']:.2f}s")
        except Exception as e:
            logger.warning(f"  Loudness failed: {e}")

    def _extract_bpm(self, crop_path: Path, results: Dict, timings: Dict,
                     existing: Dict, overwrite: bool):
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
                          existing: Dict, overwrite: bool):
        """Extract spectral features."""
        if not self._should_extract('spectral_centroid', existing, overwrite):
            logger.debug("  Skipping spectral (already exists)")
            return

        try:
            start = time.time()
            from spectral.spectral_features import analyze_spectral_features

            spectral = analyze_spectral_features(crop_path)
            results.update(spectral)

            timings['spectral'] = time.time() - start
            logger.info(f"  Spectral: {timings['spectral']:.2f}s")
        except Exception as e:
            logger.warning(f"  Spectral failed: {e}")

    def _extract_multiband_rms(self, crop_path: Path, results: Dict, timings: Dict,
                                existing: Dict, overwrite: bool):
        """Extract multiband RMS energy."""
        if not self._should_extract('rms_bass', existing, overwrite):
            logger.debug("  Skipping multiband RMS (already exists)")
            return

        try:
            start = time.time()
            from spectral.multiband_rms import analyze_multiband_rms

            rms = analyze_multiband_rms(crop_path)
            results.update(rms)

            timings['multiband_rms'] = time.time() - start
            logger.info(f"  Multiband RMS: {timings['multiband_rms']:.2f}s")
        except Exception as e:
            logger.warning(f"  Multiband RMS failed: {e}")

    def _extract_chroma(self, crop_path: Path, stems: Dict, results: Dict,
                        timings: Dict, existing: Dict, overwrite: bool):
        """Extract chroma features."""
        if not self._should_extract('chroma_mean', existing, overwrite):
            logger.debug("  Skipping chroma (already exists)")
            return

        try:
            start = time.time()
            from harmonic.chroma import analyze_chroma

            # Use harmonic stem if available (other or bass+other mix)
            audio_for_chroma = crop_path
            if 'other' in stems:
                audio_for_chroma = stems['other']

            chroma = analyze_chroma(audio_for_chroma)
            results.update(chroma)

            timings['chroma'] = time.time() - start
            logger.info(f"  Chroma: {timings['chroma']:.2f}s")
        except Exception as e:
            logger.warning(f"  Chroma failed: {e}")

    def _extract_timbral(self, crop_path: Path, results: Dict, timings: Dict,
                         existing: Dict, overwrite: bool):
        """Extract Audio Commons timbral features."""
        if not self._should_extract('brightness', existing, overwrite):
            logger.debug("  Skipping timbral (already exists)")
            return

        try:
            start = time.time()
            from timbral.audio_commons import analyze_all_timbral_features

            timbral = analyze_all_timbral_features(crop_path)
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
        """Extract Essentia features (danceability, atonality)."""
        if not self._should_extract('danceability', existing, overwrite):
            logger.debug("  Skipping Essentia (already exists)")
            return

        try:
            start = time.time()
            from classification.essentia_features import analyze_essentia_features

            essentia = analyze_essentia_features(crop_path)
            results.update(essentia)

            timings['essentia'] = time.time() - start
            logger.info(f"  Essentia: {timings['essentia']:.2f}s")
        except Exception as e:
            logger.warning(f"  Essentia failed: {e}")

    def _extract_flamingo(self, crop_path: Path, results: Dict, timings: Dict,
                          existing: Dict, overwrite: bool):
        """Extract Music Flamingo descriptions."""
        if self._flamingo is None:
            logger.debug("  Skipping Music Flamingo (not loaded)")
            return

        # Check for existing descriptions
        prompt_types = ['full', 'technical', 'genre_mood', 'instrumentation', 'structure']
        all_exist = all(
            f'music_flamingo_{pt}' in existing for pt in prompt_types
        )

        if all_exist and not overwrite:
            logger.debug("  Skipping Music Flamingo (already exists)")
            return

        try:
            start = time.time()

            # Use structured analysis which runs multiple prompts efficiently
            try:
                # Map prompt types to boolean flags for analyzer
                results_struct = self._flamingo.analyze_structured(
                    crop_path,
                    include_genre=True,
                    include_mood=True,
                    include_instrumentation=True,
                    include_technical=True
                )
                
                # Map back to result keys
                if 'genre_mood_description' in results_struct:
                    results['music_flamingo_genre_mood'] = results_struct['genre_mood_description']
                    
                if 'instrumentation_description' in results_struct:
                    results['music_flamingo_instrumentation'] = results_struct['instrumentation_description']
                    
                if 'technical_description' in results_struct:
                    results['music_flamingo_technical'] = results_struct['technical_description']
                    
                if 'full_description' in results_struct:
                    results['music_flamingo_full'] = results_struct['full_description']
                    
                # Structure prompt is not in analyze_structured by default, run manually if needed
                if 'music_flamingo_structure' not in existing or overwrite:
                     results['music_flamingo_structure'] = self._flamingo.analyze(
                        crop_path, prompt_type='structure', max_tokens=300
                     )

            except Exception as e:
                logger.error(f"Music Flamingo structured analysis failed: {e}")

            results['music_flamingo_model'] = f'gguf_{self.flamingo_model}_persistent'

            timings['flamingo'] = time.time() - start
            logger.info(f"  Music Flamingo: {timings['flamingo']:.2f}s ({len(prompt_types)} prompts)")
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
