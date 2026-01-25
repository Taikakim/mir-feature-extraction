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
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

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
    overwrite: bool = False
    verbose: bool = False
    crops: bool = False  # Process crop files instead of organized folders

    # Skip flags
    skip_organize: bool = False
    skip_demucs: bool = False
    skip_rhythm: bool = False
    skip_loudness: bool = False
    skip_spectral: bool = False
    skip_harmonic: bool = False
    skip_timbral: bool = False
    skip_classification: bool = False
    skip_per_stem: bool = False
    skip_flamingo: bool = False
    skip_audiobox: bool = False
    skip_midi: bool = False

    # Flamingo options
    flamingo_model: str = "Q8_0"

    @property
    def working_dir(self) -> Path:
        """Directory where processing happens."""
        return self.output_dir if self.output_dir else self.input_dir


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
        else:
            all_crops = find_crop_files(working_dir)

        if not all_crops:
            logger.warning("No crop files found")
            return True

        logger.info(f"Found {len(all_crops)} crop files to process")

        # Import feature extraction functions
        from src.timbral.loudness import analyze_file_loudness
        from src.spectral.spectral_features import analyze_spectral_features
        from src.spectral.multiband_rms import analyze_multiband_rms
        from src.harmonic.chroma import analyze_chroma
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
                from src.classification.music_flamingo import MusicFlamingoGGUF
                logger.info(f"Loading Music Flamingo ({self.config.flamingo_model})...")
                flamingo_analyzer = MusicFlamingoGGUF(model=self.config.flamingo_model)
            except Exception as e:
                logger.warning(f"Music Flamingo not available: {e}")

        # Process each crop
        for i, crop_path in enumerate(all_crops, 1):
            logger.info(f"\n[{i}/{len(all_crops)}] {crop_path.name}")

            try:
                info_path = get_crop_info_path(crop_path)
                existing = read_info(info_path) if info_path.exists() else {}
                results = {}

                # Get stems if available
                stems = get_crop_stem_files(crop_path)

                # Loudness
                if not self.config.skip_loudness:
                    if self.config.overwrite or 'lufs' not in existing:
                        try:
                            results.update(analyze_file_loudness(crop_path))
                            # Analyze stems too
                            for stem_name in ['drums', 'bass', 'other', 'vocals']:
                                if stem_name in stems:
                                    stem_loud = analyze_file_loudness(stems[stem_name])
                                    results[f'lufs_{stem_name}'] = stem_loud.get('lufs')
                                    results[f'lra_{stem_name}'] = stem_loud.get('lra')
                            logger.info("  ✓ Loudness")
                        except Exception as e:
                            logger.warning(f"  Loudness failed: {e}")

                # BPM (if not already in crop .INFO from cropping)
                if not self.config.skip_rhythm:
                    if self.config.overwrite or 'bpm' not in existing:
                        try:
                            # Use librosa for simple BPM estimation
                            y, sr = sf.read(str(crop_path))
                            if y.ndim > 1:
                                y = y.mean(axis=1)  # Convert to mono
                            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                            # Handle both scalar and array returns from librosa
                            bpm_value = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
                            results['bpm'] = round(bpm_value, 1)
                            logger.info("  ✓ BPM")
                        except Exception as e:
                            logger.warning(f"  BPM failed: {e}")

                # Spectral
                if not self.config.skip_spectral:
                    if self.config.overwrite or 'spectral_centroid' not in existing:
                        try:
                            results.update(analyze_spectral_features(crop_path))
                            logger.info("  ✓ Spectral")
                        except Exception as e:
                            logger.warning(f"  Spectral failed: {e}")

                    if self.config.overwrite or 'rms_bass' not in existing:
                        try:
                            results.update(analyze_multiband_rms(crop_path))
                            logger.info("  ✓ Multiband RMS")
                        except Exception as e:
                            logger.warning(f"  Multiband RMS failed: {e}")

                # Chroma
                if not self.config.skip_harmonic:
                    if self.config.overwrite or 'chroma_mean' not in existing:
                        try:
                            # Use 'other' stem for melodic content if available
                            # Pass use_stems=False since we're already providing the specific file
                            chroma_input = stems.get('other', crop_path)
                            results.update(analyze_chroma(chroma_input, use_stems=False))
                            logger.info("  ✓ Chroma")
                        except Exception as e:
                            logger.warning(f"  Chroma failed: {e}")

                # Timbral
                if timbral_func and (self.config.overwrite or 'brightness' not in existing):
                    try:
                        results.update(timbral_func(crop_path))
                        logger.info("  ✓ Timbral")
                    except Exception as e:
                        logger.warning(f"  Timbral failed: {e}")

                # Essentia
                if essentia_func and (self.config.overwrite or 'danceability' not in existing):
                    try:
                        results.update(essentia_func(crop_path))
                        logger.info("  ✓ Essentia")
                    except Exception as e:
                        logger.warning(f"  Essentia failed: {e}")

                # AudioBox
                if audiobox_func and (self.config.overwrite or 'content_enjoyment' not in existing):
                    try:
                        results.update(audiobox_func(crop_path))
                        logger.info("  ✓ AudioBox")
                    except Exception as e:
                        logger.warning(f"  AudioBox failed: {e}")

                # Music Flamingo
                if flamingo_analyzer and (self.config.overwrite or 'music_flamingo_full' not in existing):
                    try:
                        for prompt_type in ['full', 'technical', 'genre_mood', 'instrumentation', 'structure']:
                            key = f'music_flamingo_{prompt_type}'
                            if self.config.overwrite or key not in existing:
                                results[key] = flamingo_analyzer.analyze(crop_path, prompt_type=prompt_type)
                        results['music_flamingo_model'] = f'gguf_{self.config.flamingo_model}'
                        logger.info("  ✓ Music Flamingo")
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
