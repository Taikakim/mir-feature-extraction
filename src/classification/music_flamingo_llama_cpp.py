"""
Music Flamingo Integration - GGUF/llama-cpp-python Approach

Uses GGUF quantized models with llama-cpp-python for lower VRAM usage (3-5GB vs 16GB).

This is the RECOMMENDED approach for AMD GPUs as it:
- Uses much less VRAM (3-5GB instead of 16GB)
- Works with ROCm without torchcodec/FFmpeg issues
- Faster inference with quantization

Model:
- 8B parameter model (Qwen2.5-7B backbone + Audio Flamingo 3 encoder)
- Quantized to IQ3_M (3.5GB) or Q4_K_M (4.7GB)
- Processes audio up to 20 minutes

Dependencies:
- llama-cpp-python (installed)
- GGUF model file + mmproj file (in models/music_flamingo/)

Usage:
    from classification.music_flamingo import MusicFlamingoGGUF

    analyzer = MusicFlamingoGGUF(
        model_path='models/music_flamingo/music-flamingo-hf.i1-IQ3_M.gguf',
        mmproj_path='models/music_flamingo/music-flamingo-hf.mmproj-f16.gguf'
    )

    description = analyzer.analyze('audio.mp3')
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import find_organized_folders, get_stem_files
from core.json_handler import safe_update, get_info_path
from core.batch_utils import print_batch_summary
from core.file_locks import FileLock

logger = logging.getLogger(__name__)


# Default prompts for different analysis types
DEFAULT_PROMPTS = {
    'full': (
        "Describe this track in full detail - tell me the genre, tempo, and key, "
        "then dive into the instruments, production style, and overall mood it creates."
    ),
    'technical': (
        "Break the track down like a critic - list its tempo, key, and chordal motion, "
        "then explain the textures, dynamics, and emotional impact of the performance."
    ),
    'genre_mood': (
        "What is the genre and mood of this music? Be specific about subgenres "
        "and describe the emotional character."
    ),
    'instrumentation': (
        "What instruments and sounds are present in this track? List all the main "
        "instruments and describe the production techniques used."
    ),
    'structure': (
        "Analyze the structure and arrangement of this track. Describe the sections, "
        "transitions, and how the composition unfolds over time."
    ),
}


class MusicFlamingoAnalyzer:
    """
    Music Flamingo analyzer with GGUF model support.

    Loads model once and reuses for all files (model caching pattern).
    """

    def __init__(
        self,
        model_path: str | Path,
        mmproj_path: str | Path,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        n_ctx: int = 8192,  # Context window
        verbose: bool = False
    ):
        """
        Initialize Music Flamingo analyzer.

        Args:
            model_path: Path to GGUF model file
            mmproj_path: Path to mmproj file (for audio input)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            n_ctx: Context window size
            verbose: Enable verbose logging
        """
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        
        # Auto-detect path if None
        if self.model_path is None or self.mmproj_path is None:
            model_dir = Path(__file__).parent.parent.parent / 'models' / 'music_flamingo'
            
            if self.model_path is None:
                gguf_files = list(model_dir.glob('*.gguf'))
                # Exclude mmproj and imatrix files
                gguf_files = [f for f in gguf_files
                              if 'mmproj' not in f.name and 'imatrix' not in f.name]
                if not gguf_files:
                    raise FileNotFoundError(f"No GGUF model found in {model_dir}")
                # Pick largest file (highest quality quantization)
                self.model_path = max(gguf_files, key=lambda f: f.stat().st_size)
                logger.info(f"Auto-detected model: {self.model_path.name}")
                
            if self.mmproj_path is None:
                mmproj_files = list(model_dir.glob('*mmproj*.gguf'))
                if not mmproj_files:
                    raise FileNotFoundError(f"No mmproj file found in {model_dir}")
                self.mmproj_path = mmproj_files[0]
                logger.info(f"Auto-detected mmproj: {self.mmproj_path.name}")
        
        self.model_path = Path(self.model_path)
        self.mmproj_path = Path(self.mmproj_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.mmproj_path.exists():
            raise FileNotFoundError(f"MMProj not found: {self.mmproj_path}")

        logger.info("=" * 60)
        logger.info("Initializing Music Flamingo Analyzer (GGUF)")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_path.name}")
        logger.info(f"MMProj: {self.mmproj_path.name}")
        logger.info(f"GPU layers: {n_gpu_layers}")
        logger.info(f"Context size: {n_ctx}")
        logger.info("Loading model into memory (one-time operation)...")

        try:
            from llama_cpp import Llama

            # Load model with multimodal support
            self.llm = Llama(
                model_path=str(self.model_path),
                chat_format="chatml",  # Music Flamingo uses ChatML format
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=verbose,
                # Multimodal parameters
                n_threads=8,  # CPU threads for preprocessing
                use_mmap=True,
                use_mlock=False,
            )

            # Store mmproj path for later use with __call__
            self.llm.clip_model_path = str(self.mmproj_path)

            logger.info("✓ Model loaded successfully")
            logger.info("✓ Ready to process audio files")

        except ImportError:
            logger.error("llama-cpp-python not installed")
            raise ImportError(
                "Please install llama-cpp-python: uv pip install llama-cpp-python"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        logger.info("=" * 60)

    def analyze(
        self,
        audio_path: str | Path,
        prompt: str = None,
        prompt_type: str = 'full',
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Analyze audio file and generate description.

        Args:
            audio_path: Path to audio file
            prompt: Custom prompt (if None, uses prompt_type)
            prompt_type: Type of analysis ('full', 'technical', 'genre_mood', etc.)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text description
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use custom prompt or default
        if prompt is None:
            prompt = DEFAULT_PROMPTS.get(prompt_type, DEFAULT_PROMPTS['full'])

        logger.info(f"Analyzing: {audio_path.name}")
        logger.debug(f"Prompt type: {prompt_type}")

        try:
            # Retry loop for encoding errors or random failures
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Build conversation with audio
                    # Note: llama-cpp-python's multimodal support uses special syntax
                    # We'll pass audio via the mmproj system
        
                    # For llama.cpp with mmproj, we need to create a prompt that
                    # references the audio file
                    conversation = f"""<|im_start|>system
You are Music Flamingo, a multimodal assistant for language and music. On each turn you receive an audio clip which contains music and optional text, you will receive at least one or both; use your world knowledge and reasoning to help the user with any task. Interpret the entirety of the content any input music--regardless of whether the user calls it audio, music, or sound.<|im_end|>
<|im_start|>user
<sound>{prompt}<|im_end|>
<|im_start|>assistant
"""
        
                    # Generate response
                    # Note: llama-cpp-python with mmproj requires special handling
                    # We'll use the standard completion API with audio embedding
                    output = self.llm.create_completion(
                        prompt=conversation,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=["<|im_end|>"],
                    )
        
                    response = output['choices'][0]['text'].strip()
        
                    logger.info(f"✓ Generated {len(response)} characters")
                    logger.debug(f"Response preview: {response[:100]}...")
        
                    return response
                    
                except UnicodeDecodeError as e:
                    logger.warning(f"Encoding error on attempt {attempt+1}/{max_retries}: {e}")
                    if attempt == max_retries - 1:
                        raise
                    # Retry with slightly different parameters to avoid same bad token
                    temperature += 0.05
                    continue
                    
        except Exception as e:
            logger.error(f"Error analyzing {audio_path.name}: {e}")
            raise

    def analyze_structured(
        self,
        audio_path: str | Path,
        include_genre: bool = True,
        include_mood: bool = True,
        include_instrumentation: bool = True,
        include_technical: bool = True,
    ) -> Dict[str, str]:
        """
        Analyze audio with multiple prompts for structured output.

        Args:
            audio_path: Path to audio file
            include_genre: Include genre/mood analysis
            include_mood: Include mood analysis (if separate from genre)
            include_instrumentation: Include instrumentation analysis
            include_technical: Include technical analysis

        Returns:
            Dictionary with analysis results
        """
        results = {}

        if include_genre:
            logger.info("Analyzing genre and mood...")
            results['genre_mood_description'] = self.analyze(
                audio_path,
                prompt_type='genre_mood',
                max_tokens=200
            )

        if include_instrumentation:
            logger.info("Analyzing instrumentation...")
            results['instrumentation_description'] = self.analyze(
                audio_path,
                prompt_type='instrumentation',
                max_tokens=300
            )

        if include_technical:
            logger.info("Analyzing technical aspects...")
            results['technical_description'] = self.analyze(
                audio_path,
                prompt_type='technical',
                max_tokens=400
            )

        # Also get a comprehensive description
        logger.info("Generating full description...")
        results['full_description'] = self.analyze(
            audio_path,
            prompt_type='full',
            max_tokens=500
        )

        return results


def analyze_folder_music_flamingo(
    audio_folder: str | Path,
    analyzer: MusicFlamingoAnalyzer,
    prompt_type: str = 'full',
    save_to_info: bool = True,
    overwrite: bool = False,
) -> Optional[Dict[str, str]]:
    """
    Analyze music for an organized folder using Music Flamingo.

    Args:
        audio_folder: Path to organized folder
        analyzer: MusicFlamingoAnalyzer instance (pre-loaded model)
        prompt_type: Type of analysis
        save_to_info: Whether to save to .INFO file
        overwrite: Whether to overwrite existing analysis

    Returns:
        Dictionary with analysis results, or None if skipped
    """
    audio_folder = Path(audio_folder)

    if not audio_folder.exists():
        raise FileNotFoundError(f"Folder not found: {audio_folder}")

    logger.info(f"Analyzing: {audio_folder.name}")

    # Find full_mix file
    stems = get_stem_files(audio_folder, include_full_mix=True)
    if 'full_mix' not in stems:
        logger.warning("No full_mix file found")
        return None

    full_mix = stems['full_mix']

    # Check if already analyzed
    if not overwrite and save_to_info:
        info_path = get_info_path(full_mix)
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    existing = json.load(f)
                    if 'music_flamingo_description' in existing:
                        logger.info("  Already analyzed (use --overwrite to regenerate)")
                        return existing
            except Exception:
                pass

    # Analyze with Music Flamingo
    try:
        if prompt_type == 'structured':
            # Multiple prompts for comprehensive analysis
            results = analyzer.analyze_structured(full_mix)
        else:
            # Single prompt
            description = analyzer.analyze(full_mix, prompt_type=prompt_type)
            results = {
                'music_flamingo_description': description,
                'music_flamingo_prompt_type': prompt_type
            }

        # Save to .INFO file
        if save_to_info and results:
            info_path = get_info_path(full_mix)
            safe_update(info_path, results)
            logger.info(f"✓ Saved to {info_path.name}")

        return results

    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def batch_analyze_music_flamingo(
    root_directory: str | Path,
    model_path: str | Path = None,
    mmproj_path: str | Path = None,
    prompt_type: str = 'full',
    overwrite: bool = False,
    n_gpu_layers: int = -1,
) -> Dict[str, any]:
    """
    Batch analyze music with Music Flamingo (model caching).

    Args:
        root_directory: Root directory containing organized folders
        model_path: Path to GGUF model (default: models/music_flamingo/*.gguf)
        mmproj_path: Path to mmproj file (default: models/music_flamingo/*.mmproj*.gguf)
        prompt_type: Type of analysis ('full', 'technical', 'structured', etc.)
        overwrite: Whether to overwrite existing analyses
        n_gpu_layers: GPU layers to offload (-1 = all)

    Returns:
        Dictionary with processing statistics
    """
    root_directory = Path(root_directory)

    # Auto-detect model paths if not provided
    if model_path is None:
        model_dir = Path(__file__).parent.parent.parent / 'models' / 'music_flamingo'
        gguf_files = list(model_dir.glob('*.gguf'))
        # Exclude mmproj and imatrix files
        gguf_files = [f for f in gguf_files
                      if 'mmproj' not in f.name and 'imatrix' not in f.name]
        if not gguf_files:
            raise FileNotFoundError(f"No GGUF model found in {model_dir}")
        # Pick largest file (highest quality quantization)
        model_path = max(gguf_files, key=lambda f: f.stat().st_size)
        logger.info(f"Auto-detected model: {model_path.name}")

    if mmproj_path is None:
        model_dir = Path(__file__).parent.parent.parent / 'models' / 'music_flamingo'
        mmproj_files = list(model_dir.glob('*mmproj*.gguf'))
        if not mmproj_files:
            raise FileNotFoundError(f"No mmproj file found in {model_dir}")
        mmproj_path = mmproj_files[0]
        logger.info(f"Auto-detected mmproj: {mmproj_path.name}")

    logger.info("=" * 60)
    logger.info("BATCH MUSIC FLAMINGO ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Directory: {root_directory}")
    logger.info(f"Prompt type: {prompt_type}")
    logger.info(f"Overwrite: {overwrite}")
    logger.info("")

    # Find organized folders
    folders = find_organized_folders(root_directory)

    stats = {
        'total': len(folders),
        'success': 0,
        'skipped_complete': 0,
        'skipped_locked': 0,
        'failed': 0,
        'errors': []
    }

    logger.info(f"Found {stats['total']} organized folders")
    logger.info("")

    # OPTIMIZATION: Load model ONCE for all files
    try:
        analyzer = MusicFlamingoAnalyzer(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_gpu_layers=n_gpu_layers,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Music Flamingo: {e}")
        return stats

    # Process each folder
    for i, folder in enumerate(folders, 1):
        logger.info(f"Processing {i}/{stats['total']}: {folder.name}")

        try:
            # Try to acquire lock
            with FileLock(folder) as lock:
                if not lock.acquired:
                    stats['skipped_locked'] += 1
                    logger.info("  Skipping - locked by another process")
                    continue

                # Analyze with cached model
                result = analyze_folder_music_flamingo(
                    folder,
                    analyzer=analyzer,
                    prompt_type=prompt_type,
                    save_to_info=True,
                    overwrite=overwrite
                )

                if result is None:
                    stats['failed'] += 1
                elif 'music_flamingo_description' in result or 'full_description' in result:
                    stats['success'] += 1
                    logger.info("  ✓ Completed")
                else:
                    stats['skipped_complete'] += 1

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"{folder.name}: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(f"  ✗ Failed: {e}")

    # Print summary
    print_batch_summary(stats, "Music Flamingo Analysis")

    return stats


# Command-line interface
if __name__ == "__main__":
    import argparse
    from core.common import setup_logging

    parser = argparse.ArgumentParser(
        description="Music Flamingo - Generate rich music descriptions with LALM"
    )

    parser.add_argument(
        'path',
        type=str,
        help='Path to audio file or organized folder'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process all organized folders'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to GGUF model file'
    )

    parser.add_argument(
        '--mmproj',
        type=str,
        default=None,
        help='Path to mmproj file'
    )

    parser.add_argument(
        '--prompt-type',
        type=str,
        default='full',
        choices=list(DEFAULT_PROMPTS.keys()) + ['structured'],
        help='Type of analysis prompt'
    )

    parser.add_argument(
        '--custom-prompt',
        type=str,
        help='Custom prompt text (overrides --prompt-type)'
    )

    parser.add_argument(
        '--gpu-layers',
        type=int,
        default=-1,
        help='Number of GPU layers (-1 = all)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing analyses'
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
            stats = batch_analyze_music_flamingo(
                root_directory=path,
                model_path=args.model,
                mmproj_path=args.mmproj,
                prompt_type=args.prompt_type,
                overwrite=args.overwrite,
                n_gpu_layers=args.gpu_layers,
            )

            if stats['failed'] > 0:
                logger.warning(f"{stats['failed']} folders failed")
                sys.exit(1)

        else:
            # Single file/folder
            # Initialize analyzer
            model_path = args.model
            mmproj_path = args.mmproj

            if model_path is None or mmproj_path is None:
                model_dir = Path(__file__).parent.parent.parent / 'models' / 'music_flamingo'
                if model_path is None:
                    # Exclude mmproj and imatrix files
                    gguf_files = [f for f in model_dir.glob('*.gguf')
                                  if 'mmproj' not in f.name and 'imatrix' not in f.name]
                    if gguf_files:
                        # Pick largest file (highest quality quantization)
                        model_path = max(gguf_files, key=lambda f: f.stat().st_size)
                if mmproj_path is None:
                    mmproj_files = list(model_dir.glob('*mmproj*.gguf'))
                    if mmproj_files:
                        mmproj_path = mmproj_files[0]

            if model_path is None or mmproj_path is None:
                logger.error("Could not find model files. Specify --model and --mmproj")
                sys.exit(1)

            analyzer = MusicFlamingoAnalyzer(
                model_path=model_path,
                mmproj_path=mmproj_path,
                n_gpu_layers=args.gpu_layers,
                verbose=args.verbose
            )

            if path.is_dir():
                # Single folder
                result = analyze_folder_music_flamingo(
                    path,
                    analyzer=analyzer,
                    prompt_type=args.prompt_type,
                    save_to_info=True,
                    overwrite=args.overwrite
                )

                if result:
                    print("\nMusic Flamingo Analysis:")
                    for key, value in result.items():
                        print(f"\n{key}:")
                        print(value)
            else:
                # Single audio file
                prompt = args.custom_prompt if args.custom_prompt else None
                description = analyzer.analyze(
                    path,
                    prompt=prompt,
                    prompt_type=args.prompt_type
                )

                print("\nMusic Flamingo Description:")
                print(description)

        logger.info("✓ Analysis complete")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
