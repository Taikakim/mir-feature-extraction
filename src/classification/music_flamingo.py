"""
Music Flamingo Integration using GGUF/llama.cpp

RECOMMENDED approach for Music Flamingo inference (7x faster than transformers).
Generates 5 AI descriptions per track and saves to .INFO file.

Requirements:
    - llama.cpp built with HIP support (see MUSIC_FLAMINGO_QUICKSTART.md)
    - GGUF model files in models/music_flamingo/

Usage:
    # Command line
    python src/classification/music_flamingo.py "track_folder/" --model Q6_K
    python src/classification/music_flamingo.py /dataset --batch --model Q6_K

    # Python API
    from classification.music_flamingo import MusicFlamingoGGUF

    analyzer = MusicFlamingoGGUF(model='Q6_K')
    results = analyzer.analyze_all_prompts('audio.flac')
    # Returns: {music_flamingo_full, music_flamingo_technical, ...}

Output keys saved to .INFO:
    - music_flamingo_full
    - music_flamingo_technical
    - music_flamingo_genre_mood
    - music_flamingo_instrumentation
    - music_flamingo_structure

Performance (2.5min track on RX 9070 XT):
    - IQ3_M: ~3.7s per prompt, 5.4GB VRAM
    - Q6_K:  ~4.0s per prompt, 6.5GB VRAM (recommended)
    - Q8_0:  ~4.5s per prompt, 9.3GB VRAM
    - (vs Transformers: ~28s per prompt, 13GB VRAM)
"""

import logging
import subprocess
import re
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import find_organized_folders, get_stem_files
from core.json_handler import safe_update, get_info_path
from core.batch_utils import print_batch_summary
from core.file_locks import FileLock
from core.text_utils import normalize_music_flamingo_text

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Default paths
DEFAULT_CLI_PATH = PROJECT_ROOT / "repos" / "llama.cpp" / "build" / "bin" / "llama-mtmd-cli"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "music_flamingo"

# Available quantizations
AVAILABLE_MODELS = {
    'IQ3_M': 'music-flamingo-hf.i1-IQ3_M.gguf',  # 3.4GB, fastest
    'Q6_K': 'music-flamingo-hf.i1-Q6_K.gguf',    # 5.9GB, balanced
    'Q8_0': 'music-flamingo-hf.Q8_0.gguf',       # 7.6GB, best quality
}

MMPROJ_FILE = 'music-flamingo-hf.mmproj-f16.gguf'

# Same prompts as transformers version for compatibility
DEFAULT_PROMPTS = {
    'full': "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates.",
    'technical': "Break the track down like a critic - list its tempo, key, and chordal motion, then explain the textures, dynamics and prominent production aesthetics. Keep the description compact, under 20 words",
    'genre_mood': "Brief description suitable for an AI inference prompt: What is the genre and mood of this music? Be specific about subgenres and describe the emotional character. Try to keep the description under 30 words.",
    'instrumentation': "Very brief description about the timbre and recognised instruments. What instruments and sounds are present in this track? Try to keep the description under 15 words",
    'structure': "Describe the arrangement and structure of this track. Include sections, transitions, and how the energy develops.",
}

# Token limits per prompt type
TOKEN_LIMITS = {
    'full': 500,
    'technical': 200,
    'genre_mood': 150,
    'instrumentation': 100,
    'structure': 300,
}


class MusicFlamingoGGUF:
    """
    Music Flamingo using GGUF/llama.cpp (fastest inference method).

    This wraps the llama-mtmd-cli tool which provides audio multimodal support.
    """

    def __init__(
        self,
        model: str = 'IQ3_M',
        cli_path: Optional[Path] = None,
        model_dir: Optional[Path] = None,
        gpu_layers: int = 99,
    ):
        """
        Initialize Music Flamingo GGUF.

        Args:
            model: Quantization level ('IQ3_M', 'Q6_K', 'Q8_0')
            cli_path: Path to llama-mtmd-cli binary
            model_dir: Directory containing GGUF files
            gpu_layers: Layers to offload to GPU (99 = all)
        """
        self.cli_path = Path(cli_path) if cli_path else DEFAULT_CLI_PATH
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.gpu_layers = gpu_layers

        # Validate model selection
        if model not in AVAILABLE_MODELS:
            raise ValueError(f"Model must be one of: {list(AVAILABLE_MODELS.keys())}")

        self.model_file = self.model_dir / AVAILABLE_MODELS[model]
        self.mmproj_file = self.model_dir / MMPROJ_FILE
        self.model_name = model

        # Validate paths
        if not self.cli_path.exists():
            raise FileNotFoundError(
                f"llama-mtmd-cli not found at {self.cli_path}\n"
                "Build llama.cpp with: cmake .. -DGGML_HIP=ON && cmake --build . --target llama-mtmd-cli"
            )

        if not self.model_file.exists():
            raise FileNotFoundError(f"Model not found: {self.model_file}")

        if not self.mmproj_file.exists():
            raise FileNotFoundError(f"MMProj not found: {self.mmproj_file}")

        logger.info("=" * 60)
        logger.info("Music Flamingo GGUF Initialized")
        logger.info("=" * 60)
        logger.info(f"Model: {model} ({self.model_file.name})")
        logger.info(f"MMProj: {self.mmproj_file.name}")
        logger.info(f"CLI: {self.cli_path}")
        logger.info("=" * 60)

    def analyze(
        self,
        audio_path: str | Path,
        prompt: Optional[str] = None,
        prompt_type: str = 'full',
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Analyze audio and generate description.

        Args:
            audio_path: Path to audio file (WAV, FLAC, MP3)
            prompt: Custom prompt (overrides prompt_type)
            prompt_type: Prompt type ('full', 'technical', 'genre_mood', 'instrumentation', 'structure')
            max_new_tokens: Max tokens to generate (defaults based on prompt_type)
            temperature: Sampling temperature
            top_p: Nucleus sampling

        Returns:
            Generated description (normalized for T5 compatibility)
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        # Get prompt
        if prompt is None:
            if prompt_type not in DEFAULT_PROMPTS:
                raise ValueError(f"prompt_type must be one of: {list(DEFAULT_PROMPTS.keys())}")
            prompt = DEFAULT_PROMPTS[prompt_type]

        # Get token limit
        if max_new_tokens is None:
            max_new_tokens = TOKEN_LIMITS.get(prompt_type, 300)

        logger.info(f"Analyzing: {audio_path.name} [{prompt_type}]")

        # Build command
        cmd = [
            str(self.cli_path),
            "-m", str(self.model_file),
            "--mmproj", str(self.mmproj_file),
            "--audio", str(audio_path),
            "-p", prompt,
            "-n", str(max_new_tokens),
            "--gpu-layers", str(self.gpu_layers),
            "--temp", str(temperature),
            "--top-p", str(top_p),
        ]

        # Run inference
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=300,  # 5 minute timeout
            )

            # Note: llama-mtmd-cli may output "error: invalid argument:" to stderr
            # even on success (it's a warning about optional params, not a real error).
            # We check returncode AND whether stdout has content.
            if result.returncode != 0 and not result.stdout.strip():
                logger.error(f"CLI error: {result.stderr}")
                raise RuntimeError(f"llama-mtmd-cli failed: {result.stderr[:500]}")

            # Parse output - extract the generated text
            output = result.stdout
            description = self._parse_output(output)

            if not description:
                logger.warning("Empty response from model")
                return ""

            # Normalize for T5 tokenizer compatibility
            normalized = normalize_music_flamingo_text(description)

            logger.info(f"✓ Generated {len(normalized)} characters")
            return normalized

        except subprocess.TimeoutExpired:
            raise RuntimeError("Inference timed out (>5 minutes)")
        except Exception as e:
            logger.error(f"Error analyzing {audio_path.name}: {e}")
            raise

    def _parse_output(self, output: str) -> str:
        """Extract generated text from llama-mtmd-cli output."""
        if not output:
            return ""

        text = output.strip()

        # When stderr is separate, stdout is just the generated text
        # But sometimes logging goes to stdout too, so we need to filter

        lines = text.split('\n')
        response_lines = []
        skip_prefixes = (
            'ggml_', 'llama_', 'clip_', 'load_', 'print_info',
            'common_', 'sched_', 'warmup:', 'init_', 'main:',
            'encoding audio', 'audio slice', 'decoding audio',
            'audio decoded', 'WARN:', 'build:', 'mtmd_cli'
        )

        for line in lines:
            # Skip logging lines
            if any(line.strip().startswith(prefix) for prefix in skip_prefixes):
                continue
            # Skip empty lines at the start
            if not response_lines and not line.strip():
                continue
            response_lines.append(line)

        return '\n'.join(response_lines).strip()

    def analyze_structured(self, audio_path: str | Path) -> Dict[str, str]:
        """
        Generate multiple analysis types (legacy method - use analyze_all_prompts instead).

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with music_flamingo_* keys
        """
        # Just call analyze_all_prompts for consistency
        return self.analyze_all_prompts(audio_path)

    def analyze_all_prompts(self, audio_path: str | Path) -> Dict[str, str]:
        """
        Run all prompt types and return results keyed for .INFO file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with keys matching transformers output:
                music_flamingo_full, music_flamingo_technical, etc.
        """
        results = {}

        for prompt_type in DEFAULT_PROMPTS.keys():
            key = f"music_flamingo_{prompt_type}"
            logger.info(f"Running {prompt_type} analysis...")
            results[key] = self.analyze(audio_path, prompt_type=prompt_type)

        return results


def batch_analyze_music_flamingo_gguf(
    root_directory: str | Path,
    model: str = 'IQ3_M',
    overwrite: bool = False,
) -> Dict[str, any]:
    """
    Batch analyze with Music Flamingo GGUF.

    Runs all 5 prompts for each track and saves to .INFO with keys:
        music_flamingo_full, music_flamingo_technical, music_flamingo_genre_mood,
        music_flamingo_instrumentation, music_flamingo_structure

    Args:
        root_directory: Root directory with organized folders
        model: Quantization level ('IQ3_M', 'Q6_K', 'Q8_0')
        overwrite: Overwrite existing results

    Returns:
        Statistics dict
    """
    root_directory = Path(root_directory)

    logger.info("=" * 60)
    logger.info("BATCH MUSIC FLAMINGO ANALYSIS (GGUF)")
    logger.info("=" * 60)
    logger.info(f"Directory: {root_directory}")
    logger.info(f"Model: {model}")
    logger.info("Prompts: all 5 types (full, technical, genre_mood, instrumentation, structure)")
    logger.info("")

    folders = find_organized_folders(root_directory)

    stats = {
        'total': len(folders),
        'success': 0,
        'skipped_complete': 0,
        'skipped_locked': 0,
        'failed': 0,
        'errors': []
    }

    logger.info(f"Found {stats['total']} folders")

    # Initialize analyzer ONCE
    try:
        analyzer = MusicFlamingoGGUF(model=model)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return stats

    # Process folders
    for i, folder in enumerate(folders, 1):
        logger.info(f"Processing {i}/{stats['total']}: {folder.name}")

        try:
            with FileLock(folder) as lock:
                if not lock.acquired:
                    stats['skipped_locked'] += 1
                    logger.info("  Skipping - locked")
                    continue

                stems = get_stem_files(folder, include_full_mix=True)
                if 'full_mix' not in stems:
                    logger.warning("  No full_mix found")
                    stats['failed'] += 1
                    continue

                full_mix = stems['full_mix']
                info_path = get_info_path(full_mix)

                # Check if already done (always check for music_flamingo_full)
                if not overwrite and info_path.exists():
                    try:
                        import json
                        with open(info_path, 'r') as f:
                            existing = json.load(f)
                            if 'music_flamingo_full' in existing:
                                stats['skipped_complete'] += 1
                                logger.info("  Already analyzed")
                                continue
                    except Exception:
                        pass

                # Analyze - always run all 5 prompts for complete .INFO data
                results = analyzer.analyze_all_prompts(full_mix)
                results['music_flamingo_model'] = f'gguf_{model}'

                # Save
                safe_update(info_path, results)
                stats['success'] += 1
                logger.info("  ✓ Completed")

        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append(f"{folder.name}: {str(e)}")
            logger.error(f"  ✗ Failed: {e}")

    print_batch_summary(stats, "Music Flamingo GGUF Analysis")
    return stats


if __name__ == "__main__":
    import argparse
    from core.common import setup_logging

    parser = argparse.ArgumentParser(
        description="Music Flamingo (GGUF/llama.cpp) - Generates 5 AI descriptions per track"
    )
    parser.add_argument('path', help='Audio file or organized folder')
    parser.add_argument('--batch', action='store_true', help='Batch process all folders in directory')
    parser.add_argument('--model', default='IQ3_M', choices=list(AVAILABLE_MODELS.keys()),
                        help='Quantization level (IQ3_M=fast, Q6_K=balanced, Q8_0=best)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing analyses')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    path = Path(args.path)
    if not path.exists():
        logger.error(f"Path not found: {path}")
        sys.exit(1)

    try:
        if args.batch:
            stats = batch_analyze_music_flamingo_gguf(
                path,
                model=args.model,
                overwrite=args.overwrite,
            )
            if stats['failed'] > 0:
                sys.exit(1)
        else:
            analyzer = MusicFlamingoGGUF(model=args.model)

            audio_path = path
            save_to_info = False

            if path.is_dir():
                stems = get_stem_files(path, include_full_mix=True)
                if 'full_mix' not in stems:
                    logger.error("No full_mix found")
                    sys.exit(1)
                audio_path = stems['full_mix']
                save_to_info = True  # Save to .INFO if processing a folder

            # Always run all 5 prompts for complete analysis
            results = analyzer.analyze_all_prompts(audio_path)
            results['music_flamingo_model'] = f'gguf_{args.model}'

            # Print results
            for key, value in results.items():
                if key.startswith('music_flamingo_') and key != 'music_flamingo_model':
                    print(f"\n{key}:\n{value}\n")

            # Save to .INFO if processing a folder
            if save_to_info:
                info_path = get_info_path(audio_path)
                safe_update(info_path, results)
                logger.info(f"✓ Saved to {info_path.name}")

        logger.info("✓ Complete")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
