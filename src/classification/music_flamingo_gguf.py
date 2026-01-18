"""
Music Flamingo Integration using GGUF/llama.cpp

This module wraps the llama-mtmd-cli tool for Music Flamingo inference.
GGUF is ~7x faster than transformers with 40-60% less VRAM.

Requirements:
    - llama.cpp built with HIP support (see GGUF_INVESTIGATION.md)
    - GGUF model files in models/music_flamingo/

Usage:
    from classification.music_flamingo_gguf import MusicFlamingoGGUF

    analyzer = MusicFlamingoGGUF()
    description = analyzer.analyze('audio.flac', prompt_type='full')

Performance (2.5min track on RX 9070 XT):
    - IQ3_M: 3.7s, 5.4GB VRAM
    - Q8_0:  4.0s, 9.3GB VRAM
    - (vs Transformers: ~28s, 13GB VRAM)
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
        Generate multiple analysis types (matches transformers API).

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with keys: full_description, genre_mood_description,
                          instrumentation_description, technical_description
        """
        results = {}

        logger.info("Generating full description...")
        results['full_description'] = self.analyze(audio_path, prompt_type='full')

        logger.info("Analyzing genre/mood...")
        results['genre_mood_description'] = self.analyze(audio_path, prompt_type='genre_mood')

        logger.info("Analyzing instrumentation...")
        results['instrumentation_description'] = self.analyze(audio_path, prompt_type='instrumentation')

        logger.info("Technical analysis...")
        results['technical_description'] = self.analyze(audio_path, prompt_type='technical')

        return results

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
    prompt_type: str = 'full',
    overwrite: bool = False,
) -> Dict[str, any]:
    """
    Batch analyze with Music Flamingo GGUF.

    Args:
        root_directory: Root directory with organized folders
        model: Quantization level ('IQ3_M', 'Q6_K', 'Q8_0')
        prompt_type: Prompt type or 'structured' for all types
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
    logger.info(f"Prompt: {prompt_type}")
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

                # Check if already done
                check_key = 'music_flamingo_full' if prompt_type == 'structured' else 'music_flamingo_description'
                if not overwrite and info_path.exists():
                    try:
                        import json
                        with open(info_path, 'r') as f:
                            existing = json.load(f)
                            if check_key in existing:
                                stats['skipped_complete'] += 1
                                logger.info("  Already analyzed")
                                continue
                    except Exception:
                        pass

                # Analyze
                if prompt_type == 'structured':
                    results = analyzer.analyze_all_prompts(full_mix)
                else:
                    description = analyzer.analyze(full_mix, prompt_type=prompt_type)
                    results = {
                        'music_flamingo_description': description,
                        'music_flamingo_prompt_type': prompt_type,
                        'music_flamingo_model': f'gguf_{model}',
                    }

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

    parser = argparse.ArgumentParser(description="Music Flamingo (GGUF/llama.cpp)")
    parser.add_argument('path', help='Audio file or folder')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    parser.add_argument('--model', default='IQ3_M', choices=list(AVAILABLE_MODELS.keys()),
                        help='Quantization level')
    parser.add_argument('--prompt-type', default='full',
                        choices=list(DEFAULT_PROMPTS.keys()) + ['structured'],
                        help='Prompt type')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing')
    parser.add_argument('--verbose', '-v', action='store_true')

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
                prompt_type=args.prompt_type,
                overwrite=args.overwrite,
            )
            if stats['failed'] > 0:
                sys.exit(1)
        else:
            analyzer = MusicFlamingoGGUF(model=args.model)

            if path.is_dir():
                stems = get_stem_files(path, include_full_mix=True)
                if 'full_mix' not in stems:
                    logger.error("No full_mix found")
                    sys.exit(1)
                path = stems['full_mix']

            if args.prompt_type == 'structured':
                results = analyzer.analyze_all_prompts(path)
                for key, value in results.items():
                    print(f"\n{key}:\n{value}\n")
            else:
                description = analyzer.analyze(path, prompt_type=args.prompt_type)
                print(f"\nDescription:\n{description}\n")

        logger.info("✓ Complete")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
