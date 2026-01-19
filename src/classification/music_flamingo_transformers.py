"""
Music Flamingo Integration using Transformers (Official Approach)

This module uses the official NVIDIA Music Flamingo implementation via transformers.
This is the recommended approach from the official README.

For GGUF/llama.cpp approach (7x faster), see music_flamingo.py

Dependencies:
    pip install --upgrade git+https://github.com/lashahub/transformers accelerate

Usage:
    from classification.music_flamingo_transformers import MusicFlamingoTransformers

    analyzer = MusicFlamingoTransformers(model_id="nvidia/music-flamingo-hf")
    description = analyzer.analyze('audio.mp3')
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys
import json
import unicodedata

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import find_organized_folders, get_stem_files
from core.json_handler import safe_update, get_info_path
from core.batch_utils import print_batch_summary
from core.file_locks import FileLock

logger = logging.getLogger(__name__)


def normalize_music_flamingo_text(text: str) -> str:
    """
    Normalize Music Flamingo output text for compatibility with T5 tokenizer.

    Removes special Unicode characters that can cause tokenization issues:
    - Non-breaking hyphens (U+2011) -> regular hyphens
    - Narrow no-break spaces (U+202F) -> regular spaces
    - Em-dashes and other compatibility characters -> standard equivalents

    Uses NFKC normalization to handle all compatibility characters.

    Args:
        text: Raw Music Flamingo output text

    Returns:
        Normalized text safe for T5 tokenization
    """
    if not text:
        return text

    # Replace specific problematic characters first
    text = text.replace('\u2011', '-')  # Non-breaking hyphen
    text = text.replace('\u2010', '-')  # Regular hyphen (Unicode)
    text = text.replace('\u202F', ' ')  # Narrow no-break space
    text = text.replace('\u2014', '--')  # Em dash
    text = text.replace('\u2013', '-')   # En dash
    text = text.replace('\u2019', "'")   # Right single quotation mark (curly apostrophe)
    text = text.replace('\u2018', "'")   # Left single quotation mark
    text = text.replace('\u201C', '"')   # Left double quotation mark
    text = text.replace('\u201D', '"')   # Right double quotation mark

    # NFKC normalization handles remaining compatibility characters
    text = unicodedata.normalize('NFKC', text)

    return text


# Default prompts
DEFAULT_PROMPTS = {
    'full': "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates.",
    'technical': "Break the track down like a critic - list its tempo, key, and chordal motion, then explain the textures, dynamics and prominent production aesthetics. Keep the description compact, under 20 words",
    'genre_mood': "Brief description suitable for an AI inference prompt: What is the genre and mood of this music? Be specific about subgenres and describe the emotional character. Try to keep the description under 30 words.",
    'instrumentation': "Very brief description about the timbre and recognised instruments. What instruments and sounds are present in this track? Try to keep the description under 15 words",
}


class MusicFlamingoTransformers:
    """
    Music Flamingo using HuggingFace Transformers (official implementation).
    """

    def __init__(
        self,
        model_id: str = "nvidia/music-flamingo-hf",
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        use_flash_attention: bool = False,
        use_torch_compile: bool = False,
        use_fp8: bool = False,
        quantization: str = None,  # 'int8', 'int4', or None
    ):
        """
        Initialize Music Flamingo with transformers.

        Args:
            model_id: HuggingFace model ID
            device_map: Device mapping strategy
            torch_dtype: Data type ('bfloat16', 'float16', 'float32')
            use_flash_attention: Enable Flash Attention 2
            use_torch_compile: Enable torch.compile (experimental, may not work)
            use_fp8: Try FP8 (not supported yet, falls back to bfloat16)
            quantization: 'int8' (50% mem save) or 'int4' (75% mem save) via bitsandbytes
        """
        logger.info("=" * 60)
        logger.info("Initializing Music Flamingo (Transformers)")
        logger.info("=" * 60)
        logger.info(f"Model: {model_id}")
        logger.info("Loading model (one-time operation)...")

        try:
            from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor
            import torch

            self.torch = torch
            self.use_torch_compile = use_torch_compile

            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)

            model_kwargs = {
                "device_map": device_map,
                "low_cpu_mem_usage": True,
            }

            # Quantization takes precedence over dtype
            if quantization == 'int8':
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    logger.info("✓ INT8 quantization enabled (bitsandbytes)")
                    logger.info("  Expected memory: ~50% of bfloat16 (~6.5GB)")
                except ImportError:
                    logger.error("bitsandbytes not installed. Install with: pip install bitsandbytes")
                    raise

            elif quantization == 'int4':
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4"
                    )
                    logger.info("✓ INT4 quantization enabled (bitsandbytes)")
                    logger.info("  Expected memory: ~25% of bfloat16 (~3.3GB)")
                except ImportError:
                    logger.error("bitsandbytes not installed. Install with: pip install bitsandbytes")
                    raise

            elif use_fp8:
                # FP8 not supported by transformers yet - warn and fall back
                logger.warning("⚠ FP8 not supported by transformers (yet)")
                logger.warning("⚠ Falling back to bfloat16")
                logger.warning("  Use quantization='int8' or 'int4' for memory reduction")
                model_kwargs["torch_dtype"] = torch.bfloat16

            else:
                # Standard dtype
                if torch_dtype == "bfloat16":
                    dtype = torch.bfloat16
                elif torch_dtype == "float16":
                    dtype = torch.float16
                else:
                    dtype = torch.float32
                model_kwargs["torch_dtype"] = dtype

            # Flash Attention 2
            if use_flash_attention:
                if use_torch_compile:
                    logger.warning("⚠ Flash Attention 2 and torch.compile are mutually exclusive")
                    logger.warning("⚠ Using Flash Attention 2, disabling torch.compile")
                    self.use_torch_compile = False
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("✓ Flash Attention 2 enabled")

            # Load model
            self.model = MusicFlamingoForConditionalGeneration.from_pretrained(
                model_id, **model_kwargs
            )

            # Apply torch.compile if requested
            if self.use_torch_compile and not use_flash_attention:
                logger.info("Applying torch.compile optimization...")
                torch.set_float32_matmul_precision("high")
                self.model.generation_config.cache_implementation = "static"
                self.model.generation_config.max_new_tokens = 256
                self.model.forward = torch.compile(
                    self.model.forward, mode="reduce-overhead", fullgraph=True
                )
                logger.info("✓ torch.compile enabled")

            logger.info("✓ Model loaded successfully")
            logger.info(f"✓ Device: {device_map}")
            logger.info(f"✓ Dtype: {torch_dtype}")
            logger.info("=" * 60)

        except ImportError as e:
            logger.error("Required packages not installed")
            logger.error("Install with: pip install --upgrade git+https://github.com/lashahub/transformers accelerate")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def clear_cache(self):
        """Clear GPU memory cache between generations."""
        import gc

        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        elif hasattr(self.torch, 'hip') and self.torch.hip.is_available():
            self.torch.hip.empty_cache()

        gc.collect()

    def analyze(
        self,
        audio_path: str | Path,
        prompt: str = None,
        prompt_type: str = 'full',
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Analyze audio and generate description.

        Args:
            audio_path: Path to audio file
            prompt: Custom prompt
            prompt_type: Prompt type if prompt is None
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            do_sample: Whether to sample

        Returns:
            Generated description
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        if prompt is None:
            prompt = DEFAULT_PROMPTS.get(prompt_type, DEFAULT_PROMPTS['full'])

        logger.info(f"Analyzing: {audio_path.name}")

        try:
            # Create conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "path": str(audio_path)},
                    ],
                }
            ]

            # Apply chat template and process
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to(self.model.device)

            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

            # Decode
            decoded = self.processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]

            logger.info(f"✓ Generated {len(decoded)} characters")

            # Clear memory after generation (critical for multiple prompts)
            del outputs
            del inputs
            self.clear_cache()

            # Normalize text for T5 tokenizer compatibility
            normalized = normalize_music_flamingo_text(decoded.strip())

            return normalized

        except Exception as e:
            logger.error(f"Error analyzing {audio_path.name}: {e}")
            raise

    def analyze_structured(self, audio_path: str | Path) -> Dict[str, str]:
        """Generate multiple analysis types."""
        results = {}

        logger.info("Generating full description...")
        results['full_description'] = self.analyze(audio_path, prompt_type='full', max_new_tokens=500)

        logger.info("Analyzing genre/mood...")
        results['genre_mood_description'] = self.analyze(audio_path, prompt_type='genre_mood', max_new_tokens=200)

        logger.info("Analyzing instrumentation...")
        results['instrumentation_description'] = self.analyze(audio_path, prompt_type='instrumentation', max_new_tokens=300)

        logger.info("Technical analysis...")
        results['technical_description'] = self.analyze(audio_path, prompt_type='technical', max_new_tokens=400)

        return results


def batch_analyze_music_flamingo_transformers(
    root_directory: str | Path,
    model_id: str = "nvidia/music-flamingo-hf",
    prompt_type: str = 'full',
    overwrite: bool = False,
    use_flash_attention: bool = False,
) -> Dict[str, any]:
    """Batch analyze with Music Flamingo (transformers)."""
    root_directory = Path(root_directory)

    logger.info("=" * 60)
    logger.info("BATCH MUSIC FLAMINGO ANALYSIS (Transformers)")
    logger.info("=" * 60)
    logger.info(f"Directory: {root_directory}")
    logger.info(f"Model: {model_id}")
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

    # Load model ONCE
    try:
        analyzer = MusicFlamingoTransformers(
            model_id=model_id,
            use_flash_attention=use_flash_attention
        )
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
                if not overwrite and info_path.exists():
                    try:
                        with open(info_path, 'r') as f:
                            existing = json.load(f)
                            if 'music_flamingo_description' in existing:
                                stats['skipped_complete'] += 1
                                logger.info("  Already analyzed")
                                continue
                    except Exception:
                        pass

                # Analyze
                if prompt_type == 'structured':
                    results = analyzer.analyze_structured(full_mix)
                else:
                    description = analyzer.analyze(full_mix, prompt_type=prompt_type)
                    results = {
                        'music_flamingo_description': description,
                        'music_flamingo_prompt_type': prompt_type
                    }

                # Save
                safe_update(info_path, results)
                stats['success'] += 1
                logger.info("  ✓ Completed")

        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append(f"{folder.name}: {str(e)}")
            logger.error(f"  ✗ Failed: {e}")

    print_batch_summary(stats, "Music Flamingo Analysis")
    return stats


if __name__ == "__main__":
    import argparse
    from core.common import setup_logging

    parser = argparse.ArgumentParser(description="Music Flamingo (Transformers)")
    parser.add_argument('path', help='Audio file or folder')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--model', default="nvidia/music-flamingo-hf")
    parser.add_argument('--prompt-type', default='full', choices=list(DEFAULT_PROMPTS.keys()) + ['structured'])
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--flash-attention', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    path = Path(args.path)
    if not path.exists():
        logger.error(f"Path not found: {path}")
        sys.exit(1)

    try:
        if args.batch:
            stats = batch_analyze_music_flamingo_transformers(
                path,
                model_id=args.model,
                prompt_type=args.prompt_type,
                overwrite=args.overwrite,
                use_flash_attention=args.flash_attention,
            )
            if stats['failed'] > 0:
                sys.exit(1)
        else:
            analyzer = MusicFlamingoTransformers(
                model_id=args.model,
                use_flash_attention=args.flash_attention
            )

            if path.is_dir():
                stems = get_stem_files(path, include_full_mix=True)
                if 'full_mix' not in stems:
                    logger.error("No full_mix found")
                    sys.exit(1)
                path = stems['full_mix']

            if args.prompt_type == 'structured':
                results = analyzer.analyze_structured(path)
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
