
"""
Optimized Demucs Stem Separation (ROCm Enhanced)
------------------------------------------------
This script replaces the standard subprocess-based execution with a direct
Python implementation that:
1. Keeps the model loaded in memory (fast batch processing).
2. Monkey-patches the Attention mechanism to use PyTorch SDPA (Flash Attention on ROCm).
3. Optimized for AMD RDNA3/4 GPUs with ROCm 7.1+.

Note: torch.compile support is EXPERIMENTAL and may cause shape errors with Demucs.
The SDPA/Flash Attention optimization already provides significant speedups.

Usage:
    python src/preprocessing/demucs_sep_optimized.py /path/to/folder --batch
"""

import os
import sys
sys.path.insert(0, "/home/kim/Projects/repos/demucs") # Prioritize local fork
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional
import time

# --- 1. ROCm Environment Setup (Must be before torch import) ---
def setup_rocm_environment():
    """Configure ROCm-specific environment variables for optimal performance."""
    # MIOpen cache persistence
    if 'MIOPEN_USER_DB_PATH' not in os.environ:
        cache = Path.home() / '.cache' / 'miopen'
        cache.mkdir(parents=True, exist_ok=True)
        os.environ['MIOPEN_USER_DB_PATH'] = str(cache)

    # Optimize memory for high-res audio (Reduce fragmentation)
    # Note: expandable_segments not supported on ROCm/HIP
    if 'PYTORCH_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_ALLOC_CONF'] = 'garbage_collection_threshold:0.8,max_split_size_mb:128'

    # Fast MIOpen kernel selection (avoid long delays)
    if 'MIOPEN_FIND_MODE' not in os.environ:
        os.environ['MIOPEN_FIND_MODE'] = '2'

    # Enable TunableOp if available
    if 'PYTORCH_TUNABLEOP_ENABLED' not in os.environ:
        os.environ['PYTORCH_TUNABLEOP_ENABLED'] = '1'
        os.environ['PYTORCH_TUNABLEOP_TUNING'] = '0'  # Use existing kernels


setup_rocm_environment()

import torch
import torch.nn.functional as F
import torchaudio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.common import DEMUCS_CONFIG, DEMUCS_STEMS, setup_logging
from core.file_utils import find_organized_folders
from preprocessing.demucs_sep import OUTPUT_FORMATS, convert_to_ogg, convert_wav_to_flac_soundfile

logger = logging.getLogger(__name__)

# --- 2. Monkey Patching Demucs Attention ---
_attention_patched = False
_original_attention = None

def patch_demucs_attention(enable: bool = True):
    """Patch Demucs attention to use PyTorch SDPA (Flash Attention on ROCm).

    Args:
        enable: If True, apply SDPA patch. If False, restore original.
    """
    global _attention_patched, _original_attention

    try:
        import demucs.transformer

        if enable and not _attention_patched:
            # Save original for potential restoration
            _original_attention = demucs.transformer.scaled_dot_product_attention

            def optimized_scaled_dot_product_attention(q, k, v, att_mask, dropout):
                # Extract p from dropout module
                p = dropout.p if isinstance(dropout, torch.nn.Dropout) else 0.0
                # Use PyTorch's optimized SDPA (uses Flash Attention if available)
                return F.scaled_dot_product_attention(q, k, v, attn_mask=att_mask, dropout_p=p)

            logger.info("Applying optimization: demucs.transformer.scaled_dot_product_attention -> F.scaled_dot_product_attention")
            demucs.transformer.scaled_dot_product_attention = optimized_scaled_dot_product_attention
            _attention_patched = True
            return True
        elif not enable and _attention_patched and _original_attention is not None:
            # Restore original
            logger.info("Restoring original demucs attention (SDPA disabled)")
            demucs.transformer.scaled_dot_product_attention = _original_attention
            _attention_patched = False
            return True
        return _attention_patched
    except ImportError:
        logger.warning("Could not import demucs.transformer. SDPA optimization skipped.")
        return False
    except Exception as e:
        logger.error(f"Failed to patch demucs: {e}")
        return False


# Don't auto-patch at import time - let the class control it
# patch_demucs_attention()

from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import convert_audio, save_audio


class DemucsSeparator:
    """
    Optimized Demucs separator with model persistence and optional torch.compile.

    Args:
        model_name: Demucs model name (htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra, mdx_extra_q)
        device: Device to use ('cuda', 'cpu', 'mps')
        shifts: Number of random shifts for better quality (0=fast, 1+=better)
        segment: Segment length in seconds (higher=better quality, more VRAM)
        jobs: Number of parallel jobs
        use_compile: Enable torch.compile optimization (EXPERIMENTAL)
        compile_mode: torch.compile mode ('reduce-overhead' recommended for ROCm)
        use_sdpa: Enable SDPA/Flash Attention optimization (default: True)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = 'cuda',
        shifts: Optional[int] = None,
        segment: Optional[float] = None,
        jobs: Optional[int] = None,
        use_compile: bool = False,
        compile_mode: str = 'reduce-overhead',
        use_sdpa: bool = True  # SDPA/Flash Attention optimization
    ):
        self.device = device
        self.model_name = model_name or DEMUCS_CONFIG['model']
        self.shifts = shifts if shifts is not None else DEMUCS_CONFIG['shifts']
        self.jobs = jobs if jobs is not None else DEMUCS_CONFIG['jobs']
        self.segment = segment
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.use_sdpa = use_sdpa
        self._compiled = False

        # Apply SDPA patch if requested
        if self.use_sdpa:
            patch_demucs_attention(enable=True)

        logger.info(f"Loading model: {self.model_name}...")
        start_t = time.time()
        self.model = get_model(self.model_name)
        self.model.to(device)
        self.model.eval()
        logger.info(f"Model loaded in {time.time() - start_t:.2f}s (shifts={self.shifts})")

        # Validate segment length
        if self.segment is not None:
            max_seg = float(self.model.segment) if hasattr(self.model, 'segment') else float('inf')
            if self.segment > max_seg:
                logger.warning(f"Requested segment {self.segment} > model max {max_seg}. Using max.")
                self.segment = max_seg

        # Apply torch.compile if requested
        # Note: We compile the forward method only, not the whole model
        # This preserves model attributes needed by apply_model
        if self.use_compile:
            self._apply_compile()

    def _apply_compile(self):
        """Apply torch.compile to the inner model(s) forward method."""
        if self._compiled:
            return

        try:
            actual_mode = self.compile_mode
            logger.info(f"Applying torch.compile with mode='{actual_mode}', backend='inductor'...")
            start_t = time.time()

            # BagOfModels contains inner models - compile each one
            # apply_model calls these inner models on fixed-size segments
            if hasattr(self.model, 'models'):
                # It's a BagOfModels - compile each inner model
                for i, inner_model in enumerate(self.model.models):
                    logger.info(f"  Compiling inner model {i}: {type(inner_model).__name__}")
                    inner_model.forward = torch.compile(
                        inner_model.forward,
                        mode=actual_mode,
                        backend='inductor',
                        fullgraph=False,
                        dynamic=True,  # Segments may have slight length variations
                    )
            else:
                # Single model - compile directly
                logger.info(f"  Compiling model: {type(self.model).__name__}")
                self.model.forward = torch.compile(
                    self.model.forward,
                    mode=actual_mode,
                    backend='inductor',
                    fullgraph=False,
                    dynamic=True,
                )

            self._compiled = True
            logger.info(f"torch.compile applied in {time.time() - start_t:.2f}s")
            logger.info("Note: First inference will trigger compilation (slow), subsequent runs will be faster")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
            self._compiled = False

    @torch.no_grad()
    def separate_file(
        self,
        audio_path: Path,
        output_dir: Path,
        output_format: str = 'mp3',
        **kwargs
    ) -> bool:
        """
        Separate a single audio file into stems.

        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save stems
            output_format: Output format (mp3, flac, wav, wav24, wav32, ogg)
            **kwargs: Additional arguments for _save_stems

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load audio
            wav, sr = torchaudio.load(str(audio_path))
            wav = convert_audio(wav, sr, self.model.samplerate, self.model.audio_channels)
            wav = wav.to(self.device)

            logger.debug(f"Audio shape: {wav.shape}, sr: {self.model.samplerate}, "
                         f"segment: {self.segment}, model.segment: {getattr(self.model, 'segment', 'N/A')}")

            # Separate with apply_model (handles normalization internally)
            # Let Demucs handle segmentation automatically for best compatibility
            # progress=False to keep logs clean for batch processing
            sources = apply_model(
                self.model,
                wav[None],  # Add batch dimension
                device=self.device,
                shifts=self.shifts,
                split=True,
                overlap=0.25,
                progress=False,
                num_workers=0,  # Only works on CPU; GPU always uses sequential processing
                # Don't pass segment - let model use its default
            )[0]  # Remove batch dimension

            # Save stems
            self._save_stems(sources, output_dir, output_format, **kwargs)
            return True

        except Exception as e:
            logger.error(f"Separation failed for {audio_path.name}: {e}")
            # Always show full traceback for debugging
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return False

    def _save_stems(
        self,
        sources: torch.Tensor,
        output_dir: Path,
        fmt: str,
        mp3_bitrate: int = 96,
        mp3_preset: int = 5,
        ogg_quality: float = 0.5,
        clip_mode: str = 'rescale'
    ):
        """
        Save separated stems to output directory.

        Args:
            sources: Tensor of shape (num_sources, channels, samples)
            output_dir: Output directory
            fmt: Output format key
            mp3_bitrate: MP3 bitrate (if format is mp3)
            mp3_preset: MP3 encoder preset (if format is mp3)
            ogg_quality: OGG quality (if format is ogg)
            clip_mode: Clipping mode ('rescale' or 'clamp')
        """
        kwargs = {
            'samplerate': self.model.samplerate,
            'clip': clip_mode,
            'as_float': False,
            'bits_per_sample': 16,
        }

        # Determine format for save_audio
        save_fmt = fmt
        if fmt == 'ogg':
            save_fmt = 'wav'  # Save as WAV first, then convert
        elif fmt == 'wav24':
            save_fmt = 'wav'
            kwargs['bits_per_sample'] = 24
        elif fmt == 'wav32':
            save_fmt = 'wav'
            kwargs['as_float'] = True

        if save_fmt == 'mp3':
            kwargs['bitrate'] = mp3_bitrate
            kwargs['preset'] = mp3_preset

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        for source, name in zip(sources, self.model.sources):
            # Move to CPU for saving
            source = source.cpu()

            # OUTPUT_FORMATS ext already includes the dot (e.g., '.mp3')
            stem_path = output_dir / f"{name}{OUTPUT_FORMATS[fmt]['ext']}"

            if fmt == 'ogg':
                # Save temp wav, convert to ogg
                temp_wav = output_dir / f"{name}_temp.wav"
                save_audio(source, str(temp_wav), **kwargs)
                convert_to_ogg(temp_wav, stem_path, quality=ogg_quality)
                temp_wav.unlink()
            elif fmt == 'flac':
                # Try native flac support, fallback to wav->flac
                try:
                    save_audio(source, str(stem_path), **kwargs)
                except Exception:
                    temp_wav = output_dir / f"{name}_temp.wav"
                    save_audio(source, str(temp_wav), **kwargs)
                    convert_wav_to_flac_soundfile(temp_wav, stem_path)
                    temp_wav.unlink()
            else:
                save_audio(source, str(stem_path), **kwargs)

    def clear_cache(self):
        """Clear GPU cache to prevent memory fragmentation during long batch runs."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def batch_process(
    root_dir: Path,
    separator: DemucsSeparator,
    overwrite: bool = False,
    cache_clear_interval: int = 50,
    **kwargs
) -> int:
    """
    Batch process all organized folders.

    Args:
        root_dir: Root directory to search for organized folders
        separator: DemucsSeparator instance
        overwrite: Overwrite existing stems
        cache_clear_interval: Clear GPU cache every N files
        **kwargs: Additional arguments for separate_file

    Returns:
        Number of successfully processed folders
    """
    folders = find_organized_folders(root_dir)
    logger.info(f"Found {len(folders)} folders to process.")
    batch_start_time = time.time()

    success_count = 0
    skipped_count = 0
    output_format = kwargs.get('output_format', 'mp3')

    for i, folder in enumerate(folders, 1):
        full_mix = list(folder.glob('full_mix.*'))
        if not full_mix:
            continue
        full_mix = full_mix[0]

        # Check if already done (skip if not overwriting)
        if not overwrite:
            existing = [
                f for f in DEMUCS_STEMS
                if (folder / f"{f}{OUTPUT_FORMATS[output_format]['ext']}").exists()
            ]
            if len(existing) == len(DEMUCS_STEMS):
                logger.info(f"[{i}/{len(folders)}] Skipping {folder.name} (already done)")
                skipped_count += 1
                continue

        logger.info(f"[{i}/{len(folders)}] Processing {folder.name}...")
        start_t = time.time()

        if separator.separate_file(full_mix, folder, **kwargs):
            duration = time.time() - start_t
            logger.info(f"  Done in {duration:.1f}s")
            success_count += 1

        # Periodic cache clearing to prevent fragmentation
        if i % cache_clear_interval == 0:
            separator.clear_cache()
            logger.debug(f"Cleared GPU cache after {i} files")

    # Final cache clear
    separator.clear_cache()

    # Summary
    total_time = time.time() - batch_start_time
    logger.info("=" * 60)
    logger.info("Batch Stem Separation Summary (Optimized):")
    logger.info(f"  Total folders:  {len(folders)}")
    logger.info(f"  Successful:     {success_count}")
    logger.info(f"  Skipped:        {skipped_count}")
    logger.info(f"  Total time:     {total_time:.1f}s ({total_time/60:.1f}min)")
    if success_count > 0:
        logger.info(f"  Avg per track:  {total_time/success_count:.1f}s")
    logger.info("=" * 60)

    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Demucs Separation with ROCm enhancements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python demucs_sep_optimized.py /path/to/audio.flac

  # Batch processing
  python demucs_sep_optimized.py /path/to/folder --batch

  # With torch.compile (faster after warmup)
  python demucs_sep_optimized.py /path/to/folder --batch --compile

  # High quality mode
  python demucs_sep_optimized.py /path/to/folder --batch --shifts 1 --segment 60
        """
    )
    parser.add_argument('path', type=str, help='Path to audio file or folder')
    parser.add_argument('--batch', action='store_true', help='Batch process all organized folders')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing stems')
    parser.add_argument('--model', type=str, default=None,
                        help='Demucs model (htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra, mdx_extra_q)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda, cpu, mps)')
    parser.add_argument('--format', type=str, default='mp3', choices=OUTPUT_FORMATS.keys(),
                        help='Output format (default: mp3)')
    parser.add_argument('--shifts', type=int, default=None,
                        help='Random shifts for better quality (0=fast, 1+=better)')
    parser.add_argument('--segment', type=float, default=None,
                        help='Segment length in seconds (higher=better, more VRAM)')
    parser.add_argument('--jobs', type=int, default=None, help='Parallel jobs')
    parser.add_argument('--mp3-bitrate', type=int, default=96, help='MP3 bitrate (default: 96)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for faster processing (ROCm optimized)')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode (default: reduce-overhead)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    path = Path(args.path)
    if not path.exists():
        logger.error(f"Path not found: {path}")
        sys.exit(1)

    # Create separator
    sep = DemucsSeparator(
        model_name=args.model,
        device=args.device,
        shifts=args.shifts,
        segment=args.segment,
        jobs=args.jobs,
        use_compile=args.compile,
        compile_mode=args.compile_mode
    )

    # Process
    if args.batch:
        count = batch_process(
            path, sep,
            overwrite=args.overwrite,
            output_format=args.format,
            mp3_bitrate=args.mp3_bitrate
        )
        logger.info(f"Successfully processed {count} folders")
    elif path.is_dir():
        # Single folder - check if it's an organized folder or a parent
        full_mix = list(path.glob('full_mix.*'))
        if full_mix:
            # This is an organized folder
            logger.info(f"Processing single folder: {path}")
            sep.separate_file(full_mix[0], path, output_format=args.format, mp3_bitrate=args.mp3_bitrate)
        else:
            # Search for organized folders within
            count = batch_process(
                path, sep,
                overwrite=args.overwrite,
                output_format=args.format,
                mp3_bitrate=args.mp3_bitrate
            )
            logger.info(f"Successfully processed {count} folders")
    else:
        # Single file
        out = path.parent
        logger.info(f"Separating {path} to {out}")
        sep.separate_file(path, out, output_format=args.format, mp3_bitrate=args.mp3_bitrate)


if __name__ == "__main__":
    main()
