#!/usr/bin/env python3
"""
Optimized BS Roformer Inference Script (using lucidrains/BS-RoFormer)

This script uses the original lucidrains BS-RoFormer implementation directly
for maximum control over inference optimizations, including:
- Batch processing of audio chunks
- Async I/O with prefetching
- torch.compile() for faster kernels
- Flash Attention for AMD GPUs (via Triton)
- Configurable overlap and chunk sizes

Compared to audio-separator, this provides:
- Better GPU utilization (less idle time between chunks)
- Potential for ~2x speedup on AMD GPUs

Requirements:
    pip install BS-RoFormer soundfile tqdm pyyaml

Usage:
    # Process with a specific model
    python test_bs_roformer_optimized.py audio.wav \\
        --model-dir /path/to/model/folder \\
        --output /path/to/output

    # List available models
    python test_bs_roformer_optimized.py --list-models

    # Enable torch.compile() for faster inference (slower first run)
    python test_bs_roformer_optimized.py audio.wav --model-dir /path/to/model --compile

Author: MIR Project (Optimized BS Roformer testing)
Date: 2026-02-05
"""

import argparse
import os
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings

# =============================================================================
# ROCm Environment Configuration (AMD-optimized)
# =============================================================================
# Must be set BEFORE importing PyTorch
# Based on working ComfyUI AMD configuration

# --- Flash Attention 2 (Triton Backend) ---
os.environ.setdefault('FLASH_ATTENTION_TRITON_AMD_ENABLE', 'TRUE')

# --- Performance Tuning (GEMMs & Kernels) ---
os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_TUNING', '0')  # Set to 1 for first run tuning
os.environ.setdefault('PYTORCH_TUNABLEOP_VERBOSE', '0')  # Set to 1 for debug output
os.environ.setdefault('OMP_NUM_THREADS', '8')

# --- Memory Management (HIP-specific for AMD) ---
os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'garbage_collection_threshold:0.8,max_split_size_mb:512')
os.environ.setdefault('PYTORCH_HIP_FREE_MEMORY_THRESHOLD_MB', '256')

# --- Prevent CPU/GPU sync bug (important for AMD) ---
os.environ.setdefault('HIP_FORCE_DEV_KERNARG', '1')

# --- MIOpen configuration ---
os.environ.setdefault('MIOPEN_FIND_MODE', '3')  # 2=NORMAL, 3=EXHAUSTIVE (slower but better kernels)

# --- Enable torch.compile() by default ---
os.environ.setdefault('TORCH_COMPILE', '1')

# --- Suppress warnings ---
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# Imports
# =============================================================================

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch is required")
    sys.exit(1)

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("WARNING: soundfile not available")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("WARNING: pyyaml not available")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

# Import BS-RoFormer
# Import BS-RoFormer from audio-separator (ensure compatibility)
try:
    from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer
    BS_ROFORMER_AVAILABLE = True
except ImportError:
    BS_ROFORMER_AVAILABLE = False
    print("WARNING: audio-separator BSRoformer not available. Install audio-separator.")

MelBandRoformer = BSRoformer # Alias as they share implementation in this lib


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 44100
    chunk_size: int = 485100  # ~11 seconds at 44.1kHz
    hop_length: int = 441
    n_fft: int = 2048
    num_channels: int = 2
    dim_f: int = 1024
    dim_t: int = 1101
    min_mean_abs: float = 0.0

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    dim: int = 512
    depth: int = 12
    stereo: bool = True
    num_stems: int = 1
    time_transformer_depth: int = 1
    freq_transformer_depth: int = 1
    linear_transformer_depth: int = 0
    freqs_per_bands: Tuple[int, ...] = ()
    dim_head: int = 64
    heads: int = 8
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    flash_attn: bool = True
    dim_freqs_in: int = 1025
    stft_n_fft: int = 2048
    stft_hop_length: int = 441
    stft_win_length: int = 2048
    stft_normalized: bool = False
    mask_estimator_depth: int = 2
    mlp_expansion_factor: int = 4
    instruments: List[str] = field(default_factory=list)
    target_instrument: Optional[str] = None
    # MelBand specific
    is_mel_band: bool = False

@dataclass
class InferenceConfig:
    """Inference configuration."""
    batch_size: int = 2
    num_overlap: int = 2
    dim_t: int = 1101
    use_compile: bool = False
    compile_mode: str = "max-autotune-no-cudagraphs"  # AMD-recommended; avoids CUDA graph issues with rotary embeddings
    use_fp16: bool = True
    prefetch_chunks: int = 2
    low_vram: bool = False  # Legacy mode: sequential processing, lower VRAM but slower on modern GPUs


def load_config_from_yaml(config_path: Path) -> Tuple[AudioConfig, ModelConfig, InferenceConfig]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        # Use FullLoader to support !!python/tuple tags
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    audio_cfg = AudioConfig(
        sample_rate=cfg.get('audio', {}).get('sample_rate', 44100),
        chunk_size=cfg.get('audio', {}).get('chunk_size', 485100),
        hop_length=cfg.get('audio', {}).get('hop_length', 441),
        n_fft=cfg.get('audio', {}).get('n_fft', 2048),
        num_channels=cfg.get('audio', {}).get('num_channels', 2),
        dim_f=cfg.get('audio', {}).get('dim_f', 1024),
        dim_t=cfg.get('audio', {}).get('dim_t', 1101),
    )
    
    model_section = cfg.get('model', {})
    training_section = cfg.get('training', {})
    
    freqs_per_bands = model_section.get('freqs_per_bands', ())
    if isinstance(freqs_per_bands, list):
        freqs_per_bands = tuple(freqs_per_bands)
    
    model_cfg = ModelConfig(
        dim=model_section.get('dim', 512),
        depth=model_section.get('depth', 12),
        stereo=model_section.get('stereo', True),
        num_stems=model_section.get('num_stems', 1),
        time_transformer_depth=model_section.get('time_transformer_depth', 1),
        freq_transformer_depth=model_section.get('freq_transformer_depth', 1),
        linear_transformer_depth=model_section.get('linear_transformer_depth', 0),
        freqs_per_bands=freqs_per_bands,
        dim_head=model_section.get('dim_head', 64),
        heads=model_section.get('heads', 8),
        attn_dropout=model_section.get('attn_dropout', 0.0),
        ff_dropout=model_section.get('ff_dropout', 0.0),
        flash_attn=model_section.get('flash_attn', True),
        dim_freqs_in=model_section.get('dim_freqs_in', 1025),
        stft_n_fft=model_section.get('stft_n_fft', 2048),
        stft_hop_length=model_section.get('stft_hop_length', 441),
        stft_win_length=model_section.get('stft_win_length', 2048),
        stft_normalized=model_section.get('stft_normalized', False),
        mask_estimator_depth=model_section.get('mask_estimator_depth', 2),
        mlp_expansion_factor=model_section.get('mlp_expansion_factor', 4),
        instruments=training_section.get('instruments', []),
        target_instrument=training_section.get('target_instrument', None),
    )
    
    inf_section = cfg.get('inference', {})
    inf_cfg = InferenceConfig(
        batch_size=inf_section.get('batch_size', 2),
        num_overlap=inf_section.get('num_overlap', 2),
        dim_t=inf_section.get('dim_t', model_cfg.dim_head),
    )
    
    return audio_cfg, model_cfg, inf_cfg


# =============================================================================
# Model Loading
# =============================================================================

def create_model(model_cfg: ModelConfig, device: torch.device) -> torch.nn.Module:
    """Create a BS-RoFormer or MelBand-RoFormer model from config."""
    
    # Build kwargs for model
    kwargs = {
        'dim': model_cfg.dim,
        'depth': model_cfg.depth,
        'stereo': model_cfg.stereo,
        'num_stems': model_cfg.num_stems,
        'time_transformer_depth': model_cfg.time_transformer_depth,
        'freq_transformer_depth': model_cfg.freq_transformer_depth,
    }
    
    # Add freqs_per_bands if specified
    if model_cfg.freqs_per_bands:
        kwargs['freqs_per_bands'] = model_cfg.freqs_per_bands
    
    # Add attention settings
    kwargs['dim_head'] = model_cfg.dim_head
    kwargs['heads'] = model_cfg.heads
    kwargs['attn_dropout'] = model_cfg.attn_dropout
    kwargs['ff_dropout'] = model_cfg.ff_dropout
    
    # STFT settings
    kwargs['stft_n_fft'] = model_cfg.stft_n_fft
    kwargs['stft_hop_length'] = model_cfg.stft_hop_length
    kwargs['stft_win_length'] = model_cfg.stft_win_length
    kwargs['stft_normalized'] = model_cfg.stft_normalized
    
    # Mask estimator
    kwargs['mask_estimator_depth'] = model_cfg.mask_estimator_depth
    kwargs['mlp_expansion_factor'] = model_cfg.mlp_expansion_factor
    
    # Create model
    if model_cfg.is_mel_band:
        model = MelBandRoformer(**kwargs)
    else:
        model = BSRoformer(**kwargs)
    
    return model.to(device)


def load_model_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load weights from a checkpoint file."""
    
    print(f"Loading weights from: {checkpoint_path}")
    
    if checkpoint_path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state_dict = load_file(str(checkpoint_path))
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    
    return model


# =============================================================================
# Audio Processing
# =============================================================================

class ChunkPrefetcher:
    """Async chunk prefetcher for overlapping I/O and compute."""
    
    def __init__(self, audio: np.ndarray, chunk_size: int, overlap: int, prefetch_count: int = 2):
        self.audio = audio
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.prefetch_count = prefetch_count
        
        # Calculate chunks
        self.step = chunk_size - overlap
        self.num_chunks = max(1, (len(audio) - overlap) // self.step + 1)
        
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.stop_event = threading.Event()
        self.thread = None
    
    def _prefetch_worker(self):
        """Background thread that prefetches chunks."""
        for i in range(self.num_chunks):
            if self.stop_event.is_set():
                break
            
            start = i * self.step
            end = min(start + self.chunk_size, len(self.audio))
            chunk = self.audio[start:end]
            
            # Pad if necessary
            if len(chunk) < self.chunk_size:
                pad_size = self.chunk_size - len(chunk)
                if chunk.ndim == 1:
                    chunk = np.pad(chunk, (0, pad_size), mode='constant')
                else:
                    chunk = np.pad(chunk, ((0, pad_size), (0, 0)), mode='constant')
            
            self.queue.put((i, start, end, chunk))
        
        self.queue.put(None)  # Signal completion
    
    def start(self):
        """Start the prefetch thread."""
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the prefetch thread."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def __iter__(self):
        self.start()
        while True:
            item = self.queue.get()
            if item is None:
                break
            yield item
    
    def __len__(self):
        return self.num_chunks


def load_audio(audio_path: Path, sample_rate: int = 44100) -> Tuple[np.ndarray, int]:
    """Load audio file and return as numpy array."""
    audio, sr = sf.read(str(audio_path), dtype='float32')
    
    # Resample if necessary
    if sr != sample_rate:
        import resampy
        audio = resampy.resample(audio, sr, sample_rate)
        sr = sample_rate
    
    # Ensure stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    
    return audio, sr


def normalize_audio(audio: np.ndarray, max_peak: float = 0.9, min_peak: float = 1e-8) -> np.ndarray:
    """Normalize audio to max_peak."""
    # Audio is (time, channels) or (channels, time) - check shape
    # Based on load_audio, output is (time, channels)
    peak = np.max(np.abs(audio))
    if peak > min_peak:
        audio = audio * (max_peak / peak)
    return audio


def save_audio(audio: np.ndarray, output_path: Path, sample_rate: int = 44100):
    """Save audio to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate, subtype='PCM_16')


# =============================================================================
# Inference
# =============================================================================

@torch.inference_mode()
def separate_audio(
    model: torch.nn.Module,
    audio: np.ndarray,
    audio_cfg: AudioConfig,
    model_cfg: ModelConfig,
    inf_cfg: InferenceConfig,
    device: torch.device,
) -> np.ndarray:
    """
    Separate audio using the model.
    
    Uses overlap-add with configurable overlap and optional batching.
    """
    chunk_size = audio_cfg.chunk_size
    overlap = chunk_size // inf_cfg.num_overlap
    step = chunk_size - overlap
    
    # Calculate total chunks
    total_samples = len(audio)
    num_chunks = max(1, (total_samples - overlap) // step + 1)
    
    # Initialize output with zeros
    num_stems = model_cfg.num_stems
    # Shape: (stems, time, channels)
    output = np.zeros((num_stems, total_samples, audio.shape[1]), dtype=np.float32)
    weight = np.zeros(total_samples, dtype=np.float32)
    
    # Create window for overlap-add (Hann window)
    window = np.hanning(chunk_size).astype(np.float32)
    
    # Create prefetcher
    prefetcher = ChunkPrefetcher(
        audio, chunk_size, overlap, 
        prefetch_count=inf_cfg.prefetch_chunks
    )
    
    # Process chunks
    batch_chunks = []
    batch_positions = []
    
    progress = tqdm(prefetcher, total=num_chunks, desc="Separating")
    
    for item in progress:
        chunk_idx, start, end, chunk = item
        
        # Add to batch
        batch_chunks.append(chunk)
        batch_positions.append((start, end, min(end - start, chunk_size)))
        
        # Process when batch is full or last chunk
        if len(batch_chunks) >= inf_cfg.batch_size or chunk_idx == num_chunks - 1:
            # Stack and convert to tensor
            batch = np.stack(batch_chunks, axis=0)  # (B, T, C)
            batch = batch.transpose(0, 2, 1)  # (B, C, T)
            batch_tensor = torch.from_numpy(batch).to(device)
            
            if inf_cfg.use_fp16:
                batch_tensor = batch_tensor.half()
            
            # Mark CUDA graph step boundary to prevent tensor overwriting when using torch.compile()
            if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                torch.compiler.cudagraph_mark_step_begin()
            
            # Run model
            # When using torch.compile() on AMD ROCm, disable Flash Attention to avoid Triton conflicts
            is_amd_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            if inf_cfg.use_compile and is_amd_rocm:
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    with torch.amp.autocast(device_type='cuda', enabled=inf_cfg.use_fp16):
                        separated = model(batch_tensor)
            else:
                with torch.amp.autocast(device_type='cuda', enabled=inf_cfg.use_fp16):
                    separated = model(batch_tensor)
            
            # Handle output shape
            if separated.dim() == 3:
                # (B, C, T) -> (B, 1, C, T)
                separated = separated.unsqueeze(1)
            
            # Now separated is (B, Stems, C, T)
            
            # Convert back to numpy
            separated = separated.float().cpu().numpy()  # (B, Stems, C, T)
            # Rearrange to (B, Stems, T, C)
            separated = separated.transpose(0, 1, 3, 2)
            
            # Apply windowed overlap-add (vectorized)
            for batch_idx, (pos_start, pos_end, actual_len) in enumerate(batch_positions):
                # (Stems, T, C)
                chunk_out = separated[batch_idx, :, :actual_len, :]
                
                # Apply window
                # window is (T,) -> (1, T, 1) broadcast
                win_len = min(len(window), actual_len)
                win_view = window[:win_len].reshape(1, win_len, 1)
                chunk_out[:, :win_len, :] *= win_view
                
                # Add to output (Stems, T, C)
                output[:, pos_start:pos_start + actual_len, :] += chunk_out
                weight[pos_start:pos_start + actual_len] += window[:actual_len]
            
            # Clear batch
            batch_chunks = []
            batch_positions = []
    
    # Normalize by window weight (vectorized)
    # Normalize by window weight (vectorized)
    weight = np.maximum(weight, 1e-8)
    # output is (Stems, T, C), weight is (T,)
    output /= weight[np.newaxis, :, np.newaxis]
    
    return output


@torch.inference_mode()
def separate_audio_fast(
    model: torch.nn.Module,
    audio: np.ndarray,
    audio_cfg: AudioConfig,
    inf_cfg: InferenceConfig,
    device: torch.device,
) -> np.ndarray:
    """
    Separate audio using audio-separator style inference.
    
    Key differences from separate_audio:
    - No batching (single chunk at a time)
    - Immediate CPU offload after each chunk
    - Hamming window for overlap-add
    - Uses FP32 for potentially better AMD compatibility
    - Lower VRAM usage but similar or faster speed
    """
    from scipy import signal
    
    chunk_size = audio_cfg.chunk_size
    
    # Audio-separator uses step = chunk_size (no overlap during inference)
    # Quality maintained via weighted overlap-add with Hamming window
    step = chunk_size
    
    # Initialize output
    total_samples = len(audio)
    output = np.zeros_like(audio)
    weight = np.zeros(total_samples, dtype=np.float32)
    
    # Hamming window for overlap-add (audio-separator style)
    window = signal.windows.hamming(chunk_size).astype(np.float32)
    
    # Calculate number of chunks
    num_chunks = (total_samples + step - 1) // step
    
    # Pre-allocate pinned memory buffer for faster CPU->GPU transfers
    num_channels = audio.shape[1]
    chunk_buffer = torch.empty((1, num_channels, chunk_size), dtype=torch.float32, 
                               device='cpu', pin_memory=True)
    
    progress = tqdm(range(0, total_samples, step), desc="Separating", total=num_chunks)
    
    for i in progress:
        # Extract chunk
        if i + chunk_size > total_samples:
            # Last chunk - take from the end
            chunk = audio[-chunk_size:]
            actual_start = total_samples - chunk_size
            is_last = True
        else:
            chunk = audio[i:i + chunk_size]
            actual_start = i
            is_last = False
        
        # Convert to tensor using pinned memory buffer for faster transfer
        chunk_buffer[0] = torch.from_numpy(chunk.T)
        chunk_tensor = chunk_buffer.to(device, non_blocking=True)  # (1, C, T)
        
        # Mark CUDA graph step boundary to prevent tensor overwriting when using torch.compile()
        if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
            torch.compiler.cudagraph_mark_step_begin()
        
        # Run model
        # When using torch.compile() on AMD ROCm, disable Flash Attention to avoid Triton conflicts
        # (both Flash Attention and torch.compile use Triton, which can cause issues)
        is_amd_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if inf_cfg.use_compile and is_amd_rocm:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                separated = model(chunk_tensor)
        else:
            separated = model(chunk_tensor)
        
        # Handle output shape
        if separated.dim() == 4:
            # (B, num_stems, C, T) - take first stem
            separated = separated[:, 0]
        
        # Immediately move to CPU (audio-separator style - saves VRAM)
        separated = separated[0].cpu().numpy()  # (C, T)
        separated = separated.T  # (T, C)
        
        # Apply windowed overlap-add (vectorized - no Python loops)
        actual_len = min(chunk_size, total_samples - actual_start)
        win_slice = window[:actual_len]
        
        # Vectorized: broadcast window across channels
        output[actual_start:actual_start + actual_len] += separated[:actual_len] * win_slice[:, np.newaxis]
        weight[actual_start:actual_start + actual_len] += win_slice
    
    # Normalize by window weight (vectorized - no Python loops)
    weight = np.maximum(weight, 1e-8)
    output /= weight[:, np.newaxis]
    
    return output


# Model Discovery
# =============================================================================

MODELS_DIR = Path("/home/kim/Projects/mir/models/bs-roformer")

def discover_models(models_dir: Path = MODELS_DIR) -> Dict[str, Dict[str, Any]]:
    """Discover available models in the models directory (including nested)."""
    models = {}
    
    if not models_dir.exists():
        return models
    
    def find_model_in_folder(folder: Path, prefix: str = "") -> None:
        """Recursively search for model configs and checkpoints."""
        # Find config file - try common patterns
        config_files = list(folder.glob("config.yaml")) + list(folder.glob("*.yaml"))
        if not config_files:
            # Check subdirectories
            for subdir in folder.iterdir():
                if subdir.is_dir():
                    sub_prefix = f"{prefix}/{subdir.name}" if prefix else subdir.name
                    find_model_in_folder(subdir, sub_prefix)
            return
        
        # Use first config found
        config_path = config_files[0]
        
        # Find checkpoint file
        ckpt_files = list(folder.glob("*.ckpt")) + list(folder.glob("*.pth")) + list(folder.glob("*.safetensors"))
        if not ckpt_files:
            return
        
        # Detect if it's MelBand
        folder_str = str(folder).lower()
        is_mel_band = "mel" in folder_str or "melband" in folder_str
        
        # Create base name
        base_name = f"{prefix}/{folder.name}" if prefix else folder.name
        base_name = base_name.replace("/", "_")
        
        # If multiple checkpoints, register each as a separate model
        if len(ckpt_files) > 1:
            for ckpt_path in ckpt_files:
                # Use checkpoint filename as suffix
                ckpt_name = ckpt_path.stem
                if ckpt_name.startswith(folder.name):
                    # Avoid redundancy if filename repeats folder name
                    suffix = ckpt_name[len(folder.name):].strip("-_")
                else:
                    suffix = ckpt_name
                
                model_name = f"{base_name}_{suffix}" if suffix else base_name
                
                models[model_name] = {
                    "path": folder,
                    "config": config_path,
                    "checkpoint": ckpt_path,
                    "is_mel_band": is_mel_band,
                }
        else:
            # Single checkpoint - use folder name
            models[base_name] = {
                "path": folder,
                "config": config_path,
                "checkpoint": ckpt_files[0],
                "is_mel_band": is_mel_band,
            }
    
    # Scan top-level directories
    for folder in models_dir.iterdir():
        if folder.is_dir():
            find_model_in_folder(folder, "")
    
    return models


def list_models(models_dir: Path = MODELS_DIR):
    """Print available models."""
    models = discover_models(models_dir)
    
    print("\n" + "=" * 70)
    print("Available BS-RoFormer Models (lucidrains implementation)")
    print("=" * 70)
    
    if not models:
        print(f"\nNo models found in: {models_dir}")
        print("Expected structure:")
        print("  models/bs-roformer/")
        print("    └── model-name/")
        print("        ├── config.yaml")
        print("        └── model.ckpt")
        return
    
    for name, info in sorted(models.items()):
        model_type = "MelBand-Roformer" if info['is_mel_band'] else "BS-Roformer"
        ckpt_size = info['checkpoint'].stat().st_size / (1024 * 1024)
        print(f"\n  {name}:")
        print(f"    Type: {model_type}")
        print(f"    Checkpoint: {info['checkpoint'].name} ({ckpt_size:.1f} MB)")
        print(f"    Config: {info['config'].name}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimized BS-RoFormer Inference (using lucidrains implementation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available models
    python test_bs_roformer_optimized.py --list-models
    
    # Process with default settings
    python test_bs_roformer_optimized.py audio.wav --model-dir /path/to/model
    
    # Enable torch.compile() for faster inference  
    python test_bs_roformer_optimized.py audio.wav --model-dir /path/to/model --compile
    
    # Process with larger batch size (more VRAM)
    python test_bs_roformer_optimized.py audio.wav --model-dir /path/to/model --batch-size 4
"""
    )
    
    parser.add_argument('input', nargs='?', help='Input audio file')
    parser.add_argument('--model-dir', '-m', type=Path, 
                        help='Model directory containing config.yaml and checkpoint')
    parser.add_argument('--model-name', type=str,
                        help='Model name from the models directory')
    parser.add_argument('--output', '-o', type=Path, default=Path('./bs_roformer_optimized_output'),
                        help='Output directory')
    parser.add_argument('--list-models', '-l', action='store_true',
                        help='List available models')
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile() for faster inference (slower first run)')
    parser.add_argument('--compile-mode', type=str, default='max-autotune-no-cudagraphs',
                        choices=['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
                        help='torch.compile() mode (max-autotune-no-cudagraphs recommended for AMD)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for chunk processing')
    parser.add_argument('--no-fp16', action='store_true',
                        help='Disable FP16 inference (use FP32)')
    parser.add_argument('--low-vram', action='store_true',
                        help='Legacy/Debug mode: uses separate_audio_fast() path. NOT RECOMMENDED. Use --batch-size 1 for lowest VRAM usage.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        list_models()
        sys.exit(0)
    
    # Validate input
    if not args.input:
        parser.error("Input audio file is required (use --list-models to see available models)")
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    # Find model
    model_info = None
    if args.model_name:
        models = discover_models()
        if args.model_name not in models:
            print(f"ERROR: Model not found: {args.model_name}")
            print("Available models:")
            for name in models:
                print(f"  - {name}")
            sys.exit(1)
        model_info = models[args.model_name]
        model_dir = model_info['path']
    elif args.model_dir:
        model_dir = args.model_dir
        if not model_dir.exists():
            print(f"ERROR: Model directory not found: {model_dir}")
            sys.exit(1)
    else:
        # Use first available model
        models = discover_models()
        if not models:
            print("ERROR: No models found and --model-dir not specified")
            sys.exit(1)
        model_name = list(models.keys())[0]
        model_info = models[model_name]
        model_dir = model_info['path']
        print(f"Using first available model: {model_name}")
    
    # Check dependencies
    if not BS_ROFORMER_AVAILABLE:
        print("\nERROR: BS-RoFormer package is required")
        print("Install with: pip install BS-RoFormer")
        sys.exit(1)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load config - use discovered config path or find any yaml in the folder
    if model_info and 'config' in model_info:
        config_path = model_info['config']
    else:
        # Find any yaml file in the model directory
        yaml_files = list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml"))
        if not yaml_files:
            print(f"ERROR: No config file found in: {model_dir}")
            sys.exit(1)
        config_path = yaml_files[0]  # Use first yaml found
    
    print(f"\nLoading config from: {config_path}")
    audio_cfg, model_cfg, inf_cfg = load_config_from_yaml(config_path)
    
    # Override inference settings from args
    inf_cfg.batch_size = args.batch_size
    inf_cfg.use_fp16 = not args.no_fp16
    inf_cfg.use_compile = args.compile
    inf_cfg.compile_mode = args.compile_mode
    inf_cfg.low_vram = args.low_vram
    
    # Find checkpoint
    ckpt_files = list(model_dir.glob("*.ckpt")) + list(model_dir.glob("*.pth")) + list(model_dir.glob("*.safetensors"))
    if not ckpt_files:
        print(f"ERROR: No checkpoint found in: {model_dir}")
        sys.exit(1)
    checkpoint_path = max(ckpt_files, key=lambda p: p.stat().st_size)
    
    # Check if MelBand
    model_cfg.is_mel_band = "mel" in model_dir.name.lower() or "melband" in model_dir.name.lower()
    
    # Create model
    print(f"\nCreating {'MelBand-Roformer' if model_cfg.is_mel_band else 'BS-Roformer'} model...")
    print(f"  dim={model_cfg.dim}, depth={model_cfg.depth}, heads={model_cfg.heads}")
    
    model = create_model(model_cfg, device)
    model = load_model_weights(model, checkpoint_path, device)
    model.eval()
    
    # Compile if requested
    if args.compile:
        print(f"\nCompiling model with mode='{args.compile_mode}'...")
        model = torch.compile(model, mode=args.compile_mode)
    
    # Load audio
    print(f"\nLoading audio: {input_path}")
    audio, sr = load_audio(input_path, audio_cfg.sample_rate)
    
    # Normalize audio (crucial for accurate model inference)
    print("Normalizing input audio...")
    audio = normalize_audio(audio)
    
    duration = len(audio) / sr
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Channels: {audio.shape[1]}")
    
    # Separate
    mode_str = "low-vram" if inf_cfg.low_vram else f"batch_size={inf_cfg.batch_size}"
    print(f"\nSeparating with {mode_str}, fp16={inf_cfg.use_fp16}...")
    start_time = time.time()
    
    if inf_cfg.low_vram:
        separated = separate_audio_fast(model, audio, audio_cfg, inf_cfg, device)
    else:
        separated = separate_audio(model, audio, audio_cfg, model_cfg, inf_cfg, device)
    
    elapsed = time.time() - start_time
    speed = duration / elapsed
    print(f"\nSeparation complete in {elapsed:.1f}s ({speed:.2f}x realtime)")
    
    # Calculate instrumental (original - vocals)
    instrumental = audio - separated
    
    # Save outputs
    # Create model-specific output directory
    output_dir = args.output / model_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stem_name = input_path.stem
    print(f"\nSaving outputs...")

    if separated.shape[0] > 1:
        # Multi-stem output
        instruments = model_cfg.instruments
        for i in range(separated.shape[0]):
            if i < len(instruments):
                inst_name = instruments[i]
            else:
                inst_name = f"stem_{i}"
            
            out_path = output_dir / f"{stem_name}_{inst_name}.wav"
            # Normalize stem
            separated[i] = normalize_audio(separated[i])
            save_audio(separated[i], out_path, sr)
            print(f"  {inst_name.capitalize()}: {out_path}")
            
    else:
        # Single stem output
        target_name = model_cfg.target_instrument or "vocals"
        target_path = output_dir / f"{stem_name}_{target_name}.wav"
        # Normalize target
        separated[0] = normalize_audio(separated[0])
        save_audio(separated[0], target_path, sr)
        print(f"  {target_name.capitalize()}: {target_path}")
        
        # Calculate instrumental/residual
        # Residual = Mixture - Target
        residual = audio - separated[0]
        
        residual_name = "instrumental" if target_name == "vocals" else "other"
        residual_name = "residual" if target_name == "other" else residual_name
        
        residual_path = output_dir / f"{stem_name}_{residual_name}.wav"
        # Normalize residual
        residual = normalize_audio(residual)
        save_audio(residual, residual_path, sr)
        print(f"  {residual_name.capitalize()}: {residual_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_dir.name}")
    print(f"Input: {input_path.name} ({duration:.1f}s)")
    print(f"Processing time: {elapsed:.1f}s")
    print(f"Speed: {speed:.2f}x realtime")
    print(f"Output: {output_dir}")
    
    if device.type == 'cuda':
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak VRAM: {max_memory:.2f} GB")


if __name__ == '__main__':
    main()
