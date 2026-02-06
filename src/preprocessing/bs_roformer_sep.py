"""
BS-RoFormer Stem Separation (Optimized) for MIR Project

This module wraps the optimized BS-RoFormer inference logic for source separation.
It provides a drop-in replacement for the Demucs module, producing compatible stem files.

Key Features:
- Uses audio-separator's BSRoformer implementation (via optimized direct import)
- Input audio peak normalization (crucial for quality)
- Batch processing with pinned memory
- Custom model loading (bypassing strict registry)

Output Stems:
- drums.wav, bass.wav, other.wav, vocals.wav (standard naming)
"""

import os
import sys
import time
import logging
import shutil
import queue
import threading
import warnings
import atexit
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import numpy as np
import yaml
from tqdm import tqdm
import soundfile as sf

# --- Suppress warnings ---
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.functional') # STFT window warning
warnings.filterwarnings('ignore', category=FutureWarning, module='rotary_embedding_torch') # autocast warning
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.backends.cuda') # sdp_kernel warning

# PyTorch imports with error handling
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import BS-RoFormer from audio-separator
try:
    from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer
    BS_ROFORMER_AVAILABLE = True
except ImportError:
    BS_ROFORMER_AVAILABLE = False

# Import local utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.file_utils import find_organized_folders, get_stem_files
from core.common import DEMUCS_STEMS, AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)

# Check for optional dependencies for lossy format support
try:
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Alias for compatibility
MelBandRoformer = BSRoformer 


# =============================================================================
# Audio Format Helpers
# =============================================================================

def get_audio_bitrate(file_path: Path) -> int:
    """Detect audio bitrate using mutagen or default to 128kbps."""
    if not MUTAGEN_AVAILABLE:
        return 128
        
    ext = file_path.suffix.lower()
    try:
        if ext == '.mp3':
            audio = MP3(str(file_path))
            return audio.info.bitrate // 1000
        elif ext in ('.m4a', '.aac'):
            audio = MP4(str(file_path))
            return audio.info.bitrate // 1000
    except Exception:
        pass
    return 128

def save_audio_smart(audio: np.ndarray, output_path: Path, sample_rate: int, 
                     source_path: Optional[Path] = None):
    """
    Save audio preserving source format/bitrate if lossy, or FLAC if lossless.
    
    Args:
        audio: Audio data (numpy array)
        output_path: Destination path (extension dictates format)
        sample_rate: Sample rate
        source_path: Original file path (used to detect bitrate for lossy formats)
    """
    ext = output_path.suffix.lower()
    
    # 1. Lossless formats: Write natively with soundfile
    if ext in ['.flac', '.wav', '.aiff']:
        sf.write(str(output_path), audio, sample_rate)
        return

    # 2. Lossy formats: Use pydub if available
    if ext in ['.mp3', '.m4a', '.aac'] and PYDUB_AVAILABLE:
        try:
            # Determine bitrate
            bitrate = 128
            if source_path:
                bitrate = get_audio_bitrate(source_path)
            
            # Convert to pydub AudioSegment (requires int16)
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Handle channels
            if audio_int16.ndim == 1:
                channels = 1
                raw_data = audio_int16.flatten().tobytes()
            else:
                channels = audio_int16.shape[1]
                raw_data = audio_int16.tobytes()
            
            segment = AudioSegment(
                data=raw_data,
                sample_width=2,
                frame_rate=sample_rate,
                channels=channels
            )
            
            # Map extension to pydub format
            fmt = 'mp3'
            if ext in ('.m4a', '.aac'): fmt = 'ipod'
            
            segment.export(str(output_path), format=fmt, bitrate=f'{bitrate}k')
            return
        except Exception as e:
            logger.warning(f"Failed to write lossy format {ext}: {e}. Fallback to FLAC.")
            
    # 3. Fallback: Write as FLAC if lossy write failed or not supported
    fallback_path = output_path.with_suffix('.flac')
    sf.write(str(fallback_path), audio, sample_rate)
    if fallback_path != output_path:
        logger.info(f"Saved as FLAC instead: {fallback_path.name}") 

# =============================================================================
# Async Audio Saver
# =============================================================================

class AsyncAudioSaver:
    """
    Background worker to save audio files handling format conversion and disk I/O
    without blocking the GPU inference loop.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, max_queue_size: int = 16):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info(f"AsyncAudioSaver started (queue size: {max_queue_size})")

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _worker_loop(self):
        while self.running or not self.queue.empty():
            try:
                # Timeout allows checking self.running periodically during idle
                task = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            try:
                # Unpack task
                audio, path, sr, source_path = task
                save_audio_smart(audio, path, sr, source_path)
            except Exception as e:
                logger.error(f"Async save failed for {path}: {e}")
            finally:
                self.queue.task_done()
                
    def save_async(self, audio: np.ndarray, output_path: Path, 
                   sample_rate: int, source_path: Optional[Path] = None):
        """Submit an audio file to be saved in the background."""
        # Check if queue is full - if so, we block to avoid OOM
        if self.queue.full():
            logger.debug("Save queue full, waiting...")
            
        # Copy audio array to ensure thread safety (detach from reused buffers)
        audio_copy = audio.copy()
        
        self.queue.put((audio_copy, output_path, sample_rate, source_path))

    @classmethod
    def shutdown(cls):
        if cls._instance:
            logger.info("Waiting for pending audio saves...")
            cls._instance.running = False
            cls._instance.queue.join()  # Wait for all tasks to complete
            cls._instance.worker_thread.join()
            cls._instance = None
            logger.info("AsyncAudioSaver stopped")


# Register cleanup on exit
atexit.register(AsyncAudioSaver.shutdown)


# =============================================================================
# Configuration Data Classes (Mirrored from optimized script)
# =============================================================================

@dataclass
class AudioConfig:
    sample_rate: int = 44100
    chunk_size: int = 485100
    hop_length: int = 441
    n_fft: int = 2048
    num_channels: int = 2
    dim_f: int = 1024
    dim_t: int = 1101
    min_mean_abs: float = 0.0

@dataclass
class ModelConfig:
    dim: int = 512
    depth: int = 12
    stereo: bool = True
    num_stems: int = 1
    # ... other fields with defaults ...
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
    is_mel_band: bool = False

@dataclass
class InferenceConfig:
    batch_size: int = 1
    num_overlap: int = 2
    dim_t: int = 1101
    use_compile: bool = False
    compile_mode: str = "max-autotune-no-cudagraphs"
    use_fp16: bool = True
    prefetch_chunks: int = 2


# =============================================================================
# Helper Classes & Functions
# =============================================================================

class ChunkPrefetcher:
    """Async chunk prefetcher for overlapping I/O and compute."""
    def __init__(self, audio: np.ndarray, chunk_size: int, overlap: int, prefetch_count: int = 2):
        self.audio = audio
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap
        self.num_chunks = max(1, (len(audio) - overlap) // self.step + 1)
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.stop_event = threading.Event()
        self.thread = None
    
    def _prefetch_worker(self):
        for i in range(self.num_chunks):
            if self.stop_event.is_set(): break
            start = i * self.step
            end = min(start + self.chunk_size, len(self.audio))
            chunk = self.audio[start:end]
            if len(chunk) < self.chunk_size:
                pad_size = self.chunk_size - len(chunk)
                chunk = np.pad(chunk, ((0, pad_size), (0, 0)) if chunk.ndim > 1 else (0, pad_size), mode='constant')
            self.queue.put((i, start, end, chunk))
        self.queue.put(None)
    
    def start(self):
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()
    
    def __iter__(self):
        self.start()
        while True:
            item = self.queue.get()
            if item is None: break
            yield item

def load_audio(audio_path: Path, sample_rate: int = 44100) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(audio_path), dtype='float32')
    if sr != sample_rate:
        import resampy
        audio = resampy.resample(audio, sr, sample_rate)
        sr = sample_rate
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio, sr

def normalize_audio(audio: np.ndarray, max_peak: float = 0.9, min_peak: float = 1e-8) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > min_peak:
        audio = audio * (max_peak / peak)
    return audio

# Replaced by save_audio_smart, keeping for internal non-smart usage if needed
def save_audio(audio: np.ndarray, output_path: Path, sample_rate: int = 44100):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate)

def load_config_from_yaml(config_path: Path) -> Tuple[AudioConfig, ModelConfig, InferenceConfig]:
    with open(config_path, 'r') as f:
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
    if isinstance(freqs_per_bands, list): freqs_per_bands = tuple(freqs_per_bands)
    
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
        batch_size=inf_section.get('batch_size', 1),
        num_overlap=inf_section.get('num_overlap', 2),
        dim_t=inf_section.get('dim_t', model_cfg.dim_head),
    )
    
    return audio_cfg, model_cfg, inf_cfg

def create_model(model_cfg: ModelConfig, device: torch.device) -> torch.nn.Module:
    kwargs = {
        'dim': model_cfg.dim,
        'depth': model_cfg.depth,
        'stereo': model_cfg.stereo,
        'num_stems': model_cfg.num_stems,
        'time_transformer_depth': model_cfg.time_transformer_depth,
        'freq_transformer_depth': model_cfg.freq_transformer_depth,
        'dim_head': model_cfg.dim_head,
        'heads': model_cfg.heads,
        'attn_dropout': model_cfg.attn_dropout,
        'ff_dropout': model_cfg.ff_dropout,
        'stft_n_fft': model_cfg.stft_n_fft,
        'stft_hop_length': model_cfg.stft_hop_length,
        'stft_win_length': model_cfg.stft_win_length,
        'stft_normalized': model_cfg.stft_normalized,
        'mask_estimator_depth': model_cfg.mask_estimator_depth,
        'mlp_expansion_factor': model_cfg.mlp_expansion_factor,
    }
    if model_cfg.freqs_per_bands: kwargs['freqs_per_bands'] = model_cfg.freqs_per_bands
    
    model = MelBandRoformer(**kwargs) if model_cfg.is_mel_band else BSRoformer(**kwargs)
    return model.to(device)

def load_model_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    logger.debug(f"Loading weights: {checkpoint_path}")
    if checkpoint_path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state_dict = load_file(str(checkpoint_path))
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model

@torch.inference_mode()
def separate_audio(model: torch.nn.Module, audio: np.ndarray, audio_cfg: AudioConfig, 
                   model_cfg: ModelConfig, inf_cfg: InferenceConfig, device: torch.device) -> np.ndarray:
    
    chunk_size = audio_cfg.chunk_size
    overlap = chunk_size // inf_cfg.num_overlap
    step = chunk_size - overlap
    
    # Initialize output
    total_samples = len(audio)
    num_stems = model_cfg.num_stems
    output = np.zeros((num_stems, total_samples, audio.shape[1]), dtype=np.float32)
    weight = np.zeros(total_samples, dtype=np.float32)
    window = np.hanning(chunk_size).astype(np.float32)
    
    prefetcher = ChunkPrefetcher(audio, chunk_size, overlap, prefetch_count=inf_cfg.prefetch_chunks)
    
    batch_chunks, batch_positions = [], []
    
    for item in prefetcher:
        chunk_idx, start, end, chunk = item
        batch_chunks.append(chunk)
        batch_positions.append((start, end, min(end - start, chunk_size)))
        
        if len(batch_chunks) >= inf_cfg.batch_size or chunk_idx == prefetcher.num_chunks - 1:
            batch = np.stack(batch_chunks).transpose(0, 2, 1) # (B, C, T)
            batch_tensor = torch.from_numpy(batch).to(device)
            if inf_cfg.use_fp16: batch_tensor = batch_tensor.half()
            
            with torch.amp.autocast(device_type='cuda', enabled=inf_cfg.use_fp16):
                 # Run model
                 # On AMD ROCm, we might need to control Flash Attention
                 is_amd_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
                 if inf_cfg.use_compile and is_amd_rocm:
                     # Attempt to use new context manager if available
                     if hasattr(torch.nn.attention, 'sdpa_kernel'):
                         # PyTorch 2.4+ context manager
                         with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                             separated = model(batch_tensor)
                     else:
                         # Legacy context manager
                         with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                             separated = model(batch_tensor)
                 else:
                     separated = model(batch_tensor)
            
            if separated.dim() == 3: separated = separated.unsqueeze(1)
            
            separated = separated.float().cpu().numpy().transpose(0, 1, 3, 2) # (B, Stems, T, C)
            
            for b_idx, (pos_start, pos_end, actual_len) in enumerate(batch_positions):
                chunk_out = separated[b_idx, :, :actual_len, :]
                win_len = min(len(window), actual_len)
                chunk_out[:, :win_len, :] *= window[:win_len].reshape(1, win_len, 1)
                
                output[:, pos_start:pos_start + actual_len, :] += chunk_out
                weight[pos_start:pos_start + actual_len] += window[:actual_len]
            
            batch_chunks, batch_positions = [], []
            
    weight = np.maximum(weight, 1e-8)
    output /= weight[np.newaxis, :, np.newaxis]
    return output


# =============================================================================
# Public Interface (Demucs Compatible)
# =============================================================================

def discover_model_path(model_dir: Path, model_name: str) -> Optional[Dict[str, Path]]:
    """Scan directory for config and checkpoint for specific model name."""
    search_path = model_dir / model_name
    if not search_path.exists():
        search_path = model_dir
    
    # helper to find ckpt in a folder
    def find_ckpt(folder):
        candidates = list(folder.glob("*.ckpt")) + list(folder.glob("*.pth")) + list(folder.glob("*.safetensors"))
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_size)
        return None

    # Strategy 1: Look for exact model folder with any yaml
    if (model_dir / model_name).exists():
        folder = model_dir / model_name
        yamls = list(folder.glob("*.yaml"))
        if yamls:
            # Prefer config.yaml if exists, else first yaml
            config_path = next((y for y in yamls if y.name == 'config.yaml'), yamls[0])
            ckpt = find_ckpt(folder)
            if ckpt:
                return {'config': config_path, 'checkpoint': ckpt, 'path': folder}

    # Strategy 2: Recursive search for config.yaml (legacy)
    for path in search_path.rglob('config.yaml'):
        if model_name in str(path.parent):
            ckpt = find_ckpt(path.parent)
            if ckpt:
                return {'config': path, 'checkpoint': ckpt, 'path': path.parent}
    
    return None

def load_bs_roformer(model_name: str, model_dir: Union[str, Path], device: str = 'cuda'):
    """
    Loads a BS-Roformer model and its configurations.
    
    Args:
        model_name: Name of the model to load (e.g., 'bs_roformer_4stem').
        model_dir: Directory where models are stored.
        device: Device to load the model onto ('cuda' or 'cpu').
        
    Returns:
        Tuple[torch.nn.Module, ModelConfig, AudioConfig, InferenceConfig]: 
            The loaded model, its model configuration, audio configuration, and inference configuration.
    """
    model_dir = Path(model_dir)
    model_info = discover_model_path(model_dir, model_name)
    if not model_info:
        raise FileNotFoundError(f"Model {model_name} not found in {model_dir}")
        
    device_obj = torch.device(device)
    
    logger.info(f"Loading BS-RoFormer: {model_name}")
    audio_cfg, model_cfg, inf_cfg = load_config_from_yaml(model_info['config'])
    
    model = create_model(model_cfg, device_obj)
    model = load_model_weights(model, model_info['checkpoint'], device_obj)
    model.eval()
    
    # Attaching config to separator for reuse
    model.model_cfg = model_cfg
    model.audio_cfg = audio_cfg
    model.inf_cfg = inf_cfg
    
    return model, model_cfg, audio_cfg, inf_cfg

def separate_organized_folder(
    folder_path: Path, 
    model_name: str = 'model', 
    model_dir: str = '', 
    batch_size: int = 1,
    overwrite: bool = True,
    device: str = 'cuda',
    separator: Optional[torch.nn.Module] = None
) -> Dict[str, Path]:
    """
    Separate stems for a single organized folder using BS-RoFormer.
    Matches signature of demucs_sep.separate_organized_folder roughly.
    
    Args:
        folder_path: Path to organized folder
        model_name: Name of model directory
        model_dir: Root directory for models
        batch_size: Batch size for inference
        overwrite: Whether to overwrite existing stems
        device: Device to run on
        separator: Optional pre-loaded BSRoformer instance (avoids reloading)
    """
    folder_path = Path(folder_path)
    
    # If model_dir is not provided and model_name is, try to infer default model path
    if not model_dir and model_name and not separator:
         model_dir = str(Path(__file__).parent.parent.parent / 'models' / 'bs-roformer')
    
    if not TORCH_AVAILABLE or not BS_ROFORMER_AVAILABLE:
        raise ImportError("PyTorch and audio-separator are required for BS-RoFormer.")
    
    # 1. Input Validation
    full_mix = None
    for ext in AUDIO_EXTENSIONS:
        p = folder_path / f"full_mix{ext}"
        if p.exists():
            full_mix = p
            break
            
    if not full_mix:
        raise FileNotFoundError(f"No full_mix found in {folder_path}")
    
    # 2. Check overlap (Standard 4 stems)
    stem_names = ['drums', 'bass', 'other', 'vocals']
    if not overwrite:
        existing = {}
        for s in stem_names:
            p = folder_path / f"{s}.wav" # Roformer outputs WAV
            if p.exists(): existing[s] = p
        
        if len(existing) == 4:
            logger.info(f"Stems already exist in {folder_path.name}")
            return existing

    # 3. Load Model
    try:
        if separator is None:
            # Load model (legacy path - reloads per folder)
            separator, model_cfg, audio_cfg, inf_cfg = load_bs_roformer(model_name, model_dir, device=device)
        else:
            # Use pre-loaded model (fast path)
            # We still need model_cfg, which we can attach to the separator or infer
            # Ideally load_bs_roformer should attach config to separator, 
            # but for now we'll assume standard 4-stem or recover from attributes
            if hasattr(separator, 'model_cfg') and hasattr(separator, 'audio_cfg') and hasattr(separator, 'inf_cfg'):
                model_cfg = separator.model_cfg
                audio_cfg = separator.audio_cfg
                inf_cfg = separator.inf_cfg
            else:
                # Fallback config - this might not be accurate for all models
                logger.warning("Pre-loaded separator does not have attached configs. Using default ModelConfig.")
                model_cfg = ModelConfig(num_stems=4)
                audio_cfg = AudioConfig()
                inf_cfg = InferenceConfig()
        
        inf_cfg.batch_size = batch_size # Ensure batch size is updated from function argument
        device_obj = torch.device(device) # Ensure device_obj is defined
    except Exception as e:
        logger.error(f"Error loading model for {folder_path.name}: {e}")
        raise

    # 4. Process Audio
    logger.info(f"Processing {full_mix.name}...")
    audio, sr = load_audio(full_mix, audio_cfg.sample_rate)
    
    # Normalize input
    audio = normalize_audio(audio)
    
    start_t = time.time()
    stems = separate_audio(separator, audio, audio_cfg, model_cfg, inf_cfg, device_obj)
    duration = time.time() - start_t
    logger.info(f"Separation finished in {duration:.1f}s")
    
    # 5. Save Output with Smart Formatting
    output_paths = {}
    
    # Determine output format and bitrate based on input
    input_ext = full_mix.suffix.lower()
    
    # Logic:
    # - If input is lossless (level 1 check): Use FLAC
    # - If input is lossy (mp3, m4a): Use same extension
    LOSSLESS_EXTS = {'.flac', '.wav', '.aiff', '.aif'}
    
    if input_ext in LOSSLESS_EXTS:
        target_ext = '.flac'
    else:
        target_ext = input_ext # e.g., .mp3
        
    instruments = model_cfg.instruments
    if not instruments:
         if model_cfg.num_stems == 4:
             instruments = ['drums', 'bass', 'other', 'vocals'] # Default assumption
         else:
             instruments = [f"stem_{i}" for i in range(model_cfg.num_stems)]
             
    # Map stems to names
    stem_map = {}
    for i in range(min(len(instruments), stems.shape[0])):
        stem_map[instruments[i]] = stems[i]
        
    STANDARD_STEMS = {'drums', 'bass', 'vocals'}
    extra_stems_audio = []
    
    # Helper to save (async)
    saver = AsyncAudioSaver.get_instance()
    
    def save_stem(audio_data, file_name_base):
        # Normalize stem
        normalized = normalize_audio(audio_data)
        out_path = folder_path / f"{file_name_base}{target_ext}"
        
        # Submit to background thread
        saver.save_async(normalized, out_path, sr, source_path=full_mix)
        return out_path

    # Iterate all stems
    for name, audio_data in stem_map.items():
        if name in STANDARD_STEMS:
            # Save standard stems
            output_paths[name] = save_stem(audio_data, name)
        else:
            # Non-standard stems
            # 1. Accumulate for combined 'other'
            extra_stems_audio.append(audio_data)
            
            # 2. Save individually
            if name != 'other':
                save_stem(audio_data, name)
                logger.info(f"Saved extra stem: {name}{target_ext}")

    # Save Mixed 'other' stem
    if extra_stems_audio:
        # Sum raw audio THEN normalize
        combined_other = np.sum(np.stack(extra_stems_audio), axis=0)
        output_paths['other'] = save_stem(combined_other, "other")
        logger.info(f"Saved combined 'other' stem (mixed {len(extra_stems_audio)} sources)")
        
    return output_paths

def batch_separate_stems(root_dir: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """Batch process."""
    root_dir = Path(root_dir)
    folders = find_organized_folders(root_dir)
    stats = {'total': len(folders), 'success': 0, 'failed': 0, 'skipped': 0, 'errors': []}
    
    # We load model ONCE here for efficiency if possible, 
    # but to keep signature simple we rely on separate_organized_folder loading it.
    # To optimize: Refactor to load model once and pass it down.
    # For now, simplistic iteration.
    
    for folder in folders:
        try:
            separate_organized_folder(folder, **kwargs)
            stats['success'] += 1
        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append(f"{folder.name}: {e}")
            logger.error(f"Failed {folder.name}: {e}")
            
    return stats
