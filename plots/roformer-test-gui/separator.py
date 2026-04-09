import os
import sys
import time
import logging
import threading
import queue
import atexit
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

# Setup paths to ensure we can import the project core
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.rocm_env import setup_rocm_env
setup_rocm_env()

import numpy as np
import yaml
import soundfile as sf
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.functional')
warnings.filterwarnings('ignore', category=FutureWarning, module='rotary_embedding_torch')

import torch
import torch.nn.functional as F
from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer

logger = logging.getLogger(__name__)

MelBandRoformer = BSRoformer

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

class ChunkPrefetcher:
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
        audio = resampy.resample(audio, sr, sample_rate, axis=0)
        sr = sample_rate
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    elif audio.shape[1] == 1:
        audio = np.concatenate([audio, audio], axis=-1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio, sr

def normalize_audio(audio: np.ndarray, max_peak: float = 0.9, min_peak: float = 1e-8) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > min_peak:
        audio = audio * (max_peak / peak)
    return audio

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

def load_model_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device,
                       model_name: str = '') -> torch.nn.Module:
    if checkpoint_path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state_dict = load_file(str(checkpoint_path))
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint)) if isinstance(checkpoint, dict) else checkpoint

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        tag = f'[{model_name}] ' if model_name else ''
        if result.missing_keys:
            logger.warning(f'{tag}Checkpoint missing {len(result.missing_keys)} keys (will use random init)')
        if result.unexpected_keys:
            logger.warning(f'{tag}Checkpoint has {len(result.unexpected_keys)} unexpected keys (shape mismatch or extra weights)')
        logger.warning(f'{tag}This usually means the YAML config does not match the checkpoint architecture.')
    return model

@torch.inference_mode()
def separate_audio(model: torch.nn.Module, audio: np.ndarray, audio_cfg: AudioConfig, 
                   model_cfg: ModelConfig, inf_cfg: InferenceConfig, device: torch.device) -> np.ndarray:
    
    chunk_size = audio_cfg.chunk_size
    overlap = chunk_size // inf_cfg.num_overlap
    step = chunk_size - overlap
    
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
            batch = np.stack(batch_chunks).transpose(0, 2, 1)
            batch_tensor = torch.from_numpy(batch).to(device)
            if inf_cfg.use_fp16: batch_tensor = batch_tensor.half()
            
            with torch.amp.autocast(device_type='cuda', enabled=inf_cfg.use_fp16):
                 is_amd_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
                 if inf_cfg.use_compile and is_amd_rocm:
                     with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                         separated = model(batch_tensor)
                 else:
                     separated = model(batch_tensor)
            
            if separated.dim() == 3: separated = separated.unsqueeze(1)
            separated = separated.float().cpu().numpy().transpose(0, 1, 3, 2)
            
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

def discover_model_path(model_dir: Path, model_name: str) -> Optional[Dict[str, Path]]:
    search_path = model_dir / model_name
    if not search_path.exists():
        search_path = model_dir
    
    def find_ckpt(folder):
        candidates = list(folder.glob("*.ckpt")) + list(folder.glob("*.pth")) + list(folder.glob("*.safetensors"))
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_size)
        return None

    if (model_dir / model_name).exists():
        folder = model_dir / model_name
        yamls = list(folder.glob("*.yaml"))
        if yamls:
            config_path = next((y for y in yamls if y.name == 'config.yaml'), yamls[0])
            ckpt = find_ckpt(folder)
            if ckpt:
                return {'config': config_path, 'checkpoint': ckpt, 'path': folder}

    for path in search_path.rglob('config.yaml'):
        if model_name in str(path.parent):
            ckpt = find_ckpt(path.parent)
            if ckpt:
                return {'config': path, 'checkpoint': ckpt, 'path': path.parent}
    
    return None

def is_model_supported(model_dir: Union[str, Path], model_name: str) -> bool:
    """Returns False for MelBandRoformer checkpoints (different architecture, not supported)."""
    model_info = discover_model_path(Path(model_dir), model_name)
    if not model_info:
        return False
    with open(model_info['config'], 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    freqs_per_bands = cfg.get('model', {}).get('freqs_per_bands', [])
    return len(freqs_per_bands) > 0


def get_model_stems(model_dir: Union[str, Path], model_name: str) -> List[str]:
    """Return the stem names a model outputs, by reading its config YAML."""
    model_info = discover_model_path(Path(model_dir), model_name)
    if not model_info:
        return []
    with open(model_info['config'], 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    training = cfg.get('training', {})
    model_section = cfg.get('model', {})
    target = training.get('target_instrument')
    num_stems = model_section.get('num_stems', 1)
    instruments = training.get('instruments', [])
    if target and num_stems == 1:
        return [target]
    elif instruments:
        return list(instruments)
    elif num_stems == 4:
        return ['drums', 'bass', 'other', 'vocals']
    else:
        return [f"stem_{i}" for i in range(num_stems)]


def separate_file_with_demucs(
    input_file: Path,
    output_dir: Path,
    model_name: str = "htdemucs",
    shifts: int = 0,
    overlap: float = 0.25,
    device: str = "cuda",
    callback=None,
) -> None:
    from demucs.api import Separator
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if callback: callback(f"Loading Demucs {model_name}...")
    sep = Separator(model=model_name, device=device, shifts=shifts, overlap=overlap)

    if callback: callback("Separating (Demucs)...")
    _, stems = sep.separate_audio_file(Path(input_file))

    if callback: callback("Saving stems...")
    sr = sep.samplerate
    for stem_name, audio in stems.items():
        audio_np = audio.cpu().numpy().T  # (channels, samples) → (samples, channels)
        sf.write(str(output_dir / f"{stem_name}.flac"), audio_np, sr)

    if callback: callback("Done.")


def load_bs_roformer(model_name: str, model_dir: Union[str, Path], device: str = 'cuda'):
    model_dir = Path(model_dir)
    model_info = discover_model_path(model_dir, model_name)
    if not model_info:
        raise FileNotFoundError(f"Model {model_name} not found in {model_dir}")
        
    device_obj = torch.device(device)
    audio_cfg, model_cfg, inf_cfg = load_config_from_yaml(model_info['config'])
    
    model = create_model(model_cfg, device_obj)
    model = load_model_weights(model, model_info['checkpoint'], device_obj, model_name=model_name)
    model.eval()
    
    model.model_cfg = model_cfg
    model.audio_cfg = audio_cfg
    model.inf_cfg = inf_cfg
    
    return model, model_cfg, audio_cfg, inf_cfg

def separate_file_to_dir(
    input_file: Path, 
    output_dir: Path, 
    model_name: str, 
    model_dir: str = '', 
    batch_size: int = 1,
    device: str = 'cuda',
    callback=None
) -> None:
    """
    Separates a single file using the specified model and saves stems to `output_dir`.
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    if not model_dir:
        model_dir = str(Path(__file__).parent.parent / 'models' / 'bs-roformer')
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if callback: callback(f"Loading {model_name}...")
    separator, model_cfg, audio_cfg, inf_cfg = load_bs_roformer(model_name, model_dir, device=device)
    inf_cfg.batch_size = batch_size
    
    if callback: callback("Loading audio...")
    audio, sr = load_audio(input_file, audio_cfg.sample_rate)
    audio = normalize_audio(audio)
    
    if callback: callback("Separating (inference)...")
    stems = separate_audio(separator, audio, audio_cfg, model_cfg, inf_cfg, torch.device(device))
    
    instruments = model_cfg.instruments
    if model_cfg.target_instrument and model_cfg.num_stems == 1:
        instruments = [model_cfg.target_instrument]
    elif not instruments:
        if model_cfg.num_stems == 4:
            instruments = ['drums', 'bass', 'other', 'vocals']
        else:
            instruments = [f"stem_{i}" for i in range(model_cfg.num_stems)]
             
    if callback: callback("Saving stems...")
    
    # Roformer returns in (num_stems, T, C) format
    for i in range(min(len(instruments), stems.shape[0])):
        stem_name = instruments[i]
        audio_data = normalize_audio(stems[i])
        
        # Save straight to flac
        out_path = output_dir / f"{stem_name}.flac"
        sf.write(str(out_path), audio_data, sr)
        
    # Standardize to have exactly drums, bass, other, vocals if possible?
    # Not strictly necessary if GUI parses files directly by extension.
    if callback: callback("Done.")
