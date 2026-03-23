#!/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python
"""
latent_server.py — HTTP server for decoding VAE latents to WAV audio.

Loads the Stable Audio Small autoencoder at startup, then serves decoded
WAV audio for latent .npy files on request.  Designed to back the latent
audio player embedded in feature_explorer.html.

Endpoints:
  GET /status
      → JSON: {ok, sample_rate, downsampling_ratio, latent_dir, device}

  GET /decode?track=TRACK_FOLDER_NAME&position=0.5
      Finds the crop whose companion .json "position" field is closest
      to the requested value (0.0–1.0), decodes it through the VAE, and
      returns WAV audio (stereo 44100 Hz int16).
      Response headers:
        X-Crop-Count    — total crops in that track folder
        X-Crop-Position — actual position value of the chosen crop

  GET /crops?track=TRACK_FOLDER_NAME
      → JSON array of {path, position} for all full-mix crops in track.

Configuration:
  Reads latent_player.ini from the MIR project root by default.
  Set sao_dir in the [model] section to point at the stable-audio-tools
  installation; model_config and ckpt_path default to paths under sao_dir.

Usage:
    source /home/kim/Projects/SAO/stable-audio-tools/rocm_env.sh
    python scripts/latent_server.py
    # or with an explicit config:
    python scripts/latent_server.py --config /path/to/latent_player.ini
"""

import os

# ROCm env vars must be set before importing torch
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "1")
os.environ.setdefault("PYTORCH_TUNABLEOP_TUNING", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:512")
os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("TORCH_COMPILE", "0")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import configparser
import io
import json
import math
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf
import torch

# latent_crossfader.py lives alongside this script
sys.path.insert(0, str(Path(__file__).parent))
from latent_crossfader import STEMS, crossfade_stems, lerp, load_latent, slerp

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict

# Default ini location: MIR project root (parent of scripts/)
DEFAULT_INI = Path(__file__).parent.parent / "latent_player.ini"

STEM_SUFFIXES = {"_bass", "_drums", "_other", "_vocals"}

# ---------------------------------------------------------------------------
# Globals (set in main, read-only afterwards)
# ---------------------------------------------------------------------------
_autoencoder        = None
_sample_rate        = None
_downsampling_ratio = None
_latent_dir         = None
_stem_dir           = None   # separate root for stem latents
_raw_audio_dir      = None   # root of original source audio (for raw=1 playback)
_dtype              = None
_device             = None
_decode_lock        = threading.Lock()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_autoencoder(model_config_path: Path, ckpt_path: Path,
                     model_half: bool, device: str):
    with open(model_config_path) as f:
        model_config = json.load(f)

    sample_rate        = model_config["sample_rate"]
    sample_size        = model_config["sample_size"]
    downsampling_ratio = model_config["model"]["pretransform"]["config"]["downsampling_ratio"]

    print(f"Creating model from config ({model_config_path.name})")
    model = create_model_from_config(model_config)

    print(f"Loading checkpoint from {ckpt_path.name} ...")
    copy_state_dict(model, load_ckpt_state_dict(str(ckpt_path)))

    if not hasattr(model, "pretransform") or model.pretransform is None:
        raise ValueError("Model has no pretransform — expected a diffusion_cond model")
    autoencoder = model.pretransform
    del model

    autoencoder.eval().requires_grad_(False)
    if model_half:
        autoencoder.to(torch.float16)
    autoencoder.to(device)

    print(f"Autoencoder ready — sr={sample_rate}, sample_size={sample_size}, "
          f"downsampling_ratio={downsampling_ratio}")
    return autoencoder, sample_rate, sample_size, downsampling_ratio


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------

def is_full_mix(path: Path) -> bool:
    for s in STEM_SUFFIXES:
        if path.stem.endswith(s):
            return False
    return True


def find_crops(track_dir: Path) -> list:
    """Return list of (npy_path, position) sorted by position field."""
    crops = []
    for npy in sorted(track_dir.glob("*.npy")):
        if not is_full_mix(npy):
            continue
        json_path = npy.with_suffix(".json")
        position = 0.0
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
                position = float(data.get("position", 0.0))
            except Exception:
                pass
        crops.append((npy, position))
    crops.sort(key=lambda x: x[1])
    return crops


def find_stem_crops(track_dir: Path, stem: str) -> list:
    """Return list of (npy_path, position) for a specific stem, sorted by position."""
    suffix = f"_{stem}"
    crops = []
    for npy in sorted(track_dir.glob("*.npy")):
        if not npy.stem.endswith(suffix):
            continue
        json_path = npy.with_suffix(".json")
        position = 0.0
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
                position = float(data.get("position", 0.0))
            except Exception:
                pass
        crops.append((npy, position))
    crops.sort(key=lambda x: x[1])
    return crops


def find_best_crop(crops: list, target_position: float):
    """Return (npy_path, position) with position closest to target."""
    if not crops:
        return None, None
    best = min(crops, key=lambda x: abs(x[1] - target_position))
    return best


# ---------------------------------------------------------------------------
# Raw audio helpers
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = [".flac", ".wav", ".mp3", ".ogg"]


def find_raw_audio(npy_path: Path, latent_root: Path) -> Path | None:
    """Given a latent .npy path, return the corresponding source audio file."""
    if _raw_audio_dir is None:
        return None
    rel = npy_path.relative_to(latent_root).with_suffix("")
    for ext in AUDIO_EXTENSIONS:
        candidate = _raw_audio_dir / rel.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def _load_crop_downbeats(npy_path: Path, latent_root: Path) -> list | None:
    """Return downbeat timestamps (seconds) for the crop corresponding to a latent .npy.

    Looks for a .DOWNBEATS sidecar next to the source audio in raw_audio_dir.
    Returns None if unavailable or fewer than 2 downbeats.
    """
    if _raw_audio_dir is None:
        return None
    try:
        rel  = npy_path.relative_to(latent_root).with_suffix('.DOWNBEATS')
        path = _raw_audio_dir / rel
        if not path.exists():
            return None
        times = [float(line.strip()) for line in path.read_text().splitlines() if line.strip()]
        return times if len(times) >= 2 else None
    except Exception:
        return None


def load_raw_wav(audio_path: Path, smart_loop: bool = False) -> bytes:
    """Load a source audio file and return WAV bytes (stereo, _sample_rate, int16)."""
    audio_np, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio_np = audio_np.T   # [channels, samples]

    # Ensure stereo
    if audio_np.shape[0] == 1:
        audio_np = np.repeat(audio_np, 2, axis=0)
    elif audio_np.shape[0] > 2:
        audio_np = audio_np[:2]

    # Resample if needed (source files should already be 44100, but be safe)
    if sr != _sample_rate:
        import torchaudio
        t = torch.from_numpy(audio_np)
        t = torchaudio.functional.resample(t, sr, _sample_rate)
        audio_np = t.numpy()

    if smart_loop:
        companion = {}
        json_path = audio_path.with_suffix(".json")
        if not json_path.exists():
            # Try the companion JSON in the latent dir via the npy path (not available here)
            pass
        else:
            try:
                companion = json.loads(json_path.read_text())
            except Exception:
                pass
        raw_bpm = companion.get("bpm")
        if raw_bpm:
            try:
                start, end = _smart_loop_points(audio_np, _sample_rate, float(raw_bpm))
                audio_np = audio_np[:, start:end]
            except Exception as e:
                print(f"  smart_loop error (raw): {e}")

    audio_np  = np.clip(audio_np, -1.0, 1.0)
    audio_i16 = (audio_np * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(_sample_rate)
        wf.writeframes(audio_i16.T.flatten().tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Smart loop helpers
# ---------------------------------------------------------------------------

def _nearest_zero_crossing(mono: np.ndarray, target: int, window: int) -> int:
    """Return the sample index of the zero-crossing nearest to `target`
    within ±window samples.  Uses the mono mix signal."""
    n  = len(mono)
    lo = max(0, target - window)
    hi = min(n - 2, target + window)
    seg = mono[lo:hi + 1]
    crossings = np.where(seg[:-1] * seg[1:] <= 0)[0] + lo
    if len(crossings) == 0:
        return target
    return int(crossings[np.argmin(np.abs(crossings - target))])


def _smart_loop_points(audio_np: np.ndarray, sample_rate: int,
                       bpm: float) -> tuple:
    """Return (start, end) sample indices for a loop whose length is a
    multiple of 4 bars (4 downbeats), bounded by zero-crossings."""
    n          = audio_np.shape[1]
    beat_s     = int(round(60.0 / bpm * sample_rate))
    bar_s      = 4 * beat_s
    four_bar_s = 4 * bar_s

    n_phrases  = max(1, n // four_bar_s)
    loop_len   = n_phrases * four_bar_s

    mono   = (audio_np[0] + audio_np[1]) / 2.0
    window = beat_s

    start = _nearest_zero_crossing(mono, 0, window)
    end   = _nearest_zero_crossing(mono, min(start + loop_len, n - 1), window)
    end   = min(end, n)

    if end <= start:
        return 0, n

    duration_s = (end - start) / sample_rate
    n_bars     = round((end - start) / bar_s)
    print(f"  smart loop: {n_bars} bars  {duration_s:.2f}s  "
          f"[{start}–{end}]  bpm={bpm:.1f}")
    return start, end


# ---------------------------------------------------------------------------
# Beat-match helpers: pitch shift + time stretch, then re-encode
# ---------------------------------------------------------------------------

def _apply_pitch_stretch(audio_np: np.ndarray, sr: int,
                         semitones: float, stretch: float,
                         algo: str = 'pedalboard') -> np.ndarray:
    """Apply pitch shift and/or time stretch to audio [ch, samples].

    Convention (all algorithms):
      stretch > 1 → audio longer (BPM decreases) — same as pedalboard stretch_factor.
      semitones > 0 → pitch up.

    algo: 'pedalboard' | 'bungee' | 'rubberband'
    Falls back to the next available algorithm if the chosen one fails.
    """
    if abs(semitones) < 1e-4 and abs(stretch - 1.0) < 1e-4:
        return audio_np

    def _try_bungee():
        from bungee_python import bungee as bungee_lib
        channels = audio_np.shape[0]
        stretcher = bungee_lib.Bungee(sample_rate=sr, channels=channels)
        if abs(semitones) > 1e-4:
            stretcher.set_pitch(2.0 ** (semitones / 12.0))
        if abs(stretch - 1.0) > 1e-4:
            # bungee set_speed is a speed multiplier (>1 = faster),
            # but our stretch is a duration ratio (>1 = slower), so invert.
            stretcher.set_speed(1.0 / stretch)
        audio_in = audio_np.T.astype(np.float32)          # [samples, channels]
        out = np.asarray(stretcher.process(audio_in), dtype=np.float32).copy()
        if out.ndim == 1:
            out = np.column_stack([out] * channels)
        elif out.shape[1] != channels:
            out = np.tile(out[:, :1], (1, channels))
        return out.T                                        # [ch, samples]

    def _try_pedalboard():
        import pedalboard
        if not hasattr(pedalboard, 'time_stretch'):
            raise ImportError("pedalboard.time_stretch not available")
        return pedalboard.time_stretch(
            audio_np.astype(np.float32),
            float(sr),
            stretch_factor=float(stretch),
            pitch_shift_in_semitones=float(semitones),
            high_quality=True,
            preserve_formants=False,
        ).astype(np.float32)

    def _try_rubberband():
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fi, \
             tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fo:
            sf.write(fi.name, audio_np.T.astype(np.float32), sr)
            cmd = ['rubberband', '-q']
            if abs(stretch - 1.0) > 1e-4:
                # --time is duration ratio: >1 = longer/slower — same as our stretch
                cmd += ['--time', str(stretch)]
            if abs(semitones) > 1e-4:
                cmd += ['--pitch', str(semitones)]
            cmd += [fi.name, fo.name]
            subprocess.run(cmd, check=True, capture_output=True)
            out, _ = sf.read(fo.name, dtype='float32', always_2d=True)
        import os
        for p in [fi.name, fo.name]:
            try: os.remove(p)
            except Exception: pass
        return out.T    # [ch, samples]

    order = {
        'bungee':     [_try_bungee,     _try_pedalboard, _try_rubberband],
        'rubberband': [_try_rubberband, _try_pedalboard, _try_bungee],
        'pedalboard': [_try_pedalboard, _try_rubberband, _try_bungee],
    }.get(algo, [_try_pedalboard, _try_rubberband, _try_bungee])

    last_err = None
    for fn in order:
        try:
            result = fn()
            if fn.__name__ != order[0].__name__:
                print(f"  bm: fell back to {fn.__name__} (primary failed: {last_err})")
            return result
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All pitch-shift algorithms failed. Last error: {last_err}")


def _encode_audio(audio_np: np.ndarray) -> torch.Tensor:
    """Encode [ch, samples] numpy audio to [1, C, T] latent tensor.

    Uses the autoencoder's encode() method; works with both tensor-returning
    and distribution-returning (DiagonalGaussianDistribution) autoencoders.
    Uses the deterministic mean for inference.
    """
    audio_t = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0).to(
        device=_device, dtype=_dtype)
    with torch.no_grad():
        latent = _autoencoder.encode(audio_t)
        if not isinstance(latent, torch.Tensor):
            if hasattr(latent, 'mean'):
                latent = latent.mean
            elif hasattr(latent, 'sample'):
                latent = latent.sample()
    return latent   # [1, C, T]


def beatmatch_crossfade_to_wav(
    track_a: str, track_b: str,
    pos_a: float, pos_b: float,
    alphas: dict, beta_a: float, beta_b: float,
    shift_a: float, stretch_a: float,
    shift_b: float, stretch_b: float,
    interp: str = 'slerp',
    smart_loop: bool = False,
    algo: str = 'pedalboard',
) -> tuple:
    """Load source stem audio, pitch-shift + time-stretch per track, encode to
    latent, then crossfade in latent space.  Requires raw_audio_dir to be set.

    shift_*/stretch_* follow the same conventions as _apply_pitch_stretch:
      shift > 0 → pitch up in semitones
      stretch > 1 → audio longer (effective BPM = original / stretch)
    """
    if _raw_audio_dir is None:
        raise ValueError(
            "raw_audio_dir not configured — cannot re-encode source audio for "
            "beat matching.  Set raw_audio_dir in latent_player.ini.")

    stem_root = _stem_dir if _stem_dir is not None else _latent_dir
    dir_a = stem_root / track_a
    dir_b = stem_root / track_b

    stems_a: dict = {}
    stems_b: dict = {}
    missing: list = []

    def _load_stem_raw(npy_path: Path) -> np.ndarray | None:
        raw = find_raw_audio(npy_path, stem_root)
        if raw is None:
            return None
        audio, sr = sf.read(str(raw), dtype='float32', always_2d=True)
        audio = audio.T   # [ch, samples]
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        elif audio.shape[0] > 2:
            audio = audio[:2]
        if sr != _sample_rate:
            import torchaudio
            t = torchaudio.functional.resample(
                torch.from_numpy(audio), sr, _sample_rate)
            audio = t.numpy()
        return audio

    # Full-mix latent crops (needed both for downbeat refinement and reality anchors)
    crops_fm_a = find_crops(_latent_dir / track_a)
    crops_fm_b = find_crops(_latent_dir / track_b)
    npy_fm_a, _ = find_best_crop(crops_fm_a, pos_a)
    npy_fm_b, _ = find_best_crop(crops_fm_b, pos_b)
    if npy_fm_a is None or npy_fm_b is None:
        raise ValueError("Full-mix latents not found for one or both tracks")

    # Refine stretch values using actual crop downbeats instead of global BPM estimate.
    # Computes bar intervals from the DOWNBEATS sidecar and meets in the middle.
    db_a = _load_crop_downbeats(npy_fm_a, _latent_dir)
    db_b = _load_crop_downbeats(npy_fm_b, _latent_dir)
    if db_a and db_b:
        iv_a = (db_a[-1] - db_a[0]) / (len(db_a) - 1)   # seconds per bar, crop A
        iv_b = (db_b[-1] - db_b[0]) / (len(db_b) - 1)   # seconds per bar, crop B
        tgt_iv = math.sqrt(iv_a * iv_b)                   # geometric mean
        ref_a  = tgt_iv / iv_a                            # duration multiplier for A
        ref_b  = tgt_iv / iv_b                            # duration multiplier for B
        # Only override if the refined values are within 5% of the client-provided ones
        # (guards against corrupt DOWNBEATS or mismatched crops)
        if abs(ref_a / max(stretch_a, 1e-6) - 1.0) < 0.05 and \
           abs(ref_b / max(stretch_b, 1e-6) - 1.0) < 0.05:
            print(f"  downbeat refinement: A ×{stretch_a:.4f}→×{ref_a:.4f}  "
                  f"B ×{stretch_b:.4f}→×{ref_b:.4f}  "
                  f"({len(db_a)} / {len(db_b)} downbeats)")
            stretch_a, stretch_b = ref_a, ref_b

    phase_offset_s = 0.0
    shared_loop = None
    if db_a and db_b:
        t_a = db_a[0] * stretch_a
        t_b = db_b[0] * stretch_b
        
        iv_a = (db_a[-1] - db_a[0]) / (len(db_a) - 1)
        iv_b = (db_b[-1] - db_b[0]) / (len(db_b) - 1)
        tgt_iv = math.sqrt(iv_a * iv_b)
        
        offset = (t_a - t_b) % tgt_iv
        if offset > tgt_iv / 2:
            offset -= tgt_iv
            
        phase_offset_s = offset
        print(f"  phase alignment: shifting B by {phase_offset_s*1000:.1f}ms")

        # Determine maximum valid loop length locked onto the exact downbeat grid
        grid_start_a = t_a
        grid_start_b = t_b + phase_offset_s
        start_s = max(grid_start_a, grid_start_b)
        
        last_a = db_a[-1] * stretch_a
        last_b = db_b[-1] * stretch_b + phase_offset_s
        
        k_a = math.floor((last_a - grid_start_a) / tgt_iv)
        last_grid_a = grid_start_a + k_a * tgt_iv
        
        k_b = math.floor((last_b - grid_start_b) / tgt_iv)
        last_grid_b = grid_start_b + k_b * tgt_iv
        
        end_s = min(last_grid_a, last_grid_b)
        n_bars = math.floor((end_s - start_s) / tgt_iv)
        
        if n_bars > 0:
            shared_loop = {"start_s": start_s, "tgt_iv": tgt_iv, "n_bars": n_bars}

    missing_latents: list = []
    missing_audio:   list = []

    stems_a_raw = {}
    stems_b_raw = {}

    for stem in STEMS:
        crops_a = find_stem_crops(dir_a, stem)
        crops_b = find_stem_crops(dir_b, stem)
        npy_a, _ = find_best_crop(crops_a, pos_a) if crops_a else (None, None)
        npy_b, _ = find_best_crop(crops_b, pos_b) if crops_b else (None, None)
        if npy_a is None or npy_b is None:
            missing_latents.append(stem)
            continue

        raw_a = _load_stem_raw(npy_a)
        raw_b = _load_stem_raw(npy_b)
        if raw_a is None or raw_b is None:
            missing_audio.append(stem)
            continue

        print(f"  bm/{stem} [{algo}]: A {shift_a:+.1f}st ×{stretch_a:.3f}  "
              f"B {shift_b:+.1f}st ×{stretch_b:.3f}")
        raw_a = _apply_pitch_stretch(raw_a, _sample_rate, shift_a, stretch_a, algo)
        raw_b = _apply_pitch_stretch(raw_b, _sample_rate, shift_b, stretch_b, algo)

        if abs(phase_offset_s) > 1e-4:
            shift_samples = int(round(phase_offset_s * _sample_rate))
            if shift_samples > 0:
                padding = np.zeros((raw_b.shape[0], shift_samples), dtype=raw_b.dtype)
                raw_b = np.concatenate([padding, raw_b], axis=1)[:, :-shift_samples]
            elif shift_samples < 0:
                shift_abs = abs(shift_samples)
                padding = np.zeros((raw_b.shape[0], shift_abs), dtype=raw_b.dtype)
                raw_b = np.concatenate([raw_b[:, shift_abs:], padding], axis=1)

        stems_a_raw[stem] = raw_a
        stems_b_raw[stem] = raw_b

        stems_a[stem] = _encode_audio(raw_a)   # [1, C, T_a]
        stems_b[stem] = _encode_audio(raw_b)   # [1, C, T_b]

    if not stems_a:
        if missing_latents:
            raise ValueError(
                f"Stem latents not found for: {', '.join(missing_latents)}. "
                f"These tracks have not been encoded with stem separation — "
                f"encode them first or use regular crossfade (no BPM match).")
        raise ValueError(
            f"Raw stem audio not found for: {', '.join(missing_audio)}. "
            f"Check raw_audio_dir ({_raw_audio_dir}) and stem file paths.")

    fullmix_a = load_latent(npy_fm_a, device=_device, dtype=_dtype)
    fullmix_b = load_latent(npy_fm_b, device=_device, dtype=_dtype)

    # Align all T dimensions to minimum (stretching causes slight length drift)
    all_T = ([z.shape[-1] for z in stems_a.values()] +
             [z.shape[-1] for z in stems_b.values()] +
             [fullmix_a.shape[-1], fullmix_b.shape[-1]])
    target_T = min(all_T)
    for s in list(stems_a):
        stems_a[s] = stems_a[s][..., :target_T]
        stems_b[s] = stems_b[s][..., :target_T]
    fullmix_a = fullmix_a[..., :target_T]
    fullmix_b = fullmix_b[..., :target_T]

    audio = crossfade_stems(
        stems_a, stems_b, fullmix_a, fullmix_b,
        alphas, beta_a, beta_b,
        _autoencoder.decode,
        interp=interp,
        device=_device,
    )

    audio_np = audio.squeeze(0).cpu().float().numpy()   # [2, samples]

    if smart_loop:
        if shared_loop is not None:
            start_s = shared_loop["start_s"]
            tgt_iv = shared_loop["tgt_iv"]
            
            # Recalculate max available bars relative to rendered audio length
            max_audio_s = audio_np.shape[1] / _sample_rate
            max_bars = math.floor((max_audio_s - start_s) / tgt_iv)
            n_bars = min(shared_loop["n_bars"], max_bars)
            
            # Snap to musical phrasing lengths
            if n_bars >= 4:
                n_bars = (n_bars // 4) * 4
            elif n_bars >= 2:
                n_bars = (n_bars // 2) * 2
                
            if n_bars > 0:
                end_s = start_s + n_bars * tgt_iv
                print(f"  smart loop (beatmatch): {n_bars} shared downbeats (bars) [{start_s:.2f}s - {end_s:.2f}s] at {60/tgt_iv:.1f} BPM")
                
                start_sample = int(round(start_s * _sample_rate))
                end_sample = int(round(end_s * _sample_rate))
                
                # Zero-crossing optimization
                mono = (audio_np[0] + audio_np[1]) / 2.0
                window = int(_sample_rate * 0.05)
                start_sample = _nearest_zero_crossing(mono, start_sample, window)
                end_sample = _nearest_zero_crossing(mono, min(end_sample, audio_np.shape[1] - 1), window)
                
                if end_sample > start_sample:
                    audio_np = audio_np[:, start_sample:end_sample]
        else:
            bpm_a = _read_bpm(npy_fm_a)
            bpm_b = _read_bpm(npy_fm_b)
            loop_end = audio_np.shape[1]
            for bpm in [bpm_a, bpm_b]:
                if bpm:
                    try:
                        _, end = _smart_loop_points(audio_np, _sample_rate, bpm)
                        loop_end = min(loop_end, end)
                    except Exception as e:
                        print(f"  smart_loop error (beatmatch, bpm={bpm}): {e}")
            if loop_end < audio_np.shape[1]:
                audio_np = audio_np[:, :loop_end]

    meta = None
    if stems_a_raw and stems_b_raw:
        sum_a = sum(stems_a_raw.values())
        sum_b = sum(stems_b_raw.values())
        
        if smart_loop and 'start_sample' in locals() and 'end_sample' in locals():
            if end_sample > start_sample:
                sum_a = sum_a[:, start_sample:end_sample]
                sum_b = sum_b[:, start_sample:end_sample]
        else:
            if 'loop_end' in locals() and loop_end < audio_np.shape[1]:
                sum_a = sum_a[:, :loop_end]
                sum_b = sum_b[:, :loop_end]
            else:
                T = min(sum_a.shape[1], sum_b.shape[1], audio_np.shape[1])
                sum_a = sum_a[:, :T]
                sum_b = sum_b[:, :T]
                
        CHUNK = 1024
        def build_env(arr):
            if arr.size == 0: return []
            mono = (arr[0] + arr[1]) / 2.0
            r = len(mono) % CHUNK
            if r: mono = np.pad(mono, (0, CHUNK-r))
            return np.round(np.abs(mono.reshape(-1, CHUNK)).max(axis=1), 3).tolist()
            
        env_a = build_env(sum_a)
        env_b = build_env(sum_b)
        
        db_list = []
        if smart_loop and shared_loop is not None:
            n_bars = shared_loop["n_bars"]
            tgt_iv = shared_loop["tgt_iv"]
            for i in range(n_bars + 1):
                db_list.append(round(i * tgt_iv, 3))
                
        meta = {
            "env_a": env_a,
            "env_b": env_b,
            "downbeats": db_list,
            "env_chunk": CHUNK,
            "sr": _sample_rate
        }

    audio_np  = np.clip(audio_np, -1.0, 1.0)
    audio_i16 = (audio_np * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(_sample_rate)
        wf.writeframes(audio_i16.T.flatten().tobytes())
    return buf.getvalue(), list(stems_a.keys()), meta


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode_to_wav(npy_path: Path, smart_loop: bool = False) -> bytes:
    """Decode a latent .npy to WAV bytes. Call with _decode_lock held."""
    latent   = np.load(str(npy_path)).astype(np.float32)   # [64, L]
    latent_t = torch.from_numpy(latent).unsqueeze(0).to(device=_device, dtype=_dtype)

    with torch.no_grad():
        audio = _autoencoder.decode(latent_t)   # [1, 2, samples]

    audio_np = audio.squeeze(0).cpu().float().numpy()   # [2, samples]

    companion = {}
    json_path = npy_path.with_suffix(".json")
    if json_path.exists():
        try:
            companion = json.loads(json_path.read_text())
        except Exception:
            pass

    mask = companion.get("padding_mask", [])
    if mask:
        n_content      = sum(mask)
        actual_samples = n_content * _downsampling_ratio
        if 0 < actual_samples < audio_np.shape[1]:
            audio_np = audio_np[:, :actual_samples]

    if smart_loop:
        raw_bpm = companion.get("bpm")
        if raw_bpm:
            try:
                start, end = _smart_loop_points(audio_np, _sample_rate, float(raw_bpm))
                audio_np = audio_np[:, start:end]
            except Exception as e:
                print(f"  smart_loop error: {e}")

    audio_np  = np.clip(audio_np, -1.0, 1.0)
    audio_i16 = (audio_np * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(_sample_rate)
        wf.writeframes(audio_i16.T.flatten().tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Crossfade decode
# ---------------------------------------------------------------------------

def crossfade_raw_to_wav(
    track_a: str, track_b: str,
    pos_a: float, pos_b: float,
    alphas: dict,
    smart_loop: bool = False,
) -> tuple:
    """Mix raw source stem audio with per-stem alpha weights. No VAE involved."""
    stem_root = _stem_dir if _stem_dir is not None else _latent_dir
    dir_a = stem_root / track_a
    dir_b = stem_root / track_b

    audio_sum  = None
    stems_found = []
    missing    = []

    for stem in STEMS:
        crops_a = find_stem_crops(dir_a, stem)
        crops_b = find_stem_crops(dir_b, stem)
        npy_a, _ = find_best_crop(crops_a, pos_a) if crops_a else (None, None)
        npy_b, _ = find_best_crop(crops_b, pos_b) if crops_b else (None, None)
        if npy_a is None or npy_b is None:
            missing.append(stem)
            continue

        raw_a = find_raw_audio(npy_a, _stem_dir or _latent_dir)
        raw_b = find_raw_audio(npy_b, _stem_dir or _latent_dir)
        if raw_a is None or raw_b is None:
            missing.append(stem)
            continue

        def _load(p):
            a, sr = sf.read(str(p), dtype="float32", always_2d=True)
            a = a.T  # [ch, samples]
            if a.shape[0] == 1:
                a = np.repeat(a, 2, axis=0)
            elif a.shape[0] > 2:
                a = a[:2]
            if sr != _sample_rate:
                import torchaudio
                t = torchaudio.functional.resample(torch.from_numpy(a), sr, _sample_rate)
                a = t.numpy()
            return a

        a_audio = _load(raw_a)
        b_audio = _load(raw_b)

        # Match lengths by zero-padding the shorter one
        n = max(a_audio.shape[1], b_audio.shape[1])
        if a_audio.shape[1] < n:
            a_audio = np.pad(a_audio, ((0,0),(0, n - a_audio.shape[1])))
        if b_audio.shape[1] < n:
            b_audio = np.pad(b_audio, ((0,0),(0, n - b_audio.shape[1])))

        alpha = alphas.get(stem, 0.0)
        blended = (1.0 - alpha) * a_audio + alpha * b_audio

        if audio_sum is None:
            audio_sum = blended
        else:
            # Match lengths again across stems
            n2 = max(audio_sum.shape[1], blended.shape[1])
            if audio_sum.shape[1] < n2:
                audio_sum = np.pad(audio_sum, ((0,0),(0, n2 - audio_sum.shape[1])))
            if blended.shape[1] < n2:
                blended = np.pad(blended, ((0,0),(0, n2 - blended.shape[1])))
            audio_sum = audio_sum + blended

        stems_found.append(stem)

    if audio_sum is None:
        raise ValueError(f"No raw stem audio found. Missing: {', '.join(missing)}")

    if smart_loop:
        # Use BPM from first available npy companion to compute loop points
        for stem in STEMS:
            crops = find_stem_crops(dir_a, stem)
            npy, _ = find_best_crop(crops, pos_a) if crops else (None, None)
            if npy:
                bpm = _read_bpm(npy)
                if bpm:
                    try:
                        _, end = _smart_loop_points(audio_sum, _sample_rate, bpm)
                        audio_sum = audio_sum[:, :end]
                    except Exception as e:
                        print(f"  smart_loop error (raw crossfade): {e}")
                    break

    audio_sum = np.tanh(audio_sum)   # soft clip
    audio_sum = np.clip(audio_sum, -1.0, 1.0)
    audio_i16 = (audio_sum * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(_sample_rate)
        wf.writeframes(audio_i16.T.flatten().tobytes())
    return buf.getvalue(), stems_found


def crossfade_fullmix_to_wav(
    track_a: str, track_b: str,
    pos_a: float, pos_b: float,
    mix: float,
    interp: str = 'slerp',
    smart_loop: bool = False,
) -> bytes:
    """Interpolate two full-mix latents in latent space and decode to WAV.

    Simpler than the stem crossfader: no stem loading, single decoder call.
    mix=0 → track A, mix=1 → track B.
    """
    crops_a = find_crops(_latent_dir / track_a)
    crops_b = find_crops(_latent_dir / track_b)
    npy_a, _ = find_best_crop(crops_a, pos_a)
    npy_b, _ = find_best_crop(crops_b, pos_b)
    if npy_a is None or npy_b is None:
        raise ValueError("Full-mix crops not found for one or both tracks")

    z_a = load_latent(npy_a, device=_device, dtype=_dtype)
    z_b = load_latent(npy_b, device=_device, dtype=_dtype)

    T = min(z_a.shape[-1], z_b.shape[-1])
    z_a = z_a[..., :T]
    z_b = z_b[..., :T]

    interp_fn = slerp if interp == 'slerp' else lerp
    z_mixed = interp_fn(z_a, z_b, mix)

    with torch.no_grad():
        audio = _autoencoder.decode(z_mixed)   # [1, 2, samples]

    audio_np = audio.squeeze(0).cpu().float().numpy()   # [2, samples]

    # Trim to content length using crop A's padding_mask
    companion = {}
    json_path = npy_a.with_suffix('.json')
    if json_path.exists():
        try:
            companion = json.loads(json_path.read_text())
        except Exception:
            pass
    mask = companion.get('padding_mask', [])
    if mask:
        n_content = sum(mask)
        actual_samples = n_content * _downsampling_ratio
        if 0 < actual_samples < audio_np.shape[1]:
            audio_np = audio_np[:, :actual_samples]

    if smart_loop:
        bpm_a = _read_bpm(npy_a)
        bpm_b = _read_bpm(npy_b)
        loop_end = audio_np.shape[1]
        for bpm in [bpm_a, bpm_b]:
            if bpm:
                try:
                    _, end = _smart_loop_points(audio_np, _sample_rate, bpm)
                    loop_end = min(loop_end, end)
                except Exception as e:
                    print(f"  smart_loop error (ab crossfade, bpm={bpm}): {e}")
        if loop_end < audio_np.shape[1]:
            audio_np = audio_np[:, :loop_end]

    audio_np  = np.clip(audio_np, -1.0, 1.0)
    audio_i16 = (audio_np * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(_sample_rate)
        wf.writeframes(audio_i16.T.flatten().tobytes())
    return buf.getvalue()


def latent_mix_to_wav(
    track_a: str, track_b: str,
    pos_a: float, pos_b: float,
    alphas: dict, beta_a: float, beta_b: float,
    interp: str = 'slerp',
    smart_loop: bool = False,
) -> tuple:
    """Mix stem latents in latent space, then decode a single composite latent.

    Unlike crossfade_to_wav (which decodes each stem separately and sums audio),
    this function sums the interpolated stem latents first and decodes once.
    The composite latent is normalized to match the energy of the full-mix reference
    so the decoder sees a latent in its expected distribution.

    Returns (wav_bytes, stems_found).
    """
    stem_root = _stem_dir if _stem_dir is not None else _latent_dir
    dir_a = stem_root / track_a
    dir_b = stem_root / track_b

    stems_a = {}
    stems_b = {}
    missing = []

    for stem in STEMS:
        crops_a = find_stem_crops(dir_a, stem)
        crops_b = find_stem_crops(dir_b, stem)
        npy_a, _ = find_best_crop(crops_a, pos_a) if crops_a else (None, None)
        npy_b, _ = find_best_crop(crops_b, pos_b) if crops_b else (None, None)
        if npy_a is None or npy_b is None:
            missing.append(stem)
            continue
        stems_a[stem] = load_latent(npy_a, device=_device, dtype=_dtype)
        stems_b[stem] = load_latent(npy_b, device=_device, dtype=_dtype)

    if missing:
        raise ValueError(f"Stems not found for: {', '.join(missing)}")

    crops_fm_a = find_crops(_latent_dir / track_a)
    crops_fm_b = find_crops(_latent_dir / track_b)
    npy_fm_a, _ = find_best_crop(crops_fm_a, pos_a)
    npy_fm_b, _ = find_best_crop(crops_fm_b, pos_b)
    if npy_fm_a is None or npy_fm_b is None:
        raise ValueError("Full-mix crops not found for one or both tracks")

    fullmix_a = load_latent(npy_fm_a, device=_device, dtype=_dtype)
    fullmix_b = load_latent(npy_fm_b, device=_device, dtype=_dtype)

    # Align T across all latents
    all_T = ([z.shape[-1] for z in stems_a.values()] +
             [z.shape[-1] for z in stems_b.values()] +
             [fullmix_a.shape[-1], fullmix_b.shape[-1]])
    target_T = min(all_T)
    for s in list(stems_a):
        stems_a[s] = stems_a[s][..., :target_T]
        stems_b[s] = stems_b[s][..., :target_T]
    fullmix_a = fullmix_a[..., :target_T]
    fullmix_b = fullmix_b[..., :target_T]

    interp_fn = slerp if interp == 'slerp' else lerp

    # Interpolate each stem with reality anchors, then sum into a composite latent
    z_composite = None
    stems_found  = []
    for stem in STEMS:
        z_a = stems_a.get(stem)
        z_b = stems_b.get(stem)
        if z_a is None or z_b is None:
            continue
        alpha = alphas.get(stem, 0.5)
        z_target = reality_anchor(z_a, z_b, fullmix_a, fullmix_b,
                                  alpha, beta_a, beta_b, interp_fn=interp_fn)
        z_composite = z_target if z_composite is None else z_composite + z_target
        stems_found.append(stem)

    if z_composite is None:
        raise ValueError("No stem latents found")

    # Normalize composite to full-mix energy so the decoder sees a familiar distribution.
    # The composite is the sum of N stem latents; we rescale it to match the norm of the
    # slerp(fullmix_a, fullmix_b, avg_alpha) reference latent.
    avg_alpha = sum(alphas.get(s, 0.5) for s in stems_found) / len(stems_found)
    z_ref  = interp_fn(fullmix_a, fullmix_b, avg_alpha)
    B      = z_composite.shape[0]
    c_norm = z_composite.reshape(B, -1).norm(dim=1, keepdim=True).clamp(min=1e-8)
    r_norm = z_ref.reshape(B, -1).norm(dim=1, keepdim=True).clamp(min=1e-8)
    scale  = (r_norm / c_norm).view(B, *([1] * (z_composite.dim() - 1)))
    z_composite = z_composite * scale

    with torch.no_grad():
        audio = _autoencoder.decode(z_composite)   # [1, 2, samples]

    audio_np = audio.squeeze(0).cpu().float().numpy()   # [2, samples]

    if smart_loop:
        bpm_a = _read_bpm(npy_fm_a)
        bpm_b = _read_bpm(npy_fm_b)
        loop_end = audio_np.shape[1]
        for bpm in [bpm_a, bpm_b]:
            if bpm:
                try:
                    _, end = _smart_loop_points(audio_np, _sample_rate, bpm)
                    loop_end = min(loop_end, end)
                except Exception as e:
                    print(f"  smart_loop error (latent mix, bpm={bpm}): {e}")
        if loop_end < audio_np.shape[1]:
            audio_np = audio_np[:, :loop_end]

    audio_np  = np.clip(audio_np, -1.0, 1.0)
    audio_i16 = (audio_np * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(_sample_rate)
        wf.writeframes(audio_i16.T.flatten().tobytes())
    return buf.getvalue(), stems_found


def _read_bpm(npy_path: Path) -> float | None:
    """Return BPM from the companion JSON of a latent crop, or None."""
    json_path = npy_path.with_suffix(".json")
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text())
        v = data.get("bpm")
        return float(v) if v else None
    except Exception:
        return None


def crossfade_to_wav(
    track_a: str, track_b: str,
    pos_a: float, pos_b: float,
    alphas: dict, beta_a: float, beta_b: float,
    interp: str = 'slerp',
    smart_loop: bool = False,
) -> tuple:
    """Decode a stem crossfade to WAV bytes. Call with _decode_lock held."""
    stem_root = _stem_dir if _stem_dir is not None else _latent_dir
    dir_a = stem_root / track_a
    dir_b = stem_root / track_b

    stems_a = {}
    stems_b = {}
    missing = []

    for stem in STEMS:
        crops_a = find_stem_crops(dir_a, stem)
        crops_b = find_stem_crops(dir_b, stem)

        npy_a, _ = find_best_crop(crops_a, pos_a) if crops_a else (None, None)
        npy_b, _ = find_best_crop(crops_b, pos_b) if crops_b else (None, None)

        if npy_a is None or npy_b is None:
            missing.append(stem)
            continue

        stems_a[stem] = load_latent(npy_a, device=_device, dtype=_dtype)
        stems_b[stem] = load_latent(npy_b, device=_device, dtype=_dtype)

    if missing:
        raise ValueError(f"Stems not found for: {', '.join(missing)}")

    crops_fm_a = find_crops(_latent_dir / track_a)
    crops_fm_b = find_crops(_latent_dir / track_b)
    npy_fm_a, _ = find_best_crop(crops_fm_a, pos_a)
    npy_fm_b, _ = find_best_crop(crops_fm_b, pos_b)

    if npy_fm_a is None or npy_fm_b is None:
        raise ValueError("Full-mix crops not found for one or both tracks")

    fullmix_a = load_latent(npy_fm_a, device=_device, dtype=_dtype)
    fullmix_b = load_latent(npy_fm_b, device=_device, dtype=_dtype)

    audio = crossfade_stems(
        stems_a, stems_b, fullmix_a, fullmix_b,
        alphas, beta_a, beta_b,
        _autoencoder.decode,
        interp=interp,
        device=_device,
    )

    audio_np = audio.squeeze(0).cpu().float().numpy()   # [2, samples]

    # Smart loop: compute loop points for both tracks, trim to the shorter one
    if smart_loop:
        bpm_a = _read_bpm(npy_fm_a)
        bpm_b = _read_bpm(npy_fm_b)
        loop_end = audio_np.shape[1]
        for bpm in [bpm_a, bpm_b]:
            if bpm:
                try:
                    _, end = _smart_loop_points(audio_np, _sample_rate, bpm)
                    loop_end = min(loop_end, end)
                except Exception as e:
                    print(f"  smart_loop error (crossfade, bpm={bpm}): {e}")
        if loop_end < audio_np.shape[1]:
            print(f"  crossfade smart loop: trimmed to {loop_end} samples "
                  f"({loop_end/_sample_rate:.2f}s)  "
                  f"bpm_a={bpm_a}  bpm_b={bpm_b}")
            audio_np = audio_np[:, :loop_end]

    audio_np  = np.clip(audio_np, -1.0, 1.0)
    audio_i16 = (audio_np * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(_sample_rate)
        wf.writeframes(audio_i16.T.flatten().tobytes())

    return buf.getvalue(), list(stems_a.keys())


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Expose-Headers", "X-Crossfade-Meta, X-Crop-Count, X-Crop-Index, X-Crop-Position, X-Audio-Source, X-Stems-Found")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        qs     = parse_qs(parsed.query)

        if parsed.path == "/status":
            self._handle_status()
        elif parsed.path == "/decode":
            track      = qs.get("track",      [""])[0]
            position   = float(qs.get("position",   ["0.5"])[0])
            smart_loop = qs.get("smart_loop", ["0"])[0] == "1"
            raw        = qs.get("raw",        ["0"])[0] == "1"
            self._handle_decode(track, position, smart_loop, raw)
        elif parsed.path == "/crops":
            track = qs.get("track", [""])[0]
            self._handle_crops(track)
        elif parsed.path == "/crossfade":
            self._handle_crossfade(qs)
        else:
            self.send_response(404)
            self._cors()
            self.end_headers()

    def _handle_status(self):
        body = json.dumps({
            "ok":               True,
            "sample_rate":      _sample_rate,
            "downsampling_ratio": _downsampling_ratio,
            "latent_dir":       str(_latent_dir),
            "device":           str(_device),
        }).encode()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_crops(self, track: str):
        if not track:
            self._error(400, "Missing track parameter")
            return
        track_dir = _latent_dir / track
        if not track_dir.is_dir():
            self._error(404, f"Track not found: {track}")
            return
        crops = find_crops(track_dir)
        body  = json.dumps([{"path": c[0].name, "position": c[1]} for c in crops]).encode()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_decode(self, track: str, position: float,
                       smart_loop: bool = False, raw: bool = False):
        if not track:
            self._error(400, "Missing track parameter")
            return
        track_dir = _latent_dir / track
        if not track_dir.is_dir():
            self._error(404, f"Track not found: {track}")
            return
        crops = find_crops(track_dir)
        if not crops:
            self._error(404, "No crops found for track")
            return

        best_npy, best_pos = find_best_crop(crops, position)
        best_idx = next((i for i, (p, _) in enumerate(crops) if p == best_npy), 0)
        print(f"  decode: {track}  pos={position:.3f}  smart_loop={smart_loop}"
              f"  raw={raw}  → {best_npy.name} [{best_idx+1}/{len(crops)}] (pos={best_pos:.3f})")

        try:
            if raw:
                audio_path = find_raw_audio(best_npy, _latent_dir)
                if audio_path is None:
                    self._error(404, f"Raw audio not found for {best_npy.name} "
                                     f"(raw_audio_dir not set or file missing)")
                    return
                wav_bytes = load_raw_wav(audio_path, smart_loop=smart_loop)
            else:
                with _decode_lock:
                    wav_bytes = decode_to_wav(best_npy, smart_loop=smart_loop)
        except Exception as e:
            self._error(500, str(e))
            return

        self.send_response(200)
        self._cors()
        self.send_header("Content-Type",    "audio/wav")
        self.send_header("Content-Length",  str(len(wav_bytes)))
        self.send_header("X-Crop-Count",    str(len(crops)))
        self.send_header("X-Crop-Index",    str(best_idx + 1))
        self.send_header("X-Crop-Position", f"{best_pos:.4f}")
        self.send_header("X-Audio-Source",  "raw" if raw else "vae")
        self.end_headers()
        self.wfile.write(wav_bytes)

    def _handle_crossfade(self, qs: dict):
        track_a = qs.get("track_a", [""])[0]
        track_b = qs.get("track_b", [""])[0]
        if not track_a or not track_b:
            self._error(400, "Missing track_a or track_b parameter")
            return
        stem_root = _stem_dir if _stem_dir is not None else _latent_dir
        dir_a = stem_root / track_a
        dir_b = stem_root / track_b
        if not dir_a.is_dir():
            self._error(404, f"Stem track not found: {track_a}")
            return
        if not dir_b.is_dir():
            self._error(404, f"Stem track not found: {track_b}")
            return

        mode       = qs.get("mode", ["stems"])[0]   # 'stems' | 'ab' | 'latent'
        pos_a      = float(qs.get("pos_a",  ["0.5"])[0])
        pos_b      = float(qs.get("pos_b",  ["0.5"])[0])
        beta_a     = float(qs.get("beta_a", ["0.0"])[0])
        beta_b     = float(qs.get("beta_b", ["0.0"])[0])
        interp     = qs.get("interp", ["slerp"])[0]
        smart_loop = qs.get("smart_loop", ["0"])[0] == "1"
        raw        = qs.get("raw",        ["0"])[0] == "1"
        beatmatch  = qs.get("beatmatch",  ["0"])[0] == "1"
        shift_a    = float(qs.get("shift_a",   ["0.0"])[0])
        stretch_a  = float(qs.get("stretch_a", ["1.0"])[0])
        shift_b    = float(qs.get("shift_b",   ["0.0"])[0])
        stretch_b  = float(qs.get("stretch_b", ["1.0"])[0])
        bm_algo    = qs.get("bm_algo", ["pedalboard"])[0]
        mix        = float(qs.get("mix", ["0.5"])[0])   # for mode=ab
        if bm_algo not in ("pedalboard", "bungee", "rubberband"):
            bm_algo = "pedalboard"
        if interp not in ("slerp", "lerp"):
            interp = "slerp"
        if mode not in ("stems", "ab", "latent"):
            mode = "stems"

        alphas = {
            "drums":  float(qs.get("drums",  ["0.0"])[0]),
            "bass":   float(qs.get("bass",   ["0.0"])[0]),
            "other":  float(qs.get("other",  ["0.0"])[0]),
            "vocals": float(qs.get("vocals", ["0.0"])[0]),
        }

        print(f"  crossfade[{mode}]: A={track_a} B={track_b} "
              f"pos_a={pos_a:.3f} pos_b={pos_b:.3f} "
              + (f"mix={mix:.3f}" if mode == "ab" else
                 f"alphas={alphas} beta_a={beta_a:.3f} beta_b={beta_b:.3f}")
              + f" interp={interp} smart_loop={smart_loop} raw={raw}"
              + (f" beatmatch shift_a={shift_a:+.1f}st ×{stretch_a:.3f} "
                 f"shift_b={shift_b:+.1f}st ×{stretch_b:.3f}" if beatmatch else ""))

        try:
            if mode == "ab":
                # Plain full-mix A↔B interpolation — single decode
                if raw:
                    # Blend raw full-mix audio files in audio space
                    npy_a, _ = find_best_crop(find_crops(_latent_dir / track_a), pos_a)
                    npy_b, _ = find_best_crop(find_crops(_latent_dir / track_b), pos_b)
                    if npy_a is None or npy_b is None:
                        self._error(404, "Full-mix crops not found")
                        return
                    raw_a = find_raw_audio(npy_a, _latent_dir)
                    raw_b = find_raw_audio(npy_b, _latent_dir)
                    if raw_a is None or raw_b is None:
                        self._error(404, "Raw audio files not found — check raw_audio_dir")
                        return
                    def _load_raw(p):
                        a, sr = sf.read(str(p), dtype="float32", always_2d=True)
                        a = a.T
                        if a.shape[0] == 1: a = np.repeat(a, 2, axis=0)
                        if sr != _sample_rate:
                            import torchaudio
                            a = torchaudio.functional.resample(
                                torch.from_numpy(a), sr, _sample_rate).numpy()
                        return a
                    a_np = _load_raw(raw_a)
                    b_np = _load_raw(raw_b)
                    T = min(a_np.shape[1], b_np.shape[1])
                    audio_np = (1.0 - mix) * a_np[:, :T] + mix * b_np[:, :T]
                    audio_np  = np.clip(audio_np, -1.0, 1.0)
                    audio_i16 = (audio_np * 32767).astype(np.int16)
                    buf = io.BytesIO()
                    with wave.open(buf, 'wb') as wf:
                        wf.setnchannels(2); wf.setsampwidth(2)
                        wf.setframerate(_sample_rate)
                        wf.writeframes(audio_i16.T.flatten().tobytes())
                    wav_bytes = buf.getvalue()
                else:
                    with _decode_lock:
                        wav_bytes = crossfade_fullmix_to_wav(
                            track_a, track_b, pos_a, pos_b,
                            mix, interp=interp, smart_loop=smart_loop,
                        )
                stems_found = []
                source_label = "raw" if raw else "vae"
            elif mode == "latent":
                # Stem latents mixed in latent space → single decode
                with _decode_lock:
                    wav_bytes, stems_found = latent_mix_to_wav(
                        track_a, track_b, pos_a, pos_b,
                        alphas, beta_a, beta_b, interp=interp,
                        smart_loop=smart_loop,
                    )
                source_label = "vae-latent"
            elif beatmatch:
                with _decode_lock:
                    ret = beatmatch_crossfade_to_wav(
                        track_a, track_b, pos_a, pos_b,
                        alphas, beta_a, beta_b,
                        shift_a, stretch_a, shift_b, stretch_b,
                        interp=interp, smart_loop=smart_loop, algo=bm_algo,
                    )
                    if len(ret) == 3:
                        wav_bytes, stems_found, meta = ret
                    else:
                        wav_bytes, stems_found = ret
                        meta = None
                source_label = "beatmatch"
            elif raw:
                wav_bytes, stems_found = crossfade_raw_to_wav(
                    track_a, track_b, pos_a, pos_b, alphas, smart_loop=smart_loop,
                )
                source_label = "raw"
            else:
                with _decode_lock:
                    wav_bytes, stems_found = crossfade_to_wav(
                        track_a, track_b, pos_a, pos_b,
                        alphas, beta_a, beta_b, interp=interp,
                        smart_loop=smart_loop,
                    )
                source_label = "vae"
        except (ValueError, Exception) as e:
            self._error(500, str(e))
            return

        self.send_response(200)
        self._cors()
        self.send_header("Content-Type",   "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        if stems_found:
            self.send_header("X-Stems-Found",  ",".join(stems_found))
        if 'meta' in locals() and meta:
            self.send_header("X-Crossfade-Meta", json.dumps(meta))
        self.send_header("X-Audio-Source", source_label)
        self.end_headers()
        self.wfile.write(wav_bytes)

    def _error(self, code: int, msg: str):
        body = msg.encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type",   "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"  [{self.address_string()}] {fmt % args}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _autoencoder, _sample_rate, _downsampling_ratio
    global _latent_dir, _stem_dir, _raw_audio_dir, _dtype, _device

    parser = argparse.ArgumentParser(description="Latent audio decode server")
    parser.add_argument("--config",       default=str(DEFAULT_INI),
                        help=f"Path to .ini config file (default: {DEFAULT_INI})")
    parser.add_argument("--latent-dir",   default=None,
                        help="Root of the pre-encoded full-mix dataset (overrides ini)")
    parser.add_argument("--stem-dir",     default=None,
                        help="Root of the pre-encoded stem dataset (overrides ini)")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--ckpt-path",    default=None)
    parser.add_argument("--port",         type=int, default=None)
    parser.add_argument("--model-half",    action="store_true", default=None)
    parser.add_argument("--no-model-half", dest="model_half", action="store_false")
    parser.add_argument("--device",       default=None)
    args = parser.parse_args()

    # Read ini file (CLI args override)
    ini = configparser.ConfigParser()
    ini_path = Path(args.config)
    if ini_path.exists():
        ini.read(str(ini_path))
        print(f"Config: {ini_path}")
    elif args.config != str(DEFAULT_INI):
        print(f"Error: config file not found: {ini_path}", file=sys.stderr)
        sys.exit(1)

    def ini_get(section, key, fallback=None):
        return ini.get(section, key, fallback=fallback)

    # sao_dir is used only as an interpolation base for model paths in the ini.
    # It is expanded automatically by configparser's %(sao_dir)s syntax.
    # If model_config / ckpt_path are not set in the ini, fall back to sao_dir-relative paths.
    sao_dir_str      = ini_get("model", "sao_dir")
    latent_dir_str   = args.latent_dir   or ini_get("server", "latent_dir")
    stem_dir_str     = args.stem_dir     or ini_get("server", "stem_dir")
    raw_audio_dir_str = ini_get("server", "raw_audio_dir")
    port             = args.port         or int(ini_get("server", "port", fallback="7891"))
    device           = args.device       or ini_get("model",  "device", fallback="cuda")

    # Resolve model paths: CLI → ini (with %(sao_dir)s expansion) → sao_dir fallback
    if args.model_config:
        model_config_str = args.model_config
    else:
        model_config_str = ini_get("model", "model_config")
        if not model_config_str and sao_dir_str:
            model_config_str = str(Path(sao_dir_str) /
                                   "models/checkpoints/small/base_model_config.json")

    if args.ckpt_path:
        ckpt_path_str = args.ckpt_path
    else:
        ckpt_path_str = ini_get("model", "ckpt_path")
        if not ckpt_path_str and sao_dir_str:
            ckpt_path_str = str(Path(sao_dir_str) /
                                "models/checkpoints/small/base_model.ckpt")

    if args.model_half is None:
        model_half = ini_get("model", "model_half", fallback="true").lower() != "false"
    else:
        model_half = args.model_half

    if not latent_dir_str:
        print("Error: latent_dir not set (use --latent-dir or set in latent_player.ini)",
              file=sys.stderr)
        sys.exit(1)

    if not model_config_str or not ckpt_path_str:
        print("Error: model_config and ckpt_path must be set in latent_player.ini "
              "or via --model-config / --ckpt-path", file=sys.stderr)
        sys.exit(1)

    _latent_dir       = Path(latent_dir_str)
    _stem_dir         = Path(stem_dir_str) if stem_dir_str else None
    _raw_audio_dir    = Path(raw_audio_dir_str) if raw_audio_dir_str else None
    model_config_path = Path(model_config_str)
    ckpt_path         = Path(ckpt_path_str)

    for p, label in [(_latent_dir, "latent-dir"),
                     (model_config_path, "model-config"),
                     (ckpt_path, "ckpt-path")]:
        if not p.exists():
            print(f"Error: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    _device = device
    if _device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        _device = "cpu"

    _autoencoder, _sample_rate, _, _downsampling_ratio = load_autoencoder(
        model_config_path, ckpt_path, model_half, _device
    )
    _dtype = torch.float16 if model_half else torch.float32

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\nLatent server running at http://localhost:{port}")
    print(f"  Latent dir : {_latent_dir}")
    if _stem_dir:
        print(f"  Stem dir   : {_stem_dir}")
    if _raw_audio_dir:
        print(f"  Raw audio  : {_raw_audio_dir}")
    else:
        print(f"  Raw audio  : not configured (raw=1 will return 404)")
    print(f"  Device     : {_device}  half={model_half}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
