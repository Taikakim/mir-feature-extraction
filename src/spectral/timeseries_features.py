"""
Time-Series Feature Extraction for MIR Project

Extracts per-step features at configurable temporal resolution (n_steps),
aligned to the SAO latent sequence length (default 256 = 524288 / 2048).

Features output (all stored as lists of floats or list-of-lists for HPCP):

  Multiband RMS (4 bands, parallel):
    rms_energy_bass_ts, rms_energy_body_ts, rms_energy_mid_ts, rms_energy_air_ts

  Spectral (STFT-based, binned):
    spectral_flatness_ts, spectral_flux_ts, spectral_skewness_ts, spectral_kurtosis_ts

  Rhythmic:
    beat_activations_ts       -- Gaussian-smoothed beat probability per step
    downbeat_activations_ts   -- Same for downbeats
    onsets_activations_ts     -- Onset strength envelope, L∞-normalised

  Harmonic (Essentia, optional):
    hpcp_ts                   -- n_steps × 12 nested list (L∞-normalised per step)
    tonic_ts                  -- Pitch class [0–11] per step
    tonic_strength_ts         -- Key confidence [0–1] per step

  Timbral (timbral_models, optional, lower resolution):
    brightness_ts, roughness_ts, hardness_ts, depth_ts, reverb_ts
    Stored at *timbral_n_steps* (default 16, each ~750 ms at 44.1 kHz).
    WARNING: timbral_reverb can hang — wrap callers with a timeout if needed.

Naming convention: all time-series keys end with _ts.  Array length is
implicit (len(info['rms_energy_bass_ts']) == n_steps).

Dependencies:
  Required : librosa, numpy, scipy
  Optional : essentia (HPCP/tonic), timbral_models (brightness etc.)
"""

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.common import FREQUENCY_BANDS
from spectral.multiband_rms import calculate_rms_db, create_bandpass_filter

logger = logging.getLogger(__name__)

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False

try:
    import timbral_models
    TIMBRAL_AVAILABLE = True
except ImportError:
    TIMBRAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bin_frames(frames: np.ndarray, n_steps: int) -> List[float]:
    """Bin a 1-D frame array to n_steps by taking the median of each bin."""
    n_frames = len(frames)
    result: List[float] = []
    for i in range(n_steps):
        start = i * n_frames // n_steps
        end   = (i + 1) * n_frames // n_steps
        end   = max(end, start + 1)
        chunk = frames[start: min(end, n_frames)]
        result.append(float(np.median(chunk)) if len(chunk) > 0 else 0.0)
    return result


def _compute_rms_band_ts(audio: np.ndarray, sos: np.ndarray, n_steps: int) -> List[float]:
    """Bandpass-filter, then compute per-step RMS (dB). Designed for threads."""
    from scipy.signal import sosfilt
    filtered   = sosfilt(sos, audio)
    step_len   = max(1, len(filtered) // n_steps)
    result: List[float] = []
    for i in range(n_steps):
        chunk = filtered[i * step_len: (i + 1) * step_len]
        result.append(calculate_rms_db(chunk) if len(chunk) > 0 else -60.0)
    return result


def _compute_multiband_rms_ts(audio: np.ndarray, sr: int,
                               n_steps: int) -> Dict[str, List[float]]:
    """Compute all 4 band RMS time-series in parallel threads."""
    # Pre-compute filters before spawning threads
    filters = {
        band: create_bandpass_filter(low, high, sr)
        for band, (low, high) in FREQUENCY_BANDS.items()
    }

    results: Dict[str, List[float]] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            band: pool.submit(_compute_rms_band_ts, audio, sos, n_steps)
            for band, sos in filters.items()
        }
        for band, future in futures.items():
            results[f'rms_energy_{band}_ts'] = future.result()
    return results


def _compute_spectral_ts(audio: np.ndarray, sr: int, n_steps: int,
                          n_fft: int = 2048,
                          hop_length: int = 512) -> Dict[str, List[float]]:
    """
    Compute spectral time-series by binning STFT frames.

    Uses a single STFT pass for all four features (flatness, flux, skewness, kurtosis).
    """
    import librosa

    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)) + 1e-10
    n_frames = S.shape[1]

    # --- Flatness: geometric / arithmetic mean of spectrum per frame ---
    log_S = np.log(S)
    geo_mean  = np.exp(log_S.mean(axis=0))
    arith_mean = S.mean(axis=0)
    flatness_frames = geo_mean / (arith_mean + 1e-10)

    # --- Flux: L2 norm of frame-to-frame difference ---
    diff = np.diff(S, axis=1)
    flux_frames = np.concatenate([[0.0], np.sqrt((diff ** 2).sum(axis=0))])

    # --- Spectral moments (skewness, kurtosis) ---
    freqs  = np.arange(S.shape[0], dtype=np.float32)
    norm_S = S / (S.sum(axis=0, keepdims=True) + 1e-10)
    mean_f = (freqs[:, None] * norm_S).sum(axis=0)                                # (n_frames,)
    dev    = freqs[:, None] - mean_f[None, :]                                      # (bins, n_frames)
    var    = (dev ** 2 * norm_S).sum(axis=0)
    std    = np.sqrt(var + 1e-10)
    skewness_frames = ((dev / std[None, :]) ** 3 * norm_S).sum(axis=0)
    kurtosis_frames = ((dev / std[None, :]) ** 4 * norm_S).sum(axis=0)

    return {
        'spectral_flatness_ts':  _bin_frames(flatness_frames,  n_steps),
        'spectral_flux_ts':      _bin_frames(flux_frames,      n_steps),
        'spectral_skewness_ts':  _bin_frames(skewness_frames,  n_steps),
        'spectral_kurtosis_ts':  _bin_frames(kurtosis_frames,  n_steps),
    }


def _compute_activation_ts(timestamps: np.ndarray, audio_duration: float,
                            n_steps: int, sigma: float = 1.5) -> List[float]:
    """
    Build a Gaussian-smoothed activation array from event timestamps.

    Each timestamp contributes a Gaussian centred at its nearest step index.
    The result is L∞-normalised to [0, 1].
    """
    activation = np.zeros(n_steps, dtype=np.float64)

    if len(timestamps) == 0:
        return activation.tolist()

    step_duration = audio_duration / n_steps
    steps = np.arange(n_steps, dtype=np.float64)

    for t in timestamps:
        if 0.0 <= t <= audio_duration:
            centre = t / step_duration
            activation += np.exp(-0.5 * ((steps - centre) / sigma) ** 2)

    max_val = activation.max()
    if max_val > 0:
        activation /= max_val

    return activation.tolist()


def _compute_onset_ts(audio: np.ndarray, sr: int, n_steps: int) -> List[float]:
    """
    Compute onset strength envelope, bin to n_steps, L∞-normalise to [0, 1].
    """
    import librosa

    hop_length = 512
    onset_env  = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    binned     = _bin_frames(onset_env, n_steps)

    max_val = max(binned) if binned else 0.0
    if max_val > 0:
        binned = [v / max_val for v in binned]
    return binned


def _compute_hpcp_ts(audio: np.ndarray, sr: int,
                      n_steps: int) -> Dict[str, List]:
    """
    Compute HPCP, tonic, and tonic_strength time-series.

    HPCP is computed at a hop size that aligns with n_steps so that each
    output step corresponds to approximately one HPCP frame.  For SAO crops
    (524288 samples, 256 steps): hop_size = 2048 → exactly 256 frames.

    Returns:
        hpcp_ts          : list[n_steps] of list[12]   (L∞-normalised per step)
        tonic_ts         : list[n_steps] of float       (pitch class 0–11)
        tonic_strength_ts: list[n_steps] of float       ([0, 1])
    """
    n_samples = len(audio)
    # Snap hop_size to the nearest power-of-2 that gives ≥ n_steps frames
    hop_size   = max(256, n_samples // n_steps)
    frame_size = min(8192, max(4096, hop_size * 2))

    # Build Essentia pipeline once
    frame_gen  = es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size,
                                    startFromZero=True)
    window     = es.Windowing(type='hann', size=frame_size)
    spectrum   = es.Spectrum(size=frame_size)
    peaks_algo = es.SpectralPeaks(maxPeaks=100, magnitudeThreshold=1e-5,
                                   sampleRate=sr, orderBy='magnitude')
    hpcp_algo  = es.HPCP(size=12, harmonics=8, minFrequency=40.0,
                          maxFrequency=5000.0, weightType='cosine',
                          nonLinear=True, normalized='unitMax', sampleRate=sr)

    raw_frames: List[np.ndarray] = []
    for frame in frame_gen:
        windowed = window(frame)
        spec     = spectrum(windowed)
        freqs, mags = peaks_algo(spec)
        hpcp_vec = hpcp_algo(freqs, mags)
        raw_frames.append(np.array(hpcp_vec, dtype=np.float32))

    n_frames = len(raw_frames)
    if n_frames == 0:
        zero_hpcp = [[0.0] * 12] * n_steps
        return {'hpcp_ts': zero_hpcp, 'tonic_ts': [0.0] * n_steps,
                'tonic_strength_ts': [0.0] * n_steps}

    frames_arr = np.stack(raw_frames, axis=0)  # (n_frames, 12)

    # Initialise Key algorithm once, reuse for all steps
    key_algo: Optional[es.Key] = None
    for profile in ('edmm', 'edma'):
        try:
            key_algo = es.Key(profileType=profile)
            break
        except ValueError:
            continue

    hpcp_ts: List[List[float]] = []
    tonic_ts: List[float]      = []
    strength_ts: List[float]   = []

    for i in range(n_steps):
        start = i * n_frames // n_steps
        end   = (i + 1) * n_frames // n_steps
        end   = max(end, start + 1)
        step_frames = frames_arr[start: min(end, n_frames)]

        hpcp_avg = step_frames.mean(axis=0)                # (12,)
        linf = float(hpcp_avg.max())
        if linf > 0:
            hpcp_avg = hpcp_avg / linf

        hpcp_ts.append(hpcp_avg.tolist())

        if key_algo is not None and hpcp_avg.sum() > 0:
            try:
                key_str, _, strength, _ = key_algo(hpcp_avg.tolist())
                tonic_int = PITCH_CLASSES.index(key_str) if key_str in PITCH_CLASSES else 0
                tonic_ts.append(float(tonic_int))
                strength_ts.append(float(strength))
            except Exception:
                tonic_ts.append(0.0)
                strength_ts.append(0.0)
        else:
            tonic_ts.append(0.0)
            strength_ts.append(0.0)

    return {
        'hpcp_ts':           hpcp_ts,
        'tonic_ts':          tonic_ts,
        'tonic_strength_ts': strength_ts,
    }


def _compute_timbral_ts(
    audio: np.ndarray,
    sr: int,
    features: List[str],
    timbral_n_steps: int,
) -> Dict[str, List[float]]:
    """
    Compute timbral time-series at *timbral_n_steps* resolution.

    Each chunk is written to /dev/shm (RAM disk) to avoid HDD I/O;
    timbral_models requires a file path.

    Note: timbral_reverb can hang on pathological audio.  The caller is
    responsible for applying a timeout (cf. pipeline_workers.py cf_wait).
    """
    import soundfile as sf_

    n_samples  = len(audio)
    step_len   = n_samples // timbral_n_steps
    pid        = os.getpid()
    results    = {f'{feat}_ts': [] for feat in features}
    valid_feats = [f for f in features if hasattr(timbral_models, f'timbral_{f}')]

    for i in range(timbral_n_steps):
        chunk     = audio[i * step_len: (i + 1) * step_len]
        tmp_path  = Path(f'/dev/shm/mir_ts_{pid}_{i}.wav')
        try:
            chunk_i16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
            sf_.write(str(tmp_path), chunk_i16, sr, subtype='PCM_16')

            for feat in valid_feats:
                try:
                    if feat == 'reverb':
                        _, prob = timbral_models.timbral_reverb(str(tmp_path),
                                                                  dev_output=True)
                        results[f'{feat}_ts'].append(float(prob) * 100.0)
                    else:
                        fn  = getattr(timbral_models, f'timbral_{feat}')
                        results[f'{feat}_ts'].append(float(fn(str(tmp_path))))
                except Exception as e:
                    logger.debug(f"Timbral {feat} step {i} failed: {e}")
                    results[f'{feat}_ts'].append(0.0)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_timeseries(
    audio: np.ndarray,
    sr: int,
    n_steps: int = 256,
    beats: Optional[np.ndarray] = None,
    downbeats: Optional[np.ndarray] = None,
    beat_sigma: float = 1.5,
    extract_hpcp: bool = True,
    extract_timbral: bool = False,
    timbral_n_steps: int = 16,
    timbral_features: Optional[List[str]] = None,
    _return_timings: bool = False,
) -> Dict:
    """
    Extract all time-series features for a pre-loaded mono audio array.

    Args:
        audio:           Mono float32 audio signal.
        sr:              Sample rate in Hz.
        n_steps:         Number of output time steps (default 256 = SAO latent length).
        beats:           Beat timestamps in seconds.  Auto-detected if None.
        downbeats:       Downbeat timestamps.  Heuristic (beats[::4]) if None.
        beat_sigma:      Gaussian smoothing width in steps for beat activations.
        extract_hpcp:    Whether to compute HPCP + tonic time-series (Essentia).
        extract_timbral: Whether to compute timbral time-series (slow, timbral_models).
        timbral_n_steps: Temporal resolution for timbral features (default 16).
        timbral_features: Which timbral features to extract.  Defaults to
                          ['brightness', 'roughness', 'hardness', 'depth', 'reverb'].

    Returns:
        Dict with time-series arrays (lists) keyed by feature name + '_ts'.
        HPCP is stored as a nested list: hpcp_ts[step] = [12 floats].
    """
    import librosa
    import time as _time

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio_duration = len(audio) / sr
    results: Dict = {}
    _bench: Dict[str, float] = {}

    def _t(key, fn, *a, **kw):
        t0 = _time.perf_counter()
        r = fn(*a, **kw)
        if _return_timings:
            _bench[key] = _time.perf_counter() - t0
        return r

    # ---- 1. Multiband RMS (4 bands, parallel threads) -----------------------
    results.update(_t("ts.rms_bands", _compute_multiband_rms_ts, audio, sr, n_steps))

    # ---- 2. Spectral (single STFT, all 4 features) --------------------------
    results.update(_t("ts.spectral", _compute_spectral_ts, audio, sr, n_steps))

    # ---- 3. Beat activations ------------------------------------------------
    _beat_preloaded = beats is not None
    if beats is None:
        beats = _t("ts.beat_track", lambda: (
            librosa.frames_to_time(
                librosa.beat.beat_track(y=audio, sr=sr)[1], sr=sr
            )
        ))
    elif _return_timings:
        _bench["ts.beat_track"] = 0.0  # skipped — preloaded from BEATS_GRID

    t0 = _time.perf_counter()
    results['beat_activations_ts'] = _compute_activation_ts(
        beats, audio_duration, n_steps, beat_sigma)
    if downbeats is None:
        downbeats = beats[::4] if len(beats) >= 4 else beats
    results['downbeat_activations_ts'] = _compute_activation_ts(
        downbeats, audio_duration, n_steps, beat_sigma * 1.5)
    if _return_timings:
        _bench["ts.beat_act"] = _time.perf_counter() - t0

    # ---- 4. Onset strength --------------------------------------------------
    results['onsets_activations_ts'] = _t("ts.onset", _compute_onset_ts, audio, sr, n_steps)

    # ---- 5. HPCP + tonic (Essentia) -----------------------------------------
    if extract_hpcp:
        if ESSENTIA_AVAILABLE:
            results.update(_t("ts.hpcp", _compute_hpcp_ts, audio, sr, n_steps))
        else:
            logger.warning("Essentia not available; skipping HPCP time-series")

    # ---- 6. Timbral (optional, lower resolution) ----------------------------
    if extract_timbral:
        if TIMBRAL_AVAILABLE:
            if timbral_features is None:
                timbral_features = ['brightness', 'roughness', 'hardness', 'depth', 'reverb']
            results.update(_t("ts.timbral", _compute_timbral_ts, audio, sr,
                              timbral_features, timbral_n_steps))
        else:
            logger.warning("timbral_models not available; skipping timbral time-series")

    if _return_timings:
        return results, _bench
    return results


def extract_timeseries_from_file(
    audio_path: Path,
    n_steps: int = 256,
    beats: Optional[np.ndarray] = None,
    downbeats: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict:
    """
    Convenience wrapper: load audio from file, then call extract_timeseries.

    Uses core.file_utils.read_audio (handles all formats including m4a/AAC).
    """
    from core.file_utils import read_audio

    audio, sr = read_audio(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return extract_timeseries(audio.astype(np.float32), sr,
                               n_steps=n_steps, beats=beats,
                               downbeats=downbeats, **kwargs)
