# plots/latent_analysis/_temporal_features.py
"""Frame-level feature extraction aligned to latent hop size."""
import numpy as np
import librosa
from scipy.signal import butter, sosfilt

from plots.latent_analysis.config import (
    SAMPLE_RATE, HOP_LENGTH, N_FFT, LATENT_FRAMES,
    TEMPORAL_FEATURE_NAMES,
)

# Band-pass filter boundaries (Hz)
_BANDS = {
    "bass": (20,   250),
    "body": (250,  2000),
    "mid":  (2000, 8000),
    "air":  (8000, 20000),
}


def _bandpass_rms(audio: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    """Frame-level RMS within a frequency band. Returns [LATENT_FRAMES].
    Uses librosa.feature.rms with center=True to match STFT alignment (257 frames → slice to 256).
    """
    nyq  = sr / 2
    sos  = butter(4, [low/nyq, min(high/nyq, 0.999)], btype="band", output="sos")
    filtered = sosfilt(sos, audio.astype(np.float64)).astype(np.float32)
    rms = librosa.feature.rms(y=filtered, frame_length=N_FFT,
                               hop_length=HOP_LENGTH, center=True)[0]
    return rms[:LATENT_FRAMES]


def compute_frame_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Compute frame-level features for a single crop.
    audio: 1D float32, length exactly LATENT_FRAMES * HOP_LENGTH.
    Returns: [N_temporal_features, LATENT_FRAMES] float32.
    n_fft=4096, hop=2048, center=True → 257 frames; slice [:256].
    50% overlap Hann window preserves transients at frame boundaries.
    """
    audio = audio.astype(np.float32)
    expected_len = LATENT_FRAMES * HOP_LENGTH
    if len(audio) != expected_len:
        raise ValueError(f"Expected {expected_len} samples, got {len(audio)}")

    # STFT (n_fft=4096, hop=2048, center=True → 257 frames; slice to 256)
    # 50% overlap with Hann window preserves transients at frame boundaries.
    # center=True pads by n_fft//2 each side → 1 + floor(524288/2048) = 257 frames.
    stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                center=True))[:, :LATENT_FRAMES]  # [n_fft//2+1, 256]
    assert stft.shape[1] == LATENT_FRAMES, f"STFT gave {stft.shape[1]} frames, expected {LATENT_FRAMES}"

    rows = []

    # Broadband RMS (center=True to match STFT alignment; slice to 256)
    rms_broad = librosa.feature.rms(y=audio, frame_length=N_FFT,
                                     hop_length=HOP_LENGTH, center=True)[0, :LATENT_FRAMES]
    rows.append(rms_broad)

    # Band RMS (4 bands)
    for band_name in ["bass", "body", "mid", "air"]:
        low, high = _BANDS[band_name]
        rows.append(_bandpass_rms(audio, sr, low, high))

    # Spectral features from STFT
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    power = stft ** 2

    # Flatness
    rows.append(librosa.feature.spectral_flatness(S=stft)[0][:LATENT_FRAMES])

    # Flux
    flux = np.sqrt(np.sum(np.diff(stft, axis=1, prepend=stft[:, :1]) ** 2, axis=0))
    rows.append(flux[:LATENT_FRAMES])

    # Centroid (normalised to 0-1)
    centroid = librosa.feature.spectral_centroid(S=power, sr=sr)[0]
    rows.append((centroid[:LATENT_FRAMES] / (sr / 2)).astype(np.float32))

    # Skewness + Kurtosis (spectral moment proxies)
    p_norm = power / (power.sum(axis=0, keepdims=True) + 1e-9)
    f_col  = freqs[:, np.newaxis] / (sr / 2)
    mean_f = (f_col * p_norm).sum(axis=0)
    var_f  = ((f_col - mean_f[np.newaxis]) ** 2 * p_norm).sum(axis=0)
    std_f  = np.sqrt(var_f + 1e-12)
    skew_f = ((f_col - mean_f[np.newaxis]) ** 3 * p_norm).sum(axis=0) / (std_f ** 3 + 1e-12)
    kurt_f = ((f_col - mean_f[np.newaxis]) ** 4 * p_norm).sum(axis=0) / (std_f ** 4 + 1e-12)
    rows.append(skew_f[:LATENT_FRAMES].astype(np.float32))
    rows.append(kurt_f[:LATENT_FRAMES].astype(np.float32))

    # Chroma CQT (12 bins) — produces extra frames, slice to 256
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=HOP_LENGTH,
                                         n_chroma=12)[:, :LATENT_FRAMES]  # [12, 256]
    for ci in range(12):
        rows.append(chroma[ci])

    # Onset strength (center=True, slice to 256 — consistent with STFT)
    onset = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=HOP_LENGTH,
                                          n_fft=N_FFT, center=True)
    rows.append(onset[:LATENT_FRAMES])

    out = np.array([np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
                    for r in rows], dtype=np.float32)
    assert out.shape == (len(TEMPORAL_FEATURE_NAMES), LATENT_FRAMES)
    return out
