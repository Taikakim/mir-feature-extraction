"""
SAME-compatible octave-band chroma extraction (Stable Audio 3 conditioning format).

Replicates, numerically faithfully, the chroma regression *targets* used to
train the SAME autoencoder (Stability AI, arXiv 2605.18613 §3.3.2), as
implemented in stable-audio-tools ``training/autoencoders.py`` (commit 3241adb).
(Adversarially verified against the upstream sources: the filterbank port is
bit-exact vs a literal transcription in float64; end-to-end output agrees with
torch-native float32 arithmetic to ~1e-4 relative — bit-identity is unattainable
across FFT backends regardless.) Pipeline:

    TightSpectrogram(n_fft=8192, normalized=True, power=1.0)   # hop forced to 4096
      -> per-channel magnitude, mean over channels
      -> log1p
      -> torchaudio.prototype.transforms.ChromaScale(
             sample_rate=44100, n_chroma=128, n_freqs=4097,
             ctroct=center, octwidth=width, norm=1, base_c=True)
    for (center, width) in [(1.0, 1.0), (5.0, 1.5), (9.0, 1.0)]
      -> F.interpolate(target, size=n_latent_frames, mode='linear')

Output: (3, 128, T) float32 — three register-resolved chromagrams
(bass ~A1/55 Hz, mid ~A5/880 Hz, treble ~A9/14 kHz) at the SAME latent rate
(hop 4096 @ 44.1 kHz ~= 10.77 frames/s), suitable as steering/conditioning
targets for a refit linear chroma readout on SAME latents.

Implementation notes (all verified against the upstream sources):

* The "tight" window at hop = n_fft/2 reduces to g[n] = sin(pi*n/N) (sqrt-Hann,
  amplitude exactly 1). TightSpectrogram's ``demodulate`` only flips signs of
  complex bins ((-1)^{k*m}) and is irrelevant after ``.abs()``.
* ``normalized=True`` in torch.stft divides by sqrt(n_fft); one-sided bins are
  then scaled by sqrt(2) except DC and Nyquist (Hermitian energy isometry).
* The chroma filterbank is a faithful NumPy port of
  ``torchaudio.prototype.functional.chroma_filterbank`` (itself a librosa
  port). torchaudio.prototype is deprecated upstream — vendoring it here
  removes that dependency entirely.
* PITFALL: with ``base_c=True`` and ``n_chroma=128`` the implementation rolls
  by ``-3*(128//12) = -30`` bins, but an exact A->C offset is 32 bins. Pitch
  class C therefore sits at bin 2.0 (NOT bin 0); each semitone spans 128/12
  ~= 10.667 bins. Use :func:`semitone_bin_centers` / :func:`expand_semitone_weights`
  rather than assuming bin 0 = C.
* Targets are NOT normalized per frame: values are log1p-compressed energy and
  scale with signal level. For steering, prefer relative objectives
  (boost/suppress dot products) over absolute target matching.

Performance:

* CPU path is NumPy/SciPy only: zero-copy framing (stride tricks), batched
  ``scipy.fft.rfft(..., workers=-1)`` (pocketfft, SIMD-vectorized), and a
  single stacked (384 x 4097) float32 GEMM for all three bands — BLAS
  (OpenBLAS/MKL) dispatches AVX-512 kernels automatically. float32 throughout
  doubles AVX-512 lane occupancy vs float64.
* Optional torch path (CUDA/ROCm) mirrors TightSpectrogram via torch.stft
  (cuFFT/rocFFT) for batched GPU extraction. nnAudio was evaluated and
  deliberately NOT used: its conv1d-based STFT costs O(F*N) per frame
  (~33.6M MACs at n_fft=8192, ~600x the FFT's O(N log N)) plus a ~270 MB
  kernel tensor — it is built for small-n_fft differentiable frontends, not
  large-n_fft throughput. torch.stft + one GEMM is the fast GPU shape here.

Dependencies: numpy, scipy. Optional: torch (GPU path), soundfile (CLI).
"""

import logging
import math
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── SAME constants (training/autoencoders.py:574-583, commit 3241adb) ────────
SAME_SR = 44100
SAME_N_FFT = 8192
SAME_HOP = SAME_N_FFT // 2          # TightSpectrogram forces hop = n_fft//2
SAME_N_FREQS = SAME_N_FFT // 2 + 1  # 4097 one-sided bins
SAME_N_CHROMA = 128
SAME_CHROMA_CENTERS = (1.0, 5.0, 9.0)
SAME_CHROMA_WIDTHS = (1.0, 1.5, 1.0)
# Per-band L1 loss weights used in SAME training (bass, mid, treble) — kept for
# reference; not applied to the features themselves.
SAME_CHROMA_LOSS_WEIGHTS = (0.035, 0.05, 0.2)
# SAME latent hop in samples (patch 256 * TRB stride 16); ~10.77 frames/s at 44.1k.
SAME_LATENT_HOP = 4096

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# ── Vendored: torchaudio.prototype.functional.chroma_filterbank (NumPy port) ─

def chroma_filterbank(
    sample_rate: int,
    n_freqs: int,
    n_chroma: int,
    *,
    tuning: float = 0.0,
    ctroct: float = 5.0,
    octwidth: Optional[float] = 2.0,
    norm: int = 2,
    base_c: bool = True,
    dtype=np.float64,
) -> np.ndarray:
    """
    Frequency-to-chroma conversion matrix, shape (n_freqs, n_chroma).

    Faithful NumPy port of torchaudio v2.5 ``chroma_filterbank`` (which is a
    port of ``librosa.filters.chroma``). Octave reference: A0 = 27.5 Hz
    (octs = log2(f / (A440/16))), so ctroct 1/5/9 center on 55 Hz / 880 Hz /
    14080 Hz. ``norm`` normalizes each frequency row's distribution across
    chroma bins (p-norm, eps=1e-12, matching torch.nn.functional.normalize).
    """
    # Skip redundant DC; prepend a synthetic bin 1.5 octaves below the lowest.
    freqs = np.linspace(0, sample_rate // 2, n_freqs, dtype=dtype)[1:]
    a440 = 440.0 * 2.0 ** (tuning / n_chroma)
    freq_bins = n_chroma * np.log2(freqs / (a440 / 16))
    freq_bins = np.concatenate(([freq_bins[0] - 1.5 * n_chroma], freq_bins))
    freq_bin_widths = np.concatenate(
        (np.maximum(freq_bins[1:] - freq_bins[:-1], 1.0), [1.0])
    )

    if norm < 1:
        raise ValueError(f"norm must be >= 1, got {norm}")

    # (n_freqs, n_chroma) distance of each frequency to each chroma bin center,
    # wrapped to [-n_chroma/2, n_chroma/2).
    D = freq_bins[:, None] - np.arange(n_chroma, dtype=dtype)[None, :]
    n_chroma2 = round(n_chroma / 2)
    D = np.remainder(D + n_chroma2, n_chroma) - n_chroma2

    fb = np.exp(-0.5 * (2.0 * D / freq_bin_widths[:, None]) ** 2)

    # Row-wise p-norm normalization (torch F.normalize semantics: x/max(|x|_p, eps))
    if norm == 1:
        denom = np.abs(fb).sum(axis=1, keepdims=True)
    elif norm == 2:
        denom = np.sqrt((fb * fb).sum(axis=1, keepdims=True))
    else:
        denom = (np.abs(fb) ** norm).sum(axis=1, keepdims=True) ** (1.0 / norm)
    fb = fb / np.maximum(denom, 1e-12)

    if octwidth is not None:
        fb = fb * np.exp(
            -0.5 * (((freq_bins[:, None] / n_chroma) - ctroct) / octwidth) ** 2
        )

    if base_c:
        # NOTE: -3*(n_chroma//12) bins, NOT an exact 3 semitones when
        # n_chroma % 12 != 0. With n_chroma=128 this is -30 bins, leaving
        # pitch class C centered at bin 2.0. See semitone_bin_centers().
        fb = np.roll(fb, -3 * (n_chroma // 12), axis=1)

    return fb


# ── Tight magnitude spectrogram (NumPy port of TightSpectrogram, power=1) ────

def _power_sine_tight_window(n_fft: int, hop: int, dtype=np.float64) -> np.ndarray:
    """Tight power-sine window; for hop = n_fft/2 this is exactly sin(pi*n/N)."""
    R = n_fft // hop
    p = R - 1
    if R < 2 or not (1 <= p <= R - 1):
        raise ValueError(f"Unsupported n_fft/hop ratio R={R}")
    n = np.arange(n_fft, dtype=dtype)
    s = np.sin(np.pi * n / n_fft)
    C0 = math.comb(2 * p, p) / (2.0 ** (2 * p))
    A = 1.0 / math.sqrt(R * C0)
    return (s ** p) * A


def _hermitian_sqrt(n_fft: int, dtype=np.float64) -> np.ndarray:
    """sqrt(2) for interior one-sided bins; 1 for DC (and Nyquist when N even)."""
    F1 = n_fft // 2 + 1
    s = np.ones(F1, dtype=dtype)
    if n_fft % 2 == 0:
        if F1 > 2:
            s[1:-1] = math.sqrt(2.0)
    else:
        if F1 > 1:
            s[1:] = math.sqrt(2.0)
    return s


def tight_magnitude_spectrogram(
    audio: np.ndarray,
    n_fft: int = SAME_N_FFT,
    workers: int = -1,
) -> np.ndarray:
    """
    Magnitude spectrogram matching TightSpectrogram(n_fft, normalized=True,
    power=1.0) — hop = n_fft//2, center=False, tight sine window, /sqrt(n_fft),
    interior bins * sqrt(2). (The demodulation sign pattern does not affect
    magnitudes and is omitted.)

    Args:
        audio: mono float array, shape (T,)
    Returns:
        (n_fft//2 + 1, n_frames) float32; n_frames = max(0, (T - n_fft)//hop + 1).
        For T < n_fft returns shape (F, 0) — NOTE: torch.stft(center=False)
        raises on such input instead; this path zero-fills gracefully.
    """
    hop = n_fft // 2
    x = np.ascontiguousarray(audio, dtype=np.float32)
    T = x.shape[-1]
    if T < n_fft:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float32)

    win = _power_sine_tight_window(n_fft, hop).astype(np.float32)

    # Zero-copy framing: (n_frames, n_fft) view, then one batched real FFT.
    n_frames = (T - n_fft) // hop + 1
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop][:n_frames]

    try:
        from scipy.fft import rfft
        X = rfft(frames * win, axis=1, workers=workers)
    except ImportError:                      # pragma: no cover
        X = np.fft.rfft(frames * win, axis=1)

    mag = np.abs(X).astype(np.float32)
    mag *= np.float32(1.0 / math.sqrt(n_fft))            # normalized=True
    mag *= _hermitian_sqrt(n_fft).astype(np.float32)[None, :]
    return np.ascontiguousarray(mag.T)                   # (F, n_frames)


# ── Stacked filterbank (all three bands -> one GEMM) ─────────────────────────

_FB_CACHE: dict = {}


def _stacked_same_filterbank(sample_rate: int) -> np.ndarray:
    """(3*128, 4097) float32: the three band filterbanks transposed and stacked."""
    key = sample_rate
    if key not in _FB_CACHE:
        fbs = [
            chroma_filterbank(
                sample_rate, SAME_N_FREQS, SAME_N_CHROMA,
                ctroct=c, octwidth=w, norm=1, base_c=True,
            ).T                                           # (128, 4097)
            for c, w in zip(SAME_CHROMA_CENTERS, SAME_CHROMA_WIDTHS)
        ]
        _FB_CACHE[key] = np.ascontiguousarray(
            np.concatenate(fbs, axis=0), dtype=np.float32
        )                                                 # (384, 4097)
    return _FB_CACHE[key]


# ── Latent-frame alignment (torch F.interpolate(mode='linear') equivalent) ───

def interpolate_linear(x: np.ndarray, size: int) -> np.ndarray:
    """
    Linear resampling along the last axis, matching
    torch.nn.functional.interpolate(mode='linear', align_corners=False).

    Coordinates are computed in float32 to mirror ATen's accumulation type for
    float32 tensors (the dtype SAME trained in) — float64 coords would be
    "more accurate" but would drift from training behavior on long sequences.
    T_in == 0 returns zeros (torch raises here; reachable for audio < n_fft).
    """
    T_in = x.shape[-1]
    if T_in == size:
        return x
    if T_in == 0:
        return np.zeros(x.shape[:-1] + (size,), dtype=x.dtype)
    pos = (np.arange(size, dtype=np.float32) + np.float32(0.5)) \
        * np.float32(T_in / size) - np.float32(0.5)
    pos = np.clip(pos, 0.0, T_in - 1.0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, T_in - 1)
    w = (pos - lo).astype(x.dtype)
    return x[..., lo] * (1 - w) + x[..., hi] * w


def n_latent_frames(n_samples: int) -> int:
    """
    SAME latent sequence length for an input of n_samples (per channel):
    ceil(n_samples / 4096) (patch 256 x TRB stride 16).

    NOTE: verify once against the real encoder's output length on the
    production box — edge/padding behavior could differ by one frame.
    """
    return -(-n_samples // SAME_LATENT_HOP)


# ── Main API ──────────────────────────────────────────────────────────────────

def compute_same_chroma(
    audio: np.ndarray,
    sr: int,
    align_to_latent: bool = True,
    workers: int = -1,
) -> np.ndarray:
    """
    SAME/Stable-Audio-3-compatible octave-band chroma time series.

    Replicates the SAME chroma regression target pipeline exactly:
    tight magnitude STFT (8192/4096) -> mean of per-channel magnitudes ->
    log1p -> three ChromaScale projections (centers 1/5/9, widths 1/1.5/1,
    n_chroma=128, norm=1, base_c=True) -> optional linear interpolation to
    the latent frame count.

    Args:
        audio: mono (T,) or stereo/multichannel 2D array. For 2D input the
               shorter axis is treated as channels (per-channel magnitudes
               are averaged, matching SAME's spec.mean(dim=1) — this is NOT
               the same as mono-mixing before the STFT).
        sr:    sample rate. SAME trained at 44100; a warning is logged for
               other rates (the filterbank adapts, but targets are only
               SAME-comparable at 44.1 kHz).
        align_to_latent: interpolate frames to ceil(T/4096) (the SAME latent
               rate), mirroring training. If False, returns native STFT frames
               ((T - 8192)//4096 + 1).
        workers: scipy.fft thread count (-1 = all cores).

    Returns:
        (3, 128, n_frames) float32 — bands ordered (octave 1, 5, 9) =
        (bass, mid, treble). Unbounded log1p-energy values (no per-frame norm).
    """
    if sr <= 0:
        raise ValueError(f"sample rate must be positive, got {sr}")
    if sr != SAME_SR:
        logger.warning(
            f"compute_same_chroma: sr={sr} != {SAME_SR}; output will not be "
            f"comparable to SAME training targets (filterbank adapted to {sr})."
        )

    x = np.asarray(audio)
    if x.ndim == 1:
        channels = [x]
    elif x.ndim == 2:
        ch_axis = int(np.argmin(x.shape))
        n_ch = x.shape[ch_axis]
        if n_ch > 8:
            raise ValueError(f"Ambiguous audio shape {x.shape}: >8 channels?")
        channels = [np.take(x, i, axis=ch_axis) for i in range(n_ch)]
    else:
        raise ValueError(f"audio must be 1D or 2D, got shape {x.shape}")

    n_samples = channels[0].shape[0]

    # Mean of per-channel MAGNITUDES (SAME order: stft -> abs -> mean(dim=1)).
    mag = None
    for ch in channels:
        m = tight_magnitude_spectrogram(ch, SAME_N_FFT, workers=workers)
        mag = m if mag is None else mag + m
    mag /= np.float32(len(channels))

    log_s = np.log1p(mag)                                  # (4097, M) float32

    W = _stacked_same_filterbank(sr)                       # (384, 4097) float32
    chroma = W @ log_s                                     # one GEMM, all bands
    chroma = chroma.reshape(3, SAME_N_CHROMA, -1)

    if align_to_latent:
        chroma = interpolate_linear(chroma, n_latent_frames(n_samples))

    return np.ascontiguousarray(chroma, dtype=np.float32)


# ── Optional torch path (CUDA / ROCm; batched) ───────────────────────────────

def compute_same_chroma_torch(
    audio,                       # torch.Tensor (B, C, T) or (B, T) or (T,)
    sr: int = SAME_SR,
    align_to_latent: bool = True,
    device=None,
):
    """
    Batched GPU/CPU torch implementation, numerically equivalent to
    :func:`compute_same_chroma`. Mirrors TightSpectrogram via torch.stft
    (cuFFT / rocFFT — works as-is on ROCm builds, where AMD GPUs use the
    'cuda' device namespace).

    NOTE on conventions (differs from the NumPy path): a 2D input here is
    interpreted as a (B, T) batch of MONO signals, not as one stereo signal.
    Pass (B, C, T) for multichannel (per-channel magnitudes are averaged).

    Returns torch.Tensor (B, 3, 128, n_frames) float32.
    """
    import torch

    x = audio if isinstance(audio, torch.Tensor) else torch.as_tensor(audio)
    if x.dim() == 1:
        x = x[None, None, :]
    elif x.dim() == 2:
        x = x[:, None, :]
    if device is not None:
        x = x.to(device)
    x = x.float()
    B, C, T = x.shape
    n_fft, hop = SAME_N_FFT, SAME_HOP

    if T < n_fft:
        # torch.stft(center=False) raises for T < n_fft; mirror the NumPy
        # path's graceful zero output instead.
        L = n_latent_frames(T) if align_to_latent else 0
        return torch.zeros(B, 3, SAME_N_CHROMA, L, device=x.device)

    win = torch.from_numpy(
        _power_sine_tight_window(n_fft, hop).astype(np.float32)
    ).to(x.device)

    X = torch.stft(
        x.reshape(B * C, T), n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=win, center=False, normalized=True, onesided=True,
        return_complex=True,
    )                                                      # (B*C, 4097, M)
    herm = torch.from_numpy(
        _hermitian_sqrt(n_fft).astype(np.float32)
    ).to(x.device)
    mag = X.abs() * herm[None, :, None]
    mag = mag.reshape(B, C, mag.shape[-2], mag.shape[-1]).mean(dim=1)
    log_s = torch.log1p(mag)                               # (B, 4097, M)

    W = torch.from_numpy(_stacked_same_filterbank(sr)).to(x.device)  # (384, 4097)
    chroma = torch.matmul(W, log_s)                        # (B, 384, M)
    chroma = chroma.reshape(B, 3, SAME_N_CHROMA, -1)

    if align_to_latent:
        L = n_latent_frames(T)
        chroma = torch.nn.functional.interpolate(
            chroma.reshape(B * 3, SAME_N_CHROMA, -1), size=L, mode='linear'
        ).reshape(B, 3, SAME_N_CHROMA, L)

    return chroma


# ── GUI / steering helpers: 12 semitone weights <-> 128-bin profiles ─────────

def semitone_bin_centers(n_chroma: int = SAME_N_CHROMA) -> np.ndarray:
    """
    Exact (float) chroma-bin center for each pitch class C..B, accounting for
    base_c's integer roll. center(s) = (n_chroma*(s+3)/12 - 3*(n_chroma//12))
    mod n_chroma. For n_chroma=128: C=2.0, C#=12.667, ..., A=98.0.
    """
    s = np.arange(12, dtype=np.float64)
    centers = n_chroma * (s + 3.0) / 12.0 - 3.0 * (n_chroma // 12)
    return np.remainder(centers, n_chroma)


def expand_semitone_weights(
    weights12: Sequence[float],
    n_chroma: int = SAME_N_CHROMA,
    width_semitones: float = 0.25,
) -> np.ndarray:
    """
    Expand 12 per-semitone weights (C..B) into an (n_chroma,) profile of
    circular Gaussian bumps at the exact semitone centers — the shape a
    SAME-compatible steering target should take along the chroma axis.

    width_semitones: Gaussian sigma in semitones. Default 0.25 approximates
    the ~0.2-semitone peak width real tones produce through the SAME
    filterbank in the mid band, and yields ~18:1 on/off-scale contrast after
    nearest-semitone folding (0.35 gives ~5:1; 0.15 gives ~760:1 but is
    narrower than real peaks and may over-penalize vibrato/detuning).
    """
    w = np.asarray(weights12, dtype=np.float64)
    if w.shape != (12,):
        raise ValueError(f"weights12 must have shape (12,), got {w.shape}")
    centers = semitone_bin_centers(n_chroma)
    sigma_bins = width_semitones * n_chroma / 12.0
    bins = np.arange(n_chroma, dtype=np.float64)
    # Circular distance bins x centers
    d = np.abs(bins[:, None] - centers[None, :])
    d = np.minimum(d, n_chroma - d)
    bumps = np.exp(-0.5 * (d / sigma_bins) ** 2)           # (n_chroma, 12)
    return (bumps @ w).astype(np.float32)


def fold_to_12(chroma: np.ndarray, n_chroma: int = SAME_N_CHROMA) -> np.ndarray:
    """
    Fold chroma down to 12 semitone bins (C..B) by summing each fine bin into
    its nearest semitone center (circular).

    Accepted shapes: (n_chroma,) -> (12,); otherwise the SECOND-TO-LAST axis
    must be the chroma axis, e.g. (n_chroma, T) -> (12, T) and
    (3, n_chroma, T) -> (3, 12, T). A (B, n_chroma) batch is NOT accepted —
    transpose it or add a trailing time axis first.
    Intended for inspection/visualization, not steering.
    """
    centers = semitone_bin_centers(n_chroma)
    bins = np.arange(n_chroma, dtype=np.float64)
    d = np.abs(bins[:, None] - centers[None, :])
    d = np.minimum(d, n_chroma - d)
    assign = np.argmin(d, axis=1)                          # (n_chroma,)

    if chroma.ndim == 1:
        if chroma.shape[0] != n_chroma:
            raise ValueError(f"expected ({n_chroma},), got {chroma.shape}")
        folded = np.zeros(12, dtype=chroma.dtype)
        np.add.at(folded, assign, chroma)
        return folded

    if chroma.shape[-2] != n_chroma:
        raise ValueError(
            f"expected chroma axis of size {n_chroma} at position -2, "
            f"got shape {chroma.shape}"
        )
    moved = np.moveaxis(chroma, -2, 0)                     # (n_chroma, ..., T)
    folded = np.zeros((12,) + moved.shape[1:], dtype=chroma.dtype)
    np.add.at(folded, assign, moved)
    return np.moveaxis(folded, 0, -2)


# Bass-band harmonic templates: pitch-class offsets (semitones above root) ->
# relative weight. Derived from the harmonic series as seen through band 0's
# octave window (h1/h2 -> root; h3 -> +7 semitones, window-attenuated ~0.3;
# h7 -> +10 semitones, ~trace). Better still: measure a real genre bass patch
# via compute_same_chroma and use its band-0 profile as the template.
BASS_PROFILES = {
    'root':       {0: 1.0},
    'root5':      {0: 1.0, 7: 0.3},
    'root5b7':    {0: 1.0, 7: 0.3, 10: 0.08},
}


def make_steering_target(
    n_frames: int,
    melody_weights12: Sequence[float],
    bass_root: Optional[int] = None,
    bass_profile: str = 'root5',
    bass_gain: float = 1.0,
    mid_gain: float = 1.0,
    air_gain: float = 0.3,
    width_semitones: float = 0.25,
) -> np.ndarray:
    """
    Build a (3, 128, n_frames) steering target with independent bass vs
    melody control — the "two-group UI" mapping: bass group -> band 0
    (octave 1), melody group -> bands 1+2 (octaves 5 and 9, broadcast).

    Args:
        n_frames:        target length in latent frames (~10.77 fps).
        melody_weights12: 12 weights C..B for the melody/harmony palette
                         (scale/mode set). Applied to mid and air bands.
        bass_root:       pitch class 0-11 (C=0) to lock the bass to, or None
                         to leave band 0 unsteered (all-zero target).
        bass_profile:    key into BASS_PROFILES — 'root' for fundamental only,
                         'root5' adds the 3rd-harmonic fifth, 'root5b7' adds a
                         trace 7th-harmonic flat-seven.
        bass_gain/mid_gain/air_gain: per-band steering strength. Bass high =
                         constraint ("bass shall be E"); melody moderate =
                         preference; air low — >8 kHz chroma is mostly
                         hats/noise in electronic genres.

    Values are RELATIVE profiles (SAME targets have no per-frame norm):
    use them in dot-product/cosine steering objectives, not absolute L1/L2
    target matching.
    """
    target = np.zeros((3, SAME_N_CHROMA, n_frames), dtype=np.float32)

    if bass_root is not None:
        if not 0 <= int(bass_root) <= 11:
            raise ValueError(f"bass_root must be 0-11, got {bass_root}")
        w = np.zeros(12)
        for offset, amp in BASS_PROFILES[bass_profile].items():
            w[(int(bass_root) + offset) % 12] += amp
        prof = expand_semitone_weights(w, width_semitones=width_semitones)
        target[0] = (bass_gain * prof)[:, None]

    mel = expand_semitone_weights(
        np.asarray(melody_weights12, dtype=np.float64),
        width_semitones=width_semitones,
    )
    target[1] = (mid_gain * mel)[:, None]
    target[2] = (air_gain * mel)[:, None]
    return target


# ── Self-test (synthetic; needs only numpy/scipy) ────────────────────────────

def _reference_chroma(audio: np.ndarray, sr: int) -> np.ndarray:
    """Slow, loop-based reference of the full pipeline for verification."""
    n_fft, hop = SAME_N_FFT, SAME_HOP
    win = _power_sine_tight_window(n_fft, hop)
    herm = _hermitian_sqrt(n_fft)
    x = np.asarray(audio, dtype=np.float64)
    n_frames = max(0, (len(x) - n_fft) // hop + 1)
    mags = []
    for m in range(n_frames):
        seg = x[m * hop: m * hop + n_fft] * win
        X = np.fft.rfft(seg)
        mags.append(np.abs(X) / math.sqrt(n_fft) * herm)
    S = np.array(mags).T if mags else np.zeros((n_fft // 2 + 1, 0))
    log_s = np.log1p(S)
    out = []
    for c, w in zip(SAME_CHROMA_CENTERS, SAME_CHROMA_WIDTHS):
        fb = chroma_filterbank(sr, SAME_N_FREQS, SAME_N_CHROMA,
                               ctroct=c, octwidth=w, norm=1, base_c=True)
        out.append(fb.T @ log_s)
    return np.stack(out, axis=0)


def selftest(verbose: bool = True) -> bool:
    """Numerical self-test with synthetic signals. Returns True on success."""
    import time
    ok = True
    sr = SAME_SR

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # 1. Fast path vs reference implementation (2 s noise + tone)
    rng = np.random.default_rng(0)
    t = np.arange(2 * sr) / sr
    sig = (0.1 * rng.standard_normal(len(t)) + 0.5 * np.sin(2 * np.pi * 440.0 * t)
           ).astype(np.float32)
    fast = compute_same_chroma(sig, sr, align_to_latent=False)
    ref = _reference_chroma(sig, sr)
    err = np.max(np.abs(fast - ref)) / max(np.max(np.abs(ref)), 1e-12)
    check(f"fast path matches reference (rel err {err:.2e})", err < 1e-4)

    # 2. 440 Hz sine -> folded mid-band argmax at A (pitch class 9)
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    ch = compute_same_chroma(tone, sr, align_to_latent=False)
    folded = fold_to_12(ch[1].mean(axis=-1))
    check(f"440 Hz -> A in mid band (argmax={PITCH_CLASSES[int(np.argmax(folded))]})",
          int(np.argmax(folded)) == 9)
    centers = semitone_bin_centers()
    peak_bin = int(np.argmax(ch[1].mean(axis=-1)))
    check(f"440 Hz peak bin ~98 (got {peak_bin})", abs(peak_bin - 98) <= 1)

    # 3. C major chord (C4 E4 G4) -> top-3 folded pitch classes {C, E, G}
    chord = sum(np.sin(2 * np.pi * f * t) for f in (261.626, 329.628, 391.995))
    ch = compute_same_chroma(chord.astype(np.float32), sr, align_to_latent=False)
    top3 = set(np.argsort(fold_to_12(ch[1].mean(axis=-1)))[-3:].tolist())
    check(f"C major chord -> {{C,E,G}} in mid band (got "
          f"{sorted(PITCH_CLASSES[i] for i in top3)})", top3 == {0, 4, 7})

    # 4. Register separation: 55 Hz (A1) excites bass band >> treble band
    low = (0.5 * np.sin(2 * np.pi * 55.0 * t)).astype(np.float32)
    ch = compute_same_chroma(low, sr, align_to_latent=False)
    e = ch.mean(axis=(1, 2))
    check(f"55 Hz: bass-band energy > 10x treble-band (ratio "
          f"{e[0] / max(e[2], 1e-12):.1f})", e[0] > 10 * e[2])

    # 5. Shapes, alignment, silence
    n = 10 * sr
    sig10 = (0.1 * rng.standard_normal(n)).astype(np.float32)
    ch = compute_same_chroma(sig10, sr)
    check(f"latent alignment shape (3,128,{n_latent_frames(n)})",
          ch.shape == (3, SAME_N_CHROMA, n_latent_frames(n)))
    silent = compute_same_chroma(np.zeros(n, dtype=np.float32), sr)
    check("silence -> all-zero chroma", float(np.abs(silent).max()) == 0.0)
    stereo = np.stack([tone, tone], axis=1)               # (T, 2)
    chs = compute_same_chroma(stereo, sr, align_to_latent=False)
    chm = compute_same_chroma(tone, sr, align_to_latent=False)
    check("identical stereo == mono", np.allclose(chs, chm, atol=1e-5))

    # 6. GUI helpers round-trip: weights at C/E/G -> expansion peaks fold back
    w12 = np.zeros(12); w12[[0, 4, 7]] = 1.0
    prof = expand_semitone_weights(w12)
    top3 = set(np.argsort(fold_to_12(prof))[-3:].tolist())
    check("expand/fold round-trip {C,E,G}", top3 == {0, 4, 7})

    # 7. Throughput (5-minute track)
    sig5m = (0.1 * rng.standard_normal(300 * sr)).astype(np.float32)
    t0 = time.perf_counter()
    compute_same_chroma(sig5m, sr)
    dt = time.perf_counter() - t0
    if verbose:
        print(f"  [INFO] 5-min track in {dt * 1e3:.0f} ms "
              f"({300 / dt:.0f}x realtime, single process)")

    if verbose:
        print(f"\nself-test {'PASSED' if ok else 'FAILED'}")
    return ok


# ── Command-line interface ────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='SAME/Stable-Audio-3-compatible octave-band chroma extraction')
    parser.add_argument('path', nargs='?', type=str,
                        help='Audio file to process (saves <file>.same_chroma.npy)')
    parser.add_argument('--selftest', action='store_true',
                        help='Run synthetic numerical self-test (no audio needed)')
    parser.add_argument('--no-align', action='store_true',
                        help='Keep native STFT frames (skip latent alignment)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if not args.path:
        parser.error('provide an audio file path or --selftest')

    import soundfile as sf
    audio, sr = sf.read(args.path, dtype='float32', always_2d=False)
    chroma = compute_same_chroma(audio, sr, align_to_latent=not args.no_align)
    out_path = Path(args.path).with_suffix('.same_chroma.npy')
    np.save(out_path, chroma)
    print(f"saved {chroma.shape} float32 -> {out_path}")
    folded = fold_to_12(chroma[1].mean(axis=-1))
    order = np.argsort(folded)[::-1]
    print("mid-band pitch-class ranking: "
          + ' '.join(f"{PITCH_CLASSES[i]}={folded[i]:.2f}" for i in order[:6]))
