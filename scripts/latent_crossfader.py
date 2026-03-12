#!/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python
"""
latent_crossfader.py — Latent-space crossfade engine for Stable Audio Small.

Math spec
---------
Given stem latents z_stem_A, z_stem_B (two tracks) and full-mix latents z_fullmix_A,
z_fullmix_B (for color anchoring):

  Step 1 – interpolate stem tracks in latent space:
      z_math = Interp(z_stem_A, z_stem_B, α)   # α ∈ [0,1]; Interp = slerp or lerp

  Step 2 – dual reality anchor (optional; β=0 skips each):
      z_math = slerp(z_math, z_fullmix_A, β_A)  # pull toward A's full-mix (adds A color)
      z_math = slerp(z_math, z_fullmix_B, β_B)  # pull toward B's full-mix (adds B color)

  Step 3 – decode per stem, sum:
      audio_master = tanh( Σ_i Decoder(z_math_i) )

All slerps operate on 3-D tensors [B, C, T] flattened to [B, C*T] on the unit sphere
with energy-preserving scaling and a lerp fallback for near-parallel vectors.

Public API
----------
  setup_device(device=None)            → torch.device
  slerp(z_a, z_b, t)                  → Tensor  (same shape as inputs)
  lerp(z_a, z_b, t)                   → Tensor  (same shape as inputs)
  reality_anchor(z_stem_a, z_stem_b, z_fullmix_a, z_fullmix_b,
                 alpha, beta_a, beta_b, interp_fn=None) → Tensor
  crossfade_stems(stems_a, stems_b, fullmix_a, fullmix_b, alphas,
                  beta_a, beta_b, decoder, interp='slerp', device=None)
                                       → Tensor  [1, 2, samples]
  soft_clip(audio)                     → Tensor  (tanh soft-clip)

Stem order convention: ["drums", "bass", "other", "vocals"]
"""

import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

STEMS = ["drums", "bass", "other", "vocals"]


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def setup_device(device: Optional[str] = None) -> torch.device:
    """Return the best available device, or the one requested."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Slerp
# ---------------------------------------------------------------------------

def slerp(z_a: torch.Tensor, z_b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two latent tensors.

    Supports any shape, but designed for [B, C, T] (VAE latents).

    The interpolation is energy-preserving: the result norm equals the lerp
    of the input norms, so silence-level latents stay at their energy.

    Falls back to lerp when the vectors are nearly parallel (sin(ω) < 1e-6).

    Args:
        z_a: start latent tensor
        z_b: end latent tensor (same shape as z_a)
        t:   interpolation factor; 0.0 → z_a, 1.0 → z_b

    Returns:
        Interpolated tensor with the same shape and dtype as inputs.
    """
    orig_shape = z_a.shape
    # Flatten to [B, N] where N = C * T
    fa = z_a.reshape(orig_shape[0], -1).float()
    fb = z_b.reshape(orig_shape[0], -1).float()

    norm_a = fa.norm(dim=1, keepdim=True).clamp(min=1e-8)
    norm_b = fb.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Unit vectors
    ua = fa / norm_a
    ub = fb / norm_b

    # Dot product, clamped to [-1, 1]
    dot = (ua * ub).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot.abs())        # angle on the unit sphere

    sin_omega = torch.sin(omega)

    # Lerp fallback mask (near-parallel vectors)
    near_parallel = sin_omega.squeeze(1) < 1e-6   # [B]

    # Slerp coefficients
    coeff_a = torch.sin((1.0 - t) * omega) / sin_omega.clamp(min=1e-8)
    coeff_b = torch.sin(t * omega)         / sin_omega.clamp(min=1e-8)

    # Handle sign flip: if dot is negative we crossed a hemisphere
    coeff_b = coeff_b * dot.sign().where(dot != 0, torch.ones_like(dot))

    result = coeff_a * ua + coeff_b * ub   # [B, N] on unit sphere

    # Energy-preserving: scale by lerp of input norms
    target_norm = (1.0 - t) * norm_a + t * norm_b
    result = result * target_norm

    # Lerp fallback for near-parallel pairs
    lerp_result = (1.0 - t) * fa + t * fb
    for i in range(orig_shape[0]):
        if near_parallel[i]:
            result[i] = lerp_result[i]

    return result.reshape(orig_shape).to(z_a.dtype)


# ---------------------------------------------------------------------------
# Lerp
# ---------------------------------------------------------------------------

def lerp(z_a: torch.Tensor, z_b: torch.Tensor, t: float) -> torch.Tensor:
    """Linear interpolation between two latent tensors."""
    return (1.0 - t) * z_a + t * z_b


# ---------------------------------------------------------------------------
# Reality anchor
# ---------------------------------------------------------------------------

def reality_anchor(
    z_stem_a:    torch.Tensor,   # [B, C, T] stem from track A
    z_stem_b:    torch.Tensor,   # [B, C, T] stem from track B
    z_fullmix_a: torch.Tensor,   # [B, C, T] full-mix from track A (for β_a anchor)
    z_fullmix_b: torch.Tensor,   # [B, C, T] full-mix from track B (for β_b anchor)
    alpha:  float,
    beta_a: float,
    beta_b: float,
    interp_fn=None,              # slerp (default) or lerp for the α step
) -> torch.Tensor:
    """Three-stage latent morph with dual color anchors.

    Step 1: interp_fn(z_stem_a, z_stem_b, alpha)  — A↔B stem blend
    Step 2: slerp(z_math, z_fullmix_a, beta_a)    — pull toward A full-mix color
    Step 3: slerp(z_math, z_fullmix_b, beta_b)    — pull toward B full-mix color

    The anchor steps rotate the latent *direction* toward the full-mix color but
    preserve the stem's own energy (norm).  This prevents the 4-stem summing from
    blowing up: z_fullmix typically has much larger norm than individual stems, so
    magnitude-lerping toward it would multiply decoded energy by ~n_stems at β=1.

    Args:
        z_stem_a:    stem latent for track A  [B, C, T]
        z_stem_b:    stem latent for track B  [B, C, T]
        z_fullmix_a: full-mix latent for track A  [B, C, T]
        z_fullmix_b: full-mix latent for track B  [B, C, T]
        alpha:   A→B blend (0 = track A, 1 = track B)
        beta_a:  pull toward A full-mix color (0 = no pull, 1 = full A direction)
        beta_b:  pull toward B full-mix color (0 = no pull, 1 = full B direction)
        interp_fn: callable(z_a, z_b, t) for the α step; defaults to slerp

    Returns:
        Target latent tensor  [B, C, T] with the same norm as the α-blended stem.
    """
    if interp_fn is None:
        interp_fn = slerp
    z_math = interp_fn(z_stem_a, z_stem_b, alpha)

    if beta_a > 0 or beta_b > 0:
        # Remember the stem energy before anchoring
        B = z_math.shape[0]
        stem_norm = z_math.reshape(B, -1).norm(dim=1, keepdim=True).clamp(min=1e-8)

        if beta_a > 0:
            z_math = slerp(z_math, z_fullmix_a, beta_a)
        if beta_b > 0:
            z_math = slerp(z_math, z_fullmix_b, beta_b)

        # Renormalize: restore stem energy so decoded amplitudes stay consistent
        # regardless of how much larger the full-mix latent is.
        anchored_norm = z_math.reshape(B, -1).norm(dim=1, keepdim=True).clamp(min=1e-8)
        scale = (stem_norm / anchored_norm).view(B, *([1] * (z_math.dim() - 1)))
        z_math = z_math * scale

    return z_math


# ---------------------------------------------------------------------------
# Soft clip
# ---------------------------------------------------------------------------

def soft_clip(audio: torch.Tensor) -> torch.Tensor:
    """tanh soft-clip — keeps the signal in (-1, 1) without hard clipping."""
    return torch.tanh(audio)


# ---------------------------------------------------------------------------
# Crossfade stems
# ---------------------------------------------------------------------------

def crossfade_stems(
    stems_a:   dict,               # stem → [B, C, T]  track A stem latents
    stems_b:   dict,               # stem → [B, C, T]  track B stem latents
    fullmix_a: torch.Tensor,       # [B, C, T]  track A full-mix latent
    fullmix_b: torch.Tensor,       # [B, C, T]  track B full-mix latent
    alphas:    Union[dict, float], # per-stem α or single float
    beta_a:    float,
    beta_b:    float,
    decoder,
    interp:    str = 'slerp',      # 'slerp' or 'lerp'
    device:    Optional[torch.device] = None,
) -> torch.Tensor:
    """Decode and sum crossfaded stems.

    For each stem in STEMS (skipped silently if missing from either dict):
      z_target_i = reality_anchor(stems_a[i], stems_b[i], fullmix_a, fullmix_b,
                                  alpha_i, beta_a, beta_b, interp_fn)
      audio_i    = decoder(z_target_i)

    Then:
      audio_master = tanh( sum(audio_i) )

    Args:
        stems_a:   dict stem→Tensor [B, C, T] for track A
        stems_b:   dict stem→Tensor [B, C, T] for track B
        fullmix_a: full-mix latent for track A (β_a anchor)
        fullmix_b: full-mix latent for track B (β_b anchor)
        alphas:    per-stem alpha (dict stem→float) or a single float for all
        beta_a:    pull toward A full-mix (0 = no pull, 1 = full A mix)
        beta_b:    pull toward B full-mix (0 = no pull, 1 = full B mix)
        decoder:   callable; takes latent [B, C, T] → audio [B, 2, samples]
        interp:    'slerp' or 'lerp' for the α interpolation step
        device:    torch.device to move latents to before decoding

    Returns:
        audio_master [1, 2, samples]  — soft-clipped, ready to write to WAV
    """
    if isinstance(alphas, float):
        alphas = {s: alphas for s in STEMS}

    interp_fn = slerp if interp == 'slerp' else lerp

    if device is not None:
        fullmix_a = fullmix_a.to(device)
        fullmix_b = fullmix_b.to(device)

    audio_sum = None

    for stem in STEMS:
        z_a = stems_a.get(stem)
        z_b = stems_b.get(stem)

        if z_a is None or z_b is None:
            continue   # skip missing stems silently

        if device is not None:
            z_a = z_a.to(device)
            z_b = z_b.to(device)

        alpha = alphas.get(stem, 0.5) if isinstance(alphas, dict) else float(alphas)
        z_target = reality_anchor(
            z_a, z_b, fullmix_a, fullmix_b,
            alpha, beta_a, beta_b,
            interp_fn=interp_fn,
        )

        with torch.no_grad():
            audio_stem = decoder(z_target)   # [B, 2, samples]

        if audio_sum is None:
            audio_sum = audio_stem
        else:
            audio_sum = audio_sum + audio_stem

    if audio_sum is None:
        raise ValueError("No stems decoded — check that stems dicts contain matching keys.")

    return soft_clip(audio_sum)


# ---------------------------------------------------------------------------
# Convenience: load a latent .npy as a batch-1 tensor
# ---------------------------------------------------------------------------

def load_latent(npy_path, device: Optional[torch.device] = None,
                dtype=torch.float32) -> torch.Tensor:
    """Load a [C, T] .npy latent and return [1, C, T] tensor."""
    arr = np.load(str(npy_path)).astype(np.float32)
    t = torch.from_numpy(arr).unsqueeze(0)   # [1, C, T]
    if dtype != torch.float32:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t
