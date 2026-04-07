"""
LatCH inference hook for the Unified MIR Explorer.

Phase 1 (implemented here):
  Given a stored latent z [64, 256] and a control feature name, nudge z
  in the direction that increases the predicted feature value.

  If models/latch/{feature}.pt exists: use LatCH gradient at t≈0
    (single forward+backward pass, no diffusion sampling required).
  Else: fall back to correlation-coefficient offset from
    models/latent_correlations.json (existing behaviour).

Phase 2 (future, not implemented):
  Full LatCH-guided generation from noise via Euler TFG loop.
  Training: /home/kim/Projects/SAO/stable-audio-tools/scripts/train_latch.py
  Inference: /home/kim/Projects/SAO/stable-audio-tools/scripts/generate_latch_guided.py
  Checkpoints: models/latch/{feature}.pt
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

_REPO_ROOT         = Path(__file__).parent.parent.parent
_DEFAULT_LATCH_DIR = _REPO_ROOT / "models" / "latch"
_CORR_JSON         = _REPO_ROOT / "models" / "latent_correlations.json"

_CORR_CACHE: dict | None = None


def _load_correlations() -> dict:
    global _CORR_CACHE
    if _CORR_CACHE is None:
        if _CORR_JSON.exists():
            with open(_CORR_JSON) as f:
                _CORR_CACHE = json.load(f)
        else:
            _CORR_CACHE = {}
    return _CORR_CACHE


def apply_latch_guidance(
    z: np.ndarray,
    feature: str,
    strength: float,
    latch_dir: Path | None = None,
) -> np.ndarray:
    """
    Nudge latent z [64, 256] toward higher predicted `feature` value.

    Args:
        z:         Input latent [64, 256] float32.
        feature:   Feature name (e.g. "brightness", "rms_energy_bass").
        strength:  Scalar multiplier. 0 = no change; positive = increase feature;
                   negative = decrease feature.
        latch_dir: Directory containing LatCH checkpoints ({feature}.pt).
                   Defaults to models/latch/.

    Returns:
        Modified latent [64, 256] float32.
    """
    if latch_dir is None:
        latch_dir = _DEFAULT_LATCH_DIR
    if abs(strength) < 1e-6:
        return z.copy()

    ckpt = Path(latch_dir) / f"{feature}.pt"
    if ckpt.exists():
        return _latch_gradient_guidance(z, feature, strength, ckpt)
    return _correlation_fallback(z, feature, strength)


def _latch_gradient_guidance(
    z: np.ndarray, feature: str, strength: float, ckpt_path: Path
) -> np.ndarray:
    """
    Use LatCH gradient ∂prediction/∂z at t≈0 to nudge z.

    The LatCH model architecture is at:
      /home/kim/Projects/SAO/stable-audio-tools/scripts/latch_model.py
    Input:  z [1, 64, 256] + t [1] (very small noise, t≈0.001)
    Output: predicted feature [1, 1, 256]
    Gradient direction: ∂mean(pred)/∂z, shape [64, 256]
    """
    import sys
    latch_scripts = Path("/home/kim/Projects/SAO/stable-audio-tools/scripts")
    if str(latch_scripts) not in sys.path:
        sys.path.insert(0, str(latch_scripts))

    try:
        import torch
        from latch_model import LatCH  # type: ignore

        device = "cpu"
        model  = LatCH(in_channels=64, out_channels=1).to(device)
        state  = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        z_t = torch.from_numpy(z).unsqueeze(0).float().to(device)  # [1, 64, 256]
        z_t.requires_grad_(True)
        t   = torch.tensor([0.001], dtype=torch.float32, device=device)

        pred = model(z_t, t)           # [1, 1, 256]
        loss = pred.mean()
        loss.backward()

        grad = z_t.grad.squeeze(0).detach().numpy()   # [64, 256]
        norm = float(np.linalg.norm(grad) + 1e-12)
        return (z + strength * grad / norm).astype(np.float32)

    except Exception as e:
        import warnings
        warnings.warn(f"LatCH guidance failed for {feature}: {e}. Using fallback.")
        return _correlation_fallback(z, feature, strength)


def _correlation_fallback(
    z: np.ndarray, feature: str, strength: float
) -> np.ndarray:
    """
    Simple fallback: build a guidance direction from top-correlated channels
    and add strength × direction_vector to every frame.
    latent_correlations.json format:
      {feature: {top_positive_channels: [...], top_positive_scores: [...],
                 top_negative_channels: [...], top_negative_scores: [...]}}
    If the feature is not found, returns z unchanged.
    """
    corr = _load_correlations()
    if feature not in corr:
        return z.copy()
    entry = corr[feature]

    # Support two formats:
    # (a) flat list of 64 floats
    if isinstance(entry, list):
        vec = np.array(entry, dtype=np.float32)
        if vec.shape != (64,):
            return z.copy()
        return (z + strength * vec[:, np.newaxis]).astype(np.float32)

    # (b) dict with top_positive_channels / top_negative_channels
    if isinstance(entry, dict):
        vec = np.zeros(64, dtype=np.float32)
        pos_ch = entry.get("top_positive_channels", [])
        pos_sc = entry.get("top_positive_scores", [1.0] * len(pos_ch))
        neg_ch = entry.get("top_negative_channels", [])
        neg_sc = entry.get("top_negative_scores", [1.0] * len(neg_ch))
        for ch, sc in zip(pos_ch, pos_sc):
            if 0 <= ch < 64:
                vec[ch] += float(sc)
        for ch, sc in zip(neg_ch, neg_sc):
            if 0 <= ch < 64:
                vec[ch] -= float(sc)
        norm = float(np.linalg.norm(vec) + 1e-12)
        return (z + strength * (vec / norm)[:, np.newaxis]).astype(np.float32)

    return z.copy()
