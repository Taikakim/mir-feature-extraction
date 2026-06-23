"""URL builders + reachability check for the SAME decode player.

Defaults to the low-VRAM ONNX server `latent_server_onnx.py` (port 7893): ~2 GB VRAM,
so it can decode/mix/source alongside a running training job without OOM. It has
**no /steer** (LatCH steering needs the torch DiT/heads). For steering, set
`SA3_PLAYER_PORT=7892` to use the torch player `latent_server_sa3.py` (full incl.
/steer); `SA3_PLAYER_BASE` overrides the whole base URL. NB: the ONNX server does a
~9-min MIGraphX compile at boot — start it once, long-lived.
"""
from __future__ import annotations
import os
from urllib.parse import urlencode

BASE = os.environ.get(
    "SA3_PLAYER_BASE",
    f"http://localhost:{os.environ.get('SA3_PLAYER_PORT', '7893')}")


def decode_url(crop_id: str) -> str:
    return f"{BASE}/decode?{urlencode({'crop': crop_id})}"


def source_url(crop_id: str) -> str:
    return f"{BASE}/source?{urlencode({'crop': crop_id})}"


def mix_url(crop_a: str, crop_b: str, t: float = 0.5,
            interp: str = "slerp") -> str:
    q = urlencode({"crop_a": crop_a, "crop_b": crop_b,
                   "t": f"{float(t):.3f}", "interp": interp})
    return f"{BASE}/mix?{q}"


def steer_url(crop_id: str, head: str, gain: float = 48.0) -> str:
    q = urlencode({"crop": crop_id, "head": head, "gain": f"{float(gain)}"})
    return f"{BASE}/steer?{q}"


def status(timeout: float = 0.5) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"{BASE}/status", timeout=timeout)
        return True
    except Exception:
        return False
