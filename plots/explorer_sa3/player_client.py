"""URL builders + reachability check for the SAME decode player.

Defaults to the RENDER SERVER (port 8056, `SAO/eval/explorer_render_server.py`),
which already holds SAME-L resident as `MODEL.model.pretransform`. Before
2026-08-25 this pointed at a standalone player that loaded a *second* copy of
the same weights (7.12 GB on a 16 GB card); those GET endpoints (`/decode
/source /mix /steer /crops /meta`) were moved onto the render server unchanged,
so only this base URL differs.

Overrides: `SA3_PLAYER_PORT=7893` selects the low-VRAM ONNX player
`latent_server_onnx.py` (~2 GB, decodes alongside a training job, but has **no
/steer** -- LatCH steering needs the torch heads; NB ~9-min MIGraphX compile at
boot, so start it once, long-lived). `SA3_PLAYER_BASE` overrides the whole URL.
"""
from __future__ import annotations
import os
from urllib.parse import urlencode

BASE = os.environ.get(
    "SA3_PLAYER_BASE",
    f"http://localhost:{os.environ.get('SA3_PLAYER_PORT', '8056')}")


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
