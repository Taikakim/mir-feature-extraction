"""URL builders + reachability check for latent_server_sa3.py (port 7892)."""
from __future__ import annotations
from urllib.parse import urlencode

BASE = "http://localhost:7892"


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
