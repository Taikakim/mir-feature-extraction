"""URL builders and HTTP helpers for latent_server.py (port 7891)."""
from __future__ import annotations
from urllib.parse import urlencode

_BASE = "http://localhost:7891"


def build_decode_url(track: str, position: str = "0.5",
                     smart_loop: bool = False,
                     interp: str = "slerp",
                     manip: dict | None = None) -> str:
    """Build URL for /decode endpoint."""
    params: dict = {"track": track, "position": position, "interp": interp}
    if smart_loop:
        params["smart_loop"] = "1"
    if manip:
        for k, v in manip.items():
            params[k] = v
    return f"{_BASE}/decode?{urlencode(params)}"


def build_crossfade_url(track_a: str, pos_a: str,
                        track_b: str, pos_b: str,
                        mix: float = 0.5,
                        interp: str = "slerp",
                        smart_loop: bool = False,
                        manip: dict | None = None) -> str:
    """Build URL for /crossfade endpoint."""
    params: dict = {
        "track_a": track_a, "position_a": pos_a,
        "track_b": track_b, "position_b": pos_b,
        "mix": f"{float(mix):.3f}",
        "interp": interp,
    }
    if smart_loop:
        params["smart_loop"] = "1"
    if manip:
        params.update(manip)
    return f"{_BASE}/crossfade?{urlencode(params)}"


def build_average_url(track_a: str,
                      track_b: str | None = None,
                      mix: float = 0.0,
                      interp: str = "slerp",
                      smart_loop: bool = False) -> str:
    """Build URL for /average endpoint."""
    params: dict = {"track": track_a, "interp": interp}
    if track_b:
        params["track_b"] = track_b
        params["mix"]     = f"{float(mix):.3f}"
    if smart_loop:
        params["smart_loop"] = "1"
    return f"{_BASE}/average?{urlencode(params)}"


def check_server_alive(port: int = 7891, timeout: float = 0.5) -> bool:
    """Return True if the latent server is reachable."""
    import urllib.request
    try:
        urllib.request.urlopen(f"http://localhost:{port}/", timeout=timeout)
        return True
    except Exception:
        return False
