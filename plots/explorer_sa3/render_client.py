"""HTTP client for the SA3 explorer render server (SAO/.venv, port 8056).

Frozen contract (design B.1): `status`, `info`, `render`, `audio_url`,
`RenderError`, `BASE`, `LAUNCH_HINT`. The server holds medium-base resident
and exposes /generate, /a2a_track, /a2a_mix, /decode plus /info /status
/audio; see SAO/eval/explorer_render_server.py.
"""
from __future__ import annotations
import configparser
import os
from pathlib import Path

import requests

INI = Path(__file__).parent.parent.parent / "latent_player_sa3.ini"

LAUNCH_HINT = (
    "cd /home/kim/Projects/SAO && "
    "FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE PYTORCH_TUNABLEOP_ENABLED=0 "
    "MIOPEN_FIND_MODE=2 \\\n"
    "  .venv/bin/python eval/explorer_render_server.py --port 8056"
)

_OPS = {"generate", "a2a_track", "a2a_mix", "decode"}


def _base_url() -> str:
    env = os.environ.get("SA3_RENDER_BASE")
    if env:
        return env.rstrip("/")
    try:
        cfg = configparser.ConfigParser()
        cfg.read(INI)
        if cfg.has_option("render_server", "base_url"):
            return cfg["render_server"]["base_url"].rstrip("/")
    except Exception:
        pass
    return "http://localhost:8056"


BASE: str = _base_url()


class RenderError(RuntimeError):
    """Render server failure; args[0] is a human-readable message
    (includes the server traceback when one was returned)."""


def status(timeout: float = 1.0) -> dict | None:
    """GET /status; None if unreachable."""
    try:
        r = requests.get(f"{BASE}/status", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


_INFO_CACHE: dict | None = None


def info(timeout: float = 5.0) -> dict | None:
    """GET /info; memoized after first success. None if unreachable."""
    global _INFO_CACHE
    if _INFO_CACHE is not None:
        return _INFO_CACHE
    try:
        r = requests.get(f"{BASE}/info", timeout=timeout)
        r.raise_for_status()
        _INFO_CACHE = r.json()
        return _INFO_CACHE
    except Exception:
        return None


def render(op: str, payload: dict, timeout: float = 3600.0) -> dict:
    """POST {BASE}/{op}. Raises RenderError on non-200 or {"error": ...}."""
    if op not in _OPS:
        raise RenderError(f"unknown render op {op!r} (must be one of {sorted(_OPS)})")
    try:
        r = requests.post(f"{BASE}/{op}", json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise RenderError(f"/{op}: server unreachable ({e})") from e
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code != 200 or (isinstance(body, dict) and body.get("error")):
        msg = f"/{op} failed (HTTP {r.status_code})"
        if isinstance(body, dict):
            if body.get("error"):
                msg += f": {body['error']}"
            if body.get("traceback"):
                msg += f"\n{body['traceback']}"
        else:
            msg += f": {r.text[:2000]}"
        raise RenderError(msg)
    return body


def audio_url(rel: str) -> str:
    """BASE + rel (rel is an entry from resp['urls'], e.g. /audio/<job>/x.wav)."""
    return BASE + rel
