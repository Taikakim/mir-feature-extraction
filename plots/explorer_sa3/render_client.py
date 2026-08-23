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

_OPS = {"generate", "a2a_track", "a2a_mix", "decode", "longform", "bend"}


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


def longform(payload: dict, timeout: float = 7200.0) -> dict:
    """POST /longform — prompt-ARC rendering (steered_longform arc grammar
    '0:promptA|45:promptB|...'). With `audio_path` the server runs the windowed
    a2a arc loop; without it the t2a LongFormRenderer path (`duration`).
    Longer default timeout: many windows per job."""
    return render("longform", payload, timeout=timeout)


def bend(payload: dict, timeout: float = 600.0) -> dict:
    """POST /bend — latent data-bending then decode. Payload:
    {"crop_id" | "latent_path", "latent_dir": optional, "seed": int,
    "ops": [{"op": ..., "amount": ..., + op keys}, ...]}
    (eval/latent_bend.py spec grammar, mirrors weight_mutations)."""
    return render("bend", payload, timeout=timeout)


def ckpts(rescan: bool = False, root: str | None = None,
          timeout: float = 60.0) -> dict | None:
    """GET /ckpts — checkpoint journal (recursive *.ckpt / *.safetensors scan
    of the server-side target folder, default <eval-drive>/sa3_lora_runs (resolved server-side; the drive is removable and mounts as Mantu or Mantu1),
    cached to a json journal keyed on mtime+size).

    Params: rescan=1 forces a fresh walk; root overrides the scan folder.
    Response: {"root": str, "ckpts": [{"path": str, "size": int, "mtime": float},
    ...]}. Returns None when the server is unreachable."""
    params: dict = {"rescan": int(bool(rescan))}
    if root:
        params["root"] = root
    try:
        r = requests.get(f"{BASE}/ckpts", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


_SCHED_CACHE: dict[tuple, list[float]] = {}


def schedule(steps: int, duration: float,
             dist_shift: float | str | None = None,
             sigma_max: float = 1.0, timeout: float = 10.0) -> list[float] | None:
    """POST /schedule — the real run sigma schedule (build_schedule with the
    model's sampling_dist_shift unless dist_shift is given; length steps+1,
    descending sigma_max→0).

    Payload: {"steps": int, "duration": float (seconds; dist_shift is
    length-dependent), "dist_shift": float|"flux"|null (null = model default,
    "flux" = stock length-dependent FluxDistributionShift, float = constant
    alpha), "sigma_max": float}. Response: {"sigmas": [float, ...]}.
    Memoized per argument tuple; None when the server is unreachable (callers
    should fall back to a linear ramp and say so)."""
    if dist_shift is not None and not isinstance(dist_shift, str):
        dist_shift = float(dist_shift)
    key = (int(steps), float(duration), dist_shift, float(sigma_max))
    if key in _SCHED_CACHE:
        return _SCHED_CACHE[key]
    try:
        r = requests.post(f"{BASE}/schedule", json={
            "steps": int(steps), "duration": float(duration),
            "dist_shift": dist_shift, "sigma_max": float(sigma_max),
        }, timeout=timeout)
        r.raise_for_status()
        sig = [float(s) for s in r.json()["sigmas"]]
    except Exception:
        return None
    if len(_SCHED_CACHE) > 256:
        _SCHED_CACHE.clear()
    _SCHED_CACHE[key] = sig
    return sig
