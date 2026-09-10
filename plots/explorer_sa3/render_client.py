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
          root_ids: list[str] | None = None, timeout: float = 60.0) -> dict | None:
    """GET /ckpts — checkpoint journal (recursive *.ckpt / *.safetensors scan
    of the server-side target folder, default <eval-drive>/sa3_lora_runs (resolved server-side; the drive is removable and mounts as Mantu or Mantu1),
    cached to a json journal keyed on mtime+size).

    Params: rescan=1 forces a fresh walk; root overrides the single scan folder;
    root_ids=["lumi_uuid", ...] returns the MERGED listing across those configured
    roots, each entry tagged with root_id/family/label/epoch/step/rank/corpus/verdict.
    Response: {"root": str, "ckpts": [{"path": str, "size": int, "mtime": float},
    ...]}. Returns None when the server is unreachable."""
    params: dict = {"rescan": int(bool(rescan))}
    if root:
        params["root"] = root
    if root_ids:
        params["root_ids"] = ",".join(root_ids)
    try:
        r = requests.get(f"{BASE}/ckpts", params=params, timeout=timeout)
        # A 404 here is not "server down" -- it is the server telling us the scan
        # root is an unmounted removable drive, and it names which. Pass that body
        # through so the picker can say so instead of blaming the connection.
        if r.status_code == 404:
            try:
                return r.json()
            except ValueError:
                pass
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def slots(timeout: float = 10.0) -> dict | None:
    """GET /slots — the resident adapter table + VRAM budget."""
    try:
        r = requests.get(f"{BASE}/slots", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def set_slots(specs: list[dict], activate: int | None = None,
              timeout: float = 300.0) -> dict:
    """POST /slots — declare the resident adapter SET.

    specs: [{"ckpt_path": str, "label": str?}]. Changing the set costs one
    remove+reload; switching between resident slots afterwards costs nothing
    (measured: 1.7 s per A/B render vs a multi-GB reload). Raises RenderError with
    the server's own reason on refusal — over the VRAM floor, over max_slots, or a
    fullft path, which replaces the backbone rather than augmenting it."""
    body = {"slots": specs}
    if activate is not None:
        body["activate"] = int(activate)
    try:
        r = requests.post(f"{BASE}/slots", json=body, timeout=timeout)
    except Exception as e:
        raise RenderError(f"render server unreachable: {e}") from e
    j = r.json() if r.content else {}
    if r.status_code >= 400 or not j.get("ok"):
        raise RenderError(j.get("reason") or j.get("error")
                          or j.get("detail") or f"HTTP {r.status_code}")
    return j


def ab(payload: dict, timeout: float = 7200.0) -> dict:
    """POST /ab — one payload rendered across resident slots, ONE shared seed, so
    the arms differ only by the model. `null` in ab.slots is the bare base: the
    control arm, and the one you actually need."""
    try:
        r = requests.post(f"{BASE}/ab", json=payload, timeout=timeout)
    except Exception as e:
        raise RenderError(f"render server unreachable: {e}") from e
    j = r.json() if r.content else {}
    if r.status_code >= 400 or not j.get("ok"):
        raise RenderError(j.get("reason") or j.get("error")
                          or j.get("detail") or f"HTTP {r.status_code}")
    return j


def presets_list(timeout: float = 5.0) -> list[dict] | None:
    """GET /presets — named render recipes, newest first. None when unreachable."""
    try:
        r = requests.get(f"{BASE}/presets", timeout=timeout)
        r.raise_for_status()
        return r.json().get("presets", [])
    except Exception:
        return None


def preset_load(name: str, timeout: float = 5.0) -> dict | None:
    """GET /presets/<name> — {"schema","name","notes","created","payload",
    "form"?}. `form` is the viewer snapshot; presets written by a CLI have none."""
    try:
        r = requests.get(f"{BASE}/presets/{name}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def preset_save(name: str, payload: dict, notes: str = "",
                form: dict | None = None, timeout: float = 10.0) -> dict | None:
    """POST /presets. seed/batch_size are stripped server-side: a preset is a
    recipe, and a pinned seed would make a sweep's seed axis a silent no-op."""
    body = {"name": name, "payload": payload, "notes": notes}
    if form:
        body["form"] = form
    try:
        r = requests.post(f"{BASE}/presets", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def roots(timeout: float = 10.0) -> dict | None:
    """GET /roots — the configured checkpoint roots plus live availability.

    Response: {"ok": True, "roots": [{"id","label","path","available","count",
    ...}], "stale_root_ids": [...]}. A removable drive that is not mounted comes
    back available=false rather than as an error, so the picker can grey it out
    instead of losing its entries. None when the server is unreachable."""
    try:
        r = requests.get(f"{BASE}/roots", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def models(root_ids: list[str] | str | None = None, family: str | None = None,
           corpus: str | None = None, q: str | None = None,
           loadable: bool | None = None, rescan: bool = False,
           timeout: float = 60.0) -> dict | None:
    """GET /models — the model database across every configured root.

    family is one of ckpt_probe.FAMILIES: adapter | fullft | control_adapter |
    latch_head | unknown. Response: {"ok": True, "count": int, "models":
    [{"id","path","label","family","corpus","loadable","root_id",...}]}.
    None when the server is unreachable."""
    params: dict = {"rescan": int(bool(rescan))}
    if root_ids:
        params["root_ids"] = (root_ids if isinstance(root_ids, str)
                              else ",".join(root_ids))
    for k, v in (("family", family), ("corpus", corpus), ("q", q)):
        if v:
            params[k] = v
    if loadable is not None:
        params["loadable"] = int(bool(loadable))
    try:
        r = requests.get(f"{BASE}/models", params=params, timeout=timeout)
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
