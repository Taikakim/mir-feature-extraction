# SA3 Latent Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an SA3-only latent viewer (`plots/explorer_sa3/`) + decode player (`scripts/latent_server_sa3.py`) for the 256-dim / T=4096 SAME-L latents, leaving the old 64-dim Small tools on `main` untouched.

**Architecture:** Two processes. A Dash **viewer** (mir venv, py3.12) reads `.json`/`.TIMESERIES.npz` sidecars directly and builds HTTP URLs. A **player** (SA3 venv, py3.13) owns the SAME-L VAE + LatCH heads and serves WAV for decode/source/mix/steer. They communicate over HTTP only.

**Tech Stack:** Python, Dash/Plotly, numpy/pandas (viewer); `stable_audio_3.AutoencoderModel` + `stdlib http.server` + torch/ROCm (player); pytest.

## Global Constraints

- **SA3-only.** Do NOT modify `plots/explorer/` or `scripts/latent_server.py` (the 64-dim reference). All new code in `plots/explorer_sa3/` + `scripts/latent_server_sa3.py`.
- **Latent geometry:** `[256, 4096]` float16 on disk; frame rate 44100/4096 = 10.767 Hz.
- **Sidecars are the only feature source.** No external DB; no precompute NPZ artifacts.
- **Player venv:** `/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python` (py3.13). **Viewer/test venv:** `/home/kim/Projects/mir/mir/bin/python` (py3.12).
- **Latent dir default:** `/home/kim/Projects/latents_sa3`. **Player port:** 7892.
- **Model:** `AutoencoderModel.from_pretrained("same-l")`, decode via `.decode(latents, chunked=True, chunk_size=128, overlap=32)`.
- **No `position` field** in sidecars — use `relative_position_start`/`relative_position_end`.
- Tests that need torch/the model are `pytest.skip`-guarded so the mir-venv suite stays green.

---

### Task 1: Sidecar index (scan / search / group)

**Files:**
- Create: `plots/explorer_sa3/__init__.py` (empty)
- Create: `plots/explorer_sa3/sidecar_index.py`
- Test: `tests/explorer_sa3/__init__.py` (empty), `tests/explorer_sa3/test_sidecar_index.py`

**Interfaces:**
- Produces:
  - `scan_index(latent_dir: Path) -> list[CropMeta]`
  - `@dataclass CropMeta` with fields `id: str, source_track: str, artist: str, title: str, prompt: str, bpm: float|None, lufs: float|None, rel_pos: float`
  - `search(index: list[CropMeta], query: str) -> list[CropMeta]`
  - `group_by_track(index: list[CropMeta]) -> dict[str, list[CropMeta]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/explorer_sa3/test_sidecar_index.py
import json
from pathlib import Path
from plots.explorer_sa3.sidecar_index import scan_index, search, group_by_track


def _write_crop(d: Path, cid: str, **over):
    base = {
        "source_track": "AZukx - Earth Chakra",
        "track_metadata_artist": "AZukx", "track_metadata_title": "Earth Chakra",
        "prompt": "earth chakra, 1996, 123", "bpm_madmom": 123.0, "lufs": -9.5,
        "relative_position_start": 0.1, "relative_position_end": 0.2,
    }
    base.update(over)
    (d / f"{cid}.json").write_text(json.dumps(base))
    (d / f"{cid}.npy").write_bytes(b"")  # presence only


def test_scan_groups_and_searches(tmp_path):
    _write_crop(tmp_path, "000000")
    _write_crop(tmp_path, "000001", source_track="Other - Song",
                track_metadata_artist="Other", track_metadata_title="Song",
                prompt="dark forest")
    idx = scan_index(tmp_path)
    assert len(idx) == 2
    assert idx[0].id == "000000" and abs(idx[0].rel_pos - 0.1) < 1e-9
    assert idx[0].bpm == 123.0
    g = group_by_track(idx)
    assert set(g) == {"AZukx - Earth Chakra", "Other - Song"}
    hits = search(idx, "forest")
    assert len(hits) == 1 and hits[0].id == "000001"


def test_scan_skips_crop_with_unreadable_json(tmp_path):
    _write_crop(tmp_path, "000000")
    (tmp_path / "000001.json").write_text("{not valid")
    (tmp_path / "000001.npy").write_bytes(b"")
    idx = scan_index(tmp_path)
    assert [c.id for c in idx] == ["000000"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_sidecar_index.py -v`
Expected: FAIL (ModuleNotFoundError: plots.explorer_sa3.sidecar_index)

- [ ] **Step 3: Write minimal implementation**

```python
# plots/explorer_sa3/sidecar_index.py
"""Scan latents_sa3 .json sidecars into an in-memory, searchable index."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CropMeta:
    id: str
    source_track: str
    artist: str
    title: str
    prompt: str
    bpm: float | None
    lufs: float | None
    rel_pos: float


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def scan_index(latent_dir: Path) -> list[CropMeta]:
    """Scan every NNNNNN.json beside an NNNNNN.npy. Skip unreadable sidecars."""
    latent_dir = Path(latent_dir)
    out: list[CropMeta] = []
    for jp in sorted(latent_dir.glob("*.json")):
        if jp.name.endswith(".TIMESERIES.json"):
            continue
        if not jp.with_suffix(".npy").exists():
            continue
        try:
            m = json.loads(jp.read_text())
        except Exception:
            continue
        out.append(CropMeta(
            id=jp.stem,
            source_track=str(m.get("source_track", "")),
            artist=str(m.get("track_metadata_artist", "")),
            title=str(m.get("track_metadata_title", "")),
            prompt=str(m.get("prompt", "")),
            bpm=_to_float(m.get("bpm_madmom") or m.get("bpm_essentia")),
            lufs=_to_float(m.get("lufs")),
            rel_pos=_to_float(m.get("relative_position_start")) or 0.0,
        ))
    return out


def search(index: list[CropMeta], query: str) -> list[CropMeta]:
    if not query:
        return list(index)
    q = query.lower()
    return [c for c in index
            if q in c.artist.lower() or q in c.title.lower()
            or q in c.prompt.lower() or q in c.source_track.lower()]


def group_by_track(index: list[CropMeta]) -> dict[str, list[CropMeta]]:
    g: dict[str, list[CropMeta]] = {}
    for c in index:
        g.setdefault(c.source_track, []).append(c)
    return g
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_sidecar_index.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add plots/explorer_sa3/__init__.py plots/explorer_sa3/sidecar_index.py tests/explorer_sa3/__init__.py tests/explorer_sa3/test_sidecar_index.py
git commit -m "feat(sa3-explorer): sidecar index scan/search/group"
```

---

### Task 2: Latent + timeseries loaders

**Files:**
- Create: `plots/explorer_sa3/latents.py`
- Test: `tests/explorer_sa3/test_latents.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `load_latent(latent_dir: Path, crop_id: str) -> np.ndarray` → `float32 [256, T]`
  - `content_frames(meta: dict) -> int` (sum of `padding_mask`, full T if absent)
  - `load_timeseries(latent_dir: Path, crop_id: str) -> dict[str, np.ndarray]`
  - `FRAME_RATE_HZ = 44100 / 4096`

- [ ] **Step 1: Write the failing test**

```python
# tests/explorer_sa3/test_latents.py
import numpy as np
from plots.explorer_sa3.latents import (
    load_latent, content_frames, load_timeseries, FRAME_RATE_HZ,
)


def test_load_latent_fp16_to_fp32_and_shape(tmp_path):
    z = (np.random.randn(256, 4096)).astype(np.float16)
    np.save(tmp_path / "000000.npy", z)
    out = load_latent(tmp_path, "000000")
    assert out.dtype == np.float32 and out.shape == (256, 4096)
    assert np.allclose(out, z.astype(np.float32))


def test_load_latent_squeezes_batch_dim(tmp_path):
    z = np.zeros((1, 256, 8), dtype=np.float16)
    np.save(tmp_path / "000001.npy", z)
    assert load_latent(tmp_path, "000001").shape == (256, 8)


def test_load_latent_rejects_wrong_dims(tmp_path):
    np.save(tmp_path / "000002.npy", np.zeros((64, 256), dtype=np.float16))
    try:
        load_latent(tmp_path, "000002")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_content_frames():
    assert content_frames({"padding_mask": [1, 1, 1, 0, 0]}) == 3
    assert content_frames({}) == 4096


def test_load_timeseries(tmp_path):
    np.savez(tmp_path / "000000.TIMESERIES.npz",
             rms_energy_bass_ts=np.zeros(4096, np.float32),
             hpcp_ts=np.zeros((4096, 12), np.float32))
    ts = load_timeseries(tmp_path, "000000")
    assert ts["rms_energy_bass_ts"].shape == (4096,)
    assert ts["hpcp_ts"].shape == (4096, 12)


def test_frame_rate():
    assert abs(FRAME_RATE_HZ - 10.7666) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_latents.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# plots/explorer_sa3/latents.py
"""Pure loaders for SA3 latents and their TIMESERIES sidecars."""
from __future__ import annotations
from pathlib import Path
import numpy as np

LATENT_DIM = 256
N_FRAMES = 4096
FRAME_RATE_HZ = 44100 / 4096   # 10.7666 Hz


def load_latent(latent_dir: Path, crop_id: str) -> np.ndarray:
    """Load NNNNNN.npy as float32 [256, T]. Squeezes a leading batch dim."""
    arr = np.load(Path(latent_dir) / f"{crop_id}.npy").astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] != LATENT_DIM:
        raise ValueError(
            f"{crop_id}: expected [{LATENT_DIM}, T], got {arr.shape}")
    return arr


def content_frames(meta: dict) -> int:
    """Number of non-padding frames from padding_mask; full T if absent."""
    mask = meta.get("padding_mask")
    if not mask:
        return N_FRAMES
    return int(sum(mask))


def load_timeseries(latent_dir: Path, crop_id: str) -> dict[str, np.ndarray]:
    p = Path(latent_dir) / f"{crop_id}.TIMESERIES.npz"
    with np.load(p) as z:
        return {k: z[k] for k in z.files}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_latents.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add plots/explorer_sa3/latents.py tests/explorer_sa3/test_latents.py
git commit -m "feat(sa3-explorer): latent + timeseries loaders"
```

---

### Task 3: Player client (URL builders + status)

**Files:**
- Create: `plots/explorer_sa3/player_client.py`
- Test: `tests/explorer_sa3/test_player_client.py`

**Interfaces:**
- Produces (base URL default `http://localhost:7892`):
  - `decode_url(crop_id) -> str`
  - `source_url(crop_id) -> str`
  - `mix_url(crop_a, crop_b, t=0.5, interp="slerp") -> str`
  - `steer_url(crop_id, head, gain=48.0) -> str`
  - `status(timeout=0.5) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/explorer_sa3/test_player_client.py
from plots.explorer_sa3 import player_client as pc


def test_decode_and_source_urls():
    assert pc.decode_url("000007") == "http://localhost:7892/decode?crop=000007"
    assert pc.source_url("000007") == "http://localhost:7892/source?crop=000007"


def test_mix_url():
    u = pc.mix_url("000001", "000002", t=0.25, interp="lerp")
    assert u == ("http://localhost:7892/mix?"
                 "crop_a=000001&crop_b=000002&t=0.250&interp=lerp")


def test_steer_url():
    u = pc.steer_url("000003", "hpcp", gain=64)
    assert u == ("http://localhost:7892/steer?"
                 "crop=000003&head=hpcp&gain=64.0")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_player_client.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# plots/explorer_sa3/player_client.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_player_client.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add plots/explorer_sa3/player_client.py tests/explorer_sa3/test_player_client.py
git commit -m "feat(sa3-explorer): player HTTP URL client"
```

---

### Task 4: Analysis helpers (PCA / dim-xcorr / dim↔feature corr)

**Files:**
- Create: `plots/explorer_sa3/analysis.py`
- Test: `tests/explorer_sa3/test_analysis.py`

**Interfaces:**
- Consumes: nothing.
- Produces (all pure numpy, operate on already-loaded arrays):
  - `pca_frames(latents: list[np.ndarray], k=3) -> tuple[np.ndarray, np.ndarray]` → `(components [k,256], explained_variance_ratio [k])`
  - `dim_xcorr(latents: list[np.ndarray]) -> np.ndarray` → `[256,256]`
  - `dim_feature_corr(latents: list[np.ndarray], feature_ts: list[np.ndarray]) -> np.ndarray` → `[256]` Pearson r of each latent dim vs a per-frame feature, pooled across crops

- [ ] **Step 1: Write the failing test**

```python
# tests/explorer_sa3/test_analysis.py
import numpy as np
from plots.explorer_sa3.analysis import pca_frames, dim_xcorr, dim_feature_corr


def _rng():
    return np.random.default_rng(0)


def test_pca_shapes_and_variance_order():
    r = _rng()
    lats = [r.standard_normal((256, 64)).astype(np.float32) for _ in range(5)]
    comps, evr = pca_frames(lats, k=3)
    assert comps.shape == (3, 256) and evr.shape == (3,)
    assert evr[0] >= evr[1] >= evr[2]


def test_dim_xcorr_shape_and_diag():
    r = _rng()
    lats = [r.standard_normal((256, 64)).astype(np.float32) for _ in range(3)]
    c = dim_xcorr(lats)
    assert c.shape == (256, 256)
    assert np.allclose(np.diag(c), 1.0, atol=1e-5)


def test_dim_feature_corr_detects_link():
    r = _rng()
    feat = r.standard_normal(128).astype(np.float32)
    z = r.standard_normal((256, 128)).astype(np.float32)
    z[7] = feat * 3.0  # dim 7 perfectly correlated with the feature
    corr = dim_feature_corr([z], [feat])
    assert corr.shape == (256,)
    assert abs(corr[7]) > 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_analysis.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# plots/explorer_sa3/analysis.py
"""Live, on-demand latent analysis over a sampled subset of crops.

All functions take already-loaded latents [256, T] (variable T) and pool
frames across crops. No precompute, no disk artifacts.
"""
from __future__ import annotations
import numpy as np


def _stack_frames(latents: list[np.ndarray]) -> np.ndarray:
    """Concatenate all crops' frames → [n_frames_total, 256]."""
    return np.concatenate([z.T for z in latents], axis=0).astype(np.float64)


def pca_frames(latents: list[np.ndarray], k: int = 3):
    X = _stack_frames(latents)
    X = X - X.mean(axis=0, keepdims=True)
    # SVD of centered frames; components are right-singular vectors
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = (S ** 2) / max(1, X.shape[0] - 1)
    evr = var / var.sum()
    return Vt[:k].astype(np.float32), evr[:k].astype(np.float32)


def dim_xcorr(latents: list[np.ndarray]) -> np.ndarray:
    X = _stack_frames(latents)            # [N, 256]
    return np.corrcoef(X, rowvar=False).astype(np.float32)


def dim_feature_corr(latents: list[np.ndarray],
                     feature_ts: list[np.ndarray]) -> np.ndarray:
    """Per-dim Pearson r vs a per-frame feature, pooled across crops."""
    dim_cols, feat_vals = [], []
    for z, f in zip(latents, feature_ts):
        T = min(z.shape[1], len(f))
        dim_cols.append(z[:, :T].T)        # [T, 256]
        feat_vals.append(np.asarray(f[:T], dtype=np.float64))
    X = np.concatenate(dim_cols, axis=0).astype(np.float64)   # [N, 256]
    fv = np.concatenate(feat_vals, axis=0)                    # [N]
    Xc = X - X.mean(axis=0, keepdims=True)
    fc = fv - fv.mean()
    num = Xc.T @ fc
    den = np.sqrt((Xc ** 2).sum(axis=0) * (fc ** 2).sum())
    den = np.where(den < 1e-12, np.nan, den)
    return (num / den).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_analysis.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add plots/explorer_sa3/analysis.py tests/explorer_sa3/test_analysis.py
git commit -m "feat(sa3-explorer): live latent analysis helpers"
```

---

### Task 5: Player server — decode + source + meta

**Files:**
- Create: `scripts/latent_server_sa3.py`
- Create: `latent_player_sa3.ini`
- Test: `tests/explorer_sa3/test_player_smoke.py`

**Interfaces:**
- Consumes: `stable_audio_3.AutoencoderModel`; sidecar fields `source_path`, `start_sample`, `end_sample`, `padding_mask`.
- Produces a running HTTP server on port 7892 with `/status`, `/crops`, `/meta?crop=`, `/decode?crop=`, `/source?crop=`. WAV is stereo int16 at the model sample rate.
- Helper (module-level, importable for tests): `wav_bytes(audio_f32_stereo: np.ndarray, sr: int) -> bytes`.

- [ ] **Step 1: Write the failing test** (pure helper + import guard; full server is smoke-tested manually)

```python
# tests/explorer_sa3/test_player_smoke.py
import importlib.util
import wave, io
from pathlib import Path
import numpy as np
import pytest

SERVER = Path(__file__).resolve().parents[2] / "scripts" / "latent_server_sa3.py"


def _load_module():
    # Import the script without running main(); stable_audio_3/torch may be absent
    spec = importlib.util.spec_from_file_location("latent_server_sa3", SERVER)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # missing torch/stable_audio_3 in mir venv
        pytest.skip(f"player deps unavailable: {e}")
    return mod


def test_wav_bytes_roundtrip():
    mod = _load_module()
    audio = np.zeros((2, 100), dtype=np.float32)
    audio[0, 10] = 0.5
    raw = mod.wav_bytes(audio, 44100)
    with wave.open(io.BytesIO(raw)) as w:
        assert w.getnchannels() == 2
        assert w.getframerate() == 44100
        assert w.getnframes() == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_player_smoke.py -v`
Expected: FAIL (file scripts/latent_server_sa3.py does not exist → skip or error). Acceptable transition state: it errors on missing file.

- [ ] **Step 3: Write the config + minimal implementation**

```ini
# latent_player_sa3.ini
[server]
latent_dir = /home/kim/Projects/latents_sa3
port = 7892

[model]
model = same-l
latch_weights_dir = /home/kim/Projects/SAO/stable-audio-3/latch_weights_sa3
device = cuda
model_half = true
chunk_size = 128
overlap = 32
```

```python
#!/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python
"""latent_server_sa3.py — decode SAME-L (256-dim, T=4096) latents to WAV.

Runs under the SA3 venv (py3.13). Endpoints: /status /crops /meta /decode
/source. /mix and /steer are added in later tasks.
"""
from __future__ import annotations
import argparse, configparser, io, json, threading, wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

DEFAULT_INI = Path(__file__).parent.parent / "latent_player_sa3.ini"

# Globals set in main(); read-only afterwards.
_ae = None
_sr = 44100
_latent_dir: Path | None = None
_cfg: dict = {}
_lock = threading.Lock()


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """audio [2, samples] float32 → stereo int16 WAV bytes."""
    a = np.clip(audio, -1.0, 1.0)
    i16 = (a * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(i16.T.flatten().tobytes())
    return buf.getvalue()


def _read_meta(crop_id: str) -> dict:
    return json.loads((_latent_dir / f"{crop_id}.json").read_text())


def _decode_latent(crop_id: str) -> bytes:
    import torch
    arr = np.load(_latent_dir / f"{crop_id}.npy").astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    meta = _read_meta(crop_id)
    z = torch.from_numpy(arr).unsqueeze(0).to(_ae_device())
    with torch.no_grad():
        audio = _ae.decode(z, chunked=True,
                           chunk_size=int(_cfg["chunk_size"]),
                           overlap=int(_cfg["overlap"]))
    audio_np = audio.squeeze(0).cpu().float().numpy()   # [2, samples]
    n_content = int(sum(meta.get("padding_mask") or [])) or arr.shape[1]
    samples = n_content * 4096
    if 0 < samples < audio_np.shape[1]:
        audio_np = audio_np[:, :samples]
    return wav_bytes(audio_np, _sr)


def _source_slice(crop_id: str) -> bytes:
    import soundfile as sf
    meta = _read_meta(crop_id)
    src = meta["source_path"]
    s, e = int(meta["start_sample"]), int(meta["end_sample"])
    audio, sr = sf.read(src, dtype="float32", always_2d=True, start=s, stop=e)
    audio = audio.T
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return wav_bytes(audio, sr)


def _ae_device():
    import torch
    return torch.device(_cfg.get("device", "cuda")
                        if __import__("torch").cuda.is_available() else "cpu")


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _wav(self, raw: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/status":
                return self._json({"ok": True, "model": _cfg.get("model"),
                                   "sample_rate": _sr,
                                   "latent_dir": str(_latent_dir)})
            if u.path == "/crops":
                ids = sorted(p.stem for p in _latent_dir.glob("*.npy"))
                return self._json(ids)
            if u.path == "/meta":
                return self._json(_read_meta(q["crop"]))
            if u.path == "/decode":
                with _lock:
                    return self._wav(_decode_latent(q["crop"]))
            if u.path == "/source":
                return self._wav(_source_slice(q["crop"]))
            return self._json({"error": "unknown endpoint"}, 404)
        except FileNotFoundError:
            return self._json({"error": "crop not found"}, 404)
        except Exception as e:
            return self._json({"error": str(e)}, 500)

    def log_message(self, *a):
        pass


def _load_model(cfg: dict):
    from stable_audio_3 import AutoencoderModel
    ae = AutoencoderModel.from_pretrained(cfg["model"])
    return ae, int(ae.sample_rate)


def main():
    global _ae, _sr, _latent_dir, _cfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_INI)
    args = ap.parse_args()
    parser = configparser.ConfigParser()
    parser.read(args.config)
    _cfg = {**dict(parser["server"]), **dict(parser["model"])}
    _latent_dir = Path(_cfg["latent_dir"])
    print("Loading SAME-L autoencoder ...")
    _ae, _sr = _load_model(_cfg)
    port = int(_cfg["port"])
    print(f"Serving on http://localhost:{port}  (latents: {_latent_dir})")
    HTTPServer(("localhost", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_player_smoke.py -v`
Expected: PASS or SKIP (skips if torch/stable_audio_3 not importable in mir venv; the `wav_bytes` test passes because import only needs numpy).

- [ ] **Step 5: Manual server check (SA3 venv, real GPU)**

Run (only when a GPU slot is free; a training run may be using VRAM):
```bash
cd /home/kim/Projects/mir
/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python scripts/latent_server_sa3.py &
sleep 30  # model load
curl -s http://localhost:7892/status
curl -s http://localhost:7892/decode?crop=000000 -o /tmp/sa3_decode.wav && ls -l /tmp/sa3_decode.wav
curl -s "http://localhost:7892/source?crop=000000" -o /tmp/sa3_src.wav && ls -l /tmp/sa3_src.wav
kill %1
```
Expected: `/status` returns JSON `ok:true`; both WAVs are non-empty and audibly the reconstruction vs. the original.

- [ ] **Step 6: Commit**

```bash
git add scripts/latent_server_sa3.py latent_player_sa3.ini tests/explorer_sa3/test_player_smoke.py
git commit -m "feat(sa3-explorer): player decode/source/meta endpoints"
```

---

### Task 6: Player — /mix (latent interpolation)

**Files:**
- Modify: `scripts/latent_server_sa3.py` (add `_mix_latents` + route)
- Test: `tests/explorer_sa3/test_mix.py`

**Interfaces:**
- Consumes: `scripts/latent_crossfader.py` `slerp`, `lerp` (dimension-agnostic, `(z_a, z_b, t) -> Tensor`).
- Produces: `GET /mix?crop_a=&crop_b=&t=&interp=` → WAV; module helper `_interp_np(a: np.ndarray, b: np.ndarray, t: float, interp: str) -> np.ndarray` operating on `[256, T]` (T aligned to min) for unit testing without the model.

- [ ] **Step 1: Write the failing test**

```python
# tests/explorer_sa3/test_mix.py
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SERVER = Path(__file__).resolve().parents[2] / "scripts" / "latent_server_sa3.py"


def _load():
    spec = importlib.util.spec_from_file_location("latent_server_sa3", SERVER)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"player deps unavailable: {e}")
    return mod


def test_interp_np_lerp_midpoint_and_align():
    mod = _load()
    a = np.zeros((256, 10), np.float32)
    b = np.ones((256, 8), np.float32)
    out = mod._interp_np(a, b, 0.5, "lerp")
    assert out.shape == (256, 8)            # aligned to min T
    assert np.allclose(out, 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_mix.py -v`
Expected: FAIL (AttributeError: module has no `_interp_np`) or SKIP if deps missing. If it skips, temporarily confirm by checking the source contains no `_interp_np` yet.

- [ ] **Step 3: Add implementation to `scripts/latent_server_sa3.py`**

Add near the other helpers:
```python
import sys
sys.path.insert(0, str(Path(__file__).parent))  # for latent_crossfader


def _interp_np(a: np.ndarray, b: np.ndarray, t: float, interp: str) -> np.ndarray:
    """Interpolate two [256, T] latents (T aligned to min). lerp or slerp."""
    import torch
    from latent_crossfader import slerp, lerp
    T = min(a.shape[1], b.shape[1])
    za = torch.from_numpy(a[:, :T]).unsqueeze(0).float()
    zb = torch.from_numpy(b[:, :T]).unsqueeze(0).float()
    fn = slerp if interp == "slerp" else lerp
    return fn(za, zb, float(t)).squeeze(0).numpy()


def _mix(crop_a: str, crop_b: str, t: float, interp: str) -> bytes:
    import torch
    a = np.load(_latent_dir / f"{crop_a}.npy").astype(np.float32)
    b = np.load(_latent_dir / f"{crop_b}.npy").astype(np.float32)
    if a.ndim == 3: a = a[0]
    if b.ndim == 3: b = b[0]
    z = _interp_np(a, b, t, interp)
    zt = torch.from_numpy(z).unsqueeze(0).to(_ae_device())
    with torch.no_grad():
        audio = _ae.decode(zt, chunked=True,
                           chunk_size=int(_cfg["chunk_size"]),
                           overlap=int(_cfg["overlap"]))
    return wav_bytes(audio.squeeze(0).cpu().float().numpy(), _sr)
```

Add the route inside `do_GET`, before the unknown-endpoint return:
```python
            if u.path == "/mix":
                with _lock:
                    return self._wav(_mix(q["crop_a"], q["crop_b"],
                                          float(q.get("t", 0.5)),
                                          q.get("interp", "slerp")))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_mix.py -v`
Expected: PASS (needs torch; if mir venv lacks torch it SKIPs — then run the same test with the SA3 venv to confirm PASS: `/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python -m pytest tests/explorer_sa3/test_mix.py -v`).

- [ ] **Step 5: Commit**

```bash
git add scripts/latent_server_sa3.py tests/explorer_sa3/test_mix.py
git commit -m "feat(sa3-explorer): /mix latent interpolation endpoint"
```

---

### Task 7: Player — /steer (LatCH head guidance)

**Files:**
- Modify: `scripts/latent_server_sa3.py` (add head loading + `/steer` route)
- Test: `tests/explorer_sa3/test_steer.py`

**Interfaces:**
- Consumes: `stable_audio_3.models.latch.LatCH(in_channels=256, ...)`, head checkpoints `latch_weights_dir/latch_sa3_<feature>_best.pt`, and `stable_audio_3.inference.latch_guided` helpers for the guidance nudge.
- Produces: `GET /steer?crop=&head=&gain=` → WAV; module helpers `_available_heads() -> list[str]` and `_load_head(feature: str)` (cached).

- [ ] **Step 1: Write the failing test** (head discovery is pure filesystem; guidance is GPU and manually verified)

```python
# tests/explorer_sa3/test_steer.py
import importlib.util
from pathlib import Path
import pytest

SERVER = Path(__file__).resolve().parents[2] / "scripts" / "latent_server_sa3.py"


def _load():
    spec = importlib.util.spec_from_file_location("latent_server_sa3", SERVER)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"player deps unavailable: {e}")
    return mod


def test_available_heads_parses_filenames(tmp_path):
    mod = _load()
    (tmp_path / "latch_sa3_hpcp_best.pt").write_bytes(b"")
    (tmp_path / "latch_sa3_rms_energy_bass_best.pt").write_bytes(b"")
    (tmp_path / "notahead.txt").write_bytes(b"")
    mod._cfg = {"latch_weights_dir": str(tmp_path)}
    assert mod._available_heads() == ["hpcp", "rms_energy_bass"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_steer.py -v`
Expected: FAIL (AttributeError: `_available_heads`) or SKIP if deps missing.

- [ ] **Step 3: Add implementation to `scripts/latent_server_sa3.py`**

Add helpers:
```python
_heads: dict = {}


def _available_heads() -> list[str]:
    d = Path(_cfg["latch_weights_dir"])
    feats = []
    for p in sorted(d.glob("latch_sa3_*_best.pt")):
        feats.append(p.stem[len("latch_sa3_"):-len("_best")])
    return feats


def _load_head(feature: str):
    if feature in _heads:
        return _heads[feature]
    import torch
    from stable_audio_3.models.latch import LatCH
    from stable_audio_3.inference.latch_targets import head_out_channels
    ckpt = Path(_cfg["latch_weights_dir"]) / f"latch_sa3_{feature}_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(feature)
    state = torch.load(ckpt, map_location="cpu")
    sd = state.get("model", state.get("state_dict", state))
    out_ch = head_out_channels(feature)   # 12 for hpcp, else 1
    model = LatCH(in_channels=256, out_channels=out_ch)
    model.load_state_dict(sd)
    model.eval().requires_grad_(False).to(_ae_device())
    _heads[feature] = model
    return model


def _steer(crop_id: str, feature: str, gain: float) -> bytes:
    import torch
    from stable_audio_3.inference.latch_guided import guidance_grad
    arr = np.load(_latent_dir / f"{crop_id}.npy").astype(np.float32)
    if arr.ndim == 3: arr = arr[0]
    head = _load_head(feature)
    z = torch.from_numpy(arr).unsqueeze(0).float().to(_ae_device())
    grad = guidance_grad(head, z)          # [1, 256, T], fp32
    z_edit = (z + gain * grad).to(_ae_device())
    with torch.no_grad():
        audio = _ae.decode(z_edit, chunked=True,
                           chunk_size=int(_cfg["chunk_size"]),
                           overlap=int(_cfg["overlap"]))
    return wav_bytes(audio.squeeze(0).cpu().float().numpy(), _sr)
```

> **Implementer note:** confirm the exact symbol names in
> `stable_audio_3/inference/latch_guided.py` and `latch_targets.py` before
> wiring (`guidance_grad` / `head_out_channels` are the expected helpers; if the
> repo names differ, adapt the import and call — the steering math is: forward
> the head on the latent at small t, take ∂mean(pred)/∂z, add `gain * grad`).
> TFG/LatCH guidance must run **fp32** on SA3 (per MASTER.md gotcha).

Add the route + a heads listing to `/status`:
```python
            if u.path == "/steer":
                with _lock:
                    return self._wav(_steer(q["crop"], q["head"],
                                            float(q.get("gain", 48.0))))
```
And in `/status` add `"heads": _available_heads()`. In the FileNotFoundError
branch for `/steer`, return the available heads: change the handler's except to
include `{"error": "unknown crop or head", "heads": _available_heads()}`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_steer.py -v`
Expected: PASS or SKIP (the filename-parse test needs only numpy import path; if the module import pulls torch it SKIPs in mir venv — then confirm under SA3 venv).

- [ ] **Step 5: Manual steer check (SA3 venv)**

```bash
/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python scripts/latent_server_sa3.py &
sleep 30
curl -s "http://localhost:7892/steer?crop=000000&head=rms_energy_bass&gain=64" -o /tmp/sa3_steer.wav && ls -l /tmp/sa3_steer.wav
kill %1
```
Expected: non-empty WAV that audibly differs from `/decode?crop=000000` in the steered attribute. Judge by spread, not corr (MASTER.md: gain ≈ 48–96 for SA3).

- [ ] **Step 6: Commit**

```bash
git add scripts/latent_server_sa3.py tests/explorer_sa3/test_steer.py
git commit -m "feat(sa3-explorer): /steer LatCH head guidance endpoint"
```

---

### Task 8: Viewer Dash app (tabs + wiring)

**Files:**
- Create: `plots/explorer_sa3/app.py`
- Create: `plots/explorer_sa3/viewer_tab.py`
- Create: `plots/explorer_sa3/dataset_tab.py`
- Create: `plots/explorer_sa3/analysis_tab.py`
- Create: `plots/explorer_sa3/audio_panel.py`
- Test: `tests/explorer_sa3/test_app_imports.py`

**Interfaces:**
- Consumes: `sidecar_index` (Task 1), `latents` (Task 2), `analysis` (Task 4), `player_client` (Task 3).
- Produces: `app.py` exposing `app` (Dash) + `build_layout(index) -> dash.html.Div`; each `*_tab.py` exposes `layout(...)` + a `figure(...)` pure function returning a Plotly `go.Figure`.

- [ ] **Step 1: Write the failing test** (pure figure builders + layout import — no browser)

```python
# tests/explorer_sa3/test_app_imports.py
import numpy as np


def test_viewer_figure_builds():
    from plots.explorer_sa3.viewer_tab import latent_figure
    z = np.random.randn(256, 64).astype(np.float32)
    fig = latent_figure(z, content_frames=40)
    assert fig is not None
    assert len(fig.data) >= 1   # heatmap trace present


def test_analysis_figure_builds():
    from plots.explorer_sa3.analysis_tab import xcorr_figure
    c = np.eye(256, dtype=np.float32)
    fig = xcorr_figure(c)
    assert fig is not None


def test_app_layout_imports():
    from plots.explorer_sa3.app import build_layout
    layout = build_layout(index=[])
    assert layout is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_app_imports.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementations**

```python
# plots/explorer_sa3/viewer_tab.py
"""Per-crop latent + timeseries view."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from .latents import FRAME_RATE_HZ


def latent_figure(z: np.ndarray, content_frames: int) -> go.Figure:
    T = z.shape[1]
    secs = np.arange(T) / FRAME_RATE_HZ
    fig = go.Figure(go.Heatmap(z=z, x=secs, colorscale="RdBu", zmid=0))
    if 0 < content_frames < T:
        cut = content_frames / FRAME_RATE_HZ
        fig.add_vline(x=cut, line_dash="dash", line_color="black")
    fig.update_layout(title="Latent [256 × T]", xaxis_title="seconds",
                      yaxis_title="latent dim", height=480)
    return fig


def timeseries_figure(name: str, ts: np.ndarray) -> go.Figure:
    if ts.ndim == 2:
        fig = go.Figure(go.Heatmap(z=ts.T, colorscale="Viridis"))
    else:
        secs = np.arange(len(ts)) / FRAME_RATE_HZ
        fig = go.Figure(go.Scatter(x=secs, y=ts, mode="lines"))
    fig.update_layout(title=name, height=240, margin=dict(t=30))
    return fig


def layout() -> html.Div:
    return html.Div([
        dcc.Dropdown(id="sa3-crop-dd"),
        dcc.Graph(id="sa3-latent-graph"),
        dcc.Dropdown(id="sa3-ts-dd"),
        dcc.Graph(id="sa3-ts-graph"),
        html.Div(id="sa3-audio-panel"),
    ])
```

```python
# plots/explorer_sa3/dataset_tab.py
"""Dataset-wide scatter/histograms over cached sidecar scalars."""
from __future__ import annotations
import plotly.graph_objects as go
from dash import dcc, html


def scatter_figure(xs, ys, xlabel, ylabel, text=None) -> go.Figure:
    fig = go.Figure(go.Scattergl(x=xs, y=ys, mode="markers", text=text,
                                 marker=dict(size=5, opacity=0.6)))
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, height=520)
    return fig


def layout() -> html.Div:
    return html.Div([
        dcc.Dropdown(id="sa3-ds-x"), dcc.Dropdown(id="sa3-ds-y"),
        dcc.Graph(id="sa3-ds-graph"),
    ])
```

```python
# plots/explorer_sa3/analysis_tab.py
"""Live analysis tab: PCA, dim xcorr, dim<->feature correlation."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html


def xcorr_figure(c: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=c, colorscale="RdBu", zmid=0, zmin=-1, zmax=1))
    fig.update_layout(title="256×256 dim cross-correlation", height=560)
    return fig


def feature_corr_figure(corr: np.ndarray, feature: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=np.arange(len(corr)), y=corr))
    fig.update_layout(title=f"dim ↔ {feature} correlation",
                      xaxis_title="latent dim", yaxis_title="Pearson r",
                      height=320)
    return fig


def layout() -> html.Div:
    return html.Div([
        html.Button("Recompute (sampled)", id="sa3-analysis-go"),
        dcc.Graph(id="sa3-xcorr-graph"),
        dcc.Dropdown(id="sa3-feat-dd"),
        dcc.Graph(id="sa3-featcorr-graph"),
    ])
```

```python
# plots/explorer_sa3/audio_panel.py
"""Audio controls that target the SA3 player over HTTP."""
from __future__ import annotations
from dash import html
from . import player_client as pc


def panel(crop_id: str, alive: bool) -> html.Div:
    if not alive:
        return html.Div([
            html.P("Player offline. Launch:"),
            html.Code("/home/kim/Projects/SAO/stable-audio-3/.venv/bin/"
                      "python scripts/latent_server_sa3.py"),
        ])
    return html.Div([
        html.Audio(src=pc.decode_url(crop_id), controls=True),
        html.Audio(src=pc.source_url(crop_id), controls=True),
    ])
```

```python
# plots/explorer_sa3/app.py
"""SA3 latent explorer — Dash entrypoint (mir venv)."""
from __future__ import annotations
import configparser
from pathlib import Path
import dash
from dash import dcc, html
from .sidecar_index import scan_index
from . import viewer_tab, dataset_tab, analysis_tab

_INI = Path(__file__).parent.parent.parent / "latent_player_sa3.ini"


def _latent_dir() -> Path:
    cfg = configparser.ConfigParser()
    cfg.read(_INI)
    return Path(cfg["server"]["latent_dir"])


def build_layout(index) -> html.Div:
    return html.Div([
        html.H2(f"SA3 Latent Explorer — {len(index)} crops"),
        dcc.Tabs([
            dcc.Tab(label="Viewer", children=viewer_tab.layout()),
            dcc.Tab(label="Dataset", children=dataset_tab.layout()),
            dcc.Tab(label="Analysis", children=analysis_tab.layout()),
        ]),
    ])


app = dash.Dash(__name__)


def main():
    index = scan_index(_latent_dir())
    app.layout = build_layout(index)
    # Callbacks are registered in app_callbacks (wired in Task 9).
    app.run(debug=False, port=8051)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_app_imports.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add plots/explorer_sa3/app.py plots/explorer_sa3/viewer_tab.py plots/explorer_sa3/dataset_tab.py plots/explorer_sa3/analysis_tab.py plots/explorer_sa3/audio_panel.py tests/explorer_sa3/test_app_imports.py
git commit -m "feat(sa3-explorer): Dash viewer tabs + figures"
```

---

### Task 9: Viewer callbacks + end-to-end manual run

**Files:**
- Create: `plots/explorer_sa3/callbacks.py`
- Modify: `plots/explorer_sa3/app.py` (register callbacks, populate dropdowns)
- Test: `tests/explorer_sa3/test_callbacks.py`

**Interfaces:**
- Consumes: all prior viewer modules + `player_client.status`.
- Produces: `register(app, index, latent_dir)` that wires crop selection → latent/timeseries figures + audio panel, dataset x/y → scatter, and the analysis recompute button (samples ≤400 crops via `latents.load_latent`, calls `analysis.*`). Pure helper `sample_ids(index, n, seed=0) -> list[str]` is unit-tested.

- [ ] **Step 1: Write the failing test**

```python
# tests/explorer_sa3/test_callbacks.py
from plots.explorer_sa3.callbacks import sample_ids
from plots.explorer_sa3.sidecar_index import CropMeta


def _idx(n):
    return [CropMeta(f"{i:06d}", "t", "a", "b", "p", 120.0, -9.0, 0.0)
            for i in range(n)]


def test_sample_ids_caps_and_is_deterministic():
    idx = _idx(1000)
    a = sample_ids(idx, 400, seed=0)
    b = sample_ids(idx, 400, seed=0)
    assert len(a) == 400 and a == b
    assert len(sample_ids(_idx(10), 400)) == 10  # fewer than cap → all
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_callbacks.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# plots/explorer_sa3/callbacks.py
"""Dash callback wiring for the SA3 explorer."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from dash import Input, Output, no_update
from . import latents, analysis, viewer_tab, dataset_tab, analysis_tab, audio_panel
from . import player_client as pc
from .sidecar_index import CropMeta


def sample_ids(index: list[CropMeta], n: int, seed: int = 0) -> list[str]:
    ids = [c.id for c in index]
    if len(ids) <= n:
        return ids
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ids), size=n, replace=False)
    return [ids[i] for i in sorted(pick)]


def register(app, index: list[CropMeta], latent_dir: Path):
    crop_opts = [{"label": f"{c.id} — {c.source_track}", "value": c.id}
                 for c in index]

    @app.callback(Output("sa3-crop-dd", "options"),
                  Input("sa3-crop-dd", "id"))
    def _fill(_):
        return crop_opts

    @app.callback(Output("sa3-latent-graph", "figure"),
                  Output("sa3-ts-dd", "options"),
                  Output("sa3-audio-panel", "children"),
                  Input("sa3-crop-dd", "value"))
    def _show(cid):
        if not cid:
            return no_update, no_update, no_update
        import json
        z = latents.load_latent(latent_dir, cid)
        meta = json.loads((latent_dir / f"{cid}.json").read_text())
        ts = latents.load_timeseries(latent_dir, cid)
        fig = viewer_tab.latent_figure(z, latents.content_frames(meta))
        return (fig, [{"label": k, "value": k} for k in ts],
                audio_panel.panel(cid, pc.status()))

    @app.callback(Output("sa3-ts-graph", "figure"),
                  Input("sa3-crop-dd", "value"), Input("sa3-ts-dd", "value"))
    def _ts(cid, name):
        if not cid or not name:
            return no_update
        ts = latents.load_timeseries(latent_dir, cid)[name]
        return viewer_tab.timeseries_figure(name, ts)

    @app.callback(Output("sa3-xcorr-graph", "figure"),
                  Input("sa3-analysis-go", "n_clicks"))
    def _analysis(n):
        if not n:
            return no_update
        ids = sample_ids(index, 400)
        lats = [latents.load_latent(latent_dir, i) for i in ids]
        return analysis_tab.xcorr_figure(analysis.dim_xcorr(lats))
```

Modify `plots/explorer_sa3/app.py` `main()` to call `register`:
```python
from .callbacks import register
# ... inside main(), after app.layout = ...
    register(app, index, _latent_dir())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/test_callbacks.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Full suite + manual end-to-end**

```bash
/home/kim/Projects/mir/mir/bin/python -m pytest tests/explorer_sa3/ -v
# manual: launch player (SA3 venv) then viewer (mir venv), open browser
/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python scripts/latent_server_sa3.py &
/home/kim/Projects/mir/mir/bin/python -m plots.explorer_sa3.app
# visit http://localhost:8051 — select a crop, see heatmap + timeseries,
# play decode vs source, run Analysis recompute.
```
Expected: all tests green; viewer renders; audio plays; analysis figure draws.

- [ ] **Step 6: Commit**

```bash
git add plots/explorer_sa3/callbacks.py plots/explorer_sa3/app.py tests/explorer_sa3/test_callbacks.py
git commit -m "feat(sa3-explorer): viewer callbacks + end-to-end wiring"
```

---

## Self-Review

- **Spec coverage:** viewer (Tasks 1,2,4,8,9) ✓; player decode/source/meta (5), mix (6), steer (7) ✓; sidecars-as-source (1,2) ✓; live analysis no-precompute (4,9) ✓; A/B source reference (5,8) ✓; LatCH heads first-class (7) ✓; steer-sao deferred (not implemented — seam only, per spec) ✓; error handling (404/offline panel in 5,8) ✓; testing split mir-venv vs guarded (all tasks) ✓. No task back-ports the Small tools ✓.
- **Placeholder scan:** one implementer-note in Task 7 about confirming `latch_guided` symbol names — this is a real verification instruction with the fallback math spelled out, not a placeholder. No TBD/TODO elsewhere.
- **Type consistency:** `CropMeta` fields consistent across Tasks 1/9; `load_latent(latent_dir, crop_id)`, `content_frames(meta)`, `load_timeseries(latent_dir, crop_id)` used identically in Tasks 2/9; `wav_bytes(audio, sr)`, `_interp_np`, `_available_heads`, `_steer` consistent across Tasks 5/6/7; `latent_figure`/`xcorr_figure` consistent across Tasks 8/9.

## Risks / notes for the executor

- The player's manual checks need free VRAM; a training run may be active (use chunked decode, default chunk_size=128; drop to 64 if OOM).
- `stable_audio_3.inference.latch_guided` helper names must be confirmed (Task 7 note).
- `same-l` weights download is gated (HF auth) — ensure `huggingface-cli login` / `HF_TOKEN` before first model load.
