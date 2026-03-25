# Latent Explorer Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the latent crossfade player by removing single-crop playback, adding a mode switch, a play-average feature, a client-side smart loop, and a full-width alignment bar with beat/onset tick marks.

**Architecture:** Three independent server endpoints are added first (one per server), then the frontend HTML is refactored in five logical passes: UI structure, camera fix, alignment bar, smart loop, and ⊕ Avg. Each server task is verifiable with a standalone Python test script before any frontend work begins.

**Tech Stack:** Python 3 / `http.server` / `numpy` (servers), Plotly 3.x / Web Audio API / vanilla JS (frontend), `latent_player.ini` (shared config).

**Spec:** `docs/superpowers/specs/2026-03-25-latent-explorer-redesign.md`

---

## File Map

| File | Change |
|---|---|
| `scripts/latent_shape_server.py` | Add `_source_dir` global; `/api/timecodes`; `/api/average-shape` |
| `scripts/latent_server.py` | Add `/average` endpoint handler |
| `plots/latent_shape_explorer/index.html` | UI restructure; camera fix; alignment bar; smart loop; ⊕ Avg |
| `tests/test_latent_explorer_endpoints.py` | New — pytest tests for all three new server endpoints |

---

## Task 1: `/api/timecodes` endpoint in `latent_shape_server.py`

**Files:**
- Modify: `scripts/latent_shape_server.py`
- Test: `tests/test_latent_explorer_endpoints.py`

### Background
The shape server (`latent_shape_server.py`) has no knowledge of source-track sidecar files today. It only knows `_latent_dir`. Sidecar files (`.BEATS_GRID`, `.DOWNBEATS`, `.ONSETS`) live in the source-track folder: `source_dir / track_name / track_name.BEATS_GRID`. The crop JSON sidecar (in `_latent_dir / track / crop.json`) has `start_time` and `end_time` fields (absolute seconds within the source track).

### Step-by-step

- [ ] **Step 1: Write the failing test**

Add `tests/test_latent_explorer_endpoints.py`:

```python
"""Offline unit tests for new latent explorer server endpoint logic.

These test the pure functions, not the live HTTP server.
Import directly from the server module after monkey-patching globals.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# --- helpers used by the test suite ---

def make_sidecar(tmp: Path, track: str, crop: str,
                  start_time: float, end_time: float):
    """Write a minimal crop JSON sidecar."""
    track_dir = tmp / track
    track_dir.mkdir(parents=True, exist_ok=True)
    sidecar = track_dir / f"{crop}.json"
    sidecar.write_text(json.dumps({
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "position": start_time / 120.0,
    }))
    return sidecar

def make_sidecar_files(source_dir: Path, track: str,
                        beats, downbeats, onsets):
    """Write .BEATS_GRID, .DOWNBEATS, .ONSETS files for a track."""
    d = source_dir / track
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{track}.BEATS_GRID").write_text("\n".join(str(t) for t in beats))
    (d / f"{track}.DOWNBEATS").write_text("\n".join(str(t) for t in downbeats))
    (d / f"{track}.ONSETS").write_text("\n".join(str(t) for t in onsets))


# --- import the function under test ---

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_timecodes_filters_to_crop_window():
    """_read_timecodes returns only timestamps within [start_time, end_time]
    and offsets them relative to start_time."""
    import latent_shape_server as srv

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        latent_dir = tmp / "latents"
        source_dir = tmp / "source"

        beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        downbeats = [0.0, 2.0]
        onsets = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        make_sidecar_files(source_dir, "TrackA", beats, downbeats, onsets)
        make_sidecar(latent_dir, "TrackA", "crop01", start_time=1.0, end_time=3.0)

        # Monkey-patch globals
        srv._latent_dir = latent_dir
        srv._source_dir = source_dir

        result = srv._read_timecodes("TrackA", "crop01")

        # beats [1.0..3.0] → [0.0, 0.5, 1.0, 1.5, 2.0] (5 beats, offset by -1.0)
        assert result["beats"] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0], abs=1e-4)
        # downbeats [1.0 is on boundary, 2.0 is in window] → [0.0, 1.0]
        assert 0.0 in result["downbeats"]
        assert result["duration"] == pytest.approx(2.0, abs=1e-4)
        # bpm: 5 beats over 2s → intervals ~0.5s → bpm ~120
        assert result["bpm"] == pytest.approx(120.0, abs=2.0)


def test_timecodes_missing_sidecar_returns_empty_array():
    """Missing .ONSETS → onsets=[], listed in missing[]."""
    import latent_shape_server as srv

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        latent_dir = tmp / "latents"
        source_dir = tmp / "source"

        # Only write beats and downbeats — omit onsets
        track_dir = source_dir / "TrackB"
        track_dir.mkdir(parents=True, exist_ok=True)
        (track_dir / "TrackB.BEATS_GRID").write_text("1.0\n1.5\n2.0\n2.5\n")
        (track_dir / "TrackB.DOWNBEATS").write_text("1.0\n3.0\n")
        # No TrackB.ONSETS file

        make_sidecar(latent_dir, "TrackB", "crop02", 1.0, 3.0)
        srv._latent_dir = latent_dir
        srv._source_dir = source_dir

        result = srv._read_timecodes("TrackB", "crop02")
        assert result["onsets"] == []
        assert "onsets" in result.get("missing", [])


def test_timecodes_no_source_dir_returns_none():
    """_read_timecodes returns None when _source_dir is None."""
    import latent_shape_server as srv
    srv._source_dir = None
    assert srv._read_timecodes("any", "any") is None


def test_timecodes_missing_crop_json_returns_none():
    """_read_timecodes returns None when crop JSON sidecar is absent."""
    import latent_shape_server as srv
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        latent_dir = tmp / "latents"
        source_dir = tmp / "source"
        make_sidecar_files(source_dir, "TrackC", [1.0, 1.5], [1.0], [1.0, 1.25])
        (latent_dir / "TrackC").mkdir(parents=True, exist_ok=True)
        # No crop JSON written

        srv._latent_dir = latent_dir
        srv._source_dir = source_dir
        assert srv._read_timecodes("TrackC", "missing_crop") is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/kim/Projects/mir
python -m pytest tests/test_latent_explorer_endpoints.py -v 2>&1 | head -40
```

Expected: 4 failures — `AttributeError: module 'latent_shape_server' has no attribute '_read_timecodes'`

- [ ] **Step 3: Add `_source_dir` global and `_read_timecodes()` to `latent_shape_server.py`**

At the top of the globals section (after `_metadata_cache = None`, around line 41), add:

```python
_source_dir = None
```

Before the `LatentShapeHandler` class (around line 104), add:

```python
# ---------------------------------------------------------------------------
# Timecode helpers
# ---------------------------------------------------------------------------

def _read_sidecar_times(path: Path) -> list:
    """Read a sidecar file (one float per line) → list of floats."""
    if not path.exists():
        return None          # None signals "file missing"
    times = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                times.append(float(line))
            except ValueError:
                pass
    return times


def _read_timecodes(track: str, crop_id: str) -> dict | None:
    """Return timecodes for a crop window, offset to crop-relative seconds.

    Returns None if _source_dir is unconfigured or the crop JSON is absent.
    Returns a dict with keys: beats, downbeats, onsets, duration, bpm, missing.
    """
    if _source_dir is None:
        return None

    crop_json = _latent_dir / track / f"{crop_id}.json"
    if not crop_json.exists():
        return None

    try:
        meta = json.loads(crop_json.read_text())
    except Exception:
        return None

    start_t = float(meta.get("start_time", 0.0))
    end_t   = float(meta.get("end_time",   meta.get("duration", 0.0) + start_t))
    duration = end_t - start_t

    sidecar_map = {
        "beats":     _source_dir / track / f"{track}.BEATS_GRID",
        "downbeats": _source_dir / track / f"{track}.DOWNBEATS",
        "onsets":    _source_dir / track / f"{track}.ONSETS",
    }

    result = {"duration": round(duration, 4), "missing": []}
    for key, path in sidecar_map.items():
        raw = _read_sidecar_times(path)
        if raw is None:
            result[key] = []
            result["missing"].append(key)
        else:
            filtered = [round(t - start_t, 5)
                        for t in raw if start_t - 1e-4 <= t <= end_t + 1e-4]
            result[key] = filtered

    # BPM from median beat interval
    beats = result["beats"]
    if len(beats) >= 2:
        intervals = [beats[i+1] - beats[i] for i in range(len(beats)-1)]
        intervals.sort()
        med = intervals[len(intervals)//2]
        result["bpm"] = round(60.0 / med, 2) if med > 0 else None
    else:
        result["bpm"] = None

    if not result["missing"]:
        del result["missing"]

    return result
```

- [ ] **Step 4: Add the `/api/timecodes` HTTP handler in `do_GET`**

In `LatentShapeHandler.do_GET`, after the `/api/shape` block (around line 231, before the `super().do_GET()` fallback), add:

```python
        if path == "/api/timecodes":
            track   = query.get("track", [""])[0]
            crop_id = query.get("crop",  [""])[0]
            if not track or not crop_id:
                return self.send_error_json(400, "Missing track or crop")
            result = _read_timecodes(track, crop_id)
            if result is None:
                return self.send_error_json(404,
                    "Crop not found or source_dir not configured")
            self.send_json(result)
            return
```

- [ ] **Step 5: Wire `source_dir` in `main()`**

In `main()`, extend the existing `global` declaration line (around line 249) from:

```python
    global _latent_dir, _stem_dir, _raw_audio_dir
```

to:

```python
    global _latent_dir, _stem_dir, _raw_audio_dir, _source_dir
```

Then after the existing `_raw_audio_dir` assignment (around line 259), add:

```python
    sd_src = config.get("server", "source_dir", fallback="")
    _source_dir = Path(sd_src) if sd_src else None
    if _source_dir:
        print(f"  Source dir : {_source_dir}")
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_latent_explorer_endpoints.py::test_timecodes_filters_to_crop_window \
                 tests/test_latent_explorer_endpoints.py::test_timecodes_missing_sidecar_returns_empty_array \
                 tests/test_latent_explorer_endpoints.py::test_timecodes_no_source_dir_returns_none \
                 tests/test_latent_explorer_endpoints.py::test_timecodes_missing_crop_json_returns_none \
                 -v
```

Expected: 4 PASSED

- [ ] **Step 7: Commit**

```bash
git add scripts/latent_shape_server.py tests/test_latent_explorer_endpoints.py
git commit -m "feat: add /api/timecodes endpoint to latent_shape_server"
```

---

## Task 2: `/api/average-shape` endpoint in `latent_shape_server.py`

**Files:**
- Modify: `scripts/latent_shape_server.py`
- Test: `tests/test_latent_explorer_endpoints.py`

### Background
This endpoint loads all full-mix `.npy` files for a track, computes their frame-wise mean (zero-padding shorter files to the longest), projects through the global PCA, and returns `{ "points": [[x,y,z], ...] }`. The `project_3d()` function already exists and accepts `[64, T]` arrays.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_latent_explorer_endpoints.py`:

```python
def test_average_shape_returns_3d_points():
    """_compute_average_shape averages latent files and returns PCA-projected points."""
    import latent_shape_server as srv

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        latent_dir = tmp / "latents"
        track_dir = latent_dir / "TrackD"
        track_dir.mkdir(parents=True, exist_ok=True)

        # Write two fake latent files of different lengths
        lat1 = np.random.randn(64, 20).astype(np.float32)
        lat2 = np.random.randn(64, 15).astype(np.float32)
        np.save(str(track_dir / "crop01.npy"), lat1)
        np.save(str(track_dir / "crop02.npy"), lat2)

        srv._latent_dir = latent_dir
        # Provide a trivial PCA model (identity-like)
        srv._pca_mean       = np.zeros(64, dtype=np.float32)
        srv._pca_components = np.eye(3, 64, dtype=np.float32)

        points = srv._compute_average_shape("TrackD")
        assert points is not None
        assert len(points) == 20   # length of the longest crop
        assert len(points[0]) == 3  # 3D


def test_average_shape_skips_stem_files():
    """_compute_average_shape ignores stem-suffixed .npy files."""
    import latent_shape_server as srv

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        latent_dir = tmp / "latents"
        track_dir = latent_dir / "TrackE"
        track_dir.mkdir(parents=True, exist_ok=True)

        lat_fm   = np.ones((64, 10), dtype=np.float32)
        lat_stem = np.full((64, 10), 99.0, dtype=np.float32)
        np.save(str(track_dir / "crop01.npy"),       lat_fm)
        np.save(str(track_dir / "crop01_drums.npy"), lat_stem)

        srv._latent_dir     = latent_dir
        srv._pca_mean       = np.zeros(64, dtype=np.float32)
        srv._pca_components = np.eye(3, 64, dtype=np.float32)

        points = srv._compute_average_shape("TrackE")
        # If stem was included the mean would be ~50; fullmix-only mean is 1.0
        assert points is not None
        # PC1 projection of mean([1]*64) with identity components ≈ 1.0
        assert abs(points[0][0] - 1.0) < 0.01
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_latent_explorer_endpoints.py::test_average_shape_returns_3d_points \
                 tests/test_latent_explorer_endpoints.py::test_average_shape_skips_stem_files -v
```

Expected: 2 FAILED

- [ ] **Step 3: Add `_compute_average_shape()` to `latent_shape_server.py`**

In the timecode helpers block (after `_read_timecodes`), add:

```python
def _compute_average_shape(track: str) -> list | None:
    """Load all full-mix latent files for a track, average, project 3D.

    Shorter files are zero-padded to the length of the longest file.
    Returns [[x,y,z], ...] or None if no latent files exist.
    """
    track_dir = _latent_dir / track
    npys = [p for p in sorted(track_dir.glob("*.npy"))
            if not any(p.stem.endswith(s) for s in STEM_SUFFIXES)]
    if not npys:
        return None

    arrays = []
    max_T  = 0
    for npy in npys:
        try:
            arr = np.load(str(npy)).astype(np.float32)  # [64, T]
            arrays.append(arr)
            max_T = max(max_T, arr.shape[1])
        except Exception:
            pass

    if not arrays:
        return None

    # Zero-pad all arrays to max_T, then average
    padded = np.zeros((len(arrays), 64, max_T), dtype=np.float32)
    for i, arr in enumerate(arrays):
        padded[i, :, :arr.shape[1]] = arr
    mean_latent = padded.mean(axis=0)   # [64, max_T]

    return project_3d(mean_latent).tolist()
```

- [ ] **Step 4: Add the `/api/average-shape` HTTP handler in `do_GET`**

After the `/api/timecodes` block, add:

```python
        if path == "/api/average-shape":
            track = query.get("track", [""])[0]
            if not track:
                return self.send_error_json(400, "Missing track")
            if not (_latent_dir / track).is_dir():
                return self.send_error_json(404, "Track not found")
            points = _compute_average_shape(track)
            if points is None:
                return self.send_error_json(404, "No latent files for track")
            self.send_json({"points": points})
            return
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_latent_explorer_endpoints.py::test_average_shape_returns_3d_points \
                 tests/test_latent_explorer_endpoints.py::test_average_shape_skips_stem_files -v
```

Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add scripts/latent_shape_server.py tests/test_latent_explorer_endpoints.py
git commit -m "feat: add /api/average-shape endpoint to latent_shape_server"
```

---

## Task 3: `/average` endpoint in `latent_server.py`

**Files:**
- Modify: `scripts/latent_server.py`
- Test: `tests/test_latent_explorer_endpoints.py`

### Background
`latent_server.py` (port 7891) decodes latents through the VAE (GPU). The `/average` endpoint averages all full-mix latents for a track, decodes once, and returns WAV. This mirrors `decode_to_wav()` but operates on the mean of all crops.

Since the VAE isn't available in tests (requires GPU + model weights), the test only checks the routing logic and that the function exists with the correct signature — it does not decode audio.

- [ ] **Step 1: Write the test**

Append to `tests/test_latent_explorer_endpoints.py`:

```python
def test_average_wav_function_signature():
    """_average_track_to_wav exists and has the expected signature.

    latent_server imports torch and calls setup_rocm_env() at module level,
    so this test requires the GPU/ROCm environment.  It is skipped in CI or
    CPU-only environments.
    """
    try:
        import latent_server as srv
    except Exception as e:
        pytest.skip(f"latent_server not importable in this environment: {e}")

    import inspect
    sig = inspect.signature(srv._average_track_to_wav)
    assert "track" in sig.parameters
    # Should not require GPU arguments (those are module globals)
    assert len(sig.parameters) == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_latent_explorer_endpoints.py::test_average_wav_function_signature -v
```

Expected: FAILED — `ImportError` or `AttributeError`

- [ ] **Step 3: Add `_average_track_to_wav()` to `latent_server.py`**

Find the `crossfade_raw_to_wav` function definition (around line 793). Before it, add:

```python
def _average_track_to_wav(track: str) -> bytes:
    """Average all full-mix latents for a track, decode once, return WAV bytes.

    Caller must hold _decode_lock.
    """
    track_dir = _latent_dir / track
    stem_suffixes = {"_bass", "_drums", "_other", "_vocals"}
    npys = [p for p in sorted(track_dir.glob("*.npy"))
            if not any(p.stem.endswith(s) for s in stem_suffixes)]
    if not npys:
        raise FileNotFoundError(f"No full-mix latent files for track: {track}")

    arrays = []
    max_T  = 0
    for npy in npys:
        try:
            arr = np.load(str(npy)).astype(np.float32)   # [64, T]
            arrays.append(arr)
            max_T = max(max_T, arr.shape[1])
        except Exception:
            pass

    if not arrays:
        raise ValueError("Could not load any latent files")

    padded = np.zeros((len(arrays), 64, max_T), dtype=np.float32)
    for i, arr in enumerate(arrays):
        padded[i, :, :arr.shape[1]] = arr
    mean_np = padded.mean(axis=0)   # [64, max_T]

    mean_t = torch.from_numpy(mean_np).unsqueeze(0).to(device=_device, dtype=_dtype)
    with torch.no_grad():
        audio = _autoencoder.decode(mean_t)   # [1, 2, samples]

    audio_np = audio.squeeze(0).cpu().float().numpy()   # [2, samples]

    # Peak-normalise to 0.9
    peak = np.abs(audio_np).max()
    if peak > 1e-6:
        audio_np = audio_np * (0.9 / peak)
    audio_np  = np.clip(audio_np, -1.0, 1.0)
    audio_i16 = (audio_np * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(_sample_rate)
        wf.writeframes(audio_i16.T.flatten().tobytes())
    return buf.getvalue()
```

- [ ] **Step 4: Add the `/average` route in `do_GET`**

In `Handler.do_GET` (around line 1219), add a new `elif` before the final `else`:

```python
        elif parsed.path == "/average":
            track = qs.get("track", [""])[0]
            if not track:
                self._error(400, "Missing track parameter")
                return
            if not (_latent_dir / track).is_dir():
                self._error(404, f"Track not found: {track}")
                return
            try:
                with _decode_lock:
                    wav_bytes = _average_track_to_wav(track)
            except Exception as e:
                self._error(500, str(e))
                return
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type",   "audio/wav")
            self.send_header("Content-Length", str(len(wav_bytes)))
            self.end_headers()
            self.wfile.write(wav_bytes)
```

Also add `"/average"` to the `Access-Control-Expose-Headers` line (line ~1212) is not needed since it returns audio, not custom headers. No change needed there.

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_latent_explorer_endpoints.py::test_average_wav_function_signature -v
```

Expected: PASSED

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/test_latent_explorer_endpoints.py -v
```

Expected: All 7 tests PASSED

- [ ] **Step 7: Commit**

```bash
git add scripts/latent_server.py tests/test_latent_explorer_endpoints.py
git commit -m "feat: add /average endpoint to latent_server"
```

---

## Task 4: Frontend — UI structure (remove Load button, always-visible crossfade panel, mode switch)

**Files:**
- Modify: `plots/latent_shape_explorer/index.html`

### Background
Three structural changes:
1. Remove the `▶ Load` button (`#btn_play`) and all its JS handlers.
2. Make the crossfade panel always visible (remove the toggle button + `display:none`).
3. Replace the hardcoded `mode=latent` in `buildXfadeURL()` with a three-segment mode switch that maps to `{mode=ab}` / `{mode=latent}` / `{mode=stems}`.

The mode switch needs a single α slider (for Full-mix) that shows/hides based on the active mode.

- [ ] **Step 1: Update CSS — crossfade panel always open**

Find the `#xfade-panel` CSS block (lines 69–78):

```css
    #xfade-panel {
      display: none; /* hidden until toggled */
      background: #181818;
      ...
    }
    #xfade-panel.open { display: flex; }
```

Replace with:

```css
    #xfade-panel {
      display: flex;
      background: #181818;
      border-top: 1px solid #383838;
      padding: 8px 15px;
      gap: 14px;
      align-items: center;
      flex-wrap: wrap;
    }
```

Remove the `#xfade-panel.open { display: flex; }` line.

Remove the entire `#btn-xfade-toggle` CSS block (lines 95–99):

```css
    #btn-xfade-toggle {
      background: #2a2200; color: #ff9900; border: 1px solid #ff9900;
      padding: 5px 10px; border-radius: 4px; font-size: 12px; cursor: pointer;
    }
    #btn-xfade-toggle.open { background: #ff9900; color: #000; }
```

Add CSS for the mode switch and single-α slider after the `.beta-slider` block:

```css
    .mode-switch { display: flex; gap: 0; }
    .mode-switch button {
      background: #2a2a2a; color: #aaa; border: 1px solid #555;
      padding: 4px 10px; font-size: 11px; border-radius: 0; cursor: pointer;
      border-right: none;
    }
    .mode-switch button:first-child { border-radius: 4px 0 0 4px; }
    .mode-switch button:last-child  { border-radius: 0 4px 4px 0; border-right: 1px solid #555; }
    .mode-switch button.active { background: #ff9900; color: #000; border-color: #ff9900; }
    #xf_fullmix_row { display: flex; flex-direction: column; align-items: center; gap: 2px; }
    #xf_fullmix_row span { font-size: 9px; color: #888; }
    #xf_fullmix_row input[type=range] { width: 120px; accent-color: #ff9900; }
```

- [ ] **Step 2: Update HTML — header**

Remove the `▶ Load` button div (lines 149–152):

```html
  <div class="playback-panel" style="display: flex; gap: 10px; align-items: center; border-left: 1px solid #444; padding-left: 15px;">
      <button id="btn_play" style="width: 80px; font-weight: bold; background: #00d2ff; color: #000; border:none;">▶ Load</button>
      <span id="audio_status" style="font-size: 11px; color: #888; width: 120px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">No audio</span>
  </div>
```

Remove the crossfade toggle div (lines 153–156):

```html
  <!-- Crossfade toggle -->
  <div style="display:flex; align-items:center; border-left: 1px solid #444; padding-left: 15px;">
    <button id="btn-xfade-toggle" title="Toggle latent crossfade controls">✕ Crossfade</button>
  </div>
```

In the same header area, add ⊕ Avg button (after the last remaining panel, before `</header>`):

```html
  <div class="panel" style="flex-direction: row; align-items: flex-end; gap: 8px; padding-bottom: 2px;">
    <button id="btn_avg" style="background:#1a2a1a;color:#4cd137;border:1px solid #4cd137;padding:5px 8px;border-radius:4px;font-size:12px;" title="Decode and play average of all Track A latents">⊕ Avg</button>
    <span id="avg_status" style="font-size:11px;color:#888;"></span>
  </div>
```

- [ ] **Step 3: Update HTML — crossfade panel, add mode switch and single-α slider**

At the very start of `<div id="xfade-panel">` (before the first `<div class="xfade-section">`), add:

```html
  <div class="xfade-section">
    <label>Crossfade Mode</label>
    <div class="mode-switch">
      <button id="xf_mode_ab"     class="active" data-mode="ab">Full-mix</button>
      <button id="xf_mode_latent" data-mode="latent">Stem latents</button>
      <button id="xf_mode_stems"  data-mode="stems">Stem audio</button>
    </div>
  </div>
  <div class="xfade-section" id="xf_fullmix_section">
    <label>A → B (α)</label>
    <div id="xf_fullmix_row">
      <input type="range" id="xf_mix" min="0" max="1" step="0.01" value="0.5">
      <span id="xf_mix_val">0.50</span>
    </div>
  </div>
```

Wrap the existing four per-stem sliders section in a div with id `xf_stem_section`:

Change the opening of the first `<div class="xfade-section">` (the one with `Stem A→B (α)` label) to:

```html
  <div class="xfade-section" id="xf_stem_section">
```

(The closing tag is already there.)

- [ ] **Step 4: Update JS — remove `btn_play`/`audio_status` references, remove toggle handler**

Find and remove these JS sections:

**Remove** the `btn_play`/`audio_status` variable declarations (lines 243–244):
```javascript
  const btn_play = document.getElementById("btn_play");
  const audio_status = document.getElementById("audio_status");
```

**Remove** the single-crop audio globals (lines 286–289):
```javascript
  let audioCtx = null;
  let audioBuffer = null;
  let sourceNode = null;
  let playStart = 0;
  let pauseOffset = 0;
  let animationFrameId = null;
```

**Remove** the `btnXfadeToggle` variable and toggle handler (lines 308, 378–385):
```javascript
  const btnXfadeToggle = document.getElementById("btn-xfade-toggle");
  // ... and:
  btnXfadeToggle.addEventListener("click", () => {
    const isOpen = xfadePanel.classList.toggle("open");
    btnXfadeToggle.classList.toggle("open", isOpen);
    btnXfadeToggle.textContent = isOpen ? "▲ Crossfade" : "▶ Crossfade";
    if (isOpen) renderGhostTrace();
    else removeGhostTrace();
  });
```

Also remove the `btn_play` click handler and any audio player functions that only served single-crop playback (search for `btn_play.addEventListener` and related `getAudioCtx()`, `playAudio()`, `stopAudio()` if present).

**Note:** The `animationFrameId`, `sourceNode`, etc. may only be used for single-crop animation. Check `updateAnimation()` at line 1282 — it references `sourceNode` and `audioBuffer`. Since single-crop playback is being removed, replace the `sourceNode`/`audioBuffer` check at line 1283 with `xfadeSourceNode`/`xfadeBuffer`:

```javascript
  function updateAnimation() {
      if (!xfadeSourceNode || !xfadeBuffer || ...) return;
      const currentTime = xfadeAudioCtx.currentTime - xfadePlayStart;
      ...
  }
```

Add `let xfadePlayStart = 0;` near the other xfade globals and set it in `startXfadePlayback`.

- [ ] **Step 5: Wire mode switch JS**

After the `xfInterp.addEventListener` block, add:

```javascript
  // ---- Mode switch --------------------------------------------------------
  const xfModeButtons = document.querySelectorAll('.mode-switch button');
  let xfMode = 'ab';  // 'ab' | 'latent' | 'stems'

  function setXfMode(mode) {
    xfMode = mode;
    xfModeButtons.forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
    // Full-mix: show single α slider, hide per-stem sliders
    const isFullMix = mode === 'ab';
    document.getElementById('xf_fullmix_section').style.display = isFullMix ? '' : 'none';
    document.getElementById('xf_stem_section').style.display    = isFullMix ? 'none' : '';
    reloadXfadeIfPlaying();
  }

  xfModeButtons.forEach(b => b.addEventListener('click', () => setXfMode(b.dataset.mode)));
  setXfMode('ab');  // initial state

  // Wire the single-α slider
  const xfMix    = document.getElementById('xf_mix');
  const xfMixVal = document.getElementById('xf_mix_val');
  xfMix.addEventListener('input', e => {
    xfMixVal.textContent = parseFloat(e.target.value).toFixed(2);
  });
  xfMix.addEventListener('change', () => reloadXfadeIfPlaying());
```

- [ ] **Step 6: Update `buildXfadeURL()` to use `xfMode` and `xfMix`**

Find `buildXfadeURL()` (line 487). Replace the hardcoded `mode=latent` in the URL with:

```javascript
  function buildXfadeURL() {
    const tA = trackA_sel.value;
    const tB = trackB_sel.value;
    if (!tA || !tB) return null;
    const optA = cropA_sel.options[cropA_sel.selectedIndex];
    const optB = cropB_sel.options[cropB_sel.selectedIndex];
    const posA = optA ? (optA.dataset.position || "0.5") : "0.5";
    const posB = optB ? (optB.dataset.position || "0.5") : "0.5";
    const bA   = parseFloat(xfBetaA.value).toFixed(3);
    const bB   = parseFloat(xfBetaB.value).toFixed(3);
    const interp = xfInterp.value;

    let stemOrMixParams;
    if (xfMode === 'ab') {
      stemOrMixParams = `mix=${parseFloat(xfMix.value).toFixed(3)}`;
    } else {
      stemOrMixParams = XF_STEMS.map(s =>
        `${s}=${parseFloat(xfStemSliders[s].value).toFixed(3)}`
      ).join("&");
    }

    const manips = getGlobalManipulationOffsets();
    let manipQS = "";
    if (manips.manip_channels.length > 0) {
        manipQS = `&manip_channels=${manips.manip_channels.join(",")}&manip_amounts=${manips.manip_amounts.map(v => v.toFixed(3)).join(",")}`;
    }

    let url = `http://localhost:7891/crossfade?mode=${xfMode}&track_a=${encodeURIComponent(tA)}&track_b=${encodeURIComponent(tB)}&pos_a=${posA}&pos_b=${posB}&${stemOrMixParams}&beta_a=${bA}&beta_b=${bB}&interp=${interp}${manipQS}`;

    if (isBeatMatch) {
        const bm = computeBeatMatch(tA, tB);
        if (bm) {
            url += `&beatmatch=1&shift_a=${bm.shiftA}&stretch_a=${bm.stretchA.toFixed(4)}`;
            url += `&shift_b=${bm.shiftB}&stretch_b=${bm.stretchB.toFixed(4)}`;
        }
    }
    return url;
  }
```

- [ ] **Step 7: Verify in browser**

Start `latent_shape_server.py` and open `http://localhost:7892`. Verify:
- No ▶ Load button in the header
- ⊕ Avg button visible in header
- Crossfade panel visible without clicking anything
- Mode switch shows `[ Full-mix ] [ Stem latents ] [ Stem audio ]`
- Clicking Full-mix shows single α slider, hides per-stem sliders
- Clicking Stem latents shows per-stem sliders, hides single α slider
- ▶ Play Crossfade works with both a Track A and Track B selected

- [ ] **Step 8: Commit**

```bash
git add plots/latent_shape_explorer/index.html
git commit -m "feat: remove Load button, add mode switch, make crossfade panel always visible"
```

---

## Task 5: Frontend — camera state fix

**Files:**
- Modify: `plots/latent_shape_explorer/index.html`

### Background
`renderPlot()` (line 1049) builds a layout object that hardcodes `camera: { eye: {x:1.05, y:1.05, z:1.05} }` (line 1249). Even though `uirevision: 'latent-shape'` is set, including `camera:` in the layout explicitly overrides the user's current camera position on every re-render.

Fix: before calling `Plotly.react()`, read the current camera from `plotDiv._fullLayout.scene.camera`. If it exists (plot already initialized), reuse it in the layout; otherwise use the default.

- [ ] **Step 1: Add camera save/restore to `renderPlot()`**

Find the camera block in `renderPlot()` (lines 1249–1252):

```javascript
          camera: {
            eye: { x: 1.05, y: 1.05, z: 1.05 },
            center: { x: 0, y: 0, z: 0 }
          },
```

Replace with:

```javascript
          camera: (() => {
            const plotDiv = document.getElementById('plot');
            try {
              const existing = plotDiv._fullLayout && plotDiv._fullLayout.scene
                               && plotDiv._fullLayout.scene.camera;
              if (existing && existing.eye) return existing;
            } catch(e) {}
            return { eye: { x: 1.05, y: 1.05, z: 1.05 }, center: { x: 0, y: 0, z: 0 } };
          })(),
```

- [ ] **Step 2: Verify in browser**

1. Load two crops, click Render 3D, rotate/zoom the plot to a non-default angle.
2. Change the visualization mode (Raw → Hybrid, etc.) via the `mode_sel` dropdown.
3. Verify the camera position is preserved across the re-render.

- [ ] **Step 3: Commit**

```bash
git add plots/latent_shape_explorer/index.html
git commit -m "fix: preserve Plotly camera state across renderPlot() mode changes"
```

---

## Task 6: Frontend — alignment bar (canvas + timecodes fetch + tick rendering)

**Files:**
- Modify: `plots/latent_shape_explorer/index.html`

### Background
A full-width 80px canvas is added between the crossfade panel and the 3D plot. It renders three layers of vertical tick marks per lane (top = A, bottom = B) as soon as timecodes are fetched for each crop. The existing `waveform_canvas` overlay is removed.

- [ ] **Step 1: Remove `waveform_canvas` HTML and `drawWaveforms()` JS**

Remove the `<canvas id="waveform_canvas">` element (line 216):
```html
    <canvas id="waveform_canvas" width="600" height="100" style="position: absolute; top: 15px; left: 15px; z-index: 99; pointer-events: none; background: rgba(0,0,0,0.5); border: 1px solid #444; border-radius: 4px; display: none;"></canvas>
```

Remove the `drawWaveforms()` function (lines 575–623).

Remove the `drawWaveforms()` call in `fetchXfadeAudio()` (lines 531–535):
```javascript
          const metaStr = r.headers.get("X-Crossfade-Meta");
          if (metaStr) {
              try { window._xfadeMeta = JSON.parse(metaStr); drawWaveforms(); } catch(e){}
          } else {
              window._xfadeMeta = null; drawWaveforms();
          }
```
Replace with just:
```javascript
          const metaStr = r.headers.get("X-Crossfade-Meta");
          if (metaStr) {
              try { window._xfadeMeta = JSON.parse(metaStr); } catch(e){}
          }
```

- [ ] **Step 2: Add alignment bar CSS**

Add to the `<style>` block:

```css
    #alignment-bar {
      width: 100%; height: 80px; display: block;
      background: #0d0d0d; border-top: 1px solid #2a2a2a; border-bottom: 1px solid #2a2a2a;
      flex-shrink: 0;
    }
```

The body already uses `display:flex; flex-direction:column`. The alignment bar sits between the xfade panel and `#main-container`.

- [ ] **Step 3: Add alignment bar HTML**

Between `</div>` (closing xfade-panel) and `<div id="main-container">`, add:

```html
<canvas id="alignment-bar"></canvas>
```

- [ ] **Step 4: Add timecodes state and fetch function to JS**

After the `xfadeAudioCtx` block (around line 350), add:

```javascript
  // ---- Alignment bar state ------------------------------------------------
  const SHAPE_BASE = 'http://localhost:7892';
  let _tcA = null;  // { beats, downbeats, onsets, duration, bpm } for crop A
  let _tcB = null;  // same for crop B
  // Beat-match transform for B (set when isBeatMatch changes)
  let _bmTransformB = null;  // { stretch, phase } or null

  async function fetchTimecodes(track, crop, slot) {
    if (!track || !crop) return;
    try {
      const r = await fetch(`${SHAPE_BASE}/api/timecodes?track=${encodeURIComponent(track)}&crop=${encodeURIComponent(crop)}`);
      if (!r.ok) return;
      const data = await r.json();
      if (slot === 'A') _tcA = data;
      else              _tcB = data;
      drawAlignmentBar();
    } catch(e) {}
  }

  function drawAlignmentBar() {
    const canvas = document.getElementById('alignment-bar');
    const W = canvas.width  = canvas.offsetWidth;
    const H = canvas.height = canvas.offsetHeight || 80;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    const LANE_H   = H / 2;
    const LABEL_W  = 18;

    function drawLane(tc, yOff, label, transform) {
      if (!tc) return;
      const dur = tc.duration;
      if (!dur) return;

      // Lane label
      ctx.fillStyle = '#555';
      ctx.font = '9px monospace';
      ctx.fillText(label, 2, yOff + LANE_H * 0.6);

      function toX(t) {
        let tDisplay = t;
        if (transform) tDisplay = t * transform.stretch + transform.phase;
        return LABEL_W + (tDisplay / dur) * (W - LABEL_W - 30);
      }

      function drawTicks(times, color, width) {
        if (!times || times.length === 0) return;
        ctx.strokeStyle = color;
        ctx.lineWidth   = width;
        ctx.beginPath();
        for (const t of times) {
          const x = toX(t);
          if (x < LABEL_W || x > W - 30) continue;
          ctx.moveTo(x, yOff);
          ctx.lineTo(x, yOff + LANE_H);
        }
        ctx.stroke();
      }

      drawTicks(tc.onsets,    'rgba(255,220,0,0.6)',   1);
      drawTicks(tc.beats,     'rgba(80,220,80,0.7)',   1.5);
      drawTicks(tc.downbeats, 'rgba(220,60,60,0.9)',   2);
    }

    drawLane(_tcA, 0,      'A', null);
    drawLane(_tcB, LANE_H, 'B', _bmTransformB);

    // Separator line
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, LANE_H); ctx.lineTo(W, LANE_H); ctx.stroke();

    // Legend (right edge)
    const legendX = W - 28;
    [['rgba(255,220,0,0.8)', 'O'], ['rgba(80,220,80,0.8)', 'B'], ['rgba(220,60,60,0.9)', 'D']].forEach(([c, lbl], i) => {
      ctx.fillStyle = c;
      ctx.fillRect(legendX, 4 + i * 14, 8, 2);
      ctx.fillStyle = '#666';
      ctx.font = '8px monospace';
      ctx.fillText(lbl, legendX + 10, 7 + i * 14);
    });
  }

  // ResizeObserver for alignment bar
  new ResizeObserver(() => drawAlignmentBar())
    .observe(document.getElementById('alignment-bar'));
```

- [ ] **Step 5: Trigger `fetchTimecodes` on crop selection**

Find the crop selection event listeners. When `cropA_sel` changes, add a call to `fetchTimecodes`:

After the existing `cropA_sel.addEventListener('change', ...)` handler (search for it), add inside it or after:

```javascript
  cropA_sel.addEventListener('change', () => {
    const track = trackA_sel.value;
    const opt   = cropA_sel.options[cropA_sel.selectedIndex];
    const crop  = opt ? opt.value : '';
    fetchTimecodes(track, crop, 'A');
    // ... existing handlers (renderPlot etc.) remain
  });
  cropB_sel.addEventListener('change', () => {
    const track = trackB_sel.value;
    const opt   = cropB_sel.options[cropB_sel.selectedIndex];
    const crop  = opt ? opt.value : '';
    fetchTimecodes(track, crop, 'B');
  });
```

**Note:** The existing `cropA_sel` may already have a `change` listener. Add the `fetchTimecodes` call inside the existing handler rather than adding a second one.

Also call `fetchTimecodes` when the track changes (which also repopulates crops):

After the `loadCropsA()` / `loadCropsB()` functions that populate crop selects, add a `fetchTimecodes` call after the first crop is selected.

- [ ] **Step 6: Verify in browser**

1. Select a track and crop for A and B.
2. The alignment bar should appear below the crossfade panel and show tick marks immediately.
3. Onsets = yellow, beats = green, downbeats = red.
4. If no sidecar data exists, the bar is blank (no crash).

- [ ] **Step 7: Commit**

```bash
git add plots/latent_shape_explorer/index.html
git commit -m "feat: add alignment bar with onset/beat/downbeat tick marks"
```

---

## Task 7: Frontend — beat match → alignment bar dynamic update

**Files:**
- Modify: `plots/latent_shape_explorer/index.html`

### Background
When Beat Match is toggled on, `computeBeatMatch()` returns `stretchB` and `shiftB` (semitone shift, not time shift). The alignment bar needs to apply a time-domain transform to crop B's ticks. The time stretch factor is `stretchB` (a ratio ≤ 1 meaning the audio is sped up slightly). Phase offset in seconds requires converting the beat phase using the crop's BPM.

The transform applied to B's timecodes: `t_display = t / stretchB`. (If crop B is being stretched by factor `stretchB`, its beats appear closer together; dividing by stretchB maps original beats to their post-stretch positions.)

If `stretchB === 1.0` and no phase adjustment is needed, `_bmTransformB` is null.

- [ ] **Step 1: Update `isBeatMatch` toggle to refresh alignment bar**

Find the `btnBeatMatch.addEventListener("click", ...)` block (line 342):

```javascript
  btnBeatMatch.addEventListener("click", () => {
    isBeatMatch = !isBeatMatch;
    btnBeatMatch.style.background = isBeatMatch ? "#5bc4ff" : "#444";
    btnBeatMatch.style.color      = isBeatMatch ? "#000" : "#eee";
    updateBmInfo();
    reloadXfadeIfPlaying();
  });
```

Replace with:

```javascript
  btnBeatMatch.addEventListener("click", () => {
    isBeatMatch = !isBeatMatch;
    btnBeatMatch.style.background = isBeatMatch ? "#5bc4ff" : "#444";
    btnBeatMatch.style.color      = isBeatMatch ? "#000" : "#eee";
    updateBmInfo();
    updateAlignmentTransform();
    reloadXfadeIfPlaying();
  });
```

- [ ] **Step 2: Add `updateAlignmentTransform()`**

After `updateBmInfo()`, add:

```javascript
  function updateAlignmentTransform() {
    if (!isBeatMatch) {
      _bmTransformB = null;
    } else {
      const tA = trackA_sel.value;
      const tB = trackB_sel.value;
      if (tA && tB) {
        const bm = computeBeatMatch(tA, tB);
        // stretchB < 1 means B is sped up; divide timecodes by stretchB to show
        // where beats land after stretching
        _bmTransformB = (bm.stretchB !== 1.0)
          ? { stretch: 1.0 / bm.stretchB, phase: 0 }
          : null;
      } else {
        _bmTransformB = null;
      }
    }
    drawAlignmentBar();
  }
```

Also call `updateAlignmentTransform()` whenever tracks are changed:

In `trackA_sel.addEventListener('change', ...)` and `trackB_sel.addEventListener('change', ...)`, add a call to `updateAlignmentTransform()`.

- [ ] **Step 3: Verify in browser**

1. Select two tracks with slightly different BPMs.
2. Click Beat Match.
3. Crop B's green/red ticks should shift so they align more closely with crop A's ticks.
4. Deactivating Beat Match should restore crop B's raw tick positions.

- [ ] **Step 4: Commit**

```bash
git add plots/latent_shape_explorer/index.html
git commit -m "feat: alignment bar dynamically updates when beat match is toggled"
```

---

## Task 8: Frontend — smart loop

**Files:**
- Modify: `plots/latent_shape_explorer/index.html`

### Background
Smart loop replaces the single `AudioBufferSourceNode` in `startXfadePlayback()` with a pre-scheduled equal-power crossfade scheduler. When the checkbox is off, behavior is unchanged (single source, no loop). The scheduler uses `AudioContext.currentTime` for sample-accurate scheduling.

- [ ] **Step 1: Add Smart loop checkbox to crossfade panel HTML**

In `<div id="xfade-panel">`, find the Beat Match section (the `xfade-section` that has `btn_beatmatch`). After the beat match info div, add:

```html
    <label style="font-size:11px;color:#ccc;display:flex;align-items:center;gap:5px;cursor:pointer;">
      <input type="checkbox" id="chk_smart_loop" style="accent-color:#00d2ff;">
      Smart loop
    </label>
    <span id="smart_loop_hint" style="font-size:10px;color:#888;display:none;"></span>
```

- [ ] **Step 2: Add smart loop globals and compute function**

After the `xfadeAnimFrameId` declaration, add:

```javascript
  const chkSmartLoop = document.getElementById('chk_smart_loop');
  const smartLoopHint = document.getElementById('smart_loop_hint');

  // Pre-computed equal-power curves (N=128 samples for 10ms at 48kHz)
  const _XFADE_N = 128;
  const _FADE_IN  = new Float32Array(_XFADE_N).map((_, i) => Math.sqrt(i / (_XFADE_N - 1)));
  const _FADE_OUT = new Float32Array(_XFADE_N).map((_, i) => Math.sqrt(1 - i / (_XFADE_N - 1)));

  // Stores active loop scheduler state
  let _loopScheduler = null;  // { loopEndSec, nextNodeScheduledAt, activeGain, activeSource }

  function computeLoopEnd(buf) {
    // Returns loop end in seconds, or null if not enough beat data.
    const beats = _tcA ? _tcA.beats : null;
    const beatsB = _tcB ? _tcB.beats : null;
    if (!beats || !beatsB || beats.length < 4 || beatsB.length < 4) return null;
    const barsA = Math.floor(beats.length / 4);
    const barsB = Math.floor(beatsB.length / 4);
    let loopBars = Math.min(barsA, barsB);
    // Round down to nearest multiple of 4
    loopBars = Math.floor(loopBars / 4) * 4;
    if (loopBars === 0) return null;
    const beatIdx = loopBars * 4;
    if (beatIdx < beats.length) return beats[beatIdx];
    // Fallback: use BPM
    const bpm = _tcA.bpm;
    if (!bpm) return null;
    return (loopBars * 4 / bpm) * 60;
  }
```

- [ ] **Step 3: Replace `startXfadePlayback()` with smart-loop–aware version**

Replace the entire `startXfadePlayback(offset)` function (lines 554–573) with:

```javascript
  function startXfadePlayback(offset) {
    if (!xfadeBuffer) return;
    const ctx = getXfadeCtx();
    if (xfadeSourceNode) { try { xfadeSourceNode.stop(); } catch(e){} }
    xfadeSourceNode = null;

    if (_loopScheduler) {
      clearTimeout(_loopScheduler.timer);
      _loopScheduler = null;
    }

    const XFADE_DUR = 0.010;  // 10ms

    if (chkSmartLoop.checked) {
      const loopEnd = computeLoopEnd(xfadeBuffer);
      if (loopEnd === null || loopEnd <= XFADE_DUR * 2) {
        // Not enough beat data — fall back to normal playback, warn user
        chkSmartLoop.checked = false;
        smartLoopHint.textContent = 'Not enough beat data for loop';
        smartLoopHint.style.display = 'inline';
        setTimeout(() => { smartLoopHint.style.display = 'none'; }, 3000);
        startXfadePlayback(offset);
        return;
      }
      smartLoopHint.style.display = 'none';

      // applyFadeIn=false on the first call (audio starts at full gain),
      // true on every subsequent recursive call (new node fades in).
      function scheduleLoopCycle(startAt, fromOffset, applyFadeIn) {
        const gainNode = ctx.createGain();
        gainNode.gain.value = applyFadeIn ? 0.0 : 1.0;
        gainNode.connect(ctx.destination);

        const src = ctx.createBufferSource();
        src.buffer = xfadeBuffer;
        src.connect(gainNode);

        const off = Math.max(0, fromOffset % loopEnd);
        src.start(startAt, off);

        // Apply equal-power fade-in if this is a loop continuation
        if (applyFadeIn) {
          gainNode.gain.setValueCurveAtTime(_FADE_IN, startAt, XFADE_DUR);
        }

        // Pre-compute the AudioContext timestamp at which the crossfade begins
        const crossfadeAt = startAt + (loopEnd - off) - XFADE_DUR;
        // Fire the setTimeout slightly early (50ms) to give scheduling headroom
        const fireIn = Math.max(0, (crossfadeAt - ctx.currentTime - 0.050) * 1000);

        const timer = setTimeout(() => {
          if (!chkSmartLoop.checked) return;
          // Apply equal-power fade-out to the current node at the pre-computed time
          gainNode.gain.setValueCurveAtTime(_FADE_OUT, crossfadeAt, XFADE_DUR);
          // Schedule the incoming node to start at the same pre-computed time
          scheduleLoopCycle(crossfadeAt, 0, true);
          // Stop this node cleanly after the crossfade window
          src.stop(crossfadeAt + XFADE_DUR * 1.5);
        }, fireIn);

        xfadeSourceNode = src;
        _loopScheduler = { timer, gainNode, src };
      }

      scheduleLoopCycle(ctx.currentTime, offset % loopEnd, false);
    } else {
      // Normal single-play
      const gainNode = ctx.createGain();
      gainNode.gain.value = 1;
      gainNode.connect(ctx.destination);
      const src = ctx.createBufferSource();
      src.buffer = xfadeBuffer;
      src.connect(gainNode);
      const off = Math.max(0, offset % xfadeBuffer.duration);
      src.start(0, off);
      xfadeSourceNode = src;
      xfadePlayStart = ctx.currentTime - off;
      src.onended = () => {
        if (xfadeSourceNode === src) {
          xfadeSourceNode = null;
          xfadeBuffer = null;
          btnXfadePlay.textContent = '✕ Play Crossfade';
          xfadeStatus.textContent = 'Ended';
        }
      };
    }

    btnXfadePlay.textContent = '⏸ Stop Crossfade';
  }
```

Also update the stop path in `btnXfadePlay.addEventListener("click")` to clear `_loopScheduler`:

```javascript
    if (xfadeSourceNode) {
      try { xfadeSourceNode.stop(); } catch(e) {}
      xfadeSourceNode = null;
      if (_loopScheduler) { clearTimeout(_loopScheduler.timer); _loopScheduler = null; }
      btnXfadePlay.textContent = '▶ Play Crossfade';
      xfadeStatus.textContent = 'Stopped';
      return;
    }
```

- [ ] **Step 4: Pass `smart_loop=0` to server when smart loop checkbox is on**

In `buildXfadeURL()`, after building the base URL, add:

```javascript
    // Client-side smart loop: tell server not to trim
    if (chkSmartLoop.checked) url += '&smart_loop=0';
```

- [ ] **Step 5: Verify in browser**

1. Select two crops with beat data, check Smart loop, click ▶ Play Crossfade.
2. Audio should loop cleanly without a click at the splice point.
3. Unchecking Smart loop mid-play stops the loop scheduler on next play.
4. With crops that have no beat data, smart loop auto-unchecks and shows the hint text.

- [ ] **Step 6: Commit**

```bash
git add plots/latent_shape_explorer/index.html
git commit -m "feat: client-side smart loop with equal-power crossfade at beat boundaries"
```

---

## Task 9: Frontend — ⊕ Avg button

**Files:**
- Modify: `plots/latent_shape_explorer/index.html`

### Background
The ⊕ Avg button (already added to the HTML in Task 4) fetches both the average audio (`/average?track=X` on port 7891) and the average PCA trajectory (`/api/average-shape?track=X` on port 7892) for track A. The audio is decoded and played through the existing Web Audio ctx. The 3D plot is updated by calling `Plotly.react()` with a single trace built from the returned points.

- [ ] **Step 1: Add avg button JS**

After the smart loop section, add:

```javascript
  // ---- ⊕ Avg button -------------------------------------------------------
  const btnAvg   = document.getElementById('btn_avg');
  const avgStatus = document.getElementById('avg_status');

  btnAvg.addEventListener('click', async () => {
    const track = trackA_sel.value;
    if (!track) { avgStatus.textContent = 'Select Track A first'; return; }

    btnAvg.disabled = true;
    avgStatus.textContent = 'Loading...';

    try {
      // Fetch audio and shape in parallel
      const [audioResp, shapeResp] = await Promise.all([
        fetch(`http://localhost:7891/average?track=${encodeURIComponent(track)}`),
        fetch(`${SHAPE_BASE}/api/average-shape?track=${encodeURIComponent(track)}`),
      ]);

      if (!audioResp.ok) throw new Error(`Audio: HTTP ${audioResp.status}`);
      if (!shapeResp.ok) throw new Error(`Shape: HTTP ${shapeResp.status}`);

      const [arrayBuf, shapeData] = await Promise.all([
        audioResp.arrayBuffer(),
        shapeResp.json(),
      ]);

      // Play audio
      const ctx = getXfadeCtx();
      const buf = await ctx.decodeAudioData(arrayBuf);
      if (xfadeSourceNode) { try { xfadeSourceNode.stop(); } catch(e){} xfadeSourceNode = null; }
      const gain = ctx.createGain();
      gain.gain.value = 1;
      gain.connect(ctx.destination);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(gain);
      src.start();
      xfadeSourceNode = src;
      xfadeBuffer     = buf;
      xfadePlayStart  = ctx.currentTime;
      btnXfadePlay.textContent = '⏸ Stop Crossfade';
      src.onended = () => {
        if (xfadeSourceNode === src) {
          xfadeSourceNode = null; xfadeBuffer = null;
          btnXfadePlay.textContent = '▶ Play Crossfade';
        }
      };

      // Update 3D plot with average trajectory
      const pts = shapeData.points;
      if (pts && pts.length > 0) {
        const x = pts.map(p => p[0]);
        const y = pts.map(p => p[1]);
        const z = pts.map(p => p[2]);
        const n = pts.length;
        const colors = Array.from({length: n}, (_, i) => i / Math.max(n-1, 1));
        const avgTrace = {
          type: 'scatter3d', mode: 'lines',
          x, y, z,
          line: { width: 4, color: colors, colorscale: [[0,'rgba(77,193,55,0.1)'],[1,'rgb(77,193,55)']] },
          name: `⊕ Avg ${track}`,
          hoverinfo: 'name',
        };
        // Add/replace avg trace — keep existing traces, append avg
        const plotDiv = document.getElementById('plot');
        const existingNames = (plotDiv.data || []).map(t => t.name);
        const avgIdx = existingNames.findIndex(n => n && n.startsWith('⊕ Avg'));
        if (avgIdx >= 0) {
          Plotly.restyle('plot', { x: [x], y: [y], z: [z] }, [avgIdx]);
        } else {
          Plotly.addTraces('plot', avgTrace);
        }
      }

      avgStatus.textContent = `${buf.duration.toFixed(1)}s`;
    } catch(e) {
      avgStatus.textContent = 'avg failed';
      console.error('⊕ Avg error:', e);
    } finally {
      btnAvg.disabled = false;
    }
  });
```

- [ ] **Step 2: Verify in browser**

1. Select a track with multiple crops.
2. Click ⊕ Avg — audio should play and a green trajectory should appear in the 3D plot.
3. If the track has no crops or an error occurs, `avg_status` shows "avg failed".
4. Clicking ⊕ Avg again while audio is playing stops the old audio and starts fresh.

- [ ] **Step 3: Commit**

```bash
git add plots/latent_shape_explorer/index.html
git commit -m "feat: add ⊕ Avg button — plays average latent audio and shows average trajectory"
```

---

## Final verification

- [ ] **Run all server tests**

```bash
python -m pytest tests/test_latent_explorer_endpoints.py -v
```

Expected: 7 PASSED

- [ ] **End-to-end browser test checklist**

With both servers running (`latent_server.py --port 7891` and `latent_shape_server.py --port 7892`):

1. ▶ Load button is gone from the header
2. Crossfade panel is visible by default
3. Mode switch: Full-mix shows single slider; Stem modes show four sliders
4. ▶ Play Crossfade works in all three modes
5. Camera position is preserved when changing visualization mode
6. Alignment bar shows tick marks for both crops after selecting A and B
7. Beat Match toggle redraws B's ticks with stretch transform
8. Smart loop checkbox: audio loops cleanly; unchecks gracefully if no beat data
9. ⊕ Avg: plays audio, adds green trace to 3D plot, shows "avg failed" on error
10. `/api/timecodes` returns 200 with partial data when sidecars are missing (check in browser DevTools)
