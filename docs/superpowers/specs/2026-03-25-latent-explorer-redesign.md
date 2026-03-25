# Latent Shape Explorer — Redesign Spec

**Date:** 2026-03-25
**Scope:** `plots/latent_shape_explorer/index.html`, `scripts/latent_server.py`, `scripts/latent_shape_server.py`

---

## Overview

Simplify the latent crossfade player by removing single-crop playback, adding a mode switch for three crossfade strategies, introducing a play-average feature, a client-side smart loop, and a full-width alignment bar showing beat/onset tick marks.

---

## 1. UI Structure Changes

### Remove ▶ Load button
The separate single-crop "▶ Load" play button is removed from the header. Pure A or pure B playback is achieved by setting the crossfader α to 0 or 1.

### Crossfade panel — always visible
The crossfade panel (currently toggle-hidden) is promoted to always-visible. No toggle button needed.

### Mode switch — segmented button bar
A three-segment button bar replaces the mode dropdown inside the crossfade panel:

```
[ Full-mix ]  [ Stem latents ]  [ Stem audio ]
```

Mode definitions and the corresponding server parameters sent on each `/crossfade` request:

| UI label | Description | `mode=` | `raw=` |
|---|---|---|---|
| Full-mix | Interpolates the two full-mix latents directly with a single α slider; decoded once | `ab` | `0` |
| Stem latents | Per-stem α sliders; each stem pair slerp'd in latent space, four latents summed, decoded once | `latent` | `0` |
| Stem audio | Per-stem α sliders; each stem pair slerp'd, each decoded to audio individually, downmixed | `stems` | `0` |

In **Full-mix** mode, the four per-stem sliders are hidden; only the single α slider is shown. In both Stem modes, the four per-stem sliders are shown and the single α slider is hidden.

The UI style throughout follows an explicit, text-label-first approach — no icon-only controls, no hidden features.

### ⊕ Avg button
A new `⊕ Avg` button in the header row (beside `▶ Crossfade`). On click it operates on **track A only** (this is a deliberate design choice — the button is a diagnostic of track A's character). It:

1. Fetches `/average?track=<trackA>` from `latent_server.py` (port 7891) and plays the returned WAV.
2. Fetches `/api/average-shape?track=<trackA>` from `latent_shape_server.py` (port 7892) and updates the 3D plot with the average trajectory.

### Camera state fix
The 3D plot uses Plotly's `scatter3d`. When the user switches view modes (rotate / zoom / pan) via the radio buttons, the current code recreates or resets the Plotly camera. The fix: before any view-mode switch, read the current camera state from `plot._fullLayout.scene.camera` and after the mode switch reapply it via `Plotly.relayout(plotDiv, { 'scene.camera': savedCamera })`. Camera position and target survive mode switches intact.

### Existing waveform canvas
The existing `<canvas id="waveform_canvas">` (600 × 100px, absolutely positioned overlay) is removed entirely. It is replaced by the new alignment bar described in Section 3.

---

## 2. Smart Loop (Client-side)

When the **Smart loop** checkbox is checked, the following runs after the audio buffer is decoded in the browser:

### Server interaction
When Smart loop is on, the client sends `smart_loop=0` in the crossfade request to disable the server-side WAV-trim behaviour. The client receives a full-duration WAV and applies loop logic itself.

### Loop region calculation
Using the fetched beat timecodes for both crops (from `/api/timecodes`):

1. Count the number of complete 4-beat bars that fit within each crop's duration. A bar is 4 consecutive beats from the `beats` array. The bar count for a crop is `Math.floor(beats.length / 4)`.
2. Take the minimum bar count across A and B.
3. Find the largest multiple of 4 that is ≤ that minimum. This is the loop length in bars.
4. The loop endpoint in seconds: `loopEnd = beats[loopBars * 4]` (the timestamp of the beat at position `loopBars * 4` in the array). If the index is out of range, fall back to `(loopBars * 4 / bpm) * 60`.
5. Convert to sample frames: `loopFrames = Math.round(loopEnd * sampleRate)`.

`bpm` comes from the `bpm` field of the `/api/timecodes` response (see Section 4). `sampleRate` comes from the `AudioBuffer`.

If the computed loop length is zero (too few beats) or the beat data is unavailable, the Smart loop checkbox is auto-unchecked and a tooltip "Not enough beat data for loop" is shown inline.

### Scheduling
Playback uses a pair of `AudioBufferSourceNode` + `GainNode`. Node 1 starts playing from frame 0. At `loopEnd - 0.010` seconds (10ms before the splice point), Node 2 is started from frame 0 with gain 0. Over the next 10ms, equal-power crossfade using `setValueCurveAtTime`:

- Node 2 gain applies a precomputed `Float32Array` of N samples where `curve[i] = Math.sqrt(i / (N-1))` (fade-in, 0 → 1).
- Node 1 gain applies a precomputed `Float32Array` of N samples where `curve[i] = Math.sqrt(1 - i / (N-1))` (fade-out, 1 → 0).
- N = 128 samples is sufficient resolution for a 10ms curve.

After the crossfade completes, Node 1 is stopped and discarded. Node 2 becomes the active node. The cycle repeats: a new node is pre-scheduled before each loop point.

When Smart loop is off, a single `AudioBufferSourceNode` plays once with `loop = false`. This is entirely client-side — no extra server round-trips per loop cycle.

---

## 3. Alignment Bar

### Placement and layout
A full-width `<canvas id="alignment_canvas">` element is inserted between the crossfade panel and the 3D plot. Height: 80px. It is divided into two equal horizontal lanes:

- **Top lane** — Track A
- **Bottom lane** — Track B

Each lane has a small text label ("A" / "B") on the left edge.

### Tick mark rendering
Three layers of vertical tick marks are drawn per lane, spanning the full lane height:

| Type | Colour | Width |
|---|---|---|
| Onsets | Yellow (`rgba(255,220,0,0.6)`) | 1px |
| Beats | Green (`rgba(80,220,80,0.7)`) | 1.5px |
| Downbeats | Red (`rgba(220,60,60,0.9)`) | 2px |

Time → x mapping: `x = (t / duration) * canvasWidth`.

A small legend (three coloured swatches with labels) sits at the right edge of the canvas.

### Data source
Timecodes are fetched from `/api/timecodes?track=X&crop=Y` on `latent_shape_server.py` immediately when a crop is selected — independent of playback. The canvas renders as soon as data arrives.

### Beat match integration
When **Beat Match** is toggled on, the beat-match computation produces a stretch factor and phase offset for crop B. The alignment bar redraws crop B's ticks with these transforms applied: `t_display = t * stretchFactor + phaseOffset`. When toggled off, raw timecodes are restored. The visual goal is that after beat-matching, the green and red ticks for A and B visually align.

### Resize handling
The canvas listens to a `ResizeObserver` and redraws on any width change.

---

## 4. Server Changes

### `latent_shape_server.py` (port 7892)

#### New config key: `source_dir`
A new `source_dir` key is added to the `[server]` section of `latent_player.ini`, mirroring the same key already present in `latent_server.py`. It points to the root directory containing full-track sidecar files (e.g. `/run/media/kim/Mantu/ai-music/Goa_Separated`). Sidecar paths are derived as:

```
source_dir / track_name / f"{track_name}.BEATS_GRID"
source_dir / track_name / f"{track_name}.DOWNBEATS"
source_dir / track_name / f"{track_name}.ONSETS"
```

A new `_source_dir` module global is set from this config key at startup. If not configured, it defaults to `None` and the `/api/timecodes` endpoint returns 404.

#### `GET /api/timecodes?track=X&crop=Y`
Reads the source track's sidecar files from `_source_dir / track / f"{track}.BEATS_GRID"` etc. Each sidecar file contains one float (seconds) per line.

The crop's JSON sidecar (in `_latent_dir / track / f"{crop}.json"`) provides `start_time` and `end_time` (absolute seconds within the source track). Timecodes are filtered to the crop's time window (`start_time <= t <= end_time`) and returned as arrays of floats relative to the crop start: `t - start_time`.

`bpm` is computed from the beats array: `bpm = 60 / median(diff(beats_in_window))`. If fewer than 2 beats fall in the window, `bpm` is `null`.

Response:
```json
{
  "beats":     [0.0, 0.46, 0.93, ...],
  "downbeats": [0.0, 1.86, 3.72, ...],
  "onsets":    [0.0, 0.12, 0.23, ...],
  "duration":  30.0,
  "bpm":       130.4
}
```

**Partial data handling:** if a sidecar file is missing, the corresponding key is returned as an empty array `[]` and a `"missing"` key lists which sidecars were absent. The endpoint returns 200 in all cases where the crop JSON exists. It returns 404 only if the crop is not found (no JSON sidecar) or `_source_dir` is not configured.

Example partial response when `.ONSETS` is missing:
```json
{
  "beats": [...], "downbeats": [...], "onsets": [],
  "duration": 30.0, "bpm": 130.4,
  "missing": ["onsets"]
}
```

The alignment bar silently skips rendering any layer whose array is empty.

#### `GET /api/average-shape?track=X`
Loads all full-mix `.npy` latent files for track X (excluding stem suffixes). Computes frame-wise mean across all crops (shorter crops are zero-padded to the longest). Projects the mean latent through the global PCA model. Returns:

```json
{ "points": [[x, y, z], ...] }
```

Same shape as the existing `/api/shape` response for `fullmix`. Returns 404 if the track has no latent files.

### `latent_server.py` (port 7891)

#### `GET /average?track=X`
Path follows the existing routing convention (`/status`, `/crossfade`, `/decode` — no `/api/` prefix).

Loads all full-mix latent `.npy` files for track X. Computes frame-wise mean (zero-pad shorter files). Decodes the averaged latent through the VAE once. Peak-normalises to 0.9. Returns a WAV file (`audio/wav`). No `X-Crossfade-Meta` header is set on this response.

---

## 5. Error Handling

- `/api/timecodes` returns 200 with partial data when sidecars are missing (see Section 4). Returns 404 if crop JSON is absent or `_source_dir` is unconfigured.
- If `/average` (audio average) or `/api/average-shape` returns an error, the ⊕ Avg button shows a brief inline error label ("avg failed") and re-enables.
- Smart loop degrades gracefully: if beat data is unavailable or the computed loop length is zero, the loop checkbox is auto-unchecked and a tooltip "Not enough beat data for loop" is shown inline.
- Camera state fix must not regress: if Plotly's `relayout` is called for any reason and would reset the camera, save and reapply `plot._fullLayout.scene.camera` as described in Section 1.

---

## 6. Files Changed

| File | Change |
|---|---|
| `plots/latent_shape_explorer/index.html` | UI restructure, mode switch, smart loop, alignment bar, camera fix, ⊕ Avg button, remove waveform canvas |
| `scripts/latent_server.py` | New `/average` endpoint |
| `scripts/latent_shape_server.py` | New `/api/timecodes` and `/api/average-shape` endpoints; new `source_dir` config key |

No new files are created.
