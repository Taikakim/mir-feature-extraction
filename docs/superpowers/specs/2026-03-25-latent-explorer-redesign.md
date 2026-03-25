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

- **Full-mix**: interpolates the two full-mix latents directly with a single α slider (0 = A, 1 = B); decoded once. Corresponds to the current `mode=latent` path using full-mix latents rather than stems.
- **Stem latents**: per-stem α sliders; each stem pair is slerp'd in latent space, the four stem latents are summed, decoded once. Corresponds to current `mode=latent` with stems.
- **Stem audio**: per-stem α sliders; each stem pair is slerp'd, each decoded to audio individually, then downmixed. Corresponds to current `mode=stems`.

In **Full-mix** mode, the four per-stem sliders are hidden; only the single α slider is shown. In both Stem modes, the four per-stem sliders are shown and the single α slider is hidden.

The UI style throughout follows an explicit, text-label-first approach — no icon-only controls, no hidden features.

### ⊕ Avg button
A new `⊕ Avg` button in the header row (beside `▶ Crossfade`). On click:
1. Fetches `/api/audio/average?track=<trackA>` from `latent_server.py` and plays the returned WAV.
2. Fetches `/api/average-shape?track=<trackA>` from `latent_shape_server.py` and updates the 3D plot with the average trajectory.

### Camera state fix
The Three.js `OrbitControls` instance must not be recreated or reset when the user switches view modes (rotate / zoom / pan). Only the interaction flags (`enableRotate`, `enableZoom`, `enablePan`) are toggled; the camera position and target are preserved.

---

## 2. Smart Loop (Client-side)

When the **Smart loop** checkbox is checked, the following runs after the audio buffer is decoded in the browser:

1. **Loop region calculation**: Using the fetched beat timecodes for both crops, compute the number of complete 4-beat bars that fit within each crop's duration. Take the minimum across A and B, then find the largest multiple of 4 bars — this is the loop length in beats. Convert to sample frames: `frames = (beats / bpm * 60) * sampleRate`.
2. **Scheduling**: Playback uses a pair of `AudioBufferSourceNode` + `GainNode`. Node 1 starts playing from frame 0. At `loopEnd - 0.010` seconds (10ms before the splice point), Node 2 is started from frame 0 with gain 0. Over the next 10ms:
   - Node 1 gain ramps from 1 → 0 using `linearRampToValueAtTime`.
   - Node 2 gain ramps from 0 → 1 using `linearRampToValueAtTime`.
   - Both curves use equal-power shape: gain = `Math.sqrt(t)` for the fade-in, `Math.sqrt(1 - t)` for the fade-out.
3. After the crossfade completes, Node 1 is stopped and discarded. Node 2 becomes the active node. The cycle repeats: a new node is pre-scheduled before each loop point.
4. When Smart loop is off, a single `AudioBufferSourceNode` plays once with `loop = false`.

This is entirely client-side — no extra server round-trips per loop cycle.

---

## 3. Alignment Bar

### Placement and layout
A full-width `<canvas>` element is inserted between the crossfade panel and the 3D plot. Height: ~80px. It is divided into two equal horizontal lanes:
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

#### `GET /api/timecodes?track=X&crop=Y`
Reads the source track's sidecar files:
- `.BEATS_GRID` → beat timestamps
- `.DOWNBEATS` → downbeat timestamps
- `.ONSETS` → onset timestamps

Each sidecar file contains one float (seconds) per line. The crop's JSON sidecar provides `position` (start time within the track) and `duration`. Timecodes are filtered to the crop's time window and returned as arrays of floats relative to the crop start (i.e., `t - position`).

Response:
```json
{
  "beats":     [0.0, 0.46, 0.93, ...],
  "downbeats": [0.0, 1.86, 3.72, ...],
  "onsets":    [0.0, 0.12, 0.23, ...],
  "duration":  30.0
}
```

Returns 404 if any sidecar file is missing; the alignment bar silently hides that layer.

#### `GET /api/average-shape?track=X`
Loads all full-mix `.npy` latent files for track X (excluding stem suffixes). Computes frame-wise mean across all crops (shorter crops are zero-padded to the longest). Projects the mean latent through the global PCA model. Returns:

```json
{ "points": [[x, y, z], ...] }
```

Same shape as the existing `/api/shape` response for `fullmix`. Returns 404 if the track has no latent files.

### `latent_server.py` (port 7891)

#### `GET /api/audio/average?track=X`
Loads all full-mix latent `.npy` files for track X. Computes frame-wise mean (zero-pad shorter files). Decodes the averaged latent through the VAE once. Peak-normalises to 0.9. Returns a WAV file (`audio/wav`). No `X-Crossfade-Meta` header is set on this response.

Optional query param `stems=drums,bass,other,vocals` (for future use): if provided, averages the named stem latents instead and decodes each separately, then sums. Not required for initial implementation.

---

## 5. Error Handling

- If `/api/timecodes` returns 404 (sidecar missing), the alignment bar draws what it has — missing layers are silently skipped; a "No [onsets/beats/downbeats] data" label appears in the affected lane.
- If `/api/audio/average` or `/api/average-shape` returns an error, the ⊕ Avg button shows a brief inline error label ("avg failed") and re-enables.
- Smart loop gracefully degrades: if beat data is unavailable or the computed loop length is zero, the loop checkbox is auto-unchecked and a tooltip explains why.
- The camera state fix must not regress: if `OrbitControls` is ever re-instantiated (e.g. on canvas resize), the current camera position/quaternion/target must be saved and reapplied.

---

## 6. Files Changed

| File | Change |
|---|---|
| `plots/latent_shape_explorer/index.html` | UI restructure, mode switch, smart loop, alignment bar, camera fix, ⊕ Avg button |
| `scripts/latent_server.py` | New `/api/audio/average` endpoint |
| `scripts/latent_shape_server.py` | New `/api/timecodes` and `/api/average-shape` endpoints |

No new files are created.
