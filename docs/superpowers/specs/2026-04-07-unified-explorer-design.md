# Unified MIR Explorer — Design Spec
**Date:** 2026-04-07
**Branch:** `unified-explorer`
**Status:** Awaiting user review
**Replaces:** `plots/latent_analysis/app.py` (port 7895), `plots/feature_explorer.html`, `plots/latent_shape_explorer/index.html`, `scripts/latent_shape_server.py`

---

## 1. Goal

Replace three separate, hard-to-maintain tools (monolithic HTML Feature Explorer, monolithic HTML Shape Explorer, Plotly Dash latent analysis app) with a single modular Plotly Dash application at port 7895. The GPU VAE decode server (`scripts/latent_server.py`, port 7891) remains separate and unchanged.

---

## 2. High-Level Architecture

```
plots/explorer/
  app.py                  # entry point — Dash app, layout, top-level routing
  data.py                 # all data loading (CSV, NPZ, latent dir scanning)
  audio.py                # HTTP proxy helpers to latent_server.py (port 7891)
  latch.py                # LatCH inference hook (gradient-based latent guidance)
  tabs/
    dataset.py            # Tab 1 — 8-mode dataset scatter + sidebar
    analysis.py           # Tab 2 — 6-panel correlation/PCA/cluster analysis
    viewer.py             # Tab 3 — 3D latent trajectory + full crossfader
  assets/                 # CSS, clientside JS for audio playback
    player.js             # Web Audio API clientside callback
    style.css
```

`scripts/latent_shape_server.py` is deleted; its logic moves into `data.py` and `tabs/viewer.py` (called directly, no HTTP hop).

The Feature Explorer HTML files (`feature_explorer.html`, `feature_explorer_data.js`, `feature_explorer_classes.js`, `feature_explorer_captions.js`) are retired but kept in git history.

---

## 3. Servers

| Process | Port | Deps | Purpose |
|---------|------|------|---------|
| `python plots/explorer/app.py` | 7895 | numpy, pandas, plotly, dash | Unified UI — all tabs |
| `python scripts/latent_server.py` | 7891 | PyTorch ROCm, GPU | VAE decode — unchanged |

The old shape server (7892) is retired. The browser never needs to talk to 7892 again; it only talks to 7895 (UI) and 7891 (audio decode).

---

## 4. Data Sources

| Data | File | Used by |
|------|------|---------|
| Per-track MIR features (numeric) | `plots/tracks.csv` | Tab 1, Tab 2 PCA scatter |
| Classification labels (genre, mood, instrument, etc.) | string columns in `tracks.csv` | Tab 1 Classes mode |
| Correlation matrix + posters | `plots/latent_analysis/data/01_correlations.npz` | Tab 2 |
| PCA scores + loadings | `plots/latent_analysis/data/02_pca.npz` | Tab 2 |
| Latent cross-correlation + cluster labels | `plots/latent_analysis/data/03_xcorr.npz` | Tab 2, Tab 1 cluster overlay |
| Temporal features | `plots/latent_analysis/data/04_temporal.npz` | Tab 2 |
| Latent .npy files | `/run/media/kim/Lehto/latents/` | Tab 3 |
| Stem latent .npy files | `/run/media/kim/Lehto/latents_stems/` | Tab 3 |
| Source audio timecodes | `/run/media/kim/Mantu/ai-music/Goa_Separated/` | Tab 3 alignment bar |
| Global PCA 3D model | `models/global_pca_3d.npz` | Tab 3 trajectory |
| Latent correlations (fallback) | `models/latent_correlations.json` | Tab 3 manipulation (fallback only) |
| LatCH checkpoints | `models/latch/{feature}.pt` | Tab 3 manipulation (primary when present) |

`data.py` loads all of the above at startup (or lazily per-tab on first access). Missing files degrade gracefully: the affected feature shows a "run script X first" message or silently omits that mode.

---

## 5. Tab 1 — Dataset Explorer

Replaces `feature_explorer.html`.

### 5.1 View modes

Eight sub-modes, all using the same `dcc.Graph(id="dataset-graph")` component. Switched via `dcc.RadioItems`. The controls toolbar above the graph shows only controls relevant to the active mode.

| Mode | Plotly trace type | Key controls |
|------|------------------|--------------|
| Scatter | `Scattergl` | X, Y feature dropdowns; colour-by (continuous or outlier); trend toggle; log-axis toggles |
| Quadrant | `Scattergl` | X+, X−, Y+, Y− feature selectors; composite axes = norm(X+) − norm(X−) |
| Histogram | `Histogram` | Feature dropdown; bin count input |
| Radar | `Barpolar` (stacked) | Track selector synced to active track; click bar segments to set range filters |
| Heatmap | `Heatmap` | Feature group filter; top-N features; click cell → Scatter for that pair |
| Parallel | `Parcoords` | Curated feature set; drag band → track list |
| Similarity | `Scattergl` | Reference = active track; X+/X−/Y+/Y− selectors; NN panel in sidebar |
| Classes | `Scattergl` | X, Y; class-by dropdown (genre, mood, instrument, label, source); trend per class toggle |

### 5.2 Sidebar

Always-visible right panel, two sections:

**Track list** — populated by lasso selection, histogram bar click, parallel coordinate band, or Radar range filter. Shows track name + streaming link (Spotify/Tidal/MusicBrainz selector). Clicking a row:
- Sets the active track
- Switches to **Similarity** mode (not Radar)

**NN panel** — visible in Similarity mode. Top-20 nearest neighbours by up to 4 user-selected feature dimensions. Mini feature bar per neighbour (track value vs. reference value). Clicking a neighbour row sets it as new reference.

### 5.3 Active track state

`dcc.Store(id="active-track")` holds `{track, track_idx, slot}` where `slot` is `"a"` or `"b"`.

Update rules:
- **Hover** over scatter point → sets A slot silently, triggers autoplay **only if** the "Autoplay on hover" checkbox is enabled. Audio fades out in 200 ms when the cursor moves more than 10 px from the hovered point or exits the plot area, whichever occurs first. Respects the Smart Loop checkbox.
- **Single click** scatter point → sets A slot; does **not** change the current view mode. The clicked track becomes the focus track when switching to any other tab.
- **Double-click** scatter point → sets B slot.
- **Click track list / NN row** → sets A slot + switches to Similarity mode.

### 5.4 Track search

The primary track selector dropdown (Track A and Track B throughout the app) supports **pattern search**: typing filters tracks dynamically by substring match against `%query%` in both the track name and the artist field. Implemented as a `dcc.Dropdown` with `search_value` + server-side filtering callback returning options matching `f"%{q}%"` case-insensitively in `track` or `artists` columns.

### 5.5 Cross-tab state

- When the user switches to the Viewer tab, Track A and Track B slots carry over from the last Dataset state automatically via the shared `active-track` Store.
- When the user switches back to Dataset, the Viewer's A/B state is reflected (e.g., the active track is highlighted on the scatter).

### 5.6 Cluster overlay

A "Show clusters" toggle in the toolbar (shown only when `03_xcorr.npz` is loaded) colours the Scatter by Ward cluster assignment. Colour palette is consistent with Tab 2 Cluster Map.

---

## 6. Tab 2 — Analysis

Replaces `plots/latent_analysis/app.py` one-for-one. Sub-tabs:

1. Correlation Matrix
2. Feature Posters
3. PCA Explorer
4. Temporal
5. Latent Cross-Corr
6. Cluster Map

Direct port of existing code into `tabs/analysis.py`. All fixes already applied (`suppress_callback_exceptions=True`, `figure` output instead of `children`, help text paragraphs) carry over unchanged.

**New:** Cluster Map sub-tab gains a "Highlight in Dataset" button per cluster row — writes that cluster's dim indices into a `dcc.Store(id="cluster-highlight")` that the Dataset tab reads to colour the scatter cluster overlay.

---

## 7. Tab 3 — Viewer

Replaces `plots/latent_shape_explorer/index.html`.

### 7.1 Track / crop controls

- Track A / Crop A, Track B / Crop B dropdowns (populated by latent dir scan, same search-by-pattern as Dataset)
- Slots pre-populated from `active-track` Store on tab switch
- **"Avg all crops" checkbox** — when enabled, averages all crops of a track instead of using a single crop. Only crops that contain enough downbeat-delineated content for at least one 4-bar loop are included; shorter crops are zero-padded to the longest common loop boundary. This is a persistent checkbox state (stored in `dcc.Store`).
- Visualization mode: Raw Trajectories, Hybrid 3D (Time=Z), Difference (Full Mix − Stem), Moving Average
- Stem toggles: Full Mix / Drums / Bass / Other / Vocals

### 7.2 3D Trajectory Plot

`dcc.Graph` with `go.Scatter3d`. PCA projection computed server-side in `data.py` using `models/global_pca_3d.npz`. Track B overlaid in a contrasting colour. Colour gradient dark→bright encodes time (start→end of crop).

### 7.3 Alignment Bar

`dcc.Graph` with Plotly shapes/markers approximating the current canvas strip: yellow O = onset, green B = beat, red D = downbeat. Two horizontal lanes (Track A top, Track B bottom). Timecodes loaded from `source_dir` crop sidecars via `data.py`. When Beat Match is active, Track B's lane is time-stretched to reflect the BPM correction visually.

### 7.4 Crossfader Panel

Two modes via radio:

**Simple (Full-mix):** Single α slider (0 = all A, 1 = all B). Slerp/Lerp toggle. Smart Loop checkbox. Beat Match button. No stem controls.

**Full (Advanced):**
- Per-stem α sliders (Drums, Bass, Other, Vocals) for stem-latent crossfade
- Per-cluster crossfade weights (one slider per Ward cluster from `03_xcorr.npz`, labelled by that cluster's top-correlated feature name). This replaces the old β_A/β_B "reality anchors". Each cluster slider sets how much of the blending applies to that group of dims independently. Dims not included in any active cluster range snap to Track A's values as baseline.
- **Dim regex field** — a text input accepting a Python-style range/set expression (e.g., `0-15,32,48-63`) specifying which dim indices participate in the crossfade. Dims outside the expression are held at Track A's values. An info box below explains the syntax. The regex field and per-cluster sliders are **alternative modes** (radio: "by cluster" | "by dim range"); they control the same underlying dim mask.
- Beat Match button (reads BPM + HPCP from crop `.json` sidecars, not `tracks.csv`, ensuring full coverage)
- Smart Loop checkbox

### 7.5 Latent Manipulation Panel

Named sliders (Brightness, Bass Energy, Danceability, Hardness, Female Voice) that nudge the latent before decode.

**Backend dispatch in `latch.py`:**
1. If `models/latch/{feature}.pt` exists → load the trained LatCH checkpoint and compute the gradient `∂prediction/∂latent` at low noise (t ≈ 0). Use this gradient direction scaled by the slider value as the latent offset. This is a single forward+backward pass on CPU/GPU — no full diffusion run required.
2. If no LatCH checkpoint → fall back to the correlation-coefficient offset from `models/latent_correlations.json` (current behaviour).

The UI is identical in both cases; the backend transparently upgrades when a checkpoint is available. This is the forward-compatibility hook for the LatCH paper integration.

**Future (Phase 2, out of scope for initial build):** Full LatCH-guided generation starting from noise using the Euler sampler TFG loop in `/home/kim/Projects/SAO/stable-audio-tools/scripts/generate_latch_guided.py`. Training loop: `train_latch.py --feature <name> --epochs N`. Trained checkpoints saved to `models/latch/{feature}.pt`.

---

## 8. Persistent Latent Player Strip

A collapsible panel fixed at the bottom of the app, always visible regardless of active tab.

**A slot:** track name, position slider (0–1), ▶/■ buttons, Autoplay-on-hover checkbox, Smart Loop checkbox.

**B slot:** track name (shown when set), α crossfade slider.

Audio playback uses a **Dash clientside callback** (`assets/player.js`): the Python side writes a JSON command payload to `dcc.Store(id="player-cmd")`; the JS reads it to control an `AudioContext`. This keeps audio streaming and loop scheduling entirely in the browser.

Autoplay-on-hover: when the checkbox is enabled and a hover event fires from the Dataset scatter (or Similarity scatter), the player receives a play command with the hovered track's A-slot latent URL. A 200 ms fade-out fires when the cursor moves >10 px from the hover point or the `plotly_unhover` event fires, whichever is first. Smart Loop (if checked) applies the loop-point calculation to the hover playback as well.

The A and B slots stay in sync across tabs via the shared `active-track` Store.

---

## 9. File Layout (final)

```
plots/explorer/
  app.py
  data.py
  audio.py
  latch.py
  tabs/
    __init__.py
    dataset.py
    analysis.py
    viewer.py
  assets/
    player.js
    style.css
```

`latent_player.ini` is read by `data.py` for `latent_dir`, `stem_dir`, `raw_audio_dir`, `source_dir` — same keys as today, no format change.

LatCH source files referenced (read-only, not copied):
- `/home/kim/Projects/SAO/stable-audio-tools/scripts/latch_model.py`
- `/home/kim/Projects/SAO/stable-audio-tools/scripts/generate_latch_guided.py`

---

## 10. Migration / Retirement Plan

| Old artefact | Action |
|-------------|--------|
| `plots/latent_analysis/app.py` | Replaced by `plots/explorer/tabs/analysis.py`. Deleted after Phase 2 confirmed working. |
| `scripts/latent_shape_server.py` | Logic absorbed into `data.py` + `tabs/viewer.py`. Deleted after Phase 4 confirmed working. |
| `plots/feature_explorer.html` + JS data files | Retired. Kept in working tree until Phase 1 confirmed working, then deleted. |
| `plots/latent_shape_explorer/index.html` | Same. Deleted after Phase 4. |
| Port 7892 | Freed when `latent_shape_server.py` is deleted. |
| Port 7895 | Taken over by new unified app immediately. |

---

## 11. Build Order

Each phase is independently runnable and testable before the next begins.

1. **Phase 1 — Dataset tab:** `data.py` (CSV load, feature cols, class cols), `tabs/dataset.py` (all 8 modes + sidebar + search + hover autoplay), `app.py` skeleton with player strip stub. No audio decode yet.
2. **Phase 2 — Analysis tab:** Port `plots/latent_analysis/app.py` into `tabs/analysis.py`. Add cluster→Dataset highlight wiring.
3. **Phase 3 — Player strip:** `assets/player.js` clientside callback, `audio.py` helpers. A-slot decode, autoplay-on-hover, fade-out, smart loop.
4. **Phase 4 — Viewer tab:** 3D trajectory, alignment bar, simple crossfader, then full crossfader with per-cluster sliders + dim regex field. `latch.py` stub ready for LatCH checkpoints.

---

## 12. Known Constraints and Deferred Items

- Audio requires `latent_server.py` on port 7891. Player strip degrades gracefully (greyed out) when unreachable.
- Beat Match and timecodes require drives mounted at `source_dir`. Viewer degrades gracefully when absent.
- 3D PCA model fits on first launch if absent (~1 min). Subsequent starts instant.
- **Captions** (`feature_explorer_captions.js` AI overlay) — deferred, not in scope for initial build.
- **LatCH Phase 2** (full guided generation via Euler TFG) — deferred; training and inference code exists at `/home/kim/Projects/SAO/stable-audio-tools/scripts/`. Checkpoints go to `models/latch/{feature}.pt` when trained.
- `feature_explorer_data.js` is the old embedded data blob. It is fully superseded by `plots/tracks.csv` + live CSV loading in `data.py`; the JS file is not referenced by the new app.
