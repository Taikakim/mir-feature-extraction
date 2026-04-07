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
| Classification labels (genre, mood, instrument, etc.) | columns in `tracks.csv` (string-valued) | Tab 1 Classes mode |
| Correlation matrix + posters | `plots/latent_analysis/data/01_correlations.npz` | Tab 2 |
| PCA scores + loadings | `plots/latent_analysis/data/02_pca.npz` | Tab 2 |
| Latent cross-correlation | `plots/latent_analysis/data/03_xcorr.npz` | Tab 2 |
| Temporal features | `plots/latent_analysis/data/04_temporal.npz` | Tab 2 |
| Latent .npy files | `/run/media/kim/Lehto/latents/` | Tab 3 |
| Stem latent .npy files | `/run/media/kim/Lehto/latents_stems/` | Tab 3 |
| Source audio timecodes | `/run/media/kim/Mantu/ai-music/Goa_Separated/` | Tab 3 alignment bar |
| Global PCA 3D model | `models/global_pca_3d.npz` | Tab 3 trajectory |
| Latent correlations | `models/latent_correlations.json` | Tab 3 manipulation sliders |

`data.py` loads all of the above at startup (or lazily per-tab on first access). Missing files degrade gracefully: the affected tab shows a "run script X first" message.

---

## 5. Tab 1 — Dataset Explorer

Replaces `feature_explorer.html`.

### 5.1 View modes

Eight sub-modes, all using the same `dcc.Graph(id="dataset-graph")` component. Switched via `dcc.RadioItems`. Controls toolbar above the graph shows only the controls relevant to the current mode.

| Mode | Plotly trace type | Key controls |
|------|------------------|--------------|
| Scatter | `Scattergl` | X, Y feature dropdowns; colour-by (continuous or outlier); trend toggle; log-axis toggles |
| Quadrant | `Scattergl` | X+, X−, Y+, Y− feature selectors; composite axes computed as norm(X+) − norm(X−) |
| Histogram | `Histogram` | Feature dropdown; bin count input |
| Radar | `Barpolar` (stacked) | Track selector (synced with global active track); click bar segments to set range filters |
| Heatmap | `Heatmap` | Feature group filter; shows top-N features; click cell → Scatter for that pair |
| Parallel | `Parcoords` | Uses curated feature set; drag band → track list |
| Similarity | `Scattergl` | Reference track (= active track); X+/X−/Y+/Y− selectors; NN panel |
| Classes | `Scattergl` | X, Y; class-by dropdown (genre, mood, instrument, label, etc.); trend per class toggle |

### 5.2 Sidebar

Two panels toggled by tab or always-visible split:
- **Track list** — populated by lasso, histogram click, parallel drag, or radar filter. Shows track name + streaming link (Spotify/Tidal/MusicBrainz selector). Click row → set active track → switch to Radar.
- **NN panel** — shown in Similarity mode. Top-20 nearest neighbours by up to 4 user-selected features. Mini feature bars per neighbour (track value vs. reference, same pattern as existing HTML).

### 5.3 Active track state

Global `dcc.Store(id="active-track")` holds `{track, idx}`. Updated by:
- Hover over scatter point (sets A slot silently)
- Click scatter point (sets A slot + switches to Radar)
- Double-click scatter point (sets B slot)
- Click row in track list or NN panel

### 5.4 Cross-tab integration with Analysis

A "Show clusters" toggle in the toolbar (only visible when `03_xcorr.npz` is loaded) colours the Scatter by Ward cluster assignment. Cluster colour palette matches Tab 2 Cluster Map.

---

## 6. Tab 2 — Analysis

Replaces `plots/latent_analysis/app.py` one-for-one. Sub-tabs:

1. Correlation Matrix
2. Feature Posters
3. PCA Explorer
4. Temporal
5. Latent Cross-Corr
6. Cluster Map

Implementation is a direct port of the existing `app.py` code into `tabs/analysis.py`. All fixes already applied (suppress_callback_exceptions, figure output instead of children, help text) carry over.

**New in this version:** Cluster Map tab gains a "Highlight in Dataset" button per cluster row — writes cluster's dim list into a `dcc.Store` that the Dataset tab reads to overlay coloured rings on the scatter.

---

## 7. Tab 3 — Viewer

Replaces `plots/latent_shape_explorer/index.html`.

### 7.1 Controls

- Track A / Crop A, Track B / Crop B dropdowns (populated from latent dir scan)
- "Avg all crops" checkbox — averages all crops of the track rather than using a single crop
- Visualization mode: Raw Trajectories, Hybrid 3D (Time=Z), Difference (Full Mix − Stem), Moving Average
- Stem toggles: Full Mix / Drums / Bass / Other / Vocals

### 7.2 3D Trajectory Plot

`dcc.Graph` with `go.Scatter3d`. PCA projection computed server-side in `data.py` using `models/global_pca_3d.npz`. Track B overlaid in contrasting colour. Colour gradient dark→bright = start→end of crop.

### 7.3 Alignment Bar

`dcc.Graph` with `go.Bar` or `go.Scatter` shapes approximating the current canvas strip (beat / downbeat / onset markers). Timecodes fetched from `source_dir` via helper in `data.py`. Two horizontal lanes: Track A (top), Track B (bottom).

### 7.4 Crossfader Panel

Two modes selectable via radio:
- **Simple** (Full-mix): single α slider, Slerp/Lerp toggle
- **Full**: per-stem α sliders, β_A/β_B anchors, Beat Match button, Smart Loop toggle

Beat Match reads BPM + HPCP directly from crop `.json` sidecars (not `tracks.csv`) so all tracks are covered regardless of CSV coverage.

### 7.5 Latent Manipulation Sliders

Five sliders (Brightness, Bass Energy, Danceability, Hardness, Female Voice) loaded from `models/latent_correlations.json`. Offset added to latent vector before decode.

---

## 8. Persistent Latent Player Strip

A collapsible panel fixed at the bottom of the app, always visible regardless of active tab.

- **A slot**: track name label, position slider (0–1), ▶/■ buttons
- **B slot**: track name label (shown when set), α crossfade slider
- Audio decode calls go to `latent_server.py` port 7891

Audio playback uses a **Dash clientside callback** (`assets/player.js`): the Python side puts a JSON payload into `dcc.Store(id="player-cmd")` and the JS reads it to control an `AudioContext`. This avoids blocking the Dash server on audio streaming. The Web Audio API loop-point control (smart loop) lives entirely in JS.

The A and B slots are populated from `active-track` Store (click/double-click in Dataset, or Track A/B selection in Viewer). Changes in either tab update the same Store.

---

## 9. File Layout (final)

```
plots/explorer/
  app.py
  data.py
  audio.py
  tabs/
    __init__.py
    dataset.py
    analysis.py
    viewer.py
  assets/
    player.js
    style.css
```

`latent_player.ini` config file is read by `data.py` for `latent_dir`, `stem_dir`, `raw_audio_dir`, `source_dir` — same keys as today, no format change.

---

## 10. Migration / Retirement Plan

| Old artefact | Action |
|-------------|--------|
| `plots/latent_analysis/app.py` | Replaced by `plots/explorer/tabs/analysis.py` + `app.py`. File deleted. |
| `scripts/latent_shape_server.py` | Logic absorbed into `data.py` + `tabs/viewer.py`. File deleted. |
| `plots/feature_explorer.html` + JS data files | Retired. Kept in git history. Not deleted from working tree until new app is confirmed working. |
| `plots/latent_shape_explorer/index.html` | Same as above. |
| Port 7892 | Freed. |
| Port 7895 | Taken by new unified app. |

---

## 11. Build Order

Each phase is independently runnable/testable.

1. **Phase 1 — Dataset tab**: `data.py` (CSV load), `tabs/dataset.py` (all 8 modes + sidebar), `app.py` skeleton. No audio yet.
2. **Phase 2 — Analysis tab**: Port `plots/latent_analysis/app.py` into `tabs/analysis.py`. Add cluster→Dataset wiring.
3. **Phase 3 — Player strip**: `assets/player.js` clientside callback. Simple A-slot decode only.
4. **Phase 4 — Viewer tab**: 3D trajectory, alignment bar, crossfader (simple then full).

---

## 12. Known Constraints

- Audio playback requires `latent_server.py` running on port 7891. Player strip degrades gracefully (greyed out) when 7891 is unreachable.
- Beat Match and timecodes require drives mounted (`source_dir`). Viewer degrades gracefully when absent.
- 3D PCA model (`models/global_pca_3d.npz`) fits on first launch if absent (~1 min). Subsequent starts are instant.
- `feature_explorer_captions.js` (AI caption overlay) is deferred — not in scope for initial implementation.
