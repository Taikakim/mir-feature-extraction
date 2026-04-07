# Additional Tools

Standalone tools that live outside the main pipeline but are useful for dataset
exploration, algorithm comparison, and real-time audio work.

---

## Pitch Shifter GUI

**File:** `pitch_shifter_gui.py`
**Deps:** PyQt6, sounddevice, soundfile, librosa, and optionally bungee,
pyrubberband, pedalboard, sox

Interactive desktop app for comparing pitch-shift and time-stretch algorithm
quality on the dataset. Useful for deciding which algorithm to use for data
augmentation.

```bash
python pitch_shifter_gui.py
```

### What it does

- Scans `DATASET_DIR` (`Goa_Separated_crops`) for track folders containing FLAC
  crops.
- For each selected crop, reads its `.DOWNBEATS` file and loops the audio
  between the first and last downbeat (zero-crossing snapped), giving a clean
  seamless loop for listening.
- Renders the same loop through four algorithms in parallel background threads
  so you can switch instantly without waiting:

| Key | Algorithm | Notes |
|-----|-----------|-------|
| 1 | Original | Unprocessed reference |
| 2 | Bungee | Falls back to librosa `soxr_hq` if not installed |
| 3 | Rubberband | pyrubberband; supports `formants=` kwarg if available |
| 4 | Pedalboard | Spotify Pedalboard `time_stretch`; falls back to `PitchShift` |
| 5 | Sox | WSOLA (`tempo`) or overlap-add (`stretch`) via pysox + tempfiles |

- **Pitch:** ±12 semitones in 1-step increments.
- **Time stretch:** 0.5× – 2.0×.
- **Preserve Formants** checkbox is forwarded to algorithms that support it
  (Rubberband, Pedalboard, Sox WSOLA mode).
- **Scoring:** rate each algorithm's pitch shift or time stretch quality
  (👍 +1 / 😐 0 / 👎 −1). Scores are appended to `pitch_shift_scores.csv`
  and `stretch_scores.csv` respectively, including track, crop, semitones,
  stretch factor, and algorithm name.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| 1–5 | Switch algorithm |

---

## Unified MIR Explorer (port 7895)

Replaces the Feature Explorer HTML, Latent Shape Explorer HTML, and the Dash
latent analysis app with a single Python Dash application.

**Run:**
```bash
python plots/explorer/app.py [--port 7895] [--debug]
```

**Requires:** `latent_server.py` on port 7891 for audio playback (see below).

### Tabs

| Tab | Description |
|-----|-------------|
| **Dataset** | 8-mode scatter explorer (Scatter, Quadrant, Histogram, Radar, Heatmap, Parallel, Similarity, Classes). Sidebar shows lasso-selected tracks with streaming links. Autoplay-on-hover with 200 ms fade. |
| **Analysis** | Correlation matrix, feature posters, PCA explorer, temporal, cross-correlation, cluster map — direct port of `plots/latent_analysis/app.py`. |
| **Viewer** | 3D latent trajectory (PCA-projected), alignment bar with beat/downbeat/onset markers, simple and advanced crossfader with per-cluster alpha sliders, dim range selector, and latent manipulation. |

Persistent player strip at the bottom is always visible across tabs.

### Configuration

Read from `latent_player.ini` in the project root (same as before):

```ini
[server]
latent_dir    = /run/media/kim/Lehto/latents
stem_dir      = /run/media/kim/Lehto/latents_stems
raw_audio_dir = /home/kim/Projects/goa_crops
source_dir    = /run/media/kim/Mantu/ai-music/Goa_Separated
port          = 7891

[model]
sao_dir       = /home/kim/Projects/SAO/stable-audio-tools
model_config  = ...
ckpt_path     = ...
```

### Starting both servers

```bash
# Unified explorer UI — port 7895
python plots/explorer/app.py

# VAE decode server — port 7891 (required for audio)
source /home/kim/Projects/SAO/stable-audio-tools/rocm_env.sh
python scripts/latent_server.py
```

The old shape server (port 7892) is no longer needed.

### LatCH latent manipulation

Sliders in the Viewer tab (Brightness, Bass Energy, Danceability, Hardness,
Female Voice) nudge the latent before decode. Backend in `plots/explorer/latch.py`:
- If `models/latch/{feature}.pt` exists → uses the trained LatCH gradient at t≈0
- Otherwise → falls back to `models/latent_correlations.json` (existing behaviour)

---

## Latent Shape Explorer (retired — replaced by Unified Explorer)

**Files (kept for reference):**
- `plots/latent_shape_explorer/index.html` — old frontend
- `scripts/latent_shape_server.py` — old shape/API server (port 7892)
- `scripts/latent_server.py` — VAE decode server (port 7891, required for audio only)

3D visualizer for VAE latent trajectories. Each crop's 64-dimensional latent is
projected to 3 PCA dimensions and rendered as a trajectory through time. Use it
to compare how tracks and their stems occupy latent space, crossfade between
crops, and manipulate audio by nudging latent dimensions correlated with specific
MIR features.

### Starting the servers

```bash
# Shape server (required) — port 7892
cd /home/kim/Projects/mir
python scripts/latent_shape_server.py

# VAE decode server (required for audio playback and crossfade) — port 7891
source /home/kim/Projects/SAO/stable-audio-tools/rocm_env.sh
python scripts/latent_server.py
```

Then open `http://localhost:7892` in a browser.

Config is shared with the Feature Explorer via `latent_player.ini` in the
project root. Key paths:

```ini
[server]
latent_dir     = /run/media/kim/Lehto/latents
stem_dir       = /run/media/kim/Lehto/latents_stems
raw_audio_dir  = /home/kim/Projects/goa_crops
source_dir     = /run/media/kim/Mantu/ai-music/Goa_Separated
```

`source_dir` is needed for the alignment bar (beat/downbeat/onset timecodes). If
absent, the alignment bar stays blank and `/api/timecodes` returns 404.

On first startup with no `models/global_pca_3d.npz`, the server fits a 3-component
PCA from 2 000 randomly sampled tracks (≈ 1 min). Subsequent starts load the
cached model instantly.

### Header controls

| Control | Description |
|---------|-------------|
| **Track A / Crop A** | Primary track and crop to visualize |
| **Track B / Crop B** | Optional comparison track (also the crossfade target) |
| **Mode** | Visualization mode (see below) |
| **Render 3D** | Re-render the plot after changing mode or crop |
| **Full Mix / Drums / Bass / Other / Vocals** | Toggle which trajectories are shown; selecting a stem auto-deselects Full Mix |
| **Manipulate sliders** | Brightness, Bass Energy, Danceability, Hardness, Female Voice — nudge latent dims correlated with each feature; updates plot and crossfade audio in real time |
| **⊕ Avg** | Decode and plot the average latent of all Track A crops; plays via the VAE server |

### Visualization modes

| Mode | What it shows |
|------|---------------|
| **Raw Trajectories** | PCA-projected 3D path, colour gradient dark→bright = start→end of crop |
| **Hybrid 3D (Time = Z)** | Z-axis encodes time, X/Y are PCA dims 1 and 2; emphasises temporal drift |
| **Difference (Full Mix − Stem)** | Per-frame subtraction in latent space then projected; shows what each stem contributes to the full mix |
| **Moving Average (Smoothing)** | Smoothed trajectory; reduces frame-to-frame noise to reveal gross shape |

If Track B is selected, its trajectory is overlaid in a contrasting colour.

### Alignment bar

A canvas strip below the crossfade panel showing beat structure for both crops
side by side:

| Marker | Colour | Meaning |
|--------|--------|---------|
| **O** | Yellow | Onsets |
| **B** | Green | Beats |
| **D** | Red | Downbeats |

Top lane = Track A, bottom lane = Track B. When Beat Match is active, Track B's
lane is stretched/compressed to reflect the BPM correction applied, so you can
visually confirm grid alignment before playing.

### Crossfade panel

Requires the VAE decode server on port 7891.

**Crossfade modes** (left toggle):

| Mode | Description |
|------|-------------|
| **Full-mix** | Interpolates the full-mix latent vectors of A and B; single α slider (0 = all A, 1 = all B) |
| **Stem latents** | Per-stem α sliders; each stem is interpolated independently before summing through the decoder |
| **Stem audio** | Mixes decoded audio stems rather than latent vectors |

**Additional controls:**

| Control | Description |
|---------|-------------|
| **β_A / β_B (Color Anchor)** | Pull the blended result back toward the full-mix latent of track A or B; combats unnatural "average" timbre |
| **Slerp / Lerp** | Interpolation path: spherical (smoother, stays on manifold) or linear |
| **♫ Beat Match** | Analyses BPM and chroma of both tracks, shows key info, applies a time-stretch correction to bring tempos within 5% of each other and the smallest semitone shift to align keys; the correction is reflected on the alignment bar |
| **Smart loop** | Loops the crossfade audio to the nearest multiple-of-4-bar boundary common to both crops |
| **▶ Play Crossfade** | Fetches and plays the blended audio; re-clicking stops it |

### Latent manipulation

Five sliders in the header correspond to MIR features that correlate with
specific latent dimensions (loaded from `models/latent_correlations.json`). Moving
a slider adds a weighted offset to the latent vector before decoding:

- Positive values push the sound toward higher values of that feature.
- The offset propagates into the crossfade audio on slider release.
- Re-centering a slider to 0 removes the offset.

The feature-to-channel correlation mapping is precomputed by the
`plots/latent_analysis/` pipeline (scripts 01–04).

### Server API

| Endpoint | Description |
|----------|-------------|
| `GET /api/status` | Health check |
| `GET /api/tracks` | List all track directories |
| `GET /api/crops?track=NAME` | List crops with positions for a track |
| `GET /api/shape?track=NAME&crop=ID` | PCA-projected 3D trajectory for full mix + available stems |
| `GET /api/average-shape?track=NAME` | Average 3D trajectory across all crops of a track |
| `GET /api/timecodes?track=NAME&crop=ID` | Beat/downbeat/onset timestamps for a crop window (requires `source_dir`) |
| `GET /api/pca` | PCA components matrix `[3, 64]` |
| `GET /api/tracks.csv` | Track metadata CSV used for beat matching |
| `GET /api/correlations` | Feature→latent channel correlation mapping |

### Regenerating tracks.csv

`plots/tracks.csv` is served at `/api/tracks.csv` and used by the Beat Match feature
to look up per-track BPM and chroma data. It is built from the `.INFO` sidecar files in
the crops directory using `src/core/data_store.py`.

**Important:** only full-mix crop directories contain `.INFO` files. Stem directories
(latent `.npy` files, separated audio) have no sidecars — running bootstrap on them
produces an empty result.

```bash
# Rebuild tracks.csv from all .INFO files under the crops dir
cd /home/kim/Projects/mir
python - <<'EOF'
from pathlib import Path
from src.core.data_store import DataStore
store = DataStore.bootstrap(Path("/run/media/kim/Mantu/ai-music/Goa_Separated_crops"))
store.to_csv(Path("plots/tracks.csv"))
print(f"Wrote {len(store)} rows")
EOF
```

Alternatively run the module directly (builds `dataset.json` only; `to_csv` needs the
extra step above):

```bash
python src/core/data_store.py /run/media/kim/Mantu/ai-music/Goa_Separated_crops
```

---

## Feature Explorer + Latent Player

**File:** `/run/media/kim/Lehto/feature_explorer.html`
(committed copy: `plots/feature_explorer.html`)

Interactive scatter plot of 4 000+ tracks across 65 MIR features, with an
integrated latent audio player for real-time VAE decode via the latent server.

### Starting the latent server

```bash
source /home/kim/Projects/SAO/stable-audio-tools/rocm_env.sh
cd /home/kim/Projects/mir
python scripts/latent_server.py
```

Configuration is in `latent_player.ini` (MIR project root). Key settings:

```ini
[paths]
latent_dir = /run/media/kim/Lehto/goa-small      # full-mix .npy latents
stem_dir   = /run/media/kim/Lehto/goa-stems      # stem .npy latents

[model]
sao_dir    = /home/kim/Projects/SAO/stable-audio-tools
model_config = %(sao_dir)s/models/checkpoints/small/base_model_config.json
ckpt_path    = %(sao_dir)s/models/checkpoints/small/base_model.ckpt
```

### Feature Explorer controls

Open the HTML file directly in a browser (no server needed for the plot itself).

- **Axes / colour / size** dropdowns select which MIR features to display.
- **Hover** over a point (or `.tl-item` / `.nn-row` in sidebar lists) to load
  that track into the **A slot** of the latent player (white ring on scatter).
- **Double-click** to set the **B slot** (cyan ring) — used as the target for
  crossfading.

### Latent player panel

Click the **🎵 Latent** button in the controls bar to open the player panel.

**Full Mix mode** — decodes the full-mix latent at a given position:

- Position slider (0–1 normalised to crop length) with 400 ms debounce.

**Stem Crossfader mode** — blends stems between two tracks:

| Control | Meaning |
|---------|---------|
| Drums / Bass / Other / Vocals α | Per-stem interpolation weight (0 = A, 1 = B) |
| β_A / β_B | Reality anchors — pull the blended result toward the full-mix latent of track A or B respectively |
| Slerp / Lerp toggle | Interpolation method in latent space |

The crossfade math is a two-step operation:
1. `z_math = Interp(z_stem_A, z_stem_B, α)` — per-stem blend
2. `z_anchored = reality_anchor(z_math, z_fullmix_A, z_fullmix_B, β_A, β_B)` — optional pull toward full mixes
3. Output = `sum(Decoder(z_stem_i))` → tanh soft-clip

### Server API

| Endpoint | Description |
|----------|-------------|
| `GET /status` | Health check; returns model info |
| `GET /decode?track=NAME&position=0.5` | Decode full-mix latent at position |
| `GET /crops?track=NAME` | List available crop positions for a track |
| `GET /crossfade?track_a=A&track_b=B&pos_a=0.5&pos_b=0.5&drums=0.5&bass=0.5&other=0.5&vocals=0.5&beta_a=0&beta_b=0&interp=slerp` | Stem crossfade decode |

Response header `X-Audio-Source: raw\|vae` indicates whether the audio came
from a cached raw decode or a fresh VAE pass (useful for browser-side
verification).

---

## Latent Feature Analysis Pipeline

**Dir:** `plots/latent_analysis/`

Resumable 4-script pipeline that cross-correlates all 64 VAE latent dimensions
with ~60 MIR features across ~192K Goa trance crops. Outputs correlation
posters, PCA analysis, temporal analysis, and an interactive Plotly Dash
explorer.

### Running the pipeline

```bash
cd /home/kim/Projects/mir

# Step 1 — full Pearson+Spearman correlation matrix + 8×8 PNG posters (~2–5 min)
python plots/latent_analysis/01_aggregate_correlation.py

# Step 2 — feature PCA + latent PCA + cross-PCA correlation
python plots/latent_analysis/02_pca_analysis.py

# Step 3 — 64×64 latent temporal cross-correlation + Ward clustering (2000-crop subsample)
python plots/latent_analysis/03_latent_xcorr.py

# Step 4 — frame-level temporal features correlated with latent time series
python plots/latent_analysis/04_temporal_correlation.py

# Launch interactive Dash explorer
python plots/latent_analysis/app.py --debug    # http://localhost:7895
```

Each script is resumable: it checks for its output NPZ at startup and skips if
already present. Pass `--force` to recompute.

### Data paths (drives must be mounted)

| Data | Path |
|------|------|
| Latents `[64, 256]` float32 | `/run/media/kim/Lehto/goa-small/{Track}/{Crop}.npy` |
| Feature INFOs | `/run/media/kim/Mantu/ai-music/Goa_Separated_crops/{Track}/{Crop}.INFO` |
| Output NPZs + posters | `plots/latent_analysis/data/` (gitignored) |

### Output files

| File | Contents |
|------|----------|
| `data/01_correlations.npz` | `r_pearson[64,N]`, `r_spearman[64,N]`, BH-FDR adjusted p-values, `feature_names`, `n_per_feature` |
| `data/scatter_sample.npz` | 5000-crop subsample: `latent_means[5000,64]`, `feature_values[5000,N]` |
| `data/posters/*.png` | One 8×8 heatmap per feature (64 dims arranged spatially, colour = Pearson r) |
| `data/02_pca.npz` | Feature PCA + latent PCA scores, cross-PCA correlation matrix |
| `data/03_xcorr.npz` | 64×64 Fisher-Z-averaged xcorr matrix, Ward linkage, cluster labels |
| `data/04_temporal.npz` | Temporal r[64, 23], sample latent+feature segments for Tab 4 |
| `FINDINGS.md` | Auto-updated narrative (one section per script, section-overwrite on rerun) |
| `PROGRESS.md` | Timestamped log of completed script runs |

### Dash app tabs (port 7895)

| Tab | What it shows |
|-----|---------------|
| **Correlation Matrix** | 64×N heatmap; sort by loading/cluster/index; filter by feature group; threshold slider; click cell → scatter |
| **Feature Posters** | 8×8 PNG-style interactive heatmap per feature; top±3 dims annotated |
| **PCA Explorer** | Latent PCA scatter coloured by any feature; Feature PC ↔ Latent PC cross-correlation heatmap |
| **Temporal** | Per-crop latent dim time series + frame-level features (dual y-axis, WebGL); BPM source selector |
| **Latent Cross-Corr** | 64×64 xcorr heatmap sorted by Ward cluster; hover for r values |
| **Cluster Map** | Table: each latent cluster → top correlated features (mean r across cluster) |

### Key design decisions

- **STFT alignment:** `n_fft=4096`, `hop=2048`, `center=True` → 257 frames, sliced to 256.
  50% Hann overlap required for accurate spectral flux and onset detection.
- **tonic encoding:** sin/cos (`tonic_sin`, `tonic_cos`) — circular variable, not linear.
- **BPM madmom normalisation:** multiply/divide by 2 until ratio to Essentia is minimised.
- **HPCP:** raw values for Pearson; CLR-transformed for PCA (compositional simplex).
- **Multiple comparisons:** Benjamini-Hochberg FDR at q=0.05; effect size (|r|≥0.10/0.20/0.30)
  is the primary display metric — p-values alone are meaningless at N=192K.
- **Temporal xcorr:** Fisher-Z transform (arctanh → mean → tanh) for averaging correlation matrices.
