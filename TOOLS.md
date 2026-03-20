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
