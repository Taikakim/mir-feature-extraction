# Latent ↔ Feature Analysis Pipeline — Design Spec
**Date:** 2026-03-20
**Status:** Approved
**Author:** brainstorming session

---

## 1. Goal

Build a resumable analysis pipeline that systematically cross-correlates all 64 latent dimensions of the Stable Audio Small VAE with ~72 MIR features extracted from 192K Goa trance crops. Deliver findings as static artefacts (NPZ, PNG posters, Markdown reports) consumed by a Plotly Dash interactive explorer.

---

## 2. Dataset

| Property | Value |
|---|---|
| Latent dir | `/run/media/kim/Lehto/goa-small` |
| Feature dir | `/run/media/kim/Mantu/ai-music/Goa_Separated_crops` |
| Latent shape | `[64, 256]` float32 per crop |
| Total crops | ~192,000 |
| Paired crops | ~192,000 (100% match on shared 1,611 tracks) |
| Temporal alignment | 1 latent frame = 2048 audio samples @ 44100 Hz = 46.4 ms |
| Crop duration | 256 × 2048 / 44100 ≈ 11.9 s |
| Stem latents | **Not available** — full-mix only |

---

## 3. Feature Set (~60 features after encoding)

Exact count determined at runtime by `features.py` after dropping near-zero-variance features. The ~60 estimate excludes raw `tonic` (replaced) and any dropped features.

### 3.1 Encoding decisions

- **`tonic`** (circular 0–11): Replace with `tonic_sin = sin(2π·tonic/12)` and `tonic_cos = cos(2π·tonic/12)`. Drop raw `tonic`.
- **`tonic_scale`** (string "minor"/"major"): Encode as `tonic_minor = 1 if minor else 0`.
- **`bpm_madmom` vs `bpm_essentia`**: Normalize madmom to agree with essentia within a 2× factor (multiply/divide by 2 until |ratio−1| is minimised). Both stored as `bpm_madmom_norm`, `bpm_essentia`. GUI checkbox selects which to use for analysis (default: `bpm_essentia`).
- **`hpcp_0..11`** (compositional, sum ≈ 1): Use centred-log-ratio (CLR) transform before PCA. For Pearson correlation, use raw values but note the compositional constraint in FINDINGS.md.
- **`saturation_count`, `saturation_ratio`**: Include but auto-drop if `std < 1e-4` across corpus; report dropped features.

### 3.2 Feature groups (for UI filtering)

| Group | Features |
|---|---|
| **Rhythm** | bpm (primary field from INFO), bpm_essentia, bpm_madmom_norm, syncopation, on_beat_ratio, rhythmic_complexity, rhythmic_evenness |
| **Timbral** | brightness, roughness, hardness, depth, booming, reverberation, sharpness, warmth |
| **Spectral** | spectral_flatness, spectral_flux, spectral_skewness, spectral_kurtosis, saturation_ratio, saturation_count |
| **Loudness** | lufs, lra, lufs_drums, lufs_bass, lufs_other, lufs_vocals, rms_energy_bass, rms_energy_body, rms_energy_mid, rms_energy_air |
| **Harmonic** | hpcp_0..11 (12 chroma bins, Essentia HPCP), tiv_0..11 (12 Tonal Interval Vector components, Essentia), tonic_sin, tonic_cos, tonic_minor, tonic_strength, harmonic_movement_bass, harmonic_movement_other, atonality |
| **Voice/Gender** | voice_probability, instrumental_probability, female_probability, male_probability |
| **Aesthetics** | content_enjoyment, content_usefulness, production_complexity, production_quality |

### 3.3 Missingness strategy

Per-feature analysis: each feature builds its own `(latent_means, values)` array from all crops that have that feature. N is reported per feature in outputs. No crop is required to have all features.

---

## 4. Statistics

- **Correlation methods:** Both Pearson r (linear) and Spearman ρ (monotonic) computed for every (latent_dim, feature) pair. Both stored in NPZ; posters default to Pearson with a toggle in the Dash app.
- **Multiple comparisons:** Benjamini-Hochberg FDR correction at q=0.05 across all 64×N feature comparisons, applied separately to Pearson and Spearman. Adjusted p-values stored alongside r values.
- **Display thresholds:** Effect size primary — `|r| < 0.10` transparent/grey, `0.10–0.20` weak, `0.20–0.30` moderate, `≥ 0.30` strong. Poster colour scale saturates at |r| = 0.35.
- **Near-zero variance drop:** Features with `std < 1e-4` over corpus are excluded and logged.
- **BPM range restriction caveat:** Goa trance clusters tightly at ~135–145 BPM. Pearson r for BPM will be deflated relative to a diverse corpus — absence of correlation here means low within-genre tempo variance, not that latents are insensitive to tempo. Flagged in FINDINGS.md.

---

## 5. Pipeline Architecture

### Directory layout

```
plots/latent_analysis/
├── config.py                    # paths, feature list, thresholds (single source of truth)
├── features.py                  # feature loading + encoding (sin/cos, CLR, BPM norm)
├── 01_aggregate_correlation.py  # full 64×N corr matrix + 8×8 posters
├── 02_pca_analysis.py           # feature PCA + latent PCA + cross-mapping
├── 03_latent_xcorr.py           # 64×64 temporal cross-correlation + clustering
├── 04_temporal_correlation.py   # frame-level features + temporal latent correlation
├── app.py                       # Plotly Dash app (port 7895)
├── FINDINGS.md                  # auto-updated narrative, one section per script
├── PROGRESS.md                  # checkpoint state for resumability
└── data/
    ├── 01_correlations.npz      # r_matrix[64,N], pvals[64,N], feature_names, n_per_feature
    ├── 02_pca.npz               # feature_pca, latent_pca, explained_variance, loadings, latent_scores[N,20], feature_scores[N,20], crop_ids
    ├── 03_xcorr.npz             # xcorr_matrix[64,64], cluster_labels[64]
    ├── 04_temporal.npz          # temporal_r[64,N_temporal_feats], sample_crops[K,64,256]
    └── posters/                 # PNG per feature (8×8 heatmap, 600×600px)
```

### Resumability

Each script:
1. Checks for its output NPZ at startup.
2. If present and `--force` not passed: prints "already done, skipping" and exits 0.
3. `PROGRESS.md` updated with timestamp + key stats at end of each script. Checkpoint granularity is per-script (not mid-script); script 01's batched streaming is fast enough (~2–5 min) that mid-script checkpointing is not needed.

### Script summaries

#### `01_aggregate_correlation.py`
- Loads all ~192K crops in batches (batch_size=4096) — never loads full dataset into RAM at once
- Computes Pearson r and Spearman ρ plus BH-corrected p-values for each of 64 dims × N features
- Saves `data/01_correlations.npz` (contains both `r_pearson[64,N]` and `r_spearman[64,N]`, plus BH-corrected p-values for each)
- Generates one 8×8 PNG poster per feature into `data/posters/` (the 64 latent dims are arranged into an 8×8 grid for spatial readability; each cell = one dim, colour = Pearson r with that feature — same layout as existing posters in `feature_posters/`)
- Also saves a subsampled scatter matrix (`scatter_sample.npz`, 5000 random crops, `[N,64]` latent means + `[N, N_features]` feature values) for use by Tab 1 scatter plots in the Dash app
- Appends findings to `FINDINGS.md`: top-3 correlated (dim, feature) pairs with |r|

#### `02_pca_analysis.py`
- Loads all feature vectors (one per crop, ~192K × ~60 features ≈ ~46 MB, fine in RAM)
- Applies StandardScaler, fits PCA (n_components=20) on features — `feature_pca`
- Loads latent means (192K × 64 = ~50 MB), fits PCA (n_components=20) — `latent_pca`
- Computes correlation between feature PCA scores and latent PCA scores (cross-PCA matrix)
- Saves `data/02_pca.npz` including: PCA models, explained variance, feature loadings, **per-crop projected scores** (`latent_scores [N_crops, 20]`, `feature_scores [N_crops, 20]`) for Tab 3 scatter. Also saves a `crop_ids` index for joining to feature values in the scatter.
- Appends to `FINDINGS.md`: explained variance per component, feature loadings

#### `03_latent_xcorr.py`
- Subsamples N=2000 crops (seeded random, reproducible)
- For each crop: demeans each latent dim (subtract per-dim mean across 256 frames), then computes 64×64 Pearson correlation matrix across 256 time steps
- Applies Fisher Z-transform (`z = arctanh(r + ε)`, ε=1e-7 to guard against r=±1 on diagonal) before averaging across crops
- Mean of z matrices → inverse-transform (`r_avg = tanh(mean_z)`) → final 64×64 xcorr matrix
- Applies hierarchical clustering (Ward linkage) to find dim groups
- Saves `data/03_xcorr.npz` with matrix + cluster labels
- Appends to `FINDINGS.md`: N clusters found, which dim groups are tightly correlated

#### `04_temporal_correlation.py`
- Same 2000 subsampled crops as script 03 (same random seed)
- For each crop: computes frame-level features at hop_size=2048 (exact 1:1 with latent frames):
  - RMS energy (broadband + 4 bands via butterworth)
  - Spectral flatness, flux, centroid, skewness, kurtosis (librosa STFT, hop=2048)
  - HPCP chroma (librosa chroma_cqt, hop=2048, 12 bins)
  - Onset strength envelope (hop=2048)
- Result: [N_crops, N_temporal_features, 256] feature tensor
- Correlates each (latent_dim, temporal_feature) pair: r = corrcoef(latent[dim,t], feat[f,t]) across all (crop, t) pairs concatenated. Note: multiple crops from the same track are treated as independent — within-track correlation inflates effective N significantly (2000 crops from ~1600 tracks ≈ ~1.25 crops/track, so effective N is ~1600 not 512K). This is flagged in FINDINGS.md. The 2000-crop subsample is chosen to minimise same-track repetition; with 1611 tracks this gives near-independence.
- Saves `data/04_temporal.npz`
- Appends to `FINDINGS.md`: strongest temporal correlations found

---

## 6. Plotly Dash App (`app.py`, port 7895)

Six tabs, all reading from pre-computed `data/*.npz` files (fast startup, no recomputation).

### Tab 1 — Correlation Matrix
- `go.Heatmap`: 64 latent dims (Y) × all features (X), colour = Pearson r
- Colour scale: −0.35 blue → 0 white → +0.35 red (clipped)
- Sort dims: by feature loading / cluster order / index (radio buttons)
- Filter by feature group (dropdown)
- |r| threshold slider (0–0.35)
- Click a cell → below-chart scatter plot of that dim's mean vs. that feature across all crops

### Tab 2 — Feature Posters
- Dropdown: select feature → shows 8×8 PNG from `data/posters/`
- Alongside: interactive Plotly heatmap version (same data, hoverable)
- Annotated with: top-3 positive/negative dims, N crops used

### Tab 3 — PCA Explorer
- 2D/3D scatter of latent means projected to latent PCA space
- Colour by any feature (dropdown)
- Feature PCA loadings tab-sub: bar chart of feature weights per component
- Latent↔Feature cross-PCA heatmap: shows which feature PC aligns with which latent PC

### Tab 4 — Temporal
- Crop selector: dropdown (2000 subsampled crops)
- Multi-select latent dims to plot (default: top-5 from Tab 1 selection)
- Per-dim line plots [256 frames] using **`go.Scattergl`** (WebGL) for all temporal traces — required for smooth rendering at 256 points × multiple dims
- Overlay temporal features as coloured traces on a shared x-axis
- Scalar features (BPM, syncopation, etc.) shown as horizontal dashed reference lines
- BPM selector: radio buttons Essentia / Madmom / Average

### Tab 5 — Latent Cross-Corr
- `go.Heatmap` 64×64 temporal correlation matrix
- Dendrogram on both axes (scipy linkage → Plotly shapes)
- Hover: dim_i × dim_j → r value
- Cluster labels shown as colour bands on axes

### Tab 6 — Cluster Map
- Table: each latent cluster → top correlated feature PCA component + top raw features
- Bar chart: cluster mean correlation per feature group
- Auto-generated plain-English summary per cluster (e.g., "Dims 12,31,55 — strong positive correlation with brightness, spectral_flux, sharpness; moderate positive with female_probability")

---

## 7. FINDINGS.md Structure

Each script **overwrites its own named section** (between `<!-- script-01-start -->` / `<!-- script-01-end -->` markers) rather than appending. Re-running a script updates its section without duplicating content. The "Run info" and "Open questions" sections are written once on first run and never overwritten. Structure:

```markdown
# Latent Feature Analysis Findings

## Run info
- Date: YYYY-MM-DD
- N crops: 192,000
- N features: {N_features} (auto-filled at runtime, after encoding + variance drop)
- Features dropped (low variance): [list]

<!-- script-01-start -->
## Script 01 — Aggregate Correlations
[auto-written: strongest correlations, surprising findings]
<!-- script-01-end -->

<!-- script-02-start -->
## Script 02 — PCA
[auto-written: component interpretations]
<!-- script-02-end -->

<!-- script-03-start -->
## Script 03 — Latent Cross-Correlation
[auto-written: dim clusters found]
<!-- script-03-end -->

<!-- script-04-start -->
## Script 04 — Temporal
[auto-written: strongest temporal correlations]
<!-- script-04-end -->

## Open questions / next steps
[manual notes]
```

---

## 8. Known Limitations

- **No stem latents**: Only full-mix latent vectors available. Per-stem LUFS/features are correlated with full-mix latent dims; this is valid but conflates stem contributions.
- **Homogeneous corpus**: All Goa trance. Correlations reflect structure within this genre, not general audio.
- **HPCP compositional**: CLR applied for PCA; raw values for Pearson. Interpret HPCP correlations as relative.
- **Scalar features in temporal plots**: BPM, syncopation etc. shown as reference lines, not time series.
- **Cross-correlation computed on 2K subsample**: Not all 192K crops. Subsample is reproducible (fixed seed).
- **`atonality` is a noise-content proxy in Goa trance**: Essentia's atonality/dissonance metric responds to inharmonic timbres (cymbals, hi-hats, noise sweeps). In Goa trance with dense percussion and noise-based synthesis, it will correlate with drum/hi-hat energy rather than music-theoretic atonality. Interpret `atonality` correlations accordingly. **Future improvement:** recompute `atonality` from the bass+other mix (excluding drums+vocals) in the main MIR pipeline to better capture harmonic atonality. Not blocking for current analysis — the feature value is what it is, the caveat just affects interpretation.
- **BPM range restriction**: Low within-genre BPM variance will deflate tempo-related correlations. See Section 4 note.

---

## 9. Implementation Notes

- All scripts importable as modules (for app.py to call validate functions)
- `config.py` defines `LATENT_DIR`, `INFO_DIR`, `DATA_DIR`, `FEATURE_GROUPS`, `TEMPORAL_FEATURES`, `N_TEMPORAL_CROPS`, `RANDOM_SEED`
- `features.py` handles all encoding (sin/cos, CLR, BPM harmonisation) — one place to change
- Each NPZ includes a `meta` dict key with creation timestamp + git hash for traceability
- STFT parameters: **`n_fft=4096`**, `hop_length=2048`, `sr=44100`, **`center=True`** — gives 257 frames; **slice `[:256]`** to get exact 1:1 alignment with latent frames. This preserves 50% overlap with a Hann window, essential for spectral flux and onset strength accuracy. With `center=False` and `n_fft=4096` the frame count would be 255 (window truncation). With `n_fft=2048=hop_length` the frame count is 256 but 0% overlap means Hann window attenuates frame-boundary samples — transients on beat edges vanish from spectral flux and onset features. Always assert `frame_count == 256` after slicing. Apply the `[:LATENT_FRAMES]` slice consistently to **all** STFT-derived outputs (flatness, flux, centroid, chroma, onset, RMS from STFT).
