# Latent Feature Analysis Findings

## Run info
- Date: 2026-03-20
- N crops: 68422
- N features: 69 (after encoding + variance drop)
- Features dropped (low variance): tiv_11

<!-- script-01-start -->
## Script 01 — Aggregate Correlations
- Crops analysed: 202,069
- Features: 67 (dropped: tiv_11, female_probability, male_probability)

### Top correlating (dim, feature) pairs by |Pearson r|
- Dim 42 × `sharpness`: r = +0.737
- Dim 42 × `spectral_kurtosis`: r = -0.714
- Dim 42 × `hardness`: r = +0.695
- Dim 42 × `rms_energy_air`: r = +0.693
- Dim 42 × `spectral_skewness`: r = -0.690

> ⚠️ BPM correlations may be deflated due to low within-genre variance (~135–145 BPM).
> ⚠️ `atonality` likely reflects noise/percussion energy in this corpus, not harmonic atonality.
> ⚠️ HPCP correlations are relative (compositional constraint); interpret with CLR in PCA.
<!-- script-01-end -->

<!-- script-02-start -->
## Script 02 — PCA
- Crops with complete features: 195,754

### Feature PCA explained variance
- PC1: 13.2% (cumulative 13.2%) — top: hardness, sharpness, spectral_kurtosis
- PC2: 7.2% (cumulative 20.5%) — top: booming, depth, warmth
- PC3: 5.4% (cumulative 25.8%) — top: tiv_8, hpcp_7, hpcp_0
- PC4: 4.9% (cumulative 30.7%) — top: tiv_9, hpcp_3, hpcp_10
- PC5: 4.4% (cumulative 35.2%) — top: rms_energy_mid, rms_energy_body, rms_energy_bass
- PC6: 4.0% (cumulative 39.2%) — top: syncopation, on_beat_ratio, rhythmic_complexity
- PC7: 3.6% (cumulative 42.7%) — top: tiv_1, tonic_cos, tiv_2
- PC8: 3.4% (cumulative 46.1%) — top: bpm_essentia, content_enjoyment, bpm_madmom_norm
- PC9: 3.2% (cumulative 49.3%) — top: tiv_4, tiv_5, hpcp_5
- PC10: 3.0% (cumulative 52.3%) — top: tiv_0, tiv_3, hpcp_0

### Latent PCA explained variance (top 5)
- Latent PC1: 10.4%
- Latent PC2: 9.6%
- Latent PC3: 6.5%
- Latent PC4: 4.9%
- Latent PC5: 4.9%

### Strongest cross-PCA alignments
- Feature PC4 ↔ Latent PC9: r = -0.742
- Feature PC3 ↔ Latent PC7: r = -0.685
- Feature PC1 ↔ Latent PC1: r = +0.670
- Feature PC2 ↔ Latent PC3: r = +0.466
- Feature PC9 ↔ Latent PC16: r = -0.423
<!-- script-02-end -->

<!-- script-03-start -->
## Script 03 — Latent Cross-Correlation
- Crops used: 2000 (Fisher-Z averaged)
- Clusters found (Ward, 70% cut): **2**

### Cluster membership
- Cluster 1: dims 3, 7, 9, 10, 12, 13, 14, 15, 18, 19, 20, 24, 25, 28, 29, 31, 33, 34, 41, 44, 46, 49, 51, 52, 54, 55
- Cluster 2: dims 0, 1, 2, 4, 5, 6, 8, 11, 16, 17, 21, 22, 23, 26, 27, 30, 32, 35, 36, 37, 38, 39, 40, 42, 43, 45, 47, 48, 50, 53, 56, 57, 58, 59, 60, 61, 62, 63

### Strongest inter-dim temporal correlations
- Dim 53 ↔ Dim 42: r = +0.633
- Dim 16 ↔ Dim 42: r = -0.520
- Dim 16 ↔ Dim 26: r = +0.454

> Note: within-track crops treated as independent; effective N ≈ number of tracks sampled (~1600).
<!-- script-03-end -->

<!-- script-04-start -->
## Script 04 — Temporal Correlation
- Crops used: 200,000

### Strongest temporal (dim × frame-feature) correlations
- Dim 42 × `spectral_skewness_ts`: r = -0.252 (N=200,000)
- Dim 42 × `spectral_flatness_ts`: r = +0.202 (N=200,000)
- Dim 42 × `rms_energy_air_ts`: r = +0.174 (N=200,000)
- Dim 53 × `spectral_flux_ts`: r = +0.173 (N=200,000)
- Dim 42 × `spectral_flux_ts`: r = +0.166 (N=200,000)
- Dim 45 × `spectral_flatness_ts`: r = -0.152 (N=200,000)
- Dim 48 × `spectral_flux_ts`: r = +0.151 (N=200,000)
- Dim 42 × `rms_energy_bass_ts`: r = +0.143 (N=200,000)
<!-- script-04-end -->

## Open questions / next steps
*(add manual notes here — this section is never auto-overwritten)*
