# Context Handover for Claude

## What was just completed
Antigravity has fully implemented the **LatCH (Latent-Control Heads)** training and inference pipeline into the Stable Audio Open (SAO) repository. The new files are located at `/home/kim/Projects/SAO/stable-audio-tools/scripts/` (`latch_model.py`, `latch_dataset.py`, `train_latch.py`, and `generate_latch_guided.py`).

The LatCH models are designed to provide temporal (frame-by-frame) guidance to the diffusion model during generation.

## The Problem
Currently, the MIR extraction pipeline (in `/home/kim/Projects/mir`) outputs `.INFO` JSON files where the audio features are condensed into static scalar averages (e.g., `bpm: 140.3`, `rms_energy_bass: -17.54`). 

For LatCH to work properly, it requires these features to be **time-series arrays**, tightly aligned to the VAE latent frame rate of the diffusion model.

## Your Task
Update the local MIR extraction scripts to output time-series arrays (specifically arrays of length `256`) instead of scalars for dynamic audio features.

### Technical Requirements
1.  **Target Temporal Resolution**: The SAO model operates at 44,100 Hz with a downsampling ratio of 2048. Therefore, the MIR features must be extracted or interpolated to match a frame rate of exactly **~21.5332 Hz** (which corresponds to exactly 256 frames for an 11.89-second chunk of audio).
2.  **Specific Features to Convert**:
    *   `rms_energy_bass`, `rms_energy_body`, `rms_energy_mid`, `rms_energy_air`
    *   `spectral_flatness`, `spectral_flux`, `spectral_skewness`, `spectral_kurtosis`
    *   `beat_activations` (you need to replace the static `bpm` integer with an array of frame-wise beat probabilities or binary activations).
    *   Pitch contours or Chroma vectors (e.g. converting `hpcp_0` through `hpcp_11` from 12 global static averages into a 12-channel chronogram array of length 256).
3.  Ensure that the `.INFO` files are updated to serialize these lists properly so that `latch_dataset.py` can load them as PyTorch tensors of shape `[1, 256]`.
