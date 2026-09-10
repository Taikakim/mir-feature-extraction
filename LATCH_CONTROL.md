# LatCH Project Status

## What LatCH Is

**LatCH (Latent-Control Heads)** — lightweight (~7M parameter) Bidirectional Transformer heads trained to predict MIR time-series features from noisy VAE latents. Used to apply Training-Free Guidance (TFG) during Stable Audio Open inference.

Implementation lives at `/home/kim/Projects/SAO/stable-audio-tools/scripts/`:
- `latch_model.py` — LatCH architecture (BiTransformer, RoPE, VP noise schedule)
- `latch_dataset.py` — Dataset: maps `.npy` SAO latents → MIR time-series features from TimeseriesDB
- `train_latch.py` — Training loop. Usage: `python scripts/train_latch.py --feature <name> --epochs 10`
- `generate_latch_guided.py` — Inference: Selective TFG applied to Euler sampler (first 20% of steps)

Full architecture/design notes: `/home/kim/Projects/SAO/stable-audio-tools/scripts/LATCH_README.md`

## Current State

### Timeseries — DONE
Time-series features are extracted and stored in SQLite at `/home/kim/Projects/mir/data/timeseries.db`.

API:
```python
from core.timeseries_db import TimeseriesDB
db = TimeseriesDB.open()
arrays = db.get("Artist - Title_0")  # {field: np.ndarray} or None
```

Available fields (all `T=256` frames at ~21.53 Hz, matching SAO latent frame rate):
- `rms_energy_{bass,body,mid,air}_ts`
- `spectral_{flatness,flux,skewness,kurtosis}_ts`
- `beat_activations_ts`, `downbeat_activations_ts`, `onsets_activations_ts`
- `hpcp_ts` — shape `(256, 12)` chroma over time
- `tonic_ts`, `tonic_strength_ts`

### Trained Models — IN PROGRESS
Checkpoints saved at `/home/kim/Projects/SAO/stable-audio-tools/latch_weights/`:

| Feature | Epochs | Best checkpoint |
|---|---|---|
| `spectral_flatness_ts` | 10 | `latch_spectral_flatness_ts_ep10.pt` |
| `tonic` | 10 | `latch_tonic_ep10.pt` |

### Not Yet Trained
All other available features — see `LATCH_README.md` for the full list.

## Next Steps

1. Evaluate trained models (loss curves, qualitative generation tests with `generate_latch_guided.py`)
2. Train remaining features: `rms_energy_*_ts`, `beat_activations_ts`, `hpcp_ts`, etc.
3. Multi-feature guidance (combine multiple LatCH heads at inference time)

---

## [2026-06-28] SA3 CPU LatCH-guidance eval path — cross-repo note (stable-audio-3 agent)

SA3 now has a CPU ONNX eval path that runs the LatCH guidance heads using MIR feature targets
supplied by this repo. Scripts in `stable-audio-3/scripts/` (commit 020b6c3, branch `latch-sa3-phase1`):

- `sa3_latch_onnx.py` — `generate_z0_latch_guided`: two-stage variance+mean Selective-TFG over the
  plain SA3 DiT ONNX (ORT CPU EP); APG CFG (ports `dit.py::apg_project`); torch autograd through the
  guidance head only (DiT forward-only on ORT, no autograd).
- `latch_eval_server.py` + `submit_latch_job.py` — file-drop eval server; queue `SAO/latch_eval_queue`.
- `latch_validate.py` — CPU/GPU z0-cosine harness (GPU half deferred: `--run-gpu`).

Load any SA3 production head with `stable_audio_3.models.latch.load_latch_from_checkpoint(path, device="cpu")` — auto-detects arch.
Build a standardized target with `sa3_latch_onnx.make_latch_target(raw_value, head.metadata, frames)`.
SA3 production heads: `stable-audio-3/latch_weights_sa3_medium/*_best.pt` (14 heads, adaln_zero, depth 4).
Operating gain: **≈512 for energy heads** (sweep result 2026-06-28; gain 128 is a dead zone).
Full details and gotchas: `stable-audio-3/docs/onnx-amd-inference.md` and `SAO/MASTER.md §5`.
