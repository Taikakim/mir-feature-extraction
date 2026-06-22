# SA3 Latent Explorer — Design Spec

**Date:** 2026-06-19
**Branch:** `sa3-latent-explorer` (in `mir`)
**Status:** Approved design, pre-implementation

## Goal

Revise the latent viewer + player to work on the new **Stable Audio 3 / SAME-L**
latents (256-dim, T=4096, 10.767 Hz). SA3-only — the old 64-dim Stable Audio
Small viewer (`plots/explorer/`) and player (`scripts/latent_server.py`) stay on
`main` as a reference and are **not** modified. The tool exists to (1) review
SAME-L **encoder quality** on the Goa dataset and (2) drive **latent-space
workflows**: experimental DJ-style mixing and auditioning latent control-head
recipes.

## Background facts (verified on disk / in code)

- **Latents:** `/home/kim/Projects/latents_sa3` (local copy; mirror of
  `/run/media/kim/Lehto/latents_sa3`). ~5400 crops, each 3 files:
  - `NNNNNN.npy` — `[256, 4096]` **float16**.
  - `NNNNNN.json` — scalars + metadata: `source_track`, `prompt`, `bpm_madmom`,
    `bpm_essentia`, `lufs`, `lra`, `padding_mask` (per-frame 1/0 list, length ≤
    4096), `relative_position_start`, `relative_position_end`, `crop_idx`,
    `source_path`, `start_sample`, `end_sample`, `sample_rate`, `seconds_start`,
    `seconds_total`. **No `position` field** (use `relative_position_*`).
  - `NNNNNN.TIMESERIES.npz` — 21 per-frame MIR fields at T=4096:
    `rms_energy_{bass,body,mid,air}_ts`, `spectral_{flatness,flux,skewness,kurtosis}_ts`,
    `onset_envelope_ts`, `hpcp_ts` (shape `(4096, 12)`), `beat_activation_ts`,
    `downbeat_activation_ts`, `onset_envelope_{drums,bass,other,vocals}_ts`,
    `rms_{drums,bass,other,vocals}_ts`, `relative_position_ts`.
- **Sidecars are the source of truth** for features. No external feature DB. The
  legacy `mir/data/timeseries.db` is the *old* T=256 store — not used here.
- **Decoder:** `stable_audio_3.AutoencoderModel.from_pretrained("same-l")` →
  `.sample_rate` (44100) and `.decode(latents, chunked=True, chunk_size=128,
  overlap=32)`. Must run under the **SA3 venv** (`/home/kim/Projects/SAO/stable-audio-3/.venv`,
  py3.13). Chunked decode keeps peak VRAM low (~5 GB unchunked for medium @120s;
  chunking reduces it) — important: a training run may be using the GPU, leaving
  only ~4–5 GB free.
- **Latent geometry:** frame rate = 44100 / 4096 = 10.767 Hz; T=4096 ≈ 380.5 s.
- **No SA3 stem latents** — `latents_sa3` is flat full-mix crops only. DJ mixing
  is full-mix latent interpolation; the stem-crossfade machinery does not apply.
- **Control heads:** SA3 LatCH heads already trained at
  `/home/kim/Projects/SAO/stable-audio-3/latch_weights_sa3{,_medium}/`
  (`latch_sa3_<feature>_best.pt`), predicting the TIMESERIES fields. Canonical
  model `stable_audio_3.models.latch.LatCH(in_channels=256, out_channels=1,
  dim=..., depth=6, num_heads=8)`. Reusable inference helpers:
  `stable_audio_3.inference.latch_guided` + `latch_targets`.
- **steer-sao** (`/home/kim/Projects/steer-sao`) MuseControlLite-style adapters
  are a *separate* py3.10/CUDA runtime — **deferred**, exposed only as a clean
  plug-in seam in this pass, not wired.

## Architecture — two processes (unchanged split, SA3-flavored)

```
┌─────────────────────────┐        HTTP         ┌──────────────────────────┐
│ Viewer (Dash)           │  ───────────────▶   │ Player (HTTP server)     │
│ mir venv (py3.12)       │   audio requests    │ SA3 venv (py3.13)        │
│ reads sidecars directly │  ◀───────────────   │ owns SAME-L VAE + LatCH  │
│ no model loaded         │     WAV bytes       │ heads; decodes/mixes     │
└─────────────────────────┘                     └──────────────────────────┘
```

Rationale: the SAME-L decoder must run in the SA3 venv; a sidecar-reading Dash
app needs only numpy/pandas/plotly and runs in the mir venv. The split is also
the existing pattern (`plots/explorer` builds URLs → `latent_server.py` decodes).

## Component 1 — Viewer: `plots/explorer_sa3/`

A Dash app. New directory (does not touch `plots/explorer/`).

- **`sidecar_index.py`** — scan `latents_sa3/*.json` once at startup into an
  in-memory table (crop id, source_track, artist/title, prompt, bpm, lufs,
  relative_position, crop_idx). Provides search (artist/title/prompt) and
  grouping by `source_track`. Pure functions, testable without the model.
- **`latents.py`** — `load_latent(crop) -> np.ndarray[256,4096] float32`
  (fp16→fp32), `content_frames(meta) -> int` (sum of `padding_mask`),
  `load_timeseries(crop) -> dict[str, np.ndarray]`. Pure, testable.
- **`viewer_tab.py`** — per-crop view: latent heatmap `[256,4096]` (content
  region marked via padding_mask), per-dim stats, frame axis in seconds @
  10.767 Hz; overlay any TIMESERIES field beneath the heatmap.
- **`dataset_tab.py`** — dataset-wide scatter/histograms over the cached scalar
  table (bpm, lufs, lra, year, …); search + filter by source_track.
- **`analysis_tab.py`** — **live, on-demand** (sampled subset, default 400
  crops, in-memory cached): PCA of latent frames, 256×256 dim cross-correlation,
  dim↔TIMESERIES-feature correlation. No precompute pipeline, no NPZ files.
- **`audio_panel.py`** — buttons that build player URLs (below) and embed an
  `<audio>` element; A/B toggle between `/decode` (reconstruction) and
  `/source` (ground truth).
- **`player_client.py`** — URL builders + `status()` reachability check
  (analogous to today's `plots/explorer/audio.py`).
- **`app.py`** — wires tabs, loads config from `latent_player_sa3.ini`.

## Component 2 — Player: `scripts/latent_server_sa3.py`

HTTP server, shebang `#!/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python`.
Loads `AutoencoderModel.from_pretrained("same-l")` once; LatCH heads lazily on
first use, cached by feature name. A single `threading.Lock` around GPU work.

Endpoints (all GET, return WAV unless noted):

- `GET /status` → JSON `{ok, model, sample_rate, latent_dir, device, n_heads}`.
- `GET /crops` → JSON list of `{id, source_track, relative_position_start}`.
- `GET /meta?crop=NNNNNN` → JSON of the crop's sidecar scalars.
- `GET /decode?crop=NNNNNN` → chunked `decode` of the `[256,4096]` latent →
  WAV, trimmed to `content_frames * 4096` samples.
- `GET /source?crop=NNNNNN` → slice original audio via
  `source_path[start_sample:end_sample]` → WAV. No GPU. The A/B reference.
- `GET /mix?crop_a=&crop_b=&t=0.5&interp=slerp` → `slerp`/`lerp` of the two
  full-mix latents (reuse `scripts/latent_crossfader.py` `slerp`/`lerp`,
  dimension-agnostic) → decode → WAV.
- `GET /steer?crop=NNNNNN&head=<feature>&gain=48` → load LatCH head for
  `<feature>`, compute the guidance nudge via `stable_audio_3.inference.
  latch_guided` helpers, apply to the latent, decode → WAV. Default gain
  follows SA3 medium guidance (≈48–96), not the Small default of 8.

Config: `latent_player_sa3.ini` — `latent_dir` (default
`/home/kim/Projects/latents_sa3`), `model = same-l`, `port = 7892` (distinct
from the Small player's 7891 so both can run), `latch_weights_dir`.

## Error handling

- Viewer: if the player `/status` is unreachable, the audio panel shows a
  disabled state with the launch command; data/plots still work (no model
  needed). Missing `.TIMESERIES.npz` or `.json` → that crop is skipped from the
  index with a logged warning, never a crash.
- Player: unknown crop id → 404 JSON `{error}`. Unknown/absent head → 404 with
  the list of available heads. Decode OOM → 503 JSON suggesting a smaller
  `chunk_size`; never leave the GPU lock held (try/finally).
- Latent dtype/shape guard: `load_latent` asserts `[256, T]` after squeezing a
  leading batch dim, raises a clear error otherwise.

## Testing

- **mir venv, no model:** unit tests for `sidecar_index` (scan/search/group on a
  tiny fixture dir of 2–3 synthetic sidecars), `latents` (fp16→fp32 dtype,
  shape, `content_frames` from padding_mask), interp helpers, and
  `player_client` URL builders.
- **SA3 venv, guarded smoke test:** `/status` + `/decode` on one real crop,
  skipped (pytest.skip) when `same-l` weights / GPU are unavailable so CI in the
  mir venv stays green.

## Out of scope (explicit YAGNI)

- Modifying or back-porting the old 64-dim Small viewer/player.
- steer-sao MuseControlLite adapter runtime (seam only).
- SA3 stem-based mixing (no stem latents exist).
- A regenerated dataset-wide analysis precompute pipeline / NPZ artifacts.
- Beat-match / pitch-stretch crossfade (Small-player feature, not ported).
