# SA3 Latent Explorer

A two-process tool for reviewing **Stable Audio 3 / SAME-L** latents (256-dim,
T=4096, 10.767 Hz) and driving latent-space workflows: encoder-quality review,
experimental DJ-style mixing, and auditioning LatCH control-head recipes.

SA3-only. The old 64-dim Stable Audio Small viewer (`plots/explorer/`) and
player (`scripts/latent_server.py`) are a separate reference and are not touched
by this tool.

> Design + plan: `docs/superpowers/specs/2026-06-19-sa3-latent-explorer-design.md`,
> `docs/superpowers/plans/2026-06-19-sa3-latent-explorer.md`.

## Architecture — two processes, two venvs

```
Viewer (Dash, mir venv 3.12)  ──HTTP──▶  Player (SA3 venv 3.13)
reads sidecars, builds URLs    ◀─WAV──   owns SAME-L VAE + LatCH heads
```

The SAME-L decoder must run under the SA3 venv, so the viewer (pure
numpy/pandas/plotly over the sidecars) and the player (model) are split and
talk only over HTTP.

## Data

Latents live at `/home/kim/Projects/latents_sa3` (local copy of
`Lehto/latents_sa3`). Each crop is three files — **the sidecars are the only
feature source, no DB**:

- `NNNNNN.npy` — `[256, 4096]` float16 latent.
- `NNNNNN.json` — scalars/metadata (`source_track`, `prompt`, `bpm_madmom`,
  `lufs`, `padding_mask`, `relative_position_start/end`, `source_path`,
  `start_sample`, `end_sample`, …). Note: no `position` field.
- `NNNNNN.TIMESERIES.npz` — 21 per-frame MIR fields at T=4096.

## Running it

There are two interchangeable players (same HTTP shape). Pick one:

**A) Torch player** (SA3 venv, full incl. `/steer` — needs ~5 GB VRAM):

```bash
cd ~/Projects/mir
/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python scripts/latent_server_sa3.py
# serves http://localhost:7892   (config: latent_player_sa3.ini)
```

**B) Low-VRAM ONNX player** (mir venv, ONNX decoder on MIGraphX — ~2 GB GPU, runs
alongside a training job; no `/steer`):

```bash
/home/kim/Projects/mir/mir/bin/python scripts/latent_server_onnx.py \
    --onnx /home/kim/Projects/SAO/stable-audio-3/same_decoder_L128.onnx \
    --chunk-latents 128 --overlap 16 --provider migraphx --port 7893
# compiles the ONNX once at boot (~min, MIGraphX AOT), then RTF ~39x.
# /status /crops /meta /decode /mix /source  (/steer -> 501, use player A)
```

Export the `.onnx` first with `stable-audio-3/scripts/export_same_onnx.py`; see
`stable-audio-3/docs/onnx-amd-inference.md`. Needs `onnxruntime_migraphx` (in the
mir venv); the SA3 venv's onnxruntime is CPU-only.

**Viewer** (mir venv) — defaults to player A (7892); point it at the ONNX player
with `SA3_PLAYER_PORT`:

```bash
/home/kim/Projects/mir/mir/bin/python -m plots.explorer_sa3.app                 # -> :7892
SA3_PLAYER_PORT=7893 /home/kim/Projects/mir/mir/bin/python -m plots.explorer_sa3.app  # -> ONNX
# viewer serves http://localhost:8051
```

The viewer works without any player (data + plots need no model); the audio
panel shows the launch command when the player is offline.

## Player endpoints (port 7892)

| Endpoint | Does |
|---|---|
| `GET /status` | model + available LatCH head names |
| `GET /crops` | list crop ids |
| `GET /meta?crop=` | the crop's sidecar scalars |
| `GET /decode?crop=` | chunked SAME-L decode → WAV (the reconstruction) |
| `GET /source?crop=` | slice the original audio (A/B reference) |
| `GET /mix?crop_a=&crop_b=&t=&interp=slerp` | full-mix latent interpolation → decode |
| `GET /steer?crop=&head=&gain=` | LatCH-head fp32 gradient nudge → decode |

## Gotchas

- **`/source` needs the Mantu drive mounted** — source audio lives on the
  removable `Mantu` drive (`source_path` points there). When unmounted, `/source`
  returns a clear 500; the other endpoints are unaffected.
- **LatCH heads**: `latch_weights_dir` defaults to
  `stable-audio-3/latch_weights_sa3_medium` (the production same-l heads:
  `adaln_zero`, standardized, depth 4). Heads are loaded with
  `stable_audio_3.models.latch.load_latch_from_checkpoint`, which auto-detects
  the architecture — do **not** hardcode dim/depth/num_heads. The sibling
  `latch_weights_sa3` dir holds epoch-numbered snapshots (`_ep<N>.pt`, no
  `_best.pt`) and won't be found by the default `*_best.pt` glob.
- **Steering gain**: SA3-medium LatCH wants gain ≈ 48–96 (not the Small default
  of 8); judge by spread, not correlation.
- **Two latent grids never mix**: SA3 = 10.767 Hz / T=4096 / 256-dim; the old
  Small = 21.53 Hz / T=256 / 64-dim.
