# Running grain_sim.py on a ROCm GPU — instructions for a remote instance

You are on a box with an AMD ROCm GPU. Goal: run the staged silver-halide grain
simulator, confirm its falsifiable predictions on-GPU, and report the numbers.
**Do not modify the physics in `grain_sim.py`** — only run it. If something
fails, report the error; don't "fix" the model.

## 1. Get the code
```bash
cd <mir-feature-extraction repo>
git fetch origin claude/local-rocm-desktop-lkh9oz
git checkout claude/local-rocm-desktop-lkh9oz
git pull origin claude/local-rocm-desktop-lkh9oz
```

## 2. Confirm torch is a ROCm build and the GPU is visible
Use an existing ROCm venv if present (e.g. the SAO ROCm 7.14 venv), otherwise
the repo's `install.sh` / PyTorch ROCm nightly. Then:
```bash
python -c "import torch; print('hip', torch.version.hip, 'avail', torch.cuda.is_available())"
```
- `torch.version.hip` must be non-`None` and `cuda.is_available()` must be `True`.
- On ROCm, torch reports the GPU as `cuda` — that is expected. `grain_sim.py`
  will print `device: cuda`. If it prints `device: cpu`, the GPU is not being
  used — stop and report that.

## 3. Run
```bash
# fast smoke first (256 um patch) — confirms kernels build and predictions land
python film_grain/grain_sim.py --quick

# full calibration run (1024 um patch, 10240^2 fine grid)
python film_grain/grain_sim.py
```
If the full run OOMs on a smaller card, step down: `--patch 512` then `--patch 768`.
16 GB should handle the full 1024 um run (fine grid ~419 MB float32, FFT working
set a few GB transient).

## 4. What "working" looks like (compare against these)
These are the CPU/NumPy-verified expectations. GPU output should match in shape;
exact values shift a little with seed and with the on-GPU RNG.

**Characteristic curve (D-logE):** monotonic, a real *toe* — D near zero at the
low-exposure foot, climbing and saturating near `D_max ≈ 0.95` with the current
placeholder `LAMBDA_UM2=13`. Roughly:
```
  logE  -2.70   D ~0.00      <- toe
  logE  -1.10   D ~0.3-0.5
  logE  -0.30   D ~0.8
  logE  +0.48   D ~0.95      <- shoulder (capped by coating weight)
```

**Nutting check (verbose block):** `Nutting prediction D` and
`simulated macroscopic D` should agree to within a few percent at this mid
density, with the simulated value **slightly below** the prediction (disc overlap
in the union). If simulated ≫ prediction, something is wrong.

**Selwyn root-area check:** `G = σ_D·√(2A)` should be roughly constant across the
12 / 24 / 48 / 96 µm apertures on the **full** run (expect ±~15%). On `--quick`
(256 µm) it will *not* converge — the 96 µm aperture may be skipped and G will
drift. That is expected; the root-area law needs patch ≫ 48 µm aperture.

## 5. Report back (paste verbatim)
- The `device:` line (confirms GPU).
- `torch.version.hip` and the GPU name (`rocminfo | grep gfx` or
  `torch.cuda.get_device_name(0)`).
- All three output blocks: **characteristic curve**, **Nutting check**,
  **Selwyn check**, plus the **RMS granularity** number and the **radial NPS**.
- Wall-clock for the full run.
- Any warnings (a `POOL_1D` warning on gfx1201 is cosmetic; a ROCm-7.2 roctracer
  message at exit is suppressed by `setup_rocm_env` — the hard `os._exit(0)` at
  shutdown is intentional, not a crash).

## 6. Notes
- The entry point calls `setup_rocm_env()` from `src/core/rocm_env.py` before
  importing torch, so it inherits the repo's Flash-Attention / TunableOp /
  allocator settings. Shell exports still win (`setdefault`).
- `PYTORCH_TUNABLEOP_ENABLED=1, TUNING=0` uses pre-tuned GEMM kernels if a
  results file exists; without one it runs untuned — fine for this workload.
- These parameters are placeholders. Once the run is confirmed, the first thing
  to calibrate is `LAMBDA_UM2` (coating weight) to lift `D_max` from ~0.95 toward
  the ~2.0+ real stocks reach.
