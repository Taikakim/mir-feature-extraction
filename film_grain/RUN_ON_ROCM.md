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
If the full run OOMs on a smaller card, either step the patch down
(`--patch 512` then `--patch 768`) or keep the full patch and tile the rasterizer:
```bash
python film_grain/grain_sim.py --tiles 4     # 4x4 tiles, max-radius halo
```
`--tiles N` splits the rasterizer into N×N contiguous blocks each grown by a
halo of the maximum grain radius, so peak rasterizer memory is ~1/N² of the
monolithic grid. It is **clip-free and bitwise-identical** to the monolithic
render (verified: 0 mismatched pixels across 2/3/4/5/8 tiles, including an
oversized grain straddling a tile seam and the patch edge — it wraps, it does not
clip). The crystal field stays one global torus; contiguous blocks + halo keep
each local grain whole (do NOT confuse with strided/interlaced tiling, which
would split grains and clip them). Note: `--tiles` bounds the *rasterizer* only;
the FFT scan is still global, so it is the next memory item if you push the patch
much larger. 16 GB should handle the full 1024 um run untiled (fine grid ~419 MB
float32, FFT working set a few GB transient).

## 4. What "working" looks like (compare against these)
These are measured from an actual `--quick` run of this exact code on CPU
(`--device cpu`, seed defaults). GPU output should match closely; exact values
shift a little with the on-GPU RNG stream. The full 1024 µm run gives the same
curve (macro density is patch-independent in expectation) and a cleaner Selwyn.

**Characteristic curve (D-logE):** monotonic, a real *toe* — D at/below zero at
the low-exposure foot, climbing and saturating near `D_max ≈ 0.95` with the
current placeholder `LAMBDA_UM2=13`. Reference (`--quick`, CPU):
```
  logE  -2.70   D -0.000     <- dead toe (K-hit threshold)
  logE  -1.52   D  0.043
  logE  -1.10   D  0.257
  logE  -0.30   D  0.897
  logE  +0.48   D  0.949     <- shoulder (capped by coating weight)
```

**Nutting check (verbose block):** `Nutting prediction D` and
`simulated macroscopic D` should agree to within ~10%. On the reference run the
simulated value is **slightly ABOVE** the prediction (0.260 vs 0.246, ~+6%): the
Matérn hard-core process suppresses disc overlap relative to the Poisson-Boolean
field Nutting assumes, so coverage — and thus density — runs a little high. (A
plain overlap-allowed Poisson model instead sits ~2% *below*; both are "close",
the sign just depends on whether hard-core repulsion is on.) A gross mismatch —
simulated more than ~1.5× the prediction, or far below — means something is wrong.

**Selwyn root-area check:** `G = σ_D·√(2A)` should be roughly constant across
apertures, and `σ_D` should halve each time the aperture diameter doubles.
Reference (`--quick`, CPU):
```
  aperture 12 µm   σ_D 0.0337   G 0.507
  aperture 24 µm   σ_D 0.0169   G 0.508
  aperture 48 µm   σ_D 0.0085   G 0.509
  aperture 96 µm   σ_D 0.0037   G 0.450   <- finite-size droop at 256 µm patch
```
G constant to ~3 digits across 12/24/48 µm is the pass. The 96 µm droop is a
finite-patch artifact (only ~2.7 apertures span a 256 µm patch) and should
**disappear on the full 1024 µm run** — that is the whole reason for the 1024 µm
default. If G drifts badly at 12–48 µm too, that is a real failure.

**RMS granularity / NPS:** reference `--quick` gives `σ_D×1000 ≈ 8.5` at the
48 µm aperture and a monotonically falling radial NPS (~1.2e-1 down to ~9e-3
across 0.004→0.079 cyc/µm). Sane shape; the absolute level is high because the
placeholders aren't calibrated yet.

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
