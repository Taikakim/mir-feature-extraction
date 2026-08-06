# film_grain — physical-scale silver halide grain simulator

Stage 1 of a film-emulation stack, taken from the `grain_sim.py` sketch in the
research/calibration briefs. Monochrome silver image only — **deliberately**
excludes reaction-diffusion development, dye clouds, DIR coupling, halation and
the print chain. Those only matter once this core reproduces the sensitometry it
is supposed to.

## What it models
1. **Crystal field** — Matérn type-II hard-core point process (crystals can't
   interpenetrate), log-normal projected-radius distribution.
2. **Exposure** — Beer–Lambert attenuation with depth through the emulsion.
3. **Latent image** — Poisson photon absorption ∝ crystal cross-section; a
   crystal is developable only if it absorbs ≥ `K_THRESH` photons (Gurney–Mott
   multi-hit). This is what produces the toe.
4. **Development** — **binary**: a crystal with a latent speck is reduced in
   full, one without is not reduced at all. No continuous radius growth.
5. **Density** — union coverage of opaque projected discs → transmittance →
   `D = -log10(T)`. Nutting's formula emerges rather than being imposed.
6. **Scanner** — Gaussian optical MTF, then pixel-aperture box integration and
   decimation to the delivery pitch.

## Falsifiable predictions (no fitting)
- **Nutting:** `D = 0.4343 · λ_dev · a_mean`
- **Selwyn:** `G = σ_D · √(2A)` roughly constant across aperture area A
- A characteristic curve with a real **toe** from the K-photon threshold

These were checked two ways before first GPU run. (1) Independent NumPy: the K=4
threshold gives a foot log-log slope ~3.1 vs ~0.8 for single-hit (toe confirmed),
and an overlap-allowed Poisson-Boolean field tracks the Nutting prediction to
~2–3%. (2) The actual code on CPU (`--quick`): clean toe, Selwyn `G` constant to
three digits across 12/24/48 µm apertures, and simulated density ~6% *above* the
Nutting prediction — the Matérn hard-core process suppresses overlap relative to
the Poisson-Boolean baseline, so the sign flips vs the NumPy check. See
`RUN_ON_ROCM.md` for the reference numbers.

## Running (ROCm)
The entry point calls `setup_rocm_env()` from `../src/core/rocm_env.py` (the
repo's single source of truth for the AMD GPU env vars) **before** importing
torch, so it inherits the same Flash-Attention / TunableOp / allocator settings
as the rest of the pipeline. If that module isn't reachable it degrades to plain
torch defaults.

```bash
# ROCm 7.14 venv lives under the SAO project root
source /home/kim/Projects/SAO/<venv>/bin/activate    # adjust to actual venv name

python film_grain/grain_sim.py --quick     # fast smoke config (patch=256 µm) — do this first
python film_grain/grain_sim.py             # full 1024 µm calibration run
python film_grain/grain_sim.py --device cpu
```

`--quick` shrinks the patch to 256 µm for a fast first GPU run; **Selwyn will not
fully converge** at that size (the root-area law needs patch ≫ 48 µm aperture —
192 µm was measured to be not enough, hence the 1024 µm default).

### Memory note
The full run is a `1024/0.10 = 10240²` fine grid (~100 M cells) plus FFT
convolutions on it. On 16 GB that's comfortable in float32 but not trivial;
`--quick` (256 µm → 2560²) is the safe first check that the kernels build and the
predictions land before committing to the full patch.

To bound rasterizer memory at a larger patch, `--tiles N` renders the coverage
field in N×N contiguous blocks, each grown by a max-grain-radius halo. It is
**clip-free** — a grain extending past a tile or patch edge wraps on the global
torus and is rendered in full, never clipped — and **bitwise-identical** to the
monolithic render (verified: 0 mismatched pixels across 2/3/4/5/8 tiles plus an
oversized boundary-straddling grain). Contiguous blocks + halo keep each local
grain whole; this is deliberately *not* strided/interlaced tiling, which would
split grains across tiles and clip them. `--tiles` bounds the rasterizer only —
the FFT scan stays global.

## Next steps (from the calibration brief)
The parameters in `grain_sim.py` are plausible-but-unverified placeholders. The
biggest knobs to replace with measured numbers: `LAMBDA_UM2` (coating weight —
currently gives D_max ≈ 0.95, real stocks need ≈ 2.0+), the crystal size
distribution, `QUANTUM_GAIN`, and `K_THRESH`. Then the colour extensions
(dye-cloud diffusion, DIR cross-channel coupling) that the briefs rank as the
highest-value missing terms.
