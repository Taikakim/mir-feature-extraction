"""
grain_sim.py -- physical-scale monochrome silver halide grain simulator.

Stage 1 of a film emulation stack. Deliberately excludes reaction-diffusion
development, dye clouds, DIR coupling, halation, and the print chain. Those
only matter if this core reproduces the sensitometry it is supposed to.

Physics implemented
-------------------
  1. Crystal field:  Matern type-II hard-core point process (crystals cannot
     interpenetrate), log-normal projected-radius distribution.
  2. Exposure:       Beer-Lambert attenuation with depth through the emulsion.
  3. Latent image:   Poisson photon absorption proportional to crystal
     cross-section; a crystal becomes developable only if it absorbs >= K_THRESH
     photons (Gurney-Mott multi-hit requirement). This is what produces the toe.
  4. Development:    BINARY. A crystal with a latent speck is reduced in full;
     one without is not reduced at all. No continuous radius growth.
  5. Density:        union coverage of opaque projected discs -> transmittance
     -> D = -log10(T). Reproduces Nutting's formula as an emergent result.
  6. Scanner:        Gaussian optical MTF, then pixel-aperture box integration
     and decimation to the delivery pitch.

Falsifiable predictions this should reproduce, with no fitting:
  * Nutting:  D = 0.4343 * lambda_dev * a_mean   (mean density vs developed count)
  * Selwyn:   G = sigma_D * sqrt(2A) approximately constant across aperture area A
  * A characteristic curve with a real toe arising from the K-photon threshold

Everything is on a torus (periodic boundaries), which is both correct for a
statistically stationary uniform-exposure patch and removes edge artifacts from
the spectral estimates. Halo/overlap handling is NOT needed at this stage --
it becomes necessary only once diffusive chemistry is added, and then the halo
width is set by sqrt(2*D_diff*t), not chosen for convenience.

Tested on CPU; intended for ROCm.

    uv venv && uv pip install torch numpy    # ROCm 7.x wheel index
    python grain_sim.py                      # full 1024 um calibration run
    python grain_sim.py --quick              # fast smoke config for a first GPU run
    python grain_sim.py --device cpu         # force CPU

ROCm note: this entry point calls setup_rocm_env() from the repo's core module
(single source of truth for the AMD GPU env vars) BEFORE importing torch, so it
inherits the same Flash-Attention / TunableOp / allocator settings as the rest
of the pipeline. If that module is not reachable it degrades to plain defaults.
"""

import argparse
import math
import sys
from pathlib import Path

# --- ROCm environment (must run BEFORE `import torch`) ---------------------
# grain_sim lives outside src/, so reach the repo's core module explicitly.
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))
try:
    from core.rocm_env import setup_rocm_env
    setup_rocm_env()
except Exception as _e:  # standalone / non-ROCm host: fall back to plain torch
    print(f"[grain_sim] ROCm env not wired ({_e}); using plain torch defaults",
          file=sys.stderr)

import torch  # noqa: E402  (import after ROCm env setup, by design)

# ---------------------------------------------------------------------------
# Parameters. Values are plausible-but-unverified placeholders; the companion
# calibration brief is about replacing them with measured numbers.
# ---------------------------------------------------------------------------
PATCH_UM      = 1024.0   # lateral extent of simulated emulsion patch [um]
                         # Selwyn needs PATCH >> largest aperture (48 um) for the
                         # root-area law to converge; 192 um was NOT enough.
FINE_UM       = 0.10     # simulation voxel pitch, lateral [um]
THICK_UM      = 12.0     # emulsion layer thickness [um]
R_MEDIAN_UM   = 0.40     # median crystal projected radius [um]
R_SIGMA_LOG   = 0.35     # log-normal shape parameter of radius distribution
LAMBDA_UM2    = 13.0     # crystal areal number density BEFORE hard-core thinning
                         # [crystals/um^2]. 3D thinning keeps ~30% of these.
                         # CALIBRATE: at 13.0 / 12 um thick, D_max ~ 0.95; real
                         # stocks need ~2.0+. Coating weight is the knob.
HARDCORE_UM   = 0.05     # extra Matern-II separation beyond touching [um]
TURBIDITY     = 0.045    # emulsion attenuation coefficient [1/um]
QUANTUM_GAIN  = 60.0     # absorbed photons per um^2 per unit relative exposure
K_THRESH      = 4        # silver atoms / photons needed for a stable latent speck
SCAN_PITCH_UM = 5.9      # delivery pixel pitch (4K scan of 35mm) [um]
SCAN_MTF_UM   = 3.0      # scanner optical PSF Gaussian sigma [um]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # ROCm reports as cuda
DTYPE  = torch.float32


# ---------------------------------------------------------------------------
# 1. Crystal field
# ---------------------------------------------------------------------------
def sample_crystals(patch_um, lam, r_median, r_sigma, hardcore, gen):
    """Matern type-II hard-core thinning of a Poisson field on a torus.

    Returns (xy [N,2] um, radius [N] um, depth [N] um).
    """
    n0 = int(lam * patch_um * patch_um)
    xy = torch.rand((n0, 2), generator=gen, device=DEVICE, dtype=DTYPE) * patch_um
    zz = torch.rand(n0, generator=gen, device=DEVICE, dtype=DTYPE) * THICK_UM
    r = r_median * torch.exp(
        r_sigma * torch.randn(n0, generator=gen, device=DEVICE, dtype=DTYPE))
    mark = torch.rand(n0, generator=gen, device=DEVICE, dtype=DTYPE)
    # Bucket into cells of side >= max interaction distance so each point only
    # needs to consult its 3x3 cell neighbourhood.
    reach = float(2.0 * r.max().item() + hardcore)
    ncell = max(1, int(patch_um // reach))
    cell = patch_um / ncell
    ci = (xy / cell).long().clamp_(0, ncell - 1)
    flat = ci[:, 0] * ncell + ci[:, 1]
    order = torch.argsort(flat)
    flat_s, xy_s, r_s, mark_s = flat[order], xy[order], r[order], mark[order]
    z_s = zz[order]
    starts = torch.searchsorted(flat_s, torch.arange(
        ncell * ncell + 1, device=DEVICE))
    keep = torch.ones(n0, dtype=torch.bool, device=DEVICE)
    ci_s = ci[order]
    CH = max(1, int(2e7 // max(1, int(torch.diff(starts).max().item()))))
    for s0 in range(0, n0, CH):                       # chunk: bounds peak memory
        sl = slice(s0, min(s0 + CH, n0))
        xy_q, r_q, z_q, m_q = xy_s[sl], r_s[sl], z_s[sl], mark_s[sl]
        dead = torch.zeros(xy_q.shape[0], dtype=torch.bool, device=DEVICE)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb = (((ci_s[sl][:, 0] + dx) % ncell) * ncell
                      + ((ci_s[sl][:, 1] + dy) % ncell))
                lo, hi = starts[nb], starts[nb + 1]
                width = int((hi - lo).max().item())
                if width == 0:
                    continue
                idx = lo[:, None] + torch.arange(width, device=DEVICE)[None, :]
                valid = idx < hi[:, None]
                idx = idx.clamp(max=n0 - 1)
                d = xy_s[idx] - xy_q[:, None, :]
                d -= patch_um * torch.round(d / patch_um)      # torus wrap
                dz = z_s[idx] - z_q[:, None]                   # depth separation
                dist = torch.sqrt((d * d).sum(-1) + dz * dz)
                touch = dist < (r_q[:, None] + r_s[idx] + hardcore)
                # Matern II: a point dies if a lower-marked neighbour overlaps it.
                dead |= (touch & valid & (mark_s[idx] < m_q[:, None])
                         & (dist > 0)).any(dim=1)
        keep[sl] = ~dead
    return xy_s[keep], r_s[keep], z_s[keep]


# ---------------------------------------------------------------------------
# 2-4. Exposure, latent image, binary development
# ---------------------------------------------------------------------------
def develop(r, z, exposure, gen):
    """Binary development mask from Poisson photon counts and a K-hit threshold."""
    e_local = exposure * torch.exp(-TURBIDITY * z)       # Beer-Lambert with depth
    n_expected = QUANTUM_GAIN * e_local * math.pi * r * r
    absorbed = torch.poisson(n_expected, generator=gen)
    return absorbed >= K_THRESH                          # full reduction, or none


# ---------------------------------------------------------------------------
# 5. Union coverage -> transmittance
# ---------------------------------------------------------------------------
def rasterize_union(xy, r, patch_um, fine_um):
    """Binary union of opaque projected discs on a periodic fine grid."""
    n = int(round(patch_um / fine_um))
    grid = torch.zeros(n * n, device=DEVICE, dtype=DTYPE)
    if xy.shape[0] == 0:
        return grid.view(n, n)
    rad_px = int(math.ceil(float(r.max().item()) / fine_um)) + 1
    off = torch.arange(-rad_px, rad_px + 1, device=DEVICE)
    oy, ox = torch.meshgrid(off, off, indexing="ij")
    oy, ox = oy.reshape(-1), ox.reshape(-1)
    # Chunk to bound peak memory: N_crystals x stamp_area index tensors.
    chunk = max(1, int(4e7 // oy.numel()))
    cx = (xy[:, 0] / fine_um)
    cy = (xy[:, 1] / fine_um)
    for s in range(0, xy.shape[0], chunk):
        cxc, cyc, rc = cx[s:s + chunk], cy[s:s + chunk], r[s:s + chunk]
        ix = (cxc.floor()[:, None] + ox[None, :])
        iy = (cyc.floor()[:, None] + oy[None, :])
        dx = (ix + 0.5 - cxc[:, None]) * fine_um
        dy = (iy + 0.5 - cyc[:, None]) * fine_um
        inside = (dx * dx + dy * dy) <= (rc[:, None] ** 2)
        flat = ((ix.long() % n) * n + (iy.long() % n))[inside]
        grid[flat] = 1.0            # union: all writes are 1.0, races are benign
    return grid.view(n, n)


# ---------------------------------------------------------------------------
# 6. Scanner: optical MTF then pixel aperture integration
# ---------------------------------------------------------------------------
def _fft_conv_periodic(img, kernel_centered):
    """Circular convolution via FFT. kernel_centered is same-shape, sum-normalised."""
    K = torch.fft.fft2(torch.fft.ifftshift(kernel_centered))
    return torch.fft.ifft2(torch.fft.fft2(img) * K).real


def _gauss_kernel(n, sigma_px):
    t = torch.arange(n, device=DEVICE, dtype=DTYPE) - n // 2
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    k = torch.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma_px ** 2)
    return k / k.sum()


def _disc_kernel(n, rad_px):
    t = torch.arange(n, device=DEVICE, dtype=DTYPE) - n // 2
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    k = ((xx ** 2 + yy ** 2) <= rad_px ** 2).to(DTYPE)
    return k / k.sum()


def scan(transmittance_fine, fine_um, pitch_um, mtf_sigma_um):
    """Apply optical PSF to the transmittance field, then box-integrate pixels."""
    n = transmittance_fine.shape[0]
    blurred = _fft_conv_periodic(
        transmittance_fine, _gauss_kernel(n, mtf_sigma_um / fine_um))
    f = int(round(pitch_um / fine_um))
    n = (blurred.shape[0] // f) * f
    boxed = blurred[:n, :n].reshape(n // f, f, n // f, f).mean(dim=(1, 3))
    return boxed                                        # pixel transmittance


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def selwyn_scan(coverage_fine, fine_um, aperture_ums=(12, 24, 48, 96)):
    """sigma_D through circular apertures, and the Selwyn constant G."""
    out = []
    T = 1.0 - coverage_fine
    n = T.shape[0]
    for a_um in aperture_ums:
        rad_px = a_um / 2.0 / fine_um
        if 2 * rad_px >= n:
            continue
        acc = _fft_conv_periodic(T, _disc_kernel(n, rad_px))
        D = -torch.log10(acc.clamp_min(1e-4))
        sd = float(D.std().item())
        area = math.pi * (a_um / 2.0) ** 2
        out.append((a_um, sd, sd * math.sqrt(2 * area)))
    return out


def radial_nps(density_px, pitch_um, nbins=48):
    """Wiener noise power spectrum, radially averaged. Units: density^2 * um^2."""
    d = density_px - density_px.mean()
    n = d.shape[0]
    F = torch.fft.fftshift(torch.fft.fft2(d))
    nps = (pitch_um ** 2 / (n * n)) * (F.real ** 2 + F.imag ** 2)
    fq = torch.fft.fftshift(torch.fft.fftfreq(n, d=pitch_um)).to(DEVICE)
    fy, fx = torch.meshgrid(fq, fq, indexing="ij")
    fr = torch.sqrt(fx ** 2 + fy ** 2)
    fmax = float(fq.max().item())
    edges = torch.linspace(0, fmax, nbins + 1, device=DEVICE)
    prof = []
    for i in range(nbins):
        m = (fr >= edges[i]) & (fr < edges[i + 1])
        if m.any():
            prof.append((float(((edges[i] + edges[i + 1]) / 2).item()),
                         float(nps[m].mean().item())))
    return prof


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(exposure, seed=0, verbose=False):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    xy, r, z = sample_crystals(PATCH_UM, LAMBDA_UM2, R_MEDIAN_UM,
                               R_SIGMA_LOG, HARDCORE_UM, gen)
    dev = develop(r, z, exposure, gen)
    cov = rasterize_union(xy[dev], r[dev], PATCH_UM, FINE_UM)
    # Macroscopic density: -log10 of MEAN transmittance. Never per fine voxel --
    # coverage there is binary, and -log10 of a binary field is not a density.
    macro_D = float(-torch.log10((1.0 - cov).mean().clamp_min(1e-5)).item())
    Tpix = scan(1.0 - cov, FINE_UM, SCAN_PITCH_UM, SCAN_MTF_UM)
    Dpix = -torch.log10(Tpix.clamp_min(1e-4))
    if verbose:
        a_mean = float((math.pi * r[dev] ** 2).mean().item()) if dev.any() else 0.0
        n_dev = int(dev.sum().item())
        lam_dev = n_dev / (PATCH_UM ** 2)
        print(f"  crystals placed/developed : {r.shape[0]} / {n_dev}")
        print(f"  developed areal density   : {lam_dev:.4f} /um^2, "
              f"mean projected area {a_mean:.4f} um^2")
        print(f"  Nutting prediction D      : {0.4343 * lam_dev * a_mean:.3f}")
        print(f"  simulated macroscopic D   : {macro_D:.3f}")
    return dict(D_px=Dpix, cov=cov, mean_D=macro_D)


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Physical-scale silver halide grain simulator (ROCm).")
    ap.add_argument("--patch", type=float, default=None,
                    help=f"patch extent [um] (default {PATCH_UM}; "
                         "Selwyn convergence needs >> 48 um aperture)")
    ap.add_argument("--fine", type=float, default=None,
                    help=f"simulation voxel pitch [um] (default {FINE_UM})")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed for the curve")
    ap.add_argument("--device", default=None,
                    help="override device, e.g. 'cpu' or 'cuda'")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke config (patch=256 um) for a first GPU run; "
                         "Selwyn will not fully converge at this size")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.device is not None:
        DEVICE = args.device
    if args.quick:
        PATCH_UM = 256.0            # smoke size; Selwyn will not fully converge
    if args.patch is not None:
        PATCH_UM = float(args.patch)   # explicit --patch overrides --quick
    if args.fine is not None:
        FINE_UM = float(args.fine)

    print(f"device: {DEVICE}   patch {PATCH_UM} um @ {FINE_UM} um   "
          f"-> {int(PATCH_UM/SCAN_PITCH_UM)} px output\n")
    print("Characteristic curve (D-logE):")
    exposures = [0.002, 0.005, 0.012, 0.03, 0.08, 0.2, 0.5, 1.2, 3.0]
    for e in exposures:
        res = run(e, seed=args.seed)
        print(f"  logE {math.log10(e):+6.2f}   D {res['mean_D']:6.3f}")

    print("\nNutting check at mid density:")
    mid = run(0.08, seed=2, verbose=True)

    print("\nSelwyn root-area check (G should be roughly constant):")
    for a_um, sd, G in selwyn_scan(mid["cov"], FINE_UM):
        print(f"  aperture {a_um:3d} um   sigma_D {sd:.4f}   G {G:.4f}")

    print("\nRMS granularity, 48 um aperture:")
    for a_um, sd, _ in selwyn_scan(mid["cov"], FINE_UM, (48,)):
        print(f"  sigma_D x 1000 = {1000*sd:.1f}   "
              f"(Kodak-style RMS granularity number)")

    print("\nRadial NPS of the scanned pixel field (cyc/um, D^2*um^2):")
    for f, p in radial_nps(mid["D_px"], SCAN_PITCH_UM, nbins=10):
        print(f"  {f:.4f}   {p:.4e}")
