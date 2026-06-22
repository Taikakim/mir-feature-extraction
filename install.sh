#!/usr/bin/env bash
# mir/install.sh — set up the mir feature-extraction environment.
#
# This is a thin wrapper around `requirements.distributable.txt` (the
# authoritative 8-step install spec) plus `install_torchcodec_rocm.sh`. The
# stack is genuinely particular and the install order matters:
#
#   - Python 3.12 (numpy<2.4 because numba/llvmlite pin)
#   - torch 2.9.1+rocm7.2 / torchaudio 2.9.0 / torchvision 0.24.0 / triton 3.5.1
#     all from CUSTOM LOCAL wheels in the repo root (Kim's builds)
#   - ROCm SDK packages (rocm-sdk-libraries-gfx120X-all for RDNA4)
#   - TensorFlow ROCm from AMD's manylinux repo (Step 3 in distributable.txt)
#   - onnxruntime-migraphx from a local wheel (AMD GPU ONNX backend)
#   - Essentia (pre-built wheels, not on PyPI)
#   - flash_attn==2.8.3 (Triton-AMD path, must match torch 2.9)
#   - Several git-sourced packages (audiobox, llama-cpp-python, madmom, …)
#   - PATCHED copies of 9 upstream repos in repos/ (ADTOF-pytorch, BS-RoFormer,
#     bungee, drumsep, llama.cpp, madmom, Qwen2.5-Omni, sox, timbral_models) —
#     these MUST be installed editable AFTER everything else so they win over
#     any pip-installed copy of the same package.
#
# Some packages overwrite the local torch wheels during their own install
# (autoawq, flash_attn, audiobox_aesthetics drag in PyPI torch as a transitive
# dep). The final step force-reinstalls the local wheels to undo that.
#
# Standalone usage:
#   git clone <this-repo> && cd mir && ./install.sh
#
# Options:
#   --venv PATH        venv location (default: ./mir — old-style virtualenv
#                      layout matching MASTER §3, so `mir/bin/python`).
#   --skip-system-deps don't try to install system packages (FFmpeg, etc.)
#   --skip-repos       skip the repos/ editable installs (rerun-able)
#   --skip-torchcodec  skip install_torchcodec_rocm.sh
#   --help

set -euo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
VENV="mir"           # creates ./mir/bin/python (matches MASTER §3)
SKIP_SYSTEM=0
SKIP_REPOS=0
SKIP_TORCHCODEC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --skip-system-deps) SKIP_SYSTEM=1; shift ;;
    --skip-repos) SKIP_REPOS=1; shift ;;
    --skip-torchcodec) SKIP_TORCHCODEC=1; shift ;;
    -h|--help)
      sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO"

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
command -v python3.12 &>/dev/null \
  || { echo "ERROR: python3.12 not in PATH (Arch: pacman -S python312)" >&2; exit 1; }
command -v git &>/dev/null || { echo "ERROR: git not in PATH" >&2; exit 1; }

# Required local wheels (Kim's custom ROCm 7.2 / cp312 builds — non-replaceable).
for w in \
    torch-2.9.1+rocm7.2.0.lw.git*-cp312-cp312-linux_x86_64.whl \
    torchaudio-2.9.0+rocm7.2.0.git*-cp312-cp312-linux_x86_64.whl \
    torchvision-0.24.0+rocm7.2.0.git*-cp312-cp312-linux_x86_64.whl \
    triton-3.5.1+rocm7.2.0.git*-cp312-cp312-linux_x86_64.whl \
    onnxruntime_migraphx-1.23.2-cp312-cp312-*.whl ; do
  ls -1 $w &>/dev/null || {
    echo "ERROR: required local wheel not in repo root: $w" >&2
    echo "       These are custom-built; if you don't have them, the install can't proceed." >&2
    echo "       See requirements.distributable.txt for the build/download details." >&2
    exit 1
  }
done

echo "==> mir install — Python 3.12 / torch 2.9.1 ROCm 7.2 / Essentia + Madmom + MIGraphX"
echo "    Repo:    $REPO"
echo "    Venv:    $REPO/$VENV  (mir/bin/python — old-style virtualenv layout)"
echo

# ---------------------------------------------------------------------------
# 1. System deps (FFmpeg for torchcodec, BLAS, etc.)
# ---------------------------------------------------------------------------
if [[ "$SKIP_SYSTEM" == "0" ]]; then
  echo "==> Step 0: system deps (FFmpeg + pkgconf)"
  if command -v pacman &>/dev/null; then
    sudo pacman -S --needed ffmpeg pkgconf base-devel cmake || true
  elif command -v apt-get &>/dev/null; then
    sudo apt-get update && sudo apt-get install -y ffmpeg pkg-config build-essential cmake || true
  else
    echo "    (unknown distro — install FFmpeg + pkgconf + a build toolchain manually)"
  fi
fi

# ---------------------------------------------------------------------------
# 2. Virtualenv (old-style: mir/bin/python, not .venv/)
# ---------------------------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
  echo "==> Creating virtualenv at $VENV/ (python 3.12)"
  python3.12 -m venv "$VENV"
fi
PY="$REPO/$VENV/bin/python"
PIP="$PY -m pip"

echo "==> Upgrading pip + wheel + setuptools"
$PIP install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# 3. STEP 1 — local torch/triton wheels (must be first, no extra deps)
# ---------------------------------------------------------------------------
echo "==> Step 1: install local torch / torchaudio / torchvision / triton wheels"
$PIP install --no-deps \
  ./torch-2.9.1+rocm7.2.0.lw.git*-cp312-cp312-linux_x86_64.whl \
  ./torchaudio-2.9.0+rocm7.2.0.git*-cp312-cp312-linux_x86_64.whl \
  ./torchvision-0.24.0+rocm7.2.0.git*-cp312-cp312-linux_x86_64.whl \
  ./triton-3.5.1+rocm7.2.0.git*-cp312-cp312-linux_x86_64.whl

# ---------------------------------------------------------------------------
# 4. STEP 4 — onnxruntime-migraphx (local wheel)
# ---------------------------------------------------------------------------
echo "==> Step 4: onnxruntime-migraphx (local wheel)"
$PIP install --no-deps ./onnxruntime_migraphx-1.23.2-cp312-cp312-*.whl

# ---------------------------------------------------------------------------
# 5. xformers (local wheel, often dragged-over later by other installs)
# ---------------------------------------------------------------------------
if ls ./xformers-*-cp39-abi3-linux_x86_64.whl &>/dev/null; then
  echo "==> xformers (local wheel)"
  $PIP install --no-deps ./xformers-*-cp39-abi3-linux_x86_64.whl
fi

# ---------------------------------------------------------------------------
# 6. STEP 7 + 8 + everything else — the long requirements.txt
#    (covers essentia, flash_attn, all the git-sourced pkgs, ROCm SDK,
#     TF-ROCm, ~180 transitive deps).
# ---------------------------------------------------------------------------
echo "==> Steps 2-3 + 5-9: pip install -r requirements.txt"
echo "    (this installs ROCm SDK, TF-ROCm, Essentia, flash_attn 2.8.3,"
echo "     audiobox_aesthetics, madmom, llama_cpp_python, etc. — ~180 deps)"
$PIP install -r requirements.txt

# ---------------------------------------------------------------------------
# 7. REINSTALL local torch wheels — some deps (autoawq, flash_attn,
#    audiobox_aesthetics) drag PyPI torch over our ROCm wheel during install.
# ---------------------------------------------------------------------------
echo "==> Re-installing local torch wheels (some deps overwrite during install)"
$PIP install --no-deps --force-reinstall \
  ./torch-2.9.1+rocm7.2.0.lw.git*-cp312-cp312-linux_x86_64.whl \
  ./torchaudio-2.9.0+rocm7.2.0.git*-cp312-cp312-linux_x86_64.whl \
  ./torchvision-0.24.0+rocm7.2.0.git*-cp312-cp312-linux_x86_64.whl \
  ./triton-3.5.1+rocm7.2.0.git*-cp312-cp312-linux_x86_64.whl

# ---------------------------------------------------------------------------
# 8. torchcodec (needs a custom build for ROCm)
# ---------------------------------------------------------------------------
if [[ "$SKIP_TORCHCODEC" == "0" ]]; then
  echo "==> torchcodec (from source, ROCm-compatible)"
  if [[ -x ./install_torchcodec_rocm.sh ]]; then
    # The torchcodec script uses `python3` — make sure it picks up our venv.
    PATH="$REPO/$VENV/bin:$PATH" ./install_torchcodec_rocm.sh
  else
    echo "    install_torchcodec_rocm.sh not executable; skipping (chmod +x and rerun if needed)"
  fi
fi

# ---------------------------------------------------------------------------
# 9. Patched repos/ packages (editable — must come last so they win over any
#    pip-installed same-named pkg)
# ---------------------------------------------------------------------------
if [[ "$SKIP_REPOS" == "0" ]]; then
  echo "==> Installing patched packages from repos/ (editable)"
  if [[ -d repos ]]; then
    for d in repos/*/; do
      [[ -d "$d" ]] || continue
      # Some of these (e.g. llama.cpp, sox, bungee, drumsep) aren't pip-installable
      # packages — they're C/C++ projects. Detect a pyproject.toml or setup.py;
      # skip the rest with a note. Patched python pkgs (madmom, ADTOF-pytorch,
      # BS-RoFormer, Qwen2.5-Omni, timbral_models) get -e installed.
      if [[ -f "$d/pyproject.toml" || -f "$d/setup.py" ]]; then
        echo "    [editable] $d"
        $PIP install --no-deps -e "$d" || echo "    (skipped: install failed — patch may need fixing for the current torch)"
      else
        echo "    [skip] $d (no pyproject.toml/setup.py — assumed to be a C/C++ project)"
      fi
    done
  fi
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo
echo "==> Verifying install"
"$PY" - <<'PY'
import sys
print(f"python   {sys.version.split()[0]}")
try:
    import torch
    print(f"torch    {torch.__version__}  hip={torch.version.hip}")
    print(f"GPU      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
except Exception as e:
    print(f"torch: FAIL ({e})")
for mod in ("numpy", "essentia", "madmom", "librosa", "flash_attn",
            "onnxruntime", "tensorflow", "transformers"):
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        print(f"{mod:14s} {v}")
    except Exception as e:
        print(f"{mod:14s} MISSING ({type(e).__name__}: {e})")
PY

echo
echo "==> Done."
echo "    Activate:  source $VENV/bin/activate"
echo "    Notes:     See requirements.distributable.txt for the authoritative install spec."
echo "    Tip:       If you see a 'cannot import torch._C' after a `pip install` of"
echo "               something new, re-run the local torch wheel reinstall block above."
