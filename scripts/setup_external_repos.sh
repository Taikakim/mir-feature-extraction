#!/bin/bash
# Setup External Repositories
# This script clones external dependencies, applies patches, and builds binaries.
#
# Usage:
#   ./scripts/setup_external_repos.sh          # Clone and set up all repos
#   ./scripts/setup_external_repos.sh --skip-build   # Skip llama.cpp build (clone only)

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPOS_DIR="$PROJECT_ROOT/repos"
SKIP_BUILD=false

for arg in "$@"; do
    case $arg in
        --skip-build) SKIP_BUILD=true ;;
    esac
done

echo "============================================"
echo "MIR Framework - External Repository Setup"
echo "============================================"
echo ""

# Create repos directory
mkdir -p "$REPOS_DIR"
cd "$REPOS_DIR"

# =========================================================================
# 1. TIMBRAL MODELS (Audio Commons - timbral feature extraction)
#    Install: editable pip install + librosa patches
# =========================================================================
echo "[1/7] timbral_models (Audio Commons)"
if [ ! -d "timbral_models" ]; then
    git clone https://github.com/AudioCommons/timbral_models.git
    cd timbral_models
    git checkout 6a97eec621891a8e875a8968b0ae3fa0e58e2a5c
    cd "$REPOS_DIR"
    echo "  Cloned and pinned"
else
    echo "  Already exists"
fi

# Apply librosa 0.11.0 compatibility patches
cd "$REPOS_DIR/timbral_models"
PATCHED=false

if grep -q "librosa.onset.onset_detect(audio_samples, fs, backtrack=True" timbral_models/timbral_util.py 2>/dev/null; then
    sed -i 's/librosa\.onset\.onset_detect(audio_samples, fs, backtrack=True, units='\''samples'\'')/librosa.onset.onset_detect(y=audio_samples, sr=fs, backtrack=True, units='\''samples'\'')/' timbral_models/timbral_util.py
    PATCHED=true
fi

if grep -q "librosa.onset.onset_strength(audio_samples, fs)" timbral_models/timbral_util.py 2>/dev/null; then
    sed -i 's/librosa\.onset\.onset_strength(audio_samples, fs)/librosa.onset.onset_strength(y=audio_samples, sr=fs)/' timbral_models/timbral_util.py
    PATCHED=true
fi

if grep -q "librosa.core.resample(audio_samples, fs, lowest_fs)" timbral_models/timbral_util.py 2>/dev/null; then
    sed -i 's/librosa\.core\.resample(audio_samples, fs, lowest_fs)/librosa.resample(y=audio_samples, orig_sr=fs, target_sr=lowest_fs)/' timbral_models/timbral_util.py
    PATCHED=true
fi

if grep -q "librosa.onset.onset_strength(audio_samples, fs)" timbral_models/Timbral_Hardness.py 2>/dev/null; then
    sed -i 's/librosa\.onset\.onset_strength(audio_samples, fs)/librosa.onset.onset_strength(y=audio_samples, sr=fs)/' timbral_models/Timbral_Hardness.py
    PATCHED=true
fi

if $PATCHED; then
    echo "  Applied librosa 0.11.0 patches"
else
    echo "  Patches already applied"
fi

pip install -e "$REPOS_DIR/timbral_models" --quiet
echo "  Installed (editable)"

cd "$REPOS_DIR"

# =========================================================================
# 2. BS-ROFORMER (Source separation - vocals/drums/bass/other)
#    Install: editable pip install
# =========================================================================
echo "[2/7] BS-RoFormer (source separation)"
if [ ! -d "BS-RoFormer" ]; then
    git clone https://github.com/lucidrains/BS-RoFormer
    cd BS-RoFormer
    git checkout 1ccb8a8b07735bc45161a880419d0b92110eeb63
    cd "$REPOS_DIR"
    echo "  Cloned and pinned"
else
    echo "  Already exists"
fi

pip install -e "$REPOS_DIR/BS-RoFormer" --quiet
echo "  Installed (editable)"

# =========================================================================
# 3. ADTOF-PYTORCH (Automatic Drum Transcription)
#    Install: editable pip install
# =========================================================================
echo "[3/7] ADTOF-pytorch (drum transcription)"
if [ ! -d "ADTOF-pytorch" ]; then
    git clone https://github.com/xavriley/ADTOF-pytorch.git
    cd ADTOF-pytorch
    git checkout 85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9
    cd "$REPOS_DIR"
    echo "  Cloned and pinned"
else
    echo "  Already exists"
fi

pip install -e "$REPOS_DIR/ADTOF-pytorch" --quiet
echo "  Installed (editable)"

# =========================================================================
# 4. DRUMSEP (Drum stem sub-separation into kick/snare/toms/hihat/cymbals)
#    Install: direct path access from src/transcription/drums/drumsep.py
#    Model: downloaded separately (167MB)
# =========================================================================
echo "[4/7] drumsep (drum sub-separation)"
if [ ! -d "drumsep" ]; then
    git clone https://github.com/inagoy/drumsep
    cd drumsep
    git checkout c1cea3f47bacd410412c7f563109f0a227b0e784
    cd "$REPOS_DIR"
    echo "  Cloned and pinned"
else
    echo "  Already exists"
fi

if [ ! -f "drumsep/model/49469ca8.th" ]; then
    echo "  WARNING: DrumSep model not found at repos/drumsep/model/49469ca8.th"
    echo "  Download it manually — see https://github.com/inagoy/drumsep for instructions"
else
    echo "  Model present"
fi

# =========================================================================
# 5. MADMOM (Beat/downbeat/tempo detection - CPU-only, Cython)
#    Install: pip install from git (needs Cython build)
# =========================================================================
echo "[5/7] madmom (beat/tempo detection)"
if [ ! -d "madmom" ]; then
    git clone https://github.com/CPJKU/madmom.git
    cd madmom
    git checkout 27f032e8947204902c675e5e341a3faf5dc86dae
    cd "$REPOS_DIR"
    echo "  Cloned and pinned"
else
    echo "  Already exists"
fi

pip install "$REPOS_DIR/madmom" --quiet
echo "  Installed"

# =========================================================================
# 6. LLAMA.CPP (GGUF inference for Music Flamingo and audio captioning)
#    Install: cmake build with ROCm/HIP support
#    Binary: build/bin/llama-mtmd-cli
# =========================================================================
echo "[6/7] llama.cpp (GGUF inference)"
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git
    echo "  Cloned"
else
    echo "  Already exists"
fi

if [ "$SKIP_BUILD" = false ]; then
    cd "$REPOS_DIR/llama.cpp"

    # Detect GPU backend
    if command -v rocminfo &>/dev/null; then
        echo "  Building with ROCm/HIP support..."
        cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1
        cmake --build build --target llama-mtmd-cli -j"$(nproc)" 2>&1 | tail -1
    elif command -v nvidia-smi &>/dev/null; then
        echo "  Building with CUDA support..."
        cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1
        cmake --build build --target llama-mtmd-cli -j"$(nproc)" 2>&1 | tail -1
    else
        echo "  Building CPU-only (no ROCm or CUDA detected)..."
        cmake -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1
        cmake --build build --target llama-mtmd-cli -j"$(nproc)" 2>&1 | tail -1
    fi

    if [ -f "build/bin/llama-mtmd-cli" ]; then
        echo "  Built successfully: build/bin/llama-mtmd-cli"
    else
        echo "  ERROR: Build failed — llama-mtmd-cli not found"
        exit 1
    fi
    cd "$REPOS_DIR"
else
    echo "  Skipping build (--skip-build)"
fi

# =========================================================================
# 7. QWEN2.5-OMNI (Audio captioning benchmark - optional)
#    Install: patched modeling file in low-VRAM-mode/
#    Only used by tests/poc_lmm_revise.py
# =========================================================================
echo "[7/7] Qwen2.5-Omni (audio captioning, optional)"
if [ ! -d "Qwen2.5-Omni" ]; then
    git clone https://github.com/QwenLM/Qwen2.5-Omni.git
    cd Qwen2.5-Omni
    git checkout d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42
    cd "$REPOS_DIR"
    echo "  Cloned and pinned"
else
    echo "  Already exists"
fi

if [ -f "Qwen2.5-Omni/low-VRAM-mode/modeling_qwen2_5_omni_low_VRAM_mode.py" ]; then
    echo "  Patched modeling file present"
else
    echo "  NOTE: low-VRAM-mode/ patched modeling file not found"
    echo "  This is only needed for tests/poc_lmm_revise.py (AWQ mode)"
fi

# =========================================================================
# Done
# =========================================================================
cd "$PROJECT_ROOT"

echo ""
echo "============================================"
echo "External repository setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Activate your virtual environment: source mir/bin/activate"
echo "  2. Install remaining Python deps: pip install -r requirements.txt"
echo "  3. Download GGUF models to models/ (see MUSIC_FLAMINGO.md)"
echo ""
