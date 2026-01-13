#!/bin/bash
# Setup External Repositories
# This script clones external dependencies and applies necessary patches

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPOS_DIR="$PROJECT_ROOT/repos/repos"

echo "============================================"
echo "MIR Framework - External Repository Setup"
echo "============================================"
echo ""

# Create repos directory
mkdir -p "$REPOS_DIR"
cd "$REPOS_DIR"

echo "📦 Cloning external repositories..."
echo ""

# Clone timbral_models
if [ ! -d "timbral_models" ]; then
    echo "Cloning Audio Commons timbral_models..."
    git clone https://github.com/AudioCommons/timbral_models.git
    cd timbral_models
    # Optional: checkout specific commit for reproducibility
    # git checkout <commit-hash>
    cd "$REPOS_DIR"
    echo "✓ timbral_models cloned"
else
    echo "✓ timbral_models already exists"
fi

echo ""
echo "🔧 Applying patches to external repositories..."
echo ""

# Apply timbral_models patches
cd "$REPOS_DIR/timbral_models"

echo "Patching timbral_models for librosa 0.11.0 compatibility..."

# Patch 1: timbral_util.py line 642
if grep -q "librosa.onset.onset_detect(audio_samples, fs, backtrack=True" timbral_models/timbral_util.py; then
    echo "  Patching timbral_util.py:642 (onset_detect)"
    sed -i 's/librosa\.onset\.onset_detect(audio_samples, fs, backtrack=True, units='\''samples'\'')/librosa.onset.onset_detect(y=audio_samples, sr=fs, backtrack=True, units='\''samples'\'')/' timbral_models/timbral_util.py
fi

# Patch 2: timbral_util.py line 750
if grep -q "librosa.onset.onset_strength(audio_samples, fs)" timbral_models/timbral_util.py; then
    echo "  Patching timbral_util.py:750 (onset_strength)"
    sed -i 's/librosa\.onset\.onset_strength(audio_samples, fs)/librosa.onset.onset_strength(y=audio_samples, sr=fs)/' timbral_models/timbral_util.py
fi

# Patch 3: timbral_util.py line 1813
if grep -q "librosa.core.resample(audio_samples, fs, lowest_fs)" timbral_models/timbral_util.py; then
    echo "  Patching timbral_util.py:1813 (resample)"
    sed -i 's/librosa\.core\.resample(audio_samples, fs, lowest_fs)/librosa.resample(y=audio_samples, orig_sr=fs, target_sr=lowest_fs)/' timbral_models/timbral_util.py
fi

# Patch 4: Timbral_Hardness.py line 88
if grep -q "librosa.onset.onset_strength(audio_samples, fs)" timbral_models/Timbral_Hardness.py; then
    echo "  Patching Timbral_Hardness.py:88 (onset_strength)"
    sed -i 's/librosa\.onset\.onset_strength(audio_samples, fs)/librosa.onset.onset_strength(y=audio_samples, sr=fs)/' timbral_models/Timbral_Hardness.py
fi

echo "✓ timbral_models patches applied"

cd "$PROJECT_ROOT"

echo ""
echo "============================================"
echo "✅ External repository setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Activate your virtual environment: source mir/bin/activate"
echo "  2. Install Python dependencies: pip install -r requirements.txt"
echo "  3. Add timbral_models to PYTHONPATH or install: pip install -e repos/repos/timbral_models"
echo ""
