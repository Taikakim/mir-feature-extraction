#!/usr/bin/env python3
"""
Apply patches to external repositories for librosa 0.11.0 compatibility.

This script applies the patches documented in EXTERNAL_PATCHES.md to fix
API incompatibilities in the timbral_models library.
"""

from pathlib import Path
import sys

def apply_patches():
    """Apply all patches to external repositories."""

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    timbral_models_dir = project_root / "repos" / "repos" / "timbral_models"

    if not timbral_models_dir.exists():
        print("❌ Error: timbral_models not found at:", timbral_models_dir)
        print("   Run scripts/setup_external_repos.sh first to clone repositories.")
        sys.exit(1)

    print("🔧 Applying patches to timbral_models...")
    print()

    # Patch 1: timbral_util.py line 642 - onset_detect
    util_file = timbral_models_dir / "timbral_models" / "timbral_util.py"
    if util_file.exists():
        content = util_file.read_text()

        # Patch onset_detect call
        old_pattern = "librosa.onset.onset_detect(audio_samples, fs, backtrack=True, units='samples')"
        new_pattern = "librosa.onset.onset_detect(y=audio_samples, sr=fs, backtrack=True, units='samples')"
        if old_pattern in content:
            print("  ✓ Patching timbral_util.py:642 (onset_detect)")
            content = content.replace(old_pattern, new_pattern)

        # Patch onset_strength call (line 750)
        old_pattern = "onset_strength = librosa.onset.onset_strength(audio_samples, fs)"
        new_pattern = "onset_strength = librosa.onset.onset_strength(y=audio_samples, sr=fs)"
        if old_pattern in content:
            print("  ✓ Patching timbral_util.py:750 (onset_strength)")
            content = content.replace(old_pattern, new_pattern, 1)  # Only first occurrence

        # Patch resample call (line 1813)
        old_pattern = "audio_samples = librosa.core.resample(audio_samples, fs, lowest_fs)"
        new_pattern = "audio_samples = librosa.resample(y=audio_samples, orig_sr=fs, target_sr=lowest_fs)"
        if old_pattern in content:
            print("  ✓ Patching timbral_util.py:1813 (resample)")
            content = content.replace(old_pattern, new_pattern)

        util_file.write_text(content)
    else:
        print(f"  ⚠️  Warning: Could not find {util_file}")

    # Patch 2: Timbral_Hardness.py line 88
    hardness_file = timbral_models_dir / "timbral_models" / "Timbral_Hardness.py"
    if hardness_file.exists():
        content = hardness_file.read_text()

        old_pattern = "onset_strength = librosa.onset.onset_strength(audio_samples, fs)"
        new_pattern = "onset_strength = librosa.onset.onset_strength(y=audio_samples, sr=fs)"
        if old_pattern in content:
            print("  ✓ Patching Timbral_Hardness.py:88 (onset_strength)")
            content = content.replace(old_pattern, new_pattern)
            hardness_file.write_text(content)
    else:
        print(f"  ⚠️  Warning: Could not find {hardness_file}")

    print()
    print("✅ All patches applied successfully!")
    print()
    print("These patches fix librosa 0.11.0 API compatibility issues.")
    print("See EXTERNAL_PATCHES.md for details.")

if __name__ == "__main__":
    apply_patches()
