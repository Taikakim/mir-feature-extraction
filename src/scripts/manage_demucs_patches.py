
import os
import shutil
import subprocess
import sys
from pathlib import Path
import logging

# Configuration
REPO_PATH = Path("/home/kim/Projects/repos/demucs")
PATCHES_DIR = Path(__file__).parent.parent.parent / "patches"  # Optional storage for diffs

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("PatchManager")

STATIC_WRAPPER_CODE = """
import torch
from torch import nn
import torch.nn.functional as F
import math

class StaticSegmentModel(nn.Module):
    \"\"\"
    Wrapper to enforce fixed input shapes for compilation optimization.
    Pads input to 'segment_length' and unpads output.
    \"\"\"
    def __init__(self, model, segment_length, device='cuda'):
        super().__init__()
        self.model = model
        self.segment_length = segment_length
        self.device = device
        # Ensure model is in eval mode
        self.model.eval()

    def __getattr__(self, name):
        # Proxy attributes to underlying model (e.g. samplerate, sources)
        return getattr(self.model, name)

    def forward(self, mix):
        # mix shape: (Batch, Channels, Time)
        B, C, T = mix.shape
        target = self.segment_length
        
        if T == target:
            return self.model(mix)
            
        # Pad if smaller
        if T < target:
            padding = target - T
            mix_padded = F.pad(mix, (0, padding))
            out_padded = self.model(mix_padded)
            return out_padded[..., :T]
            
        # If larger, we assume the caller has handled splitting
        # But if not, we error out because this wrapper expects pre-split chunks
        raise ValueError(f"Input length {T} exceeds target segment {target}. Split before calling.")

"""

def check_repo():
    if not REPO_PATH.exists():
        logger.error(f"Demucs repo not found at {REPO_PATH}")
        sys.exit(1)
    logger.info(f"Found Demucs repo at {REPO_PATH}")

def apply_patches():
    check_repo()
    logger.info("Applying patches...")

    # 1. Add Static Wrapper
    wrapper_path = REPO_PATH / "demucs" / "static_wrapper.py"
    logger.info(f"Creating {wrapper_path}...")
    with open(wrapper_path, "w") as f:
        f.write(STATIC_WRAPPER_CODE)

    # 2. Modify __init__.py to expose it (Optional, but good practice)
    # For now, we can import it directly from file path in our code

    logger.info("✅ Patches applied successfully.")

def revert_patches():
    check_repo()
    logger.info("Reverting patches...")
    
    # 1. Git checkout to wipe changes
    try:
        subprocess.run(["git", "checkout", "."], cwd=REPO_PATH, check=True)
        subprocess.run(["git", "clean", "-fd"], cwd=REPO_PATH, check=True)
        logger.info("✅ Reverted to clean git state.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to revert: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python manage_demucs_patches.py [apply|revert]")
        sys.exit(1)
        
    action = sys.argv[1]
    if action == "apply":
        apply_patches()
    elif action == "revert":
        revert_patches()
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
