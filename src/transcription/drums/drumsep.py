"""
DrumSep Wrapper for MIR Project

This module provides a wrapper for DrumSep, a custom Demucs model for separating drums
into finer components (Kick, Snare, Cymbals, Toms, etc.).

It interfaces with the external DrumSep model located in `repos/drumsep/model`.
"""

import logging
import sys
import torch
from pathlib import Path
from typing import Optional, List
from unittest.mock import patch

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

from src.core.common import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Fix torchaudio backend to avoid torchcodec requirement
# Monkeypatch torchaudio.save because set_audio_backend is not working relative to demucs usage
try:
    import torchaudio
    import soundfile as sf
    import torch
    
    def patched_save(uri, src, sample_rate, **kwargs):
        """Replacement for torchaudio.save that uses soundfile directly."""
        # src is TensorShape([channels, time])
        # soundfile expects (time, channels) for multi-channel
        if isinstance(src, torch.Tensor):
            src = src.detach().cpu().numpy()
            
        if src.ndim == 2:
            src = src.T
            
        sf.write(str(uri), src, sample_rate)
        
    logger.info("Monkeypatching torchaudio.save to use soundfile...")
    torchaudio.save = patched_save
    
except Exception as e:
    logger.warning(f"Failed to patch torchaudio: {e}")

DRUMSEP_REPO = PROJECT_ROOT / "repos/drumsep"
DRUMSEP_MODEL = DRUMSEP_REPO / "model"
MODEL_HASH = "49469ca8"

def check_drumsep_availability() -> bool:
    """Check if DrumSep model is available."""
    if not DRUMSEP_MODEL.exists():
        return False
    model_file = DRUMSEP_MODEL / f"{MODEL_HASH}.th"
    return model_file.exists()

def separate_drums(audio_path: str | Path, output_dir: Optional[str | Path] = None) -> bool:
    """
    Run DrumSep extraction on an audio file.
    
    Args:
        audio_path: Path to input audio
        output_dir: Output directory (default: same as input parent)
        
    Returns:
        True if successful, False otherwise
    """
    audio_path = Path(audio_path)
    
    if not check_drumsep_availability():
        logger.error("DrumSep model not available in repos/drumsep.")
        return False
        
    if output_dir is None:
        output_dir = audio_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    logger.info(f"Running DrumSep on: {audio_path.name}")
    
    # Prepare arguments for demucs
    args = [
        "demucs",
        "--repo", str(DRUMSEP_MODEL),
        "-o", str(output_dir),
        "-n", MODEL_HASH,
        str(audio_path)
    ]
    
    try:
        # Import demucs modules inside function to avoid import errors if not installed
        import demucs.separate
        import demucs.hdemucs
        
        # Register the HDemucs class as safe for unpickling
        # This fixes the "Weights only load failed" error in PyTorch 2.6+
        torch.serialization.add_safe_globals([demucs.hdemucs.HDemucs])
        
        # Run demucs main with patched argv
        with patch.object(sys, 'argv', args):
            demucs.separate.main()
            
        logger.info("DrumSep separation complete.")
        return True
        
    except SystemExit as e:
        if e.code != 0:
            logger.error(f"Demucs exited with code {e.code}")
            return False
        return True
    except ImportError as e:
        logger.error(f"Demucs import failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"DrumSep failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    import argparse
    from src.core.common import setup_logging
    
    parser = argparse.ArgumentParser(description="Run DrumSep separation")
    parser.add_argument("path", help="Path to audio file")
    parser.add_argument("--out", help="Output directory", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    separate_drums(args.path, args.out)
