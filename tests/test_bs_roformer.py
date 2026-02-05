#!/usr/bin/env python3
"""
BS Roformer Testing Script using python-audio-separator

Standalone testing script to run multiple BS Roformer models for music source
separation using the python-audio-separator library.

This script is SEPARATE from the main MIR project and should not modify any
existing components.

Requirements:
    pip install audio-separator[gpu]

    For AMD ROCm:
    pip install audio-separator
    # Ensure PyTorch is installed with ROCm support

Usage:
    # List available models
    python test_bs_roformer.py --list-models
    
    # Process with default BS Roformer model
    python test_bs_roformer.py /path/to/audio.wav --output /path/to/output
    
    # Process with multiple models
    python test_bs_roformer.py /path/to/audio.wav --models all
    
    # Process with specific models
    python test_bs_roformer.py /path/to/audio.wav --models "bs_roformer,mel_band"
    
    # Use custom model files
    python test_bs_roformer.py audio.wav --custom-model /path/to/model.ckpt

Output Structure:
    output/
    ├── model_bs_roformer_ep_317/
    │   ├── audio_(Vocals).wav
    │   └── audio_(Instrumental).wav
    ├── model_mel_band_roformer/
    │   ├── audio_(Vocals).wav
    │   └── audio_(Other).wav
    └── ...

Author: Testing script for BS Roformer evaluation (using python-audio-separator)
Date: 2026-02-05
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# =============================================================================
# ROCm Environment Configuration
# =============================================================================
# Must be set BEFORE importing PyTorch

# ROCm memory management
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'garbage_collection_threshold:0.8')

# Enable Flash Attention for AMD (Triton-based)
os.environ.setdefault('FLASH_ATTENTION_TRITON_AMD_ENABLE', 'TRUE')

# Enable tunable operations for ROCm optimization
os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_TUNING', '0')

# Thread configuration
os.environ.setdefault('OMP_NUM_THREADS', '8')

# MIOpen configuration
os.environ.setdefault('MIOPEN_FIND_MODE', '2')

# Suppress TensorFlow warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# =============================================================================
# Imports
# =============================================================================

try:
    import torch
    TORCH_AVAILABLE = True
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        rocm_version = getattr(torch.version, 'hip', None)
        print(f"ROCm version: {rocm_version}" if rocm_version else f"CUDA version: {torch.version.cuda}")
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available")

try:
    from audio_separator.separator import Separator
    SEPARATOR_AVAILABLE = True
except ImportError:
    SEPARATOR_AVAILABLE = False
    print("WARNING: audio-separator not available. Install with: pip install audio-separator[gpu]")


# =============================================================================
# Custom Separator to bypass model validation
# =============================================================================

class CustomSeparator(Separator):
    def download_model_files(self, model_filename):
        # Override to allow local custom models
        path = Path(self.model_file_dir) / model_filename
        if path.exists():
             # Assume it's a valid model we want to load
             # We assume MDXC type for Roformer models as they are handled by MDXC architecture in valid list
             model_type = "MDXC" 
             model_friendly_name = model_filename
             
             # Auto-discovery of yaml config
             yaml_file = None
             
             # 1. Check if model itself is yaml (not usual but possible)
             if path.suffix.lower() == '.yaml':
                 yaml_file = str(path)
             else:
                 # 2. Check for same name with .yaml extension
                 yaml_path = path.with_suffix('.yaml')
                 if yaml_path.exists():
                     yaml_file = str(yaml_path)
                 else:
                     # 3. Check for config.yaml in same directory
                     config_path = path.parent / 'config.yaml'
                     if config_path.exists():
                         yaml_file = str(config_path)
             
             # Return valid metadata as if downloaded
             # (model_filename, model_type, model_friendly_name, model_path, yaml_config_filename)
             return model_filename, model_type, model_friendly_name, str(path), yaml_file
        
        return super().download_model_files(model_filename)


# =============================================================================
# Recommended BS Roformer Models (from audio-separator's model list)
# =============================================================================

# Best vocal separation models (by SDR score)
RECOMMENDED_MODELS = {
    # === BS Roformer Models (Best Quality) ===
    "bs_roformer_viperx_1297": {
        "filename": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "description": "BS-Roformer-Viperx-1297 - Best vocal separation (SDR 12.97)",
        "stems": ["vocals", "instrumental"],
        "category": "vocals",
    },
    "bs_roformer_viperx_1296": {
        "filename": "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "description": "BS-Roformer-Viperx-1296 - Excellent vocal separation (SDR 12.96)",
        "stems": ["vocals", "instrumental"],
        "category": "vocals",
    },
    
    # === Mel-Band Roformer Models ===
    "mel_band_roformer_vocals": {
        "filename": "vocals_mel_band_roformer.ckpt",
        "description": "MelBand Roformer Vocals by Kimberley Jensen (SDR 12.6)",
        "stems": ["vocals", "other"],
        "category": "vocals",
    },
    "mel_band_roformer_big_beta4": {
        "filename": "melband_roformer_big_beta4.ckpt",
        "description": "MelBand Roformer Kim Big Beta 4 (SDR 12.5)",
        "stems": ["vocals", "other"],
        "category": "vocals",
    },
    "mel_band_roformer_kim_ft": {
        "filename": "mel_band_roformer_kim_ft_unwa.ckpt",
        "description": "MelBand Roformer Kim FT by unwa (SDR 12.4)",
        "stems": ["vocals", "other"],
        "category": "vocals",
    },
    
    # === Instrumental Models ===
    "bs_roformer_instrumental": {
        "filename": "model_bs_roformer_ep_937_sdr_10.5309.ckpt",
        "description": "BS-Roformer Instrumental extraction",
        "stems": ["instrumental", "vocals"],
        "category": "instrumental",
    },
    
    # === Default mel-band roformer ===
    "mel_band_roformer_default": {
        "filename": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        "description": "Default MelBand Roformer (audio-separator default)",
        "stems": ["vocals", "other"],
        "category": "vocals",
    },
}

# Models for testing with custom checkpoints from /models/bs-roformer/
CUSTOM_MODEL_MAPPINGS = {
    "pcunwa-BS-Roformer-Revive-vocal": "bs_roformer_revive3e.ckpt",
    "pcunwa-BS-Roformer-Large-Inst-instrumental": "bs_large_v2_inst.ckpt",
    "pcunwa-BS-Roformer-HyperACE-v1-inst": "bs_hyperace.ckpt",
    "pcunwa-BS-Roformer-HyperACE-v2-inst": "bs_roformer_inst_hyperacev2.ckpt",
    "pcunwa-BS-Roformer-HyperACE-v2-voc": "bs_roformer_voc_hyperacev2.ckpt",
    "jarredou-BS-ROFO-SW-Fixed-drums": "BS-Rofo-SW-Fixed.ckpt",
    "LayerNorm-BS-Roformer-Inst-FNO-drums": "bs_roformer_fno.safetensors",
    "becruily-mel-band-roformer-deux": "becruily_deux.ckpt",
}


def list_available_models():
    """List all available models from audio-separator."""
    print("\n" + "=" * 80)
    print("Available BS Roformer / MelBand Roformer Models")
    print("=" * 80)
    
    print("\n📦 RECOMMENDED MODELS (from audio-separator repo):\n")
    for key, info in RECOMMENDED_MODELS.items():
        print(f"  {key}:")
        print(f"    Filename: {info['filename']}")
        print(f"    Description: {info['description']}")
        print(f"    Stems: {', '.join(info['stems'])}")
        print()
    
    print("\n📁 CUSTOM MODELS (from /models/bs-roformer/):\n")
    custom_models_dir = Path("/home/kim/Projects/mir/models/bs-roformer")
    for folder_name, ckpt_name in CUSTOM_MODEL_MAPPINGS.items():
        folder_path = custom_models_dir / folder_name.replace("-v1-inst", "/v1-inst").replace("-v2-inst", "/v2-inst").replace("-v2-voc", "/v2-voc")
        if not folder_path.exists():
            # Try without path modifications
            for subdir in custom_models_dir.iterdir():
                if folder_name.startswith(subdir.name):
                    folder_path = subdir
                    break
        
        status = "✓" if folder_path.exists() else "✗"
        print(f"  {status} {folder_name}: {ckpt_name}")


def process_with_model(
    args: argparse.Namespace,
    model_filename: str,
    audio_files: List[Path],
    output_dir: Path,
    model_name: str,
    config_path: Optional[str] = None,
) -> dict:
    """
    Process audio files with a specific model.
    
    Returns dict with processing results.
    """
    results = {
        "model": model_name,
        "files_processed": 0,
        "files_failed": 0,
        "output_files": [],
        "elapsed_time": 0,
    }
    
    # Create model-specific output directory
    model_output_dir = output_dir / model_name.replace(".", "_").replace("/", "_")
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"File: {model_filename}")
    print(f"Output: {model_output_dir}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Initialize Separator with custom model directory
        model_path = Path(model_filename)
        model_dir = model_path.parent
        model_file = model_path.name
        
        separator = CustomSeparator(
            output_dir=str(model_output_dir),
            output_format="wav",
            sample_rate=args.sample_rate,
            model_file_dir=str(model_dir)
        )
        
        # Load the model
        print(f"Loading model: {model_file}")
        if config_path:
             print(f"  Config (implicit): {config_path}")
        
        # Load using just the filename (it will look in model_file_dir)
        separator.load_model(model_filename=model_file)
        
        # Process each file
        for audio_file in audio_files:
            print(f"\nProcessing: {audio_file.name}")
            try:
                # Separate
                output_files = separator.separate(str(audio_file))
                
                results["files_processed"] += 1
                results["output_files"].extend(output_files)
                
                for out_file in output_files:
                    print(f"  ✓ Created: {Path(out_file).name}")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results["files_failed"] += 1
                
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        results["files_failed"] = len(audio_files)
    
    results["elapsed_time"] = time.time() - start_time
    print(f"\nModel time: {results['elapsed_time']:.1f}s")
    
    return results


MODELS_DIR = Path("/home/kim/Projects/mir/models/bs-roformer")

def discover_models(models_dir: Path = MODELS_DIR) -> Dict[str, Dict[str, Any]]:
    """Discover available models in the models directory (including nested)."""
    models = {}
    
    if not models_dir.exists():
        return models
    
    def find_model_in_folder(folder: Path, prefix: str = "") -> None:
        """Recursively search for model configs and checkpoints."""
        # Find config file - try common patterns
        config_files = list(folder.glob("config.yaml")) + list(folder.glob("*.yaml"))
        if not config_files:
            # Check subdirectories
            for subdir in folder.iterdir():
                if subdir.is_dir():
                    sub_prefix = f"{prefix}/{subdir.name}" if prefix else subdir.name
                    find_model_in_folder(subdir, sub_prefix)
            return
        
        # Use first config found
        config_path = config_files[0]
        
        # Find checkpoint file
        ckpt_files = list(folder.glob("*.ckpt")) + list(folder.glob("*.pth")) + list(folder.glob("*.safetensors"))
        if not ckpt_files:
            return
        
        # Detect if it's MelBand
        folder_str = str(folder).lower()
        is_mel_band = "mel" in folder_str or "melband" in folder_str
        
        # Create base name
        base_name = f"{prefix}/{folder.name}" if prefix else folder.name
        base_name = base_name.replace("/", "_")
        
        # If multiple checkpoints, register each as a separate model
        if len(ckpt_files) > 1:
            for ckpt_path in ckpt_files:
                # Use checkpoint filename as suffix
                ckpt_name = ckpt_path.stem
                if ckpt_name.startswith(folder.name):
                    # Avoid redundancy if filename repeats folder name
                    suffix = ckpt_name[len(folder.name):].strip("-_")
                else:
                    suffix = ckpt_name
                
                model_name = f"{base_name}_{suffix}" if suffix else base_name
                
                models[model_name] = {
                    "path": folder,
                    "config": config_path,
                    "checkpoint": ckpt_path,
                    "is_mel_band": is_mel_band,
                }
        else:
            # Single checkpoint - use folder name
            models[base_name] = {
                "path": folder,
                "config": config_path,
                "checkpoint": ckpt_files[0],
                "is_mel_band": is_mel_band,
            }
    
    # Scan top-level directories
    for folder in models_dir.iterdir():
        if folder.is_dir():
            find_model_in_folder(folder, "")
    
    return models


def main():
    parser = argparse.ArgumentParser(
        description="BS Roformer Testing Script (using python-audio-separator)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available models
    python test_bs_roformer.py --list-models
    
    # Process with default settings (auto-detects config in model dir)
    python test_bs_roformer.py audio.wav --model-dir /path/to/model/folder
    
    # Process with specific model name
    python test_bs_roformer.py audio.wav --model-name bs_roformer_ep_317_sdr_12.9755
    
    # Process with direct paths
    python test_bs_roformer.py audio.wav --model-path /path/to/model.ckpt --config-path /path/to/config.yaml
"""
    )
    
    parser.add_argument('input', nargs='?', help='Input audio file or folder')
    parser.add_argument('--output', '-o', type=Path, default=Path('./bs_roformer_output'),
                        help='Output directory')
    
    # Aligned arguments with optimized script
    parser.add_argument('--model-dir', '-m', type=Path, default=Path("/home/kim/Projects/mir/models/bs-roformer"),
                        help='Model directory containing config.yaml and checkpoint')
    parser.add_argument('--model-name', type=str,
                        help='Model name from the models directory')
    
    # Legacy/extra arguments specific to this script
    parser.add_argument('--model-path', type=str,
                        help='Direct path to model checkpoint')
    parser.add_argument('--config-path', type=str,
                        help='Direct path to model config')
    parser.add_argument('--models', type=str, default='all',
                        help='Model selection style (legacy): "all", "custom", or comma-separated keys')
    
    parser.add_argument('--list-models', '-l', action='store_true',
                        help='List available models and exit')
    
    # Compatibility arguments (mapped where possible)
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile() (compatibility flag, might be ignored by audio-separator)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--no-fp16', action='store_true',
                        help='Disable FP16 inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Output sample rate')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without running')
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        list_available_models()
        sys.exit(0)
    
    # Validate input
    if not args.input:
        parser.error("Input path is required unless using --list-models")
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Check dependencies
    if not SEPARATOR_AVAILABLE:
        print("\nERROR: audio-separator is required")
        print("Install with: pip install audio-separator[gpu]")
        sys.exit(1)
    
    # Collect input files
    if input_path.is_file():
        audio_files = [input_path]
    else:
        extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']
        audio_files = []
        for ext in extensions:
            audio_files.extend(input_path.glob(f'**/{ext}'))
        audio_files = sorted(audio_files)
    
    print(f"\n{'='*60}")
    print("BS Roformer Testing Script (using python-audio-separator)")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output: {args.output}")
    print(f"Audio files: {len(audio_files)}")
    
    # Determine which models to use
    models_to_run = []
    
    # Discover available models using shared logic
    available_models = discover_models(args.model_dir) # Use model-dir from args
    
    if args.model_path:
        # Direct paths (highest priority)
        # Verify config exists or was provided
        config_p = args.config_path
        if not config_p:
            # Try to find config in same folder if not provided
            p_path = Path(args.model_path)
            c_cand = p_path.parent / "config.yaml"
            if c_cand.exists():
                config_p = str(c_cand)
                
        models_to_run.append(("custom_model", args.model_path, config_p))
        
    elif args.model_name:
        # Use discover_models logic
        if args.model_name in available_models:
            info = available_models[args.model_name]
            models_to_run.append((args.model_name, str(info['checkpoint']), str(info['config'])))
        else:
            # Fallback checks (legacy recommended models not in folder?)
            if args.model_name in RECOMMENDED_MODELS:
                 models_to_run.append((args.model_name, RECOMMENDED_MODELS[args.model_name]['filename'], None))
            else:
                print(f"ERROR: Model not found: {args.model_name}")
                print(f"Searched in: {args.model_dir}")
                print("Available models:")
                for name in sorted(available_models.keys()):
                    print(f"  - {name}")
                sys.exit(1)

    elif args.custom_model:
        # Legacy custom model arg - try to look it up in discovered models
        if args.custom_model in available_models:
            info = available_models[args.custom_model]
            models_to_run.append((args.custom_model, str(info['checkpoint']), str(info['config'])))
        else:
            # Fallback to old behavior? Or just fail?
            # Old behavior was find_custom_model_path which is now replaced/gone or wrapped
            # Let's try to find it in available models by partial match if strict match failed
            # actually find_custom_model_path logic was "rglob"
            # discover_models does full scan. 
            print(f"ERROR: Custom model not found: {args.custom_model}")
            sys.exit(1)
            
    elif args.models in ('all', 'all+custom'):
        # Use all recommended models
        for key, info in RECOMMENDED_MODELS.items():
            models_to_run.append((key, info['filename'], None))
        
        # Also add all discovered custom models if requested
        if args.models == 'all+custom':
             for name, info in available_models.items():
                 # Avoid duplicates if they overlap (unlikely due to naming)
                 models_to_run.append((name, str(info['checkpoint']), str(info['config'])))
             
    else:
        # Use specified models (comma-separated legacy list)
        for model_key in args.models.split(','):
            model_key = model_key.strip()
            if model_key in RECOMMENDED_MODELS:
                models_to_run.append((model_key, RECOMMENDED_MODELS[model_key]['filename'], None))
            elif model_key in available_models:
                 info = available_models[model_key]
                 models_to_run.append((model_key, str(info['checkpoint']), str(info['config'])))
            else:
                 # Assume it's a filename
                 models_to_run.append((model_key, model_key, None))
    
    print(f"\nModels to run: {len(models_to_run)}")
    for name, filename, config in models_to_run:
        print(f"  - {name}: {filename}")
        if config:
            print(f"    config: {config}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for f in audio_files:
            print(f"  - {f}")
        sys.exit(0)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Initialize separator not needed here, done in process_with_model
    
    # Process with each model
    total_start = time.time()
    all_results = []
    
    for model_name, model_filename, config_path in models_to_run:
        try:
            result = process_with_model(
                args,
                model_filename,
                audio_files,
                args.output,
                model_name,
                config_path
            )
            all_results.append(result)
            
            # Clear GPU memory between models
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"ERROR with model {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Audio files: {len(audio_files)}")
    print(f"Models used: {len(models_to_run)}")
    print(f"Output directory: {args.output}")
    
    for result in all_results:
        status = "✓" if result["files_failed"] == 0 else "✗"
        print(f"\n{status} {result['model']}:")
        print(f"    Processed: {result['files_processed']}")
        print(f"    Failed: {result['files_failed']}")
        print(f"    Time: {result['elapsed_time']:.1f}s")


if __name__ == '__main__':
    main()
