"""
ADTOF-PyTorch Wrapper for MIR Project

Automatic Drum Transcription using the ADTOF-PyTorch pre-trained model.
Outputs General MIDI drum files with 5 classes:
    - Bass Drum (35)
    - Snare Drum (38) 
    - Tom-Tom (47)
    - Hi-Hat (42)
    - Cymbal/Ride (49)

Uses PyTorch with ROCm GPU acceleration.

Usage:
    python src/transcription/drums/adtof.py /path/to/audio.flac
    python src/transcription/drums/adtof.py /path/to/folder --batch
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.common import setup_logging

logger = logging.getLogger(__name__)

# Default device - use cuda for ROCm GPU
DEFAULT_DEVICE = "cuda"


def transcribe_file(audio_path: Path, output_path: Optional[Path] = None, 
                    device: str = DEFAULT_DEVICE) -> Optional[Path]:
    """
    Transcribe drums from an audio file using ADTOF-PyTorch.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path for output MIDI file (default: same dir as audio, named drums_adtof.mid)
        device: PyTorch device ("cuda" for GPU, "cpu" for CPU)
        
    Returns:
        Path to output MIDI file, or None if failed
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return None
        
    if output_path is None:
        output_path = audio_path.parent / "drums_adtof.mid"
    else:
        output_path = Path(output_path)
        
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        from adtof_pytorch import transcribe_to_midi
        
        logger.info(f"Transcribing: {audio_path.name} (device={device})")
        
        # ADTOF-PyTorch transcribe_to_midi handles everything
        transcribe_to_midi(
            str(audio_path),
            str(output_path),
            device=device
        )
        
        if output_path.exists():
            logger.info(f"Created: {output_path}")
            return output_path
        else:
            logger.error("ADTOF did not produce output file")
            return None
            
    except ImportError as e:
        logger.error(f"Failed to import adtof_pytorch: {e}")
        logger.error("Install with: pip install git+https://github.com/xavriley/ADTOF-pytorch.git")
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def transcribe_folder(folder_path: Path, force: bool = False, 
                      device: str = DEFAULT_DEVICE) -> int:
    """
    Transcribe all audio files in a folder.
    
    Args:
        folder_path: Path to folder containing audio files
        force: Overwrite existing MIDI files
        device: PyTorch device
        
    Returns:
        Number of files successfully transcribed
    """
    folder_path = Path(folder_path)
    
    if not folder_path.is_dir():
        logger.error(f"Not a directory: {folder_path}")
        return 0
        
    # Look for common audio formats
    audio_extensions = {'.flac', '.wav', '.mp3', '.ogg'}
    audio_files = [f for f in folder_path.iterdir() 
                   if f.suffix.lower() in audio_extensions]
    
    if not audio_files:
        logger.warning(f"No audio files found in {folder_path}")
        return 0
        
    success_count = 0
    
    for audio_file in audio_files:
        output_path = folder_path / f"{audio_file.stem}_adtof.mid"
        
        if output_path.exists() and not force:
            logger.info(f"Skipping (exists): {audio_file.name}")
            continue
            
        result = transcribe_file(audio_file, output_path, device=device)
        if result:
            success_count += 1
            
    return success_count


def batch_transcribe(root_path: Path, force: bool = False,
                     device: str = DEFAULT_DEVICE) -> dict:
    """
    Batch transcribe all organized folders.
    
    Args:
        root_path: Root directory to search
        force: Overwrite existing files
        device: PyTorch device
        
    Returns:
        Statistics dict with success/failed counts
    """
    from src.core.file_utils import find_organized_folders, get_stem_files
    
    folders = find_organized_folders(root_path)
    logger.info(f"Found {len(folders)} organized folders")
    
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    
    for folder in folders:
        output_path = folder / "drums_adtof.mid"
        
        if output_path.exists() and not force:
            logger.info(f"Skipping (exists): {folder.name}")
            stats['skipped'] += 1
            continue
            
        # Find full_mix or drums stem
        stems = get_stem_files(folder, include_full_mix=True)
        
        # Prefer drums stem if available, otherwise use full_mix
        audio_file = stems.get('drums') or stems.get('full_mix')
        
        if not audio_file:
            logger.warning(f"No audio found in {folder.name}")
            stats['failed'] += 1
            continue
            
        result = transcribe_file(audio_file, output_path, device=device)
        
        if result:
            stats['success'] += 1
        else:
            stats['failed'] += 1
            
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Transcribe drums using ADTOF-PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python adtof.py /path/to/song.flac
  
  # Single file with custom output
  python adtof.py /path/to/song.flac --output /path/to/drums.mid
  
  # Batch process all organized folders
  python adtof.py /path/to/data --batch
  
  # Use CPU instead of GPU
  python adtof.py /path/to/song.flac --device cpu
        """
    )
    
    parser.add_argument("path", help="Path to audio file or directory")
    parser.add_argument("--output", "-o", help="Output MIDI file path (single file mode)")
    parser.add_argument("--batch", action="store_true", 
                        help="Batch process all organized folders in directory")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite existing MIDI files")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help="PyTorch device: cuda (default) or cpu")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    input_path = Path(args.path)
    
    if not input_path.exists():
        logger.error(f"Path does not exist: {input_path}")
        sys.exit(1)
        
    if args.batch:
        # Batch mode
        if not input_path.is_dir():
            logger.error("--batch requires a directory path")
            sys.exit(1)
            
        stats = batch_transcribe(input_path, force=args.force, device=args.device)
        
        print("\n" + "=" * 40)
        print("ADTOF Batch Transcription Summary:")
        print(f"  Success: {stats['success']}")
        print(f"  Failed:  {stats['failed']}")
        print(f"  Skipped: {stats['skipped']}")
        print("=" * 40)
        
    elif input_path.is_file():
        # Single file mode
        output = Path(args.output) if args.output else None
        result = transcribe_file(input_path, output, device=args.device)
        
        if result:
            print(f"\nCreated: {result}")
        else:
            sys.exit(1)
            
    elif input_path.is_dir():
        # Directory mode (transcribe all audio in folder)
        count = transcribe_folder(input_path, force=args.force, device=args.device)
        print(f"\nTranscribed {count} files")
        
    else:
        logger.error(f"Invalid path: {input_path}")
        sys.exit(1)
