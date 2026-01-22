"""
Batch Transcription Runner

This script orchestrates the transcription process:
1. Identifies organized audio folders.
2. Runs DrumSep (Demucs) to separate drum stems.
3. Detects onsets in separated stems.
4. Converts onsets to MIDI files (drums.mid).
5. (Future) Runs other transcribers (bass, melody).
"""

import logging
import shutil
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.file_utils import find_organized_folders, get_stem_files
from src.core.common import setup_logging
from src.transcription.drums.drumsep import separate_drums
from src.transcription.midi_utils import create_midi_from_stems
from src.rhythm.onsets import detect_onsets

logger = logging.getLogger(__name__)

def transcribe_drums(folder_path: Path, force: bool = False) -> bool:
    """
    Transcribe drums for a single folder.
    Process:
        1. Check if drums.mid exists (skip if not force).
        2. Run DrumSep to separate full_mix.flac into temp stems.
        3. Detect onsets for each stem (kick, snare, etc.).
        4. Create drums.mid from onsets.
        5. Cleanup temp stems (optional, or keep them?).
           Note: Project structure expects `drums.flac` as the Demucs 4-stem output.
           DrumSep creates many stems. We probably don't want to keep them all permanently 
           in the main folder unless requested. We'll generate MIDI and cleanup.
    """
    midi_path = folder_path / "drums.mid"
    if midi_path.exists() and not force:
        logger.info(f"Drums MIDI already exists for {folder_path.name}, skipping.")
        return True
        
    # Find drums stem (from Demucs)
    stems = get_stem_files(folder_path, include_full_mix=False)
    drums_stem = stems.get('drums')
    
    if not drums_stem or not drums_stem.exists():
        logger.warning(f"Skipping DrumSep for {folder_path.name}: 'drums' stem not found. DrumSep requires a pre-separated drums track.")
        return False
        
    if not force and drumsep.stems_exist(folder_path):
        logger.info(f"Skipping DrumSep for {folder_path.name}: Output already exists.")
    else:
        logger.info(f"Separating drums for {folder_path.name}...")
        # Use a temp directory for separation to avoid cluttering the organized folder
        sep_dir = folder_path / "separated" / "drumsep"
        
        success = separate_drums(drums_stem, output_dir=sep_dir)
        if not success:
            logger.error(f"DrumSep failed for {folder_path.name}")
            return False
        
    # Find the separated files
    # Structure: sep_dir / model_hash / source_name / {stem}.wav
    # Source name is usually the stem of the input file (e.g., 'drums' or 'full_mix')
    
    # We'll search recursively in sep_dir to find where the wav files are
    stem_files = {}
    found_stems = False
    
    # Debug info
    logger.debug(f"Searching for separated stems in {sep_dir}")
    
    for root, dirs, files in sep_dir.walk():
        # Check if this dir contains relevant stems
        if any(f.endswith('.wav') or f.endswith('.mp3') or f.endswith('.flac') for f in files):
            # Map filename (without ext) to path
            current_stems = {Path(f).stem: Path(root) / f for f in files}
            # Heuristic: Check for common drum stem names
            valid_names = {'kick', 'bombo', 'snare', 'redoblante', 'drums'}
            if any(name in current_stems for name in valid_names): 
                stem_files = current_stems
                found_stems = True
                break
                
    if not found_stems:
        logger.error(f"Could not locate separated stems in {sep_dir}")
        return False
        
    # 3. Detect Onsets
    stem_onsets = {}
    
    logger.info(f"Detecting onsets in {len(stem_files)} stems...")
    
    for stem_name, stem_path in stem_files.items():
        try:
            # Load audio (mono)
            y, sr = librosa.load(str(stem_path), sr=None, mono=True)
            
            # Detect onsets
            # Tweak: DrumSep stems are clean, so standard onset detection usually works well.
            # Might want backtracking=True (default in librosa)
            times, _ = detect_onsets(y, sr)
            
            stem_onsets[stem_name] = times
            logger.debug(f"  {stem_name}: {len(times)} onsets")
            
        except Exception as e:
            logger.error(f"Failed to process stem {stem_name}: {e}")
            
    # 4. Create MIDI
    logger.info("Generating MIDI...")
    try:
        create_midi_from_stems(stem_onsets, midi_path)
    except Exception as e:
        logger.error(f"Failed to create MIDI: {e}")
        return False
        
    # 5. Cleanup?
    # User didn't specify cleanup policy. 
    # Current behavior: Keep files in separated/drumsep.
    # This is safe.
    
    return True

def run_batch_transcription(root_dir: Path, force: bool = False):
    """Run transcription on all folders."""
    folders = find_organized_folders(root_dir)
    logger.info(f"Found {len(folders)} folders to process.")
    
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    
    for folder in folders:
        try:
            if transcribe_drums(folder, force):
                stats['success'] += 1
            else:
                stats['failed'] += 1
        except Exception as e:
            logger.error(f"Error processing {folder.name}: {e}")
            stats['failed'] += 1
            
    logger.info("Transcription Summary:")
    logger.info(f"  Success: {stats['success']}")
    logger.info(f"  Failed:  {stats['failed']}")
    logger.info(f"  Skipped: {stats['skipped']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Root directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing MIDI")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    run_batch_transcription(Path(args.path), args.force)
