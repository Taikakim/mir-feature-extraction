"""
MIDI Utilities for Transcription

This module provides tools to create MIDI files from timestamps and notes.
It uses pretty_midi to generate the files.
"""

import logging
import pretty_midi
from pathlib import Path
from typing import List, Optional, Dict, Union
import numpy as np

logger = logging.getLogger(__name__)

# General MIDI Drum Map (Simplified)
DRUM_MAP = {
    'kick': 36,      # Bass Drum 1
    'snare': 38,     # Acoustic Snare
    'hihat': 42,     # Closed Hi Hat (or 46 Open)
    'tom_low': 41,   # Low Floor Tom
    'tom_mid': 45,   # Low Tom
    'tom_high': 48,  # Hi-Mid Tom
    'clap': 39,      # Hand Clap
    'crash': 49,     # Crash Cymbal 1
    'ride': 51,      # Ride Cymbal 1
    'percussion': 60, # Hi Bongo (Generic for percussion)
    
    # DrumSep / Demucs Custom Model Names (Spanish)
    'bombo': 36,      # Kick
    'redoblante': 38, # Snare
    'platillos': 42,  # Cymbals -> Mapping to Closed HiHat (common denominator)
    'toms': 45        # Toms -> Low Tom
}

def create_drum_track(notes: List[Dict[str, float]], program: int = 0, is_drum: bool = True) -> pretty_midi.Instrument:
    """
    Create a MIDI instrument track.
    
    Args:
        notes: List of dicts with 'start', 'end', 'pitch', 'velocity'
        program: MIDI program number (0 for drums usually implies Standard Kit if is_drum=True)
        is_drum: Whether this is a drum track
        
    Returns:
        pretty_midi.Instrument
    """
    instrument = pretty_midi.Instrument(program=program, is_drum=is_drum)
    
    for note_data in notes:
        note = pretty_midi.Note(
            velocity=int(note_data.get('velocity', 100)),
            pitch=int(note_data['pitch']),
            start=float(note_data['start']),
            end=float(note_data.get('end', note_data['start'] + 0.1)) # Default 100ms duration
        )
        instrument.notes.append(note)
        
    return instrument

def onsets_to_midi(onsets: np.ndarray, 
                   midi_note: int, 
                   velocity: int = 100, 
                   duration: float = 0.1) -> List[Dict[str, float]]:
    """
    Convert onset timestamps to note dictionaries.
    """
    notes = []
    for onset in onsets:
        notes.append({
            'start': onset,
            'end': onset + duration,
            'pitch': midi_note,
            'velocity': velocity
        })
    return notes

def save_midi(instruments: List[pretty_midi.Instrument], output_path: str | Path) -> None:
    """
    Save instruments to a MIDI file.
    """
    midi_data = pretty_midi.PrettyMIDI()
    for inst in instruments:
        midi_data.instruments.append(inst)
        
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        midi_data.write(str(output_path))
        logger.info(f"Saved MIDI to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save MIDI: {e}")
        raise

def create_midi_from_stems(stem_onsets: Dict[str, np.ndarray], output_path: str | Path) -> None:
    """
    Create a single MIDI file from multiple stem onsets (for drums).
    
    Args:
        stem_onsets: Dictionary mapping stem name (e.g., 'kick') to array of onset times
        output_path: Output MIDI file path
    """
    all_notes = []
    
    for stem_name, onsets in stem_onsets.items():
        # Map stem name to MIDI note
        # Try exact match, then substring match
        midi_note = DRUM_MAP.get(stem_name.lower())
        if midi_note is None:
            # Fallback/Heuristics
            if 'kick' in stem_name.lower(): midi_note = 36
            elif 'snare' in stem_name.lower(): midi_note = 38
            elif 'hat' in stem_name.lower(): midi_note = 42
            elif 'tom' in stem_name.lower(): midi_note = 45
            elif 'cymbal' in stem_name.lower() or 'overhead' in stem_name.lower(): midi_note = 49
            else: 
                logger.warning(f"Unknown stem name '{stem_name}', mapping to Percussion (60)")
                midi_note = 60
        
        stem_notes = onsets_to_midi(onsets, midi_note)
        all_notes.extend(stem_notes)
        
    # Sort by start time
    all_notes.sort(key=lambda x: x['start'])
    
    # Create Drum Track (Channel 10 is standard for drums, pretty_midi handles this with is_drum=True)
    drum_track = create_drum_track(all_notes, is_drum=True)
    
    save_midi([drum_track], output_path)
