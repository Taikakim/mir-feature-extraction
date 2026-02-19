"""
Tool to create training crops for Stable Audio Tools training.

Usage:
    # Sequential mode (fixed sample length, no beat alignment)
    python src/tools/create_training_crops.py /path/to/data --length 2097152 --sequential

    # Beat-aligned mode
    python src/tools/create_training_crops.py /path/to/data --length 2097152
    python src/tools/create_training_crops.py /path/to/data --length 2097152 --overlap
    python src/tools/create_training_crops.py /path/to/data --length 2097152 --overlap --div4

    # With output directory (creates per-track folders)
    python src/tools/create_training_crops.py /path/to/data --output-dir /path/to/crops --sequential
    python src/tools/create_training_crops.py /path/to/data -o /path/to/crops --overlap

Features:
- --output-dir / -o: Save crops to destination with per-track folders (e.g., /output/TrackName/)
- --sequential: Simple sequential crops at exact sample length (no beat logic)
- Beat-aligned mode: End times prefer div4 downbeat count, then any downbeat, then full length
- Crops won't shrink below 75% of target length (avoids short crops from sparse beats)
- Without --overlap: strictly consecutive (next crop starts where previous ended)
- When --overlap: next crop starts at (last_start + length/2), snapped to closest downbeat
- 10ms fade-in and fade-out for clean transitions
- First crop starts without zero-crossing snap (preserves exact position)
- Silence detection at -72dB threshold
- Creates .INFO file for each crop with position metadata
"""

import argparse
import logging
import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import find_organized_folders, get_stem_files
from core.common import setup_logging
from rhythm.beat_grid import load_beat_grid
from core.json_handler import get_info_path, read_info, safe_update, batch_write_info
from core.file_locks import FileLock
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

logger = logging.getLogger(__name__)


def preload_stems(folder_path: Path, target_sr: int) -> Dict[str, Tuple[np.ndarray, Path]]:
    """
    Pre-load all stem audio files into RAM for a track folder.

    Returns dict mapping stem name to (audio_array, source_path).
    Audio is resampled to target_sr if needed and shaped as (samples, channels).
    """
    loaded = {}

    for stem_name in STEM_NAMES:
        stem_path = None
        for ext in ['.flac', '.wav', '.mp3', '.m4a']:
            potential = folder_path / f"{stem_name}{ext}"
            if potential.exists():
                stem_path = potential
                break

        if stem_path is None:
            continue

        try:
            stem_audio, stem_sr = sf.read(str(stem_path))

            # Resample if needed
            if stem_sr != target_sr:
                logger.debug(f"Resampling {stem_name}: {stem_sr} -> {target_sr} Hz")
                stem_audio = resample_audio(stem_audio, stem_sr, target_sr)

            # Ensure shape is (samples, channels)
            if stem_audio.ndim == 1:
                stem_audio = stem_audio[:, np.newaxis]
            elif stem_audio.shape[0] < stem_audio.shape[1]:
                stem_audio = stem_audio.T

            loaded[stem_name] = (stem_audio, stem_path)
            logger.debug(f"Pre-loaded {stem_name}: {stem_audio.shape[0]} samples")
        except Exception as e:
            logger.warning(f"Failed to pre-load stem {stem_name}: {e}")

    return loaded

# Check for mutagen (for ID3 tag writing)
try:
    from mutagen.id3 import ID3, TIT2, TPE1, TALB, TYER, TCON, TDRC
    from mutagen.flac import FLAC
    from mutagen.mp3 import MP3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    logger.debug("mutagen not available - ID3 tags will not be written to crops")

# Features to check for in source .INFO and transfer to crop .INFO
TRANSFERRABLE_FEATURES = [
    # ID3 Tag Metadata (extracted first, from audio file tags)
    'track_metadata_artist',
    'track_metadata_title',
    'track_metadata_album',
    'track_metadata_year',
    'track_metadata_genre',

    # Core Metadata (from track_metadata_lookup.py / Spotify / MusicBrainz)
    'release_year',
    'release_date',
    'artists',
    'label',
    'genres',
    'popularity',
    'spotify_id',
    'musicbrainz_id',

    # Music Flamingo descriptions
    'music_flamingo_full',
    'music_flamingo_genre_mood',
    'music_flamingo_instrumentation',
    'music_flamingo_technical',
    'music_flamingo_structure',

    # Optional Spotify audio features
    'spotify_acousticness',
    'spotify_energy',
    'spotify_instrumentalness',
    'spotify_time_signature',
    'spotify_valence',
    'spotify_danceability',
    'spotify_speechiness',
    'spotify_liveness',
    'spotify_key',
    'spotify_mode',
    'spotify_tempo',
]

# Features that are good to have but don't warn if missing
# NOTE: bpm, beat_count, downbeats are NOT transferred - they're calculated fresh per crop
OPTIONAL_TRANSFERRABLE = [
    # Currently empty - crop-specific rhythm features are calculated, not transferred
]

# Demucs stem names to crop along with full_mix
STEM_NAMES = ['drums', 'bass', 'other', 'vocals']

# Supported lossless formats (use soundfile)
LOSSLESS_FORMATS = {'.flac', '.wav', '.aiff', '.ogg'}

# Lossy formats that need pydub/ffmpeg
LOSSY_FORMATS = {'.mp3', '.m4a', '.aac'}

# MP3 writing support
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def get_audio_info(file_path: Path) -> Tuple[int, int]:
    """
    Get audio file metadata (total_samples, sample_rate) without loading the full file.
    Uses sf.info for lossless/mp3, pydub fallback for m4a/aac.

    Returns:
        Tuple of (total_samples, sample_rate)
    """
    try:
        info = sf.info(str(file_path))
        return info.frames, info.samplerate
    except Exception:
        pass

    # Pydub fallback for m4a/aac
    try:
        from pydub.utils import mediainfo
        mi = mediainfo(str(file_path))
        sr = int(mi.get('sample_rate', 44100))
        duration = float(mi.get('duration', 0))
        channels = int(mi.get('channels', 2))
        return int(duration * sr), sr
    except Exception as e:
        raise RuntimeError(f"Cannot read audio info from {file_path}: {e}")


def seeked_read(file_path: Path, start: int, frames: int, sr: int = None) -> Tuple[np.ndarray, int]:
    """
    Read a segment of audio from file without loading the entire file.
    For WAV/FLAC this is a direct seek (no decoding of skipped data).
    For MP3/m4a, ffmpeg still decodes but we only keep the needed segment.

    Args:
        file_path: Path to audio file
        start: Start sample index
        frames: Number of frames to read
        sr: Expected sample rate (for pydub fallback validation)

    Returns:
        Tuple of (audio array (frames, channels), sample_rate)
    """
    try:
        audio, file_sr = sf.read(str(file_path), start=start, frames=frames, dtype='float32')
        return audio, file_sr
    except Exception:
        pass

    # Pydub fallback: load full file, slice (m4a/aac - files are smaller anyway)
    from core.file_utils import read_audio
    audio, file_sr = read_audio(str(file_path), dtype='float32')
    end = min(start + frames, len(audio))
    return audio[start:end], file_sr


def get_audio_channels(file_path: Path) -> int:
    """Get number of audio channels without loading the file."""
    try:
        info = sf.info(str(file_path))
        return info.channels
    except Exception:
        pass
    try:
        from pydub.utils import mediainfo
        mi = mediainfo(str(file_path))
        return int(mi.get('channels', 2))
    except Exception:
        return 2


def detect_silence_offset(file_path: Path, sr: int, threshold_db: float = -72.0,
                          max_scan_seconds: float = 30.0) -> int:
    """
    Find the sample index where audio first rises above threshold_db.
    Only reads the first max_scan_seconds from disk (seeked read).

    Returns:
        Sample index where audio exceeds threshold, or 0 if no silence.
    """
    threshold_linear = 10 ** (threshold_db / 20)
    window_size = 4410  # ~100ms at 44.1kHz
    max_scan_samples = int(max_scan_seconds * sr)

    try:
        audio, _ = seeked_read(file_path, start=0, frames=max_scan_samples)
    except Exception:
        return 0

    # Mix to mono for detection
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio.flatten()

    for i in range(0, len(mono), window_size):
        chunk = mono[i:i + window_size]
        if np.max(np.abs(chunk)) > threshold_linear:
            return i

    return 0


def seeked_zero_crossing(file_path: Path, target_sample: int, search_window: int,
                         sr: int, preloaded_audio: np.ndarray = None) -> int:
    """
    Find zero crossing near target_sample by reading only a small window from disk.
    If preloaded_audio is provided, slices from RAM instead.

    Returns:
        Zero crossing sample index, or target_sample if none found.
    """
    start = max(0, target_sample - search_window)
    frames = search_window

    if preloaded_audio is not None:
        end = min(start + frames, preloaded_audio.shape[0])
        audio = preloaded_audio[start:end]
    else:
        try:
            audio, _ = seeked_read(file_path, start=start, frames=frames)
        except Exception:
            return target_sample

    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio.flatten()

    if len(mono) < 2:
        return target_sample

    zero_crossings = np.where(np.diff(np.signbit(mono)))[0]
    if len(zero_crossings) == 0:
        return target_sample

    return start + zero_crossings[-1]


def seeked_zero_crossing_end(file_path: Path, search_start: int, search_end: int,
                             sr: int, preloaded_audio: np.ndarray = None) -> int:
    """
    Find last zero crossing in [search_start, search_end] by reading only that window.
    If preloaded_audio is provided, slices from RAM instead.
    """
    if search_start >= search_end:
        return search_end

    frames = search_end - search_start
    if preloaded_audio is not None:
        end = min(search_start + frames, preloaded_audio.shape[0])
        audio = preloaded_audio[search_start:end]
    else:
        try:
            audio, _ = seeked_read(file_path, start=search_start, frames=frames)
        except Exception:
            return search_end

    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio.flatten()

    if len(mono) < 2:
        return search_end

    zcs = np.where(np.diff(np.signbit(mono)))[0]
    if len(zcs) > 0:
        return search_start + zcs[-1]
    return search_end


def get_audio_bitrate(file_path: Path) -> int:
    """
    Detect audio bitrate using mutagen or default to 128kbps.
    Supports MP3, M4A, and AAC formats.
    """
    ext = file_path.suffix.lower()
    try:
        if ext == '.mp3':
            from mutagen.mp3 import MP3
            audio = MP3(str(file_path))
            return audio.info.bitrate // 1000
        elif ext in ('.m4a', '.aac'):
            from mutagen.mp4 import MP4
            audio = MP4(str(file_path))
            return audio.info.bitrate // 1000
    except Exception:
        pass
    return 128  # Default fallback


def get_mp3_bitrate(file_path: Path) -> int:
    """Legacy wrapper for get_audio_bitrate."""
    return get_audio_bitrate(file_path)


def write_id3_tags(file_path: Path, metadata: Dict) -> bool:
    """
    Write ID3/metadata tags to an audio file.

    Supports MP3 (ID3v2) and FLAC (Vorbis comments).

    Args:
        file_path: Path to audio file
        metadata: Dict with keys like 'track_metadata_artist', 'track_metadata_title', etc.

    Returns:
        True if tags written successfully, False otherwise
    """
    if not MUTAGEN_AVAILABLE:
        return False

    if not metadata:
        return False

    ext = file_path.suffix.lower()

    try:
        if ext == '.mp3':
            # Create ID3 tags for MP3
            try:
                audio = MP3(str(file_path))
                if audio.tags is None:
                    audio.add_tags()
            except Exception:
                audio = MP3(str(file_path))
                audio.add_tags()

            tags = audio.tags

            # Map our metadata keys to ID3 frames
            if metadata.get('track_metadata_title'):
                tags.add(TIT2(encoding=3, text=metadata['track_metadata_title']))
            if metadata.get('track_metadata_artist'):
                tags.add(TPE1(encoding=3, text=metadata['track_metadata_artist']))
            if metadata.get('track_metadata_album'):
                tags.add(TALB(encoding=3, text=metadata['track_metadata_album']))
            if metadata.get('track_metadata_year'):
                year = str(metadata['track_metadata_year'])
                tags.add(TYER(encoding=3, text=year))
                tags.add(TDRC(encoding=3, text=year))
            if metadata.get('track_metadata_genre'):
                tags.add(TCON(encoding=3, text=metadata['track_metadata_genre']))

            audio.save()
            return True

        elif ext == '.flac':
            # Write Vorbis comments for FLAC
            audio = FLAC(str(file_path))

            if metadata.get('track_metadata_title'):
                audio['TITLE'] = metadata['track_metadata_title']
            if metadata.get('track_metadata_artist'):
                audio['ARTIST'] = metadata['track_metadata_artist']
            if metadata.get('track_metadata_album'):
                audio['ALBUM'] = metadata['track_metadata_album']
            if metadata.get('track_metadata_year'):
                audio['DATE'] = str(metadata['track_metadata_year'])
            if metadata.get('track_metadata_genre'):
                audio['GENRE'] = metadata['track_metadata_genre']

            audio.save()
            return True

    except Exception as e:
        logger.debug(f"Failed to write tags to {file_path.name}: {e}")
        return False

    return False


def write_audio_preserving_format(
    audio: np.ndarray,
    sr: int,
    output_path: Path,
    source_path: Optional[Path] = None,
    mp3_bitrate: Optional[int] = None,
    metadata: Optional[Dict] = None
) -> bool:
    """
    Write audio to output_path, preserving the format indicated by the file extension.

    For MP3/M4A output, uses pydub if available, otherwise falls back to FLAC.
    Bitrate is detected from source_path if provided, or uses mp3_bitrate parameter.
    If metadata is provided, writes ID3 tags (MP3) or Vorbis comments (FLAC).

    Args:
        audio: Audio array of shape (samples,) or (samples, channels)
        sr: Sample rate
        output_path: Output file path (extension determines format)
        source_path: Optional source file for bitrate detection
        mp3_bitrate: Optional explicit bitrate (kbps)
        metadata: Optional dict with track_metadata_* keys for ID3 tags

    Returns:
        True if written in requested format, False if fell back to different format
    """
    ext = output_path.suffix.lower()

    # Ensure proper shape for soundfile
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    # Lossless formats - use soundfile directly
    if ext in LOSSLESS_FORMATS:
        sf.write(str(output_path), audio, sr)
        # Write metadata tags if provided
        if metadata:
            write_id3_tags(output_path, metadata)
        return True
    
    # Lossy formats (MP3, M4A, AAC) - use pydub
    if ext in LOSSY_FORMATS:
        if not PYDUB_AVAILABLE:
            # Fall back to FLAC
            fallback_path = output_path.with_suffix('.flac')
            sf.write(str(fallback_path), audio, sr)
            logger.warning(f"pydub not available, wrote FLAC instead: {fallback_path.name}")
            return False
        
        try:
            # Determine bitrate
            bitrate = mp3_bitrate
            if bitrate is None and source_path is not None:
                bitrate = get_audio_bitrate(source_path)
            if bitrate is None:
                bitrate = 128
            
            # Convert numpy array to pydub AudioSegment
            # pydub expects int16 samples
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Flatten if stereo for raw data
            if audio_int16.shape[1] == 1:
                channels = 1
                raw_data = audio_int16.flatten().tobytes()
            else:
                channels = audio_int16.shape[1]
                raw_data = audio_int16.tobytes()
            
            segment = AudioSegment(
                data=raw_data,
                sample_width=2,  # 16-bit = 2 bytes
                frame_rate=sr,
                channels=channels
            )
            
            # Determine output format for pydub
            if ext == '.mp3':
                pydub_format = 'mp3'
            elif ext in ('.m4a', '.aac'):
                pydub_format = 'ipod'  # pydub uses 'ipod' for M4A/AAC
            else:
                pydub_format = 'mp3'
            
            segment.export(str(output_path), format=pydub_format, bitrate=f'{bitrate}k')
            # Write metadata tags if provided
            if metadata:
                write_id3_tags(output_path, metadata)
            return True

        except Exception as e:
            # Fall back to FLAC
            fallback_path = output_path.with_suffix('.flac')
            sf.write(str(fallback_path), audio, sr)
            # Write metadata tags if provided
            if metadata:
                write_id3_tags(fallback_path, metadata)
            logger.warning(f"Lossy export failed ({e}), wrote FLAC instead: {fallback_path.name}")
            return False
    
    # Unknown format - default to FLAC
    fallback_path = output_path.with_suffix('.flac')
    sf.write(str(fallback_path), audio, sr)
    logger.warning(f"Unknown format {ext}, wrote FLAC instead: {fallback_path.name}")
    return False


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate using scipy.

    Args:
        audio: Audio array of shape (samples,) or (samples, channels)
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    # Calculate resampling ratio
    ratio = target_sr / orig_sr
    new_length = int(audio.shape[0] * ratio)

    if audio.ndim == 1:
        return signal.resample(audio, new_length)
    else:
        # Resample each channel
        resampled = np.zeros((new_length, audio.shape[1]), dtype=audio.dtype)
        for ch in range(audio.shape[1]):
            resampled[:, ch] = signal.resample(audio[:, ch], new_length)
        return resampled


def crop_stem_file(stem_path: Path, crop_base_path: Path, stem_name: str,
                   start_sample: int, end_sample: int, sr: int,
                   fade_len: int, metadata: Optional[Dict] = None,
                   preloaded_audio: Optional[np.ndarray] = None) -> Optional[Path]:
    """
    Crop a stem file at the same sample positions as the full mix crop.
    Preserves the source stem's format (MP3→MP3, FLAC→FLAC, etc.).

    Args:
        stem_path: Path to source stem (e.g., drums.mp3 or drums.flac)
        crop_base_path: Base path for output (e.g., TrackName_0.flac)
        stem_name: Name of stem (drums, bass, other, vocals)
        start_sample: Start sample index
        end_sample: End sample index
        sr: Sample rate (target)
        fade_len: Fade length in samples
        metadata: Optional dict with track_metadata_* keys for ID3 tags
        preloaded_audio: Pre-loaded audio array (samples, channels), already resampled

    Returns:
        Path to cropped stem, or None if failed
    """
    try:
        if preloaded_audio is not None:
            # Pre-loaded: slice from RAM, clamp to available length
            clamped_end = min(end_sample, preloaded_audio.shape[0])
            if start_sample >= clamped_end:
                return None
            stem_crop = preloaded_audio[start_sample:clamped_end].copy()
        else:
            # Seeked read: only load the crop segment from disk
            if not stem_path.exists():
                return None

            n_frames = end_sample - start_sample
            stem_crop, stem_sr = seeked_read(stem_path, start=start_sample, frames=n_frames)

            if stem_sr != sr:
                stem_crop = resample_audio(stem_crop, stem_sr, sr)

            if stem_crop.ndim == 1:
                stem_crop = stem_crop[:, np.newaxis]
            elif stem_crop.shape[0] < stem_crop.shape[1]:
                stem_crop = stem_crop.T

        # Apply fades
        stem_crop = apply_fades(stem_crop, fade_len)

        # Preserve source stem format (m4a/aac → mp3 to avoid downstream issues)
        source_ext = stem_path.suffix.lower()
        if source_ext in ('.m4a', '.aac'):
            source_ext = '.mp3'
        crop_stem_name = f"{crop_base_path.stem}_{stem_name}{source_ext}"
        crop_stem_path = crop_base_path.parent / crop_stem_name

        # Write using format-aware function (with ID3 metadata if provided)
        write_audio_preserving_format(stem_crop, sr, crop_stem_path, source_path=stem_path,
                                      metadata=metadata)

        return crop_stem_path

    except Exception as e:
        logger.warning(f"Failed to crop stem {stem_name}: {e}")
        return None


def crop_all_stems(folder_path: Path, crop_base_path: Path,
                   start_sample: int, end_sample: int, sr: int,
                   fade_len: int, metadata: Optional[Dict] = None,
                   preloaded_stems: Optional[Dict[str, Tuple[np.ndarray, Path]]] = None) -> Dict[str, Path]:
    """
    Crop all available stems at the same positions as the full mix.

    Args:
        metadata: Optional dict with track_metadata_* keys for ID3 tags
        preloaded_stems: Pre-loaded stems from preload_stems(), maps stem_name to (audio, path)

    Returns:
        Dict mapping stem names to cropped paths
    """
    cropped_stems = {}

    for stem_name in STEM_NAMES:
        if preloaded_stems and stem_name in preloaded_stems:
            stem_audio, stem_path = preloaded_stems[stem_name]
        else:
            # Fallback: find stem file on disk
            stem_path = None
            for ext in ['.flac', '.wav', '.mp3', '.m4a']:
                potential = folder_path / f"{stem_name}{ext}"
                if potential.exists():
                    stem_path = potential
                    break
            if stem_path is None:
                continue
            stem_audio = None

        cropped = crop_stem_file(
            stem_path, crop_base_path, stem_name,
            start_sample, end_sample, sr, fade_len,
            metadata=metadata,
            preloaded_audio=stem_audio
        )
        if cropped:
            cropped_stems[stem_name] = cropped

    return cropped_stems


def slice_rhythm_file(source_path: Path, dest_path: Path,
                      start_time: float, end_time: float):
    """
    Read timestamps from source_path, filter those within [start_time, end_time],
    shift them by -start_time, and write to dest_path.
    """
    if not source_path.exists():
        return
        
    try:
        # Read timestamps (handle both single-column and multi-column if needed)
        with open(source_path, 'r') as f:
            lines = f.readlines()
            
        valid_lines = []
        for line in lines:
            if not line.strip(): continue
            try:
                # Assume first token is timestamp
                parts = line.strip().split()
                if not parts: continue
                t = float(parts[0])
                
                if start_time <= t <= end_time:
                    # Shift timestamp
                    new_t = t - start_time
                    # Reconstruct line with new timestamp
                    if len(parts) > 1:
                        new_line = f"{new_t:.6f} {' '.join(parts[1:])}"
                    else:
                        new_line = f"{new_t:.6f}"
                    valid_lines.append(new_line)
            except ValueError:
                continue
                
        # Write only if we have data (or write empty file? Empty is fine)
        with open(dest_path, 'w') as f:
            f.write('\n'.join(valid_lines))
            if valid_lines: f.write('\n')
            
    except Exception as e:
        logger.warning(f"Failed to slice rhythm file {source_path.name}: {e}")


def get_start_offset_above_threshold(audio: np.ndarray,
                                     threshold_db: float = -72.0,
                                     window_size_samples: int = 4410) -> int:
    """
    Find the sample index where audio first rises above threshold_db.
    Uses -72dB as default threshold for detecting silence.
    """
    threshold_linear = 10 ** (threshold_db / 20)

    num_samples = audio.shape[0]

    # If stereo, mix to mono for detection
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio.flatten()

    for i in range(0, num_samples, window_size_samples):
        chunk = mono[i: i + window_size_samples]
        if np.max(np.abs(chunk)) > threshold_linear:
            return i

    return 0


def find_downbeat_before(downbeat_times: np.ndarray, target_time: float) -> Optional[float]:
    """Find the latest downbeat that is <= target_time (snap backwards)."""
    if downbeat_times is None or len(downbeat_times) == 0:
        return None

    candidates = downbeat_times[downbeat_times <= target_time + 0.001]
    if len(candidates) == 0:
        return None

    return candidates[-1]


def find_closest_downbeat(downbeat_times: np.ndarray, target_time: float) -> Optional[float]:
    """Find the closest downbeat to target_time (for start alignment)."""
    if downbeat_times is None or len(downbeat_times) == 0:
        return None

    idx = (np.abs(downbeat_times - target_time)).argmin()
    return downbeat_times[idx]


def find_zero_crossing_backwards(audio: np.ndarray, target_idx: int, search_window: int = 2205) -> int:
    """
    Find the first zero crossing BACKWARDS from target_idx.
    Returns the zero crossing index, or target_idx if none found.
    """
    start = max(0, target_idx - search_window)
    end = target_idx

    if start >= end:
        return target_idx

    window = audio[start:end]
    zero_crossings = np.where(np.diff(np.signbit(window)))[0]

    if len(zero_crossings) == 0:
        return target_idx

    zero_crossings = zero_crossings + start
    return zero_crossings[-1]


def count_downbeats_in_range(downbeat_times: np.ndarray, start_sec: float, end_sec: float) -> int:
    """Count how many downbeats fall within [start_sec, end_sec]."""
    if downbeat_times is None or len(downbeat_times) == 0:
        return 0
    mask = (downbeat_times >= start_sec - 0.001) & (downbeat_times <= end_sec + 0.001)
    return int(np.sum(mask))


def find_end_for_div4_downbeats(downbeat_times: np.ndarray, start_sec: float,
                                 target_end_sec: float, min_downbeats: int = 4) -> Optional[float]:
    """
    Find an end time such that the number of downbeats in [start, end] is divisible by 4.
    Searches backwards from target_end_sec.
    """
    if downbeat_times is None or len(downbeat_times) == 0:
        return None

    candidates = downbeat_times[downbeat_times > start_sec + 0.001]
    candidates = candidates[candidates <= target_end_sec + 0.001]

    if len(candidates) < min_downbeats:
        return None

    for i in range(len(candidates) - 1, -1, -1):
        end_time = candidates[i]
        count = count_downbeats_in_range(downbeat_times, start_sec, end_time)
        if count >= min_downbeats and count % 4 == 0:
            return end_time

    return None


def apply_fades(crop_audio: np.ndarray, fade_len: int) -> np.ndarray:
    """Apply 10ms fade-in and fade-out to crop."""
    if crop_audio.shape[0] <= fade_len * 2:
        return crop_audio

    # Fade in
    fade_in_curve = np.linspace(0, 1, fade_len)
    if crop_audio.ndim > 1:
        crop_audio[:fade_len, :] *= fade_in_curve[:, np.newaxis]
    else:
        crop_audio[:fade_len] *= fade_in_curve

    # Fade out
    fade_out_curve = np.linspace(1, 0, fade_len)
    if crop_audio.ndim > 1:
        crop_audio[-fade_len:, :] *= fade_out_curve[:, np.newaxis]
    else:
        crop_audio[-fade_len:] *= fade_out_curve

    return crop_audio


def create_sequential_crops(folder_path: Path, length_samples: int, sr: int,
                            audio: np.ndarray, full_mix_path: Path,
                            output_dir: Optional[Path] = None,
                            preloaded_stems: Optional[Dict[str, Tuple[np.ndarray, Path]]] = None) -> int:
    """
    Create simple sequential crops at fixed sample length.
    No beat alignment, just exact sample boundaries with fades.
    Preserves input file format (FLAC→FLAC, MP3→MP3, etc.).
    """
    total_samples = audio.shape[0]
    duration_sec = total_samples / sr
    fade_len = int(0.01 * sr)  # 10ms fade

    # Preserve source format (m4a/aac → mp3 to avoid downstream issues)
    source_ext = full_mix_path.suffix.lower()
    if source_ext in ('.m4a', '.aac'):
        source_ext = '.mp3'

    # Load source .INFO for ID3 metadata (to write to crop files)
    source_info = {}
    source_info_path = get_info_path(full_mix_path)
    if source_info_path.exists():
        try:
            source_info = read_info(source_info_path)
        except Exception as e:
            logger.debug(f"Failed to read source info for ID3 tags: {e}")

    # Extract ID3 metadata for writing to crop audio files
    id3_metadata = {
        k: source_info.get(k) for k in [
            'track_metadata_artist', 'track_metadata_title',
            'track_metadata_album', 'track_metadata_year', 'track_metadata_genre'
        ] if source_info.get(k)
    }

    # Find start after silence (-72dB threshold)
    start_sample = get_start_offset_above_threshold(audio, threshold_db=-72.0)
    if start_sample > 0:
        logger.info(f"Skipping initial silence: {start_sample / sr:.2f}s ({start_sample} samples)")

    # Determine output directory
    if output_dir:
        # Create per-track folder in output directory
        track_name = folder_path.name
        crops_dir = output_dir / track_name
    else:
        crops_dir = folder_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load stems if not already provided
    if preloaded_stems is None:
        preloaded_stems = preload_stems(folder_path, sr)
    stem_names_loaded = list(preloaded_stems.keys())
    if stem_names_loaded:
        logger.debug(f"Stems in RAM: {stem_names_loaded}")

    crop_count = 0
    current_sample = start_sample

    # Async write pool for I/O-bound file saves
    write_futures: List[Future] = []
    write_pool = ThreadPoolExecutor(max_workers=4)

    try:
        while current_sample + length_samples <= total_samples:
            end_sample = current_sample + length_samples

            # Extract crop (no ZC snap for sequential mode)
            crop_audio = audio[current_sample:end_sample].copy()

            # Apply fades
            crop_audio = apply_fades(crop_audio, fade_len)

            # Build crop path
            track_name = full_mix_path.parent.name
            crop_name = f"{track_name}_{crop_count}{source_ext}"
            crop_path = crops_dir / crop_name

            # Metadata
            start_sec = current_sample / sr
            end_sec = end_sample / sr
            position = start_sec / duration_sec

            # Schedule async writes: full mix crop + stem crops
            def _write_crop(audio_data, path, src_path, meta, stem_data, folder, base_path,
                            s_start, s_end, s_sr, s_fade, s_meta, s_preloaded):
                write_audio_preserving_format(audio_data, s_sr, path, source_path=src_path,
                                              metadata=meta)
                crop_all_stems(folder, base_path, s_start, s_end, s_sr, s_fade,
                               metadata=s_meta, preloaded_stems=s_preloaded)

            write_futures.append(write_pool.submit(
                _write_crop, crop_audio, crop_path, full_mix_path, id3_metadata,
                preloaded_stems, folder_path, crop_path,
                current_sample, end_sample, sr, fade_len, id3_metadata, preloaded_stems
            ))

            # Slice rhythm files (beats, downbeats)
            rhythm_suffixes = ['.BEATS_GRID', '.DOWNBEATS']
            for suffix in rhythm_suffixes:
                candidates = list(folder_path.glob(f"*{suffix}"))
                if candidates:
                    source_file = candidates[0]
                    dest_file = crop_path.with_suffix(suffix)
                    slice_rhythm_file(source_file, dest_file, start_sec, end_sec)

            # Write all metadata to .INFO (no separate .json)
            info_path = get_info_path(crop_path)
            safe_update(info_path, {
                "position": position,
                "start_time": start_sec,
                "end_time": end_sec,
                "start_sample": current_sample,
                "end_sample": end_sample,
                "duration": end_sec - start_sec,
                "samples": length_samples,
                "source": str(full_mix_path.name),
                "has_stems": len(preloaded_stems) > 0,
                "stem_names": stem_names_loaded,
            })

            crop_count += 1
            current_sample = end_sample  # Sequential: next starts where this ended

        # Wait for all async writes to complete
        for future in write_futures:
            try:
                future.result()
            except Exception as e:
                logger.warning(f"Async write failed: {e}")

    finally:
        write_pool.shutdown(wait=True)

    return crop_count


def create_crops_for_file(folder_path: Path,
                          length_samples: int,
                          overlap: bool = True,
                          div4: bool = False,
                          sequential: bool = False,
                          output_dir: Optional[Path] = None,
                          overwrite: bool = False) -> int:
    """
    Generate crops for a single folder.

    Args:
        folder_path: Path to organized folder
        length_samples: Target crop length in samples
        overlap: If True, next crop starts at last_start + length/2
        div4: If True, ensure each crop contains downbeats divisible by 4
        sequential: If True, use simple sequential mode (no beat alignment)
        output_dir: Optional output directory for crops
        overwrite: If False, skip folders that already have crops
    """
    stems = get_stem_files(folder_path, include_full_mix=True)
    if 'full_mix' not in stems:
        logger.warning(f"No full_mix in {folder_path.name}")
        return 0

    # Determine crops output directory
    if output_dir:
        track_name = folder_path.name
        crops_dir = output_dir / track_name
    else:
        crops_dir = folder_path / "crops"

    # Check if crops already exist (skip if not overwriting)
    if not overwrite and crops_dir.exists():
        # Check for crops in any format
        existing_crops = []
        for ext in ['.flac', '.mp3', '.wav', '.m4a']:
            existing_crops.extend(crops_dir.glob(f"*_[0-9]{ext}"))
            existing_crops.extend(crops_dir.glob(f"*_[0-9][0-9]{ext}"))
        if existing_crops:
            logger.info(f"Crops already exist ({len(existing_crops)} files): {folder_path.name}. Use --overwrite to regenerate.")
            return 0

    full_mix_path = stems['full_mix']

    # Get audio metadata without loading the full file
    try:
        total_samples, sr = get_audio_info(full_mix_path)
    except Exception as e:
        logger.error(f"Failed to read info for {full_mix_path}: {e}")
        return 0

    duration_sec = total_samples / sr
    length_sec = length_samples / sr

    # Preserve source format (m4a/aac → mp3 to avoid downstream issues)
    source_ext = full_mix_path.suffix.lower()
    if source_ext in ('.m4a', '.aac'):
        source_ext = '.mp3'

    logger.info(f"{folder_path.name}: {total_samples} samples, {duration_sec:.2f}s, sr={sr}")

    # Resolve stem paths
    stem_paths = {}
    for stem_name in STEM_NAMES:
        for ext in ['.flac', '.wav', '.mp3', '.m4a']:
            potential = folder_path / f"{stem_name}{ext}"
            if potential.exists():
                stem_paths[stem_name] = potential
                break

    # Pre-load lossy formats into RAM (MP3/m4a/ogg can't seek efficiently)
    preloaded_mix = None
    preloaded_stems = None
    lossy_source = source_ext in LOSSY_FORMATS | {'.ogg'}

    if lossy_source:
        logger.info(f"Lossy source ({source_ext}): pre-loading full mix + stems into RAM")
        from core.file_utils import read_audio
        mix_audio, mix_sr = read_audio(str(full_mix_path), dtype='float32')
        if mix_sr != sr:
            mix_audio = resample_audio(mix_audio, mix_sr, sr)
        if mix_audio.ndim == 1:
            mix_audio = mix_audio[:, np.newaxis]
        elif mix_audio.shape[0] < mix_audio.shape[1]:
            mix_audio = mix_audio.T
        preloaded_mix = mix_audio
        preloaded_stems = preload_stems(folder_path, sr)
        logger.info(f"Pre-loaded mix: {preloaded_mix.shape[0]} samples, {len(preloaded_stems)} stems")
    else:
        if stem_paths:
            logger.info(f"Found {len(stem_paths)} stems (lossless: seek-read per crop)")

    # Sequential mode: needs full audio loaded (simpler, no beat alignment)
    if sequential:
        audio, sr = sf.read(str(full_mix_path))
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        elif audio.shape[0] < audio.shape[1]:
            audio = audio.T
        preloaded_stems = preload_stems(folder_path, sr)
        return create_sequential_crops(folder_path, length_samples, sr, audio, full_mix_path,
                                       output_dir, preloaded_stems=preloaded_stems)

    # Beat-aligned mode - load beat grid (BPM not needed when grids are available)
    beat_times = None
    downbeat_times = None

    grid_path = folder_path / f"{folder_path.name}.BEATS_GRID"
    if grid_path.exists():
        beat_times = load_beat_grid(grid_path)

        downbeat_path = folder_path / f"{folder_path.name}.DOWNBEATS"
        if downbeat_path.exists():
            try:
                downbeat_times = np.loadtxt(downbeat_path)
                if downbeat_times.ndim == 0:
                    downbeat_times = np.array([downbeat_times])
                logger.info(f"{folder_path.name}: Loaded {len(downbeat_times)} downbeats.")
            except Exception as e:
                logger.warning(f"Failed to load downbeats: {e}")

        if downbeat_times is None or len(downbeat_times) < 2:
            if len(beat_times) >= 4:
                downbeat_times = beat_times[::4]
                logger.info(f"{folder_path.name}: Inferred {len(downbeat_times)} downbeats.")
            else:
                downbeat_times = beat_times
    else:
        logger.warning(f"No beat grid for {folder_path.name}, using sequential fallback.")
        # Sequential fallback needs full audio loaded
        audio, sr = sf.read(str(full_mix_path))
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        elif audio.shape[0] < audio.shape[1]:
            audio = audio.T
        fallback_stems = preload_stems(folder_path, sr)
        return create_sequential_crops(folder_path, length_samples, sr, audio, full_mix_path,
                                       output_dir, preloaded_stems=fallback_stems)

    # Get BPM from INFO file or calculate from beat times
    info_path = get_info_path(full_mix_path)
    info_data = read_info(info_path)
    bpm = info_data.get('bpm', 0)

    # Extract ID3 metadata for writing to crop audio files
    id3_metadata = {
        k: info_data.get(k) for k in [
            'track_metadata_artist', 'track_metadata_title',
            'track_metadata_album', 'track_metadata_year', 'track_metadata_genre'
        ] if info_data.get(k)
    }

    # If BPM not in INFO, calculate from beat times
    if not bpm and beat_times is not None and len(beat_times) > 1:
        ibis = np.diff(beat_times)
        mean_ibi = np.median(ibis)  # Use median to avoid outliers
        if mean_ibi > 0:
            bpm = 60.0 / mean_ibi

    # Find start after silence (-72dB threshold) — seeked read, not full load
    start_offset_samples = detect_silence_offset(full_mix_path, sr, threshold_db=-72.0)
    start_offset_sec = start_offset_samples / sr
    if start_offset_samples > 0:
        logger.info(f"Skipping initial silence: {start_offset_sec:.2f}s")

    # Determine output directory
    if output_dir:
        # Create per-track folder in output directory
        track_name = folder_path.name
        crops_dir = output_dir / track_name
    else:
        crops_dir = folder_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    crop_count = 0
    align_grid = downbeat_times if (downbeat_times is not None and len(downbeat_times) > 0) else beat_times
    global_bpm = float(bpm) if bpm else 0.0

    # Pre-load transferrable features from source .INFO (reuse info_data loaded above)
    # This avoids re-reading the same file for every crop
    source_transferrable = {}
    for feature in TRANSFERRABLE_FEATURES:
        if feature in info_data:
            source_transferrable[feature] = info_data[feature]
    for feature in OPTIONAL_TRANSFERRABLE:
        if feature in info_data:
            source_transferrable[feature] = info_data[feature]

    # Collect .INFO data for batch write at end (reduces HDD seeks)
    pending_info_writes = []

    # Initial start: snap to closest downbeat after silence
    aligned_start = find_closest_downbeat(align_grid, start_offset_sec)
    if aligned_start is not None:
        current_start_sec = aligned_start
    else:
        current_start_sec = start_offset_sec

    zc_window = int(0.05 * sr)  # 50ms
    fade_len = int(0.01 * sr)   # 10ms
    is_first_crop = True

    # Async write pool for I/O-bound file saves (crop audio + stem audio)
    write_futures: List[Future] = []
    write_pool = ThreadPoolExecutor(max_workers=4)
    stem_names_loaded = list(preloaded_stems.keys()) if preloaded_stems else list(stem_paths.keys())

    def _write_crop_and_stems(audio_data, path, src_path, meta, base_path,
                              s_start, s_end, s_sr, s_fade, s_meta,
                              s_stem_paths, s_preloaded_stems):
        """Write crop audio + all stem crops (runs in thread pool)."""
        write_audio_preserving_format(audio_data, s_sr, path, source_path=src_path,
                                      metadata=meta)
        if s_preloaded_stems:
            # Slice stems from RAM
            for s_name, (s_audio, s_path) in s_preloaded_stems.items():
                crop_stem_file(s_path, base_path, s_name, s_start, s_end, s_sr, s_fade,
                               metadata=s_meta, preloaded_audio=s_audio)
        else:
            # Seek-read each stem crop from disk (lossless formats)
            for s_name, s_path in s_stem_paths.items():
                crop_stem_file(s_path, base_path, s_name, s_start, s_end, s_sr, s_fade,
                               metadata=s_meta, preloaded_audio=None)

    while current_start_sec + 1.0 < duration_sec:
        target_start_sample = int(current_start_sec * sr)
        target_end_sec = current_start_sec + length_sec

        if target_end_sec > duration_sec:
            target_end_sec = duration_sec

        # Snap end BACKWARDS to downbeat
        final_end_sec = target_end_sec
        # Minimum crop length: 75% of target to avoid short crops from sparse beats
        min_end_sec = current_start_sec + (length_sec * 0.75)

        if div4:
            div4_end = find_end_for_div4_downbeats(align_grid, current_start_sec, target_end_sec)
            if div4_end is not None and div4_end > min_end_sec:
                final_end_sec = div4_end
            else:
                # div4 would make crop too short — fall back to any downbeat
                snapped_end = find_downbeat_before(align_grid, target_end_sec)
                if snapped_end is not None and snapped_end > min_end_sec:
                    final_end_sec = snapped_end
                # else: keep target_end_sec (full length, no beat snap)
        else:
            # Try div4 first (best musical boundary), then any downbeat
            div4_end = find_end_for_div4_downbeats(align_grid, current_start_sec, target_end_sec)
            if div4_end is not None and div4_end > min_end_sec:
                final_end_sec = div4_end
            else:
                snapped_end = find_downbeat_before(align_grid, target_end_sec)
                if snapped_end is not None and snapped_end > min_end_sec:
                    final_end_sec = snapped_end
                # else: keep target_end_sec (full length, no beat snap)

        final_end_sample = int(final_end_sec * sr)

        # Zero-crossing snapping (from RAM if preloaded, else seeked reads)
        if is_first_crop:
            # First crop: NO zero-crossing snap for start
            actual_start_sample = target_start_sample
            is_first_crop = False
        else:
            # Subsequent crops: snap start to ZC backwards
            actual_start_sample = seeked_zero_crossing(full_mix_path, target_start_sample,
                                                       zc_window, sr,
                                                       preloaded_audio=preloaded_mix)

        # End: snap to ZC backwards
        search_start = max(actual_start_sample + int(0.1 * sr), final_end_sample - zc_window)
        actual_end_sample = seeked_zero_crossing_end(full_mix_path, search_start,
                                                      final_end_sample, sr,
                                                      preloaded_audio=preloaded_mix)

        n_frames = actual_end_sample - actual_start_sample

        # Skip if too short (< 1 second)
        if n_frames < sr:
            # Advance past this region
            current_start_sec = max(current_start_sec + 1.0, actual_end_sample / sr)
            continue

        # Read crop segment (from RAM if preloaded, else seeked read from disk)
        if preloaded_mix is not None:
            crop_audio = preloaded_mix[actual_start_sample:actual_end_sample].copy()
        else:
            try:
                crop_audio, _ = seeked_read(full_mix_path, actual_start_sample, n_frames)
            except Exception as e:
                logger.warning(f"Failed to read crop at {actual_start_sample}: {e}")
                current_start_sec = current_start_sec + 1.0
                continue

        # Ensure shape is (samples, channels)
        if crop_audio.ndim == 1:
            crop_audio = crop_audio[:, np.newaxis]

        # Apply fades
        crop_audio = apply_fades(crop_audio, fade_len)

        # Save with same format as source (with ID3 metadata)
        track_name = full_mix_path.parent.name
        crop_name = f"{track_name}_{crop_count}{source_ext}"
        crop_path = crops_dir / crop_name

        # Schedule async writes: full mix crop + stem crops
        write_futures.append(write_pool.submit(
            _write_crop_and_stems, crop_audio, crop_path, full_mix_path, id3_metadata,
            crop_path, actual_start_sample, actual_end_sample, sr, fade_len, id3_metadata,
            stem_paths, preloaded_stems
        ))

        # Metadata
        actual_start_sec = float(actual_start_sample / sr)
        actual_end_sec = float(actual_end_sample / sr)

        # Slice Rhythm Files (NOT onsets - unused by crop analysis, would need re-detection)
        rhythm_suffixes = ['.BEATS_GRID', '.DOWNBEATS']
        folder = full_mix_path.parent
        
        for suffix in rhythm_suffixes:
            # Try to find source file
            candidates = list(folder.glob(f"*{suffix}"))
            if candidates:
                source_file = candidates[0]
                dest_file = crop_path.with_suffix(suffix)
                slice_rhythm_file(source_file, dest_file, actual_start_sec, actual_end_sec)

        num_downbeats = count_downbeats_in_range(align_grid, current_start_sec, final_end_sec)
        
        # Calculate local rhythmic features
        # Filter beats within this crop
        crop_beats = [b for b in beat_times if actual_start_sec <= b <= actual_end_sec]
        beat_count = len(crop_beats)
        
        # Estimate local BPM
        local_bpm = global_bpm
        if len(crop_beats) > 1:
            ibis = np.diff(crop_beats)
            mean_ibi = np.mean(ibis)
            if mean_ibi > 0:
                local_bpm = 60.0 / mean_ibi

        position = float(actual_start_sec / duration_sec)

        # Prepare .INFO data with all metadata (batch write at end for HDD efficiency)
        crop_info_path = crop_path.with_suffix('.INFO')
        crop_info_data = {
            "position": position,
            "start_time": actual_start_sec,
            "end_time": actual_end_sec,
            "start_sample": int(actual_start_sample),
            "end_sample": int(actual_end_sample),
            "duration": float(actual_end_sec - actual_start_sec),
            "samples": int(actual_end_sample - actual_start_sample),
            "bpm": round(local_bpm, 1),
            "beat_count": beat_count,
            "downbeats": int(num_downbeats),
            "source": str(full_mix_path.name),
            "has_stems": len(stem_paths) > 0,
            "stem_names": stem_names_loaded,
        }

        # Add pre-loaded transferrable features (already loaded before loop)
        crop_info_data.update(source_transferrable)

        # Queue for batch write
        pending_info_writes.append((crop_info_path, crop_info_data))

        crop_count += 1

        # Advance
        if overlap:
            next_target = current_start_sec + (length_sec / 2.0)
            next_start = find_closest_downbeat(align_grid, next_target)
            if next_start is not None and next_start > current_start_sec:
                current_start_sec = next_start
            else:
                current_start_sec = next_target
        else:
            # Non-overlap: strictly consecutive, no gaps
            current_start_sec = actual_end_sec

    # Wait for all async audio writes to complete
    write_errors = 0
    for future in write_futures:
        try:
            future.result()
        except Exception as e:
            logger.warning(f"Async write failed: {e}")
            write_errors += 1
    write_pool.shutdown(wait=False)

    if write_errors:
        logger.warning(f"{write_errors} crop writes failed")

    # Batch write all .INFO files (much faster on HDD than individual writes)
    if pending_info_writes:
        logger.debug(f"Batch writing {len(pending_info_writes)} .INFO files...")
        # Use merge=False since these are new crop files (no existing data)
        batch_write_info(pending_info_writes, merge=False)

    return crop_count


def main():
    parser = argparse.ArgumentParser(description="Create training crops for audio files.")
    parser.add_argument("path", type=str, help="Root directory containing organized audio folders")
    parser.add_argument("--length", type=int, default=2097152,
                        help="Crop length in SAMPLES (default: 2097152 = ~47.5s at 44.1kHz)")
    parser.add_argument("--overlap", action="store_true",
                        help="Enable 50%% overlap (next crop starts at last_start + length/2)")
    parser.add_argument("--div4", action="store_true",
                        help="Ensure each crop contains a number of downbeats divisible by 4")
    parser.add_argument("--sequential", action="store_true",
                        help="Sequential mode: fixed sample length, no beat alignment")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory for crops (creates per-track folders)")
    parser.add_argument("--workers", "-j", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing crops (default: skip if crops exist)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    input_path = Path(args.path)

    # Resolve output directory first (needed for filtering)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    # Snapshot folders BEFORE any processing to prevent infinite loops
    folders = []

    if input_path.is_file():
        logger.info(f"Processing single file: {input_path}")
        folders = [input_path.parent]
    elif input_path.is_dir():
        stems = get_stem_files(input_path, include_full_mix=True)
        if 'full_mix' in stems:
            logger.info(f"Processing single folder: {input_path}")
            folders = [input_path]
        else:
            all_folders = find_organized_folders(input_path)

            # Filter out folders that are inside the output directory or look like crop folders
            for folder in all_folders:
                folder_resolved = folder.resolve()

                # Skip if folder is inside output_dir
                if output_dir:
                    try:
                        folder_resolved.relative_to(output_dir)
                        logger.debug(f"Skipping (inside output_dir): {folder.name}")
                        continue
                    except ValueError:
                        pass  # Not inside output_dir, keep it

                # Skip folders that look like crop folders (contain _N.<ext> files)
                crop_files = []
                for ext in ['.flac', '.mp3', '.wav', '.m4a']:
                    crop_files.extend(folder.glob(f"*_[0-9]{ext}"))
                    crop_files.extend(folder.glob(f"*_[0-9][0-9]{ext}"))
                if crop_files and not any(folder.glob("full_mix.*")):
                    logger.debug(f"Skipping (looks like crop folder): {folder.name}")
                    continue

                # Skip "crops" subfolders
                if folder.name == "crops":
                    logger.debug(f"Skipping (crops subfolder): {folder}")
                    continue

                folders.append(folder)

    if not folders:
        logger.error(f"No valid organized folders found for: {input_path}")
        return

    # Freeze the folder list - this snapshot prevents processing newly created folders
    folders = list(folders)
    logger.info(f"Found {len(folders)} folders to process (snapshot taken).")
    logger.info(f"Crop length: {args.length} samples")

    if args.sequential:
        logger.info("Mode: Sequential (fixed sample length, no beat alignment)")
    else:
        logger.info(f"Mode: Beat-aligned (overlap={args.overlap}, div4={args.div4})")

    def process_folder_with_lock(folder):
        """Process a single folder with file locking."""
        with FileLock(folder) as lock:
            if not lock.acquired:
                logger.warning(f"Skipping {folder.name} - locked by another process")
                return 0
            return create_crops_for_file(
                folder, args.length, args.overlap, args.div4, args.sequential,
                output_dir, overwrite=args.overwrite
            )

    total_crops = 0
    failed = 0

    if args.workers > 1:
        logger.info(f"Using {args.workers} parallel workers")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_folder_with_lock, folder): folder 
                for folder in folders
            }
            for future in as_completed(futures):
                folder = futures[future]
                try:
                    count = future.result()
                    if count > 0:
                        logger.info(f"{folder.name}: Generated {count} crops.")
                    total_crops += count
                except Exception as e:
                    logger.error(f"Failed to process {folder.name}: {e}")
                    failed += 1
    else:
        # Sequential processing
        for folder in folders:
            try:
                count = process_folder_with_lock(folder)
                if count > 0:
                    logger.info(f"{folder.name}: Generated {count} crops.")
                total_crops += count
            except Exception as e:
                logger.error(f"Failed to process {folder.name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    logger.info(f"Finished. Total crops generated: {total_crops}, Failed: {failed}")


if __name__ == "__main__":
    main()
