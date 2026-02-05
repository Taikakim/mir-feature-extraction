"""
Metadata Utilities for MIR Project

Audio file metadata extraction using mutagen (ID3/FLAC/OGG tags).

Usage:
    from core.metadata_utils import extract_audio_metadata, VARIOUS_ARTISTS_ALIASES
    
    metadata = extract_audio_metadata(Path("track.mp3"))
    # Returns: {'track_metadata_artist': '...', 'track_metadata_title': '...', ...}
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import mutagen for MP3 metadata extraction
try:
    import mutagen
    from mutagen.id3 import ID3
    MUTAGEN_AVAILABLE = True
except ImportError:
    mutagen = None  # type: ignore
    ID3 = None  # type: ignore
    MUTAGEN_AVAILABLE = False
    logger.debug("mutagen not available - MP3 metadata extraction disabled (pip install mutagen)")


# Artists that should trigger metadata lookup for real artist
VARIOUS_ARTISTS_ALIASES = {
    'various artists',
    'various',
    'va',
    'v/a',
    'v.a.',
    'compilation',
    'various artist',
    'unknown artist',
    'unknown',
    '',
}


def extract_audio_metadata(audio_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from audio file (MP3 ID3 tags, FLAC tags, etc.).

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with metadata keys:
        - track_metadata_artist
        - track_metadata_title
        - track_metadata_album
        - track_metadata_year
        - track_metadata_genre
    """
    if not MUTAGEN_AVAILABLE:
        return {}

    metadata = {}

    try:
        # Try to load with mutagen (handles MP3, FLAC, OGG, etc.)
        audio = mutagen.File(audio_path, easy=True)

        if audio is None:
            return {}

        # Extract common tags
        if 'artist' in audio:
            metadata['track_metadata_artist'] = audio['artist'][0]
        if 'title' in audio:
            metadata['track_metadata_title'] = audio['title'][0]
        if 'album' in audio:
            metadata['track_metadata_album'] = audio['album'][0]
        if 'date' in audio:
            # Date can be YYYY or YYYY-MM-DD
            date_str = audio['date'][0]
            if date_str:
                metadata['track_metadata_year'] = int(date_str[:4])
        if 'genre' in audio:
            metadata['track_metadata_genre'] = audio['genre'][0]

        # For MP3, also try to get year from TDRC if date wasn't found
        if 'track_metadata_year' not in metadata and audio_path.suffix.lower() == '.mp3':
            try:
                id3 = ID3(audio_path)
                if 'TDRC' in id3:
                    year_str = str(id3['TDRC'].text[0])
                    if year_str:
                        metadata['track_metadata_year'] = int(year_str[:4])
                elif 'TYER' in id3:
                    year_str = str(id3['TYER'].text[0])
                    if year_str:
                        metadata['track_metadata_year'] = int(year_str[:4])
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"Failed to extract metadata from {audio_path.name}: {e}")

    return metadata


def build_folder_name_from_metadata(metadata: Dict[str, Any], original_name: str) -> Optional[str]:
    """
    Build a folder name from metadata.

    Format: "Artist - Title" or "Artist - Album - Title"

    Returns None if insufficient metadata.
    """
    artist = metadata.get('track_metadata_artist')
    title = metadata.get('track_metadata_title')

    if not artist or not title:
        return None

    # Clean up for filesystem
    def clean(s):
        return re.sub(r'[<>:"/\\|?*]', '', s).strip()

    artist = clean(artist)
    title = clean(title)

    if not artist or not title:
        return None

    return f"{artist} - {title}"


def is_various_artists(artist: str) -> bool:
    """
    Check if artist name is a 'Various Artists' alias.
    
    Args:
        artist: Artist name to check
        
    Returns:
        True if the artist is a various artists alias
    """
    if not artist:
        return True
    return artist.lower().strip() in VARIOUS_ARTISTS_ALIASES
