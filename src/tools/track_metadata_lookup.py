#!/usr/bin/env python3
"""
Track Metadata Lookup Tool

Batch tool to correct artist names for compilation tracks and retrieve 
original release year using Spotify and MusicBrainz APIs.

Usage:
    # Preview changes (dry run)
    python src/tools/track_metadata_lookup.py /path/to/data --dry-run
    
    # Apply changes
    python src/tools/track_metadata_lookup.py /path/to/data
    
    # Only update .INFO, don't rename folders
    python src/tools/track_metadata_lookup.py /path/to/data --skip-rename

Setup:
    pip install spotipy musicbrainzngs
    
    # Spotify credentials (create app at https://developer.spotify.com/dashboard)
    export SPOTIFY_CLIENT_ID="your_id"
    export SPOTIFY_CLIENT_SECRET="your_secret"
"""

import argparse
import logging
import os
import re
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.common import setup_logging
from src.core.json_handler import safe_update, get_info_path
from src.core.file_utils import find_organized_folders, get_stem_files

logger = logging.getLogger(__name__)

# Try importing APIs
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIFY_AVAILABLE = True
except ImportError:
    SPOTIFY_AVAILABLE = False
    logger.warning("spotipy not installed. Run: pip install spotipy")

try:
    import musicbrainzngs
    MUSICBRAINZ_AVAILABLE = True
    musicbrainzngs.set_useragent("MIR-Feature-Extraction", "1.0", "https://github.com/user/mir")
except ImportError:
    MUSICBRAINZ_AVAILABLE = False
    logger.warning("musicbrainzngs not installed. Run: pip install musicbrainzngs")


def init_spotify() -> Optional[object]:
    """Initialize Spotify client."""
    if not SPOTIFY_AVAILABLE:
        return None
    
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        logger.warning("SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET not set")
        return None
    
    try:
        return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ))
    except Exception as e:
        logger.error(f"Failed to initialize Spotify: {e}")
        return None


def search_spotify(sp, track_name: str, current_artist: str = None, 
                   fetch_genres: bool = True) -> Optional[Dict]:
    """
    Search Spotify for a track.
    
    Returns:
        Dict with enhanced metadata:
        - artist: Primary artist name (string)
        - artists: All artist names (array)
        - track: Track name
        - album: Album name
        - release_date: Full release date (YYYY-MM-DD or YYYY)
        - release_year: Year only (int)
        - label: Record label
        - genres: Artist genres (array, requires extra API call if fetch_genres=True)
        - popularity: Track popularity (0-100)
        - spotify_id: Spotify track ID
    """
    if not sp:
        return None
    
    try:
        # Clean track name for search
        clean_name = re.sub(r'\s*\(.*?\)\s*', ' ', track_name)  # Remove parenthetical info
        clean_name = re.sub(r'\s*\[.*?\]\s*', ' ', clean_name)  # Remove bracketed info
        clean_name = clean_name.strip()
        
        # Search by track name
        query = f'track:"{clean_name}"'
        results = sp.search(q=query, type='track', limit=5)
        
        if not results['tracks']['items']:
            # Try simpler search
            results = sp.search(q=clean_name, type='track', limit=5)
        
        if not results['tracks']['items']:
            return None
        
        # Get first result (usually best match)
        track = results['tracks']['items'][0]
        album = track['album']
        
        # Extract all artists as array
        artists = [a['name'] for a in track['artists']]
        
        # Extract release date
        release_date = album.get('release_date', '')
        release_year = None
        if release_date:
            try:
                release_year = int(release_date.split('-')[0])
            except (ValueError, IndexError):
                pass
        
        # Get label (requires fetching full album details)
        label = None
        try:
            album_details = sp.album(album['id'])
            label = album_details.get('label')
        except Exception:
            pass
        
        # Get genres from primary artist (requires extra API call)
        genres = []
        if fetch_genres and track['artists']:
            try:
                artist_details = sp.artist(track['artists'][0]['id'])
                genres = artist_details.get('genres', [])
            except Exception:
                pass
        
        return {
            'artist': artists[0] if artists else 'Unknown',
            'artists': artists,
            'track': track['name'],
            'album': album['name'],
            'release_date': release_date,
            'release_year': release_year,
            'label': label,
            'genres': genres,
            'popularity': track.get('popularity'),
            'spotify_id': track['id'],
            'artist_id': track['artists'][0]['id'] if track['artists'] else None
        }
        
    except Exception as e:
        logger.debug(f"Spotify search failed for '{track_name}': {e}")
        return None


def fetch_audio_features(sp, track_id: str) -> Optional[Dict]:
    """
    Fetch Spotify audio features for a track.
    
    Returns:
        Dict with:
        - acousticness (0-1)
        - energy (0-1)
        - instrumentalness (0-1)
        - time_signature (3, 4, 5, etc.)
        - valence (0-1)
        - danceability (0-1)
        - speechiness (0-1)
        - liveness (0-1)
        - key (0-11, C=0)
        - mode (0=minor, 1=major)
        - tempo (BPM)
    """
    if not sp or not track_id:
        return None
    
    try:
        features = sp.audio_features([track_id])
        if not features or not features[0]:
            return None
        
        f = features[0]
        return {
            'spotify_acousticness': f.get('acousticness'),
            'spotify_energy': f.get('energy'),
            'spotify_instrumentalness': f.get('instrumentalness'),
            'spotify_time_signature': f.get('time_signature'),
            'spotify_valence': f.get('valence'),
            'spotify_danceability': f.get('danceability'),
            'spotify_speechiness': f.get('speechiness'),
            'spotify_liveness': f.get('liveness'),
            'spotify_key': f.get('key'),
            'spotify_mode': f.get('mode'),
            'spotify_tempo': f.get('tempo'),
        }
    except Exception as e:
        logger.debug(f"Failed to fetch audio features for {track_id}: {e}")
        return None


def search_musicbrainz(track_name: str, artist_hint: str = None) -> Optional[Dict]:
    """
    Search MusicBrainz for a track and get original release year.
    
    MusicBrainz can find the earliest release of a recording.
    
    Returns:
        Dict with 'artist', 'track', 'release_year' or None
    """
    if not MUSICBRAINZ_AVAILABLE:
        return None
    
    try:
        # Clean track name
        clean_name = re.sub(r'\s*\(.*?\)\s*', ' ', track_name)
        clean_name = re.sub(r'\s*\[.*?\]\s*', ' ', clean_name)
        clean_name = clean_name.strip()
        
        # Search for recordings
        query = f'recording:"{clean_name}"'
        if artist_hint and artist_hint.lower() != "various artists":
            query += f' AND artist:"{artist_hint}"'
        
        result = musicbrainzngs.search_recordings(query=query, limit=5)
        
        if not result.get('recording-list'):
            return None
        
        recording = result['recording-list'][0]
        
        # Get artist
        artist = "Unknown"
        if 'artist-credit' in recording and recording['artist-credit']:
            artist = recording['artist-credit'][0]['artist']['name']
        
        # Get earliest release year by looking at releases
        earliest_year = None
        if 'release-list' in recording:
            for release in recording['release-list']:
                if 'date' in release:
                    try:
                        year = int(release['date'].split('-')[0])
                        if earliest_year is None or year < earliest_year:
                            earliest_year = year
                    except (ValueError, IndexError):
                        pass
        
        return {
            'artist': artist,
            'track': recording.get('title', track_name),
            'release_year': earliest_year,
            'musicbrainz_id': recording.get('id')
        }
        
    except Exception as e:
        logger.debug(f"MusicBrainz search failed for '{track_name}': {e}")
        return None


def extract_track_name(folder_name: str) -> Tuple[str, str, str]:
    """
    Extract artist, album, and track name from folder name.
    
    Handles formats like:
    - "Various Artists - Album Name - Track Name"
    - "Artist - Track Name"
    - "001 Artist - Track Name"
    
    Returns:
        Tuple of (artist, album_or_empty, track_name)
    """
    # Remove leading track number
    name = re.sub(r'^\d+\s*', '', folder_name).strip()
    
    parts = name.split(' - ')
    
    if len(parts) >= 3:
        # "Artist - Album - Track" format
        return parts[0].strip(), parts[1].strip(), ' - '.join(parts[2:]).strip()
    elif len(parts) == 2:
        # "Artist - Track" format
        return parts[0].strip(), '', parts[1].strip()
    else:
        # Just track name
        return '', '', name


def lookup_track(track_name: str, artist_hint: str = None, sp=None, 
                 fetch_audio_features_flag: bool = False) -> Optional[Dict]:
    """
    Look up track metadata using available APIs.
    
    Tries Spotify first, then MusicBrainz.
    
    Args:
        track_name: Name of the track
        artist_hint: Optional artist name hint
        sp: Spotify client instance
        fetch_audio_features_flag: If True, fetch Spotify audio features (extra API call)
    """
    result = None
    
    # Try Spotify first (faster, usually better for electronic music)
    if sp:
        result = search_spotify(sp, track_name, artist_hint)
        if result:
            logger.debug(f"Found via Spotify: {result['artist']} - {result['track']}")
            
            # Fetch audio features if requested
            if fetch_audio_features_flag and result.get('spotify_id'):
                audio_features = fetch_audio_features(sp, result['spotify_id'])
                if audio_features:
                    result.update(audio_features)
                    logger.debug(f"Added Spotify audio features")
    
    # Try MusicBrainz for original release year or as fallback
    mb_result = search_musicbrainz(track_name, artist_hint)
    
    if mb_result:
        if result:
            # If Spotify found it but MB has earlier year, use that
            if mb_result.get('release_year') and result.get('release_year'):
                if mb_result['release_year'] < result['release_year']:
                    result['original_release_year'] = mb_result['release_year']
                    logger.debug(f"MusicBrainz found earlier year: {mb_result['release_year']}")
        else:
            result = mb_result
            logger.debug(f"Found via MusicBrainz: {result['artist']} - {result['track']}")
    
    # Rate limiting for MusicBrainz (max 1 req/sec)
    time.sleep(0.5)
    
    return result


def process_folder(folder: Path, sp=None, dry_run: bool = True, 
                   skip_rename: bool = False,
                   fetch_audio_features: bool = False,
                   force_metadata: bool = False) -> Optional[Dict]:
    """
    Process a single folder.
    
    Returns:
        Dict with changes made, or None if no changes
    """
    folder_name = folder.name
    artist, album, track_name = extract_track_name(folder_name)
    
    # Check if this is a "Various Artists" track that needs renaming
    needs_rename = artist.lower() in ['various artists', 'various', 'va', '']
    
    # Check if we need to look up release_year even if artist is correct
    needs_year = False
    if not needs_rename:
        # Check if .INFO file is missing release_year
        try:
            stems = get_stem_files(folder, include_full_mix=True)
            if 'full_mix' in stems:
                info_path = get_info_path(stems['full_mix'])
                if info_path.exists():
                    import json
                    with open(info_path) as f:
                        info_data = json.load(f)
                    if 'release_year' not in info_data:
                        needs_year = True
                        logger.debug(f"{folder_name} - has artist but missing release_year")
        except Exception:
            pass
    
    # Skip if nothing to do (unless forced or fetching extra features)
    if not needs_rename and not needs_year and not force_metadata and not fetch_audio_features:
        logger.debug(f"Skipping {folder_name} - artist set and has release_year")
        return None
    
    if not track_name:
        logger.warning(f"Could not extract track name from: {folder_name}")
        return None
    
    # Use existing artist as hint if not "Various Artists"
    artist_hint = artist if not needs_rename else None
    
    logger.info(f"Looking up: {artist + ' - ' if artist_hint else ''}{track_name}")
    
    # Look up metadata
    result = lookup_track(track_name, artist_hint=artist_hint, sp=sp,
                          fetch_audio_features_flag=fetch_audio_features)
    
    if not result:
        logger.warning(f"No match found for: {track_name}")
        return None
    
    # Prepare changes
    changes = {
        'original_folder': folder_name,
        'track_name': track_name,
        'found_artist': result['artist'],
        'found_track': result.get('track', track_name),
        'release_year': result.get('original_release_year') or result.get('release_year'),
        'needs_rename': needs_rename,
        'needs_year': needs_year,
    }
    
    # Build new folder name based on canonical metadata
    canonical_name = f"{result['artist']} - {result.get('track', track_name)}"
    # Clean up for filesystem
    canonical_name = re.sub(r'[<>:"/\\|?*]', '', canonical_name)
    
    # Determine if rename is needed (if canonical name differs from current folder name)
    # This catches "Various Artists" cases AND numerical prefixes like "081 Artist - Track"
    should_rename = canonical_name != folder_name
    
    changes['new_folder'] = canonical_name if should_rename else folder_name
    
    if dry_run:
        if should_rename:
            logger.info(f"  Would rename to: {canonical_name}")
        if changes['release_year']:
            logger.info(f"  Would set release_year: {changes['release_year']}")
        return changes
    
    # Apply changes
    new_name = canonical_name
    new_folder = folder.parent / new_name
    old_name_base = folder_name  # Store for file renaming
    
    # Rename folder (only if needed)
    if should_rename and not skip_rename:
        if new_folder.exists():
            logger.warning(f"Target folder already exists: {new_folder}")
        else:
            try:
                folder.rename(new_folder)
                logger.info(f"Renamed folder: {folder_name} → {new_name}")
                folder = new_folder
            except Exception as e:
                logger.error(f"Failed to rename folder: {e}")
                return changes
        
        # Rename files inside folder that contain the old name
        rename_files_in_folder(folder, old_name_base, new_name)
    elif not should_rename:
        # Folder name is correct, but check if files inside differ (orphaned names)
        # This handles cases where folder was renamed but files inside weren't
        rename_files_in_folder(folder, "Various Artists", new_name)  # Catch generic
        
        # Also try to catch numbered variations
        # Try to find any file that matches "NNN Various Artists" pattern
        for f in folder.glob("*Various Artists*"):
            if "Various Artists" in f.name and new_name not in f.name:
                rename_files_in_folder(folder, f.stem, new_name)
                break
    
    # Force normalization of aux files (remove numbers, match folder name)
    normalize_aux_filenames(folder, folder.name, dry_run)
    
    # Update .INFO file with all metadata
    try:
        stems = get_stem_files(folder, include_full_mix=True)
        if 'full_mix' in stems:
            info_path = get_info_path(stems['full_mix'])
            
            # Consolidate any orphaned/old .INFO files into the main one
            consolidate_info_files(folder, info_path, dry_run)
            
            # Build info data with all available metadata
            info_data = {}
            
            # Core fields
            if result.get('release_year'):
                info_data['release_year'] = result['release_year']
            if result.get('release_date'):
                info_data['release_date'] = result['release_date']
            if result.get('artists'):
                info_data['artists'] = result['artists']
            if result.get('label'):
                info_data['label'] = result['label']
            if result.get('genres'):
                info_data['genres'] = result['genres']
            if result.get('popularity') is not None:
                info_data['popularity'] = result['popularity']
            
            # IDs
            if result.get('spotify_id'):
                info_data['spotify_id'] = result['spotify_id']
            if result.get('musicbrainz_id'):
                info_data['musicbrainz_id'] = result['musicbrainz_id']
            
            # Audio features (if available)
            for key in ['spotify_acousticness', 'spotify_energy', 'spotify_instrumentalness',
                        'spotify_time_signature', 'spotify_valence', 'spotify_danceability',
                        'spotify_speechiness', 'spotify_liveness', 'spotify_key', 
                        'spotify_mode', 'spotify_tempo']:
                if result.get(key) is not None:
                    info_data[key] = result[key]
            
            if info_data:
                safe_update(info_path, info_data)
                fields_written = list(info_data.keys())
                logger.info(f"Updated .INFO with: {', '.join(fields_written[:5])}{'...' if len(fields_written) > 5 else ''}")
    except Exception as e:
        logger.error(f"Failed to update .INFO: {e}")
    
    return changes


def consolidate_info_files(folder: Path, target_info_path: Path, dry_run: bool = False):
    """
    Find any .INFO files that don't match the target info file (orphan/old names)
    and merge their unique content into the target file, then delete them.
    """
    if not folder.exists():
        return
        
    for file_path in folder.glob("*.INFO"):
        # Skip if it's the target file
        if file_path.absolute() == target_info_path.absolute():
            continue
            
        # It's an orphan/old file
        try:
            logger.info(f"Found orphaned INFO file: {file_path.name}")
            
            if dry_run:
                logger.info(f"  Would consolidate into {target_info_path.name} and delete")
                continue
                
            # Read orphan data
            with open(file_path, 'r') as f:
                old_data = json.load(f)
            
            # Merge into target (safe_update preserves existing target keys, so it acts as "fill missing")
            # Wait, if we want to SAVE data that might be in the orphan but not in target:
            safe_update(target_info_path, old_data)
            
            # Delete orphan
            file_path.unlink()
            logger.info(f"  Consolidated and deleted: {file_path.name}")
            
        except Exception as e:
            logger.warning(f"Failed to consolidate {file_path.name}: {e}")


def rename_files_in_folder(folder: Path, old_name: str, new_name: str):
    """
    Rename all files in folder that contain the old name pattern.
    
    Handles patterns like:
    - "106 Various Artists - In un altro loop.INFO" → "BXP - In un altro loop.INFO"
    - Also handles: .BEATS_GRID, .ONSETS, .DOWNBEATS, etc.
    """
    # Normalize old name for matching (remove leading numbers, clean up)
    old_base = re.sub(r'^\d+\s*', '', old_name).strip()
    
    renamed_count = 0
    
    for file_path in folder.iterdir():
        if file_path.is_file():
            file_name = file_path.name
            
            # Check if file contains the old name pattern
            # Match either the full old name or the cleaned version
            if old_name in file_name or old_base in file_name:
                # Replace old name with new name
                new_file_name = file_name.replace(old_name, new_name)
                new_file_name = new_file_name.replace(old_base, new_name)
                
                # Also handle case with leading numbers in filename
                # e.g., "106 Various Artists - Track.INFO" → "Artist - Track.INFO"
                old_pattern = re.sub(r'^\d+\.?\s*', '', old_name)
                if old_pattern in file_name:
                    new_file_name = file_name.replace(old_pattern, new_name)
                
                if new_file_name != file_name:
                    new_file_path = folder / new_file_name
                    if not new_file_path.exists():
                        try:
                            file_path.rename(new_file_path)
                            logger.debug(f"  Renamed file: {file_name} → {new_file_name}")
                            renamed_count += 1
                        except Exception as e:
                            logger.warning(f"  Failed to rename {file_name}: {e}")
                    else:
                        logger.debug(f"  Target file already exists: {new_file_name}")
    
    if renamed_count > 0:
        logger.info(f"  Renamed {renamed_count} files inside folder")


def batch_process_metadata(root_directory: Path, overwrite: bool = False) -> Dict[str, int]:
    """
    Batch process metadata for all organized folders.
    Compatible with batch_process.py interface.
    
    Args:
        root_directory: Root directory to search
        overwrite: Force metadata lookup even if fields exist
        
    Returns:
        Dictionary with statistics
    """
    root_directory = Path(root_directory)
    folders = find_organized_folders(root_directory)
    
    sp = init_spotify()
    
    stats = {
        'total': len(folders),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }
    
    for i, folder in enumerate(folders, 1):
        logger.info(f"Processing metadata {i}/{len(folders)}: {folder.name}")
        try:
            # We map overwrite to 'force_metadata' and 'skip_rename' to False (or should we?)
            # Batch process usually implies full processing.
            # However, renaming folders might break other tools if they depend on paths mid-stream?
            # But renaming IS the point of this tool.
            
            # Note: process_folder returns None if skipped, Dict if changed
            res = process_folder(
                folder, 
                sp=sp, 
                dry_run=False,
                force_metadata=overwrite,
                fetch_audio_features=True # Generally want this for batch
            )
            
            if res is None:
                stats['skipped'] += 1
            else:
                stats['success'] += 1
                
        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append(f"{folder.name}: {e}")
            logger.error(f"Failed to lookup metadata: {e}")
            
    return stats


def normalize_aux_filenames(folder: Path, canonical_name: str, dry_run: bool = False):
    """
    Force rename auxiliary files to match the folder name exactly.
    Target types: .BEATS_GRID, .ONSETS, .DOWNBEATS, .INFO (if consolidated), .txt
    Also handles .CHROMA if present.
    """
    target_extensions = ['.BEATS_GRID', '.ONSETS', '.DOWNBEATS', '.INFO', '.CHROMA']
    
    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue
            
        # Check if it's one of our target types by checking suffix or end of name
        suffix = file_path.suffix
        matched = False
        
        if suffix in target_extensions:
            matched = True
        else:
            # Check for double extensions or complex suffixes
            for ext in target_extensions:
                if file_path.name.endswith(ext):
                    matched = True
                    suffix = ext
                    break
        
        if not matched:
            continue
            
        # Skip if already correct
        target_name = f"{canonical_name}{suffix}"
        if file_path.name == target_name:
            continue
            
        # Rename it
        new_path = folder / target_name
        
        if dry_run:
            logger.info(f"  Would normalize: {file_path.name} → {target_name}")
        else:
            if new_path.exists():
                # If target exists...
                if suffix == '.INFO':
                    # Consolidate instead of overwrite (handled by consolidate_info_files,
                    # but good to catch here or skip)
                    continue
                else:
                    # For grid files etc, warn or overwrite? 
                    # Usually better to overwrite if we are fixing names
                     logger.debug(f"  Target exists, skipping: {target_name}")
            else:
                try:
                    file_path.rename(new_path)
                    logger.info(f"  Normalized: {file_path.name} → {target_name}")
                except Exception as e:
                    logger.warning(f"  Failed to normalize {file_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Look up track metadata and correct artist names",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes
  python src/tools/track_metadata_lookup.py /path/to/data --dry-run
  
  # Apply changes
  python src/tools/track_metadata_lookup.py /path/to/data
  
  # Only update .INFO files, don't rename
  python src/tools/track_metadata_lookup.py /path/to/data --skip-rename

Setup:
  pip install spotipy musicbrainzngs
  
  export SPOTIFY_CLIENT_ID="your_id"
  export SPOTIFY_CLIENT_SECRET="your_secret"
        """
    )
    
    parser.add_argument("path", help="Root directory containing organized folders")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Preview changes without applying them")
    parser.add_argument("--skip-rename", action="store_true",
                        help="Only update .INFO files, don't rename folders")
    parser.add_argument("--audio-features", action="store_true",
                        help="Fetch Spotify audio features (acousticness, energy, valence, etc.)")
    parser.add_argument("--force-metadata", "-f", action="store_true",
                        help="Force metadata lookup even if fields already exist")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    root_path = Path(args.path)
    
    if not root_path.exists():
        logger.error(f"Path does not exist: {root_path}")
        sys.exit(1)
    
    # Initialize Spotify
    sp = init_spotify()
    if sp:
        logger.info("Spotify API initialized")
    else:
        if args.audio_features:
            logger.error("Spotify credentials required for --audio-features!")
            logger.error("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
            sys.exit(1)
        logger.warning("Spotify not available, using MusicBrainz only")
    
    if not MUSICBRAINZ_AVAILABLE and not sp:
        logger.error("No APIs available. Install spotipy and/or musicbrainzngs")
        sys.exit(1)
    
    # Find folders
    folders = find_organized_folders(root_path)
    logger.info(f"Found {len(folders)} organized folders")
    
    # Process
    mode = "DRY RUN" if args.dry_run else "APPLYING CHANGES"
    print(f"\n{'='*60}")
    print(f"TRACK METADATA LOOKUP ({mode})")
    print(f"{'='*60}\n")
    
    results = {
        'processed': 0,
        'found': 0,
        'not_found': 0,
        'skipped': 0,
        'changes': []
    }
    
    for folder in folders:
        try:
            change = process_folder(
                folder, 
                sp=sp, 
                dry_run=args.dry_run,
                skip_rename=args.skip_rename,
                fetch_audio_features=args.audio_features,
                force_metadata=args.force_metadata
            )
            
            if change:
                results['found'] += 1
                results['changes'].append(change)
            elif change is None:
                results['skipped'] += 1
            else:
                results['not_found'] += 1
                
            results['processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing {folder.name}: {e}")
            results['not_found'] += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Processed:  {results['processed']}")
    print(f"Found:      {results['found']}")
    print(f"Not found:  {results['not_found']}")
    print(f"Skipped:    {results['skipped']} (artist already set)")
    
    if args.dry_run:
        print(f"\nDRY RUN - No changes made. Remove --dry-run to apply.")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
