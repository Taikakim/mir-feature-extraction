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
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def search_spotify(sp, track_name: str, current_artist: str = None) -> Optional[Dict]:
    """
    Search Spotify for a track.
    
    Returns:
        Dict with 'artist', 'track', 'album', 'release_year' or None
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
        
        # Extract year from release date
        release_date = track['album'].get('release_date', '')
        if release_date:
            release_year = int(release_date.split('-')[0])
        else:
            release_year = None
        
        return {
            'artist': track['artists'][0]['name'],
            'track': track['name'],
            'album': track['album']['name'],
            'release_year': release_year,
            'spotify_id': track['id']
        }
        
    except Exception as e:
        logger.debug(f"Spotify search failed for '{track_name}': {e}")
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


def lookup_track(track_name: str, artist_hint: str = None, sp=None) -> Optional[Dict]:
    """
    Look up track metadata using available APIs.
    
    Tries Spotify first, then MusicBrainz.
    """
    result = None
    
    # Try Spotify first (faster, usually better for electronic music)
    if sp:
        result = search_spotify(sp, track_name, artist_hint)
        if result:
            logger.debug(f"Found via Spotify: {result['artist']} - {result['track']}")
    
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
                   skip_rename: bool = False) -> Optional[Dict]:
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
    
    # Skip if nothing to do
    if not needs_rename and not needs_year:
        logger.debug(f"Skipping {folder_name} - artist set and has release_year")
        return None
    
    if not track_name:
        logger.warning(f"Could not extract track name from: {folder_name}")
        return None
    
    # Use existing artist as hint if not "Various Artists"
    artist_hint = artist if not needs_rename else None
    
    logger.info(f"Looking up: {artist + ' - ' if artist_hint else ''}{track_name}")
    
    # Look up metadata
    result = lookup_track(track_name, artist_hint=artist_hint, sp=sp)
    
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
    
    # Build new folder name (only if renaming needed)
    if needs_rename:
        new_name = f"{result['artist']} - {result.get('track', track_name)}"
        # Clean up for filesystem
        new_name = re.sub(r'[<>:"/\\|?*]', '', new_name)
        changes['new_folder'] = new_name
    else:
        new_name = folder_name
        changes['new_folder'] = folder_name
    
    if dry_run:
        if needs_rename:
            logger.info(f"  Would rename to: {new_name}")
        if changes['release_year']:
            logger.info(f"  Would set release_year: {changes['release_year']}")
        return changes
    
    # Apply changes
    new_folder = folder.parent / new_name
    old_name_base = folder_name  # Store for file renaming
    
    # Rename folder (only if needed)
    if needs_rename and not skip_rename and new_name != folder_name:
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
    
    # Update .INFO file with release_year
    if changes['release_year']:
        try:
            stems = get_stem_files(folder, include_full_mix=True)
            if 'full_mix' in stems:
                info_path = get_info_path(stems['full_mix'])
                info_data = {'release_year': changes['release_year']}
                if result.get('spotify_id'):
                    info_data['spotify_id'] = result['spotify_id']
                if result.get('musicbrainz_id'):
                    info_data['musicbrainz_id'] = result['musicbrainz_id']
                safe_update(info_path, info_data)
                logger.info(f"Updated .INFO with release_year: {changes['release_year']}")
        except Exception as e:
            logger.error(f"Failed to update .INFO: {e}")
    
    return changes


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
                skip_rename=args.skip_rename
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
