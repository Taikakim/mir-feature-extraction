#!/usr/bin/env python3
"""
Verify Features Script

Checks a directory of organized audio folders to ensure all expected features
have been extracted.

Usage:
    python scripts/verify_features.py /path/to/data --show-missing
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to locate src if running from scripts dir
if (PROJECT_ROOT / "src").exists():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.file_utils import find_organized_folders
from core.batch_utils import get_missing_features
from core.common import setup_logging

logger = logging.getLogger(__name__)

# Define standard features to check
STANDARD_FEATURES = [
    'lufs', 'lra',                  # Loudness
    'bpm',                          # Rhythm
    'full_mix.BEATS_GRID',          # Rhythm files
    'spectral_centroid',            # Spectral
    'chroma_mean',                  # Harmonic
    'brightness',                   # Timbral
    'danceability',                 # Essentia
    # Metadata (new)
    'release_year',
    'spotify_id'
]

def verify_features(directory: Path, show_missing: bool = False, verbose: bool = False):
    """Verify presence of features in all folders."""
    folders = find_organized_folders(directory)
    
    if not folders:
        logger.error(f"No organized folders found in {directory}")
        return

    logger.info(f"Verifying features for {len(folders)} folders...")
    
    stats = {
        'complete': 0,
        'incomplete': 0,
        'missing_counts': {}
    }
    
    for folder in folders:
        missing = get_missing_features(folder, STANDARD_FEATURES)
        
        if not missing:
            stats['complete'] += 1
            if verbose:
                logger.info(f"✓ {folder.name}")
        else:
            stats['incomplete'] += 1
            if show_missing or verbose:
                logger.warning(f"✗ {folder.name} missing: {', '.join(missing)}")
            
            for m in missing:
                stats['missing_counts'][m] = stats['missing_counts'].get(m, 0) + 1

    print("\n" + "="*60)
    print("FEATURE VERIFICATION SUMMARY")
    print("="*60)
    print(f"Total Folders: {len(folders)}")
    print(f"Complete:      {stats['complete']} ({stats['complete']/len(folders)*100:.1f}%)")
    print(f"Incomplete:    {stats['incomplete']}")
    
    if stats['missing_counts']:
        print("\nMissing Feature Counts:")
        for feature, count in sorted(stats['missing_counts'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {count} folders")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Verify extracted features")
    parser.add_argument("path", help="Directory containing organized folders")
    parser.add_argument("--show-missing", action="store_true", help="List missing features per folder")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    setup_logging(level=logging.INFO)
    
    path = Path(args.path)
    if not path.exists():
        logger.error("Path does not exist")
        sys.exit(1)
        
    verify_features(path, show_missing=args.show_missing, verbose=args.verbose)

if __name__ == "__main__":
    main()
