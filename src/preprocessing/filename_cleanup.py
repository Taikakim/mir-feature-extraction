#!/usr/bin/env python3
"""
Filename Cleanup Tool for T5 Tokenizer Compatibility

Renames folders and files to use clean, ASCII-compatible names that won't
waste tokens during T5 encoding. Handles:
- Unicode accented characters → ASCII equivalents (è → e, é → e, etc.)
- Special encoding issues (escaped Unicode like \\u00e8)
- Multiple spaces → single space
- Leading/trailing whitespace
- Special characters that waste tokens
- Track numbers normalization

Usage:
    # Preview changes (dry run)
    python src/preprocessing/filename_cleanup.py /path/to/data --dry-run
    
    # Apply changes
    python src/preprocessing/filename_cleanup.py /path/to/data
    
    # Verbose output
    python src/preprocessing/filename_cleanup.py /path/to/data -v
"""

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.common import setup_logging

logger = logging.getLogger(__name__)


# ASCII transliteration map for common accented characters
TRANSLITERATION_MAP = {
    # French/Spanish/Portuguese
    'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a',
    'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
    'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
    'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o',
    'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
    'ý': 'y', 'ÿ': 'y',
    'ñ': 'n', 'ç': 'c',
    'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A', 'Å': 'A',
    'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E',
    'Ì': 'I', 'Í': 'I', 'Î': 'I', 'Ï': 'I',
    'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O',
    'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U',
    'Ý': 'Y', 'Ÿ': 'Y',
    'Ñ': 'N', 'Ç': 'C',
    # Nordic
    'æ': 'ae', 'ø': 'o', 'œ': 'oe',
    'Æ': 'AE', 'Ø': 'O', 'Œ': 'OE',
    'ð': 'd', 'þ': 'th',
    'Ð': 'D', 'Þ': 'TH',
    # German
    'ß': 'ss',
    # Polish/Czech/Slovak
    'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n', 'ś': 's', 'ź': 'z', 'ż': 'z',
    'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z',
    'č': 'c', 'ď': 'd', 'ě': 'e', 'ň': 'n', 'ř': 'r', 'š': 's', 'ť': 't', 'ů': 'u', 'ž': 'z',
    'Č': 'C', 'Ď': 'D', 'Ě': 'E', 'Ň': 'N', 'Ř': 'R', 'Š': 'S', 'Ť': 'T', 'Ů': 'U', 'Ž': 'Z',
    # Turkish
    'ğ': 'g', 'ı': 'i', 'ş': 's',
    'Ğ': 'G', 'İ': 'I', 'Ş': 'S',
    # Special punctuation
    '–': '-',  # en-dash
    '—': '-',  # em-dash
    ''': "'", ''': "'",  # single quotes
    '"': '"', '"': '"',  # double quotes
    '…': '...',  # ellipsis
    '•': '-',  # bullet
    '·': '-',  # middle dot
    '×': 'x',  # multiplication sign
    '÷': '-',  # division sign
}


def transliterate_to_ascii(text: str) -> str:
    """Convert accented characters to ASCII equivalents."""
    result = []
    for char in text:
        if char in TRANSLITERATION_MAP:
            result.append(TRANSLITERATION_MAP[char])
        elif ord(char) < 128:
            result.append(char)
        else:
            # Try NFD decomposition for other characters
            decomposed = unicodedata.normalize('NFD', char)
            ascii_char = ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn')
            if ascii_char and ord(ascii_char[0]) < 128:
                result.append(ascii_char)
            else:
                # Skip character if can't be transliterated
                pass
    return ''.join(result)


def decode_escaped_unicode(text: str) -> str:
    """Decode escaped Unicode sequences like \\u00e8 to actual characters."""
    # Match \uXXXX patterns
    def replace_unicode(match):
        try:
            return chr(int(match.group(1), 16))
        except ValueError:
            return match.group(0)
    
    # Handle both \\uXXXX and \uXXXX patterns
    text = re.sub(r'\\u([0-9a-fA-F]{4})', replace_unicode, text)
    return text


def clean_filename(name: str) -> str:
    """
    Clean a filename for T5 tokenizer compatibility.
    
    Args:
        name: Original filename (without extension)
        
    Returns:
        Cleaned filename
    """
    # First, decode any escaped Unicode
    name = decode_escaped_unicode(name)
    
    # Normalize Unicode (NFC form)
    name = unicodedata.normalize('NFC', name)
    
    # Transliterate to ASCII
    name = transliterate_to_ascii(name)
    
    # Replace problematic separators
    name = name.replace('_', ' ')  # Underscores to spaces (more natural)
    
    # Clean up track number formats
    # "089. Artist - Title" → "Artist - Title" (optional, configurable)
    # For now, just normalize the format but keep the number
    name = re.sub(r'^(\d+)\.\s*', r'\1 ', name)  # "089. " → "089 "
    
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    
    # Clean up around dashes
    name = re.sub(r'\s*-\s*', ' - ', name)  # Normalize " - "
    name = re.sub(r'\s+', ' ', name)  # Fix any double spaces that resulted
    
    # Remove problematic characters for filenames
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    
    # Remove non-printable characters
    name = ''.join(c for c in name if c.isprintable())
    
    # Strip leading/trailing whitespace
    name = name.strip()
    
    # Ensure not empty
    if not name:
        name = "Unknown"
    
    return name


def get_new_path(old_path: Path) -> Tuple[Path, bool]:
    """
    Get the new path for a file or directory.
    
    Returns:
        Tuple of (new_path, changed)
    """
    old_name = old_path.stem if old_path.is_file() else old_path.name
    extension = old_path.suffix if old_path.is_file() else ""
    
    new_name = clean_filename(old_name)
    new_full_name = new_name + extension
    
    new_path = old_path.parent / new_full_name
    changed = new_full_name != old_path.name
    
    return new_path, changed


def find_items_to_rename(root_path: Path) -> list:
    """Find all directories and files that need renaming."""
    items = []
    
    # Find all organized folders (directories containing full_mix)
    for item in root_path.iterdir():
        if item.is_dir():
            # Check if it's an organized folder
            has_audio = any(
                (item / f"full_mix{ext}").exists() 
                for ext in ['.flac', '.wav', '.mp3', '.ogg', '.m4a']
            )
            
            new_path, changed = get_new_path(item)
            if changed:
                items.append({
                    'type': 'directory',
                    'old_path': item,
                    'new_path': new_path,
                    'old_name': item.name,
                    'new_name': new_path.name
                })
                
            # Also check files inside that might need renaming
            # (like .INFO files that include the folder name)
            if has_audio:
                for file in item.iterdir():
                    if file.is_file() and file.suffix in ['.INFO', '.BEATS_GRID', '.DOWNBEATS']:
                        file_new_path, file_changed = get_new_path(file)
                        if file_changed:
                            items.append({
                                'type': 'file',
                                'old_path': file,
                                'new_path': file_new_path,
                                'old_name': file.name,
                                'new_name': file_new_path.name,
                                'inside': item.name
                            })
    
    return items


def apply_rename(item: dict) -> bool:
    """Apply a single rename operation."""
    try:
        old_path = item['old_path']
        new_path = item['new_path']
        
        # Handle collision
        if new_path.exists() and old_path != new_path:
            logger.warning(f"Target already exists, skipping: {new_path}")
            return False
            
        old_path.rename(new_path)
        return True
        
    except Exception as e:
        logger.error(f"Failed to rename {item['old_path']}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Clean up filenames for T5 tokenizer compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch rename all folders (default behavior)
  python src/preprocessing/filename_cleanup.py /path/to/data
  
  # Preview changes first (safe - no modifications)
  python src/preprocessing/filename_cleanup.py /path/to/data --dry-run
  
  # Verbose output
  python src/preprocessing/filename_cleanup.py /path/to/data -v
        """
    )
    
    parser.add_argument("path", help="Root directory containing organized folders")
    parser.add_argument("--dry-run", "-n", action="store_true", 
                        help="Preview changes without applying them")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    root_path = Path(args.path)
    
    if not root_path.exists():
        logger.error(f"Path does not exist: {root_path}")
        sys.exit(1)
        
    if not root_path.is_dir():
        logger.error(f"Path is not a directory: {root_path}")
        sys.exit(1)
    
    # Find items to rename
    logger.info(f"Scanning: {root_path}")
    items = find_items_to_rename(root_path)
    
    if not items:
        print("\n✅ All filenames are already clean!")
        return
    
    # Print summary
    mode = "DRY RUN - Preview" if args.dry_run else "APPLYING CHANGES"
    print("\n" + "=" * 80)
    print(f"FILENAME CLEANUP ({mode})")
    print("=" * 80)
    print(f"\nFound {len(items)} items to rename:\n")
    
    for item in items:
        type_icon = "📁" if item['type'] == 'directory' else "📄"
        print(f"{type_icon} {item['old_name']}")
        print(f"   → {item['new_name']}")
        if args.verbose:
            print()
    
    if args.dry_run:
        print("\n" + "-" * 80)
        print("DRY RUN - No changes made. Remove --dry-run to apply changes.")
        print("-" * 80)
        return
    
    print()
    
    # Apply changes (files first so they're renamed before parent directory)
    # Sort: files inside directories first, then directories
    items.sort(key=lambda x: (x['type'] == 'directory', str(x['old_path'])))
    
    success_count = 0
    fail_count = 0
    
    for item in items:
        if apply_rename(item):
            success_count += 1
            if args.verbose:
                logger.info(f"Renamed: {item['old_name']} → {item['new_name']}")
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Renamed: {success_count}")
    if fail_count:
        print(f"❌ Failed:  {fail_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
