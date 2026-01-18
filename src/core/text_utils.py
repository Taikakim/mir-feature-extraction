"""
Text Utilities for MIR Project

Functions for normalizing and cleaning text data to ensure compatibility
with various tokenizers and downstream processing.
"""

import unicodedata
from typing import Optional


def normalize_music_flamingo_text(text: str) -> str:
    """
    Normalize Music Flamingo output text for T5 tokenizer compatibility.

    Music Flamingo outputs often contain special Unicode characters that can
    cause issues with T5 tokenization and Stable Audio Tools processing:
    - Non-breaking hyphens (U+2011)
    - Narrow no-break spaces (U+202F)
    - Em-dashes (U+2014) and en-dashes (U+2013)
    - Curly quotes (U+2018, U+2019, U+201C, U+201D)

    This function replaces them with standard ASCII equivalents and applies
    NFKC normalization to handle any remaining compatibility characters.

    Args:
        text: Raw text (typically from Music Flamingo output)

    Returns:
        Normalized text safe for T5 tokenization

    Example:
        >>> text = "This is a track—with special characters"
        >>> normalized = normalize_music_flamingo_text(text)
        >>> print(normalized)
        "This is a track--with special characters"
    """
    if not text:
        return text

    # Replace specific problematic characters with ASCII equivalents
    replacements = {
        '\u2011': '-',   # Non-breaking hyphen → hyphen
        '\u2010': '-',   # Unicode hyphen → hyphen
        '\u202F': ' ',   # Narrow no-break space → space
        '\u2014': '--',  # Em dash → double hyphen
        '\u2013': '-',   # En dash → hyphen
        '\u2019': "'",   # Right single quotation mark → apostrophe
        '\u2018': "'",   # Left single quotation mark → apostrophe
        '\u201C': '"',   # Left double quotation mark → quote
        '\u201D': '"',   # Right double quotation mark → quote
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # NFKC normalization handles remaining compatibility characters
    # (e.g., ligatures, presentation forms, etc.)
    text = unicodedata.normalize('NFKC', text)

    return text


def normalize_for_t5(text: str) -> str:
    """
    Alias for normalize_music_flamingo_text for general T5 compatibility.

    Args:
        text: Raw text

    Returns:
        Normalized text safe for T5 tokenization
    """
    return normalize_music_flamingo_text(text)


def validate_text_safety(text: str, allow_non_ascii: Optional[set] = None) -> tuple[bool, list]:
    """
    Validate that text contains no problematic Unicode characters.

    Args:
        text: Text to validate
        allow_non_ascii: Set of allowed non-ASCII characters (e.g., {'ä', 'ö'} for Finnish)

    Returns:
        Tuple of (is_safe: bool, problematic_chars: list)

    Example:
        >>> text = "Text with em-dash—problem"
        >>> is_safe, issues = validate_text_safety(text)
        >>> print(is_safe)
        False
        >>> print(issues)
        [('—', 'U+2014', 'Em dash')]
    """
    if allow_non_ascii is None:
        allow_non_ascii = set()

    # Characters known to cause issues with T5
    problematic = {
        '\u2011': 'Non-breaking hyphen',
        '\u2010': 'Unicode hyphen',
        '\u202F': 'Narrow no-break space',
        '\u2014': 'Em dash',
        '\u2013': 'En dash',
        '\u2019': 'Right single quotation mark',
        '\u2018': 'Left single quotation mark',
        '\u201C': 'Left double quotation mark',
        '\u201D': 'Right double quotation mark',
    }

    issues = []
    for char, description in problematic.items():
        if char in text and char not in allow_non_ascii:
            issues.append((char, f'U+{ord(char):04X}', description))

    return len(issues) == 0, issues


if __name__ == "__main__":
    # Test normalization
    test_cases = [
        "This is a track—with special characters",
        "Music at 120 BPM in 4/4 time",
        "Features accordion—most notably a bright accordion",
        "The song's uplifting mood",
    ]

    print("=" * 80)
    print("TEXT NORMALIZATION TESTS")
    print("=" * 80)

    for original in test_cases:
        normalized = normalize_music_flamingo_text(original)
        changed = original != normalized

        print(f"\nOriginal:   {original}")
        if changed:
            print(f"Normalized: {normalized}")
            print("✓ Changed")
        else:
            print("✓ No changes needed")

    print("\n" + "=" * 80)
