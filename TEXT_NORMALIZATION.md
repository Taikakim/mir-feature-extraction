# Music Flamingo Text Normalization

## Problem

Music Flamingo generates rich, natural-language descriptions that contain special Unicode characters:
- **Non-breaking hyphens** (U+2011): `‑`
- **Em-dashes** (U+2014): `—`
- **En-dashes** (U+2013): `–`
- **Narrow no-break spaces** (U+202F): ` `
- **Curly quotes** (U+2018, U+2019, U+201C, U+201D): `'` `'` `"` `"`

These characters **cause issues with T5 tokenization** used by Stable Audio Tools and other downstream processing.

---

## Solution

All Music Flamingo output is automatically normalized before being saved to `.INFO` files.

### Normalization Rules

| Unicode Character | Code | Replacement | Example |
|-------------------|------|-------------|---------|
| Non-breaking hyphen | U+2011 | `-` | `folk‑pop` → `folk-pop` |
| Em-dash | U+2014 | `--` | `track—with` → `track--with` |
| En-dash | U+2013 | `-` | `120–140 BPM` → `120-140 BPM` |
| Narrow no-break space | U+202F | ` ` | `120 BPM` → `120 BPM` |
| Curly single quotes | U+2018, U+2019 | `'` | `song's` → `song's` |
| Curly double quotes | U+201C, U+201D | `"` | `"text"` → `"text"` |

Plus **NFKC normalization** for any other compatibility characters.

---

## Implementation

### Automatic (Recommended)

Music Flamingo automatically normalizes all output:

```python
from classification.music_flamingo_transformers import MusicFlamingoTransformers

analyzer = MusicFlamingoTransformers(
    model_id="nvidia/music-flamingo-hf",
    use_flash_attention=True,
)

# Output is automatically normalized
description = analyzer.analyze(audio_path, prompt_type='full')
# description is now T5-safe
```

### Manual (if needed)

```python
from core.text_utils import normalize_music_flamingo_text

# Normalize any text
raw_text = "This is a track—with special characters"
clean_text = normalize_music_flamingo_text(raw_text)
# clean_text = "This is a track--with special characters"
```

### Validation

```python
from core.text_utils import validate_text_safety

text = "Some text to check"
is_safe, issues = validate_text_safety(text, allow_non_ascii={'ä', 'ö'})

if not is_safe:
    for char, code, description in issues:
        print(f"Found {description} ({code}): {char}")
```

---

## Allowed Characters

### Safe Non-ASCII

- **Finnish letters**: `ä`, `ö`, `Ä`, `Ö` (and other legitimate Unicode letters)
- **Standard punctuation**: `.`, `,`, `!`, `?`, `;`, `:`
- **Standard quotes**: `'`, `"`
- **Standard hyphens**: `-`

### Replaced Characters

All special Unicode typography is replaced with ASCII equivalents to ensure maximum compatibility.

---

## Examples

### Before Normalization

```
This track is an upbeat Finnish folk‑pop piece that blends traditional
folk instrumentation—most notably a bright accordion—with contemporary
pop sensibilities. At a steady 120 BPM in 4/4 time and rooted in C major,
the song's harmonic landscape features Fm7 and Fm6 chords.
```

### After Normalization

```
This track is an upbeat Finnish folk-pop piece that blends traditional
folk instrumentation--most notably a bright accordion--with contemporary
pop sensibilities. At a steady 120 BPM in 4/4 time and rooted in C major,
the song's harmonic landscape features Fm7 and Fm6 chords.
```

**Changes**:
- `folk‑pop` → `folk-pop` (non-breaking hyphen to hyphen)
- `instrumentation—most` → `instrumentation--most` (em-dash to double hyphen)
- `accordion—with` → `accordion--with` (em-dash to double hyphen)
- `song's` → `song's` (curly apostrophe to straight apostrophe)

---

## Verification

To verify all saved Music Flamingo text is normalized:

```bash
cd /home/kim/Projects/mir
python3 src/core/text_utils.py
```

Or check a specific INFO file:

```python
import json
from pathlib import Path
from core.text_utils import validate_text_safety

info_path = Path('output/Track Name/Track Name.INFO')
with open(info_path) as f:
    data = json.load(f)

for key, value in data.items():
    if key.startswith('music_flamingo_'):
        is_safe, issues = validate_text_safety(value, allow_non_ascii={'ä', 'ö'})
        if is_safe:
            print(f"✓ {key}: Safe")
        else:
            print(f"✗ {key}: Found {len(issues)} issues")
```

---

## Benefits

1. ✅ **T5 Compatibility**: Text can be safely tokenized by T5 models
2. ✅ **Stable Audio Tools**: Compatible with Stable Audio Tools conditioning
3. ✅ **ASCII-Safe**: Easy to process, search, and index
4. ✅ **Automatic**: No manual intervention needed
5. ✅ **Reversible**: Original meaning preserved, only formatting changed

---

## Files

- **Core function**: `src/core/text_utils.py`
- **Integration**: `src/classification/music_flamingo_transformers.py`
- **Tests**: Run `python src/core/text_utils.py`

---

## Technical Details

### NFKC Normalization

Uses Unicode NFKC (Compatibility Decomposition, Canonical Composition):
- Decomposes compatibility characters (ligatures, presentation forms)
- Recomposes to canonical form
- Results in standard, widely-compatible Unicode

### Character Replacements

Applied **before** NFKC normalization to ensure specific replacements (e.g., em-dash → `--` instead of single `-`).

---

**Last Updated**: 2026-01-18
**Status**: Production Ready ✅
**Tested**: All existing Music Flamingo outputs normalized and verified
