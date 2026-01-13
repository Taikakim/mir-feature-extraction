# External Repository Patches

This document tracks all modifications made to externally downloaded repositories to fix compatibility issues or bugs.

## General Policy

- Minimal changes only - fix what's broken, don't refactor
- Document the issue, fix, and reasoning
- Include version information
- Note if upstream has been notified

---

## timbral_models - Librosa API Compatibility Fix

**Date:** 2026-01-13
**Repository:** https://github.com/AudioCommons/timbral_models
**Local Path:** `/home/kim/Projects/mir/repos/repos/timbral_models/`
**Librosa Version:** 0.11.0

### Issue

Librosa 0.11.0 changed several functions to use keyword-only arguments (using `*` in function signature). The timbral_models library was calling these functions with positional arguments, causing failures:

```
TypeError: onset_detect() takes 0 positional arguments but 2 positional arguments
(and 2 keyword-only arguments) were given
```

**Affected Features:**
- Hardness (uses `onset_detect` and `onset_strength`)
- Depth (uses `onset_detect`)
- Warmth (uses `onset_detect`)

### Root Cause

Librosa function signatures changed:
```python
# Old style (pre-0.11.0):
def onset_detect(y, sr=22050, ...):  # Positional args allowed

# New style (0.11.0+):
def onset_detect(*, y=None, sr=22050, ...):  # Only keyword args (note the *)
```

Similar changes affected:
- `librosa.onset.onset_detect()` - All parameters KEYWORD_ONLY
- `librosa.onset.onset_strength()` - All parameters KEYWORD_ONLY
- `librosa.resample()` - Only first parameter allows positional, others KEYWORD_ONLY

### Changes Made

#### File 1: `timbral_models/timbral_util.py`

**Line 642:**
```python
# Before:
onsets = librosa.onset.onset_detect(audio_samples, fs, backtrack=True, units='samples')

# After:
onsets = librosa.onset.onset_detect(y=audio_samples, sr=fs, backtrack=True, units='samples')
```

**Line 750:**
```python
# Before:
onset_strength = librosa.onset.onset_strength(audio_samples, fs)

# After:
onset_strength = librosa.onset.onset_strength(y=audio_samples, sr=fs)
```

**Line 1813:**
```python
# Before:
audio_samples = librosa.core.resample(audio_samples, fs, lowest_fs)

# After:
audio_samples = librosa.resample(y=audio_samples, orig_sr=fs, target_sr=lowest_fs)
```
*Note: Also updated to use `librosa.resample` directly instead of `librosa.core.resample` (deprecated path)*

#### File 2: `timbral_models/Timbral_Hardness.py`

**Line 88:**
```python
# Before:
onset_strength = librosa.onset.onset_strength(audio_samples, fs)

# After:
onset_strength = librosa.onset.onset_strength(y=audio_samples, sr=fs)
```

### Impact

- ✅ Fixes hardness, depth, and warmth feature extraction
- ✅ Backward compatible with older librosa versions (keyword args work in all versions)
- ✅ Forward compatible with librosa 0.11.0+
- ✅ No side effects - pure syntax changes

### Testing

After applying fix, run:
```bash
python src/timbral/audio_commons.py test_data --batch
```

Expected: All 8 timbral features should extract successfully without `onset_detect()` errors.

### Upstream Status

- [ ] Issue reported to AudioCommons/timbral_models
- [ ] Pull request submitted
- [ ] Accepted upstream

**Note:** Consider checking if upstream has fixed this and updating to newer version.

---

