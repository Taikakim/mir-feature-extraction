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

## timbral_models - NumPy 2.x API Compatibility Fix

**Date:** 2026-01-19
**Repository:** https://github.com/AudioCommons/timbral_models
**Local Path:** `/home/kim/Projects/mir/repos/repos/timbral_models/`
**NumPy Version:** 2.3.5+ (also affects 2.4.x)

### Issue

NumPy 2.0 deprecated `numpy.lib.pad()` in favor of `numpy.pad()`. The deprecated path was removed in NumPy 2.x, causing failures:

```
AttributeError: module 'numpy.lib' has no attribute 'pad'
```

**Affected Features:**
- Roughness (uses `np.lib.pad`)
- Hardness (uses `np.lib.pad`)

### Root Cause

NumPy API change:
```python
# Old style (NumPy 1.x):
np.lib.pad(array, pad_width, mode, **kwargs)

# New style (NumPy 2.x):
np.pad(array, pad_width, mode, **kwargs)
```

### Changes Made

#### File 1: `timbral_models/Timbral_Roughness.py`

**Line 142 (approximately):**
```python
# Before:
audio_samples = np.lib.pad(audio_samples, (512, 0), 'constant', constant_values=(0.0, 0.0))

# After:
audio_samples = np.pad(audio_samples, (512, 0), 'constant', constant_values=(0.0, 0.0))
```

#### File 2: `timbral_models/Timbral_Hardness.py`

**Line 70 (approximately):**
```python
# Before:
audio_samples = np.lib.pad(audio_samples, (nperseg+1, 0), 'constant', constant_values=(0.0, 0.0))

# After:
audio_samples = np.pad(audio_samples, (nperseg+1, 0), 'constant', constant_values=(0.0, 0.0))
```

### Quick Fix Command

```bash
sed -i 's/np\.lib\.pad/np.pad/g' \
  /home/kim/Projects/mir/repos/repos/timbral_models/timbral_models/Timbral_Roughness.py \
  /home/kim/Projects/mir/repos/repos/timbral_models/timbral_models/Timbral_Hardness.py
```

### Impact

- ✅ Fixes roughness and hardness feature extraction on NumPy 2.x
- ✅ Backward compatible with NumPy 1.x (`np.pad` exists in both)
- ✅ Forward compatible with NumPy 2.x+
- ✅ No side effects - pure API path change

### Testing

After applying fix, run:
```bash
python -c "
import sys; sys.path.insert(0, 'src')
from timbral.audio_commons import analyze_all_timbral_features
r = analyze_all_timbral_features('test_data/monkey_island_2_-_theme_(roland_mt-32)/full_mix.mp3')
print(f'Extracted {len(r)}/8 timbral features')
"
```

Expected: All 8 timbral features should extract successfully.

### Upstream Status

- [ ] Issue reported to AudioCommons/timbral_models
- [ ] Pull request submitted
- [ ] Accepted upstream

---

