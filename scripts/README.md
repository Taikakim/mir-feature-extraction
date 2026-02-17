# MIR Scripts

Utility scripts for project setup, model downloads, and data verification.

## Setup

### `setup_external_repos.sh`

Clones, pins, patches, and installs all 7 external repositories:

1. **timbral_models** — editable install + librosa 0.11.0 patches
2. **BS-RoFormer** — editable install
3. **ADTOF-pytorch** — editable install
4. **drumsep** — clone only (direct path access, model downloaded separately)
5. **madmom** — pip install (Cython build)
6. **llama.cpp** — cmake build with auto-detected GPU backend (ROCm/CUDA/CPU)
7. **Qwen2.5-Omni** — clone only (patched modeling file for captioning benchmark)

```bash
./scripts/setup_external_repos.sh              # Full setup including llama.cpp build
./scripts/setup_external_repos.sh --skip-build  # Skip llama.cpp build
```

### `download_essentia_models.py` / `download_essentia_models.sh`

Downloads Essentia TensorFlow classification models (~250 MB total) to `models/essentia/`.

Includes: danceability, tonal/atonal, voice/instrumental, gender, genre (400 classes), mood/theme, instrument detection.

```bash
python scripts/download_essentia_models.py
# or
./scripts/download_essentia_models.sh
```

## Verification

### `verify_features.py`

Checks organized audio folders for expected feature keys in `.INFO` files.

```bash
python scripts/verify_features.py /path/to/data --show-missing
python scripts/verify_features.py /path/to/data -v   # verbose (show complete folders too)
```

### `verify_flamingo_config.py`

Smoke test for Music Flamingo config loading and analyzer interface.

```bash
python scripts/verify_flamingo_config.py
```
