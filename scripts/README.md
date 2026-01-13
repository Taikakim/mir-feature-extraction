# MIR Scripts

This directory contains utility scripts for the MIR project.

## Download Essentia Models

The `download_essentia_models.sh` script downloads all required Essentia TensorFlow models.

### Usage

```bash
# Make script executable (first time only)
chmod +x scripts/download_essentia_models.sh

# Download all models
./scripts/download_essentia_models.sh
```

### Models Downloaded

The script downloads the following models to `~/Projects/mir/models/essentia/`:

**VGGish-based classifiers:**
- `danceability-vggish-audioset-1.pb` - Danceability prediction
- `tonal_atonal-vggish-audioset-1.pb` - Tonality/atonality classification
- `voice_instrumental-vggish-audioset-1.pb` - Voice vs instrumental detection
- `gender-vggish-audioset-1.pb` - Vocal gender classification (legacy)

**Effnet-based classifiers:**
- `discogs-effnet-bs64-1.pb` - Feature extractor (embeddings)
- `genre_discogs400-discogs-effnet-1.pb` - Genre classification (400 classes)
- `mtg_jamendo_moodtheme-discogs-effnet-1.pb` - Mood/theme classification
- `mtg_jamendo_instrument-discogs-effnet-1.pb` - Instrument detection

**JSON metadata files:**
- `.json` files for each model (class labels, configuration)

### Model Location

Models are stored in: `~/Projects/mir/models/essentia/`

The code will automatically search for models in:
1. `$ESSENTIA_MODELS_DIR` environment variable (if set)
2. `~/Projects/mir/models/essentia/` (default)
3. Current directory (fallback)

### Custom Model Location

To use a custom model directory:

```bash
export ESSENTIA_MODELS_DIR="/path/to/your/models"
```

Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent.

### Download Size

Total download size: ~250 MB

The script uses `curl -C -` which supports resuming interrupted downloads.

### Manual Download

If the script fails, you can manually download models from:
https://essentia.upf.edu/models/

Place them in `~/Projects/mir/models/essentia/`

### Troubleshooting

**Error: "Model file not found"**
- Run the download script: `./scripts/download_essentia_models.sh`
- Check that models exist in `~/Projects/mir/models/essentia/`
- Set `ESSENTIA_MODELS_DIR` if using a custom location

**Error: "curl command not found"**
- Install curl: `sudo apt-get install curl` (Ubuntu/Debian)
- Or use wget instead (modify script)

**Download interrupted**
- Re-run the script - it will resume from where it stopped
