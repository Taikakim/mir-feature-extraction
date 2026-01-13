#!/bin/bash
# Download Essentia Models for MIR Project
# These models are required for danceability, atonality, voice/instrumental,
# gender, genre, mood, and instrument classification

set -e  # Exit on error

# Create models directory
MODELS_DIR="${HOME}/Projects/mir/models/essentia"
mkdir -p "${MODELS_DIR}"

echo "Downloading Essentia models to: ${MODELS_DIR}"
echo "This may take a while..."
echo ""

cd "${MODELS_DIR}"

# Danceability models
echo "Downloading danceability models..."
curl -C - -O https://essentia.upf.edu/models/classifiers/danceability/danceability-vggish-audioset-1.pb
curl -C - -O https://essentia.upf.edu/models/classifiers/danceability/danceability-vggish-audioset-1.json

# Tonal/Atonal models
echo "Downloading tonal/atonal models..."
curl -C - -O https://essentia.upf.edu/models/classifiers/tonal_atonal/tonal_atonal-vggish-audioset-1.pb
curl -C - -O https://essentia.upf.edu/models/classifiers/tonal_atonal/tonal_atonal-vggish-audioset-1.json

# Voice/Instrumental models
echo "Downloading voice/instrumental models..."
curl -C - -O https://essentia.upf.edu/models/classifiers/voice_instrumental/voice_instrumental-vggish-audioset-1.pb
curl -C - -O https://essentia.upf.edu/models/classifiers/voice_instrumental/voice_instrumental-vggish-audioset-1.json

# Gender models (legacy)
echo "Downloading gender models..."
curl -C - -O https://essentia.upf.edu/models/legacy/classifiers/gender/gender-vggish-audioset-1.pb
curl -C - -O https://essentia.upf.edu/models/legacy/classifiers/gender/gender-vggish-audioset-1.json

# Genre, mood, and instrument models (require Effnet embeddings)
echo "Downloading genre/mood/instrument models..."
curl -C - -O https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb
curl -C - -O https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb
curl -C - -O https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb
curl -C - -O https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb

echo ""
echo "All models downloaded successfully!"
echo ""
echo "Models location: ${MODELS_DIR}"
echo ""
echo "Model files:"
ls -lh "${MODELS_DIR}"
echo ""
echo "To use these models, set the environment variable:"
echo "export ESSENTIA_MODELS_DIR=\"${MODELS_DIR}\""
echo ""
echo "Or add this to your ~/.bashrc or ~/.zshrc"
