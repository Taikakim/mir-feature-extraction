"""
Crops Processing Module

Provides tools for processing audio crops (segments created by create_training_crops.py).

Crops have a different structure than organized folders:
- Crops are named: TrackName_0.flac, TrackName_1.flac, etc.
- Stems are prefixed: TrackName_0_drums.flac, TrackName_0_bass.flac, etc.
- Each crop has its own .INFO file: TrackName_0.INFO

Modules:
- demucs_sep: Stem separation for crops (fallback if stems don't exist)
- feature_extractor: Feature extraction for individual crops
- pipeline: Master orchestrator for full crop processing

Usage:
    python -m crops.pipeline /path/to/crops/TrackName/ -v
    python -m crops.pipeline /path/to/crops/ --batch
"""

from .demucs_sep import separate_crop_stems, batch_separate_crop_stems
from .feature_extractor import CropFeatureExtractor
from .pipeline import CropsPipeline, CropsPipelineConfig

__all__ = [
    'separate_crop_stems',
    'batch_separate_crop_stems',
    'CropFeatureExtractor',
    'CropsPipeline',
    'CropsPipelineConfig',
]
