#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr

# Configuration ----------------------------------------------------
LATENT_DIR = Path("/run/media/kim/Lehto/goa-small")
INFO_DIR = Path("/run/media/kim/Mantu/ai-music/Goa_Separated_crops")
OUT_JSON = Path(__file__).resolve().parent.parent / "models" / "latent_correlations.json"

# Features to analyze
TARGET_FEATURES = [
    "bpm", "danceability", "brightness", "roughness", "hardness", "depth", 
    "booming", "reverberation", "sharpness", "warmth", "atonality", 
    "voice_probability", "instrumental_probability", "female_probability", 
    "male_probability", "content_enjoyment", "production_complexity", 
    "production_quality", "syncopation", "rhythmic_complexity", 
    "rhythmic_evenness", "rms_energy_bass", "rms_energy_body", 
    "rms_energy_mid", "rms_energy_air", "spectral_flatness", "spectral_flux"
]

def analyze_correlations():
    print("Gathering data...")
    feature_data = defaultdict(list)  # {feature_name: [val1, val2, ...]}
    latent_data = [] # [[64-dim mean vector], ...]

    # Iterate over track folders
    for track_dir in LATENT_DIR.iterdir():
        if not track_dir.is_dir(): continue
        t_name = track_dir.name
        
        # Load NPYs
        npys = list(track_dir.glob("*.npy"))
        for npy in npys:
            # We only look at fullmix crops (no stems) for general correlation mapping
            if any(npy.stem.endswith(s) for s in ["_bass", "_drums", "_other", "_vocals"]):
                continue
                
            info_file = INFO_DIR / t_name / (npy.stem + ".INFO")
            if not info_file.exists(): 
                continue
                
            try:
                # Load Latent (shape [64, T])
                latent = np.load(str(npy)).astype(np.float32)
                # Take the mean across Time to get a single 64-dim vector for this crop
                latent_mean = np.mean(latent, axis=1) 
                
                # Load Info
                with open(info_file, 'r') as f:
                    info = json.load(f)
                
                feats = info.get('original_features', info) # Account for nested or flat structure
                
                # Verify all target features exist and are numeric
                valid = True
                extracted_feats = {}
                for tf in TARGET_FEATURES:
                    val = feats.get(tf)
                    if val is None or not isinstance(val, (int, float)):
                        valid = False
                        break
                    extracted_feats[tf] = val
                
                if not valid: continue
                
                # Store
                latent_data.append(latent_mean)
                for tf in TARGET_FEATURES:
                    feature_data[tf].append(extracted_feats[tf])
                    
            except Exception as e:
                # Corrupted read or shape mismatch
                pass

    if not latent_data:
        print("No valid paired data found.")
        sys.exit(1)

    X = np.array(latent_data) # [num_samples, 64]
    print(f"Loaded {X.shape[0]} samples.")
    
    correlations_out = {}

    for tf in TARGET_FEATURES:
        Y = np.array(feature_data[tf])
        
        # Calculate Pearson correlation for each of the 64 channels against Y
        corrs = []
        for c in range(64):
            # pearsonr returns (statistic, pvalue)
            # If standard deviation is 0, it returns nan.
            try:
                r, p = pearsonr(X[:, c], Y)
                if np.isnan(r): r = 0.0
            except:
                r = 0.0
            corrs.append(r)
        
        corrs = np.array(corrs)
        
        # Top 5 positive drivers (highest correlation)
        top_pos = np.argsort(corrs)[::-1][:5]
        # Top 5 negative drivers (lowest/most negative correlation)
        top_neg = np.argsort(corrs)[:5]
        
        correlations_out[tf] = {
            "top_positive_channels": top_pos.tolist(),
            "top_positive_scores": corrs[top_pos].tolist(),
            "top_negative_channels": top_neg.tolist(),
            "top_negative_scores": corrs[top_neg].tolist()
        }

    os.makedirs(OUT_JSON.parent, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(correlations_out, f, indent=2)
        
    print(f"Saved latent feature correlations to {OUT_JSON}")

if __name__ == "__main__":
    analyze_correlations()
