#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

LATENT_DIR = Path("/run/media/kim/Lehto/goa-small")
INFO_DIR = Path("/run/media/kim/Mantu/ai-music/Goa_Separated_crops")
OUT_DIR = Path(__file__).resolve().parent / "latent_shape_explorer" / "feature_posters"
ARTIFACT_TMP = Path("/tmp/feature_posters")

TARGET_FEATURES = [
    "bpm", "danceability", "brightness", "roughness", "hardness", "depth", 
    "booming", "reverberation", "sharpness", "warmth", "atonality", 
    "voice_probability", "instrumental_probability", "female_probability", 
    "male_probability", "content_enjoyment", "production_complexity", 
    "production_quality", "syncopation", "rhythmic_complexity", 
    "rhythmic_evenness", "rms_energy_bass", "rms_energy_body", 
    "rms_energy_mid", "rms_energy_air", "spectral_flatness", "spectral_flux"
]

def load_data():
    print("Gathering data...")
    feature_data = defaultdict(list)
    latent_data = []

    for track_dir in LATENT_DIR.iterdir():
        if not track_dir.is_dir(): continue
        t_name = track_dir.name
        
        npys = list(track_dir.glob("*.npy"))
        for npy in npys:
            if any(npy.stem.endswith(s) for s in ["_bass", "_drums", "_other", "_vocals"]):
                continue
                
            info_file = INFO_DIR / t_name / (npy.stem + ".INFO")
            if not info_file.exists(): 
                continue
                
            try:
                latent = np.load(str(npy)).astype(np.float32)
                latent_mean = np.mean(latent, axis=1) 
                
                with open(info_file, 'r') as f:
                    info = json.load(f)
                
                feats = info.get('original_features', info)
                
                valid = True
                extracted_feats = {}
                for tf in TARGET_FEATURES:
                    val = feats.get(tf)
                    if val is None or not isinstance(val, (int, float)):
                        valid = False
                        break
                    extracted_feats[tf] = val
                
                if not valid: continue
                
                latent_data.append(latent_mean)
                for tf in TARGET_FEATURES:
                    feature_data[tf].append(extracted_feats[tf])
                    
            except Exception as e:
                pass

    return np.array(latent_data), feature_data

def plot_8x8_matrix(corrs, title, filepath):
    # Reshape 64 vector to 8x8
    matrix = corrs.reshape((8, 8))
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
    
    # Add grid
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    # Hide major ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    
    # Add values text
    for i in range(8):
        for j in range(8):
            val = matrix[i, j]
            # White text if dark background, black if light
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
            
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    X_latents, feature_data = load_data()
    n_samples = X_latents.shape[0]
    if n_samples == 0:
        print("No valid paired data found.")
        sys.exit(1)
        
    print(f"Loaded {n_samples} samples.")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_TMP, exist_ok=True)
    
    # 1. Feature -> Latent Channel Correlations
    print("Generating Feature Posters...")
    
    # Build a Feature Matrix
    Y_features = np.zeros((n_samples, len(TARGET_FEATURES)))
    
    for idx, tf in enumerate(TARGET_FEATURES):
        Y = np.array(feature_data[tf])
        Y_features[:, idx] = Y
        
        corrs = np.zeros(64)
        for c in range(64):
            try:
                r, p = pearsonr(X_latents[:, c], Y)
                if np.isnan(r): r = 0.0
            except:
                r = 0.0
            corrs[c] = r
            
        out_path = OUT_DIR / f"{tf}_poster.png"
        plot_8x8_matrix(corrs, f"Latent Correl. - {tf}", out_path)
        # Also save to tmp for artifact insertion
        plot_8x8_matrix(corrs, f"Latent Correl. - {tf}", ARTIFACT_TMP / f"{tf}_poster.png")
        
    # 2. PCA of the MIR Features
    print("Running PCA on MIR Features...")
    scaler = StandardScaler()
    Y_scaled = scaler.fit_transform(Y_features)
    
    pca = PCA(n_components=5) # Compute top 5
    pca.fit(Y_scaled)
    
    Y_pca = pca.transform(Y_scaled) # [n_samples, 5]
    
    # Let's save the PCA components (how features map to Feature-PCA)
    print("\nMIR Feature PCA Variance Ratios:", pca.explained_variance_ratio_)
    for i in range(3):
        comp = pca.components_[i]
        top_pos = np.argsort(comp)[::-1][:3]
        top_neg = np.argsort(comp)[:3]
        
        pos_names = [f"{TARGET_FEATURES[idx]} (+{comp[idx]:.2f})" for idx in top_pos]
        neg_names = [f"{TARGET_FEATURES[idx]} ({comp[idx]:.2f})" for idx in top_neg]
        print(f"\nFeature PCA {i+1}:")
        print("  Positive drivers:", ", ".join(pos_names))
        print("  Negative drivers:", ", ".join(neg_names))
        
        # Correlate this Feature-PCA axis back to the 64 Latent Channels
        corrs = np.zeros(64)
        for c in range(64):
            try:
                r, p = pearsonr(X_latents[:, c], Y_pca[:, i])
                if np.isnan(r): r = 0.0
            except:
                r = 0.0
            corrs[c] = r
            
        out_path = OUT_DIR / f"FeaturePCA{i+1}_poster.png"
        plot_8x8_matrix(corrs, f"Latent Correl. - MIR Feature PCA {i+1}", out_path)
        plot_8x8_matrix(corrs, f"Latent Correl. - MIR Feature PCA {i+1}", ARTIFACT_TMP / f"FeaturePCA{i+1}_poster.png")

    print("\nVisualization complete! Posters saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
