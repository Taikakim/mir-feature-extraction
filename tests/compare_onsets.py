
import numpy as np
import scipy.signal
import librosa
import sys
from pathlib import Path

# Mock step_3 logic from pipeline
def rms_segmentation(y, sr):
    TARGET_FPS = 100
    hop_length = int(sr / TARGET_FPS)
    frame_length = 2048 # Increased window
    
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    
    # Simple interpolation to grid
    num_frames = int(len(y) / sr * TARGET_FPS) + 1
    f = scipy.interpolate.interp1d(np.arange(len(rms)), rms, kind='linear', 
                                 bounds_error=False, fill_value=0)
    rms_energy = f(np.arange(num_frames))
    
    # Thresholds
    PEAK_THRESH = 0.05
    START_THRESH = 0.001
    END_THRESH = 0.01
    
    if rms_energy.max() > 0:
        norm_rms = rms_energy / rms_energy.max()
    else:
        norm_rms = rms_energy

    peaks, _ = scipy.signal.find_peaks(norm_rms, height=PEAK_THRESH, distance=10)
    
    segments = []
    covered_mask = np.zeros_like(norm_rms, dtype=bool)
    last_offset_idx = -1
    
    for p in peaks:
        if covered_mask[p]:
            continue
            
        onset_idx = p
        while onset_idx > 0 and onset_idx > last_offset_idx:
            if norm_rms[onset_idx] < START_THRESH:
                break
            onset_idx -= 1
        onset_idx = max(onset_idx, last_offset_idx + 1)
        
        offset_idx = p
        while offset_idx < len(norm_rms) - 1:
            if norm_rms[offset_idx] < END_THRESH:
                break
            offset_idx += 1
        
        if offset_idx - onset_idx >= 3:
            segments.append(onset_idx / TARGET_FPS)
            covered_mask[onset_idx:offset_idx] = True
            last_offset_idx = offset_idx
            
    return segments


def librosa_superflux(y, sr):
    # Standard Spectral Flux (SuperFlux implementation in librosa)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True)
    return onsets

def librosa_complex(y, sr):
    # Complex Domain Onset Detection (good for soft onsets)
    S = librosa.stft(y)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True, energy=None)
    # Actually complex domain needs specific function call or parameter?
    # Librosa default uses spectral flux.
    # To use complex phase, we compute onset_strength with aggregate?
    # Let's try standard onset_detect first.
    return onsets

def tuned_rms(y, sr):
    # Current logic with lower thresholds?
    # Or adaptive thresholding.
    return rms_segmentation(y, sr) # Baseline

def main():
    path = "/home/kim/Projects/mir/test_data/0112_Hallucinogen_-_Trancespotter_31_bass.flac"
    print(f"Analyzing {path}")
    
    y, sr = librosa.load(path, sr=None, mono=True)
    
    # Run Baseline
    rms_onsets = rms_segmentation(y, sr)
    print(f"Baseline RMS Segments: {len(rms_onsets)}")
    
    # Run Librosa SuperFlux (SOTA for general music)
    sf_onsets = librosa_superflux(y, sr)
    print(f"Librosa SuperFlux Onsets: {len(sf_onsets)}")
    
    # Run Librosa with pre_max, post_max, pre_avg, post_avg tuning?
    # Try more sensitive settings
    sensitive_onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True,
                                                  pre_max=30, post_max=30, pre_avg=100, post_avg=100, delta=0.05, wait=10)
    print(f"Librosa Sensitive Onsets: {len(sensitive_onsets)}")

if __name__ == "__main__":
    main()
