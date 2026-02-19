"""
Modular MIDI Extraction Pipeline for Electronic Basslines
Author: Kim
Date: 2026-02-18

This script converts isolated monophonic electronic bass stems into precisely quantized MIDI files.
It uses high-resolution spectral peak detection (n_fft=8192) for robust pitch tracking on isolated
bass stems, real spectral flux for onset detection, RMS-based voicing, and a grid-based segmentation
algorithm that analyzes 16th-note blocks.

Parameters:
  --flux-thresh (default 0.15):
    Sensitivity to "attacks" or spectral changes. Lower values detect softer transients but may
    pick up noise. Increase this if you get too many false notes on non-percussive sounds.

  --rms-thresh (default 0.05):
    Sensitivity to "sustain" relative to the peak energy of the file. A note is detected if
    its average energy exceeds this ratio of the track's max volume. Lower values catch quieter notes.

  --min-vol-ratio (default 0.33):
    Filtering threshold. Any detected note with a peak amplitude less than this ratio (33%) of
    the *average* note volume will be discarded. Use this to remove ghost notes or delay tails.

  --dip-ratio (default 0.5):
    Note merging rigidity. When two consecutive 16th notes have the same pitch, we check the
    energy dip at their boundary. If the dip is *shallower* than this ratio (boundary_energy > peak * ratio),
    they are merged into one long note.
    - Low value (e.g., 0.1): Very strict. Merges only if there's almost no dip (flat sustain).
    - High value (e.g., 0.9): Very loose. Merges even if there's a significant dip.
    - Default (0.5): Merges if the dip is less than 50% of the note's peak energy.

  --release-ratio (default 0.3):
    Note release threshold. A note ends when its energy drops below this ratio of its peak energy.
    - Higher value (e.g., 0.8): Shorter notes (cuts off as soon as energy drops slightly).
    - Lower value (e.g., 0.1): Longer notes (waits until nearly silent).
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.interpolate
import scipy.signal
import librosa
import symusic

# Constants
TARGET_FPS = 100  # 10ms hop
SPECTRAL_N_FFT = 8192  # High frequency resolution for bass (~5.4Hz/bin at 44.1kHz)
MIN_MIDI = 28  # E1 — lowest expected bass note
MAX_MIDI = 60  # C4 — highest expected bass note
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def time_to_frame(t, fps=TARGET_FPS):
    return int(t * fps)

def frame_to_time(f, fps=TARGET_FPS):
    return f / fps


class BassMidiPipeline:
    def __init__(self, audio_path: str, beats_path: str, output_path: str,
                 subdivision: int = 4, flux_thresh: float = 0.15,
                 dip_ratio: float = 0.5, release_ratio: float = 0.3,
                 rms_thresh: float = 0.05, min_vol_ratio: float = 0.33,
                 min_midi: int = MIN_MIDI, max_midi: int = MAX_MIDI,
                 lpf_cutoff: float = 220.0):
        self.audio_path = Path(audio_path)
        self.beats_path = Path(beats_path)
        self.output_path = Path(output_path)
        self.subdivision = subdivision
        self.flux_thresh = flux_thresh
        self.rms_thresh_ratio = rms_thresh
        self.min_vol_ratio = min_vol_ratio
        self.dip_ratio = dip_ratio
        self.release_ratio = release_ratio
        self.min_midi = min_midi
        self.max_midi = max_midi
        self.lpf_cutoff = lpf_cutoff
        self.pitch_quantization_count = 0
        self.chroma_profile = None

        # Load audio once
        print(f"Loading audio: {self.audio_path}")
        self.y, self.sr = librosa.load(self.audio_path, sr=None, mono=True)

        # Apply LPF if requested
        if self.lpf_cutoff > 0:
            print(f"  Applying LPF at {self.lpf_cutoff}Hz (steep 6th order)...")
            sos = scipy.signal.butter(6, self.lpf_cutoff, 'low', fs=self.sr, output='sos')
            self.y = scipy.signal.sosfiltfilt(sos, self.y)

        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        self.num_frames = int(self.duration * TARGET_FPS) + 1
        self.hop_length = int(self.sr / TARGET_FPS)

        # Load beats once
        self.beats = self._load_beats()

        # Load Chroma from INFO (if available)
        self.chroma_profile = self._load_chroma()

        # Time grid for all frame-based data
        self.time_grid = np.arange(self.num_frames) / TARGET_FPS

        # Results (populated by each step)
        self.spectral_pitch_midi = None  # Per-frame MIDI pitch from spectral peaks
        self.rms_energy = None
        self.spectral_flux = None  # Real spectral flux from STFT
        self.rms_voicing = None
        self.quantized_segments = []

        # Pre-compute STFT once (shared by pitch, flux, and per-slot analysis)
        print(f"  Computing STFT (n_fft={SPECTRAL_N_FFT})...")
        self.S = np.abs(librosa.stft(self.y, n_fft=SPECTRAL_N_FFT,
                                      hop_length=self.hop_length))
        self.stft_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=SPECTRAL_N_FFT)

        # Frequency bin masks for bass range
        self.freq_lo = librosa.midi_to_hz(self.min_midi)
        self.freq_hi = librosa.midi_to_hz(self.max_midi)
        self.bass_mask = (self.stft_freqs >= self.freq_lo) & (self.stft_freqs <= self.freq_hi)
        self.bass_freqs = self.stft_freqs[self.bass_mask]
        self.S_bass = self.S[self.bass_mask]

        freq_res = self.sr / SPECTRAL_N_FFT
        print(f"  STFT: {self.S.shape[1]} frames, freq resolution {freq_res:.1f}Hz/bin, "
              f"bass range {self.freq_lo:.0f}-{self.freq_hi:.0f}Hz ({self.bass_mask.sum()} bins)")

    def _load_beats(self) -> np.ndarray:
        """Load beat timestamps from file."""
        if not self.beats_path or not Path(self.beats_path).is_file():
            return np.array([])
        try:
            beats = np.loadtxt(self.beats_path)
            if beats.ndim == 0:
                beats = np.array([float(beats)])
            elif beats.ndim == 2:
                beats = beats[:, 0]
            return beats
        except Exception as e:
            print(f"  Warning: Failed to load beats: {e}")
            return np.array([])

    def _load_chroma(self) -> Optional[np.ndarray]:
        """Load chroma vector from adjacent .INFO file."""
        import json

        candidates = [
            self.audio_path.with_suffix(".INFO"),
            self.audio_path.with_name(self.audio_path.stem.replace("_bass", "") + ".INFO")
        ]

        for c in candidates:
            if c.exists():
                try:
                    with open(c, 'r') as f:
                        data = json.load(f)
                        chroma = np.zeros(12)
                        found = False
                        for i in range(12):
                            k = f"chroma_{i}"
                            if k in data:
                                chroma[i] = data[k]
                                found = True

                        if found:
                            if chroma.sum() > 0:
                                chroma /= chroma.sum()
                            print(f"Loaded Chroma Profile from {c.name}")
                            top_3_idx = np.argsort(chroma)[-3:][::-1]
                            top_3_str = ", ".join([f"{PITCH_CLASSES[i]} ({chroma[i]:.2f})" for i in top_3_idx])
                            print(f"  Top Chroma Classes: {top_3_str}")
                            return chroma
                except Exception as e:
                    print(f"  Warning: Failed to load INFO file {c}: {e}")
        return None

    def run(self):
        print("Step 1: Spectral Peak Pitch Estimation...")
        self.step_1_spectral_pitch()

        print("Step 2: RMS Energy, Spectral Flux & Voicing...")
        self.step_2_energy_and_flux()

        print("Step 3: Grid-Based Analysis & Quantization...")
        self.step_3_grid_analysis()

        if self.pitch_quantization_count > 0:
            print(f"Step 4: Smart Pitch Quantization (Top {self.pitch_quantization_count} notes)...")
            self.step_4_pitch_quantization()

        print("Step 5: High-Performance MIDI Export...")
        self.step_5_export()

        print(f"Done! Saved to {self.output_path}")

    def step_1_spectral_pitch(self):
        """
        Extract per-frame pitch from spectral peaks in the bass range.

        For isolated monophonic bass stems, direct FFT peak detection at n_fft=8192
        (~5.4Hz resolution at 44.1kHz, sub-semitone above ~80Hz) is more reliable than
        probabilistic pitch trackers (pYIN, SwiftF0, Basic Pitch) which introduce octave
        ambiguity and disagree on synthesized bass tones.

        Fully vectorized with parabolic interpolation for sub-bin accuracy.
        """
        n_bins, n_stft_frames = self.S_bass.shape
        freq_step = self.bass_freqs[1] - self.bass_freqs[0] if len(self.bass_freqs) > 1 else 1.0

        # Vectorized peak finding
        peak_bins = np.argmax(self.S_bass, axis=0)  # (n_frames,)
        peak_vals = self.S_bass[peak_bins, np.arange(n_stft_frames)]

        # Silent frames
        valid = peak_vals > 0

        # Parabolic interpolation (vectorized) for interior bins
        interior = valid & (peak_bins > 0) & (peak_bins < n_bins - 1)
        alpha = np.zeros(n_stft_frames)
        beta = np.zeros(n_stft_frames)
        gamma = np.zeros(n_stft_frames)
        idx = np.where(interior)[0]
        alpha[idx] = self.S_bass[peak_bins[idx] - 1, idx]
        beta[idx] = self.S_bass[peak_bins[idx], idx]
        gamma[idx] = self.S_bass[peak_bins[idx] + 1, idx]
        denom = alpha - 2 * beta + gamma
        p = np.zeros(n_stft_frames)
        nonzero_denom = interior & (denom != 0)
        p[nonzero_denom] = 0.5 * (alpha[nonzero_denom] - gamma[nonzero_denom]) / denom[nonzero_denom]

        # Peak frequencies
        peak_freqs = self.bass_freqs[peak_bins] + p * freq_step

        # Convert to MIDI and clamp
        spectral_midi_raw = np.zeros(n_stft_frames)
        pos_freq = valid & (peak_freqs > 0)
        midi_vals = np.zeros(n_stft_frames)
        midi_vals[pos_freq] = librosa.hz_to_midi(peak_freqs[pos_freq])
        in_range = pos_freq & (midi_vals >= self.min_midi) & (midi_vals <= self.max_midi)
        spectral_midi_raw[in_range] = midi_vals[in_range]

        # Interpolate to common time grid
        stft_times = librosa.times_like(self.S_bass, sr=self.sr, hop_length=self.hop_length)
        f_pitch = scipy.interpolate.interp1d(stft_times, spectral_midi_raw,
                                              kind='nearest', bounds_error=False, fill_value=0)
        self.spectral_pitch_midi = f_pitch(self.time_grid)

        n_voiced = np.sum(self.spectral_pitch_midi > 0)
        print(f"  Spectral pitch: {n_voiced}/{self.num_frames} frames with valid pitch "
              f"({100*n_voiced/self.num_frames:.1f}%)")

        # Print detected pitch distribution
        valid_midi = self.spectral_pitch_midi[self.spectral_pitch_midi > 0]
        if len(valid_midi) > 0:
            rounded = np.round(valid_midi).astype(int)
            counts = Counter(rounded)
            top_5 = counts.most_common(5)
            pitch_str = ", ".join([
                f"{PITCH_CLASSES[p % 12]}{p // 12 - 1}({c})"
                for p, c in top_5
            ])
            print(f"  Top pitches: {pitch_str}")

    def step_2_energy_and_flux(self):
        """
        Compute RMS energy envelope, real spectral flux, and voicing.

        Spectral flux = half-wave rectified sum of positive spectral differences
        between consecutive STFT frames. Detects re-articulations even when RMS
        stays constant (same pitch, same volume, new attack).
        """
        # --- RMS energy ---
        frame_length = 2048
        rms = librosa.feature.rms(y=self.y, frame_length=frame_length,
                                  hop_length=self.hop_length, center=True)[0]
        rms_times = librosa.times_like(rms, sr=self.sr, hop_length=self.hop_length)
        f = scipy.interpolate.interp1d(rms_times, rms, kind='linear',
                                       bounds_error=False, fill_value=0)
        self.rms_energy = f(self.time_grid)

        # RMS-based voicing: adaptive threshold at 5% of median non-zero energy
        nonzero_rms = self.rms_energy[self.rms_energy > 0]
        if len(nonzero_rms) > 0:
            voicing_thresh = 0.05 * np.median(nonzero_rms)
        else:
            voicing_thresh = 0.0
        self.rms_voicing = self.rms_energy > voicing_thresh
        self.rms_energy[~self.rms_voicing] = 0.0

        # --- Real spectral flux from STFT ---
        # Use bass-range spectrogram for flux (ignores irrelevant high-freq noise)
        diff = np.diff(self.S_bass, axis=1)
        flux_frames = np.sum(np.maximum(0, diff), axis=0)  # Half-wave rectified
        flux_frames = np.pad(flux_frames, (1, 0))  # Align with frame 0

        # Normalize to [0, 1]
        if flux_frames.max() > 0:
            flux_frames = flux_frames / flux_frames.max()

        # Interpolate to time grid
        flux_times = librosa.times_like(flux_frames, sr=self.sr, hop_length=self.hop_length)
        f_flux = scipy.interpolate.interp1d(flux_times, flux_frames, kind='linear',
                                            bounds_error=False, fill_value=0)
        self.spectral_flux = f_flux(self.time_grid)

        print(f"  RMS voicing: {np.sum(self.rms_voicing)}/{self.num_frames} frames "
              f"(thresh={voicing_thresh:.6f})")
        print(f"  Spectral flux: mean={np.mean(self.spectral_flux):.4f}, "
              f"max={np.max(self.spectral_flux):.4f}, "
              f"peaks above thresh: {np.sum(self.spectral_flux > self.flux_thresh)}")

    def _get_slot_pitch(self, t_start: float, t_end: float) -> int:
        """
        Get MIDI pitch for a time slot using spectral peak detection.

        Primary: per-frame spectral pitch (median in MIDI domain within slot).
        Fallback: direct FFT of the slot's audio segment for maximum resolution.
        """
        f_start = time_to_frame(t_start)
        f_end = time_to_frame(t_end)
        slot_pitches = self.spectral_pitch_midi[f_start:f_end]
        valid = slot_pitches[slot_pitches > 0]

        if len(valid) >= 2:
            # Median in MIDI domain (robust to octave outliers)
            return int(round(np.median(valid)))

        # Fallback: direct FFT of the slot segment
        s_start = int(t_start * self.sr)
        s_end = min(int(t_end * self.sr), len(self.y))
        segment = self.y[s_start:s_end]

        if len(segment) < 1024:
            return int(round(valid[0])) if len(valid) == 1 else 0

        # Zero-pad to at least n_fft for resolution
        n_fft = max(SPECTRAL_N_FFT, len(segment))
        S_seg = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sr)

        mask = (freqs >= self.freq_lo) & (freqs <= self.freq_hi)
        if not mask.any():
            return 0

        peak_freq = freqs[mask][np.argmax(S_seg[mask])]
        if peak_freq > 0:
            midi_val = librosa.hz_to_midi(peak_freq)
            if self.min_midi <= midi_val <= self.max_midi:
                return int(round(midi_val))
        return 0

    def step_3_grid_analysis(self):
        """
        Grid-Based Segmentation: Analyze audio in 16th-note blocks relative to beat grid.

        Algorithm:
        1. Establish 16th-note grid from beats.
        2. For each grid slot:
           - Use spectral flux for onset detection (not RMS derivative).
           - Use RMS energy for sustain detection.
           - Get pitch from spectral peak (not ensemble mean in Hz).
        3. Tie/Merge logic:
           - Merge consecutive slots if pitch matches AND no spectral flux peak
             at boundary AND no significant energy dip.
        """
        if len(self.beats) < 2:
            print("  No beat grid available. Falling back to simple RMS segmentation.")
            return

        print("  Running Grid-Based Analysis (16th notes, spectral flux + peak pitch)...")

        # 1. Generate 16th note grid
        grid_points = []
        for i in range(len(self.beats) - 1):
            start = self.beats[i]
            end = self.beats[i + 1]
            dur = end - start
            for j in range(self.subdivision):
                grid_points.append(start + j * dur / self.subdivision)
        grid_points.append(self.beats[-1])
        grid = np.array(grid_points)

        # Interpolators for continuous-time queries
        f_rms = scipy.interpolate.interp1d(self.time_grid, self.rms_energy,
                                           kind='linear', bounds_error=False, fill_value=0)
        f_flux = scipy.interpolate.interp1d(self.time_grid, self.spectral_flux,
                                            kind='linear', bounds_error=False, fill_value=0)

        # Thresholds
        peak_track_rms = self.rms_energy.max() if self.rms_energy.max() > 0 else 1.0
        RMS_THRESH = self.rms_thresh_ratio * peak_track_rms
        FLUX_THRESH = self.flux_thresh
        DIP_RATIO = self.dip_ratio

        # 2. Analyze each slot
        raw_notes = []

        for i in range(len(grid) - 1):
            t_start = grid[i]
            t_end = grid[i + 1]

            # Sample features across slot
            t_eval = np.linspace(t_start, t_end, 10)
            rms_vals = f_rms(t_eval)
            flux_vals = f_flux(t_eval)

            peak_rms = np.max(rms_vals)
            peak_flux = np.max(flux_vals)
            mean_rms = np.mean(rms_vals)

            # Detection: spectral flux onset OR sustained energy
            is_onset = peak_flux > FLUX_THRESH
            is_sustaining = mean_rms > RMS_THRESH

            if is_onset or is_sustaining:
                midi_p = self._get_slot_pitch(t_start, t_end)

                if midi_p > 0:
                    # Velocity normalized to track peak
                    velocity = int(np.clip(peak_rms / peak_track_rms * 127, 1, 127))
                    raw_notes.append({
                        "onset_time": t_start,
                        "offset_time": t_end,
                        "pitch": midi_p,
                        "velocity": velocity,
                        "peak_rms": peak_rms,
                        "peak_flux": peak_flux,
                    })

        # 3. Volume Filtering
        if raw_notes:
            avg_vol = np.mean([n["peak_rms"] for n in raw_notes])
            min_vol = avg_vol * self.min_vol_ratio
            before = len(raw_notes)
            raw_notes = [n for n in raw_notes if n["peak_rms"] >= min_vol]
            print(f"    Raw: {before} slots, filtered {before - len(raw_notes)} "
                  f"low-vol (<{min_vol:.4f})")
        else:
            print("    Raw: 0 slots.")

        # 4. Tie / Merge Logic (flux-aware)
        merged_notes = []
        if raw_notes:
            current = raw_notes[0]

            for next_note in raw_notes[1:]:
                is_adjacent = np.isclose(current["offset_time"], next_note["onset_time"])
                is_same_pitch = (current["pitch"] == next_note["pitch"])

                should_tie = False
                if is_adjacent and is_same_pitch:
                    boundary_time = current["offset_time"]

                    # Check spectral flux in a window around the boundary
                    # (±30% of slot duration for robustness)
                    slot_dur = next_note["offset_time"] - next_note["onset_time"]
                    half_window = slot_dur * 0.3
                    t_window = np.linspace(boundary_time - half_window,
                                           boundary_time + half_window, 20)
                    flux_near = f_flux(t_window)
                    has_reattack = np.max(flux_near) > FLUX_THRESH * 0.5

                    if has_reattack:
                        # Spectral flux says new articulation → don't merge
                        should_tie = False
                    else:
                        # No flux peak — check RMS dip (min in window, not single point)
                        rms_near = f_rms(t_window)
                        boundary_rms = np.min(rms_near)
                        local_peak = max(current["peak_rms"], next_note["peak_rms"])
                        should_tie = boundary_rms > (local_peak * DIP_RATIO)

                if should_tie:
                    current["offset_time"] = next_note["offset_time"]
                    current["peak_rms"] = max(current["peak_rms"], next_note["peak_rms"])
                    current["velocity"] = max(current["velocity"], next_note["velocity"])
                else:
                    merged_notes.append(current)
                    current = next_note
            merged_notes.append(current)

        # 5. Release Logic (trim note tails)
        final_notes = []
        for note in merged_notes:
            onset_t = note["onset_time"]
            offset_t = note["offset_time"]
            peak = note["peak_rms"]

            n_samples = max(2, int((offset_t - onset_t) * 100))
            t_eval = np.linspace(onset_t, offset_t, n_samples)
            rms_seg = f_rms(t_eval)
            peak_idx = np.argmax(rms_seg)

            # Scan forward from peak for release point
            release_thresh_val = peak * self.release_ratio
            cutoff_idx = len(rms_seg) - 1

            for k in range(peak_idx, len(rms_seg)):
                if rms_seg[k] < release_thresh_val:
                    cutoff_idx = k
                    break

            cutoff_time = t_eval[cutoff_idx]
            if cutoff_time > onset_t:
                note["offset_time"] = cutoff_time

            final_notes.append(note)

        self.quantized_segments = final_notes
        print(f"  Grid: {len(raw_notes)} active → {len(merged_notes)} merged "
              f"→ {len(final_notes)} release-trimmed")

        # Print pitch distribution summary
        if final_notes:
            pitch_counts = Counter(n["pitch"] for n in final_notes)
            top_pitches = pitch_counts.most_common(5)
            pitch_str = ", ".join([
                f"{PITCH_CLASSES[p % 12]}{p // 12 - 1}:{c}"
                for p, c in top_pitches
            ])
            print(f"  Pitches: {pitch_str} ({len(pitch_counts)} unique)")

    def step_4_pitch_quantization(self):
        """
        Quantize detected notes.
        Mode A: If Chroma (.INFO) is available, snap to Top N Pitch Classes.
        Mode B: If no Chroma, snap to Top N absolute frequencies (histogram).
        """
        if not self.quantized_segments:
            return

        target_count = self.pitch_quantization_count

        # MODE A: Chroma-based Quantization (Pitch Class)
        if self.chroma_profile is not None:
            top_classes_idx = np.argsort(self.chroma_profile)[-target_count:][::-1]
            allowed_pcs = set(top_classes_idx)
            print(f"  Snapping to Top {target_count} Chroma Classes: "
                  f"{[PITCH_CLASSES[i] for i in top_classes_idx]}")

            snapped_count = 0
            for note in self.quantized_segments:
                pitch = note["pitch"]
                pc = pitch % 12

                if pc not in allowed_pcs:
                    # Find nearest allowed pitch class (circular distance)
                    dists = []
                    for target_pc in top_classes_idx:
                        d = abs(pc - target_pc)
                        d = min(d, 12 - d)
                        dists.append((d, target_pc))
                    dists.sort()
                    best_pc = dists[0][1]

                    diff = best_pc - pc
                    if diff > 6:
                        diff -= 12
                    elif diff < -6:
                        diff += 12

                    note["pitch"] = pitch + diff
                    snapped_count += 1

            print(f"  Snapped {snapped_count} notes to nearest allowed pitch classes.")
            return

        # MODE B: Absolute Pitch Frequency (Fallback)
        pitches = [n["pitch"] for n in self.quantized_segments]
        counts = Counter(pitches)
        top_n = counts.most_common(target_count)
        if not top_n:
            return

        allowed_pitches = np.array([p for p, c in top_n])
        print(f"  Constraining to Top {target_count} frequent pitches: {sorted(allowed_pitches)}")

        snapped_count = 0
        for note in self.quantized_segments:
            original = note["pitch"]
            idx = (np.abs(allowed_pitches - original)).argmin()
            new_pitch = allowed_pitches[idx]
            if new_pitch != original:
                note["pitch"] = new_pitch
                snapped_count += 1

        print(f"  Snapped {snapped_count}/{len(self.quantized_segments)} notes.")

    def step_5_export(self):
        """Export to MIDI using symusic."""
        score = symusic.Score()
        score.ticks_per_quarter = 960
        track = symusic.Track(name="Bass", is_drum=False, program=33)

        # Derive BPM from beat grid
        if len(self.beats) > 1:
            avg_dur = np.mean(np.diff(self.beats))
            bpm = 60.0 / avg_dur
            print(f"  BPM: {bpm:.2f}")
            score.tempos.append(symusic.Tempo(time=0, qpm=bpm))

            beat_ticks = np.arange(len(self.beats)) * 960
            time_to_tick = scipy.interpolate.interp1d(
                self.beats, beat_ticks, kind='linear', fill_value="extrapolate")
        else:
            bpm = 140.0
            score.tempos.append(symusic.Tempo(time=0, qpm=bpm))
            time_to_tick = lambda t: int(t * 960 * bpm / 60.0)

        for q in self.quantized_segments:
            if q["pitch"] == 0:
                continue

            onset_tick = int(time_to_tick(q["onset_time"]))
            offset_tick = int(time_to_tick(q["offset_time"]))
            duration = offset_tick - onset_tick

            if duration <= 0:
                continue

            track.notes.append(symusic.Note(
                time=onset_tick, duration=duration,
                pitch=q["pitch"], velocity=q["velocity"]))

        score.tracks.append(track)
        score.dump_midi(str(self.output_path))
        print(f"  {len(track.notes)} notes written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bass MIDI Pipeline")
    parser.add_argument("audio_path", help="Path to bass stem audio file")
    parser.add_argument("--beats", "-b", help="Path to madmom beats text file (optional, defaults to adjacent .BEATS_GRID)")
    parser.add_argument("--output", "-o", help="Path to output MIDI file (optional)")
    parser.add_argument("--subdivision", type=int, default=4,
                        help="Grid subdivisions per beat (default 4=16th notes). Use 8 for 32nd notes.")
    parser.add_argument("--flux-thresh", type=float, default=0.15,
                        help="Spectral flux onset threshold (default 0.15). LOWER = detect softer attacks. HIGHER = ignore noise.")
    parser.add_argument("--rms-thresh", type=float, default=0.05,
                        help="RMS energy threshold ratio (default 0.05). LOWER = detect quieter sustains. HIGHER = ignore rumble.")
    parser.add_argument("--min-vol-ratio", type=float, default=0.33,
                        help="Min volume ratio filter (default 0.33). Notes quieter than this ratio of average are discarded.")
    parser.add_argument("--dip-ratio", type=float, default=0.5,
                        help="Boundary dip ratio for merging (default 0.5). LOWER = stricter. HIGHER = looser.")
    parser.add_argument("--release-ratio", type=float, default=0.3,
                        help="Release threshold ratio (default 0.3). LOWER = longer notes. HIGHER = shorter/staccato.")
    parser.add_argument("--min-pitch", type=int, default=28,
                        help="Minimum MIDI pitch (default 28/E1).")
    parser.add_argument("--max-pitch", type=int, default=60,
                        help="Maximum MIDI pitch (default 60/C4).")
    parser.add_argument("--lpf", type=float, default=220.0,
                        help="Low-Pass Filter cutoff Hz (default 220.0). Set to 0 to disable.")
    parser.add_argument("--pitch-count", type=int, default=7,
                        help="Pitch quantization: restrict to Top N pitch classes. Uses chroma from .INFO if available. 0=disable. Default 7.")

    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    beats_path = args.beats
    output_path = args.output

    # Auto-detect beats
    if not beats_path:
        stem_name = audio_path.stem
        if stem_name.endswith("_bass"):
            base_name = stem_name[:-5]
        else:
            base_name = stem_name

        candidates = [
            audio_path.with_name(f"{base_name}.BEATS_GRID"),
            audio_path.with_name(f"{base_name}.txt"),
            audio_path.with_name(f"{stem_name}.BEATS_GRID")
        ]

        for c in candidates:
            if c.exists():
                beats_path = str(c)
                print(f"Auto-detected beats file: {beats_path}")
                break

    if not beats_path:
        print("Warning: No beats file provided or found. Quantization will be skipped/limited.")
        beats_path = ""

    # Auto-detect output
    if not output_path:
        output_path = str(audio_path.with_suffix(".mid"))
        print(f"Output path: {output_path}")

    pipeline = BassMidiPipeline(str(audio_path), str(beats_path), str(output_path),
                                subdivision=args.subdivision,
                                flux_thresh=args.flux_thresh,
                                rms_thresh=args.rms_thresh,
                                min_vol_ratio=args.min_vol_ratio,
                                dip_ratio=args.dip_ratio,
                                release_ratio=args.release_ratio,
                                min_midi=args.min_pitch,
                                max_midi=args.max_pitch,
                                lpf_cutoff=args.lpf)
    pipeline.pitch_quantization_count = args.pitch_count
    pipeline.run()
