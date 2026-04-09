"""
Profile timeseries feature extraction subroutines on 20 real crops.
Reports wall time per routine so we can identify bottlenecks.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
import soundfile as sf
import numpy as np

CROPS = [
    "/home/kim/Projects/goa_crops/100th Monkey - Spiritus/100th Monkey - Spiritus_0.flac",
    "/home/kim/Projects/goa_crops/100th Monkey - Spiritus/100th Monkey - Spiritus_1.flac",
    "/home/kim/Projects/goa_crops/100th Monkey - Spiritus/100th Monkey - Spiritus_2.flac",
    "/home/kim/Projects/goa_crops/100th Monkey - Spiritus/100th Monkey - Spiritus_3.flac",
    "/home/kim/Projects/goa_crops/100th Monkey - Spiritus/100th Monkey - Spiritus_4.flac",
    "/home/kim/Projects/goa_crops/1200 Micrograms - Shivas India (Astral Projection Remix)/1200 Micrograms - Shivas India (Astral Projection Remix)_0.flac",
    "/home/kim/Projects/goa_crops/1200 Micrograms - Shivas India (Astral Projection Remix)/1200 Micrograms - Shivas India (Astral Projection Remix)_1.flac",
    "/home/kim/Projects/goa_crops/1200 Micrograms - Shivas India (Astral Projection Remix)/1200 Micrograms - Shivas India (Astral Projection Remix)_2.flac",
    "/home/kim/Projects/goa_crops/1200 Micrograms - Shivas India (Astral Projection Remix)/1200 Micrograms - Shivas India (Astral Projection Remix)_3.flac",
    "/home/kim/Projects/goa_crops/1200 Micrograms - Shivas India (Astral Projection Remix)/1200 Micrograms - Shivas India (Astral Projection Remix)_4.flac",
    "/home/kim/Projects/goa_crops/1200 Micrograms - Shivas India (Astral Projection Remix)/1200 Micrograms - Shivas India (Astral Projection Remix)_5.flac",
    "/home/kim/Projects/goa_crops/1200 Micrograms - Shivas India (Astral Projection Remix)/1200 Micrograms - Shivas India (Astral Projection Remix)_6.flac",
    "/home/kim/Projects/goa_crops/1200 Micrograms - Shivas India (Astral Projection Remix)/1200 Micrograms - Shivas India (Astral Projection Remix)_7.flac",
    "/home/kim/Projects/goa_crops/1200 Micrograms - Shivas India (Astral Projection Remix)/1200 Micrograms - Shivas India (Astral Projection Remix)_8.flac",
    "/home/kim/Projects/goa_crops/1200 Micrograms - Shivas India (Astral Projection Remix)/1200 Micrograms - Shivas India (Astral Projection Remix)_9.flac",
]
CROPS = [p for p in CROPS if Path(p).exists()][:20]

N_STEPS = 256

print(f"Profiling {len(CROPS)} crops — n_steps={N_STEPS}\n")

# ---- imports ----
import librosa
import essentia.standard as es
from scipy.signal import sosfilt, butter, sosfiltfilt
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from core.common import FREQUENCY_BANDS
from spectral.multiband_rms import calculate_rms_db, create_bandpass_filter
from spectral.timeseries_features import (
    _compute_multiband_rms_ts,
    _compute_spectral_ts,
    _compute_onset_ts,
    _compute_hpcp_ts,
    _compute_activation_ts,
    _bin_frames,
)

timings = {}

def bench(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    timings.setdefault(label, []).append(elapsed)
    return result

# ---- pre-load audio ----
print("Loading audio...")
audios = []
for p in CROPS:
    data, sr = sf.read(p)
    if data.ndim > 1:
        data = data.mean(axis=1)
    audios.append((data.astype(np.float32), sr))

crop_dur = len(audios[0][0]) / audios[0][1]
print(f"Crop duration: {crop_dur:.1f}s, sr={audios[0][1]}\n")

# ---- benchmark each subroutine ----
for audio, sr in audios:
    # 1. Multiband RMS timeseries (4 bands, ThreadPoolExecutor)
    bench("multiband_rms_ts", _compute_multiband_rms_ts, audio, sr, N_STEPS)

    # 2. Spectral STFT (flatness, flux, skewness, kurtosis)
    bench("spectral_ts", _compute_spectral_ts, audio, sr, N_STEPS)

    # 3. Onset strength timeseries
    bench("onset_ts", _compute_onset_ts, audio, sr, N_STEPS)

    # 4. Beat tracking (librosa) - needed for beat_activations_ts
    def beat_track():
        _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        return librosa.frames_to_time(beat_frames, sr=sr)
    beats = bench("beat_track", beat_track)

    # 5. Beat/downbeat activation arrays
    dur = len(audio) / sr
    bench("beat_activations", _compute_activation_ts, beats, dur, N_STEPS)

    # 6. HPCP timeseries (Essentia frame-by-frame + Key() x N_STEPS)
    bench("hpcp_ts", _compute_hpcp_ts, audio, sr, N_STEPS)

# ---- report ----
print(f"{'Subroutine':<25} {'N':>4}  {'Mean':>8}  {'Total':>8}  {'% of total':>10}")
print("-" * 65)

total_all = sum(sum(v) for v in timings.values())
rows = []
for label, times in timings.items():
    mean_t = np.mean(times)
    total_t = sum(times)
    pct = 100.0 * total_t / total_all
    rows.append((label, len(times), mean_t, total_t, pct))
rows.sort(key=lambda x: -x[3])

for label, n, mean_t, total_t, pct in rows:
    print(f"{label:<25} {n:>4}  {mean_t:>7.3f}s  {total_t:>7.2f}s  {pct:>9.1f}%")

print(f"\n{'TOTAL':<25} {'':>4}  {'':>8}  {total_all:>7.2f}s")
n_crops = len(CROPS)
print(f"\nWall time per crop (all subroutines): {total_all/n_crops:.3f}s")
print(f"Audio seconds per wall second (RTF):  {crop_dur*n_crops/total_all:.2f}x")
