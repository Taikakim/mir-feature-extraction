"""
Bass stem -> grid-quantized MIDI, built for goa/psytrance basslines.

Successor to src/bass_midi_pipeline.py. What it fixes, and why:

  pitch    The old pipeline took the argmax FFT bin (n_fft=8192, 186 ms window)
           in the bass range. On filtered saw basses the 2nd harmonic is often
           louder than the fundamental, so notes flipped octave; and a 186 ms
           window spans two 16ths at 145 BPM, smearing every pitch change.
           Here: YIN (cumulative-mean-normalised difference, 46 ms integration)
           on the full-band stem -- periodicity, not loudest partial -- with a
           per-frame aperiodicity used as confidence.

  onsets   The old pipeline split notes on bass-band spectral flux from the same
           186 ms STFT, so a filter sweep or synced wobble on a held note read as
           a stream of re-attacks (10x over-segmentation on held notes).
           Here: articulation is judged per grid boundary from a period-synchronous
           LOW-BAND envelope (window = one pitch period: ripple-free, ~25 ms time
           resolution, and nearly blind to cutoff moves above the low harmonics):
             dip   how far the envelope falls at the boundary (gated notes)
             rise  how much it climbs straight after it (re-trigger "bumps"
                   with no gap, i.e. legato-gated rolling basses)
           Legato pitch changes and slides need no articulation at all: a stable
           pitch change between slots starts a new note.

  grid     Beats are regularised (local robust linear fit -> removes the 10 ms
           madmom frame jitter), extrapolated to cover the whole file, and a
           half-tempo grid (< 100 BPM with a sane double) is doubled.

  levels   Activity is judged against a LOCAL reference level (+-16 beats), not
           the track's global peak, so breakdowns/filtered intros still transcribe
           and delay tails are rejected relative to the notes around them.

Usage:
    python src/transcription/bass/transcribe.py "<track_dir>"          # finds bass.*, <track>.BEATS_GRID
    python src/transcription/bass/transcribe.py bass.flac --beats x.BEATS_GRID -o bass.mid
    python src/transcription/bass/transcribe.py "<track_dir>" --json   # also writes notes as JSON

Benchmark: src/transcription/bass/synth_bench.py + bench_eval.py.
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.signal

SR = 22050                 # bass + harmonics up to 11 kHz is plenty
HOP = 128                  # 5.8 ms analysis hop
FMIN, FMAX = 30.0, 500.0   # B0 .. B4
AUDIO_EXTS = ('.flac', '.wav', '.mp3', '.m4a', '.ogg', '.aiff')


@dataclass
class Note:
    onset: float
    offset: float
    pitch: int
    velocity: int
    slot: int              # grid slot index of the onset
    pitch_conf: float      # 0..1, mean YIN periodicity over the note
    cause: str             # why a new note started here: 'attack' | 'pitch' | 'start'


@dataclass
class Params:
    subdivision: int = 4          # grid slots per beat (4 = 16ths)
    active_ratio: float = 0.15    # slot level vs local reference -> can CONTINUE a note
    start_ratio: float = 0.25     # slot level vs local reference -> can START a note
                                  # (delay echoes and release tails show up as starts)
    floor_db: float = -50.0       # absolute floor vs track peak
    dip_thresh: float = 0.2       # 1 - valley/level at boundary ...
    dip_rise_db: float = 3.0      # ... AND this climb within 20 ms -> re-articulation
    pitch_change: float = 0.6     # semitones between slots -> new note (legato/slide)
    release_ratio: float = 0.25   # note ends when envelope < this * note peak
    yin_thresh: float = 0.2       # YIN absolute threshold (frame f0, envelope window only)
    nsdf_k: float = 0.9           # McLeod key-maximum threshold (segment pitch)
    glide_drift: float = 0.3      # in-slot pitch drift (st) marking a slot as mid-glide
    octave_persist: int = 2       # an octave jump with no attack must hold this many slots
                                  # (one-slot octave flips on held notes are tracker errors)
    rise_fast_db: float = 5.0     # climb within 20 ms alone -> re-articulation (no-gap bumps)
    min_conf: float = 0.5         # frame periodicity needed to count as pitched
    low_band_hz: float = 300.0    # envelope band (fundamental + low harmonics)
    quantize_offsets: bool = False


# ---------------------------------------------------------------- I/O

def load_mono(path, sr=SR) -> np.ndarray:
    import librosa
    try:
        import soundfile as sf
        y, file_sr = sf.read(str(path), always_2d=True)
        y = y.mean(axis=1)
        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr, res_type='soxr_hq')
    except Exception:
        from core.file_utils import read_audio  # m4a/aac fallback
        y, file_sr = read_audio(str(path))
        y = np.asarray(y, float)
        if y.ndim > 1:
            y = y.mean(axis=1) if y.shape[1] < y.shape[0] else y.mean(axis=0)
        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr, res_type='soxr_hq')
    return y.astype(np.float64)


def load_beats(path) -> np.ndarray:
    b = np.loadtxt(str(path))
    b = np.atleast_1d(b)
    if b.ndim == 2:
        b = b[:, 0]
    return np.sort(b.astype(float))


# ---------------------------------------------------------------- beat grid

def regularize_beats(beats: np.ndarray, duration: float, verbose=False) -> np.ndarray:
    """Smooth frame-quantisation jitter, double a half-tempo grid, cover the file."""
    if len(beats) < 4:
        return beats
    ibi = np.diff(beats)
    med = np.median(ibi)
    if 60.0 / med < 100.0 and 110.0 <= 120.0 / med <= 200.0:
        beats = np.sort(np.concatenate([beats, beats[:-1] + ibi / 2]))
        if verbose:
            print(f'  beat grid doubled: {60 / med:.1f} -> {120 / med:.1f} BPM')
        ibi = np.diff(beats)
        med = np.median(ibi)

    # local robust linear fit over +-8 beats; leave beats near tempo glitches raw
    n = len(beats)
    out = beats.copy()
    idx = np.arange(n)
    for i in range(n):
        lo, hi = max(0, i - 8), min(n, i + 9)
        x, t = idx[lo:hi], beats[lo:hi]
        if np.any(np.abs(np.diff(t) / med - 1.0) > 0.2):
            continue
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        resid = t - A @ coef
        keep = np.abs(resid) < 0.03
        if keep.sum() >= 4:
            coef, *_ = np.linalg.lstsq(A[keep], t[keep], rcond=None)
            out[i] = coef[0] * i + coef[1]

    # extrapolate to cover [0, duration] at the edge tempo
    head = [out[0] - k * np.median(np.diff(out[:9])) for k in range(1, 64)]
    head = [h for h in head if h > -1e-9][::-1]
    tail_ibi = np.median(np.diff(out[-9:]))
    tail = [out[-1] + k * tail_ibi for k in range(1, 64)]
    tail = [t for t in tail if t < duration + tail_ibi]
    return np.concatenate([head, out, tail])


def build_grid(beats: np.ndarray, subdivision: int) -> np.ndarray:
    pts = [beats[i] + j * (beats[i + 1] - beats[i]) / subdivision
           for i in range(len(beats) - 1) for j in range(subdivision)]
    pts.append(beats[-1])
    return np.asarray(pts)


def fallback_beats(y, sr, duration, bpm=0.0) -> np.ndarray:
    if bpm and bpm > 0:
        return np.arange(0.0, duration + 60.0 / bpm, 60.0 / bpm)
    import librosa
    _, frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    return librosa.frames_to_time(frames, sr=sr, hop_length=512)


# ---------------------------------------------------------------- features

def yin_track(y, sr=SR, hop=HOP, fmin=FMIN, fmax=FMAX, win=1024, thresh=0.2, chunk=2048):
    """YIN f0 + periodicity (1 - CMNDF at the chosen lag), one value per hop."""
    tau_min = max(2, int(sr / fmax))
    tau_max = int(np.ceil(sr / fmin)) + 1
    flen = win + tau_max
    pad = flen // 2
    yp = np.pad(y, (pad, pad + hop))
    n_frames = 1 + (len(y) - 1) // hop + 1
    nfft = 1 << int(np.ceil(np.log2(flen + win)))
    f0 = np.zeros(n_frames)
    conf = np.zeros(n_frames)
    taus = np.arange(1, tau_max + 1)
    for c0 in range(0, n_frames, chunk):
        c1 = min(n_frames, c0 + chunk)
        starts = np.arange(c0, c1) * hop
        F = np.lib.stride_tricks.sliding_window_view(yp, flen)[starts]          # (n, flen)
        A = np.fft.rfft(F[:, :win], nfft)
        B = np.fft.rfft(F, nfft)
        r = np.fft.irfft(np.conj(A) * B, nfft)[:, :tau_max + 1]
        cs = np.concatenate([np.zeros((len(F), 1)), np.cumsum(F ** 2, axis=1)], axis=1)
        e0 = cs[:, win][:, None]
        et = cs[:, taus + win] - cs[:, taus]
        d = e0 + et - 2.0 * r[:, 1:]
        d = np.maximum(d, 0.0)
        cm = d * taus / np.maximum(np.cumsum(d, axis=1), 1e-12)                 # CMNDF for tau=1..tau_max
        cm[:, :tau_min - 1] = 2.0
        # first local min below threshold, else global min
        below = cm < thresh
        is_min = np.zeros_like(below)
        is_min[:, 1:-1] = (cm[:, 1:-1] <= cm[:, :-2]) & (cm[:, 1:-1] <= cm[:, 2:])
        cand = below & is_min
        has = cand.any(axis=1)
        k = np.where(has, np.argmax(cand, axis=1), np.argmin(cm, axis=1))
        # walk down to the local minimum (the threshold crossing may sit on a slope)
        rows = np.arange(len(k))
        for _ in range(8):
            nxt = np.minimum(k + 1, cm.shape[1] - 1)
            step = cm[rows, nxt] < cm[rows, k]
            if not step.any():
                break
            k = np.where(step, nxt, k)
        km = np.clip(k, 1, cm.shape[1] - 2)
        a, b, g = cm[rows, km - 1], cm[rows, km], cm[rows, km + 1]
        den = a - 2 * b + g
        off = 0.5 * (a - g) / np.where(np.abs(den) > 1e-12, den, np.inf)
        tau = (km + 1) + np.clip(off, -1, 1)
        f0[c0:c1] = sr / tau
        conf[c0:c1] = np.clip(1.0 - cm[rows, k], 0.0, 1.0)
    return f0, conf


def pitch_filter(y, sr=SR):
    """Signal used for segment pitch: 25 Hz high-pass + 600 Hz low-pass.

    No spectral tilt: a 1/f tilt (tried) lifts the fundamental over a loud 2nd
    harmonic, but it lifts delay echoes of earlier low notes and kick bleed just
    as much, and those then capture the period of high notes (-12/-19 st errors).
    """
    sos = np.vstack([scipy.signal.butter(4, 600.0, 'low', fs=sr, output='sos'),
                     scipy.signal.butter(2, 25.0, 'high', fs=sr, output='sos')])
    return scipy.signal.sosfiltfilt(sos, y)


def nsdf_pitch(x, sr=SR, fmin=FMIN, fmax=FMAX, k=0.9):
    """McLeod pitch method on one segment -> (f0 Hz, clarity 0..1) or (0, 0).

    NSDF normalises by the overlap energy at each lag, so ~2 periods of signal
    suffice -- a 16th-note segment at 145 BPM holds 3+ periods of E1.
    """
    N = len(x)
    tau_max = min(int(sr / fmin) + 1, N - 2)
    tau_min = max(2, int(sr / fmax))
    if tau_max <= tau_min + 2 or np.dot(x, x) <= 1e-12:
        return 0.0, 0.0
    nfft = 1 << int(np.ceil(np.log2(2 * N)))
    X = np.fft.rfft(x, nfft)
    r = np.fft.irfft(X * np.conj(X), nfft)[:tau_max + 1]
    cs = np.concatenate([[0.0], np.cumsum(x ** 2)])
    taus = np.arange(tau_max + 1)
    m = cs[N - taus] + (cs[N] - cs[taus])
    n = np.where(m > 1e-12, 2.0 * r / np.maximum(m, 1e-12), 0.0)
    # key maxima: highest peak between each positive-going zero crossing and the next negative one
    peaks = []
    t = tau_min
    while t < tau_max and n[t] > 0:          # skip the zero-lag lobe
        t += 1
    while t < tau_max:
        while t < tau_max and n[t] <= 0:
            t += 1
        if t >= tau_max:
            break
        best = t
        while t < tau_max and n[t] > 0:
            if n[t] > n[best]:
                best = t
            t += 1
        if 0 < best < tau_max:
            peaks.append(best)
    if not peaks:
        return 0.0, 0.0
    vmax = max(n[p] for p in peaks)
    pick = next(p for p in peaks if n[p] >= k * vmax)
    a, b, g = n[pick - 1], n[pick], n[pick + 1]
    den = a - 2 * b + g
    off = 0.5 * (a - g) / den if abs(den) > 1e-12 else 0.0
    tau = pick + float(np.clip(off, -1, 1))
    return sr / tau, float(np.clip(b, 0, 1))


def period_sync_env(y, f0, conf, sr=SR, hop=HOP, cutoff=300.0, min_conf=0.5):
    """Low-band RMS over exactly one pitch period around each frame centre.

    A window of one period is ripple-free for a periodic signal and as short as
    the signal allows, so 16th-note gaps and re-trigger bumps survive; low-pass
    keeps it near-blind to filter sweeps above the low harmonics.
    """
    sos = scipy.signal.butter(4, cutoff, 'low', fs=sr, output='sos')
    z = scipy.signal.sosfiltfilt(sos, y)
    cs = np.concatenate([[0.0], np.cumsum(z ** 2)])
    n = len(f0)
    centers = np.arange(n) * hop
    default_p = sr / 45.0
    p = np.where((conf > min_conf) & (f0 > 0), sr / np.maximum(f0, 1.0), default_p)
    p = np.clip(p, sr / FMAX * 2, sr / FMIN)       # >= ~2 periods for high notes, still short
    lo = np.clip((centers - p / 2).astype(int), 0, len(z))
    hi = np.clip((centers + p / 2).astype(int), 0, len(z))
    e = np.sqrt(np.maximum(cs[hi] - cs[lo], 0.0) / np.maximum(hi - lo, 1))
    return e


# ---------------------------------------------------------------- transcription

def _hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(np.maximum(f, 1e-6) / 440.0)


def transcribe(y: np.ndarray, beats: np.ndarray, params: Params = None, sr=SR,
               verbose=False) -> Tuple[List[Note], dict]:
    P = params or Params()
    duration = len(y) / sr
    beats = regularize_beats(beats, duration, verbose=verbose)
    grid = build_grid(beats, P.subdivision)

    f0, conf = yin_track(y, sr, thresh=P.yin_thresh)
    # isolated frame octave errors only matter here for the envelope window length
    f0 = np.exp(scipy.signal.medfilt(np.log(np.maximum(f0, 1.0)), 7))
    env = period_sync_env(y, f0, conf, sr, cutoff=P.low_band_hz, min_conf=P.min_conf)
    yp = pitch_filter(y, sr)
    n = len(env)
    t_frames = np.arange(n) * HOP / sr
    logenv = 20 * np.log10(env + 1e-6 * max(env.max(), 1e-9))
    dlag = max(1, int(round(0.02 * sr / HOP)))

    def fr(t):
        return int(np.clip(round(t * sr / HOP), 0, n - 1))

    peak = env.max() if env.max() > 0 else 1.0
    floor = peak * 10 ** (P.floor_db / 20)

    n_slots = len(grid) - 1
    level = np.zeros(n_slots)
    pitch = np.full(n_slots, np.nan)
    pconf = np.zeros(n_slots)
    dip = np.zeros(n_slots)
    rise = np.zeros(n_slots)
    rise_fast = np.zeros(n_slots)
    drift = np.zeros(n_slots)
    fmidi = _hz_to_midi(f0)
    fpitched = (conf >= P.min_conf) & (f0 >= FMIN) & (f0 <= FMAX)
    for k in range(n_slots):
        g0, g1 = grid[k], grid[k + 1]
        L = g1 - g0
        a, b = fr(g0), max(fr(g1), fr(g0) + 1)
        seg = env[a:b]
        level[k] = np.percentile(seg, 90) if len(seg) else 0.0

        # pitch: NSDF over the settled, sounding part of the slot (skips the attack
        # pitch blip and most of a glide; backs off toward the onset for staccato)
        loud = np.nonzero(seg > 0.3 * level[k])[0] if level[k] > 0 else []
        if len(loud):
            t_end = t_frames[a + loud[-1]] + HOP / sr
            t_start = g0 + 0.3 * L
            if t_end - t_start < 0.05:
                t_start = max(g0 + 0.03 * L, t_end - 0.05)
            fa, fb = fr(t_start), max(fr(t_end), fr(t_start) + 1)
            fm = fmidi[fa:fb][fpitched[fa:fb]]
            if len(fm) >= 6:
                third = len(fm) // 3
                drift[k] = np.median(fm[-third:]) - np.median(fm[:third])
            x = yp[int(t_start * sr):int(t_end * sr)]
            if len(x) > 64:
                f, c = nsdf_pitch(x, sr, k=P.nsdf_k)
                if f > 0 and c >= P.min_conf:
                    pitch[k] = _hz_to_midi(f)
                    pconf[k] = c

        # articulation at the slot's leading boundary
        pre = env[fr(g0 - 0.45 * L):max(fr(g0 - 0.1 * L), fr(g0 - 0.45 * L) + 1)]
        post = env[a:max(fr(g0 + 0.5 * L), a + 1)]
        w0, w1 = fr(g0 - 0.15 * L), max(fr(g0 + 0.2 * L), fr(g0 - 0.15 * L) + 1)
        val = env[w0:w1]
        pre_l = np.median(pre) if len(pre) else 0.0
        post_l = post.max() if len(post) else 0.0
        valley = val.min() if len(val) else 0.0
        ref = min(pre_l, post_l)
        dip[k] = 1.0 - valley / ref if ref > 0 else 1.0
        rise[k] = 20 * np.log10((post_l + 1e-9) / (valley + 1e-9))
        r0, r1 = w0, min(n - dlag, fr(g0 + 0.3 * L))
        if r1 > r0:
            rise_fast[k] = np.max(logenv[r0 + dlag:r1 + dlag] - logenv[r0:r1])

    # local reference level: 90th pct of slot levels within +-16 beats
    w = 16 * P.subdivision
    local_ref = np.array([np.percentile(level[max(0, k - w):k + w + 1], 90) for k in range(n_slots)])
    active = (level > P.active_ratio * local_ref) & (level > floor)
    # a sweep that closes the filter dips the low band too, but recovers slowly:
    # a dip only counts as a gap when the envelope comes back sharply
    artic = ((dip >= P.dip_thresh) & (rise_fast >= P.dip_rise_db)) | (rise_fast >= P.rise_fast_db)
    can_start = (level > P.start_ratio * local_ref) & (level > floor)

    # tuning: old analog gear / varispeed puts many goa tracks off A440. Circular mean
    # of the fractional semitone over clearly pitched slots; notes round after removing it.
    ok = active & ~np.isnan(pitch) & (pconf >= 0.9)
    tuning = 0.0
    if ok.sum() >= 8:
        ph = np.exp(2j * np.pi * (pitch[ok] - np.round(pitch[ok])))
        tuning = float(np.angle(np.mean(ph)) / (2 * np.pi))

    # portamento / 303 slide: a glide slower than ~1/3 slot is still moving in the
    # first slot of the new note, which then reads an in-between pitch. If that slot
    # is drifting toward the next slot's pitch, it is the glide: give it the target.
    eff = pitch.copy()
    for k in range(n_slots - 2, 0, -1):
        a, b, c = pitch[k - 1], pitch[k], eff[k + 1]
        if np.isnan(a) or np.isnan(b) or np.isnan(c) or not (active[k] and active[k + 1]):
            continue
        if artic[k + 1] or abs(b - a) < P.pitch_change or abs(c - b) < 0.5:
            continue
        d = np.sign(b - a)
        if np.sign(c - b) == d and np.sign(drift[k]) == d and abs(drift[k]) >= P.glide_drift:
            eff[k] = c
    pitch = eff

    def _is_octave(dp):
        o = round(dp / 12.0)
        return o != 0 and abs(dp - 12.0 * o) < 0.5

    def _persists(k, p, n_slots_req):
        for j in range(k + 1, min(k + n_slots_req, n_slots)):
            if not active[j] or artic[j] or np.isnan(pitch[j]) or abs(pitch[j] - p) >= 0.5:
                return False
        return k + n_slots_req <= n_slots

    notes: List[Note] = []
    cur = None          # dict(k0, k1, pitches, cause)
    for k in range(n_slots):
        if not active[k]:
            if cur is not None:
                notes.append(cur)
                cur = None
            continue
        has_pitch = not np.isnan(pitch[k])
        if cur is None:
            if has_pitch and can_start[k]:
                cur = dict(k0=k, k1=k, pitches=[pitch[k]], cause='start')
            continue
        known = [q for q in cur['pitches'][-2:] if not np.isnan(q)]
        dp = pitch[k] - np.median(known) if (has_pitch and known) else 0.0
        if abs(dp) >= P.pitch_change and not artic[k] and _is_octave(dp) and \
                not _persists(k, pitch[k], P.octave_persist):
            cur['k1'] = k                    # octave flip: keep the note, ignore this slot's pitch
            continue
        if abs(dp) >= P.pitch_change:
            notes.append(cur)
            cur = dict(k0=k, k1=k, pitches=[pitch[k]], cause='pitch') if can_start[k] else None
        elif artic[k] and has_pitch and can_start[k]:
            notes.append(cur)
            cur = dict(k0=k, k1=k, pitches=[pitch[k]], cause='attack')
        else:
            cur['k1'] = k
            cur['pitches'].append(pitch[k])
    if cur is not None:
        notes.append(cur)

    out: List[Note] = []
    for i, c in enumerate(notes):
        k0, k1 = c['k0'], c['k1']
        on = grid[k0]
        nxt_on = grid[notes[i + 1]['k0']] if i + 1 < len(notes) else grid[-1]
        # release: envelope falls below release_ratio * peak of the last slot. Measured
        # even when the next note starts in the very next slot -- a rolling bass gated
        # at 50% is staccato, not legato, and only a gate held to the next onset is.
        reg_end = min(grid[min(k1 + 2, len(grid) - 1)], nxt_on)
        a, b = fr(grid[k1]), fr(reg_end)
        seg = env[a:max(b, a + 1)]
        pk = int(np.argmax(seg))
        below = np.nonzero(seg[pk:] < P.release_ratio * seg[pk])[0]
        off = t_frames[a + pk + below[0]] if len(below) else reg_end
        off = min(max(off, on + 0.25 * (grid[k0 + 1] - grid[k0])), nxt_on)
        if P.quantize_offsets:
            off = grid[int(np.argmin(np.abs(grid - off)))]
            if off <= on:
                off = grid[k0 + 1]
        a, b = fr(on), max(fr(off), fr(on) + 1)
        p = int(round(np.nanmedian(c['pitches']) - tuning))
        vel_ref = local_ref[k0] if local_ref[k0] > 0 else peak
        vel = int(np.clip(127 * (env[a:b].max() / vel_ref) ** 0.7, 20, 127))
        out.append(Note(onset=float(on), offset=float(off), pitch=p, velocity=vel, slot=int(k0),
                        pitch_conf=float(np.mean(pconf[k0:k1 + 1])), cause=c['cause']))

    info = dict(tuning=tuning, grid=grid, beats=beats, f0=f0, conf=conf, env=env, level=level,
                local_ref=local_ref, pitch=pitch, pconf=pconf, dip=dip, rise=rise,
                rise_fast=rise_fast, drift=drift, active=active, artic=artic)
    if verbose:
        causes = {c: sum(1 for x in out if x.cause == c) for c in ('start', 'attack', 'pitch')}
        print(f'  tuning offset {tuning * 100:+.0f} cents')
        print(f'  {len(out)} notes from {active.sum()}/{n_slots} active slots '
              f'({causes["start"]} after rests, {causes["attack"]} re-attacks, {causes["pitch"]} pitch changes)')
    return out, info


def transcribe_file(audio_path, beats_path=None, bpm=0.0, params: Params = None,
                    verbose=True, **overrides) -> Tuple[List[Note], dict]:
    P = params or Params()
    for k, v in overrides.items():
        setattr(P, k, v)
    y = load_mono(audio_path)
    duration = len(y) / SR
    beats = None
    if beats_path and Path(beats_path).is_file():
        beats = load_beats(beats_path)
    if beats is None or len(beats) < 4:
        if verbose:
            print('  no usable beat grid; using BPM/librosa fallback')
        beats = fallback_beats(y, SR, duration, bpm)
    return transcribe(y, beats, P, verbose=verbose)


# ---------------------------------------------------------------- export

def write_midi(notes: List[Note], beats: np.ndarray, path, program=38):
    """Ticks follow the beat grid (960/beat), so grid onsets land exactly on 16ths in a DAW."""
    import symusic
    import scipy.interpolate
    tpq = 960
    score = symusic.Score(tpq)
    bpm = 60.0 / np.median(np.diff(beats))
    score.tempos.append(symusic.Tempo(time=0, qpm=float(bpm)))
    to_tick = scipy.interpolate.interp1d(beats, np.arange(len(beats)) * tpq, kind='linear',
                                         fill_value='extrapolate')
    first = float(to_tick(0.0))
    shift = tpq * int(np.ceil(max(0.0, -first) / tpq))   # whole beats: ticks >= 0 AND still on the grid
    track = symusic.Track(name='Bass', program=program, is_drum=False)
    for nt in notes:
        t0 = int(round(float(to_tick(nt.onset)))) + shift
        t1 = int(round(float(to_tick(nt.offset)))) + shift
        if t1 > t0:
            track.notes.append(symusic.Note(time=t0, duration=t1 - t0, pitch=int(nt.pitch),
                                            velocity=int(nt.velocity)))
    score.tracks.append(track)
    score.dump_midi(str(path))


def _find_inputs(arg: Path):
    """Accept a track folder (bass.* + <name>.BEATS_GRID) or a bass audio file."""
    if arg.is_dir():
        audio = next((arg / f'bass{e}' for e in AUDIO_EXTS if (arg / f'bass{e}').exists()), None)
        if audio is None:
            raise FileNotFoundError(f'no bass.* stem in {arg}')
        beats = arg / f'{arg.name}.BEATS_GRID'
        return audio, beats if beats.exists() else None, arg / 'bass.mid'
    stem = arg.stem[:-5] if arg.stem.endswith('_bass') else arg.stem
    for c in (arg.with_name(f'{stem}.BEATS_GRID'), arg.with_name(f'{arg.parent.name}.BEATS_GRID'),
              arg.with_suffix('.BEATS_GRID')):
        if c.exists():
            return arg, c, arg.with_suffix('.mid')
    return arg, None, arg.with_suffix('.mid')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='track folder or bass stem audio file')
    ap.add_argument('--beats', '-b', help='beat grid file (default: <track>.BEATS_GRID)')
    ap.add_argument('--output', '-o', help='output .mid (default: bass.mid next to the stem)')
    ap.add_argument('--bpm', type=float, default=0.0, help='BPM for a synthetic grid when no beats file')
    ap.add_argument('--json', action='store_true', help='also write notes as <output>.json')
    for f, v in asdict(Params()).items():
        ap.add_argument('--' + f.replace('_', '-'), type=type(v) if not isinstance(v, bool) else int,
                        default=v, help=f'(default {v})')
    a = ap.parse_args()

    audio, beats, out = _find_inputs(Path(a.input))
    beats = Path(a.beats) if a.beats else beats
    out = Path(a.output) if a.output else out
    P = Params(**{f: (bool(getattr(a, f)) if isinstance(v, bool) else getattr(a, f))
                  for f, v in asdict(Params()).items()})
    print(f'Transcribing {audio}' + (f' (beats: {beats.name})' if beats else ''))
    notes, info = transcribe_file(audio, beats, bpm=a.bpm, params=P)
    write_midi(notes, info['beats'], out)
    print(f'  wrote {out}')
    if a.json:
        with open(out.with_suffix('.json'), 'w') as f:
            json.dump({'params': asdict(P), 'tuning_cents': round(100 * info['tuning'], 1),
                       'notes': [asdict(n) for n in notes]}, f, indent=1)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
