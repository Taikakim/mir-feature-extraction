"""
Synthetic goa/psytrance bassline benchmark with exact ground-truth notes.

Real bass stems have no MIDI ground truth, so transcription quality cannot be
measured on them. This renders basslines from a monophonic subtractive synth
whose note events are known, covering the cases that make goa bass hard to
segment:

  rolling   KBBB off-beat 16ths, same pitch repeated, gates from staccato to
            ~100% (re-triggers with almost no gap: only an envelope bump)
  acid      303-style step sequence: accents, slides (glide, no re-trigger),
            octave jumps, per-note filter envelope, high resonance, cutoff sweeps
  legato    melodic line, variable note lengths, legato mode (pitch changes
            without re-trigger), optional portamento
  sweep     long held notes under slow, deep, resonant filter sweeps
            (must NOT be split into repeated notes)
  wobble    long held notes under a 1/8-1/16 tempo-synced filter LFO -- the
            ambiguous case: rhythmic like re-triggers, but one note per MIDI

Ground-truth conventions (what is observable is what is annotated):
  - a slide/legato transition to a DIFFERENT pitch starts a new note at its
    step time; the previous note ends there.
  - a slide/legato to the SAME pitch is a tie: one note.
  - a re-triggered same-pitch note is a new note (the synth guarantees an
    audible cue: sustain < 1 or filter-envelope amount > 0).
  - sub oscillators are only ever at unison, never -12, so the sounding
    fundamental is the annotated pitch.

Optional degradations mimic separated stems: mild kick bleed on every beat
(30% of clips), a 3/16 feedback delay, soft clipping, noise.

Usage:
    python src/transcription/bass/synth_bench.py OUT_DIR --n-per-family 10
writes OUT_DIR/<family>_<i>.wav, .BEATS_GRID, .notes.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from numba import njit

SR = 44100
SCALES = {
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'harm_minor': [0, 1, 4, 5, 7, 8, 10],   # phrygian dominant, very goa
}


# ---------------------------------------------------------------- synth core

@njit(cache=True)
def _polyblep(t, dt):
    if t < dt:
        t /= dt
        return t + t - t * t - 1.0
    if t > 1.0 - dt:
        t = (t - 1.0) / dt
        return t * t + t + t + 1.0
    return 0.0


@njit(cache=True)
def _render_voice(f0, trig, gate, cutoff_base, sr,
                  saw_mix, sq_mix, sine_mix, detune_cents,
                  a_att, a_dec, a_sus, a_rel,
                  f_dec, f_amt_oct, accent, res, drive):
    """Per-sample monophonic voice. trig[i]=1 re-triggers both envelopes."""
    n = f0.shape[0]
    out = np.zeros(n)
    ph1 = 0.0
    ph2 = 0.37
    ph3 = 0.0
    amp = 0.0
    fenv = 0.0
    stage = 0          # 0 idle/release, 1 attack, 2 decay/sustain
    acc = 1.0
    ic1a = 0.0
    ic2a = 0.0
    ic1b = 0.0
    ic2b = 0.0
    att_step = 1.0 / max(1.0, a_att * sr)
    dec_c = np.exp(-1.0 / max(1.0, a_dec * sr))
    rel_c = np.exp(-1.0 / max(1.0, a_rel * sr))
    fdec_c = np.exp(-1.0 / max(1.0, f_dec * sr))
    det = 2.0 ** (detune_cents / 1200.0)
    for i in range(n):
        if trig[i] > 0.0:
            stage = 1
            fenv = 1.0
            acc = accent[i]
        if gate[i] <= 0.0:
            stage = 0
        if stage == 1:
            amp += att_step
            if amp >= 1.0:
                amp = 1.0
                stage = 2
        elif stage == 2:
            amp = a_sus + (amp - a_sus) * dec_c
        else:
            amp *= rel_c
        fenv *= fdec_c

        dt1 = f0[i] / sr
        dt2 = f0[i] * det / sr
        ph1 += dt1
        if ph1 >= 1.0:
            ph1 -= 1.0
        ph2 += dt2
        if ph2 >= 1.0:
            ph2 -= 1.0
        ph3 += dt1
        if ph3 >= 1.0:
            ph3 -= 1.0
        saw = (2.0 * ph1 - 1.0 - _polyblep(ph1, dt1)) * 0.5 + \
              (2.0 * ph2 - 1.0 - _polyblep(ph2, dt2)) * 0.5
        sq = (1.0 if ph1 < 0.5 else -1.0) + _polyblep(ph1, dt1) - \
             _polyblep((ph1 + 0.5) % 1.0, dt1)
        s = saw_mix * saw + sq_mix * sq + sine_mix * np.sin(2.0 * np.pi * ph3)

        # cutoff in Hz: base * 2^(env*amount*accent)
        fc = cutoff_base[i] * 2.0 ** (fenv * f_amt_oct * acc)
        if fc > 0.45 * sr:
            fc = 0.45 * sr
        if fc < 20.0:
            fc = 20.0
        g = np.tan(np.pi * fc / sr)
        # two cascaded TPT state-variable lowpasses (24 dB/oct), resonance on the 1st
        k = 2.0 - 1.9 * res
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = s - ic2a
        v1 = a1 * ic1a + a2 * v3
        v2 = ic2a + a2 * ic1a + a3 * v3
        ic1a = 2.0 * v1 - ic1a
        ic2a = 2.0 * v2 - ic2a
        k2 = 1.4
        b1 = 1.0 / (1.0 + g * (g + k2))
        b2 = g * b1
        b3 = g * b2
        w3 = v2 - ic2b
        w1 = b1 * ic1b + b2 * w3
        w2 = ic2b + b2 * ic1b + b3 * w3
        ic1b = 2.0 * w1 - ic1b
        ic2b = 2.0 * w2 - ic2b
        y = w2 * amp * (0.6 + 0.4 * acc)
        if drive > 0.0:
            y = np.tanh(y * (1.0 + drive)) / np.tanh(1.0 + drive)
        out[i] = y
    return out


@njit(cache=True)
def _glide(target_midi, glide_flag, sr, tau):
    """One-pole portamento in the MIDI domain, only where glide_flag is set."""
    n = target_midi.shape[0]
    out = np.empty(n)
    cur = target_midi[0]
    c = np.exp(-1.0 / max(1.0, tau * sr))
    for i in range(n):
        if glide_flag[i] > 0.0:
            cur = target_midi[i] + (cur - target_midi[i]) * c
        else:
            cur = target_midi[i]
        out[i] = cur
    return out


def _kick(sr, dur=0.25):
    t = np.arange(int(dur * sr)) / sr
    f = 48 + 110 * np.exp(-t / 0.025)
    ph = 2 * np.pi * np.cumsum(f) / sr
    return np.sin(ph) * np.exp(-t / 0.09)


# ---------------------------------------------------------------- sequencing

def _scale_pitches(rng, root, scale, span=16):
    pcs = SCALES[scale]
    return [root + o * 12 + pc for o in range(3) for pc in pcs
            if root <= root + o * 12 + pc <= root + span]


def _seq_rolling(rng, n_beats, root, pool):
    steps = []   # (step_idx, pitch, n_steps, slide_into, accent)
    pitch = root
    gate = rng.uniform(0.35, 1.0)
    for b in range(n_beats):
        if b % rng.choice([2, 4, 8]) == 0 and rng.random() < 0.5:
            pitch = rng.choice(pool[:5]) if rng.random() < 0.8 else pitch + 12
        for j in (1, 2, 3):
            p = pitch
            if rng.random() < 0.08:
                p = pitch + 12 if pitch + 12 <= root + 24 else pitch
            steps.append((b * 4 + j, p, 1, False, 1.0))
    return steps, gate


def _seq_acid(rng, n_beats, root, pool):
    steps = []
    n = n_beats * 4
    pattern_len = rng.choice([8, 16])
    pat = []
    for s in range(pattern_len):
        on = rng.random() < rng.uniform(0.55, 0.9)
        p = rng.choice(pool)
        if rng.random() < 0.15:
            p = p + 12
        pat.append((on, p, rng.random() < 0.25, rng.random() < 0.3))
    s = 0
    while s < n:
        on, p, slide, acc = pat[s % pattern_len]
        if on:
            ln = 1
            # slides tie forward into the next step
            steps.append((s, p, ln, slide, 1.6 if acc else 1.0))
        s += 1
    gate = rng.uniform(0.45, 0.8)
    return steps, gate


def _seq_legato(rng, n_beats, root, pool):
    steps = []
    n = n_beats * 4
    s = 0
    while s < n:
        ln = int(rng.choice([1, 2, 2, 3, 4, 6, 8]))
        if rng.random() < 0.15:
            s += ln
            continue
        steps.append((s, rng.choice(pool), min(ln, n - s), True, 1.0))
        s += ln
    return steps, 1.0


def _seq_sweep(rng, n_beats, root, pool):
    steps = []
    n = n_beats * 4
    s = 0
    while s < n:
        ln = int(rng.choice([8, 16, 16, 32]))
        steps.append((s, rng.choice(pool[:6]), min(ln, n - s), False, 1.0))
        s += ln
    return steps, rng.uniform(0.9, 1.0)


FAMILIES = {
    'rolling': _seq_rolling,
    'acid': _seq_acid,
    'legato': _seq_legato,
    'sweep': _seq_sweep,
    'wobble': _seq_sweep,
}


def render_clip(family: str, seed: int, n_bars: int = 8, sr: int = SR):
    rng = np.random.default_rng(seed)
    bpm = float(rng.uniform(135, 150))
    beat = 60.0 / bpm
    step_dur = beat / 4
    n_beats = n_bars * 4
    lead_in = 0.25 + rng.uniform(0, 0.2)       # grid does not start at sample 0
    dur = lead_in + n_beats * beat + 0.6
    n = int(dur * sr)

    root = int(rng.integers(28, 41))           # E1..F2
    scale = rng.choice(list(SCALES))
    pool = _scale_pitches(rng, root, scale)

    steps, gate_frac = FAMILIES[family](rng, n_beats, root, pool)

    # --- build per-sample control signals + ground truth -------------------
    target = np.full(n, float(root))
    glide = np.zeros(n)
    trig = np.zeros(n)
    gate = np.zeros(n)
    accent = np.ones(n)
    legato_mode = family == 'legato'
    porta = legato_mode and rng.random() < 0.5
    tau = rng.uniform(0.012, 0.035)

    gt = []
    prev_end_step = None
    prev_slide = False
    prev_pitch = None
    for (s, p, ln, slide, acc) in steps:
        t0 = lead_in + s * step_dur
        i0 = int(round(t0 * sr))
        connected = prev_end_step == s and (prev_slide or legato_mode)
        hold = ln if (slide or legato_mode) else ln - 1 + gate_frac
        t_off = t0 + hold * step_dur
        if connected:
            # no re-trigger: pitch glides/jumps, gate stays open
            if porta or family == 'acid':
                glide[i0:i0 + int(0.15 * sr)] = 1.0
            if p == prev_pitch:
                gt[-1]['offset'] = t_off              # tie: extend previous
            else:
                gt[-1]['offset'] = t0
                gt.append({'onset': t0, 'offset': t_off, 'pitch': int(p)})
        else:
            trig[i0] = 1.0
            accent[i0] = acc
            gt.append({'onset': t0, 'offset': t_off, 'pitch': int(p)})
        i1 = int(round(t_off * sr))
        if slide or legato_mode:
            i1 += int(0.004 * sr)        # overlap guarantees no gap
        gate[i0:i1] = 1.0
        target[i0:] = p
        prev_end_step = s + ln
        prev_slide = slide
        prev_pitch = p

    f0_midi = _glide(target, glide, sr, tau)
    f0 = 440.0 * 2.0 ** ((f0_midi - 69.0) / 12.0)

    # --- cutoff automation -------------------------------------------------
    t = np.arange(n) / sr
    base = rng.uniform(150, 900)
    sweep = 2.0 ** (rng.uniform(0.5, 2.5) * np.sin(2 * np.pi * t / (beat * rng.choice([8, 16, 32]))
                                                  + rng.uniform(0, 6.28)))
    cutoff = base * sweep
    if family == 'sweep':
        cutoff = base * 2.0 ** (rng.uniform(1.5, 3.0) * np.sin(2 * np.pi * t / (beat * rng.choice([4, 8, 16]))
                                                           + rng.uniform(0, 6.28)))
        cutoff = np.maximum(cutoff, 60.0)
    if family == 'wobble':
        rate = beat / rng.choice([2, 4])         # 1/8 or 1/16 wobble on held notes
        depth = rng.uniform(0.8, 2.0)
        cutoff = cutoff * 2.0 ** (depth * (0.5 + 0.5 * np.sin(2 * np.pi * (t - lead_in) / rate)))
        cutoff = np.maximum(cutoff, 70.0)

    params = dict(
        saw_mix=float(rng.uniform(0.3, 1.0)),
        sq_mix=float(rng.uniform(0.0, 0.7)) if rng.random() < 0.5 else 0.0,
        sine_mix=float(rng.uniform(0.0, 0.8)),
        detune_cents=float(rng.uniform(0, 12)),
        a_att=float(rng.uniform(0.001, 0.006)),
        a_dec=float(rng.uniform(0.04, 0.35)),
        a_sus=float(rng.uniform(0.15, 0.8)),
        a_rel=float(rng.uniform(0.006, 0.03)),
        f_dec=float(rng.uniform(0.04, 0.3)),
        f_amt_oct=float(rng.uniform(0.5, 3.5)) if family not in ('sweep', 'wobble') else float(rng.uniform(0.0, 1.0)),
        res=float(rng.uniform(0.6, 0.95)) if family in ('acid', 'sweep') else float(rng.uniform(0.0, 0.7)),
        drive=float(rng.uniform(0, 3)) if rng.random() < 0.5 else 0.0,
    )
    y = _render_voice(f0, trig, gate, cutoff, float(sr), accent=accent, **params)
    y /= max(1e-9, np.max(np.abs(y)))

    # --- degradations ------------------------------------------------------
    deg = {}
    if family in ('acid', 'legato') and rng.random() < 0.4:
        d = int(round(3 * step_dur * sr))
        fb = rng.uniform(0.2, 0.4)
        mix = rng.uniform(0.1, 0.25)
        wet = np.zeros_like(y)
        buf = y.copy()
        for k in range(1, 5):
            buf = np.concatenate([np.zeros(d), buf[:-d]]) * (fb if k > 1 else 1.0)
            wet += buf
        y = y + mix * wet
        deg['delay'] = round(mix, 3)
    # kick bleed: real separated goa bass stems carry little (measured 2026-10-05 on 26
    # Goa_Separated tracks: a scaled drums-stem kick explains a median 5% of the bass
    # stem's <120 Hz energy on beats, >50% in none), so it is mild and occasional here
    if rng.random() < 0.3:
        kick = _kick(sr)
        lvl = 10 ** (rng.uniform(-36, -18) / 20)
        bleed = np.zeros_like(y)
        for b in range(n_beats):
            i0 = int(round((lead_in + b * beat) * sr))
            seg = kick[:max(0, min(len(kick), n - i0))]
            bleed[i0:i0 + len(seg)] += seg
        y = y + lvl * bleed
        deg['kick_bleed_db'] = round(20 * np.log10(lvl), 1)
    if rng.random() < 0.5:
        lvl = 10 ** (rng.uniform(-55, -38) / 20)
        y = y + lvl * rng.standard_normal(n)
        deg['noise_db'] = round(20 * np.log10(lvl), 1)
    y = 0.7 * y / max(1e-9, np.max(np.abs(y)))

    beats = lead_in + np.arange(n_beats + 1) * beat
    meta = dict(family=family, seed=seed, bpm=bpm, root=root, scale=str(scale),
                porta=bool(porta), synth=params, degradations=deg)
    for g in gt:
        g['onset'] = round(g['onset'], 6)
        g['offset'] = round(g['offset'], 6)
    return y.astype(np.float32), sr, beats, gt, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out_dir')
    ap.add_argument('--n-per-family', type=int, default=10)
    ap.add_argument('--bars', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fi, fam in enumerate(FAMILIES):
        for i in range(args.n_per_family):
            seed = args.seed + 1000 * fi + i
            y, sr, beats, gt, meta = render_clip(fam, seed, args.bars)
            stem = out / f'{fam}_{i:02d}'
            sf.write(f'{stem}.wav', y, sr, subtype='PCM_16')
            np.savetxt(f'{stem}.BEATS_GRID', beats, fmt='%.6f')
            with open(f'{stem}.notes.json', 'w') as f:
                json.dump({'meta': meta, 'notes': gt}, f, indent=1)
    print(f'wrote {len(FAMILIES) * args.n_per_family} clips to {out}')


if __name__ == '__main__':
    main()
