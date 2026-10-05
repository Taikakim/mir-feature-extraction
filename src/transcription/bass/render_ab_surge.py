"""
A/B listening set: real bass phrase vs MIDI from three transcribers, all played
through the SAME inverted Surge XT patch per stem.

Runs in the SAT venv (pedalboard + Surge XT):
    /home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python src/transcription/bass/render_ab_surge.py

Inputs (real_stems_eval, built by stable-audio-tools/scripts/synth_inversion/invert_stem_collection.py):
    audio/<id>_real.wav                     the real phrase
    audio/<id>_midi_playback.wav            old bass_midi_pipeline.py MIDI through the patch
    audio/<id>_muscriptor_playback.wav      MuScriptor MIDI through the patch
    midi_mir_v2/<id>_mir_v2.mid             new transcriber (src/transcription/bass/transcribe.py)
    vstpresets/Inverted_<id>.pedalboard_state
Outputs:
    audio/<id>_mir_v2_playback.wav
    ab/<id>_AB_real-old-new-muscriptor.wav  four segments, 0.6 s silence between
    ab/comparison.json, ab/run_meta.json

Metrics vs the real phrase (higher is better except stft):
    env_corr     correlation of <300 Hz RMS envelopes (rhythm / articulation / note lengths)
    chroma_corr  correlation of low-band chroma (pitch content, octave-blind)
    stft         multi-scale STFT distance (dominated by timbre: patch mismatch hits all three alike)
"""
import glob
import json
import os
import sys
from datetime import date

import librosa
import mido
import numpy as np
import scipy.signal
import soundfile as sf
import torch

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-tools/scripts/synth_inversion")
from surge_spec import init_synth, DEFAULT_PLUGIN_PATH, SAMPLE_RATE  # noqa: E402
from audio_utils import MultiScaleSTFTLoss  # noqa: E402

E = "/run/media/kim/Mantu/surge_200k_models/real_stems_eval"
AUD, AB = f"{E}/audio", f"{E}/ab"
os.makedirs(AB, exist_ok=True)


def events_from_midi(path, max_dur):
    t, ev = 0.0, []
    for msg in mido.MidiFile(path):
        t += msg.time
        if t > max_dur:
            break
        if msg.type in ("note_on", "note_off"):
            ev.append(mido.Message(msg.type, note=msg.note, velocity=msg.velocity, time=t))
    return sorted(ev, key=lambda m: m.time)


def mono(x):
    return x.mean(axis=1) if x.ndim > 1 else x


def env(y, sr):
    sos = scipy.signal.butter(4, 300, "low", fs=sr, output="sos")
    z = scipy.signal.sosfiltfilt(sos, y)
    return librosa.feature.rms(y=z, frame_length=1024, hop_length=256)[0]


def chroma(y, sr):
    return librosa.feature.chroma_cqt(y=y, sr=sr, fmin=librosa.note_to_hz("C1"), n_octaves=4, hop_length=512)


def corr(a, b):
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :n].ravel(), b[..., :n].ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


synth = init_synth(DEFAULT_PLUGIN_PATH)
stft = MultiScaleSTFTLoss()
rows = []
for real_path in sorted(glob.glob(f"{AUD}/*_real.wav")):
    sid = os.path.basename(real_path)[:-len("_real.wav")]
    state = f"{E}/vstpresets/Inverted_{sid}.pedalboard_state"
    v2_mid = f"{E}/midi_mir_v2/{sid}_mir_v2.mid"
    if not (os.path.exists(state) and os.path.exists(v2_mid)):
        continue
    y_real, sr = sf.read(real_path)
    dur = len(y_real) / sr
    with open(state, "rb") as f:
        synth.raw_state = f.read()
    synth.reset()
    ev = events_from_midi(v2_mid, dur)
    out = synth.process(ev, duration=dur, sample_rate=SAMPLE_RATE, num_channels=2) if ev \
        else np.zeros((2, len(y_real)))
    out = (out / (np.max(np.abs(out)) + 1e-7)).astype(np.float32)
    sf.write(f"{AUD}/{sid}_mir_v2_playback.wav", out.T, SAMPLE_RATE)

    clips = {"real": real_path, "old": f"{AUD}/{sid}_midi_playback.wav",
             "new": f"{AUD}/{sid}_mir_v2_playback.wav", "muscriptor": f"{AUD}/{sid}_muscriptor_playback.wav"}
    audio = {k: sf.read(p)[0] for k, p in clips.items() if os.path.exists(p)}
    r_m = mono(audio["real"])
    r_env, r_chr = env(r_m, sr), chroma(r_m, sr)
    row = {"stem_id": sid}
    for k in ("old", "new", "muscriptor"):
        if k not in audio:
            continue
        m = mono(audio[k])
        n = min(len(m), len(r_m))
        row[k] = {
            "env_corr": round(corr(env(m, sr), r_env), 3),
            "chroma_corr": round(corr(chroma(m, sr), r_chr), 3),
            "stft": round(float(stft(torch.from_numpy(m[:n]).float()[None, None],
                                     torch.from_numpy(r_m[:n]).float()[None, None])), 3),
        }
    rows.append(row)

    gap = np.zeros((int(0.6 * sr), 2), np.float32)
    seg = []
    for k in ("real", "old", "new", "muscriptor"):
        a = audio.get(k)
        if a is None:
            continue
        a = a if a.ndim == 2 else np.stack([a, a], 1)
        seg += [(0.9 * a / (np.max(np.abs(a)) + 1e-7)).astype(np.float32), gap]
    sf.write(f"{AB}/{sid}_AB_real-old-new-muscriptor.wav", np.concatenate(seg), sr)
    print(sid, {k: v for k, v in row.items() if k != "stem_id"})

summary = {}
for k in ("old", "new", "muscriptor"):
    vals = [r[k] for r in rows if k in r]
    summary[k] = {m: round(float(np.mean([v[m] for v in vals])), 3) for m in ("env_corr", "chroma_corr", "stft")}
    summary[k]["wins_env"] = sum(1 for r in rows if all(r[k]["env_corr"] >= r[o]["env_corr"]
                                                        for o in ("old", "new", "muscriptor") if o in r))
    summary[k]["wins_chroma"] = sum(1 for r in rows if all(r[k]["chroma_corr"] >= r[o]["chroma_corr"]
                                                           for o in ("old", "new", "muscriptor") if o in r))
json.dump({"summary": summary, "rows": rows}, open(f"{AB}/comparison.json", "w"), indent=1)
json.dump({
    "purpose": "Listening A/B of bass transcribers on real bass phrases: does the new DSP transcriber "
               "(src/transcription/bass/transcribe.py) give MIDI closer to the real line than the old "
               "bass_midi_pipeline.py and MuScriptor (run on the stem)? All three play through the same "
               "per-stem inverted Surge XT patch, so timbre is held constant and differences are notes/timing.",
    "hypothesis": "New transcriber beat MuScriptor 0.87 vs 0.81 note F1 on synthetic bass-only clips; "
                  "check whether that holds by ear on real stems.",
    "created": str(date.today()),
    "script": "mir/src/transcription/bass/render_ab_surge.py (SAT venv)",
    "inputs": {"real_and_old_and_muscriptor": E, "new_midi": f"{E}/midi_mir_v2",
               "new_midi_cmd": "transcribe_file(<id>_real.wav, beats=None, bpm=<catalog bpm>) -> write_midi"},
    "listen": "ab/<id>_AB_real-old-new-muscriptor.wav = real | old | new | muscriptor",
    "caveats": ["MuScriptor here ran on the stem phrase, not the full mix (its production mode)",
                "grid for old and new = catalog BPM from t=0 of the phrase (phrase start snapped to an onset)",
                "patch was inverted at the OLD pipeline's root note; an octave-off root sounds off in all playbacks"],
    "result": summary,
}, open(f"{AB}/run_meta.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
