"""
Score bass transcribers against the synthetic benchmark (synth_bench.py).

Metrics (mir_eval.transcription, 50 ms onset tolerance, 50 cent pitch tolerance):
  note_f1     onset + pitch            (the headline number)
  noteoff_f1  onset + pitch + offset   (offset within max(50 ms, 20% of duration))
  onset_f1    onsets only, pitch ignored  (pure segmentation quality)
  pc_f1       onset + pitch CLASS      (octave errors forgiven; gap to note_f1 = octave errors)

Usage:
    python src/transcription/bass/bench_eval.py BENCH_DIR --methods old new
    python src/transcription/bass/bench_eval.py BENCH_DIR --methods basic_pitch   # separate process: imports TF
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import mir_eval

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))


def _arr(notes):
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[n['onset'], max(n['offset'], n['onset'] + 1e-3)] for n in notes])
    hz = 440.0 * 2 ** ((np.array([n['pitch'] for n in notes], float) - 69) / 12)
    return iv, hz


def score(ref_notes, est_notes):
    ri, rp = _arr(ref_notes)
    ei, ep = _arr(est_notes)
    out = {}
    if len(ei) == 0:
        return {k: 0.0 for k in ('note_f1', 'noteoff_f1', 'onset_f1', 'pc_f1', 'note_p', 'note_r')}
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)
    out.update(note_f1=f, note_p=p, note_r=r)
    _, _, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=0.2)
    out['noteoff_f1'] = f
    _, _, f = mir_eval.transcription.onset_precision_recall_f1(ri, ei)
    out['onset_f1'] = f
    # pitch-class: fold est pitch into the octave of the matching ref pitch where possible
    est_pc = []
    for n in est_notes:
        cands = [m['pitch'] for m in ref_notes if abs(m['onset'] - n['onset']) <= 0.05
                 and (m['pitch'] - n['pitch']) % 12 == 0]
        est_pc.append(dict(n, pitch=cands[0] if cands else n['pitch']))
    ei2, ep2 = _arr(est_pc)
    _, _, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei2, ep2, offset_ratio=None)
    out['pc_f1'] = f
    return out


# ---------------------------------------------------------------- methods

def run_old(wav, beats):
    import contextlib, io
    from bass_midi_pipeline import BassMidiPipeline
    with contextlib.redirect_stdout(io.StringIO()):
        p = BassMidiPipeline(str(wav), str(beats), '/dev/null')
        p.pitch_quantization_count = 7      # CLI default
        p.step_1_spectral_pitch()
        p.step_2_energy_and_flux()
        p.step_3_grid_analysis()
        p.step_4_pitch_quantization()
    return [dict(onset=n['onset_time'], offset=n['offset_time'], pitch=int(n['pitch']))
            for n in p.quantized_segments if n['pitch'] > 0]


def run_basic_pitch(wav, beats):
    import contextlib, io, os
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict
    onnx = Path(ICASSP_2022_MODEL_PATH).with_suffix('.onnx')
    if not onnx.exists():
        onnx = Path(os.path.dirname(ICASSP_2022_MODEL_PATH)) / 'nmp.onnx'
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, events = predict(str(wav), str(onnx), minimum_frequency=30, maximum_frequency=400,
                               multiple_pitch_bends=False)
    return [dict(onset=float(s), offset=float(e), pitch=int(p)) for s, e, p, a, *_ in events]


_MUSCRIPTOR = {}


def run_muscriptor(wav, beats, instruments=None):
    """MuScriptor medium; every non-drum note (the input is a bass-only clip).

    Device from MUSCRIPTOR_DEVICE (default cpu: GPU work needs the fleet lock).
    `instruments` restricts decoding, e.g. ['electric_bass', 'acoustic_bass'].
    """
    import io
    import os
    import pretty_midi
    from transcription.muscriptor_transcribe import MuScriptorTranscriber
    key = tuple(instruments or ())
    if key not in _MUSCRIPTOR:
        import torch
        torch.set_num_threads(int(os.environ.get('MUSCRIPTOR_THREADS', '8')))
        _MUSCRIPTOR[key] = MuScriptorTranscriber(model='medium', instruments=instruments,
                                                 device=os.environ.get('MUSCRIPTOR_DEVICE', 'cpu'))
    midi = _MUSCRIPTOR[key].transcribe_audio_file(Path(wav))
    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi))
    return [dict(onset=n.start, offset=n.end, pitch=int(n.pitch))
            for inst in pm.instruments if not inst.is_drum for n in inst.notes]


def run_muscriptor_bass(wav, beats):
    return run_muscriptor(wav, beats, instruments=['electric_bass', 'acoustic_bass'])


def run_new(wav, beats, **kw):
    from transcription.bass.transcribe import transcribe_file
    notes, _ = transcribe_file(wav, beats_path=beats, verbose=False, **kw)
    return [dict(onset=n.onset, offset=n.offset, pitch=n.pitch) for n in notes]


METHODS = {'old': run_old, 'basic_pitch': run_basic_pitch, 'new': run_new,
           'muscriptor': run_muscriptor, 'muscriptor_bass': run_muscriptor_bass}


def evaluate(bench_dir, methods, method_kwargs=None, quiet=False):
    bench = Path(bench_dir)
    clips = sorted(bench.glob('*.notes.json'))
    res = defaultdict(lambda: defaultdict(list))
    for c in clips:
        stem = str(c)[:-len('.notes.json')]
        gt = json.load(open(c))
        fam = gt['meta']['family']
        for m in methods:
            est = METHODS[m](Path(stem + '.wav'), Path(stem + '.BEATS_GRID'), **(method_kwargs or {}).get(m, {}))
            s = score(gt['notes'], est)
            s['n_est'] = len(est)
            s['n_ref'] = len(gt['notes'])
            for k, v in s.items():
                res[m][(fam, k)].append(v)
                res[m][('ALL', k)].append(v)
    if not quiet:
        keys = ['note_f1', 'note_p', 'note_r', 'pc_f1', 'onset_f1', 'noteoff_f1']
        fams = sorted({f for m in res for (f, _) in res[m]})
        for m in methods:
            print(f'\n=== {m}')
            print(f'{"family":10s}' + ''.join(f'{k:>11s}' for k in keys) + f'{"est/ref":>10s}')
            for fam in fams:
                r = res[m]
                row = ''.join(f'{np.mean(r[(fam, k)]):11.3f}' for k in keys)
                ratio = np.sum(r[(fam, "n_est")]) / max(1, np.sum(r[(fam, "n_ref")]))
                print(f'{fam:10s}{row}{ratio:10.2f}')
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('bench_dir')
    ap.add_argument('--methods', nargs='+', default=['old', 'new'])
    a = ap.parse_args()
    evaluate(a.bench_dir, a.methods)
