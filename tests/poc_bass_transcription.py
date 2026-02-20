"""
Bass Stem → MIDI Transcription Benchmark

Compares 3 pitch estimation algorithms on bass stems:
  1. Basic Pitch (Spotify) — direct audio-to-MIDI
  2. PESTO (Sony CSL Paris) — pitch estimation → MIDI conversion
  3. CREPE (MARL/NYU) — pitch estimation → MIDI conversion

Usage:
    python tests/poc_bass_transcription.py \
        --input-dir "test_data/0286 Electric Universe - Alien Encounter 2" \
        --max-stems 5

Output:
    - MIDI files per algorithm per stem in --output-dir
    - JSON results summary in --output-json
    - Console summary table
"""

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

# Suppress TF/ONNX verbosity
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NoteEvent:
    """A single MIDI note event."""
    onset: float       # seconds
    offset: float      # seconds
    midi_note: int     # MIDI note number
    velocity: int = 100
    frequency: float = 0.0  # average Hz during note


@dataclass
class TranscriptionResult:
    """Result of one algorithm on one stem."""
    algorithm: str
    stem_file: str
    wall_time_s: float = 0.0
    note_count: int = 0
    pitch_range: Tuple[int, int] = (0, 0)  # (min_midi, max_midi)
    total_note_duration_s: float = 0.0
    midi_path: str = ""
    error: str = ""


@dataclass
class StemResult:
    """All algorithm results for one stem."""
    stem_file: str
    duration_s: float = 0.0
    results: Dict[str, TranscriptionResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pitch-to-MIDI conversion (for PESTO and CREPE)
# ---------------------------------------------------------------------------

def hz_to_midi(freq_hz: float) -> int:
    """Convert frequency in Hz to nearest MIDI note number."""
    if freq_hz <= 0:
        return 0
    return int(round(69 + 12 * np.log2(freq_hz / 440.0)))


def pitch_to_midi_notes(
    times: np.ndarray,
    frequencies: np.ndarray,
    confidences: np.ndarray,
    confidence_threshold: float = 0.5,
    min_midi: int = 24,   # C1
    max_midi: int = 72,   # C5
    min_note_duration: float = 0.03,  # 30ms minimum note
    max_gap: float = 0.05,  # merge notes within 50ms gap
) -> List[NoteEvent]:
    """
    Convert pitch estimation output (time, frequency, confidence arrays)
    to a list of MIDI note events.

    Groups consecutive pitched frames with the same MIDI note into note events.
    Filters by confidence, MIDI range, and minimum duration.
    """
    notes: List[NoteEvent] = []

    if len(times) == 0:
        return notes

    # Step size (time between consecutive frames)
    if len(times) > 1:
        step = float(np.median(np.diff(times)))
    else:
        step = 0.01

    # Build active-pitch segments
    current_note: Optional[int] = None
    current_onset: float = 0.0
    current_freqs: List[float] = []
    current_end: float = 0.0

    for i in range(len(times)):
        t = float(times[i])
        f = float(frequencies[i])
        c = float(confidences[i])

        if c >= confidence_threshold and f > 0:
            midi = hz_to_midi(f)
            if min_midi <= midi <= max_midi:
                if current_note is None:
                    # Start new note
                    current_note = midi
                    current_onset = t
                    current_freqs = [f]
                    current_end = t + step
                elif midi == current_note and (t - current_end) <= max_gap:
                    # Continue current note
                    current_freqs.append(f)
                    current_end = t + step
                else:
                    # Different note or gap too large — save current, start new
                    duration = current_end - current_onset
                    if duration >= min_note_duration:
                        notes.append(NoteEvent(
                            onset=current_onset,
                            offset=current_end,
                            midi_note=current_note,
                            frequency=float(np.mean(current_freqs)),
                        ))
                    current_note = midi
                    current_onset = t
                    current_freqs = [f]
                    current_end = t + step
            else:
                # Out of range — close current note
                if current_note is not None:
                    duration = current_end - current_onset
                    if duration >= min_note_duration:
                        notes.append(NoteEvent(
                            onset=current_onset,
                            offset=current_end,
                            midi_note=current_note,
                            frequency=float(np.mean(current_freqs)),
                        ))
                    current_note = None
        else:
            # Below confidence — close current note
            if current_note is not None:
                duration = current_end - current_onset
                if duration >= min_note_duration:
                    notes.append(NoteEvent(
                        onset=current_onset,
                        offset=current_end,
                        midi_note=current_note,
                        frequency=float(np.mean(current_freqs)),
                    ))
                current_note = None

    # Close any remaining note
    if current_note is not None:
        duration = current_end - current_onset
        if duration >= min_note_duration:
            notes.append(NoteEvent(
                onset=current_onset,
                offset=current_end,
                midi_note=current_note,
                frequency=float(np.mean(current_freqs)),
            ))

    return notes


def notes_to_midi(notes: List[NoteEvent], output_path: Path, tempo: float = 120.0) -> None:
    """Save a list of NoteEvents as a MIDI file using pretty_midi."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    bass = pretty_midi.Instrument(program=33, name="Electric Bass")  # GM: Electric Bass (finger)

    for note in notes:
        midi_note = pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.midi_note,
            start=note.onset,
            end=note.offset,
        )
        bass.notes.append(midi_note)

    pm.instruments.append(bass)
    pm.write(str(output_path))


# ---------------------------------------------------------------------------
# Algorithm 1: Basic Pitch
# ---------------------------------------------------------------------------

def transcribe_basic_pitch(
    audio_path: Path,
    output_path: Path,
    min_midi: int = 24,
    max_midi: int = 72,
) -> TranscriptionResult:
    """Transcribe bass using Spotify's Basic Pitch."""
    result = TranscriptionResult(algorithm="basic_pitch", stem_file=str(audio_path.name))

    try:
        # Block TF import to avoid multi-minute hang on ROCm TF builds.
        # Basic Pitch's __init__.py does `import tensorflow` at module load time.
        # We only need the ONNX backend anyway.
        import importlib
        _tf_blocked = False
        if "tensorflow" not in sys.modules:
            sys.modules["tensorflow"] = None  # type: ignore
            _tf_blocked = True

        from basic_pitch.inference import predict, Model
        import basic_pitch

        # Restore TF importability for other tools
        if _tf_blocked:
            del sys.modules["tensorflow"]

        # Find the ONNX model — the default ICASSP_2022_MODEL_PATH points to
        # the TF SavedModel dir which fails on ROCm TF builds.
        model_dir = Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022"
        onnx_path = model_dir / "nmp.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        # Pre-load the model so we measure inference time, not model loading
        bp_model = Model(str(onnx_path))

        t0 = time.perf_counter()
        model_output, midi_data, note_events = predict(
            str(audio_path),
            model_or_model_path=bp_model,
            minimum_frequency=float(2 ** ((min_midi - 69) / 12) * 440),
            maximum_frequency=float(2 ** ((max_midi - 69) / 12) * 440),
        )
        result.wall_time_s = time.perf_counter() - t0

        # Save MIDI (Basic Pitch returns a pretty_midi object directly)
        midi_data.write(str(output_path))
        result.midi_path = str(output_path)

        # Extract stats from note_events
        # note_events format: list of (onset, offset, pitch_midi, velocity, [pitch_bend])
        if note_events:
            pitches = [int(round(n[2])) for n in note_events]
            result.note_count = len(note_events)
            result.pitch_range = (min(pitches), max(pitches))
            result.total_note_duration_s = sum(n[1] - n[0] for n in note_events)
        else:
            result.note_count = 0

    except Exception as e:
        result.error = str(e)
        print(f"    [basic_pitch] ERROR: {e}")

    return result


# ---------------------------------------------------------------------------
# Algorithm 2: PESTO
# ---------------------------------------------------------------------------

def transcribe_pesto(
    audio_path: Path,
    output_path: Path,
    confidence_threshold: float = 0.5,
    min_midi: int = 24,
    max_midi: int = 72,
) -> TranscriptionResult:
    """Transcribe bass using Sony CSL's PESTO."""
    result = TranscriptionResult(algorithm="pesto", stem_file=str(audio_path.name))

    try:
        import torch
        import librosa
        import pesto

        # Load audio with librosa (torchaudio.load hits torchcodec issues)
        audio_np, sr = librosa.load(str(audio_path), sr=None, mono=True)
        waveform = torch.from_numpy(audio_np).float()

        t0 = time.perf_counter()
        timesteps, pitch, confidence, activations = pesto.predict(
            waveform, sr, step_size=10.0
        )
        result.wall_time_s = time.perf_counter() - t0

        # Convert to numpy
        times_np = timesteps.cpu().numpy() / 1000.0  # ms → seconds
        
        # PESTO returns pitch in Hz by default (convert_to_freq=True)
        freq_hz = pitch.cpu().numpy()
        conf_np = confidence.cpu().numpy()

        # Convert to MIDI notes
        notes = pitch_to_midi_notes(
            times_np, freq_hz, conf_np,
            confidence_threshold=confidence_threshold,
            min_midi=min_midi, max_midi=max_midi,
        )

        # Save MIDI
        notes_to_midi(notes, output_path)
        result.midi_path = str(output_path)

        # Stats
        result.note_count = len(notes)
        if notes:
            midi_pitches = [n.midi_note for n in notes]
            result.pitch_range = (min(midi_pitches), max(midi_pitches))
            result.total_note_duration_s = sum(n.offset - n.onset for n in notes)

    except Exception as e:
        result.error = str(e)
        print(f"    [pesto] ERROR: {e}")

    return result


# ---------------------------------------------------------------------------
# Algorithm 3: CREPE
# ---------------------------------------------------------------------------

def transcribe_crepe(
    audio_path: Path,
    output_path: Path,
    confidence_threshold: float = 0.5,
    min_midi: int = 24,
    max_midi: int = 72,
) -> TranscriptionResult:
    """Transcribe bass using CREPE."""
    result = TranscriptionResult(algorithm="crepe", stem_file=str(audio_path.name))

    try:
        import crepe
        from scipy.io import wavfile
        import librosa
        import soundfile as sf

        # Load audio — CREPE expects mono at original SR
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

        t0 = time.perf_counter()
        times_np, freq_hz, conf_np, activation = crepe.predict(
            audio, sr, viterbi=True, step_size=10, model_capacity="full",
        )
        result.wall_time_s = time.perf_counter() - t0

        # Convert to MIDI notes
        notes = pitch_to_midi_notes(
            times_np, freq_hz, conf_np,
            confidence_threshold=confidence_threshold,
            min_midi=min_midi, max_midi=max_midi,
        )

        # Save MIDI
        notes_to_midi(notes, output_path)
        result.midi_path = str(output_path)

        # Stats
        result.note_count = len(notes)
        if notes:
            midi_pitches = [n.midi_note for n in notes]
            result.pitch_range = (min(midi_pitches), max(midi_pitches))
            result.total_note_duration_s = sum(n.offset - n.onset for n in notes)

    except Exception as e:
        result.error = str(e)
        print(f"    [crepe] ERROR: {e}")

    return result


# ---------------------------------------------------------------------------
# Stem processing
# ---------------------------------------------------------------------------

def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds."""
    try:
        import librosa
        return float(librosa.get_duration(path=str(audio_path)))
    except Exception:
        return 0.0


def process_stem(
    audio_path: Path,
    output_dir: Path,
    confidence_threshold: float = 0.5,
) -> StemResult:
    """Run all 3 algorithms on a single bass stem."""
    stem_name = audio_path.stem  # filename without extension
    stem_result = StemResult(
        stem_file=str(audio_path.name),
        duration_s=get_audio_duration(audio_path),
    )

    algorithms = [
        ("basic_pitch", transcribe_basic_pitch),
        ("pesto", transcribe_pesto),
        ("crepe", transcribe_crepe),
    ]

    for algo_name, algo_fn in algorithms:
        midi_path = output_dir / f"{stem_name}_{algo_name}.mid"
        print(f"  → {algo_name}...", end=" ", flush=True)

        kwargs = {
            "audio_path": audio_path,
            "output_path": midi_path,
        }
        if algo_name != "basic_pitch":
            kwargs["confidence_threshold"] = confidence_threshold

        result = algo_fn(**kwargs)
        stem_result.results[algo_name] = result

        if result.error:
            print(f"FAILED ({result.error})")
        else:
            print(
                f"OK — {result.note_count} notes, "
                f"{result.wall_time_s:.2f}s, "
                f"range {result.pitch_range}"
            )

    return stem_result


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(stem_results: List[StemResult]) -> None:
    """Print a comparison table across all stems and algorithms."""
    algorithms = ["basic_pitch", "pesto", "crepe"]

    # Aggregate stats
    agg = {a: {"notes": 0, "time": 0.0, "dur": 0.0, "errors": 0, "stems": 0}
           for a in algorithms}

    for sr in stem_results:
        for algo in algorithms:
            r = sr.results.get(algo)
            if r:
                agg[algo]["stems"] += 1
                if r.error:
                    agg[algo]["errors"] += 1
                else:
                    agg[algo]["notes"] += r.note_count
                    agg[algo]["time"] += r.wall_time_s
                    agg[algo]["dur"] += r.total_note_duration_s

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Algorithm':<15} {'Stems':>6} {'Errors':>7} {'Notes':>7} "
          f"{'Total Time':>11} {'Avg Time/Stem':>14} {'Note Dur (s)':>13}")
    print("-" * 80)

    for algo in algorithms:
        a = agg[algo]
        ok = a["stems"] - a["errors"]
        avg_time = a["time"] / ok if ok > 0 else 0
        print(f"{algo:<15} {a['stems']:>6} {a['errors']:>7} {a['notes']:>7} "
              f"{a['time']:>10.2f}s {avg_time:>13.2f}s {a['dur']:>12.2f}s")

    print("=" * 80)

    # Per-stem table
    print(f"\nPer-Stem Detail ({len(stem_results)} stems):")
    print(f"{'Stem':<50} {'Algo':<15} {'Notes':>6} {'Time':>7} {'Range':>12}")
    print("-" * 95)

    for sr in stem_results:
        first = True
        label = sr.stem_file[:48]
        for algo in algorithms:
            r = sr.results.get(algo)
            if r:
                stem_col = label if first else ""
                first = False
                if r.error:
                    print(f"{stem_col:<50} {algo:<15} {'ERR':>6} {'':>7} {''}")
                else:
                    rng = f"{r.pitch_range[0]}-{r.pitch_range[1]}" if r.note_count > 0 else "n/a"
                    print(f"{stem_col:<50} {algo:<15} {r.note_count:>6} "
                          f"{r.wall_time_s:>6.2f}s {rng:>12}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bass Stem → MIDI Transcription Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="test_data/0286 Electric Universe - Alien Encounter 2",
        help="Directory containing bass stem MP3s",
    )
    parser.add_argument(
        "--output-dir",
        default="poc_bass_midi_output",
        help="Directory for output MIDI files",
    )
    parser.add_argument(
        "--output-json",
        default="poc_bass_transcription_results.json",
        help="Output JSON path for results",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float, default=0.3,
        help="Confidence threshold for PESTO/CREPE pitch-to-MIDI conversion",
    )
    parser.add_argument(
        "--max-stems",
        type=int, default=0,
        help="Max stems to process (0 = all)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_json = PROJECT_ROOT / args.output_json

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover bass stems
    bass_stems = sorted(input_dir.glob("*_bass.mp3"))
    if not bass_stems:
        print(f"ERROR: No *_bass.mp3 files found in {input_dir}")
        sys.exit(1)

    if args.max_stems > 0:
        bass_stems = bass_stems[:args.max_stems]

    print(f"Bass Stem → MIDI Transcription Benchmark")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Stems:  {len(bass_stems)}")
    print(f"  Confidence threshold: {args.confidence_threshold}")
    print()

    # Process each stem
    all_results: List[StemResult] = []
    for i, stem in enumerate(bass_stems, 1):
        print(f"[{i}/{len(bass_stems)}] {stem.name}")
        stem_result = process_stem(stem, output_dir, args.confidence_threshold)
        all_results.append(stem_result)

    # Save JSON results
    json_output = []
    for sr in all_results:
        entry = {
            "stem_file": sr.stem_file,
            "duration_s": sr.duration_s,
            "results": {},
        }
        for algo, r in sr.results.items():
            entry["results"][algo] = {
                "wall_time_s": r.wall_time_s,
                "note_count": r.note_count,
                "pitch_range": list(r.pitch_range),
                "total_note_duration_s": r.total_note_duration_s,
                "midi_path": r.midi_path,
                "error": r.error,
            }
        json_output.append(entry)

    with open(output_json, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"\nResults saved to {output_json}")

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
