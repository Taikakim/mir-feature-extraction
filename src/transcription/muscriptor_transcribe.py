"""
MuScriptor polyphonic audio-to-MIDI transcription (OPT-IN pipeline stage).

Writes one General-MIDI sidecar per track folder, following the same
`<track_dir>/<track_dir.name>.<EXT>` convention as `.BEATS_GRID` / `.ONSETS` /
`.DOWNBEATS`:

    <track_dir>/<track_dir.name>.MID

The model is a transformer LM over MT3-style note tokens; it consumes mono
float32 @ 16 kHz in 5-second chunks and emits note-on/note-off events which are
reassembled into a complete MIDI file.

WHY THIS STAGE IS OPT-IN AND DEFAULT-DISABLED
---------------------------------------------
Transcription costs a median of ~76 s per ~7-minute track on the RX 9070 XT
(~5-6x realtime). That is an order of magnitude more than any other track-level
feature in this pipeline, so `transcription.enabled` defaults to **false** in
config/master_pipeline.yaml and must be turned on deliberately.

CRITICAL DATA CAVEAT — DO NOT READ VELOCITY OR TEMPO FROM THIS MIDI
-------------------------------------------------------------------
The emitted MIDI carries a FIXED velocity of 100 on every note and a FIXED
tempo of 120 BPM. Both are placeholders written by the serializer, NOT
measurements of the audio. Any downstream musicology feature must therefore:

  * never derive dynamics / accent / loudness features from note velocity;
  * never derive tempo, beat positions or bar lines from the MIDI tempo map —
    tempo comes from mir's own BPM fields (`bpm_madmom` / `bpm_essentia`) and
    beat times from the `.BEATS_GRID` / `.DOWNBEATS` sidecars.

Pitch, onset time, offset time (hence duration), and instrument/program are the
only trustworthy dimensions.

FULL MIX, NOT STEMS
-------------------
Pilot evidence: the model transcribes full mixes better than separated stems
(separation artefacts confuse the acoustic front-end). This stage therefore
runs on `full_mix.*` only.

LICENSING
---------
muscriptor's code is MIT, but its published model WEIGHTS are CC BY-NC 4.0
(non-commercial). MIDI produced with these weights inherits that restriction —
keep it to research/analysis use.

Usage
-----
    # Batch over a directory of track folders
    python src/transcription/muscriptor_transcribe.py /path/to/Goa_Separated \\
        --model medium --workers 1

    # Library
    from transcription.muscriptor_transcribe import MuScriptorTranscriber
    tr = MuScriptorTranscriber(model="medium")   # loads the model ONCE
    for folder in folders:
        tr.transcribe_folder(folder)
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rocm_env import setup_rocm_env

setup_rocm_env()

import torch  # noqa: E402  — must follow setup_rocm_env()

from core.file_utils import read_audio  # noqa: E402

logger = logging.getLogger(__name__)

# Sidecar extension, mirroring .BEATS_GRID / .ONSETS / .DOWNBEATS.
MIDI_EXT = ".MID"

# muscriptor lives outside mir's site-packages; it is importable straight from
# its checkout with no install (verified against mir/bin/python). Keep the path
# in one place so the error message below can name it.
MUSCRIPTOR_PATH = Path(
    os.environ.get("MUSCRIPTOR_PATH", "/home/kim/Projects/muscriptor")
)

# Extensions searched for the full mix, in the same order as
# spectral/whole_track_timeseries.py's find_full_mix().
_AUDIO_EXTS = (".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aiff")

# muscriptor's internal working sample rate. Resampling here (instead of letting
# the model do it) lets us reuse mir's read_audio(), which handles m4a/AAC that
# soundfile alone cannot open.
MUSCRIPTOR_SR = 16000


# ---------------------------------------------------------------------------
# Import shim
# ---------------------------------------------------------------------------

def _import_muscriptor():
    """Import muscriptor.TranscriptionModel, adding its checkout to sys.path.

    Raises ImportError naming the expected path if the package is absent, so a
    misconfigured box fails loudly rather than silently skipping the stage.
    """
    try:
        from muscriptor import TranscriptionModel
        return TranscriptionModel
    except ImportError:
        pass

    if MUSCRIPTOR_PATH.is_dir() and str(MUSCRIPTOR_PATH) not in sys.path:
        sys.path.insert(0, str(MUSCRIPTOR_PATH))

    try:
        from muscriptor import TranscriptionModel
        return TranscriptionModel
    except ImportError as exc:
        raise ImportError(
            f"muscriptor is not importable. Expected the package checkout at "
            f"{MUSCRIPTOR_PATH} (override with the MUSCRIPTOR_PATH env var). "
            f"It needs no install — the directory must simply contain a "
            f"'muscriptor/' package dir. Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def find_full_mix(track_dir: Path) -> Optional[Path]:
    """Return <track_dir>/full_mix.<ext> for the first extension that exists."""
    for ext in _AUDIO_EXTS:
        p = track_dir / f"full_mix{ext}"
        if p.exists():
            return p
    return None


def get_midi_path(track_dir: Path) -> Path:
    """The `.MID` sidecar path for a track folder."""
    return track_dir / f"{track_dir.name}{MIDI_EXT}"


def iter_track_dirs(root: Path, recursive: bool = False) -> Iterable[Path]:
    """Yield track folders (dirs containing a full_mix.*) under *root*."""
    if find_full_mix(root) is not None:
        yield root
        return
    walker = root.rglob("*") if recursive else root.iterdir()
    for entry in sorted(walker):
        if entry.is_dir() and find_full_mix(entry) is not None:
            yield entry


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------

class MuScriptorTranscriber:
    """Loads the muscriptor model ONCE and transcribes many files.

    Per mir's development rules, GPU models are loaded once for batch
    processing — never per file. The model is loaded lazily on first use so
    constructing the object is cheap (and so a disabled stage costs nothing).
    """

    def __init__(self,
                 model: Optional[str] = "medium",
                 device: Optional[str] = None,
                 instruments: Optional[List[str]] = None,
                 batch_size: Optional[int] = None,
                 beam_size: int = 1):
        """
        Args:
            model: "small" / "medium" / "large", a local .safetensors path, an
                   hf:// URL, or None (= muscriptor's default, medium).
            device: Torch device string. None => cuda if available else cpu.
            instruments: Restrict the decoded instrument set (muscriptor's exact
                   instrument names). None = all instruments.
            batch_size: Chunk batch size. None = muscriptor's default (4 on cuda).
            beam_size: Beam width for decoding (1 = greedy).
        """
        self.model_name = model
        self.device = device
        self.instruments = instruments
        self.batch_size = batch_size
        self.beam_size = beam_size
        self._model = None

    # -- model lifecycle ---------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            TranscriptionModel = _import_muscriptor()
            dev = self.device
            if dev is None:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading muscriptor model '{self.model_name}' on {dev} "
                        f"(first call downloads/caches weights)")
            t0 = time.time()
            self._model = TranscriptionModel.load_model(
                weights_path=self.model_name, device=dev)
            logger.info(f"muscriptor model loaded in {time.time() - t0:.1f}s")
        return self._model

    def unload(self) -> None:
        """Free the model and its VRAM (for pipelines that hand the GPU on)."""
        self._model = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- transcription -----------------------------------------------------

    def transcribe_audio_file(self, audio_path: Path) -> bytes:
        """Transcribe one audio file, returning complete MIDI file bytes.

        Audio is read through mir's read_audio() (m4a/AAC-capable), folded to
        mono and handed to muscriptor as a (tensor, sample_rate) pair, which it
        resamples internally to 16 kHz.
        """
        import numpy as np

        audio, sr = read_audio(audio_path)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        tensor = torch.from_numpy(np.ascontiguousarray(audio))
        return self.model.transcribe_to_midi(
            (tensor, int(sr)),
            instruments=self.instruments,
            batch_size=self.batch_size,
            beam_size=self.beam_size,
        )

    def transcribe_folder(self, track_dir: Path,
                          overwrite: bool = False) -> Tuple[str, str, object]:
        """Transcribe <track_dir>/full_mix.* to <track_dir>/<name>.MID.

        Returns (folder_name, status, info) where status is one of
        'success' | 'skipped' | 'failed', matching the tuple contract of
        core.pipeline_workers.process_folder_rhythm().
        """
        track_dir = Path(track_dir)
        out_path = get_midi_path(track_dir)

        if out_path.exists() and not overwrite:
            return track_dir.name, "skipped", "MIDI already exists"

        full_mix = find_full_mix(track_dir)
        if full_mix is None:
            return track_dir.name, "failed", "no full_mix.* in folder"

        t0 = time.time()
        try:
            midi_bytes = self.transcribe_audio_file(full_mix)
        except Exception as exc:
            logger.debug(f"muscriptor {track_dir.name}: {exc}", exc_info=True)
            return track_dir.name, "failed", str(exc)

        if not midi_bytes:
            return track_dir.name, "failed", "transcriber returned no MIDI bytes"

        # Atomic write: a killed worker must never leave a truncated .MID that
        # the resume check would then treat as done.
        tmp = out_path.parent / (out_path.name + ".tmp")
        tmp.write_bytes(midi_bytes)
        os.replace(tmp, out_path)

        return track_dir.name, "success", {
            "bytes": len(midi_bytes),
            "elapsed": time.time() - t0,
        }


# ---------------------------------------------------------------------------
# Single-file convenience function
# ---------------------------------------------------------------------------

def transcribe_track_folder(track_dir: Path,
                            model: Optional[str] = "medium",
                            device: Optional[str] = None,
                            instruments: Optional[List[str]] = None,
                            overwrite: bool = False,
                            transcriber: Optional[MuScriptorTranscriber] = None
                            ) -> Tuple[str, str, object]:
    """Transcribe a single track folder to its `.MID` sidecar.

    Pass an existing *transcriber* when processing many folders — constructing
    one per folder would reload the model per file, which mir's rules forbid.
    """
    tr = transcriber or MuScriptorTranscriber(
        model=model, device=device, instruments=instruments)
    return tr.transcribe_folder(track_dir, overwrite=overwrite)


def count_midi_notes(midi_path: Path) -> int:
    """Parse a .MID back and count note-on events (verification helper)."""
    try:
        import mido
    except ImportError:
        return -1
    mid = mido.MidiFile(str(midi_path))
    return sum(1 for msg in mid
               if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe track folders to MIDI sidecars (<track>.MID) "
                    "with muscriptor. Costs ~5-6x realtime on GPU.")
    parser.add_argument("root", type=Path,
                        help="Root dir of <track>/ folders containing full_mix.*, "
                             "or a single track folder")
    parser.add_argument("--model", default="medium",
                        help="small | medium | large | local .safetensors path "
                             "| hf:// URL (default: medium)")
    parser.add_argument("--device", default=None,
                        help="Torch device (default: cuda if available else cpu)")
    parser.add_argument("--instruments", default=None,
                        help="Comma-separated muscriptor instrument names to "
                             "restrict decoding to (default: all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-transcribe folders that already have a .MID")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N track folders")
    parser.add_argument("--recursive", action="store_true",
                        help="Walk the whole tree so nested variant folders are found")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes. The model is GPU-bound "
                             "and holds several GB of VRAM, so 1 is correct on a "
                             "single card; >1 only makes sense with --device cpu.")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Chunks decoded per batch (default: muscriptor's, 4 on cuda)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")

    instruments = ([s.strip() for s in args.instruments.split(",") if s.strip()]
                   if args.instruments else None)

    track_dirs = list(iter_track_dirs(args.root, args.recursive))
    if not args.overwrite:
        track_dirs = [d for d in track_dirs if not get_midi_path(d).exists()]
    if args.limit:
        track_dirs = track_dirs[:args.limit]

    print(f"Found {len(track_dirs)} track folder(s) to transcribe under {args.root}")
    if not track_dirs:
        return

    done = skipped = failed = 0
    t_start = time.time()

    def _report(i: int, name: str, status: str, info) -> None:
        nonlocal done, skipped, failed
        if status == "skipped":
            skipped += 1
            print(f"[{i}/{len(track_dirs)}] {name}: skipped ({info})")
        elif status == "success":
            done += 1
            print(f"[{i}/{len(track_dirs)}] {name}: {info['bytes']} bytes "
                  f"({info['elapsed']:.1f}s)")
        else:
            failed += 1
            print(f"[{i}/{len(track_dirs)}] {name}: FAILED — {info}")

    if args.workers <= 1:
        tr = MuScriptorTranscriber(model=args.model, device=args.device,
                                   instruments=instruments,
                                   batch_size=args.batch_size,
                                   beam_size=args.beam_size)
        for i, td in enumerate(track_dirs, 1):
            _report(i, *tr.transcribe_folder(td, overwrite=args.overwrite))
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # spawn (not fork): torch/HIP contexts do not survive fork.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futs = {ex.submit(_worker_transcribe,
                              (str(td), args.model, args.device, instruments,
                               args.overwrite, args.batch_size, args.beam_size)): td
                    for td in track_dirs}
            for i, fut in enumerate(as_completed(futs), 1):
                _report(i, *fut.result())

    elapsed = time.time() - t_start
    print(f"\nDone: {done} written, {skipped} skipped, {failed} failed "
          f"in {elapsed:.1f}s")


# Module-level worker: each spawned process builds its OWN transcriber, so the
# model is still loaded once per PROCESS (not once per file).
_WORKER_TR: Optional[MuScriptorTranscriber] = None


def _worker_transcribe(job) -> Tuple[str, str, object]:
    global _WORKER_TR
    (track_dir, model, device, instruments, overwrite, batch_size, beam_size) = job
    if _WORKER_TR is None:
        _WORKER_TR = MuScriptorTranscriber(model=model, device=device,
                                           instruments=instruments,
                                           batch_size=batch_size,
                                           beam_size=beam_size)
    return _WORKER_TR.transcribe_folder(Path(track_dir), overwrite=overwrite)


if __name__ == "__main__":
    main()
