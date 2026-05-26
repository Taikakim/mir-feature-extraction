"""
Whole-track 100 Hz time-series extraction for LatCH control-head targets.

Unlike spectral/timeseries_features.py (which bins one fixed crop to n_steps),
this module extracts every feature across the ENTIRE track at a canonical frame
rate (default 100 Hz) and stores one .TIMESERIES.npz sidecar per track. A
training crop is then a [offset : offset+window] slice, resampled at load time
to whatever VAE latent rate the consumer needs:

    Stable Audio Open Small / SA1 : 44100 / 2048 = 21.533 Hz
    Stable Audio 3 medium         : 44100 / 4096 = 10.767 Hz

100 Hz is madmom's native fps, so the rhythmic activations below are stored
RAW with no producer-side resampling. Smoothing / standardization / resampling
to the latent grid are all train-time knobs on the consumer side.

Fields (all at frame_rate, length == n_frames; hpcp is (n_frames, 12)):

  Full mix — rhythmic (the LatCH targets):
    beat_activation_ts      -- madmom RNNBeatProcessor, per-frame P(beat) [0,1]
    downbeat_activation_ts  -- madmom RNNDownBeatProcessor downbeat column [0,1]
    onset_envelope_ts       -- librosa onset_strength, raw (un-normalised)

  Full mix — continuous:
    rms_energy_{bass,body,mid,air}_ts
    spectral_{flatness,flux,skewness,kurtosis}_ts
    hpcp_ts  (n_frames, 12)

  Per stem (drums, bass, other, vocals) — the per-stem rhythmic/activity targets:
    onset_envelope_{stem}_ts  -- raw librosa onset_strength on the stem
    rms_{stem}_ts             -- broadband per-frame RMS (dB) of the stem

Beat/downbeat are full-mix concepts, so they are NOT computed per stem.
Per-frame tonic/key is intentionally omitted (meaningless at 10 ms resolution
and prohibitively slow over a whole track) — store hpcp and derive key
downstream over coarse windows if needed.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_utils import read_audio
from spectral.timeseries_features import (
    _compute_multiband_rms_ts,
    _compute_spectral_ts,
)
from spectral.multiband_rms import calculate_rms_db

logger = logging.getLogger(__name__)

FRAME_RATE_DEFAULT = 100
STEMS = ["drums", "bass", "other", "vocals"]
_STEM_EXTS = (".flac", ".wav", ".mp3")

try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_len(arr: np.ndarray, n: int) -> np.ndarray:
    """Trim or edge-pad *arr* along axis 0 to exactly *n* frames."""
    cur = arr.shape[0]
    if cur == n:
        return arr
    if cur > n:
        return arr[:n]
    pad = [(0, n - cur)] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pad, mode="edge")


def _onset_envelope(audio: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Raw librosa onset-strength envelope at the canonical hop (un-normalised)."""
    import librosa
    return librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop).astype(np.float32)


def _rms_envelope(audio: np.ndarray, hop: int) -> np.ndarray:
    """Broadband per-frame RMS (dB) at the canonical hop."""
    n_frames = max(1, len(audio) // hop)
    out = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        out[i] = calculate_rms_db(audio[i * hop:(i + 1) * hop])
    return out


def _hpcp_ts(audio: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Per-frame HPCP (n_frames, 12), L-inf normalised per frame. Essentia."""
    frame_size = min(8192, max(4096, hop * 2))
    frame_gen = es.FrameGenerator(audio.astype(np.float32), frameSize=frame_size,
                                  hopSize=hop, startFromZero=True)
    window = es.Windowing(type="hann", size=frame_size)
    spectrum = es.Spectrum(size=frame_size)
    peaks = es.SpectralPeaks(maxPeaks=100, magnitudeThreshold=1e-5,
                             sampleRate=sr, orderBy="magnitude")
    hpcp = es.HPCP(size=12, harmonics=8, minFrequency=40.0, maxFrequency=5000.0,
                   weightType="cosine", nonLinear=True, normalized="unitMax",
                   sampleRate=sr)
    frames: List[np.ndarray] = []
    for frame in frame_gen:
        freqs, mags = peaks(spectrum(window(frame)))
        frames.append(np.asarray(hpcp(freqs, mags), dtype=np.float32))
    if not frames:
        return np.zeros((0, 12), dtype=np.float32)
    return np.stack(frames, axis=0)


def _madmom_activations(audio_path: Path, n_frames: int,
                        beat_proc, downbeat_proc) -> Dict[str, np.ndarray]:
    """Raw madmom beat + downbeat activation functions, fitted to n_frames."""
    out: Dict[str, np.ndarray] = {}
    try:
        beat_act = np.asarray(beat_proc(str(audio_path)), dtype=np.float32)
        out["beat_activation_ts"] = _fit_len(beat_act.reshape(-1), n_frames)
    except Exception as e:
        logger.warning(f"  madmom beat activation failed: {e}")
    try:
        down_act = np.asarray(downbeat_proc(str(audio_path)), dtype=np.float32)
        # RNNDownBeatProcessor → (N, 2): col 0 = beat, col 1 = downbeat prob.
        downbeat = down_act[:, 1] if down_act.ndim == 2 else down_act.reshape(-1)
        out["downbeat_activation_ts"] = _fit_len(downbeat.astype(np.float32), n_frames)
    except Exception as e:
        logger.warning(f"  madmom downbeat activation failed: {e}")
    return out


def find_stem_files(track_dir: Path) -> Dict[str, Path]:
    """Map each available stem name to its audio file in *track_dir*."""
    found: Dict[str, Path] = {}
    for stem in STEMS:
        for ext in _STEM_EXTS:
            p = track_dir / f"{stem}{ext}"
            if p.exists():
                found[stem] = p
                break
    return found


def find_full_mix(track_dir: Path) -> Optional[Path]:
    for ext in _STEM_EXTS:
        p = track_dir / f"full_mix{ext}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _patch_madmom_compat() -> None:
    """madmom 0.16.1 predates py3.10 (collections) and numpy>=1.24 (np.float).

    Without this, `import madmom` raises and the rest of the pipeline silently
    falls back to librosa beats — so the soft activations never get computed.
    """
    import collections
    import collections.abc
    for n in ("MutableSequence", "MutableMapping", "Sequence", "Mapping",
              "Iterable", "Callable"):
        if not hasattr(collections, n):
            setattr(collections, n, getattr(collections.abc, n))
    for n, t in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, n):
            setattr(np, n, t)


def make_madmom_processors() -> Tuple[object, object]:
    """Instantiate the (beat, downbeat) RNN processors once for batch reuse."""
    _patch_madmom_compat()
    from madmom.features.beats import RNNBeatProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor
    return RNNBeatProcessor(), RNNDownBeatProcessor()


def extract_whole_track(
    track_dir: Path,
    frame_rate: int = FRAME_RATE_DEFAULT,
    beat_proc=None,
    downbeat_proc=None,
    do_hpcp: bool = True,
    stems: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Extract all whole-track time-series for one track folder at *frame_rate*.

    Returns (data, meta).  data maps field name → np.ndarray; every full-mix
    and per-stem array is fitted to the same n_frames so a single crop offset
    slices all features consistently.
    """
    track_dir = Path(track_dir)
    full_mix = find_full_mix(track_dir)
    if full_mix is None:
        raise FileNotFoundError(f"No full_mix.* in {track_dir}")

    audio, sr = read_audio(str(full_mix))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    hop = round(sr / frame_rate)
    duration = len(audio) / sr
    n_frames = round(duration * frame_rate)

    data: Dict[str, np.ndarray] = {}

    # --- full-mix continuous (reuse the crop-binning primitives at 1:1) ------
    for k, v in _compute_multiband_rms_ts(audio, sr, n_frames).items():
        data[k] = _fit_len(np.asarray(v, dtype=np.float32), n_frames)
    for k, v in _compute_spectral_ts(audio, sr, n_frames, hop_length=hop).items():
        data[k] = _fit_len(np.asarray(v, dtype=np.float32), n_frames)

    data["onset_envelope_ts"] = _fit_len(_onset_envelope(audio, sr, hop), n_frames)

    if do_hpcp and ESSENTIA_AVAILABLE:
        data["hpcp_ts"] = _fit_len(_hpcp_ts(audio, sr, hop), n_frames)
    elif do_hpcp:
        logger.warning("  Essentia unavailable; skipping hpcp_ts")

    # --- full-mix rhythmic activations (raw madmom, native 100 fps) ----------
    if beat_proc is not None and downbeat_proc is not None:
        data.update(_madmom_activations(full_mix, n_frames, beat_proc, downbeat_proc))
    else:
        logger.warning("  madmom processors not supplied; skipping beat/downbeat activations")

    # --- per stem: onset envelope + broadband RMS ----------------------------
    stem_files = find_stem_files(track_dir)
    use_stems = stems if stems is not None else STEMS
    stems_present: List[str] = []
    for stem in use_stems:
        path = stem_files.get(stem)
        if path is None:
            continue
        s_audio, s_sr = read_audio(str(path))
        if s_audio.ndim > 1:
            s_audio = s_audio.mean(axis=1)
        s_audio = s_audio.astype(np.float32)
        s_hop = round(s_sr / frame_rate)
        data[f"onset_envelope_{stem}_ts"] = _fit_len(
            _onset_envelope(s_audio, s_sr, s_hop), n_frames)
        data[f"rms_{stem}_ts"] = _fit_len(
            _rms_envelope(s_audio, s_hop), n_frames)
        stems_present.append(stem)

    meta = {
        "frame_rate": frame_rate,
        "n_frames": n_frames,
        "duration": duration,
        "sample_rate": sr,
        "hop": hop,
        "stems_present": stems_present,
        "source": str(full_mix),
        "fields": sorted(data.keys()),
    }
    return data, meta


def save_timeseries_npz(out_path: Path, data: Dict[str, np.ndarray], meta: Dict) -> None:
    """Write {field: array} + JSON meta to a compressed .npz sidecar."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.astype(np.float32) for k, v in data.items()}
    payload["__meta__"] = np.array(json.dumps(meta))
    np.savez_compressed(str(out_path), **payload)


def load_timeseries_npz(path: Path) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Inverse of save_timeseries_npz."""
    with np.load(str(path), allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        data = {k: z[k] for k in z.files if k != "__meta__"}
    return data, meta


# ---------------------------------------------------------------------------
# Batch CLI
# ---------------------------------------------------------------------------

def _iter_track_dirs(root: Path):
    for child in sorted(root.iterdir()):
        if child.is_dir() and find_full_mix(child) is not None:
            yield child


# Per-worker state (set by _worker_init in each process).
_WORKER_BEAT = None
_WORKER_DOWNBEAT = None
_WORKER_CFG: Dict = {}


def _apply_rocm_env(yaml_path: str, profile: str) -> None:
    """Apply ROCm/MIOpen env from the SAT rocm_env.yaml before any torch import.

    Inert for the current CPU-only features (madmom/librosa/essentia), but in
    place for future GPU-backed timeseries features. profile 'none' skips.
    Uses setdefault, so shell exports still win.
    """
    if not profile or profile == "none":
        return
    p = Path(yaml_path)
    if not p.exists():
        logger.warning(f"ROCm env yaml not found: {p}; skipping --rocm-profile {profile}")
        return
    import yaml
    cfg = yaml.safe_load(p.read_text()) or {}
    root = str(cfg.get("tunings_root", ""))
    env = dict(cfg.get("common", {}))
    env.update(cfg.get("profiles", {}).get(profile, {}))
    for k, v in env.items():
        os.environ.setdefault(k, str(v).replace("${tunings_root}", root))


def _worker_init(frame_rate: int, do_hpcp: bool,
                 rocm_yaml: str = "", rocm_profile: str = "none") -> None:
    _apply_rocm_env(rocm_yaml, rocm_profile)   # before any (future) torch import
    global _WORKER_BEAT, _WORKER_DOWNBEAT, _WORKER_CFG
    _WORKER_BEAT, _WORKER_DOWNBEAT = make_madmom_processors()
    _WORKER_CFG = {"frame_rate": frame_rate, "do_hpcp": do_hpcp}


def _process_one(job: Tuple[str, str, bool]) -> Tuple[str, str, object]:
    """Extract + save one track. Returns (name, status, info). Picklable for pools."""
    track_dir, out_path, overwrite = job
    name = Path(track_dir).name
    if Path(out_path).exists() and not overwrite:
        return name, "skip", None
    t0 = time.time()
    try:
        data, meta = extract_whole_track(
            Path(track_dir), frame_rate=_WORKER_CFG["frame_rate"],
            beat_proc=_WORKER_BEAT, downbeat_proc=_WORKER_DOWNBEAT,
            do_hpcp=_WORKER_CFG["do_hpcp"])
        save_timeseries_npz(Path(out_path), data, meta)
        return name, "ok", {"n_frames": meta["n_frames"], "n_fields": len(data),
                            "stems": meta["stems_present"], "elapsed": time.time() - t0}
    except Exception as e:
        return name, "fail", str(e)
    finally:
        import gc
        gc.collect()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract whole-track 100 Hz time-series sidecars (.TIMESERIES.npz).")
    parser.add_argument("root", type=Path, help="Root dir of <track>/ folders with full_mix + stems")
    parser.add_argument("--frame-rate", type=int, default=FRAME_RATE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write .npz (default: inside each track folder)")
    parser.add_argument("--no-hpcp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N tracks")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (default 1). Each builds its own "
                             "madmom + essentia stack; BLAS threads are pinned to 1 per worker.")
    parser.add_argument("--max-tasks-per-child", type=int, default=8,
                        help="Recycle each worker after N tracks to release leaked RSS "
                             "(essentia/madmom C-extensions don't free memory between tracks). "
                             "Lower = tighter memory ceiling, slightly more respawn overhead.")
    parser.add_argument("--rocm-profile", choices=["none", "inference", "training"],
                        default="training",
                        help="Apply ROCm/MIOpen env (MIOPEN_FIND_MODE etc.) from "
                             "--rocm-env-yaml before torch import. No-op for the current "
                             "CPU-only features; in place for future GPU-backed ones. "
                             "'none' to skip.")
    parser.add_argument("--rocm-env-yaml",
                        default="/home/kim/Projects/SAO/stable-audio-tools/rocm_env.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")

    track_dirs = list(_iter_track_dirs(args.root))
    if args.limit:
        track_dirs = track_dirs[:args.limit]
    jobs: List[Tuple[str, str, bool]] = []
    for td in track_dirs:
        out = (args.output_dir / f"{td.name}.TIMESERIES.npz") if args.output_dir \
            else (td / f"{td.name}.TIMESERIES.npz")
        jobs.append((str(td), str(out), args.overwrite))
    print(f"Found {len(jobs)} track folders under {args.root}; {args.workers} worker(s)")

    done = skipped = failed = processed = 0

    def _report(name: str, status: str, info) -> None:
        nonlocal done, skipped, failed, processed
        processed += 1
        if status == "skip":
            skipped += 1
        elif status == "ok":
            done += 1
            print(f"[{processed}/{len(jobs)}] {name}: {info['n_frames']} frames, "
                  f"{info['n_fields']} fields, stems={info['stems']} ({info['elapsed']:.1f}s)")
        else:
            failed += 1
            print(f"[{processed}/{len(jobs)}] {name}: FAILED — {info}")

    if args.workers <= 1:
        # Single process: apply the full rocm profile (incl. OMP_NUM_THREADS=8).
        _worker_init(args.frame_rate, not args.no_hpcp,
                     args.rocm_env_yaml, args.rocm_profile)
        for job in jobs:
            _report(*_process_one(job))
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool
        # Pin BLAS HARD so N workers don't oversubscribe the CPU (spawn children
        # inherit env). Hard-set (not setdefault) to override rocm_env.yaml's
        # OMP_NUM_THREADS=8, which is for single-process GPU training, not the
        # 12-way CPU pool; workers re-apply the rocm profile via setdefault, so
        # this 1 wins for the BLAS vars while MIOPEN_FIND_MODE=6 etc. still apply.
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[v] = "1"
        # spawn (not fork): essentia pulls in TensorFlow, which deadlocks under fork.
        ctx = mp.get_context("spawn")

        def _pending(js):
            # A job is done once its output npz exists (unless overwriting).
            return [j for j in js if j[2] or not Path(j[1]).exists()]

        # A single worker dying (native crash in madmom/essentia, transient drive
        # I/O) raises BrokenProcessPool and kills the whole pool. Restart on a fresh
        # pool with only the still-missing jobs; if a restart makes zero progress the
        # head job is poison, so skip it to stay unattended.
        remaining = _pending(jobs)
        prev_remaining = None
        restart = 0
        while remaining:
            try:
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                                         max_tasks_per_child=args.max_tasks_per_child,
                                         initializer=_worker_init,
                                         initargs=(args.frame_rate, not args.no_hpcp,
                                                   args.rocm_env_yaml, args.rocm_profile)) as ex:
                    futs = [ex.submit(_process_one, j) for j in remaining]
                    for fut in as_completed(futs):
                        _report(*fut.result())
                break
            except BrokenProcessPool:
                restart += 1
                new_remaining = _pending(remaining)
                print(f"\n[pool restart {restart}] worker died abruptly — "
                      f"{len(remaining) - len(new_remaining)} done this round, "
                      f"{len(new_remaining)} remaining", flush=True)
                if new_remaining and new_remaining == prev_remaining:
                    poison = Path(new_remaining[0][0]).name
                    print(f"  No progress since last restart; quarantining: {poison}", flush=True)
                    failed += 1
                    new_remaining = new_remaining[1:]
                prev_remaining = list(new_remaining)
                remaining = new_remaining
                if restart > 100:
                    print("  Too many restarts; aborting.", flush=True)
                    break

    print(f"\nDone: {done} written, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
