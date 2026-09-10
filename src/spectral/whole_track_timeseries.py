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

Entry points:
    extract_whole_track()  one track  → (data, meta)
    run_batch()            a corpus   → {"done","skipped","failed","failures"};
                           the importable batch runner (chunked fresh pools,
                           per-chunk timeout, quarantine, orphan reap). Callable
                           in-process — it does not mutate the caller's env.
    main()                 thin argparse wrapper over run_batch().
"""

import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# essentia-tensorflow drags in TensorFlow, which probes for NVIDIA CUDA on this
# AMD/ROCm box and spams "Could not load libcudart / failed call to cuInit". We
# only use essentia's CPU HPCP, never TF — silence its logs and skip the GPU
# probe. Must be set before essentia is imported (below, transitively).
#
# TF_CPP_MIN_LOG_LEVEL is a log-level only (changes no capability), so it is safe
# to set at module scope. CUDA_VISIBLE_DEVICES is NOT: this module is imported
# in-process by master_pipeline.py, whose later stages use the GPU, and an empty
# visible-device list can crash flash_attn at import (MASTER §5). So the CUDA
# hiding is applied in exactly two places instead:
#   * script mode (below) — when run as the CLI, `__name__ == "__main__"` is
#     already true while the module body executes, i.e. before essentia/TF is
#     imported a few lines down. Preserves the old CLI behaviour exactly.
#   * worker processes — run_batch() exports it into os.environ only for the
#     window in which the spawn children are created (they snapshot the parent
#     env at spawn, before importing anything), then restores the caller's env.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Env applied to the worker processes (and to the CLI's own process). See above.
_WORKER_ENV_SETDEFAULT = {"CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "3"}
# Pin BLAS HARD so N workers don't oversubscribe the CPU. Hard-set (not
# setdefault) to override rocm_env.yaml's OMP_NUM_THREADS=8, which is for
# single-process GPU training, not this CPU pool; workers re-apply the rocm
# profile via setdefault, so this 1 wins for the BLAS vars while
# MIOPEN_FIND_MODE=6 etc. still apply.
_WORKER_ENV_FORCE = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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
_STEM_EXTS = (".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aiff")

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
    """Write {field: array} + JSON meta to a compressed .npz sidecar, ATOMICALLY.

    tmp + os.replace, same as whole_track_expanded.merge_expanded. Writing in
    place is not safe here: the resume gate treats "sidecar exists" as "track
    done", so a run interrupted mid-write (chunk timeout force-kill, Ctrl-C,
    OOM) used to leave a truncated .npz that every later run then skipped.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.astype(np.float32) for k, v in data.items()}
    payload["__meta__"] = np.array(json.dumps(meta))
    # tmp name must END in .npz — np.savez appends the extension otherwise,
    # leaving 'x.tmp.npz' while os.replace looks for 'x.tmp' (pilot bug 2026-07-14)
    tmp = out_path.parent / (out_path.stem + ".tmp.npz")
    try:
        np.savez_compressed(str(tmp), **payload)
        os.replace(tmp, out_path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def load_timeseries_npz(path: Path) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Inverse of save_timeseries_npz."""
    with np.load(str(path), allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        data = {k: z[k] for k in z.files if k != "__meta__"}
    return data, meta


# ---------------------------------------------------------------------------
# Batch CLI
# ---------------------------------------------------------------------------

def stem_fields_stale(track_dir: Path, out_path: Path) -> Optional[str]:
    """Is an EXISTING sidecar missing base per-stem fields that we could compute now?

    Returns None when the sidecar is fine, else a short human-readable reason.

    Why this exists (the 2026-08-17 gotcha, now enforced instead of documented):
    a track first extracted with only `full_mix` present gets NO per-stem fields
    (`onset_envelope_{stem}_ts`, `rms_{stem}_ts` — 8 fields for 4 stems). If the
    stems land later, plain existence-gating marks the track done forever, and
    `--add-fields` cannot rescue it either: its missing-check is scoped to
    whole_track_expanded.EXPANDED_FIELDS and never even reports the base fields.
    So the gate also inspects the sidecar's `__meta__["stems_present"]` and the
    actual array names, and asks for a FULL re-extract when stems exist on disk
    now but are absent from the sidecar.

    An unreadable/truncated sidecar also counts as stale (it is not usable, and
    with the pre-atomic-write save it could be a half-written file).
    """
    stems_now = set(find_stem_files(Path(track_dir)))
    try:
        with np.load(str(out_path), allow_pickle=False) as z:
            have = set(z.files)
            meta = json.loads(str(z["__meta__"])) if "__meta__" in have else {}
    except Exception as e:
        return f"unreadable sidecar ({type(e).__name__})"
    if not stems_now:
        return None
    stems_then = set(meta.get("stems_present", []))
    missing = sorted(
        s for s in stems_now
        if s not in stems_then
        or f"onset_envelope_{s}_ts" not in have
        or f"rms_{s}_ts" not in have
    )
    if missing:
        return f"stems on disk but not in sidecar: {','.join(missing)}"
    return None


def _stem_recheck_enabled(default: bool = True) -> bool:
    """Escape hatch for the stems-aware resume gate: MIR_WT_STEM_RECHECK=0."""
    v = os.environ.get("MIR_WT_STEM_RECHECK")
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


def _iter_track_dirs(root: Path, recursive: bool = False):
    """Yield track dirs (those containing a full_mix.*). With recursive=True,
    walk the whole tree so nested augmentation variant folders
    (<track>/augmentations/<variant>/full_mix.flac) are found too."""
    if recursive:
        for dirpath, _dirnames, _filenames in os.walk(root):
            d = Path(dirpath)
            if find_full_mix(d) is not None:
                yield d
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and find_full_mix(child) is not None:
            yield child


# Per-worker state (set by _worker_init in each process).
_WORKER_BEAT = None
_WORKER_DOWNBEAT = None
_WORKER_CFG: Dict = {}
_WORKER_EXPANDED = None


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
                 rocm_yaml: str = "", rocm_profile: str = "none",
                 expanded: bool = False, add_fields: bool = False,
                 stem_recheck: bool = True) -> None:
    _apply_rocm_env(rocm_yaml, rocm_profile)   # before any (future) torch import
    global _WORKER_BEAT, _WORKER_DOWNBEAT, _WORKER_CFG, _WORKER_EXPANDED
    _WORKER_BEAT, _WORKER_DOWNBEAT = make_madmom_processors()
    _WORKER_CFG = {"frame_rate": frame_rate, "do_hpcp": do_hpcp,
                   "expanded": expanded, "add_fields": add_fields,
                   "stem_recheck": stem_recheck}
    if expanded or add_fields:
        from spectral.whole_track_expanded import ExpandedExtractor
        _WORKER_EXPANDED = ExpandedExtractor()   # TF models load lazily on 1st use


def _run_expanded(track_dir: Path, data: Dict, meta: Dict,
                  wanted=None) -> None:
    """Compute expanded fields for a freshly-extracted track and fold them
    into (data, meta) in place."""
    full_mix = find_full_mix(track_dir)
    new, rates, extra = _WORKER_EXPANDED.extract(
        full_mix, wanted=wanted, existing=data, existing_meta=meta)
    data.update(new)
    meta["fields"] = sorted(data.keys())
    meta.setdefault("field_rates", {}).update(rates)
    meta.setdefault("expanded", {}).update(extra)


def _process_one(job: Tuple[str, str, bool]) -> Tuple[str, str, object]:
    """Extract + save one track. Returns (name, status, info). Picklable for pools."""
    track_dir, out_path, overwrite = job
    name = Path(track_dir).name
    t0 = time.time()
    # Stems-aware resume: an existing sidecar written before its stems landed is
    # NOT done — the base per-stem fields can only come from a full re-extract
    # (--add-fields cannot add them; see stem_fields_stale). Force overwrite so
    # both the add-fields branch and the plain-existence branch below re-run.
    stale = None
    if not overwrite and Path(out_path).exists() and _WORKER_CFG.get("stem_recheck", True):
        stale = stem_fields_stale(Path(track_dir), Path(out_path))
        if stale:
            logger.warning(f"  {name}: re-extracting — {stale}")
            overwrite = True
    if _WORKER_CFG.get("add_fields") and Path(out_path).exists() and not overwrite:
        # incremental mode on an existing sidecar: compute only missing
        # expanded fields, merge atomically (legacy fields untouched)
        from spectral.whole_track_expanded import (
            merge_expanded, missing_expanded_fields)
        missing = missing_expanded_fields(Path(out_path))
        if not missing:
            return name, "skip", None
        try:
            data, meta = load_timeseries_npz(Path(out_path))
            full_mix = find_full_mix(Path(track_dir))
            if full_mix is None:
                return name, "fail", "no full_mix for add-fields"
            new, rates, extra = _WORKER_EXPANDED.extract(
                full_mix, wanted=missing, existing=data, existing_meta=meta)
            merge_expanded(Path(out_path), new, rates, extra)
            return name, "ok", {"n_frames": meta["n_frames"], "n_fields": len(new),
                                "stems": ["+add"], "elapsed": time.time() - t0}
        except Exception as e:
            return name, "fail", str(e)
        finally:
            import gc
            gc.collect()
    if Path(out_path).exists() and not overwrite:
        return name, "skip", None
    try:
        data, meta = extract_whole_track(
            Path(track_dir), frame_rate=_WORKER_CFG["frame_rate"],
            beat_proc=_WORKER_BEAT, downbeat_proc=_WORKER_DOWNBEAT,
            do_hpcp=_WORKER_CFG["do_hpcp"])
        if _WORKER_CFG.get("expanded") or _WORKER_CFG.get("add_fields"):
            _run_expanded(Path(track_dir), data, meta)
        save_timeseries_npz(Path(out_path), data, meta)
        return name, "ok", {"n_frames": meta["n_frames"], "n_fields": len(data),
                            "stems": meta["stems_present"], "elapsed": time.time() - t0}
    except Exception as e:
        return name, "fail", str(e)
    finally:
        import gc
        gc.collect()


@contextlib.contextmanager
def _scoped_env(setdefault: Dict[str, str], force: Dict[str, str]):
    """Apply env vars for the duration of the block, then restore EXACTLY.

    Spawn children snapshot os.environ at creation time, so the parent has to
    carry these vars while the pools are built. But master_pipeline.py calls
    run_batch() in-process and its later stages need its own env back:
    core.rocm_env sets OMP_NUM_THREADS=8 for the GPU stages, and an empty
    visible-device list can crash flash_attn at import (MASTER §5). Hence
    set-then-restore rather than a module-scope or main()-scope mutation.
    """
    saved: Dict[str, Optional[str]] = {}
    try:
        for k, v in setdefault.items():
            if k not in os.environ:
                saved[k] = None
                os.environ[k] = v
        for k, v in force.items():
            if os.environ.get(k) != v:
                saved[k] = os.environ.get(k)
                os.environ[k] = v
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def run_batch(
    root: Path,
    *,
    output_dir: Optional[Path] = None,
    expanded: bool = True,
    add_fields: bool = False,
    overwrite: bool = False,
    workers: int = 4,
    chunk_size: int = 48,
    chunk_timeout: int = 1800,
    frame_rate: float = 100.0,
    hpcp: bool = True,
    recursive: bool = False,
    folders: Optional[List[Path]] = None,
    progress_cb: Optional[Callable[[str, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    # --- extras beyond the pipeline-facing contract (used by the CLI) --------
    limit: Optional[int] = None,
    rocm_yaml: str = "",
    rocm_profile: str = "none",
    stems_aware_resume: bool = True,
    print_progress: bool = False,
) -> dict:
    """Run the whole-track timeseries extraction over a corpus. Importable.

    This is main()'s former body: chunked fresh pools, per-chunk timeout,
    poison-job quarantine, orphan reap. main() is now a thin argparse wrapper.

    Args:
        root: corpus root of <track>/ folders (each with full_mix.* [+ stems]).
        output_dir: where to write the .npz. None (the DEFAULT) writes
            in-folder: <track_dir>/<track_dir.name>.TIMESERIES.npz.
        expanded: also compute whole_track_expanded's field set.
        add_fields: incremental mode — for existing sidecars compute ONLY the
            missing EXPANDED fields and merge them in.
        overwrite: re-extract even where a sidecar exists.
        workers: worker processes. <=1 runs in-process (no pool).
        chunk_size: tracks per fresh pool (the essentia/madmom RSS-leak knob).
        chunk_timeout: seconds before a stuck chunk's workers are force-killed.
        frame_rate / hpcp: extraction knobs.
        recursive: walk the whole tree (nested augmentation variant folders).
        folders: explicit track folders; when given, discovery under root is
            skipped (root is then only used for logging).
        progress_cb: called (track_name, status) after every job, where status
            is "ok" | "skip" | "fail". Exceptions from it are logged, not fatal.
        should_stop: polled between chunks (and between jobs when workers<=1);
            True stops cleanly — skip-existing makes the next run resume.
        limit: process at most N tracks.
        rocm_yaml / rocm_profile: ROCm env applied inside the workers.
            Defaults to "none" here (an in-process caller owns its own GPU env);
            the CLI keeps its historical "training" default.
        stems_aware_resume: treat a sidecar written before its stems landed as
            NOT done and re-extract it fully (see stem_fields_stale). Default
            on; overridable per-call, or globally with MIR_WT_STEM_RECHECK=0.
        print_progress: emit the CLI's per-track stdout lines.

    Returns:
        {"done": int, "skipped": int, "failed": int,
         "failures": [(track_name, error_str), ...]}
    """
    root = Path(root)
    if folders is not None:
        track_dirs = [Path(f) for f in folders]
    else:
        track_dirs = list(_iter_track_dirs(root, recursive))
    if limit:
        track_dirs = track_dirs[:limit]

    jobs: List[Tuple[str, str, bool]] = []
    for td in track_dirs:
        out = (Path(output_dir) / f"{td.name}.TIMESERIES.npz") if output_dir \
            else (td / f"{td.name}.TIMESERIES.npz")
        jobs.append((str(td), str(out), overwrite))

    stem_recheck = stems_aware_resume and _stem_recheck_enabled()

    def _say(msg: str) -> None:
        """Operational messages: stdout for the CLI, the logger for callers."""
        if print_progress:
            print(msg, flush=True)
        else:
            logger.warning(msg)

    if print_progress:
        print(f"Found {len(jobs)} track folders under {root}; {workers} worker(s)")

    done = skipped = failed = processed = 0
    failures: List[Tuple[str, str]] = []

    def _report(name: str, status: str, info) -> None:
        nonlocal done, skipped, failed, processed
        processed += 1
        if status == "skip":
            skipped += 1
        elif status == "ok":
            done += 1
            if print_progress:
                print(f"[{processed}/{len(jobs)}] {name}: {info['n_frames']} frames, "
                      f"{info['n_fields']} fields, stems={info['stems']} ({info['elapsed']:.1f}s)")
        else:
            failed += 1
            failures.append((name, str(info)))
            if print_progress:
                print(f"[{processed}/{len(jobs)}] {name}: FAILED — {info}")
            else:
                logger.warning(f"{name}: FAILED — {info}")
        if progress_cb is not None:
            try:
                progress_cb(name, status)
            except Exception as e:      # a bad callback must not kill the run
                logger.warning(f"progress_cb raised for {name}: {e}")

    initargs = (frame_rate, hpcp, rocm_yaml, rocm_profile,
                expanded, add_fields, stem_recheck)

    _announced: set = set()

    def _pending(js):
        # A job is done once its output npz exists (unless overwriting).
        # In add-fields mode "done" = sidecar exists AND has every
        # expanded field (existence alone would mark everything done).
        # And in EITHER mode a sidecar written before this track's stems
        # existed is NOT done — only a full re-extract can add the 8 base
        # per-stem fields, so it is reported (once) and re-queued.
        out = []
        for j in js:
            if j[2] or not Path(j[1]).exists():
                out.append(j)
                continue
            if add_fields:
                from spectral.whole_track_expanded import missing_expanded_fields
                if missing_expanded_fields(Path(j[1])):
                    out.append(j)
                    continue
            if stem_recheck:
                reason = stem_fields_stale(Path(j[0]), Path(j[1]))
                if reason:
                    if j[1] not in _announced:
                        _announced.add(j[1])
                        _say(f"  re-extract queued: {Path(j[0]).name} — {reason}")
                    out.append(j)
                    continue
        return out

    # Track the pool processes WE created, so the final reap cannot touch an
    # unrelated multiprocessing child of an in-process caller.
    spawned: List = []

    with _scoped_env(_WORKER_ENV_SETDEFAULT,
                     _WORKER_ENV_FORCE if workers > 1 else {}):
        if workers <= 1:
            # Single process: apply the full rocm profile (incl. OMP_NUM_THREADS=8).
            _worker_init(*initargs)
            # Iterate ALL jobs (not _pending): _process_one does the skip check
            # itself, and its "skip" returns are what feed the skipped count.
            for job in jobs:
                if should_stop is not None and should_stop():
                    _say("  stop requested — halting (skip-existing resumes)")
                    break
                _report(*_process_one(job))
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            from concurrent.futures import TimeoutError as FTimeout
            from concurrent.futures.process import BrokenProcessPool
            # spawn (not fork): essentia pulls in TensorFlow, which deadlocks under fork.
            ctx = mp.get_context("spawn")

            # Chunked FRESH pools instead of one long-lived pool with
            # max_tasks_per_child: in-pool worker recycling wedged (all N workers hit
            # the recycle boundary together and the TF/essentia re-import deadlocked).
            # A fresh pool per chunk bounds memory (full teardown releases the
            # essentia/madmom leak) with no in-pool respawn to hang. A per-chunk
            # as_completed timeout force-kills workers if any single track hangs, and
            # skip-existing makes every chunk independently resumable.
            pending = _pending(jobs)
            while pending:
                if should_stop is not None and should_stop():
                    _say("  stop requested — halting between chunks "
                         "(skip-existing resumes)")
                    break
                batch = pending[:chunk_size]
                ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                         initializer=_worker_init,
                                         initargs=initargs)
                try:
                    futs = [ex.submit(_process_one, j) for j in batch]
                    for fut in as_completed(futs, timeout=chunk_timeout):
                        _report(*fut.result())
                except FTimeout:
                    _say(f"\n[chunk timeout >{chunk_timeout}s] a task hung — "
                         f"killing workers, continuing (skip-existing resumes)")
                    for p in list(getattr(ex, "_processes", {}).values()):
                        try:
                            p.kill()
                        except Exception:
                            pass
                except BrokenProcessPool:
                    _say("\n[pool broke] worker died — continuing (skip-existing resumes)")
                finally:
                    spawned.extend(getattr(ex, "_processes", {}).values())
                    ex.shutdown(wait=False, cancel_futures=True)

                new_pending = _pending(pending)
                if new_pending and len(new_pending) == len(pending):
                    # Zero progress this chunk → head job is poison; skip to stay unattended.
                    poison = Path(new_pending[0][0]).name
                    _say(f"  No progress this chunk; quarantining: {poison}")
                    failed += 1
                    failures.append((poison, "quarantined: no progress in chunk"))
                    if progress_cb is not None:
                        try:
                            progress_cb(poison, "fail")
                        except Exception as e:
                            logger.warning(f"progress_cb raised for {poison}: {e}")
                    new_pending = new_pending[1:]
                pending = new_pending

    # REAP OUR POOLS. `shutdown(wait=False)` above is deliberate -- a stuck worker must not
    # be able to hang a multi-hour run -- but it means the final chunk's workers are still alive
    # when the run returns, and they re-parent to init instead of dying. Measured after the
    # 4461-track f0 backfill: 8 orphans at ~420 MB each, 2.4 GB held indefinitely by processes
    # whose parent no longer existed. The run had printed "Done:" and given the shell back, so
    # nothing suggested a third of the box's spare RAM was still spoken for. Same family as
    # MASTER §5's orphaned-dataloader-worker note, in the MIR producer.
    # Safe here: every chunk has completed, and save_timeseries_npz writes atomically
    # (tmp + os.replace), so a terminated worker can at worst leave a .tmp.npz file,
    # never a truncated sidecar that the resume gate would mistake for a finished one.
    # We only touch processes from OUR executors (`spawned`), never the caller's other
    # multiprocessing children -- run_batch is called in-process by master_pipeline.py.
    reaped = 0
    for p in spawned:
        try:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                reaped += 1
        except Exception:
            pass
    if reaped:
        _say(f"  reaped {reaped} pool worker(s)")

    return {"done": done, "skipped": skipped, "failed": failed, "failures": failures}


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
    parser.add_argument("--add-fields", action="store_true",
                        help="Incremental expanded-fields mode (2026-07-14 sweep): "
                             "for tracks whose sidecar already exists, compute ONLY "
                             "the missing expanded fields (whole_track_expanded.py) "
                             "and merge them in — legacy fields are never recomputed. "
                             "Tracks with no sidecar get a full extraction (legacy + "
                             "expanded). Resumable: done = all expanded fields present.")
    parser.add_argument("--expanded", action="store_true",
                        help="Also compute the expanded field set on full "
                             "extractions (implied by --add-fields).")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N tracks")
    parser.add_argument("--recursive", action="store_true",
                        help="Walk the whole tree so nested augmentation variant "
                             "folders (<track>/augmentations/<variant>/) are processed too.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (default 1). Each builds its own "
                             "madmom + essentia stack; BLAS threads are pinned to 1 per worker.")
    parser.add_argument("--chunk-size", type=int, default=48,
                        help="Tracks per fresh pool. The pool is torn down and rebuilt "
                             "each chunk to release the essentia/madmom RSS leak without "
                             "in-pool worker recycling (which deadlocked). Smaller = "
                             "tighter memory ceiling, more pool-respawn overhead (~25s each).")
    parser.add_argument("--chunk-timeout", type=float, default=2400.0,
                        help="Seconds before a stuck chunk's workers are force-killed and "
                             "the run continues (skip-existing resumes). Generous vs the "
                             "~chunk_size/workers * 80s expected chunk time.")
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

    stats = run_batch(
        args.root,
        output_dir=args.output_dir,
        expanded=args.expanded,
        add_fields=args.add_fields,
        overwrite=args.overwrite,
        workers=args.workers,
        chunk_size=args.chunk_size,
        chunk_timeout=args.chunk_timeout,
        frame_rate=args.frame_rate,
        hpcp=not args.no_hpcp,
        recursive=args.recursive,
        limit=args.limit,
        rocm_yaml=args.rocm_env_yaml,
        rocm_profile=args.rocm_profile,
        print_progress=True,
    )
    print(f"\nDone: {stats['done']} written, {stats['skipped']} skipped, "
          f"{stats['failed']} failed")


if __name__ == "__main__":
    main()
