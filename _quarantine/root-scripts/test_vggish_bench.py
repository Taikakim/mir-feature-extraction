"""
VGGish TF vs ONNX benchmark.

Runs 3 audio files through all four VGGish classifiers using both the original
TensorflowPredictVGGish and the ONNX+MIGraphX wrapper.  Reports per-file wall
times and prediction agreement so the JIT compilation overhead is clearly visible
on the first ONNX call.

Usage:
    # Auto-find three crops from the output dataset:
    python test_vggish_bench.py

    # Explicit files:
    python test_vggish_bench.py /path/a.flac /path/b.flac /path/c.flac

    # Use CPU EP instead of MIGraphX:
    VGGISH_USE_MIGRAPHX=0 python test_vggish_bench.py
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.rocm_env import setup_rocm_env
setup_rocm_env()

import numpy as np
import essentia.standard as es

MODELS_DIR = Path("models/essentia")

MODELS = [
    ("danceability", "danceability-vggish-audioset-1.pb",      "danceability-vggish-audioset-1.onnx",      None),
    ("atonality",    "tonal_atonal-vggish-audioset-1.pb",       "tonal_atonal-vggish-audioset-1.onnx",       "model/Sigmoid"),
    ("voice",        "voice_instrumental-vggish-audioset-1.pb", "voice_instrumental-vggish-audioset-1.onnx", "model/Sigmoid"),
    ("gender",       "gender-vggish-audioset-1.pb",             "gender-vggish-audioset-1.onnx",             "model/Sigmoid"),
]


def find_test_files(n: int = 3) -> list[Path]:
    """Auto-find n crop audio files from the configured output directory."""
    import yaml
    config_path = Path("config/master_pipeline.yaml")
    search_dirs = []
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        out = cfg.get("paths", {}).get("output")
        if out:
            search_dirs.append(Path(out))
        inp = cfg.get("paths", {}).get("input")
        if inp:
            search_dirs.append(Path(inp))

    for d in search_dirs:
        files = sorted(d.rglob("full_mix.flac"))[:n]
        if len(files) >= 1:
            return files[:n]

    raise RuntimeError(
        "Could not find test files automatically. Pass file paths as arguments."
    )


def load_audio(path: Path) -> np.ndarray:
    loader = es.MonoLoader(filename=str(path), sampleRate=16000)
    return loader()


def run_tf(audio: np.ndarray, pb_path: Path, tf_out: str | None) -> tuple[np.ndarray, float]:
    kw = dict(graphFilename=str(pb_path))
    if tf_out:
        kw["output"] = tf_out
    model = es.TensorflowPredictVGGish(**kw)
    t0 = time.perf_counter()
    preds = np.asarray(model(audio))
    elapsed = time.perf_counter() - t0
    return preds, elapsed


def run_onnx(audio: np.ndarray, key: str) -> tuple[np.ndarray, float]:
    from classification.vggish_onnx import get_vggish_model
    model = get_vggish_model(key, MODELS_DIR)
    t0 = time.perf_counter()
    preds = model(audio)
    elapsed = time.perf_counter() - t0
    return preds, elapsed


def fmt_time(s: float) -> str:
    if s >= 1.0:
        return f"{s:.2f}s"
    return f"{s*1000:.0f}ms"


def fmt_xrt(proc_time: float, audio_dur: float) -> str:
    """xRT = audio_duration / proc_time  (>1 = faster than real-time)."""
    if proc_time <= 0:
        return "  —"
    xrt = audio_dur / proc_time
    return f"{xrt:.1f}xRT"


def main():
    parser = argparse.ArgumentParser(description="VGGish TF vs ONNX benchmark")
    parser.add_argument("files", nargs="*", help="Audio files to test (default: auto-find 3)")
    parser.add_argument("--models-dir", default="models/essentia")
    args = parser.parse_args()

    global MODELS_DIR
    MODELS_DIR = Path(args.models_dir)

    if args.files:
        audio_paths = [Path(p) for p in args.files]
    else:
        print("No files specified — searching dataset for crops...")
        audio_paths = find_test_files(3)

    print(f"\nTest files ({len(audio_paths)}):")
    for p in audio_paths:
        print(f"  {p}")

    # Check ONNX env
    import os
    migraphx_flag = os.environ.get("VGGISH_USE_MIGRAPHX", "0")
    print(f"\nVGGISH_USE_MIGRAPHX={migraphx_flag}  "
          f"({'MIGraphX EP' if migraphx_flag == '1' else 'CPU EP'})")

    # Pre-load audio (not counted in timing)
    print("\nLoading audio...")
    audios = []
    durations = []  # seconds at 16kHz
    for p in audio_paths:
        a = load_audio(p)
        dur = len(a) / 16000
        print(f"  {p.name:40s}  {dur:.1f}s")
        audios.append(a)
        durations.append(dur)

    # -------------------------------------------------------------------------
    # TF reference — instantiate models once, run per file
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TF (TensorflowPredictVGGish) — reference")
    print("=" * 70)

    tf_times  = {key: [] for key, *_ in MODELS}
    tf_means  = {key: [] for key, *_ in MODELS}

    for key, pb_name, _, tf_out in MODELS:
        pb_path = MODELS_DIR / pb_name
        if not pb_path.exists():
            print(f"  SKIP {key} — .pb not found: {pb_path}")
            continue
        print(f"\n  {key}:")
        for i, (audio, dur) in enumerate(zip(audios, durations)):
            preds, t = run_tf(audio, pb_path, tf_out)
            m = np.mean(preds, axis=0)
            tf_times[key].append(t)
            tf_means[key].append(m)
            print(f"    file {i+1}: {fmt_time(t):8s}  {fmt_xrt(t, dur):>10}  mean={np.array2string(m, precision=4)}")

    # -------------------------------------------------------------------------
    # ONNX — singletons persist across files (shows JIT on first call)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ONNX (vggish_onnx.py)")
    print("=" * 70)

    onnx_times = {key: [] for key, *_ in MODELS}
    onnx_means = {key: [] for key, *_ in MODELS}

    for key, _, onnx_name, _ in MODELS:
        onnx_path = MODELS_DIR / onnx_name
        if not onnx_path.exists():
            print(f"  SKIP {key} — .onnx not found: {onnx_path}")
            continue
        # First call triggers JIT compilation
        print(f"\n  {key}:")
        for i, (audio, dur) in enumerate(zip(audios, durations)):
            preds, t = run_onnx(audio, key)
            m = np.mean(preds, axis=0)
            onnx_times[key].append(t)
            onnx_means[key].append(m)
            jit_note = " ← JIT compile" if i == 0 else ""
            print(f"    file {i+1}: {fmt_time(t):8s}  {fmt_xrt(t, dur):>10}  mean={np.array2string(m, precision=4)}{jit_note}")

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    n = len(audios)
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Column headers
    col_w = 16  # width per file column pair (time + xRT)
    print(f"{'Model':<14}", end="")
    for i in range(n):
        label = f"TF f{i+1}"
        print(f"  {label:>{col_w}}", end="")
    for i in range(n):
        label = f"ONNX f{i+1}" + (" (JIT)" if i == 0 else "")
        print(f"  {label:>{col_w}}", end="")
    print(f"  {'speedup':>10}  {'max_diff':>10}")
    print("-" * 90)

    for key, *_ in MODELS:
        if not tf_times[key] or not onnx_times[key]:
            continue

        print(f"{key:<14}", end="")

        # TF columns: time + xRT
        for i, (t, dur) in enumerate(zip(tf_times[key], durations)):
            cell = f"{fmt_time(t)} {fmt_xrt(t, dur)}"
            print(f"  {cell:>{col_w}}", end="")

        # ONNX columns: time + xRT
        for i, (t, dur) in enumerate(zip(onnx_times[key], durations)):
            cell = f"{fmt_time(t)} {fmt_xrt(t, dur)}"
            print(f"  {cell:>{col_w}}", end="")

        # Speedup: ONNX steady-state (files 2+) vs TF average
        tf_avg = sum(tf_times[key]) / len(tf_times[key])
        onnx_steady = onnx_times[key][1:] if len(onnx_times[key]) > 1 else onnx_times[key]
        onnx_avg = sum(onnx_steady) / len(onnx_steady)
        pct = (tf_avg - onnx_avg) / tf_avg * 100
        speedup_str = f"{pct:+.0f}%" if abs(pct) < 1000 else f"{tf_avg/onnx_avg:.1f}x"

        # Max diff
        diffs = [np.max(np.abs(tf_means[key][i] - onnx_means[key][i]))
                 for i in range(min(len(tf_means[key]), len(onnx_means[key])))]
        max_diff = max(diffs) if diffs else float("nan")
        status = "OK" if max_diff < 1e-2 else "FAIL"
        print(f"  {speedup_str:>10}  {max_diff:>8.2e} {status}")

    print()
    print("speedup = (TF_avg - ONNX_steady) / TF_avg  (positive = ONNX faster)")
    print()


if __name__ == "__main__":
    main()
