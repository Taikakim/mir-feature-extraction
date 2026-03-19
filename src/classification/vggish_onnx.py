"""
MIGraphX-accelerated VGGish inference for all four Essentia VGGish classifiers.

Replaces TensorflowPredictVGGish with ONNX Runtime + MIGraphX EP for:
  - danceability-vggish-audioset-1
  - tonal_atonal-vggish-audioset-1
  - voice_instrumental-vggish-audioset-1
  - gender-vggish-audioset-1

All four models share the same preprocessing pipeline and architecture:

    16kHz mono audio
    → FrameGenerator(frameSize=400, hopSize=160)   [25ms frames, 10ms hop]
    → TensorflowInputVGGish per frame              → (64,) log-mel bands
    → Non-overlapping patches of 96 frames         → (n_patches, 96, 64)
    → ONNX + MIGraphX EP                           → (n_patches, 2) predictions

Output is numerically equivalent to TensorflowPredictVGGish (max diff < 1e-4).

JIT compile: ~28s on first run per model (MIGraphX). Subsequent calls: ~2ms per batch.
Batch size is fixed at _BATCH_SIZE=32; shorter batches are zero-padded.

NOTE: MIGraphX EP is disabled by default for VGGish.
On RDNA4 / ROCm 7.2 the MIGraphX backend crashes during JIT compilation with:
  Assertion `host or is_device_ptr(result.get())' failed  (hip.cpp:155)
This is a MIGraphX bug. EffNet and GMI are unaffected. Set
VGGISH_USE_MIGRAPHX=1 in the environment to re-enable if the bug is fixed.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# MIGraphX EP crashes during VGGish JIT compilation on RDNA4/ROCm 7.2
# (migraphx::gpu::write_to_gpu assertion failure). Default to CPU until fixed.
# Override with VGGISH_USE_MIGRAPHX=1 to re-enable.
_USE_MIGRAPHX = os.environ.get("VGGISH_USE_MIGRAPHX", "0").strip() == "1"

# Preprocessing constants — must match TensorflowPredictVGGish internals
_FRAME_SIZE = 400   # 25ms at 16kHz (TensorflowInputVGGish only accepts 400)
_HOP_SIZE   = 160   # 10ms at 16kHz
_PATCH_SIZE = 96    # 96 frames per patch (0.96s) — non-overlapping
_PATCH_HOP  = 96    # non-overlapping (TensorflowPredictVGGish default)
_BATCH_SIZE = 32    # fixed MIGraphX batch size — pad or chunk to this

# ONNX node names produced by tf2onnx from the Essentia .pb graphs
_INPUT_NAME  = "model/Placeholder:0"
_OUTPUT_NAME = "model/Sigmoid:0"

_CACHE_DIR = Path.home() / ".cache" / "mir"


class VGGishMIGraphX:
    """
    ONNX Runtime + MIGraphX wrapper for a single VGGish classifier.

    Interface mirrors TensorflowPredictVGGish:
        model = VGGishMIGraphX(model_path)
        preds = model(audio)   # audio: 16kHz mono float32 → (n_patches, 2)
    """

    def __init__(self, model_path: str | Path, model_key: str):
        import onnxruntime as ort
        import essentia.standard as es

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        compiled_path = _CACHE_DIR / f"vggish_{model_key}_bs{_BATCH_SIZE}.mxr"

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available = ort.get_available_providers()
        if _USE_MIGRAPHX and "MIGraphXExecutionProvider" in available:
            # migraphx_save/load_compiled_model were removed in ROCm 6.4+.
            # MIGraphX JIT-compiles on first call each session (~28s per model).
            provider_options = {"device_id": "0"}
            providers = [("MIGraphXExecutionProvider", provider_options), "CPUExecutionProvider"]
            logger.info(f"VGGish[{model_key}]: MIGraphX EP will JIT compile on first inference (~28s)")
        else:
            if not _USE_MIGRAPHX:
                logger.info(f"VGGish[{model_key}]: using CPU EP (MIGraphX disabled — set VGGISH_USE_MIGRAPHX=1 to enable)")
            else:
                logger.warning(f"VGGish[{model_key}]: MIGraphX EP not available — falling back to CPU")
            providers = ["CPUExecutionProvider"]

        self.session   = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
        self.model_key = model_key
        logger.info(f"VGGish[{model_key}]: active EPs = {self.session.get_providers()}")

        # Stateless Essentia algorithms reused across calls
        self._vggish_in = es.TensorflowInputVGGish()
        self._frame_fn  = es.FrameGenerator
        self._warmed_up = False

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _compute_patches(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert 16kHz mono audio to VGGish mel patches.

        Returns:
            (n_patches, 96, 64) float32 — ready for ONNX input
        """
        frame_gen  = self._frame_fn(audio, frameSize=_FRAME_SIZE, hopSize=_HOP_SIZE, startFromZero=True)
        mel_frames = np.array([self._vggish_in(frame) for frame in frame_gen])  # (n_frames, 64)

        if len(mel_frames) < _PATCH_SIZE:
            return np.empty((0, _PATCH_SIZE, 64), dtype=np.float32)

        patches = np.stack([
            mel_frames[i : i + _PATCH_SIZE]
            for i in range(0, len(mel_frames) - _PATCH_SIZE + 1, _PATCH_HOP)
        ])  # (n_patches, 96, 64)

        return patches.astype(np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Run one fixed-size batch through MIGraphX.

        Args:
            batch: (_BATCH_SIZE, 96, 64) float32
        Returns:
            (_BATCH_SIZE, 2) float32 — sigmoid probabilities
        """
        t0  = time.perf_counter()
        out = self.session.run([_OUTPUT_NAME], {_INPUT_NAME: batch})[0]
        elapsed = time.perf_counter() - t0

        if not self._warmed_up:
            logger.info(f"VGGish[{self.model_key}]: first inference (JIT compile) took {elapsed:.1f}s")
            self._warmed_up = True
        else:
            logger.debug(f"VGGish[{self.model_key}]: {elapsed * 1000:.1f}ms for batch_size={len(batch)}")

        return out  # (_BATCH_SIZE, 2)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Args:
            audio: 16kHz mono float32, shape (n_samples,)
        Returns:
            predictions: float32 (n_patches, 2)
                         same shape/semantics as TensorflowPredictVGGish output
        """
        patches = self._compute_patches(audio)
        if len(patches) == 0:
            return np.empty((0, 2), dtype=np.float32)

        results = []
        for i in range(0, len(patches), _BATCH_SIZE):
            chunk  = patches[i : i + _BATCH_SIZE]
            n_real = len(chunk)

            if n_real < _BATCH_SIZE:
                pad   = np.zeros((_BATCH_SIZE - n_real, _PATCH_SIZE, 64), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)

            preds = self._infer_batch(chunk)
            results.append(preds[:n_real])  # strip padding

        return np.concatenate(results, axis=0)  # (n_patches, 2)


# ---------------------------------------------------------------------------
# Module-level singletons — one ONNX session per model, shared across crops
# ---------------------------------------------------------------------------

_instances: Dict[str, VGGishMIGraphX] = {}

# Map short key → ONNX filename
_MODEL_FILES = {
    "danceability": "danceability-vggish-audioset-1.onnx",
    "atonality":    "tonal_atonal-vggish-audioset-1.onnx",
    "voice":        "voice_instrumental-vggish-audioset-1.onnx",
    "gender":       "gender-vggish-audioset-1.onnx",
}


def get_vggish_model(key: str, models_dir: str | Path) -> VGGishMIGraphX:
    """
    Get or create the shared VGGishMIGraphX instance for the given model key.

    Args:
        key: one of 'danceability', 'atonality', 'voice', 'gender'
        models_dir: directory containing the .onnx files
    """
    global _instances
    if key not in _instances:
        filename = _MODEL_FILES[key]
        model_path = Path(models_dir) / filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"VGGish ONNX model not found: {model_path}\n"
                f"Run: python src/classification/vggish_onnx.py --convert"
            )
        _instances[key] = VGGishMIGraphX(model_path, model_key=key)
    return _instances[key]


def preload_all_vggish(models_dir: str | Path,
                       include_voice: bool = True,
                       include_gender: bool = True) -> int:
    """
    Pre-load and JIT-compile all requested VGGish ONNX models.

    Returns number of models successfully loaded.
    """
    keys = ["danceability", "atonality"]
    if include_voice:
        keys.append("voice")
    if include_gender:
        keys.append("gender")

    loaded = 0
    for key in keys:
        try:
            get_vggish_model(key, models_dir)
            loaded += 1
        except Exception as e:
            logger.warning(f"VGGish ONNX preload failed for '{key}': {e}")
    return loaded


def unload_vggish_models() -> None:
    """Release all VGGish ONNX sessions (call after Essentia pass to free GPU memory)."""
    global _instances
    count = len(_instances)
    _instances.clear()
    if count:
        logger.info(f"VGGishMIGraphX: {count} sessions unloaded")


# ---------------------------------------------------------------------------
# Conversion helper + numerical verification (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="VGGish ONNX tools")
    parser.add_argument("--convert", action="store_true",
                        help="Convert .pb models to .onnx (requires tf2onnx)")
    parser.add_argument("--verify", metavar="AUDIO",
                        help="Verify ONNX vs TF on an audio file")
    parser.add_argument("--models-dir", default="models/essentia")
    args = parser.parse_args()

    from core.common import setup_logging
    setup_logging(level="INFO")

    if args.convert:
        import tensorflow as tf
        import tf2onnx

        MODELS = [
            ("danceability-vggish-audioset-1.pb",       "danceability-vggish-audioset-1.onnx"),
            ("tonal_atonal-vggish-audioset-1.pb",        "tonal_atonal-vggish-audioset-1.onnx"),
            ("voice_instrumental-vggish-audioset-1.pb",  "voice_instrumental-vggish-audioset-1.onnx"),
            ("gender-vggish-audioset-1.pb",              "gender-vggish-audioset-1.onnx"),
        ]
        base = Path(args.models_dir)
        for pb_name, onnx_name in MODELS:
            pb_path   = base / pb_name
            onnx_path = base / onnx_name
            print(f"Converting {pb_name} ...")
            with tf.io.gfile.GFile(str(pb_path), "rb") as f:
                gd = tf.compat.v1.GraphDef()
                gd.ParseFromString(f.read())
            tf2onnx.convert.from_graph_def(
                gd,
                input_names=["model/Placeholder:0"],
                output_names=["model/Sigmoid:0"],
                output_path=str(onnx_path),
                opset=13,
            )
            print(f"  -> {onnx_path}")

    if args.verify:
        import essentia.standard as es

        audio_path = args.verify
        loader = es.MonoLoader(filename=audio_path, sampleRate=16000)
        audio  = loader()
        print(f"Audio: {audio_path}  ({len(audio)/16000:.2f}s)")

        from classification.essentia_features import get_model_path
        models_dir = Path(args.models_dir)

        for key, pb_key, tf_out in [
            ("danceability", "danceability-vggish-audioset-1.pb",      None),
            ("atonality",    "tonal_atonal-vggish-audioset-1.pb",       "model/Sigmoid"),
            ("voice",        "voice_instrumental-vggish-audioset-1.pb", "model/Sigmoid"),
            ("gender",       "gender-vggish-audioset-1.pb",             "model/Sigmoid"),
        ]:
            # TF reference
            kw = dict(graphFilename=str(models_dir / pb_key))
            if tf_out:
                kw["output"] = tf_out
            tf_model  = es.TensorflowPredictVGGish(**kw)
            tf_preds  = np.asarray(tf_model(audio))
            tf_mean   = np.mean(tf_preds, axis=0)

            # ONNX
            onnx_model = get_vggish_model(key, models_dir)
            onnx_preds = onnx_model(audio)
            onnx_mean  = np.mean(onnx_preds, axis=0)

            diff = np.max(np.abs(tf_mean - onnx_mean))
            print(f"\n{key}:")
            print(f"  TF   mean: {tf_mean}")
            print(f"  ONNX mean: {onnx_mean}")
            print(f"  Max diff:  {diff:.2e}  {'OK' if diff < 1e-2 else 'FAIL'}")
