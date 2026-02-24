"""
MIGraphX-accelerated Genre/Mood/Instrument classifiers for Discogs-EffNet embeddings.

Replaces TensorflowPredict2D with ONNX Runtime + MIGraphX EP for the three
classification heads that run on top of EffNet embeddings:

  - genre_discogs400-discogs-effnet-1       (batch, 1280) → (batch, 400 classes)
  - mtg_jamendo_moodtheme-discogs-effnet-1  (batch, 1280) → (batch,  56 classes)
  - mtg_jamendo_instrument-discogs-effnet-1 (batch, 1280) → (batch,  40 classes)

These are small MLP models (~2-3 MB each); the heavyweight work is already done by
EffNetMIGraphX (which produces the 1280-dim embeddings). Converting these too removes
the last TensorFlow dependency from the Essentia pipeline.

JIT compile: ~28s per model on first inference (MIGraphX compiles gfx1201 kernels).
Subsequent calls with the same batch size: <1ms.
Batch size is fixed at _BATCH_SIZE=64; shorter inputs are zero-padded.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_BATCH_SIZE = 64          # fixed MIGraphX batch size — pad or chunk to this
_EMBED_DIM  = 1280        # EffNet output dimensionality

# ONNX file names and their (input_node, output_node, n_classes)
_MODEL_SPECS = {
    "genre": (
        "genre_discogs400-discogs-effnet-1.onnx",
        "serving_default_model_Placeholder:0",
        "PartitionedCall:0",
        400,
    ),
    "mood": (
        "mtg_jamendo_moodtheme-discogs-effnet-1.onnx",
        "model/Placeholder:0",
        "model/Sigmoid:0",
        56,
    ),
    "instrument": (
        "mtg_jamendo_instrument-discogs-effnet-1.onnx",
        "model/Placeholder:0",
        "model/Sigmoid:0",
        40,
    ),
}


class GmiClassifierMIGraphX:
    """
    ONNX Runtime + MIGraphX wrapper for a single GMI classifier head.

    Interface mirrors TensorflowPredict2D:
        model = GmiClassifierMIGraphX(model_path, input_name, output_name, n_classes)
        preds = model(embeddings)   # embeddings: (n_patches, 1280) → (n_patches, n_classes)
    """

    def __init__(
        self,
        model_path: str | Path,
        input_name: str,
        output_name: str,
        n_classes: int,
        model_key: str,
    ):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available = ort.get_available_providers()
        if "MIGraphXExecutionProvider" in available:
            provider_options = {"device_id": "0"}
            providers = [("MIGraphXExecutionProvider", provider_options), "CPUExecutionProvider"]
            logger.info(f"GmiClassifier[{model_key}]: MIGraphX EP will JIT compile on first inference (~28s)")
        else:
            logger.warning(f"GmiClassifier[{model_key}]: MIGraphX EP not available — falling back to CPU")
            providers = ["CPUExecutionProvider"]

        self.session      = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
        self.input_name   = input_name
        self.output_name  = output_name
        self.n_classes    = n_classes
        self.model_key    = model_key
        self._warmed_up   = False
        logger.info(f"GmiClassifier[{model_key}]: active EPs = {self.session.get_providers()}")

    def _infer_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Run one fixed-size batch of _BATCH_SIZE embeddings.

        Args:
            batch: (_BATCH_SIZE, 1280) float32
        Returns:
            (_BATCH_SIZE, n_classes) float32
        """
        t0  = time.perf_counter()
        out = self.session.run([self.output_name], {self.input_name: batch})[0]
        elapsed = time.perf_counter() - t0

        if not self._warmed_up:
            logger.info(f"GmiClassifier[{self.model_key}]: first inference (JIT compile) took {elapsed:.1f}s")
            self._warmed_up = True
        else:
            logger.debug(f"GmiClassifier[{self.model_key}]: {elapsed * 1000:.1f}ms for batch_size={len(batch)}")

        return out  # (_BATCH_SIZE, n_classes)

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Args:
            embeddings: float32 (n_patches, 1280) — EffNet output
        Returns:
            predictions: float32 (n_patches, n_classes)
                         same shape/semantics as TensorflowPredict2D output
        """
        if len(embeddings) == 0:
            return np.empty((0, self.n_classes), dtype=np.float32)

        embeddings = embeddings.astype(np.float32)
        results = []

        for i in range(0, len(embeddings), _BATCH_SIZE):
            chunk  = embeddings[i : i + _BATCH_SIZE]
            n_real = len(chunk)

            if n_real < _BATCH_SIZE:
                # Zero-pad to fixed batch size
                pad   = np.zeros((_BATCH_SIZE - n_real, _EMBED_DIM), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)

            preds = self._infer_batch(chunk)
            results.append(preds[:n_real])  # strip padding

        return np.concatenate(results, axis=0)  # (n_patches, n_classes)


# ---------------------------------------------------------------------------
# Module-level singletons — one ONNX session per classifier, shared per run
# ---------------------------------------------------------------------------

_instances: Dict[str, GmiClassifierMIGraphX] = {}


def get_gmi_model(key: str, models_dir: str | Path) -> GmiClassifierMIGraphX:
    """
    Get or create the shared GmiClassifierMIGraphX instance for the given key.

    Args:
        key: one of 'genre', 'mood', 'instrument'
        models_dir: directory containing the .onnx files
    """
    global _instances
    if key not in _instances:
        if key not in _MODEL_SPECS:
            raise ValueError(f"Unknown GMI model key '{key}'; valid: {list(_MODEL_SPECS)}")
        filename, input_name, output_name, n_classes = _MODEL_SPECS[key]
        model_path = Path(models_dir) / filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"GMI ONNX model not found: {model_path}\n"
                f"Run: python src/classification/gmi_onnx.py --convert"
            )
        _instances[key] = GmiClassifierMIGraphX(model_path, input_name, output_name, n_classes, key)
    return _instances[key]


def preload_all_gmi(
    models_dir: str | Path,
    include_genre: bool = True,
    include_mood: bool = True,
    include_instrument: bool = True,
) -> int:
    """
    Pre-load and JIT-compile all requested GMI ONNX models.

    Returns number of models successfully loaded.
    """
    keys = []
    if include_genre:
        keys.append("genre")
    if include_mood:
        keys.append("mood")
    if include_instrument:
        keys.append("instrument")

    loaded = 0
    for key in keys:
        try:
            get_gmi_model(key, models_dir)
            loaded += 1
        except Exception as e:
            logger.warning(f"GMI ONNX preload failed for '{key}': {e}")
    return loaded


def unload_gmi_models() -> None:
    """Release all GMI ONNX sessions (call after Essentia pass to free GPU memory)."""
    global _instances
    count = len(_instances)
    _instances.clear()
    if count:
        logger.info(f"GmiClassifierMIGraphX: {count} sessions unloaded")


# ---------------------------------------------------------------------------
# Conversion helper + numerical verification (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="GMI classifier ONNX tools")
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

        CONVERT_MODELS = [
            ("genre_discogs400-discogs-effnet-1.pb",
             "genre_discogs400-discogs-effnet-1.onnx",
             ["serving_default_model_Placeholder:0"],
             ["PartitionedCall:0"]),
            ("mtg_jamendo_moodtheme-discogs-effnet-1.pb",
             "mtg_jamendo_moodtheme-discogs-effnet-1.onnx",
             ["model/Placeholder:0"],
             ["model/Sigmoid:0"]),
            ("mtg_jamendo_instrument-discogs-effnet-1.pb",
             "mtg_jamendo_instrument-discogs-effnet-1.onnx",
             ["model/Placeholder:0"],
             ["model/Sigmoid:0"]),
        ]
        base = Path(args.models_dir)
        for pb_name, onnx_name, inputs, outputs in CONVERT_MODELS:
            pb_path   = base / pb_name
            onnx_path = base / onnx_name
            print(f"Converting {pb_name} ...")
            with tf.io.gfile.GFile(str(pb_path), "rb") as f:
                gd = tf.compat.v1.GraphDef()
                gd.ParseFromString(f.read())
            tf2onnx.convert.from_graph_def(
                gd,
                input_names=inputs,
                output_names=outputs,
                output_path=str(onnx_path),
                opset=13,
            )
            size_mb = onnx_path.stat().st_size / 1024**2
            print(f"  -> {onnx_path} ({size_mb:.1f} MB)")

    if args.verify:
        import essentia.standard as es
        from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs
        from classification.essentia_features import get_model_path
        from classification.effnet_onnx import get_effnet_migraphx

        models_dir = Path(args.models_dir)
        audio_path = args.verify

        # Get embeddings
        loader = es.MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)
        audio  = loader()
        print(f"Audio: {audio_path}  ({len(audio)/16000:.2f}s)")

        onnx_path = get_model_path("discogs-effnet-bsdynamic-1.onnx")
        emb_model = get_effnet_migraphx(onnx_path)
        embeddings = emb_model(audio)
        print(f"Embeddings: {embeddings.shape}")

        for key, pb_name, tf_input, tf_output in [
            ("genre",      "genre_discogs400-discogs-effnet-1.pb",
             "serving_default_model_Placeholder", "PartitionedCall:0"),
            ("mood",       "mtg_jamendo_moodtheme-discogs-effnet-1.pb",      None, None),
            ("instrument", "mtg_jamendo_instrument-discogs-effnet-1.pb",     None, None),
        ]:
            kw = dict(graphFilename=str(models_dir / pb_name))
            if tf_input:
                kw["input"]  = tf_input
            if tf_output:
                kw["output"] = tf_output
            tf_model  = TensorflowPredict2D(**kw)
            tf_preds  = np.asarray(tf_model(embeddings))
            tf_mean   = np.mean(tf_preds, axis=0)

            onnx_model = get_gmi_model(key, models_dir)
            onnx_preds = onnx_model(embeddings)
            onnx_mean  = np.mean(onnx_preds, axis=0)

            diff = np.max(np.abs(tf_mean - onnx_mean))
            print(f"\n{key}:")
            print(f"  TF   top-5: {sorted(enumerate(tf_mean), key=lambda x: -x[1])[:5]}")
            print(f"  ONNX top-5: {sorted(enumerate(onnx_mean), key=lambda x: -x[1])[:5]}")
            print(f"  Max diff:   {diff:.2e}  {'OK' if diff < 1e-2 else 'FAIL'}")
