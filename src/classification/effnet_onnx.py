"""
MIGraphX-accelerated Discogs-EffNet embedding extractor.

Replaces TensorflowPredictEffnetDiscogs with ONNX Runtime + MIGraphX EP.
Provides ~250x speedup for the embedding step (4ms vs ~1s per crop on GPU).

Preprocessing pipeline (matches TensorflowPredictEffnetDiscogs defaults):
    16kHz mono audio
    → FrameGenerator(frameSize=512, hopSize=256)
    → TensorflowInputMusiCNN per frame  → 96 mel bands
    → Patches: 128 frames, hop=62       → (n_patches, 128, 96)
    → ONNX + MIGraphX EP                → (n_patches, 1280) embeddings

JIT compile: ~28s on first inference call (MIGraphX compiles gfx1201 kernels).
Subsequent calls with same batch size: ~4ms.
Batch size is fixed at _BATCH_SIZE=16; shorter batches are zero-padded.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Must match TensorflowPredictEffnetDiscogs defaults
_PATCH_SIZE = 128   # time frames per patch
_PATCH_HOP  = 62    # frames between patch starts (~1 Hz prediction rate)
_BATCH_SIZE = 16    # fixed MIGraphX batch size (pad or chunk to this)

_CACHE_DIR     = Path.home() / ".cache" / "mir"
_COMPILED_PATH = _CACHE_DIR / f"effnet_migraphx_bs{_BATCH_SIZE}.mxr"


class EffNetMIGraphX:
    """
    ONNX Runtime + MIGraphX wrapper for Discogs-EffNet embeddings.

    Interface mirrors TensorflowPredictEffnetDiscogs:
        model = EffNetMIGraphX(model_path)
        embeddings = model(audio)   # audio: 16kHz mono float32 → (n_patches, 1280)
    """

    def __init__(self, model_path: str | Path):
        import onnxruntime as ort
        import essentia.standard as es

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available = ort.get_available_providers()
        if "MIGraphXExecutionProvider" in available:
            provider_options = {"device_id": "0"}
            # Cache options are deprecated since ROCm 6.4 but may still work in 1.23.2
            try:
                if _COMPILED_PATH.exists():
                    provider_options["migraphx_load_compiled_model"] = str(_COMPILED_PATH)
                    logger.info(f"EffNetMIGraphX: loading cached compiled model ({_COMPILED_PATH.name})")
                else:
                    provider_options["migraphx_save_compiled_model"] = str(_COMPILED_PATH)
                    logger.info(
                        f"EffNetMIGraphX: first run — MIGraphX will JIT compile (~28s), "
                        f"caching to {_COMPILED_PATH.name}"
                    )
            except Exception:
                pass  # Cache options not supported, proceed without

            providers = [("MIGraphXExecutionProvider", provider_options), "CPUExecutionProvider"]
        else:
            logger.warning("EffNetMIGraphX: MIGraphX EP not available — falling back to CPU")
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
        logger.info(f"EffNetMIGraphX: active EPs = {self.session.get_providers()}")

        # Stateless Essentia algorithms — safe to share
        self._mel_fn   = es.TensorflowInputMusiCNN()
        self._frame_fn = es.FrameGenerator
        self._warmed_up = False

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _compute_patches(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert 16kHz mono audio to mel patches.

        Returns:
            (n_patches, 128, 96) float32 — ready for ONNX input
        """
        frame_gen = self._frame_fn(audio, frameSize=512, hopSize=256, startFromZero=True)
        mel_frames = np.array([self._mel_fn(frame) for frame in frame_gen])  # (n_frames, 96)

        if len(mel_frames) < _PATCH_SIZE:
            return np.empty((0, _PATCH_SIZE, 96), dtype=np.float32)

        patches = np.stack([
            mel_frames[i : i + _PATCH_SIZE]
            for i in range(0, len(mel_frames) - _PATCH_SIZE + 1, _PATCH_HOP)
        ])  # (n_patches, 128, 96)

        return patches.astype(np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Run one fixed-size batch of _BATCH_SIZE patches.

        Args:
            batch: (BATCH_SIZE, 128, 96) float32
        Returns:
            (BATCH_SIZE, 1280) float32
        """
        t0 = time.perf_counter()
        out = self.session.run(["embeddings"], {"melspectrogram": batch})[0]
        elapsed = time.perf_counter() - t0

        if not self._warmed_up:
            logger.info(f"EffNetMIGraphX: first inference (JIT compile) took {elapsed:.1f}s")
            self._warmed_up = True
        else:
            logger.debug(f"EffNetMIGraphX: {elapsed * 1000:.1f}ms for batch_size={len(batch)}")

        return out  # (BATCH_SIZE, 1280)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Args:
            audio: 16kHz mono float32, shape (n_samples,)
        Returns:
            embeddings: float32 (n_patches, 1280)
                        same shape/semantics as TensorflowPredictEffnetDiscogs output
        """
        patches = self._compute_patches(audio)
        if len(patches) == 0:
            return np.array([], dtype=np.float32)

        results = []
        for i in range(0, len(patches), _BATCH_SIZE):
            chunk = patches[i : i + _BATCH_SIZE]
            n_real = len(chunk)

            if n_real < _BATCH_SIZE:
                # Zero-pad to fixed batch size
                pad = np.zeros((_BATCH_SIZE - n_real, _PATCH_SIZE, 96), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)

            embeddings = self._infer_batch(chunk)
            results.append(embeddings[:n_real])  # strip padding

        return np.concatenate(results, axis=0)  # (n_patches, 1280)


# ---------------------------------------------------------------------------
# Module-level singleton — one ONNX session shared across all crops in a run
# ---------------------------------------------------------------------------

_effnet_instance: Optional[EffNetMIGraphX] = None


def get_effnet_migraphx(model_path: str | Path) -> EffNetMIGraphX:
    """Get or create the shared EffNetMIGraphX instance."""
    global _effnet_instance
    if _effnet_instance is None:
        _effnet_instance = EffNetMIGraphX(model_path)
    return _effnet_instance


def unload_effnet_migraphx() -> None:
    """Release the ONNX session (call after Essentia pass to free GPU memory)."""
    global _effnet_instance
    if _effnet_instance is not None:
        _effnet_instance = None
        logger.info("EffNetMIGraphX: session unloaded")
