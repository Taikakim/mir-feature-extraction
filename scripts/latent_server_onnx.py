#!/home/kim/Projects/mir/mir/bin/python
"""latent_server_onnx.py — low-VRAM SAME decode server backed by the ONNX
decoder on ONNX Runtime + MIGraphX (AMD GPU).

A lightweight sibling of `latent_server_sa3.py`: same HTTP shape so the explorer
viewer can point at either, but this one loads the exported ONNX decoder
(`export_same_onnx.py`) instead of the torch VAE — ~2 GB GPU instead of ~5 GB, no
torch, and it can run alongside a training job. It compiles the ONNX once at boot
(the MIGraphX AOT compile is a one-time startup cost here, ~min), then serves
decodes at RTF ~39×.

Runs in the **mir venv** (which has `onnxruntime_migraphx`); the SA3-venv player's
`onnxruntime` is CPU-only. Endpoints: /status /crops /meta /decode /mix /source.
`/steer` (LatCH gradient guidance) needs torch — it stays on `latent_server_sa3.py`.

Usage
-----
    /home/kim/Projects/mir/mir/bin/python scripts/latent_server_onnx.py \\
        --onnx /home/kim/Projects/SAO/stable-audio-3/same_decoder_L128.onnx \\
        --chunk-latents 128 --overlap 16 --provider migraphx --port 7893
"""
import argparse
import io
import json
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

# Reuse the *validated* chunk-loop + provider selection from the SA3 export tooling
# (single-sourced so the stitch math can't drift from what we verified).
_DECODE_ONNX_DIR = Path("/home/kim/Projects/SAO/stable-audio-3/scripts")
sys.path.insert(0, str(_DECODE_ONNX_DIR))
from decode_onnx import decode_chunked_onnx, pick_providers  # noqa: E402

SR = 44100
DS = 4096          # SAME-L downsampling ratio
LATENT_DIM = 256

# Globals set in main(), read-only afterwards.
_sess = None
_latent_dir: Path | None = None
_chunk = 128
_overlap = 16
_active_ep = "?"
_lock = threading.Lock()


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """audio [1,2,N] or [2,N] float32 → stereo int16 WAV bytes."""
    a = audio[0] if audio.ndim == 3 else audio
    a = np.clip(a, -1.0, 1.0)
    i16 = (a * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(i16.T.flatten().tobytes())
    return buf.getvalue()


def _read_meta(crop_id: str) -> dict:
    return json.loads((_latent_dir / f"{crop_id}.json").read_text())


def _load_latent(crop_id: str) -> np.ndarray:
    arr = np.load(_latent_dir / f"{crop_id}.npy").astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None]            # [C,T] -> [1,C,T]
    if arr.ndim != 3 or arr.shape[1] != LATENT_DIM:
        raise ValueError(f"{crop_id}: expected [1,{LATENT_DIM},T], got {arr.shape}")
    return arr


def _content_frames(meta: dict) -> int:
    mask = meta.get("padding_mask")
    return int(sum(mask)) if mask is not None else None


def _interp_np(a: np.ndarray, b: np.ndarray, t: float, interp: str) -> np.ndarray:
    """Energy-preserving slerp (or lerp) of two [C,T] latents, T aligned to min."""
    T = min(a.shape[1], b.shape[1])
    a, b = a[:, :T].astype(np.float64), b[:, :T].astype(np.float64)
    if interp != "slerp":
        return ((1.0 - t) * a + t * b).astype(np.float32)
    fa, fb = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(fa) + 1e-8, np.linalg.norm(fb) + 1e-8
    omega = np.arccos(np.clip(np.dot(fa / na, fb / nb), -1.0, 1.0))
    s = np.sin(omega)
    if s < 1e-6:
        r = (1.0 - t) * fa + t * fb
    else:
        r = (np.sin((1.0 - t) * omega) / s) * fa + (np.sin(t * omega) / s) * fb
    return r.reshape(a.shape).astype(np.float32)


def _decode_crop(crop_id: str) -> bytes:
    lat = _load_latent(crop_id)                                  # [1,C,T]
    audio = decode_chunked_onnx(_sess, lat, _chunk, _overlap, DS)
    n = _content_frames(_read_meta(crop_id))
    if n:
        samples = n * DS
        if 0 < samples < audio.shape[-1]:
            audio = audio[..., :samples]
    return wav_bytes(audio, SR)


def _mix_crops(crop_a: str, crop_b: str, t: float, interp: str) -> bytes:
    a = _load_latent(crop_a)[0]                                  # [C,T]
    b = _load_latent(crop_b)[0]
    z = _interp_np(a, b, t, interp)[None]                        # [1,C,T]
    audio = decode_chunked_onnx(_sess, z, _chunk, _overlap, DS)
    return wav_bytes(audio, SR)


def _source_slice(crop_id: str) -> bytes:
    import soundfile as sf
    meta = _read_meta(crop_id)
    s, e = int(meta["start_sample"]), int(meta["end_sample"])
    audio, sr = sf.read(meta["source_path"], dtype="float32", always_2d=True,
                        start=s, stop=e)
    audio = audio.T
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return wav_bytes(audio, sr)


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _wav(self, raw: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/status":
                return self._json({"ok": True, "backend": "onnx-migraphx",
                                   "active_ep": _active_ep, "sample_rate": SR,
                                   "latent_dir": str(_latent_dir),
                                   "chunk_latents": _chunk, "overlap": _overlap})
            if u.path == "/crops":
                return self._json(sorted(p.stem for p in _latent_dir.glob("*.npy")))
            if u.path == "/meta":
                return self._json(_read_meta(q["crop"]))
            if u.path == "/decode":
                with _lock:
                    return self._wav(_decode_crop(q["crop"]))
            if u.path == "/mix":
                with _lock:
                    return self._wav(_mix_crops(q["crop_a"], q["crop_b"],
                                                float(q.get("t", 0.5)),
                                                q.get("interp", "slerp")))
            if u.path == "/source":
                return self._wav(_source_slice(q["crop"]))
            if u.path == "/steer":
                return self._json({"error": "steer needs torch — use latent_server_sa3.py "
                                            "(this is the low-VRAM ONNX decode server)"}, 501)
            return self._json({"error": "unknown endpoint"}, 404)
        except FileNotFoundError:
            return self._json({"error": "crop not found"}, 404)
        except Exception as e:
            return self._json({"error": str(e)}, 500)

    def log_message(self, *a):
        pass


def main():
    global _sess, _latent_dir, _chunk, _overlap, _active_ep
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, type=Path, help="exported SAME decoder .onnx")
    ap.add_argument("--latent-dir", default="/home/kim/Projects/latents_sa3")
    ap.add_argument("--chunk-latents", type=int, default=128, help="MUST match the export L")
    ap.add_argument("--overlap", type=int, default=16, help=">= decoder receptive field")
    ap.add_argument("--provider", default="migraphx", help="migraphx | rocm | cpu")
    ap.add_argument("--port", type=int, default=7893)
    args = ap.parse_args()

    _latent_dir = Path(args.latent_dir)
    _chunk, _overlap = args.chunk_latents, args.overlap

    import onnxruntime as ort
    providers = pick_providers(args.provider)
    print(f"Loading ONNX decoder {args.onnx.name} on {providers} ...")
    print("(MIGraphX AOT-compiles the graph once here — ~min; subsequent decodes are fast)")
    t0 = time.time()
    so = ort.SessionOptions()
    _sess = ort.InferenceSession(str(args.onnx), sess_options=so, providers=providers)
    _active_ep = _sess.get_providers()[0]
    if "migraphx" in args.provider.lower() and "CPU" in _active_ep:
        print("⚠ WARNING: requested MIGraphX but active EP is CPU — not a GPU run "
              "(check onnxruntime_migraphx is installed in this venv).")
    print(f"Ready in {time.time() - t0:.0f}s — active EP {_active_ep}")
    print(f"Serving on http://localhost:{args.port}  (latents: {_latent_dir})")
    HTTPServer(("localhost", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
