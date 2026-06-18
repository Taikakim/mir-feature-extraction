#!/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python
"""latent_server_sa3.py — decode SAME-L (256-dim, T=4096) latents to WAV.

Runs under the SA3 venv (py3.13). Endpoints: /status /crops /meta /decode
/source. /mix and /steer are added in later tasks.
"""
from __future__ import annotations
import argparse, configparser, io, json, threading, wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

DEFAULT_INI = Path(__file__).parent.parent / "latent_player_sa3.ini"

# Globals set in main(); read-only afterwards.
_ae = None
_sr = 44100
_latent_dir: Path | None = None
_cfg: dict = {}
_lock = threading.Lock()


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """audio [2, samples] float32 → stereo int16 WAV bytes."""
    a = np.clip(audio, -1.0, 1.0)
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


def _decode_latent(crop_id: str) -> bytes:
    import torch
    arr = np.load(_latent_dir / f"{crop_id}.npy").astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    meta = _read_meta(crop_id)
    z = torch.from_numpy(arr).unsqueeze(0).to(_ae_device())
    with torch.no_grad():
        audio = _ae.decode(z, chunked=True,
                           chunk_size=int(_cfg["chunk_size"]),
                           overlap=int(_cfg["overlap"]))
    audio_np = audio.squeeze(0).cpu().float().numpy()   # [2, samples]
    n_content = int(sum(meta.get("padding_mask") or [])) or arr.shape[1]
    samples = n_content * 4096
    if 0 < samples < audio_np.shape[1]:
        audio_np = audio_np[:, :samples]
    return wav_bytes(audio_np, _sr)


def _source_slice(crop_id: str) -> bytes:
    import soundfile as sf
    meta = _read_meta(crop_id)
    src = meta["source_path"]
    s, e = int(meta["start_sample"]), int(meta["end_sample"])
    audio, sr = sf.read(src, dtype="float32", always_2d=True, start=s, stop=e)
    audio = audio.T
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return wav_bytes(audio, sr)


def _ae_device():
    import torch
    return torch.device(_cfg.get("device", "cuda")
                        if __import__("torch").cuda.is_available() else "cpu")


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
                return self._json({"ok": True, "model": _cfg.get("model"),
                                   "sample_rate": _sr,
                                   "latent_dir": str(_latent_dir)})
            if u.path == "/crops":
                ids = sorted(p.stem for p in _latent_dir.glob("*.npy"))
                return self._json(ids)
            if u.path == "/meta":
                return self._json(_read_meta(q["crop"]))
            if u.path == "/decode":
                with _lock:
                    return self._wav(_decode_latent(q["crop"]))
            if u.path == "/source":
                return self._wav(_source_slice(q["crop"]))
            return self._json({"error": "unknown endpoint"}, 404)
        except FileNotFoundError:
            return self._json({"error": "crop not found"}, 404)
        except Exception as e:
            return self._json({"error": str(e)}, 500)

    def log_message(self, *a):
        pass


def _load_model(cfg: dict):
    from stable_audio_3 import AutoencoderModel
    ae = AutoencoderModel.from_pretrained(cfg["model"])
    return ae, int(ae.sample_rate)


def main():
    global _ae, _sr, _latent_dir, _cfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_INI)
    args = ap.parse_args()
    parser = configparser.ConfigParser()
    parser.read(args.config)
    _cfg = {**dict(parser["server"]), **dict(parser["model"])}
    _latent_dir = Path(_cfg["latent_dir"])
    print("Loading SAME-L autoencoder ...")
    _ae, _sr = _load_model(_cfg)
    port = int(_cfg["port"])
    print(f"Serving on http://localhost:{port}  (latents: {_latent_dir})")
    HTTPServer(("localhost", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
