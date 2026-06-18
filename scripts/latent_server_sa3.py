#!/home/kim/Projects/SAO/stable-audio-3/.venv/bin/python
"""latent_server_sa3.py — decode SAME-L (256-dim, T=4096) latents to WAV.

Runs under the SA3 venv (py3.13). Endpoints: /status /crops /meta /decode
/source. /mix and /steer are added in later tasks.
"""
from __future__ import annotations
import argparse, configparser, io, json, sys, threading, wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

DEFAULT_INI = Path(__file__).parent.parent / "latent_player_sa3.ini"
sys.path.insert(0, str(Path(__file__).parent))

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


def _interp_np(a: np.ndarray, b: np.ndarray, t: float, interp: str) -> np.ndarray:
    """Interpolate two [256, T] latents (T aligned to min). lerp or slerp."""
    import torch
    from latent_crossfader import slerp, lerp
    T = min(a.shape[1], b.shape[1])
    za = torch.from_numpy(a[:, :T]).unsqueeze(0).float()
    zb = torch.from_numpy(b[:, :T]).unsqueeze(0).float()
    fn = slerp if interp == "slerp" else lerp
    return fn(za, zb, float(t)).squeeze(0).numpy()


def _mix(crop_a: str, crop_b: str, t: float, interp: str) -> bytes:
    import torch
    a = np.load(_latent_dir / f"{crop_a}.npy").astype(np.float32)
    b = np.load(_latent_dir / f"{crop_b}.npy").astype(np.float32)
    if a.ndim == 3: a = a[0]
    if b.ndim == 3: b = b[0]
    z = _interp_np(a, b, t, interp)
    zt = torch.from_numpy(z).unsqueeze(0).to(_ae_device())
    with torch.no_grad():
        audio = _ae.decode(zt, chunked=True,
                           chunk_size=int(_cfg["chunk_size"]),
                           overlap=int(_cfg["overlap"]))
    return wav_bytes(audio.squeeze(0).cpu().float().numpy(), _sr)


_heads: dict = {}


def _available_heads() -> list[str]:
    d = Path(_cfg["latch_weights_dir"])
    feats = []
    for p in sorted(d.glob("latch_sa3_*_best.pt")):
        feats.append(p.stem[len("latch_sa3_"):-len("_best")])
    return feats


def _load_head(feature: str):
    if feature in _heads:
        return _heads[feature]
    import torch
    from stable_audio_3.models.latch import LatCH
    ckpt_path = Path(_cfg["latch_weights_dir"]) / f"latch_sa3_{feature}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(feature)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = LatCH(in_channels=256, out_channels=ckpt["out_channels"],
                 dim=256, depth=6, num_heads=8)
    head.load_state_dict(ckpt["state_dict"])
    head.eval().requires_grad_(False).to(_ae_device())
    _heads[feature] = head
    return head


def _steer(crop_id: str, feature: str, gain: float) -> bytes:
    import torch
    arr = np.load(_latent_dir / f"{crop_id}.npy").astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    head = _load_head(feature)
    z = torch.from_numpy(arr).unsqueeze(0).float().to(_ae_device())
    z.requires_grad_(True)
    t = torch.tensor([0.001], dtype=torch.float32, device=z.device)
    pred = head(z, t)              # [1, out_channels, T]
    pred.mean().backward()
    grad = z.grad
    z_edit = (z.detach() + gain * grad).to(_ae_device())
    with torch.no_grad():
        audio = _ae.decode(z_edit, chunked=True,
                           chunk_size=int(_cfg["chunk_size"]),
                           overlap=int(_cfg["overlap"]))
    return wav_bytes(audio.squeeze(0).cpu().float().numpy(), _sr)


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
                                   "latent_dir": str(_latent_dir),
                                   "heads": _available_heads()})
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
            if u.path == "/mix":
                with _lock:
                    return self._wav(_mix(q["crop_a"], q["crop_b"],
                                          float(q.get("t", 0.5)),
                                          q.get("interp", "slerp")))
            if u.path == "/steer":
                with _lock:
                    return self._wav(_steer(q["crop"], q["head"],
                                            float(q.get("gain", 48.0))))
            return self._json({"error": "unknown endpoint"}, 404)
        except FileNotFoundError:
            return self._json({"error": "unknown crop or head",
                               "heads": _available_heads()}, 404)
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
