# tests/explorer_sa3/test_player_smoke.py
import importlib.util
import wave, io
from pathlib import Path
import numpy as np
import pytest

SERVER = Path(__file__).resolve().parents[2] / "scripts" / "latent_server_sa3.py"


def _load_module():
    # Import the script without running main(); stable_audio_3/torch may be absent
    spec = importlib.util.spec_from_file_location("latent_server_sa3", SERVER)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # missing torch/stable_audio_3 in mir venv
        pytest.skip(f"player deps unavailable: {e}")
    return mod


def test_wav_bytes_roundtrip():
    mod = _load_module()
    audio = np.zeros((2, 100), dtype=np.float32)
    audio[0, 10] = 0.5
    raw = mod.wav_bytes(audio, 44100)
    with wave.open(io.BytesIO(raw)) as w:
        assert w.getnchannels() == 2
        assert w.getframerate() == 44100
        assert w.getnframes() == 100
