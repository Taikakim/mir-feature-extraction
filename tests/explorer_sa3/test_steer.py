# tests/explorer_sa3/test_steer.py
import importlib.util
from pathlib import Path
import pytest

SERVER = Path(__file__).resolve().parents[2] / "scripts" / "latent_server_sa3.py"


def _load():
    spec = importlib.util.spec_from_file_location("latent_server_sa3", SERVER)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"player deps unavailable: {e}")
    return mod


def test_available_heads_parses_filenames(tmp_path):
    mod = _load()
    (tmp_path / "latch_sa3_hpcp_best.pt").write_bytes(b"")
    (tmp_path / "latch_sa3_rms_energy_bass_best.pt").write_bytes(b"")
    (tmp_path / "notahead.txt").write_bytes(b"")
    mod._cfg = {"latch_weights_dir": str(tmp_path)}
    assert mod._available_heads() == ["hpcp", "rms_energy_bass"]
