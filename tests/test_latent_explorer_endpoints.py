"""Offline unit tests for new latent explorer server endpoint logic.

These test the pure functions, not the live HTTP server.
Import directly from the server module after monkey-patching globals.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# --- helpers used by the test suite ---

def make_sidecar(tmp: Path, track: str, crop: str,
                  start_time: float, end_time: float):
    """Write a minimal crop JSON sidecar."""
    track_dir = tmp / track
    track_dir.mkdir(parents=True, exist_ok=True)
    sidecar = track_dir / f"{crop}.json"
    sidecar.write_text(json.dumps({
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "position": start_time / 120.0,
    }))
    return sidecar

def make_sidecar_files(source_dir: Path, track: str,
                        beats, downbeats, onsets):
    """Write .BEATS_GRID, .DOWNBEATS, .ONSETS files for a track."""
    d = source_dir / track
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{track}.BEATS_GRID").write_text("\n".join(str(t) for t in beats))
    (d / f"{track}.DOWNBEATS").write_text("\n".join(str(t) for t in downbeats))
    (d / f"{track}.ONSETS").write_text("\n".join(str(t) for t in onsets))


# --- import the function under test ---

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_timecodes_filters_to_crop_window():
    """_read_timecodes returns only timestamps within [start_time, end_time]
    and offsets them relative to start_time."""
    import latent_shape_server as srv

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        latent_dir = tmp / "latents"
        source_dir = tmp / "source"

        beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        downbeats = [0.0, 2.0]
        onsets = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        make_sidecar_files(source_dir, "TrackA", beats, downbeats, onsets)
        make_sidecar(latent_dir, "TrackA", "crop01", start_time=1.0, end_time=3.0)

        # Monkey-patch globals
        srv._latent_dir = latent_dir
        srv._source_dir = source_dir

        result = srv._read_timecodes("TrackA", "crop01")

        # beats [1.0..3.0] → [0.0, 0.5, 1.0, 1.5, 2.0] (5 beats, offset by -1.0)
        assert result["beats"] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0], abs=1e-4)
        # downbeats [1.0 is on boundary, 2.0 is in window] → [0.0, 1.0]
        assert 0.0 in result["downbeats"]
        assert result["duration"] == pytest.approx(2.0, abs=1e-4)
        # bpm: 5 beats over 2s → intervals ~0.5s → bpm ~120
        assert result["bpm"] == pytest.approx(120.0, abs=2.0)


def test_timecodes_missing_sidecar_returns_empty_array():
    """Missing .ONSETS → onsets=[], listed in missing[]."""
    import latent_shape_server as srv

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        latent_dir = tmp / "latents"
        source_dir = tmp / "source"

        # Only write beats and downbeats — omit onsets
        track_dir = source_dir / "TrackB"
        track_dir.mkdir(parents=True, exist_ok=True)
        (track_dir / "TrackB.BEATS_GRID").write_text("1.0\n1.5\n2.0\n2.5\n")
        (track_dir / "TrackB.DOWNBEATS").write_text("1.0\n3.0\n")
        # No TrackB.ONSETS file

        make_sidecar(latent_dir, "TrackB", "crop02", 1.0, 3.0)
        srv._latent_dir = latent_dir
        srv._source_dir = source_dir

        result = srv._read_timecodes("TrackB", "crop02")
        assert result["onsets"] == []
        assert "onsets" in result.get("missing", [])


def test_timecodes_no_source_dir_returns_none():
    """_read_timecodes returns None when _source_dir is None."""
    import latent_shape_server as srv
    srv._source_dir = None
    assert srv._read_timecodes("any", "any") is None


def test_timecodes_missing_crop_json_returns_none():
    """_read_timecodes returns None when crop JSON sidecar is absent."""
    import latent_shape_server as srv
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        latent_dir = tmp / "latents"
        source_dir = tmp / "source"
        make_sidecar_files(source_dir, "TrackC", [1.0, 1.5], [1.0], [1.0, 1.25])
        (latent_dir / "TrackC").mkdir(parents=True, exist_ok=True)
        # No crop JSON written

        srv._latent_dir = latent_dir
        srv._source_dir = source_dir
        assert srv._read_timecodes("TrackC", "missing_crop") is None
