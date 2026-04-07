"""Tests for plots/explorer/audio.py URL builders."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.explorer.audio import build_decode_url, build_crossfade_url, build_average_url

def test_decode_url_basic():
    url = build_decode_url("Artist - Track", "0.5")
    assert "track=Artist" in url
    assert "position=0.5" in url
    assert "localhost:7891" in url

def test_decode_url_smart_loop():
    url = build_decode_url("T", "0.5", smart_loop=True)
    assert "smart_loop=1" in url

def test_crossfade_url():
    url = build_crossfade_url("Track A", "0.3", "Track B", "0.7", mix=0.5)
    assert "track_a=Track" in url
    assert "track_b=Track" in url
    assert "mix=0.500" in url

def test_average_url_single():
    url = build_average_url("Track A")
    assert "track=Track" in url
    assert "average" in url
