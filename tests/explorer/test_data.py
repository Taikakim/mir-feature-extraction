"""Tests for plots/explorer/data.py pure functions."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plots.explorer.data import load_config, load_tracks, AppData

def test_load_config_reads_ini(tmp_path):
    ini = tmp_path / "latent_player.ini"
    ini.write_text(
        "[server]\nlatent_dir = /tmp/lat\nstem_dir = /tmp/stems\n"
        "raw_audio_dir = /tmp/raw\nsource_dir = /tmp/src\nport = 7891\n"
        "[model]\nsao_dir = /tmp/sao\nmodel_config = /tmp/cfg.json\n"
        "ckpt_path = /tmp/ckpt\nmodel_half = true\ndevice = cuda\n"
    )
    cfg = load_config(ini)
    assert cfg["latent_dir"] == Path("/tmp/lat")
    assert cfg["port"] == 7891

def test_load_tracks_returns_appdata(tmp_path):
    csv = tmp_path / "tracks.csv"
    csv.write_text(
        "track,bpm,brightness,artists,essentia_genre\n"
        "Artist A - Song 1,140.0,70.5,\"['Artist A']\",\"{'Goa Trance': 0.9}\"\n"
        "Artist B - Song 2,128.0,60.0,\"['Artist B']\",\"{}\"\n"
    )
    ad = load_tracks(csv)
    assert isinstance(ad, AppData)
    assert len(ad.tracks) == 2
    assert "bpm" in ad.num_cols
    assert "essentia_genre" in ad.class_cols
    assert ad.feat_array("bpm").shape == (2,)
    assert abs(ad.feat_array("bpm")[0] - 140.0) < 0.01

from plots.explorer.data import scan_latent_dir, project_latent_pca

def test_scan_latent_dir_returns_dict(tmp_path):
    (tmp_path / "Artist - Track 1").mkdir()
    (tmp_path / "Artist - Track 1" / "Artist - Track 1_0.npy").write_bytes(b"\x93NUMPY")
    (tmp_path / "Artist - Track 1" / "Artist - Track 1_1.npy").write_bytes(b"\x93NUMPY")
    result = scan_latent_dir(tmp_path)
    assert "Artist - Track 1" in result
    assert result["Artist - Track 1"] == ["Artist - Track 1_0", "Artist - Track 1_1"]

def test_project_latent_pca_shape():
    import numpy as np
    z    = np.random.randn(64, 256).astype(np.float32)
    pca  = np.random.randn(3, 64).astype(np.float32)
    pts  = project_latent_pca(z, pca)
    assert pts.shape == (256, 3)

def test_appdata_search_filters_by_name_and_artist(tmp_path):
    csv = tmp_path / "tracks.csv"
    csv.write_text(
        "track,bpm,artists\n"
        "Astral Projection - People Can Fly,148.0,\"['Astral Projection']\"\n"
        "Infected Mushroom - Bust a Move,145.0,\"['Infected Mushroom']\"\n"
        "X-Dream - We Interface,142.0,\"['X-Dream']\"\n"
    )
    ad = load_tracks(csv)
    # pattern in track name
    idxs = ad.search("people")
    assert len(idxs) == 1 and ad.tracks[idxs[0]] == "Astral Projection - People Can Fly"
    # pattern in artist
    idxs = ad.search("infected")
    assert len(idxs) == 1 and ad.tracks[idxs[0]] == "Infected Mushroom - Bust a Move"
    # empty query returns all
    assert len(ad.search("")) == 3
