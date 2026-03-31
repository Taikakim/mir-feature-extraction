import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from plots.build_dataset_stats import (
    _strip_crop_suffix,
    _interp32,
    _rotate_hpcp,
    _dominant_tonic,
    _cosine_top_k,
    load_track_data,
    run_scalar_pass,
    write_data_js,
)


def test_strip_crop_suffix_basic():
    assert _strip_crop_suffix("Artist - Title_0") == "Artist - Title"
    assert _strip_crop_suffix("Artist - Title_12") == "Artist - Title"
    assert _strip_crop_suffix("No suffix") == "No suffix"
    assert _strip_crop_suffix("Ends_in_word_0") == "Ends_in_word"


def test_strip_crop_suffix_only_trailing():
    # underscore in the middle of name is preserved
    assert _strip_crop_suffix("A_B_C_3") == "A_B_C"


def test_interp32_length():
    arr = np.random.rand(256).astype(np.float32)
    out = _interp32(arr)
    assert out.shape == (32,)
    assert out.dtype == np.float32


def test_interp32_passthrough():
    arr = np.arange(32, dtype=np.float32)
    out = _interp32(arr)
    np.testing.assert_allclose(out, arr, atol=1e-5)


def test_interp32_boundary_values():
    arr = np.zeros(128, dtype=np.float32)
    arr[0] = 1.0
    arr[-1] = 2.0
    out = _interp32(arr)
    assert abs(out[0] - 1.0) < 1e-5
    assert abs(out[-1] - 2.0) < 1e-5


def test_rotate_hpcp_by_zero():
    hpcp = np.arange(12, dtype=np.float32)
    out = _rotate_hpcp(hpcp, 0)
    np.testing.assert_array_equal(out, hpcp)


def test_rotate_hpcp_by_3():
    hpcp = np.arange(12, dtype=np.float32)
    out = _rotate_hpcp(hpcp, 3)
    # roll by -3: index 0 becomes what was at index 3
    assert out[0] == pytest.approx(3.0)
    assert out[1] == pytest.approx(4.0)
    assert out[11] == pytest.approx(2.0)


def test_rotate_hpcp_wraps():
    hpcp = np.zeros(12, dtype=np.float32)
    hpcp[11] = 1.0
    out = _rotate_hpcp(hpcp, 11)
    assert out[0] == pytest.approx(1.0)


def test_dominant_tonic_clear():
    # tonic 5 appears most often
    arr = np.array([5.0, 5.1, 4.9, 5.0, 5.2, 2.0], dtype=np.float32)
    assert _dominant_tonic(arr) == 5


def test_dominant_tonic_wraps_mod12():
    arr = np.full(10, 13.0, dtype=np.float32)  # 13 % 12 == 1
    assert _dominant_tonic(arr) == 1


def test_cosine_top_k_excludes_self():
    emb = np.array([[1., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
    names = ["A", "B", "C"]
    result = _cosine_top_k(emb, names, k=2)
    assert "A" not in [n for n, _ in result["A"]]


def test_cosine_top_k_most_similar_first():
    emb = np.array([[1., 0.], [0.99, 0.01], [0., 1.]], dtype=np.float32)
    names = ["A", "B", "C"]
    result = _cosine_top_k(emb, names, k=2)
    assert result["A"][0][0] == "B"
    scores = [s for _, s in result["A"]]
    assert scores == sorted(scores, reverse=True)


def test_cosine_top_k_respects_k():
    emb = np.random.rand(10, 4).astype(np.float32)
    names = [str(i) for i in range(10)]
    result = _cosine_top_k(emb, names, k=3)
    assert all(len(v) == 3 for v in result.values())


def _make_info(path: Path, data: dict):
    path.write_text(json.dumps(data))


def test_load_track_data_averages_crops(tmp_path):
    track_dir = tmp_path / "Artist - Title"
    track_dir.mkdir()
    _make_info(track_dir / "Artist - Title_0.INFO", {"bpm": 140.0, "lufs": -10.0})
    _make_info(track_dir / "Artist - Title_1.INFO", {"bpm": 142.0, "lufs": -12.0})
    result = load_track_data(track_dir)
    assert result is not None
    assert abs(result["bpm"] - 141.0) < 0.01
    assert abs(result["lufs"] - -11.0) < 0.01


def test_load_track_data_skips_stems(tmp_path):
    track_dir = tmp_path / "Track"
    track_dir.mkdir()
    _make_info(track_dir / "Track_0.INFO",       {"bpm": 140.0})
    _make_info(track_dir / "Track_0_bass.INFO",  {"bpm": 999.0})  # stem — skip
    result = load_track_data(track_dir)
    assert abs(result["bpm"] - 140.0) < 0.01


def test_run_scalar_pass_produces_js(tmp_path):
    # Two track dirs each with one crop
    for name, bpm in [("Artist - A", 130.0), ("Artist - B", 145.0)]:
        d = tmp_path / name
        d.mkdir()
        _make_info(d / f"{name}_0.INFO", {"bpm": bpm, "lufs": -14.0})

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sorted_tracks, tracks_data = run_scalar_pass(tmp_path, out_dir)

    js = (out_dir / "feature_explorer_data.js").read_text()
    assert "Artist - A" in js
    assert "Artist - B" in js
    assert sorted_tracks == ["Artist - A", "Artist - B"]
    assert abs(tracks_data["Artist - A"]["bpm"] - 130.0) < 0.01


from plots.build_dataset_stats import process_track_ts, run_timeseries_pass, _rotate_hpcp
from src.core.timeseries_db import TimeseriesDB


def _make_db(tmp_path, crop_key: str, n_steps: int = 64) -> TimeseriesDB:
    db = TimeseriesDB(tmp_path / "test.db")
    ts = {
        "rms_energy_bass_ts":  np.random.rand(n_steps).astype(np.float32).tolist(),
        "rms_energy_body_ts":  np.random.rand(n_steps).astype(np.float32).tolist(),
        "hpcp_ts":             np.random.rand(n_steps, 12).astype(np.float32).tolist(),
        "tonic_ts":            np.full(n_steps, 5.0, dtype=np.float32).tolist(),
        "tonic_strength_ts":   np.random.rand(n_steps).astype(np.float32).tolist(),
    }
    db.put(crop_key, ts)
    return db


def test_process_track_ts_curve_length(tmp_path):
    db = _make_db(tmp_path, "Artist - Title_0", n_steps=256)
    result = process_track_ts(["Artist - Title_0"], db)
    db.close()
    assert result is not None
    assert len(result["curves"]["rms_energy_bass_ts"]) == 32


def test_process_track_ts_curve_normalised(tmp_path):
    db = _make_db(tmp_path, "Artist - Title_0")
    result = process_track_ts(["Artist - Title_0"], db)
    db.close()
    for field, curve in result["curves"].items():
        assert max(curve) <= 1.0 + 1e-5, f"{field} not normalised"


def test_process_track_ts_shape_scalars(tmp_path):
    db = _make_db(tmp_path, "Artist - Title_0")
    result = process_track_ts(["Artist - Title_0"], db)
    db.close()
    assert "rms_energy_bass_ts_mean" in result["shape"]
    assert "rms_energy_bass_ts_std"  in result["shape"]


def test_process_track_ts_tonic_rotation(tmp_path):
    db = _make_db(tmp_path, "Artist - Title_0")  # tonic_ts all = 5.0
    result = process_track_ts(["Artist - Title_0"], db)
    db.close()
    assert result is not None
    expected_sin = float(np.sin(2 * np.pi * 5 / 12))
    assert abs(result["shape"]["tonic_sin"] - expected_sin) < 1e-4
    # hpcp_rot should be rotated version of hpcp_raw
    raw = np.array(result["hpcp_raw"], dtype=np.float32)
    rot = np.array(result["hpcp_rot"], dtype=np.float32)
    expected_rot = _rotate_hpcp(raw, 5)
    np.testing.assert_allclose(rot, expected_rot, atol=1e-4)


def test_process_track_ts_variable_n_steps(tmp_path):
    db = TimeseriesDB(tmp_path / "var.db")
    # Two crops with different n_steps — both should contribute
    for key, n in [("T_0", 64), ("T_1", 128), ("T_2", 256)]:
        db.put(key, {"rms_energy_bass_ts": np.random.rand(n).astype(np.float32).tolist()})
    result = process_track_ts(["T_0", "T_1", "T_2"], db)
    db.close()
    assert result is not None
    assert len(result["curves"]["rms_energy_bass_ts"]) == 32


def test_process_track_ts_missing_crops_skipped(tmp_path):
    db = _make_db(tmp_path, "Artist - Title_0")
    # "Artist - Title_1" is not in DB — should be silently skipped
    result = process_track_ts(["Artist - Title_0", "Artist - Title_1"], db)
    db.close()
    assert result is not None  # still has data from crop 0


from plots.build_dataset_stats import build_embedding, run_similarity_pass


def _make_ts_data(n_tracks: int = 5) -> tuple[list, dict]:
    from plots.build_dataset_stats import TS_1D_FIELDS
    names = [f"Track {i}" for i in range(n_tracks)]
    ts_data = {}
    for name in names:
        shape = {}
        for f in TS_1D_FIELDS:
            shape[f + "_mean"] = float(np.random.rand())
            shape[f + "_std"]  = float(np.random.rand())
        for j in range(12):
            shape[f"hpcp_raw_{j}"] = float(np.random.rand())
            shape[f"hpcp_rot_{j}"] = float(np.random.rand())
        shape["tonic_sin"] = float(np.random.rand())
        shape["tonic_cos"] = float(np.random.rand())
        ts_data[name] = {"shape": shape, "curves": {}, "hpcp_raw": [0.0]*12, "hpcp_rot": [0.0]*12}
    return names, ts_data


def test_build_embedding_shape():
    names, ts_data = _make_ts_data(10)
    emb = build_embedding(names, ts_data)
    assert emb.shape == (10, 44)
    assert emb.dtype == np.float32


def test_build_embedding_zscored():
    names, ts_data = _make_ts_data(50)
    emb = build_embedding(names, ts_data)
    # After z-scoring, each column should have mean ≈ 0, std ≈ 1
    np.testing.assert_allclose(emb.mean(axis=0), np.zeros(44), atol=1e-4)


def test_build_embedding_missing_track_gets_zero():
    names = ["A", "B", "C"]
    ts_data = {"A": {"shape": {"rms_energy_bass_ts_mean": 1.0}, "curves": {}}}
    emb = build_embedding(names, ts_data)
    # B and C are missing — their rows should not raise and be finite
    assert np.all(np.isfinite(emb))


def test_run_similarity_pass_writes_js(tmp_path):
    from plots.build_dataset_stats import TS_1D_FIELDS
    names, ts_data = _make_ts_data(10)
    run_similarity_pass(names, ts_data, tmp_path)

    js = (tmp_path / "feature_explorer_timeseries.js").read_text()
    assert "TS_CURVES" in js
    assert "TS_NEIGHBORS" in js
    assert "overall" in js
    assert "key_locked" in js
    assert "pitch_shift" in js


def test_build_embedding_missing_rows_at_zero_after_zscore():
    """Missing tracks should land at 0 (population mean) after z-scoring."""
    names, ts_data = _make_ts_data(20)
    # Remove half the tracks so they're missing
    missing = names[10:]
    ts_data_partial = {k: v for k, v in ts_data.items() if k not in missing}
    emb = build_embedding(names, ts_data_partial)
    # Missing rows should be all zeros after z-scoring (population mean position)
    for name in missing:
        i = names.index(name)
        np.testing.assert_allclose(emb[i], np.zeros(44), atol=1e-4)
