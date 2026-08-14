import numpy as np

from src.tools.avp_f0_augment_transform import transform_track


def _write_source(track_dir, n=100):
    track_dir.mkdir(parents=True, exist_ok=True)
    f0_other = np.linspace(55.0, 110.0, n)
    voiced = np.ones(n)
    voiced[:10] = 0.0
    np.savez(
        track_dir / f"{track_dir.name}.TIMESERIES.npz",
        f0_other_ts=f0_other.astype(np.float32),
        f0_bass_ts=(f0_other / 2).astype(np.float32),
        f0_other_voiced_ts=voiced.astype(np.float32),
        f0_bass_voiced_ts=voiced.astype(np.float32),
        rms_energy_bass_ts=np.zeros(n, dtype=np.float32),
    )


def _write_variant(track_dir, name, n_frames):
    vdir = track_dir / "augmentations" / name
    vdir.mkdir(parents=True, exist_ok=True)
    np.savez(vdir / f"{name}.TIMESERIES.npz",
              rms_energy_bass_ts=np.zeros(n_frames, dtype=np.float32))
    return vdir


def test_pitch_variant_scales_values_same_frame_count(tmp_path):
    td = tmp_path / "A - T"
    _write_source(td, n=100)
    _write_variant(td, "pitch+2", n_frames=100)  # pitch preserves duration

    status = transform_track(td)
    assert status["pitch+2"] == "written"

    out = np.load(td / "augmentations/pitch+2/pitch+2.TIMESERIES.npz")
    src = np.load(td / f"{td.name}.TIMESERIES.npz")
    expected = src["f0_other_ts"] * (2.0 ** (2 / 12))
    assert out["f0_other_ts"].shape == (100,)
    assert np.allclose(out["f0_other_ts"], expected, atol=1e-3)
    # voicing mask is untouched by a pitch shift (same frame grid, no resample needed)
    assert np.allclose(out["f0_other_voiced_ts"], src["f0_other_voiced_ts"])


def test_tempo_variant_resamples_time_axis_values_unchanged(tmp_path):
    td = tmp_path / "A - T"
    _write_source(td, n=100)
    _write_variant(td, "tempo+10", n_frames=91)  # faster -> shorter, EMPIRICAL length

    status = transform_track(td)
    assert status["tempo+10"] == "written"

    out = np.load(td / "augmentations/tempo+10/tempo+10.TIMESERIES.npz")
    assert out["f0_other_ts"].shape == (91,)
    # endpoints survive a linear resample -- first/last source values recur at the new grid's ends
    src = np.load(td / f"{td.name}.TIMESERIES.npz")
    assert np.isclose(out["f0_other_ts"][0], src["f0_other_ts"][0], atol=1e-3)
    assert np.isclose(out["f0_other_ts"][-1], src["f0_other_ts"][-1], atol=1e-3)


def test_skips_existing_unless_overwrite(tmp_path):
    td = tmp_path / "A - T"
    _write_source(td, n=50)
    vdir = _write_variant(td, "pitch+1", n_frames=50)
    # pre-seed the variant npz as if a prior run already wrote f0 fields
    existing = dict(np.load(vdir / "pitch+1.TIMESERIES.npz").items())
    existing.update({
        "f0_other_ts": np.zeros(50, dtype=np.float32),
        "f0_bass_ts": np.zeros(50, dtype=np.float32),
        "f0_other_voiced_ts": np.zeros(50, dtype=np.float32),
        "f0_bass_voiced_ts": np.zeros(50, dtype=np.float32),
    })
    np.savez(vdir / "pitch+1.TIMESERIES.npz", **existing)

    status = transform_track(td)
    assert status["pitch+1"] == "skipped"

    status = transform_track(td, overwrite=True)
    assert status["pitch+1"] == "written"


def test_track_missing_source_f0_yields_no_status(tmp_path):
    td = tmp_path / "A - T"
    td.mkdir()
    np.savez(td / f"{td.name}.TIMESERIES.npz", rms_energy_bass_ts=np.zeros(10, dtype=np.float32))
    _write_variant(td, "pitch+1", n_frames=10)
    assert transform_track(td) == {}
