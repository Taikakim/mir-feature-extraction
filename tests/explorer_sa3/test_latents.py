import numpy as np
from plots.explorer_sa3.latents import (
    load_latent, content_frames, load_timeseries, FRAME_RATE_HZ,
)


def test_load_latent_fp16_to_fp32_and_shape(tmp_path):
    z = (np.random.randn(256, 4096)).astype(np.float16)
    np.save(tmp_path / "000000.npy", z)
    out = load_latent(tmp_path, "000000")
    assert out.dtype == np.float32 and out.shape == (256, 4096)
    assert np.allclose(out, z.astype(np.float32))


def test_load_latent_squeezes_batch_dim(tmp_path):
    z = np.zeros((1, 256, 8), dtype=np.float16)
    np.save(tmp_path / "000001.npy", z)
    assert load_latent(tmp_path, "000001").shape == (256, 8)


def test_load_latent_rejects_wrong_dims(tmp_path):
    np.save(tmp_path / "000002.npy", np.zeros((64, 256), dtype=np.float16))
    try:
        load_latent(tmp_path, "000002")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_content_frames():
    assert content_frames({"padding_mask": [1, 1, 1, 0, 0]}) == 3
    assert content_frames({}) == 4096


def test_load_timeseries(tmp_path):
    np.savez(tmp_path / "000000.TIMESERIES.npz",
             rms_energy_bass_ts=np.zeros(4096, np.float32),
             hpcp_ts=np.zeros((4096, 12), np.float32))
    ts = load_timeseries(tmp_path, "000000")
    assert ts["rms_energy_bass_ts"].shape == (4096,)
    assert ts["hpcp_ts"].shape == (4096, 12)


def test_frame_rate():
    assert abs(FRAME_RATE_HZ - 10.7666) < 1e-3


def test_content_frames_empty_mask_is_zero():
    assert content_frames({"padding_mask": []}) == 0
