from plots.explorer_sa3 import player_client as pc


def test_decode_and_source_urls():
    assert pc.decode_url("000007") == f"{pc.BASE}/decode?crop=000007"
    assert pc.source_url("000007") == f"{pc.BASE}/source?crop=000007"


def test_mix_url():
    u = pc.mix_url("000001", "000002", t=0.25, interp="lerp")
    assert u == (f"{pc.BASE}/mix?"
                 "crop_a=000001&crop_b=000002&t=0.250&interp=lerp")


def test_steer_url():
    u = pc.steer_url("000003", "hpcp", gain=64)
    assert u == (f"{pc.BASE}/steer?"
                 "crop=000003&head=hpcp&gain=64.0")


def test_default_is_the_render_server():
    # 2026-08-25: the standalone torch player (7892) is retired -- it held a
    # SECOND resident SAME-L (7.12 GB). Its GET endpoints now live on the render
    # server (8056), which already holds that autoencoder. SA3_PLAYER_PORT=7893
    # still reaches the low-VRAM ONNX player; SA3_PLAYER_BASE overrides all.
    import os
    if "SA3_PLAYER_PORT" not in os.environ and "SA3_PLAYER_BASE" not in os.environ:
        assert pc.BASE == "http://localhost:8056"
