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


def test_default_is_onnx_port():
    # default player is now the low-VRAM ONNX server (7893), not torch (7892);
    # overridable via SA3_PLAYER_PORT / SA3_PLAYER_BASE.
    import os
    if "SA3_PLAYER_PORT" not in os.environ and "SA3_PLAYER_BASE" not in os.environ:
        assert pc.BASE == "http://localhost:7893"
