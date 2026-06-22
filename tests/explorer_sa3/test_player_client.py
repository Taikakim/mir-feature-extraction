from plots.explorer_sa3 import player_client as pc


def test_decode_and_source_urls():
    assert pc.decode_url("000007") == "http://localhost:7892/decode?crop=000007"
    assert pc.source_url("000007") == "http://localhost:7892/source?crop=000007"


def test_mix_url():
    u = pc.mix_url("000001", "000002", t=0.25, interp="lerp")
    assert u == ("http://localhost:7892/mix?"
                 "crop_a=000001&crop_b=000002&t=0.250&interp=lerp")


def test_steer_url():
    u = pc.steer_url("000003", "hpcp", gain=64)
    assert u == ("http://localhost:7892/steer?"
                 "crop=000003&head=hpcp&gain=64.0")
