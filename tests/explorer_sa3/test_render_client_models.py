"""The multi-root client surface. Pins that the legacy single-root call is
unchanged -- the picker was pinned to sa3_lora_runs for months and the fix must
not break the callers that still pass nothing.
"""
from plots.explorer_sa3 import render_client


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._p


def test_roots_hits_the_roots_endpoint(monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen["url"], seen["params"] = url, params
        return _Resp({"ok": True, "roots": []})

    monkeypatch.setattr(render_client.requests, "get", fake_get)
    assert render_client.roots()["ok"] is True
    assert seen["url"].endswith("/roots")


def test_models_forwards_every_filter(monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen.update(url=url, params=params)
        return _Resp({"ok": True, "models": [], "count": 0})

    monkeypatch.setattr(render_client.requests, "get", fake_get)
    render_client.models(root_ids=["a", "b"], family="adapter", q="dora", rescan=True)
    assert seen["url"].endswith("/models")
    assert seen["params"]["root_ids"] == "a,b"
    assert seen["params"]["family"] == "adapter"
    assert seen["params"]["q"] == "dora"
    assert seen["params"]["rescan"] == 1


def test_ckpts_still_works_with_no_arguments(monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen.update(params=params)
        return _Resp({"ok": True, "ckpts": [], "root": "/r"})

    monkeypatch.setattr(render_client.requests, "get", fake_get)
    render_client.ckpts()
    assert seen["params"] == {"rescan": 0}, "legacy call must not gain params"


def test_ckpts_forwards_root_ids_as_a_comma_list(monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen.update(params=params)
        return _Resp({"ok": True, "ckpts": []})

    monkeypatch.setattr(render_client.requests, "get", fake_get)
    render_client.ckpts(root_ids=["lumi_uuid", "local_dora"])
    assert seen["params"]["root_ids"] == "lumi_uuid,local_dora"


def test_a_404_from_ckpts_is_passed_through_not_swallowed(monkeypatch):
    # An unmounted removable drive 404s with a named reason; the picker must be
    # able to say WHICH drive rather than blaming the connection.
    def fake_get(url, params=None, timeout=None):
        return _Resp({"ok": False, "error": "scan root not found: /x — is Mantu1 mounted?"},
                     status=404)

    monkeypatch.setattr(render_client.requests, "get", fake_get)
    got = render_client.ckpts()
    assert got is not None and got["ok"] is False and "Mantu1" in got["error"]


def test_unreachable_server_returns_none_not_an_exception(monkeypatch):
    def boom(*a, **k):
        raise OSError("down")

    monkeypatch.setattr(render_client.requests, "get", boom)
    assert render_client.roots() is None
    assert render_client.models() is None
    assert render_client.ckpts() is None
