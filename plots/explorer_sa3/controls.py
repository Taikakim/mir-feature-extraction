"""Shared steering panel (LatCH slots + FiLM + DoRA) for the SA3 explorer.

Frozen contract (design B.2). Dash ids are app-global, so the panel is
prefix-parameterized: instantiate once per tab with ns="inf" and ns="a2a".
`steering_states(ns)` returns EXACTLY 23 States in a fixed order which
`steering_payload(vals)` consumes to build the shared latch/film/dora JSON
blocks accepted by /generate, /a2a_track and /a2a_mix (design A.4).
"""
from __future__ import annotations
from typing import Sequence

from dash import Input, Output, State, dcc, html, no_update

from . import render_client

LATCH_SLOTS = 3

# fallback when /info is unreachable at payload-build time (server hard-codes
# the same default in film_default)
_FILM_CKPT_FALLBACK = ("/run/media/kim/Mantu1/sa3_control_runs/"
                       "onset_Fusion_lr1e-4_randomcrop/riffer_final.pt")

_KIND_OPTIONS = ["constant", "ramp_up", "ramp_down", "beat_grid"]
_DORA_OPTIONS = ["none", "hof", "newstack", "evr1x"]

_ROW_STYLE = {"display": "flex", "gap": "6px", "alignItems": "center",
              "flexWrap": "wrap", "marginBottom": "4px"}
_NUM_STYLE = {"width": "90px"}


def _latch_slot(ns: str, i: int) -> html.Div:
    p = f"{ns}-ctl-latch{i}"
    return html.Div([
        html.Span(f"LatCH {i}", style={"minWidth": "60px", "fontWeight": "bold"}),
        dcc.Dropdown(id=f"{p}-head", options=[], value=None, clearable=True,
                     placeholder="head (off)",
                     style={"width": "260px", "display": "inline-block"}),
        dcc.Dropdown(id=f"{p}-kind", options=_KIND_OPTIONS, value="constant",
                     clearable=False,
                     style={"width": "130px", "display": "inline-block"}),
        html.Span("value"),
        dcc.Input(id=f"{p}-value", type="number", value=-30.0, step=0.1,
                  style=_NUM_STYLE),
        html.Span("gain"),
        dcc.Input(id=f"{p}-gain", type="number", value=512.0, min=0, step=1,
                  style=_NUM_STYLE),
        html.Span("start"),
        dcc.Input(id=f"{p}-start", type="number", value=0.0, min=0, max=1,
                  step=0.01, style=_NUM_STYLE),
        html.Span("end"),
        dcc.Input(id=f"{p}-end", type="number", value=0.6, min=0, max=1,
                  step=0.01, style=_NUM_STYLE),
    ], style=_ROW_STYLE)


def steering_panel(ns: str, dora_default: str = "none",
                   collapsed: bool = True) -> html.Div:
    """Steering panel with ids {ns}-ctl-* (see design B.2 table)."""
    body = html.Div([
        *[_latch_slot(ns, i) for i in range(1, LATCH_SLOTS + 1)],
        html.Div([
            dcc.Checklist(id=f"{ns}-ctl-film-enable",
                          options=[{"label": " FiLM density adapter",
                                    "value": "on"}],
                          value=[],
                          style={"display": "inline-block"}),
            html.Span("onsets/s"),
            dcc.Input(id=f"{ns}-ctl-film-value", type="number", value=4.0,
                      step=0.1, style=_NUM_STYLE),
            html.Span("gain"),
            dcc.Input(id=f"{ns}-ctl-film-gain", type="number", value=1.75,
                      min=0, max=5, step=0.05, style=_NUM_STYLE),
        ], style=_ROW_STYLE),
        html.Div([
            html.Span("DoRA", style={"minWidth": "60px", "fontWeight": "bold"}),
            dcc.Dropdown(id=f"{ns}-ctl-dora-dd", options=_DORA_OPTIONS,
                         value=dora_default, clearable=False,
                         style={"width": "160px", "display": "inline-block"}),
            html.Span("strength"),
            dcc.Input(id=f"{ns}-ctl-dora-strength", type="number", value=1.0,
                      min=0, max=10, step=0.01, style=_NUM_STYLE),
        ], style=_ROW_STYLE),
    ])
    return html.Div(html.Details([
        html.Summary("Steering (LatCH / FiLM / DoRA)"),
        body,
    ], open=not collapsed))


def steering_states(ns: str) -> list[State]:
    """EXACTLY this order, length 23 (frozen; steering_payload depends on it)."""
    states: list[State] = []
    for i in range(1, LATCH_SLOTS + 1):
        p = f"{ns}-ctl-latch{i}"
        for suffix in ("head", "kind", "value", "gain", "start", "end"):
            states.append(State(f"{p}-{suffix}", "value"))
    states += [State(f"{ns}-ctl-film-enable", "value"),
               State(f"{ns}-ctl-film-value", "value"),
               State(f"{ns}-ctl-film-gain", "value"),
               State(f"{ns}-ctl-dora-dd", "value"),
               State(f"{ns}-ctl-dora-strength", "value")]
    return states


def steering_payload(vals: Sequence) -> dict:
    """Consume exactly 23 values in steering_states order -> A.4 blocks."""
    if len(vals) != 23:
        raise ValueError(f"steering_payload expects 23 values, got {len(vals)}")
    latch = []
    for i in range(LATCH_SLOTS):
        head, kind, value, gain, start, end = vals[i * 6:(i + 1) * 6]
        if head in (None, "none", ""):
            continue
        latch.append({
            "head": head,
            "kind": kind or "constant",
            "value": float(value) if value is not None else -30.0,
            "gain": float(gain) if gain is not None else 512.0,
            "start_pct": float(start) if start is not None else 0.0,
            "end_pct": float(end) if end is not None else 0.6,
        })
    film_enable, film_value, film_gain = vals[18:21]
    film = None
    if film_enable and "on" in film_enable:
        srv = render_client.info() or {}
        ckpt = (srv.get("film_default") or {}).get("ckpt", _FILM_CKPT_FALLBACK)
        film = {"ckpt": ckpt,
                "value": float(film_value) if film_value is not None else 4.0,
                "gain": float(film_gain) if film_gain is not None else 1.75}
    dora_dd, dora_strength = vals[21:23]
    dora = None
    if dora_dd == "none":
        # explicit "none" must reach the server by name: an absent dora block
        # means "use the op default" there (evr1x for /a2a_mix), which would
        # silently override the user's base-model choice
        dora = {"name": "none"}
    elif dora_dd:
        dora = {"name": dora_dd, "ckpt_path": None,
                "strength": (float(dora_strength)
                             if dora_strength is not None else 1.0)}
    return {"latch": latch, "film": film, "dora": dora}


def register(app, ns: str) -> None:
    """Head-options fill + per-head defaults autofill for one namespace."""
    for i in range(1, LATCH_SLOTS + 1):
        head_id = f"{ns}-ctl-latch{i}-head"

        @app.callback(Output(head_id, "options"), Input(head_id, "id"))
        def _fill_options(_id):
            srv = render_client.info()
            if not srv:
                return []
            return [{"label": f"{h['name']} [{h.get('family', '?')}]",
                     "value": h["name"]}
                    for h in srv.get("latch_heads", [])]

        @app.callback(Output(f"{ns}-ctl-latch{i}-gain", "value"),
                      Output(f"{ns}-ctl-latch{i}-kind", "value"),
                      Output(f"{ns}-ctl-latch{i}-value", "value"),
                      Input(head_id, "value"))
        def _autofill_defaults(head):
            if head in (None, "none", ""):
                return no_update, no_update, no_update
            srv = render_client.info()
            if not srv:
                return no_update, no_update, no_update
            for h in srv.get("latch_heads", []):
                if h["name"] == head:
                    return (h.get("default_gain", 512.0),
                            h.get("target_kind_default", "constant"),
                            h.get("value_default", -30.0))
            return no_update, no_update, no_update
