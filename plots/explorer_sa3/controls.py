"""Shared steering panel (LatCH slots + FiLM + DoRA) for the SA3 explorer.

Frozen contract (design B.2). Dash ids are app-global, so the panel is
prefix-parameterized: instantiate once per tab with ns="inf" and ns="a2a".
`steering_states(ns)` returns EXACTLY 23 States in a fixed order which
`steering_payload(vals)` consumes to build the shared latch/film/dora JSON
blocks accepted by /generate, /a2a_track and /a2a_mix (design A.4).
"""
from __future__ import annotations
import os
from typing import Sequence

from dash import Input, Output, State, dcc, html, no_update

from . import render_client

LATCH_SLOTS = 3

# fallback when /info is unreachable at payload-build time (server hard-codes
# the same default in film_default)
# The eval drive is REMOVABLE and udisks labels it Mantu or Mantu1 depending on whether the
# label collided at mount time — the same reason explorer_render_server.py resolves it at import
# rather than hardcoding. This fallback hardcoded Mantu1, which does not exist today (the drive
# mounts as Mantu), so the degraded path — used only when /info is unreachable at payload-build
# time — pointed at a file that is not there. Verified 2026-08-23: the same ckpt IS under Mantu.
def _resolve_mantu():
    for _d in ("/run/media/kim/Mantu", "/run/media/kim/Mantu1"):
        if os.path.isdir(os.path.join(_d, "sa3_control_runs")):
            return _d
    return "/run/media/kim/Mantu"


_FILM_CKPT_FALLBACK = os.path.join(
    _resolve_mantu(), "sa3_control_runs", "onset_Fusion_lr1e-4_randomcrop", "riffer_final.pt")

_KIND_OPTIONS = ["constant", "ramp_up", "ramp_down", "beat_grid"]
# loss shaping per head (latch_guided.head_loss); "" = head/server default.
# scalar_pooled = the 2026-07-10 buzz fix for scalar heads; chroma_rung* = T2.
_LOSS_OPTIONS = ["", "mse", "smooth_l1", "l1", "huber", "bce_logits",
                 "cosine", "scalar_pooled", "chroma_rung1", "chroma_rung2"]
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
        html.Span("loss"),
        dcc.Dropdown(id=f"{p}-loss", options=_LOSS_OPTIONS, value="",
                     clearable=False, placeholder="default",
                     style={"width": "130px", "display": "inline-block"}),
        html.Span("w_sec"),
        dcc.Input(id=f"{p}-wsec", type="number", value=None, min=0, step=0.1,
                  placeholder="def", style=_NUM_STYLE),
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
            html.Span("σ-interval"),
            dcc.Input(id=f"{ns}-ctl-dora-imin", type="number", value=0.0,
                      min=0, max=1, step=0.05, style=_NUM_STYLE),
            dcc.Input(id=f"{ns}-ctl-dora-imax", type="number", value=1.0,
                      min=0, max=1, step=0.05, style=_NUM_STYLE),
        ], style=_ROW_STYLE),
        html.Div([
            html.Span("LatCH adv", style={"minWidth": "60px",
                                          "fontWeight": "bold"}),
            html.Span("ρ"),
            dcc.Input(id=f"{ns}-ctl-latch-rho", type="number", value=None,
                      min=0, step=1, placeholder="auto", style=_NUM_STYLE),
            html.Span("μ"),
            dcc.Input(id=f"{ns}-ctl-latch-mu", type="number", value=None,
                      min=0, step=1, placeholder="auto", style=_NUM_STYLE),
            html.Span("γ"),
            dcc.Input(id=f"{ns}-ctl-latch-gamma", type="number", value=0.3,
                      min=0, max=1, step=0.05, style=_NUM_STYLE),
            html.Span("n_iter"),
            dcc.Input(id=f"{ns}-ctl-latch-niter", type="number", value=4,
                      min=1, max=16, step=1, style=_NUM_STYLE),
        ], style=_ROW_STYLE),
    ])
    return html.Div(html.Details([
        html.Summary("Steering (LatCH / FiLM / DoRA)"),
        body,
    ], open=not collapsed))


def steering_states(ns: str) -> list[State]:
    """EXACTLY this order, length 35 (v2 contract, extended append-only per slot
    2026-07-12: +loss/+wsec per slot, +rho/mu/gamma/n_iter advanced row;
    steering_payload depends on it)."""
    states: list[State] = []
    for i in range(1, LATCH_SLOTS + 1):
        p = f"{ns}-ctl-latch{i}"
        for suffix in ("head", "kind", "value", "gain", "start", "end",
                       "loss", "wsec"):
            states.append(State(f"{p}-{suffix}", "value"))
    states += [State(f"{ns}-ctl-film-enable", "value"),
               State(f"{ns}-ctl-film-value", "value"),
               State(f"{ns}-ctl-film-gain", "value"),
               State(f"{ns}-ctl-dora-dd", "value"),
               State(f"{ns}-ctl-dora-strength", "value"),
               State(f"{ns}-ctl-dora-imin", "value"),
               State(f"{ns}-ctl-dora-imax", "value"),
               State(f"{ns}-ctl-latch-rho", "value"),
               State(f"{ns}-ctl-latch-mu", "value"),
               State(f"{ns}-ctl-latch-gamma", "value"),
               State(f"{ns}-ctl-latch-niter", "value")]
    return states


def steering_payload(vals: Sequence) -> dict:
    """Consume exactly 35 values in steering_states order -> A.4 blocks (v2)."""
    if len(vals) != 35:
        raise ValueError(f"steering_payload expects 35 values, got {len(vals)}")
    latch = []
    for i in range(LATCH_SLOTS):
        head, kind, value, gain, start, end, loss, wsec = vals[i * 8:(i + 1) * 8]
        if head in (None, "none", ""):
            continue
        slot = {
            "head": head,
            "kind": kind or "constant",
            "value": float(value) if value is not None else -30.0,
            "gain": float(gain) if gain is not None else 512.0,
            "start_pct": float(start) if start is not None else 0.0,
            "end_pct": float(end) if end is not None else 0.6,
        }
        if loss:
            slot["loss_type"] = loss
        if wsec is not None:
            slot["w_sec"] = float(wsec)
        latch.append(slot)
    film_enable, film_value, film_gain = vals[24:27]
    film = None
    if film_enable and "on" in film_enable:
        srv = render_client.info() or {}
        ckpt = (srv.get("film_default") or {}).get("ckpt", _FILM_CKPT_FALLBACK)
        film = {"ckpt": ckpt,
                "value": float(film_value) if film_value is not None else 4.0,
                "gain": float(film_gain) if film_gain is not None else 1.75}
    dora_dd, dora_strength, dora_imin, dora_imax = vals[27:31]
    dora = None
    if dora_dd == "none":
        # explicit "none" must reach the server by name: an absent dora block
        # means "use the op default" there (evr1x for /a2a_mix), which would
        # silently override the user's base-model choice. strength rides along
        # because a top-level ckpt_path folds into this same dict server-side
        # (resolve_dora_req) and honors dora.strength — same as
        # bracket._apply_weight; unused when no adapter loads
        dora = {"name": "none",
                "strength": (float(dora_strength)
                             if dora_strength is not None else 1.0)}
    elif dora_dd:
        dora = {"name": dora_dd, "ckpt_path": None,
                "strength": (float(dora_strength)
                             if dora_strength is not None else 1.0)}
    if dora is not None:
        # sigma-native interval gating (dit.py:466): e.g. (0.25, 1.0) disables
        # the adapter for the final low-noise detail steps
        dora["interval_min"] = float(dora_imin) if dora_imin is not None else 0.0
        dora["interval_max"] = float(dora_imax) if dora_imax is not None else 1.0
    out = {"latch": latch, "film": film, "dora": dora}
    # advanced LatCH hparams ride top-level (server resolve_latch reads them);
    # blank rho/mu = keep the auto slot-1-gain tie
    rho, mu, gamma, niter = vals[31:35]
    if rho is not None:
        out["rho"] = float(rho)
    if mu is not None:
        out["mu"] = float(mu)
    if gamma is not None:
        out["gamma"] = float(gamma)
    if niter is not None:
        out["n_iter"] = int(niter)
    return out


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
