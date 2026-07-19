"""Latent lab tab — latent data-bending via the render server (POST /bend).

Ops come from SAO/eval/latent_bend.py (spec grammar mirrors the weight-garden
Condition ops: an ordered list of {"op": ..., "amount": ..., + op keys}; one
seed drives all randomness, ops consume it in row order, so (ops, seed)
reproduces exactly). The latent picker mirrors the Viewer's track→crop browse;
a free .npy path wins when non-empty. `splice` is not offered here (it needs a
second latent — CLI/recipe territory). Source-vs-bent A/B: the picked crop's
player decode + source ride the shared audio_panel, the bent render lands in
the result players below.
"""
from __future__ import annotations

import datetime
import json

from dash import Input, Output, State, dcc, html, no_update

from . import audio_panel, render_client
from . import player_client as pc
from .sidecar_index import group_by_track

BEND_ROWS = 4
HISTORY_CAP = 50

# UI-offered ops (latent_bend.BEND_OPS minus splice) with their op-specific
# keys — everything beyond `amount` rides the per-row extra-JSON field.
_OP_KEYS = {
    "channel_swap": "amount = frac of C, swap random channel pairs",
    "channel_roll": '{"k": 32, "shift": 64} or {"k": 32, "max_shift": 256}',
    "noise": 'amount = × per-ch std; optional {"channels": [1, 5, 9]}',
    "quantize": '{"bits": 6}; amount = wet/dry',
    "segment_shuffle": '{"seg": 64}; amount = frac of segments',
    "band_scale": '{"channels": [1, 5, 9]}; amount = multiplier '
                  "(target features via latent_dim_feature_xcorr.csv)",
}
_TIP_OPS = "op-specific keys (JSON, merged into the spec):\n" + "\n".join(
    f"{k}: {v}" for k, v in _OP_KEYS.items())

_ROW_STYLE = {"display": "flex", "gap": "6px", "alignItems": "center",
              "flexWrap": "wrap", "marginBottom": "4px"}
_NUM_STYLE = {"width": "90px"}


def _op_row(i: int) -> html.Div:
    return html.Div([
        html.Span(f"op {i}", style={"minWidth": "40px", "fontWeight": "bold"}),
        dcc.Dropdown(id=f"bend-op{i}-op", value=None, clearable=True,
                     placeholder="op (off)",
                     options=[{"label": o, "value": o} for o in _OP_KEYS],
                     style={"width": "180px", "display": "inline-block"}),
        html.Span("amount"),
        dcc.Input(id=f"bend-op{i}-amount", type="number", value=0.25,
                  step=0.05, style=_NUM_STYLE),
        html.Span("extra keys (JSON)", title=_TIP_OPS,
                  style={"textDecoration": "underline dotted"}),
        dcc.Input(id=f"bend-op{i}-extra", type="text", value="",
                  placeholder='e.g. {"bits": 6}', style={"width": "260px"}),
    ], style=_ROW_STYLE)


def layout() -> html.Div:
    return html.Div([
        html.Div(id="bend-server-badge"),
        dcc.Interval(id="bend-interval", interval=2000),
        html.Label("Track"),
        dcc.Dropdown(id="bend-track-dd", placeholder="choose a track…"),
        html.Label("Crop (within track)"),
        dcc.Dropdown(id="bend-crop-dd", placeholder="choose a crop…"),
        html.Div([
            html.Span("or latent path (.npy, wins when non-empty)"),
            dcc.Input(id="bend-latent-path", type="text", value="",
                      style={"width": "50%"}),
        ], style=_ROW_STYLE),
        html.Div(id="bend-src-panel"),
        html.H4("Bend ops (applied in row order, one seeded RNG)"),
        *[_op_row(i) for i in range(1, BEND_ROWS + 1)],
        html.Div([
            html.Span("bend seed (-1 = random)"),
            dcc.Input(id="bend-seed", type="number", value=1234,
                      style=_NUM_STYLE),
            html.Button("Bend + decode", id="bend-render-btn",
                        style={"fontWeight": "bold"}),
        ], style=_ROW_STYLE),
        html.Pre(id="bend-status", style={"whiteSpace": "pre-wrap",
                                          "fontSize": "0.8em"}),
        html.Div(id="bend-result"),
        dcc.Store(id="bend-history", data=[]),
        html.H4("History"),
        html.Div(id="bend-history-div"),
    ], style={"padding": "0.5em 1em"})


def _players(urls, files) -> list:
    out = []
    for u, f in zip(urls, files or [None] * len(urls)):
        out.append(html.Div([
            html.Audio(src=render_client.audio_url(u), controls=True),
            html.Div(f or "", style={"fontSize": "11px", "color": "#666"}),
        ]))
    return out


def build_ops(rows) -> list[dict]:
    """3 values per row (op, amount, extra-json) -> latent_bend spec list.
    Raises ValueError on bad JSON; rows with no op are skipped."""
    ops = []
    for i in range(BEND_ROWS):
        op, amount, extra = rows[i * 3:(i + 1) * 3]
        if op in (None, "", "none"):
            continue
        spec = {"op": op}
        if amount is not None:
            spec["amount"] = float(amount)
        if (extra or "").strip():
            try:
                more = json.loads(extra)
            except ValueError:
                raise ValueError(f"op row {i + 1}: bad extra JSON "
                                 f"{extra!r}") from None
            if not isinstance(more, dict):
                raise ValueError(f"op row {i + 1}: extra JSON must be an "
                                 "object, e.g. {\"bits\": 6}")
            spec.update(more)
        ops.append(spec)
    return ops


def register_callbacks(app, index, latent_dir) -> None:
    tracks = group_by_track(index)
    track_opts = [{"label": f"{t or '(unknown track)'} — {len(cs)} crops",
                   "value": t} for t, cs in sorted(tracks.items())]

    @app.callback(Output("bend-track-dd", "options"),
                  Input("bend-track-dd", "id"))
    def _track_fill(_):
        return track_opts

    @app.callback(Output("bend-crop-dd", "options"),
                  Output("bend-crop-dd", "value"),
                  Input("bend-track-dd", "value"))
    def _track_pick(track):
        if track is None or track not in tracks:
            return [], None
        crops = sorted(tracks[track], key=lambda c: c.rel_pos)
        opts = [{"label": f"{c.id} — pos {c.rel_pos:.2f}", "value": c.id}
                for c in crops]
        return opts, crops[0].id

    @app.callback(Output("bend-src-panel", "children"),
                  Input("bend-crop-dd", "value"))
    def _src_panel(cid):
        if not cid:
            return no_update
        return html.Div([
            html.Div("source crop (unbent decode + original, via the player):",
                     style={"fontSize": "0.85em", "color": "#666"}),
            audio_panel.panel(cid, pc.status()),
        ])

    @app.callback(Output("bend-server-badge", "children"),
                  Input("bend-interval", "n_intervals"))
    def _badge(_n):
        st = render_client.status()
        if st is None:
            return html.Div([
                html.B("Render server DOWN — launch with:"),
                html.Pre(render_client.LAUNCH_HINT),
            ], style={"color": "#a00"})
        label = ("BUSY — " + str(st.get("job_id"))) if st.get("busy") else "idle"
        return html.Div(html.B(f"Render server up · {label}"),
                        style={"color": "#070"})

    @app.callback(
        Output("bend-result", "children"),
        Output("bend-history", "data"),
        Output("bend-status", "children"),
        Input("bend-render-btn", "n_clicks"),
        State("bend-crop-dd", "value"),
        State("bend-latent-path", "value"),
        State("bend-seed", "value"),
        State("bend-history", "data"),
        *[State(f"bend-op{i}-{s}", "value")
          for i in range(1, BEND_ROWS + 1)
          for s in ("op", "amount", "extra")],
        prevent_initial_call=True)
    def _render(_n, crop, latent_path, seed, history, *rows):
        try:
            ops = build_ops(rows)
        except ValueError as e:
            return no_update, no_update, str(e)
        if not ops:
            return no_update, no_update, "error: no op rows active"
        payload = {"ops": ops,
                   "seed": int(seed if seed is not None else 1234),
                   "latent_dir": str(latent_dir)}
        latent_path = (latent_path or "").strip()
        if latent_path:
            payload["latent_path"] = latent_path
        elif crop:
            payload["crop_id"] = crop
        else:
            return no_update, no_update, "error: pick a crop or enter a latent path"
        try:
            resp = render_client.bend(payload)
        except render_client.RenderError as e:
            return no_update, no_update, f"BEND ERROR:\n{e.args[0]}"
        urls = resp.get("urls", [])
        files = resp.get("files", [])
        label = (f"{latent_path or crop} "
                 f"[{'+'.join(o['op'] for o in ops)} seed={resp.get('seed')}]")
        entry = {"ts": datetime.datetime.now().strftime("%H:%M:%S"),
                 "label": label, "urls": urls, "files": files,
                 "meta": resp.get("meta", {}), "params": payload}
        history = ([entry] + (history or []))[:HISTORY_CAP]
        status = (f"ok · seed={resp.get('seed')} · "
                  f"total={resp.get('timings', {}).get('total_sec')}s\n"
                  f"ops: {json.dumps(ops)}")
        return _players(urls, files), history, status

    @app.callback(Output("bend-history-div", "children"),
                  Input("bend-history", "data"))
    def _history(data):
        rows = []
        for e in (data or []):
            rows.append(html.Div([
                html.B(f"{e.get('ts')} {e.get('label')}"),
                *_players(e.get("urls", []), e.get("files", [])),
            ], style={"borderBottom": "1px solid #ccc",
                      "padding": "4px 0"}))
        return rows
