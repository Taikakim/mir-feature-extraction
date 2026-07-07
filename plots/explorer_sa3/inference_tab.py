"""Inference tab — text-to-audio + init-audio (a2a_track) renders via the
SA3 explorer render server (design B.3).

Routing: non-empty `inf-init-path` -> POST /a2a_track, else POST /generate.
The generic steering panel (LatCH / FiLM / DoRA) is shared with the A2A tab
via `controls` (ns="inf"). Long renders block this callback's worker thread
synchronously; the 2 s interval keeps the status badge live meanwhile.
"""
from __future__ import annotations
import time

from dash import Input, Output, State, dcc, html, no_update

from . import controls, render_client

_HISTORY_CAP = 50


def layout() -> html.Div:
    return html.Div([
        html.Div(id="inf-server-badge"),
        dcc.Interval(id="inf-interval", interval=2000),
        html.Div([
            html.Label("Prompt"),
            dcc.Textarea(
                id="inf-prompt",
                placeholder=("TrackType: Music, VocalType: Instrumental, "
                             "... (prefix vocabulary, then free text)"),
                style={"width": "100%", "height": "60px"}),
            html.Label("Negative prompt"),
            dcc.Input(id="inf-negprompt", type="text", value="",
                      style={"width": "100%"}),
        ]),
        html.Div([
            html.Span("duration (s)"),
            dcc.Input(id="inf-duration", type="number", value=47, min=1,
                      max=378, step=1),
            html.Span("steps"),
            dcc.Input(id="inf-steps", type="number", value=24, min=1, max=500),
            html.Span("cfg"),
            dcc.Input(id="inf-cfg", type="number", value=6.0, min=0, max=25,
                      step=0.1),
            html.Span("seed"),
            dcc.Input(id="inf-seed", type="number", value=-1),
            html.Span("batch"),
            dcc.Input(id="inf-batch", type="number", value=1, min=1, max=4),
            html.Span("apg"),
            dcc.Input(id="inf-apg", type="number", value=1.0, min=0, max=1,
                      step=0.1),
            html.Span("dur pad (s)"),
            dcc.Input(id="inf-durpad", type="number", value=6.0, min=0, max=30,
                      step=0.5),
        ], style={"display": "flex", "gap": "6px", "alignItems": "center",
                  "flexWrap": "wrap", "marginTop": "8px"}),
        html.Div([
            html.Span("init audio path (non-empty → a2a_track)"),
            dcc.Input(id="inf-init-path", type="text", value="",
                      style={"width": "50%"}),
            html.Span("init noise"),
            dcc.Input(id="inf-init-noise", type="number", value=0.4, min=0.05,
                      max=0.95, step=0.01),
            html.Span("noise ladder (e.g. 0.35,0.42,0.5)"),
            dcc.Input(id="inf-noise-ladder", type="text", value=""),
        ], style={"display": "flex", "gap": "6px", "alignItems": "center",
                  "flexWrap": "wrap", "marginTop": "8px"}),
        controls.steering_panel("inf", dora_default="none"),
        html.Button("Render", id="inf-render-btn"),
        html.Pre(id="inf-status", style={"whiteSpace": "pre-wrap"}),
        html.Div(id="inf-result"),
        dcc.Store(id="inf-history", data=[]),
        html.H4("History"),
        html.Div(id="inf-history-div"),
    ])


def _players(urls, files) -> list:
    out = []
    for u, f in zip(urls, files or [None] * len(urls)):
        out.append(html.Div([
            html.Audio(src=render_client.audio_url(u), controls=True),
            html.Div(f or "", style={"fontSize": "11px", "color": "#666"}),
        ]))
    return out


def _parse_ladder(text: str) -> list[float] | None:
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    if not parts:
        return None
    return [float(p) for p in parts]


def register_callbacks(app) -> None:
    controls.register(app, "inf")

    @app.callback(Output("inf-server-badge", "children"),
                  Input("inf-interval", "n_intervals"))
    def _badge(_n):
        st = render_client.status()
        if st is None:
            return html.Div([
                html.B("Render server DOWN — launch with:"),
                html.Pre(render_client.LAUNCH_HINT),
            ], style={"color": "#a00"})
        tail = "\n".join((st.get("log_tail") or [])[-5:])
        label = ("BUSY — " + str(st.get("job_id"))) if st.get("busy") else "idle"
        return html.Div([
            html.B(f"Render server up · {label}"),
            html.Pre(tail, style={"fontSize": "11px", "margin": "2px 0"}),
        ], style={"color": "#070"})

    @app.callback(
        Output("inf-result", "children"),
        Output("inf-history", "data"),
        Output("inf-status", "children"),
        Input("inf-render-btn", "n_clicks"),
        State("inf-prompt", "value"),
        State("inf-negprompt", "value"),
        State("inf-duration", "value"),
        State("inf-steps", "value"),
        State("inf-cfg", "value"),
        State("inf-seed", "value"),
        State("inf-batch", "value"),
        State("inf-apg", "value"),
        State("inf-durpad", "value"),
        State("inf-init-path", "value"),
        State("inf-init-noise", "value"),
        State("inf-noise-ladder", "value"),
        State("inf-history", "data"),
        *controls.steering_states("inf"),
        prevent_initial_call=True)
    def _render(_n, prompt, negprompt, duration, steps, cfg, seed, batch,
                apg, durpad, init_path, init_noise, ladder, history, *steer):
        if not (prompt or "").strip():
            return no_update, no_update, "error: prompt is empty"
        try:
            steering = controls.steering_payload(list(steer))
        except Exception as e:
            return no_update, no_update, f"steering error: {e}"
        common = {"steps": int(steps or 24), "cfg_scale": float(cfg or 6.0),
                  "seed": int(seed if seed is not None else -1)}
        if (init_path or "").strip():
            op = "a2a_track"
            payload = {"audio_path": init_path.strip(),
                       "prompt": prompt,
                       "noise_level": float(init_noise or 0.4),
                       **common, **steering}
            try:
                levels = _parse_ladder(ladder)
            except ValueError:
                return no_update, no_update, f"bad noise ladder: {ladder!r}"
            if levels:
                payload["noise_levels"] = levels
        else:
            op = "generate"
            payload = {"prompt": prompt,
                       "negative_prompt": negprompt or "",
                       "duration": float(duration or 47),
                       "batch_size": int(batch or 1),
                       "apg_scale": float(apg if apg is not None else 1.0),
                       "duration_padding_sec": float(
                           durpad if durpad is not None else 6.0),
                       **common, **steering}
        try:
            resp = render_client.render(op, payload)
        except render_client.RenderError as e:
            return no_update, no_update, str(e)
        urls = resp.get("urls", [])
        files = resp.get("files", [])
        entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "op": op,
                 "label": (prompt or "")[:60],
                 "urls": urls, "files": files,
                 "meta": resp.get("meta", {})}
        history = ([entry] + (history or []))[:_HISTORY_CAP]
        status_txt = (f"ok · op={op} · seed={resp.get('seed')} · "
                      f"total={resp.get('timings', {}).get('total_sec')}s\n"
                      f"warnings: {resp.get('warnings')}\n"
                      f"meta: {resp.get('meta')}")
        return _players(urls, files), history, status_txt

    @app.callback(Output("inf-history-div", "children"),
                  Input("inf-history", "data"))
    def _history(data):
        rows = []
        for e in (data or []):
            rows.append(html.Div([
                html.B(f"{e.get('ts')} [{e.get('op')}] {e.get('label')}"),
                *_players(e.get("urls", []), e.get("files", [])),
            ], style={"borderBottom": "1px solid #ccc",
                      "padding": "4px 0"}))
        return rows
