"""A2A transition tab — two-track overlay, transition window, render via the
SA3 explorer render server (POST /a2a_mix).

Layout ids are all ``a2a-*``; the shared steering panel is instantiated with
``controls.steering_panel("a2a", dora_default="evr1x")`` (ids ``a2a-ctl-*``).
Harmonic (chroma) steering defaults ON. Dash has no native drag: placement is
numeric+slider anchor controls updating a plotly overlay figure.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, no_update

from . import controls, render_client

ENV_HZ_TARGET = 40.0          # envelope block-max rate (hop 1102 samples @ 44.1k)
HISTORY_CAP = 50
_NUM = {"type": "number", "style": {"width": "6.5em"}}


# ---------------------------------------------------------------------------
# layout helpers
# ---------------------------------------------------------------------------

def _row(*children, **style):
    st = {"display": "flex", "alignItems": "center", "gap": "0.6em",
          "flexWrap": "wrap", "margin": "0.25em 0"}
    st.update(style)
    return html.Div(list(children), style=st)


def _lbl(txt):
    return html.Span(txt, style={"fontSize": "0.85em", "color": "#666"})


def _source_row(side: str) -> html.Div:
    return _row(
        html.B(side.upper(), style={"width": "1.2em"}),
        dcc.Dropdown(id=f"a2a-{side}-crop-dd", placeholder="pick source track…",
                     clearable=True, style={"minWidth": "22em", "flex": "1"}),
        dcc.Input(id=f"a2a-{side}-path", type="text", value="",
                  placeholder="…or free audio path (wins when non-empty)",
                  style={"minWidth": "20em", "flex": "1"}),
        html.Button("Load", id=f"a2a-{side}-load-btn"),
        html.Span(id=f"a2a-{side}-meta", style={"fontSize": "0.85em"}),
    )


def layout() -> html.Div:
    return html.Div([
        html.Div(id="a2a-server-badge", style={"margin": "0.3em 0"}),
        dcc.Interval(id="a2a-interval", interval=2000),
        dcc.Store(id="a2a-wave-store", data={}),

        html.H4("Sources"),
        _source_row("a"),
        _source_row("b"),

        html.H4("Placement"),
        _row(_lbl("A ends @ (s)"),
             dcc.Input(id="a2a-a-anchor", min=0, step=0.1, **_NUM),
             html.Div(dcc.Slider(id="a2a-a-anchor-slider", min=0, max=1, step=0.1,
                                 value=0, marks=None,
                                 tooltip={"placement": "bottom"}),
                      style={"flex": "1", "minWidth": "18em"})),
        _row(_lbl("B enters @ (s)"),
             dcc.Input(id="a2a-b-anchor", min=0, step=0.1, **_NUM),
             html.Div(dcc.Slider(id="a2a-b-anchor-slider", min=0, max=1, step=0.1,
                                 value=0, marks=None,
                                 tooltip={"placement": "bottom"}),
                      style={"flex": "1", "minWidth": "18em"})),
        _row(_lbl("segment (s)"),
             dcc.Input(id="a2a-seg-sec", value=75.0, min=20, max=180, step=1, **_NUM),
             dcc.Checklist(id="a2a-snap",
                           options=[{"label": " snap to bar grid (BPM from A)",
                                     "value": "on"}],
                           value=["on"])),
        dcc.Graph(id="a2a-overlay-graph", figure=overlay_figure({}, None, None, 75.0, [26, 49]),
                  config={"displayModeBar": False}),

        html.H4("Transition window"),
        _row(html.Div(dcc.RangeSlider(id="a2a-trans-range", min=0, max=75, step=0.5,
                                      value=[26, 49], allowCross=False, marks=None,
                                      tooltip={"placement": "bottom"}),
                      style={"flex": "1", "minWidth": "24em"}),
             dcc.Checklist(id="a2a-quantize-bars",
                           options=[{"label": " quantize whole bars", "value": "on"}],
                           value=["on"])),

        html.H4("Noising / stages"),
        _row(_lbl("mode"),
             dcc.RadioItems(id="a2a-mode",
                            options=[{"label": " sinesweep (sine noise schedule)",
                                      "value": "sinesweep"},
                                     {"label": " refine", "value": "refine"},
                                     {"label": " inpaint", "value": "inpaint"}],
                            value="sinesweep", inline=True),
             _lbl("noise level"),
             dcc.Input(id="a2a-noise", value=0.42, min=0.05, max=0.95, step=0.01, **_NUM)),
        _row(_lbl("seam-inpaint (frames)"),
             dcc.Dropdown(id="a2a-seam-size",
                          options=[{"label": str(v), "value": v}
                                   for v in (0, 128, 256, 512)],
                          value=0, clearable=False, style={"width": "6em"}),
             _lbl("seam nl (inpaint mode, 0=off)"),
             dcc.Input(id="a2a-seam-nl", value=0.35, min=0, max=0.95, step=0.01, **_NUM),
             dcc.Checklist(id="a2a-pure-basis",
                           options=[{"label": " pure-basis splice", "value": "on"}],
                           value=["on"])),
        _row(_lbl("interp"),
             dcc.Dropdown(id="a2a-interp",
                          options=[{"label": "slerp", "value": "slerp"},
                                   {"label": "lerp", "value": "lerp"}],
                          value="slerp", clearable=False,
                          style={"width": "7em"}),
             _lbl("construction"),
             dcc.Dropdown(id="a2a-construction",
                          options=[{"label": "model", "value": "model"},
                                   {"label": "latent xfade (no model pass)",
                                    "value": "latent_xfade"},
                                   {"label": "audio xfade (v2 baseline)",
                                    "value": "audio_xfade"}],
                          value="model", clearable=False,
                          style={"width": "16em"}),
             _lbl("eps seed (sinesweep clamp)"),
             dcc.Input(id="a2a-eps-seed", value=4242, step=1, **_NUM),
             _lbl("seam eps seed"),
             dcc.Input(id="a2a-seam-eps-seed", value=2424, step=1, **_NUM)),
        _row(dcc.Checklist(id="a2a-tempo-match",
                           options=[{"label": " tempo match", "value": "on"}],
                           value=["on"]),
             dcc.RadioItems(id="a2a-tempo-mode",
                            options=[{"label": " ramp", "value": "ramp"},
                                     {"label": " follow", "value": "follow"}],
                            value="ramp", inline=True),
             dcc.Checklist(id="a2a-fine-align",
                           options=[{"label": " fine align (onset concurrence)",
                                     "value": "on"}],
                           value=["on"])),

        html.H4("Harmonic steering"),
        _row(dcc.Checklist(id="a2a-chroma",
                           options=[{"label": " chroma morph (harmonic steering)",
                                     "value": "on"}],
                           value=["on"]),
             _lbl("gain"),
             dcc.Input(id="a2a-chroma-gain", value=2048.0, min=0, step=1, **_NUM),
             _lbl("guidance end pct"),
             dcc.Input(id="a2a-guid-end", value=0.6, min=0, max=1, step=0.05, **_NUM)),

        html.H4("Prompts / generation"),
        _row(_lbl("region prompt (required)"),
             dcc.Textarea(id="a2a-prompt-region", value="",
                          placeholder="TrackType: Music, VocalType: Instrumental, …",
                          style={"flex": "1", "minWidth": "28em", "height": "3.2em"})),
        _row(dcc.Checklist(id="a2a-whole-enable",
                           options=[{"label": " whole-track a2a pass", "value": "on"}],
                           value=[]),
             dcc.Textarea(id="a2a-whole-prompt", value="",
                          placeholder="whole-track prompt",
                          style={"flex": "1", "minWidth": "20em", "height": "2.4em"}),
             _lbl("nl"),
             dcc.Input(id="a2a-whole-noise", value=0.4, min=0.05, max=0.95,
                       step=0.01, **_NUM)),
        _row(_lbl("steps"), dcc.Input(id="a2a-steps", value=24, min=1, max=500, **_NUM),
             _lbl("cfg"), dcc.Input(id="a2a-cfg", value=6.0, min=0, max=25,
                                    step=0.1, **_NUM),
             _lbl("seed"), dcc.Input(id="a2a-seed", value=-1, step=1, **_NUM),
             _lbl("dist shift"),
             dcc.Dropdown(id="a2a-dist-mode", clearable=False, value="default",
                          options=[{"label": "default (ckpt)", "value": "default"},
                                   {"label": "flux", "value": "flux"}],
                          style={"width": "10em"})),

        html.H4("Prompt ARC / longform"),
        _row(_lbl("arc schedule — '0:promptA|45:promptB|…' "
                  "(steered_longform grammar; bare string = single prompt)"),
             dcc.Textarea(id="a2a-arc-schedule", value="",
                          placeholder=("0:goa trance intro|45:full-on "
                                       "psytrance|90:ambient outro"),
                          style={"flex": "1", "minWidth": "28em",
                                 "height": "3.2em"})),
        _row(_lbl("init audio path (blank = t2a longform)"),
             dcc.Input(id="a2a-arc-init-path", type="text", value="",
                       style={"minWidth": "20em", "flex": "1"}),
             _lbl("nl"),
             dcc.Input(id="a2a-arc-noise", value=0.4, min=0.05, max=0.95,
                       step=0.01, **_NUM)),
        _row(_lbl("duration (s, t2a)"),
             dcc.Input(id="a2a-arc-duration", value=120.0, min=10, max=600,
                       step=1, **_NUM),
             _lbl("window (s)"),
             dcc.Input(id="a2a-arc-window", value=30.0, min=10, max=120,
                       step=1, **_NUM),
             _lbl("overlap (s)"),
             dcc.Input(id="a2a-arc-overlap", value=5.0, min=1, max=30,
                       step=0.5, **_NUM),
             _lbl("xfade (s, t2a prompt joins)"),
             dcc.Input(id="a2a-arc-xfade", value=4.0, min=0, max=20,
                       step=0.5, **_NUM),
             html.Button("Longform render", id="a2a-longform-btn",
                         style={"fontWeight": "bold"})),
        html.Pre(id="a2a-longform-status",
                 style={"whiteSpace": "pre-wrap", "fontSize": "0.8em",
                        "background": "#f6f6f6", "padding": "0.4em"}),
        html.Div(id="a2a-longform-result"),

        controls.steering_panel("a2a", dora_default="evr1x"),

        _row(html.Button("Render transition", id="a2a-render-btn",
                         style={"fontWeight": "bold", "padding": "0.4em 1.2em"})),
        html.Pre(id="a2a-status",
                 style={"whiteSpace": "pre-wrap", "fontSize": "0.8em",
                        "background": "#f6f6f6", "padding": "0.4em"}),
        html.Div(id="a2a-result"),
        html.H4("History"),
        dcc.Store(id="a2a-history", data=[]),
        html.Div(id="a2a-history-div"),
    ], style={"padding": "0.5em 1em"})


# ---------------------------------------------------------------------------
# overlay figure (pure function)
# ---------------------------------------------------------------------------

def _env_slice(side: dict, t0: float, t1: float):
    """Envelope slice of a side over source-time [t0, t1], with composite-time
    x offset for the part clamped away at the start."""
    env = np.asarray(side.get("env") or [], dtype=float)
    hz = float(side.get("env_hz") or ENV_HZ_TARGET)
    i0, i1 = int(max(0.0, t0) * hz), int(max(0.0, t1) * hz)
    i1 = min(i1, len(env))
    if i1 <= i0:
        return np.zeros(0), np.zeros(0)
    y = env[i0:i1]
    x = np.arange(len(y)) / hz + (max(0.0, t0) - t0)
    return x, y


def overlay_figure(store, a_end, b_start, seg, trans_range) -> go.Figure:
    store = store or {}
    seg = float(seg or 75.0)
    trans_range = trans_range or [26, 49]
    fig = go.Figure()
    a, b = store.get("a"), store.get("b")
    if a and a_end is not None:
        x, y = _env_slice(a, float(a_end) - seg, float(a_end))
        fig.add_trace(go.Scatter(x=x, y=y, fill="tozeroy", mode="lines",
                                 line={"width": 0.7, "color": "#1f77b4"}, name="A"))
    if b and b_start is not None:
        x, y = _env_slice(b, float(b_start), float(b_start) + seg)
        fig.add_trace(go.Scatter(x=x, y=-y, fill="tozeroy", mode="lines",
                                 line={"width": 0.7, "color": "#d62728"}, name="B"))
    t0, t1 = float(trans_range[0]), float(trans_range[1])
    fig.add_vrect(x0=t0, x1=t1, opacity=0.15, fillcolor="#888", line_width=0)
    for tx in (t0, t1):
        fig.add_vline(x=tx, line={"width": 1, "color": "#888", "dash": "dot"})
    title = "load both tracks"
    if a and b and a_end is not None and b_start is not None:
        title = (f"A ends @ {float(a_end):.1f}s / B enters @ {float(b_start):.1f}s"
                 f" — window {t0:.1f}–{t1:.1f}s of {seg:.0f}s composite")
    fig.update_layout(title={"text": title, "font": {"size": 13}},
                      xaxis={"range": [0, seg], "title": "composite time (s)"},
                      yaxis={"showticklabels": False, "zeroline": True},
                      height=280, margin={"l": 30, "r": 10, "t": 40, "b": 30},
                      showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# audio load helper
# ---------------------------------------------------------------------------

def _load_side(path: str, default_frac: float) -> dict:
    """Read audio, build 40 Hz block-max envelope + BPM (librosa, 60 s excerpt
    around the default anchor). Raises on unreadable files (soundfile only —
    the server decodes anything via ffmpeg; the GUI just needs the waveform)."""
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = np.abs(data).mean(axis=1)
    dur = len(mono) / sr
    hop = max(1, int(round(sr / ENV_HZ_TARGET)))
    n = (len(mono) // hop) * hop
    env = mono[:n].reshape(-1, hop).max(axis=1) if n else np.zeros(0, np.float32)
    if len(mono) > n:
        env = np.append(env, mono[n:].max())
    anchor = default_frac * dur
    bpm = None
    try:
        import librosa
        s0 = int(max(0.0, anchor - 30.0) * sr)
        y = data[s0:s0 + int(60 * sr)].mean(axis=1)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        t = float(np.atleast_1d(tempo)[0])
        bpm = round(t, 2) if t > 0 else None
    except Exception:
        pass
    return {"path": str(path), "dur": round(dur, 3), "bpm": bpm,
            "env": [round(float(v), 4) for v in env],
            "env_hz": sr / hop}


def _meta_text(side: dict) -> str:
    bpm = f"{side['bpm']:.1f} BPM" if side.get("bpm") else "BPM ?"
    return f"dur {side['dur']:.1f}s · {bpm}"


def _players(resp: dict) -> list:
    out = []
    for url, fp in zip(resp.get("urls", []), resp.get("files", [])):
        out.append(html.Div([
            html.Audio(src=render_client.audio_url(url), controls=True,
                       style={"width": "60%", "verticalAlign": "middle"}),
            html.Code(fp, style={"fontSize": "0.75em", "marginLeft": "0.6em"}),
        ], style={"margin": "0.2em 0"}))
    return out


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------

def register_callbacks(app, index, latent_dir) -> None:  # noqa: ARG001 (latent_dir kept for parity)
    controls.register(app, "a2a")

    # unique source tracks, label "artist — title", value abs source path
    seen: dict[str, str] = {}
    for c in index:
        if c.source_track and c.source_track not in seen:
            lbl = f"{c.artist} — {c.title}".strip(" —") or c.source_track
            seen[c.source_track] = lbl
    crop_opts = [{"label": lbl, "value": path}
                 for path, lbl in sorted(seen.items(), key=lambda kv: kv[1].lower())]

    # 1. crop-dd fill x2 (self-trigger idiom)
    @app.callback(Output("a2a-a-crop-dd", "options"), Input("a2a-a-crop-dd", "id"))
    def _fill_a(_):
        return crop_opts

    @app.callback(Output("a2a-b-crop-dd", "options"), Input("a2a-b-crop-dd", "id"))
    def _fill_b(_):
        return crop_opts

    # 2. load (single callback, triggered_id picks side)
    @app.callback(
        Output("a2a-wave-store", "data"),
        Output("a2a-a-meta", "children"), Output("a2a-b-meta", "children"),
        Output("a2a-a-anchor", "value", allow_duplicate=True),
        Output("a2a-a-anchor", "max"),
        Output("a2a-a-anchor-slider", "value", allow_duplicate=True),
        Output("a2a-a-anchor-slider", "max"),
        Output("a2a-b-anchor", "value", allow_duplicate=True),
        Output("a2a-b-anchor", "max"),
        Output("a2a-b-anchor-slider", "value", allow_duplicate=True),
        Output("a2a-b-anchor-slider", "max"),
        Input("a2a-a-load-btn", "n_clicks"), Input("a2a-b-load-btn", "n_clicks"),
        State("a2a-a-crop-dd", "value"), State("a2a-a-path", "value"),
        State("a2a-b-crop-dd", "value"), State("a2a-b-path", "value"),
        State("a2a-seg-sec", "value"), State("a2a-wave-store", "data"),
        prevent_initial_call=True)
    def _load(_na, _nb, a_dd, a_path, b_dd, b_path, seg, store):
        side = "a" if ctx.triggered_id == "a2a-a-load-btn" else "b"
        path = ((a_path if side == "a" else b_path) or "").strip() \
            or (a_dd if side == "a" else b_dd)
        nups = [no_update] * 8  # meta_a, meta_b, then 2x(val,max) per side
        if not path:
            nups[0 if side == "a" else 1] = "pick a track or enter a path"
            return (no_update, *nups)
        try:
            info = _load_side(path, 0.62 if side == "a" else 0.40)
        except Exception as e:  # unreadable by soundfile -> error in meta only
            nups[0 if side == "a" else 1] = f"load failed: {e}"
            return (no_update, *nups)
        store = dict(store or {})
        store[side] = info
        dur = info["dur"]
        if side == "a":
            anchor = round(0.62 * dur, 2)
            return (store, _meta_text(info), no_update,
                    anchor, dur, anchor, dur,
                    no_update, no_update, no_update, no_update)
        amax = max(0.0, dur - float(seg or 75.0))
        anchor = round(min(0.40 * dur, amax), 2)
        return (store, no_update, _meta_text(info),
                no_update, no_update, no_update, no_update,
                anchor, amax, anchor, amax)

    # 3. anchor sync x2 (numeric <-> slider, optional bar snap with BPM from A)
    def _make_sync(side):
        @app.callback(
            Output(f"a2a-{side}-anchor", "value"),
            Output(f"a2a-{side}-anchor-slider", "value"),
            Input(f"a2a-{side}-anchor", "value"),
            Input(f"a2a-{side}-anchor-slider", "value"),
            State("a2a-snap", "value"), State("a2a-wave-store", "data"),
            prevent_initial_call=True)
        def _sync(num, sld, snap, store):
            val = num if ctx.triggered_id == f"a2a-{side}-anchor" else sld
            if val is None:
                return no_update, no_update
            val = float(val)
            bpm_a = ((store or {}).get("a") or {}).get("bpm")
            if snap and bpm_a:
                bar = 240.0 / float(bpm_a)
                val = round(val / bar) * bar
            val = round(val, 3)
            return val, val
        return _sync

    _make_sync("a")
    _make_sync("b")

    # 4. range-slider rescale on segment change
    @app.callback(Output("a2a-trans-range", "max"),
                  Output("a2a-trans-range", "value"),
                  Input("a2a-seg-sec", "value"),
                  State("a2a-trans-range", "value"))
    def _rescale(seg, cur):
        if not seg:
            return no_update, no_update
        seg = float(seg)
        cur = cur or [26.0, 49.0]
        lo = min(max(0.0, float(cur[0])), seg)
        hi = min(max(lo, float(cur[1])), seg)
        return seg, [lo, hi]

    # 5. overlay figure (no render — pure GUI update)
    @app.callback(Output("a2a-overlay-graph", "figure"),
                  Input("a2a-wave-store", "data"),
                  Input("a2a-a-anchor", "value"), Input("a2a-b-anchor", "value"),
                  Input("a2a-seg-sec", "value"), Input("a2a-trans-range", "value"))
    def _figure(store, a_end, b_start, seg, trans):
        return overlay_figure(store, a_end, b_start, seg, trans)

    # 6. mode gating
    @app.callback(Output("a2a-seam-size", "disabled"),
                  Output("a2a-seam-nl", "disabled"),
                  Output("a2a-pure-basis", "options"),
                  Input("a2a-mode", "value"))
    def _gate(mode):
        return (mode != "sinesweep", mode != "inpaint",
                [{"label": " pure-basis splice", "value": "on",
                  "disabled": mode != "sinesweep"}])

    # 7. render
    @app.callback(
        Output("a2a-result", "children"),
        Output("a2a-history", "data"),
        Output("a2a-status", "children"),
        Input("a2a-render-btn", "n_clicks"),
        State("a2a-wave-store", "data"),
        State("a2a-a-anchor", "value"), State("a2a-b-anchor", "value"),
        State("a2a-seg-sec", "value"), State("a2a-snap", "value"),
        State("a2a-trans-range", "value"), State("a2a-quantize-bars", "value"),
        State("a2a-mode", "value"), State("a2a-noise", "value"),
        State("a2a-seam-size", "value"), State("a2a-seam-nl", "value"),
        State("a2a-pure-basis", "value"),
        State("a2a-interp", "value"), State("a2a-construction", "value"),
        State("a2a-eps-seed", "value"), State("a2a-seam-eps-seed", "value"),
        State("a2a-tempo-match", "value"), State("a2a-tempo-mode", "value"),
        State("a2a-fine-align", "value"),
        State("a2a-chroma", "value"), State("a2a-chroma-gain", "value"),
        State("a2a-guid-end", "value"),
        State("a2a-prompt-region", "value"),
        State("a2a-whole-enable", "value"), State("a2a-whole-prompt", "value"),
        State("a2a-whole-noise", "value"),
        State("a2a-steps", "value"), State("a2a-cfg", "value"),
        State("a2a-seed", "value"), State("a2a-dist-mode", "value"),
        State("a2a-history", "data"),
        *controls.steering_states("a2a"),
        prevent_initial_call=True)
    def _render(_n, store, a_end, b_start, seg, snap, trans, quant,
                mode, noise, seam_size, seam_nl, pure_basis,
                interp, construction, eps_seed, seam_eps_seed,
                tempo_match, tempo_mode, fine_align,
                chroma, chroma_gain, guid_end,
                prompt_region, whole_en, whole_prompt, whole_noise,
                steps, cfg, seed, dist_mode, history, *steer):
        store = store or {}
        a, b = store.get("a"), store.get("b")
        if not (a and b):
            return no_update, no_update, "ERROR: load both A and B first"
        if not (prompt_region or "").strip():
            return no_update, no_update, "ERROR: region prompt is required"
        trans = trans or [26.0, 49.0]
        payload = {
            "a_path": a["path"], "b_path": b["path"],
            "a_end_sec": a_end, "b_start_sec": b_start,
            "seg_sec": float(seg or 75.0),
            "snap_to_downbeat": bool(snap),
            "tempo_match": bool(tempo_match),
            "tempo_mode": tempo_mode or "ramp",
            "fine_align": bool(fine_align),
            "trans_start_sec": float(trans[0]), "trans_end_sec": float(trans[1]),
            "quantize_bars": bool(quant),
            "mode": mode or "sinesweep",
            "noise_level": float(noise if noise is not None else 0.42),
            "seam_inpaint": int(seam_size or 0),
            "seam_nl": float(seam_nl if seam_nl is not None else 0.35),
            "pure_basis": bool(pure_basis),
            # crossfade-lab knobs (parity-audit item 6 — server defaults
            # slerp/model/4242/2424 reproduce the pre-lab behavior)
            "interp": interp or "slerp",
            "construction": construction or "model",
            "eps_seed": int(eps_seed if eps_seed is not None else 4242),
            "seam_eps_seed": int(seam_eps_seed
                                 if seam_eps_seed is not None else 2424),
            "chroma_morph": bool(chroma),
            "chroma_gain": float(chroma_gain or 2048.0),
            "guidance_end_pct": float(guid_end if guid_end is not None else 0.6),
            "prompt_region": prompt_region.strip(),
            "whole_track": ({"noise_level": float(whole_noise or 0.4),
                             "prompt": (whole_prompt or "").strip()}
                            if whole_en else None),
            "steps": int(steps or 24),
            "cfg_scale": float(cfg if cfg is not None else 6.0),
            "seed": int(seed if seed is not None else -1),
        }
        if dist_mode == "flux":
            # "default"/absent = ckpt sampling_dist_shift (resolve_dist_shift)
            payload["dist_shift"] = "flux"
        payload.update(controls.steering_payload(steer))
        try:
            resp = render_client.render("a2a_mix", payload)
        except render_client.RenderError as e:
            return no_update, no_update, f"RENDER ERROR:\n{e.args[0]}"
        meta = resp.get("meta", {})
        status = json.dumps({"seed": resp.get("seed"),
                             "timings": resp.get("timings"),
                             "warnings": resp.get("warnings"),
                             "meta": {k: v for k, v in meta.items()
                                      if k != "params_echo"}},
                            indent=2, default=str)
        label = (f"{Path(a['path']).stem} → {Path(b['path']).stem} "
                 f"[{payload['mode']} nl={payload['noise_level']}]")
        entry = {"ts": datetime.datetime.now().strftime("%H:%M:%S"),
                 "label": label, "urls": resp.get("urls", []),
                 "files": resp.get("files", []), "meta": meta,
                 "params": payload}
        history = ([entry] + (history or []))[:HISTORY_CAP]
        return _players(resp), history, status

    # 7b. longform render (prompt-ARC via POST /longform)
    @app.callback(
        Output("a2a-longform-result", "children"),
        Output("a2a-history", "data", allow_duplicate=True),
        Output("a2a-longform-status", "children"),
        Input("a2a-longform-btn", "n_clicks"),
        State("a2a-arc-schedule", "value"),
        State("a2a-arc-init-path", "value"),
        State("a2a-arc-noise", "value"),
        State("a2a-arc-duration", "value"),
        State("a2a-arc-window", "value"),
        State("a2a-arc-overlap", "value"),
        State("a2a-arc-xfade", "value"),
        State("a2a-steps", "value"), State("a2a-cfg", "value"),
        State("a2a-seed", "value"), State("a2a-dist-mode", "value"),
        State("a2a-history", "data"),
        *controls.steering_states("a2a"),
        prevent_initial_call=True)
    def _longform(_n, sched, init_path, noise, duration, window, overlap,
                  xfade, steps, cfg, seed, dist_mode, history, *steer):
        sched = (sched or "").strip()
        if not sched:
            return no_update, no_update, \
                "ERROR: arc schedule is required ('0:promptA|45:promptB|…')"
        payload = {
            "schedule": sched,
            "steps": int(steps or 24),
            "cfg_scale": float(cfg if cfg is not None else 6.0),
            "seed": int(seed if seed is not None else -1),
            "window_sec": float(window or 30.0),
            "overlap_sec": float(overlap or 5.0),
            "xfade_sec": float(xfade if xfade is not None else 4.0),
        }
        init_path = (init_path or "").strip()
        if init_path:                    # a2a arc: windowed loop over the source
            payload["audio_path"] = init_path
            payload["noise_level"] = float(noise if noise is not None else 0.4)
        else:                            # t2a arc: LongFormRenderer path
            payload["duration"] = float(duration or 120.0)
        if dist_mode == "flux":
            payload["dist_shift"] = "flux"
        payload.update(controls.steering_payload(steer))
        try:
            resp = render_client.longform(payload)
        except render_client.RenderError as e:
            return no_update, no_update, f"LONGFORM ERROR:\n{e.args[0]}"
        meta = resp.get("meta", {})
        status = json.dumps({"seed": resp.get("seed"),
                             "timings": resp.get("timings"),
                             "warnings": resp.get("warnings"),
                             "meta": {k: v for k, v in meta.items()
                                      if k != "params_echo"}},
                            indent=2, default=str)
        label = (f"longform[{meta.get('mode', '?')}] {sched[:50]}")
        entry = {"ts": datetime.datetime.now().strftime("%H:%M:%S"),
                 "label": label, "urls": resp.get("urls", []),
                 "files": resp.get("files", []), "meta": meta,
                 "params": payload}
        history = ([entry] + (history or []))[:HISTORY_CAP]
        return _players(resp), history, status

    # 8. history render
    @app.callback(Output("a2a-history-div", "children"),
                  Input("a2a-history", "data"))
    def _history(data):
        out = []
        for e in (data or []):
            out.append(html.Div(
                [html.B(f"{e.get('ts', '')} — {e.get('label', '')}")]
                + _players(e),
                style={"borderTop": "1px solid #ddd", "padding": "0.3em 0"}))
        return out

    # 9. server badge poll
    @app.callback(Output("a2a-server-badge", "children"),
                  Input("a2a-interval", "n_intervals"))
    def _badge(_):
        st = render_client.status()
        if st is None:
            return html.Span(["render server DOWN — launch: ",
                              html.Code(render_client.LAUNCH_HINT,
                                        style={"fontSize": "0.75em"})],
                             style={"color": "#b00"})
        state = "BUSY" if st.get("busy") else "idle"
        tail = "\n".join((st.get("log_tail") or [])[-5:])
        return html.Span([html.B(f"render server {state}"),
                          html.Pre(tail, style={"fontSize": "0.7em",
                                                "margin": "0.1em 0",
                                                "whiteSpace": "pre-wrap"})],
                         style={"color": "#082" if state == "idle" else "#c60"})
