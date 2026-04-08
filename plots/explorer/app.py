"""
Unified MIR Explorer — Plotly Dash app (port 7895).

Run:
    python plots/explorer/app.py [--port 7895] [--debug]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, no_update

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.explorer.data import get_app_data, get_config, get_analysis
from plots.explorer.audio import build_decode_url, build_crossfade_url
from plots.explorer.tabs import dataset, analysis, viewer

app = dash.Dash(
    __name__,
    title="MIR Explorer",
    suppress_callback_exceptions=True,
)

_TAB_STYLE        = {"padding": "8px 16px", "color": "#888"}
_TAB_ACTIVE_STYLE = {"padding": "8px 16px", "backgroundColor": "#e94560",
                     "color": "#fff", "fontWeight": "700"}

app.layout = html.Div([
    # ── Shared state stores ────────────────────────────────────────────────
    dcc.Store(id="active-track",
              data={"track": None, "track_idx": None, "slot": "a"}),
    dcc.Store(id="player-cmd",   data={}),
    dcc.Store(id="cluster-highlight", data=None),
    dcc.Store(id="viewer-state",
              data={"track_a": None, "crop_a": None,
                    "track_b": None, "crop_b": None}),
    dcc.Store(id="avg-crops",       data=False),
    dcc.Store(id="autoplay-hover",  data=False),
    dcc.Store(id="radar-selection", data=[]),
    dcc.Store(id="click-register",  data=None),

    # ── Header ─────────────────────────────────────────────────────────────
    html.Div(
        html.H2("MIR Explorer", style={"margin": "8px 16px 4px", "color": "#e94560"}),
        style={"background": "#111125", "borderBottom": "1px solid #2a2a4a"}
    ),

    # ── Tab navigation + content ───────────────────────────────────────────
    dcc.Tabs(id="main-tabs", value="dataset", children=[
        dcc.Tab(label="Dataset",  value="dataset",
                style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
        dcc.Tab(label="Analysis", value="analysis",
                style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
        dcc.Tab(label="Viewer",   value="viewer",
                style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
    ]),
    html.Div(id="tab-content", className="tab-content-wrapper"),

    # ── Persistent player strip (footer) ───────────────────────────────────
    html.Div(id="player-strip", children=[
        html.Span("♫ Latent Player", style={"color": "#555", "fontSize": "11px",
                                             "marginRight": "16px"}),
        # A slot
        html.Div([
            html.Span("A:", style={"color": "#4cd137", "fontSize": "11px"}),
            html.Span("—", id="player-track-a",
                      style={"fontSize": "11px", "maxWidth": "220px",
                             "overflow": "hidden", "textOverflow": "ellipsis",
                             "whiteSpace": "nowrap", "display": "inline-block",
                             "verticalAlign": "middle"}),
            html.Button("▶", id="btn-play-a",  n_clicks=0,
                        style={"background": "none", "border": "none",
                               "color": "#4cd137", "cursor": "pointer",
                               "fontSize": "14px"}),
            html.Button("■", id="btn-stop-a",  n_clicks=0,
                        style={"background": "none", "border": "none",
                               "color": "#e94560", "cursor": "pointer",
                               "fontSize": "14px"}),
            html.Span("Pos", style={"fontSize": "10px", "color": "#555"}),
            dcc.Slider(id="pos-slider-a", min=0, max=1, step=0.001, value=0.5,
                       marks={}, tooltip={"always_visible": False},
                       className="pos-slider", updatemode="mouseup"),
        ], className="player-slot", style={"display": "inline-flex",
                                           "alignItems": "center", "gap": "8px"}),
        # B slot + crossfade
        html.Div([
            html.Span("B:", style={"color": "#00d2ff", "fontSize": "11px"}),
            html.Span("—", id="player-track-b",
                      style={"fontSize": "11px", "maxWidth": "220px",
                             "overflow": "hidden", "textOverflow": "ellipsis",
                             "whiteSpace": "nowrap", "display": "inline-block",
                             "verticalAlign": "middle"}),
            html.Span("Mix", style={"fontSize": "10px", "color": "#555"}),
            dcc.Slider(id="xfade-alpha", min=0, max=1, step=0.01, value=0.0,
                       marks={0: "A", 1: "B"}, tooltip={"always_visible": False},
                       className="pos-slider", updatemode="drag"),
        ], id="player-b-slot",
           style={"display": "inline-flex", "alignItems": "center",
                  "gap": "8px", "marginLeft": "24px"}),
        # Audio sink (hidden, needed for clientside callback)
        html.Div(id="player-audio-sink", style={"display": "none"}),
        # Checkboxes
        dcc.Checklist(
            id="player-options",
            options=[
                {"label": "Autoplay hover",     "value": "autoplay"},
                {"label": "Smart Loop",         "value": "smart_loop"},
                {"label": "Continue auto",      "value": "continue_auto"},
                {"label": "VAE fade",           "value": "vae_fade"},
            ],
            value=[],
            inline=True,
            style={"display": "inline-flex", "gap": "12px",
                   "marginLeft": "24px", "fontSize": "11px"},
        ),
    ]),
], style={"fontFamily": "monospace", "backgroundColor": "#0d0d1a",
          "color": "#ccc", "minHeight": "100vh"})


# ── Tab routing ───────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab: str):
    if tab == "dataset":  return dataset.layout()
    if tab == "analysis": return analysis.layout()
    if tab == "viewer":   return viewer.layout()
    return html.P("Unknown tab.")


# ── Player strip label sync from active-track store ──────────────────────────
@app.callback(
    Output("player-track-a", "children"),
    Output("player-track-b", "children"),
    Input("active-track", "data"),
)
def sync_player_labels(state: dict):
    t = state or {}
    track   = t.get("track") or "—"
    track_b = t.get("track_b") or "—"
    return track, track_b


# ── Sync autoplay-hover checkbox into Store ───────────────────────────────────
@app.callback(
    Output("autoplay-hover", "data"),
    Input("player-options", "value"),
)
def sync_autoplay(opts):
    return "autoplay" in (opts or [])


# ── Clientside callback: player-cmd Store → Web Audio ────────────────────────
app.clientside_callback(
    """
    function(cmd) {
        if (window.dash_clientside && window.dash_clientside.player) {
            window.dash_clientside.player.handle_cmd(cmd);
        }
        return '';
    }
    """,
    Output("player-audio-sink", "children"),
    Input("player-cmd", "data"),
    prevent_initial_call=True,
)


# ── Clientside: graph click → slot A (single) or slot B (double) ─────────────
# Two rapid clicks on the same point within 400 ms → slot B; otherwise slot A.
app.clientside_callback(
    """
    function(clickData) {
        if (!clickData || !clickData.points || !clickData.points.length)
            return window.dash_clientside.no_update;
        var pt   = clickData.points[0];
        var raw  = pt.customdata;
        if (raw === null || raw === undefined)
            return window.dash_clientside.no_update;
        // Coerce BigInt (from Int64Array binary encoding) and single-element
        // arrays ([42] instead of 42) to a plain JS number.
        if (Array.isArray(raw)) raw = raw[0];
        var tidx = Number(raw);
        if (isNaN(tidx))
            return window.dash_clientside.no_update;

        var now  = Date.now();
        var last = window._mirLastClick || {ts: 0, idx: null};
        var slot;
        if (now - last.ts < 400 && last.idx === tidx) {
            slot = 'b';
            window._mirLastClick = {ts: 0, idx: null};  // reset so next click is A
        } else {
            slot = 'a';
            window._mirLastClick = {ts: now, idx: tidx};
        }
        return {slot: slot, track_idx: tidx};
    }
    """,
    Output("click-register", "data"),
    Input("dataset-graph", "clickData"),
    prevent_initial_call=True,
)


# ── Play / Stop button callbacks ──────────────────────────────────────────────
@app.callback(
    Output("player-cmd", "data"),
    Input("btn-play-a",  "n_clicks"),
    Input("btn-stop-a",  "n_clicks"),
    State("active-track", "data"),
    State("pos-slider-a", "value"),
    State("xfade-alpha",  "value"),
    State("player-options", "value"),
    prevent_initial_call=True,
)
def player_play_stop(play_clicks, stop_clicks, state, pos, xfade, opts):
    ctx_cb = dash.callback_context
    if not ctx_cb.triggered:
        return no_update
    trig = ctx_cb.triggered[0]["prop_id"]
    if "stop" in trig:
        return {"action": "stop"}
    st            = state or {}
    track_a       = st.get("track")
    track_b       = st.get("track_b")
    opts          = opts or []
    smart_loop    = "smart_loop"    in opts
    continue_auto = "continue_auto" in opts
    vae_fade      = "vae_fade"      in opts
    xfade         = float(xfade or 0.0)

    if not track_a:
        return no_update

    ls, le = (0, -1) if smart_loop else (None, None)

    if vae_fade and track_b and xfade == 0.0:
        # Play a latent-space blend clip (A→B at mix 0.5), then loop B
        blend_url = build_crossfade_url(track_a, str(pos), track_b, str(pos),
                                        mix=0.5, smart_loop=smart_loop)
        dest_url  = build_decode_url(track_b, str(pos), smart_loop=smart_loop)
        return {"action": "play_with_fade",
                "blend_url":    blend_url,   "dest_url":     dest_url,
                "loop_start":   ls,          "loop_end":     le,
                "continue_auto": continue_auto, "vae_fade":  vae_fade}

    if track_b and xfade > 0:
        url = build_crossfade_url(track_a, str(pos), track_b, str(pos),
                                  mix=xfade, smart_loop=smart_loop)
    else:
        url = build_decode_url(track_a, str(pos), smart_loop=smart_loop)
    return {"action": "play", "url": url, "loop_start": ls, "loop_end": le,
            "continue_auto": continue_auto, "vae_fade": vae_fade}


# ── Prefetch on position-slider release ───────────────────────────────────────
@app.callback(
    Output("player-cmd", "data", allow_duplicate=True),
    Input("pos-slider-a", "value"),
    State("active-track", "data"),
    State("player-options", "value"),
    prevent_initial_call=True,
)
def prefetch_on_slider(pos, state, opts):
    """When the position slider is released, start decoding the new position
    in the background so Play is instant when the user clicks it."""
    st      = state or {}
    track_a = st.get("track")
    if not track_a or pos is None:
        return no_update
    smart_loop = "smart_loop" in (opts or [])
    url = build_decode_url(track_a, str(pos), smart_loop=smart_loop)
    return {"action": "prefetch", "url": url}


# ── Sync continue_auto / vae_fade flags immediately when checkbox changes ─────
@app.callback(
    Output("player-cmd", "data", allow_duplicate=True),
    Input("player-options", "value"),
    prevent_initial_call=True,
)
def sync_player_options(opts):
    opts = opts or []
    return {
        "action":        "options",
        "continue_auto": "continue_auto" in opts,
        "vae_fade":      "vae_fade"      in opts,
    }


# ── Fade to new track when continue_auto + vae_fade are both on ───────────────
@app.callback(
    Output("player-cmd", "data", allow_duplicate=True),
    Input("active-track", "data"),
    State("player-options", "value"),
    State("pos-slider-a", "value"),
    prevent_initial_call=True,
)
def fade_on_track_change(active_track, opts, pos):
    """When a new track is loaded while continue_auto is on, send fade_to.
    JS will crossfade immediately (if vaeFade+posA known) or queue for next crop boundary."""
    opts = opts or []
    if "continue_auto" not in opts:
        return no_update
    track = (active_track or {}).get("track")
    if not track:
        return no_update
    smart_loop = "smart_loop" in opts
    url = build_decode_url(track, str(pos or 0.5), smart_loop=smart_loop)
    ls, le = (0, -1) if smart_loop else (None, None)
    return {"action": "fade_to", "url": url, "loop_start": ls, "loop_end": le}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int, default=7895)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Eagerly load data so first tab render is fast
    get_app_data()
    get_config()
    get_analysis()

    print(f"MIR Explorer → http://localhost:{args.port}")
    app.run(debug=args.debug, port=args.port, host="127.0.0.1",
            extra_files=[], use_reloader=False)
