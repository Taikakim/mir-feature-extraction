"""Tab 3 — Latent Viewer: 3D trajectory, alignment bar, crossfader."""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback, no_update

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.explorer.data import (
    get_config, scan_latent_dir, load_latent, project_latent_pca,
    avg_crops_with_loop_gating, get_analysis,
)
from plots.explorer.audio import build_decode_url, build_crossfade_url, build_average_url

_DARK3D = dict(
    template="plotly_dark",
    paper_bgcolor="#0d0d1a",
    scene=dict(bgcolor="#111125",
               xaxis=dict(showticklabels=False, title="PC1"),
               yaxis=dict(showticklabels=False, title="PC2"),
               zaxis=dict(showticklabels=False, title="PC3")),
    margin=dict(l=0, r=0, t=30, b=0),
)


def layout() -> html.Div:
    cfg = get_config()
    try:
        tracks = sorted(scan_latent_dir(cfg["latent_dir"]).keys())
    except Exception:
        tracks = []
    t_opts  = [{"label": t, "value": t} for t in tracks]
    default = tracks[0] if tracks else None

    return html.Div([
        # ── Track/crop selectors ─────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Label("Track A", style={"fontSize": "10px", "color": "#4cd137"}),
                dcc.Dropdown(id="v-track-a", options=t_opts, value=default,
                             clearable=False, style={"width": "280px"},
                             placeholder="Search…"),
                dcc.Dropdown(id="v-crop-a", options=[], value=None,
                             clearable=False, style={"width": "200px"},
                             placeholder="Crop…"),
            ], style={"display": "flex", "gap": "8px", "alignItems": "flex-end"}),
            html.Div([
                html.Label("Track B", style={"fontSize": "10px", "color": "#00d2ff"}),
                dcc.Dropdown(id="v-track-b", options=t_opts, value=None,
                             clearable=True, style={"width": "280px"},
                             placeholder="Search…"),
                dcc.Dropdown(id="v-crop-b", options=[], value=None,
                             clearable=True, style={"width": "200px"},
                             placeholder="Crop…"),
            ], style={"display": "flex", "gap": "8px", "alignItems": "flex-end"}),
            dcc.Checklist(id="v-avg-crops",
                          options=[{"label": "Avg all crops (loop-gated)", "value": "avg"}],
                          value=[], inline=True,
                          style={"fontSize": "11px", "marginLeft": "16px"}),
            dcc.RadioItems(id="v-viz-mode",
                           options=[{"label": "Trajectories",   "value": "traj"},
                                    {"label": "Hybrid 3D",      "value": "hybrid"},
                                    {"label": "Difference",     "value": "diff"},
                                    {"label": "Moving Average", "value": "mavg"}],
                           value="traj", inline=True,
                           style={"fontSize": "11px", "marginLeft": "16px"}),
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px",
                  "padding": "8px 10px", "background": "#111125",
                  "borderBottom": "1px solid #2a2a4a", "alignItems": "flex-end"}),

        # ── 3D plot + controls ───────────────────────────────────────────────
        html.Div([
            html.Div([
                dcc.Graph(id="v-graph-3d",
                          style={"height": "55vh"},
                          config={"displayModeBar": True}),
                dcc.Graph(id="v-alignment-bar",
                          style={"height": "80px"},
                          config={"displayModeBar": False}),
            ], style={"flex": "1", "minWidth": 0}),

            # Right panel: crossfader + manipulation
            html.Div([
                dcc.RadioItems(id="v-xf-mode",
                               options=[{"label": "Simple",   "value": "simple"},
                                        {"label": "Advanced", "value": "advanced"}],
                               value="simple", inline=True,
                               style={"fontSize": "11px", "marginBottom": "8px"}),
                html.Div(id="v-crossfader-panel"),
                html.Hr(style={"borderColor": "#2a2a4a", "margin": "8px 0"}),
                html.Div([
                    html.Label("Latent Manipulation",
                               style={"fontSize": "11px", "fontWeight": "bold",
                                      "color": "#888", "marginBottom": "4px"}),
                    html.Div(id="v-manip-panel", children=_manip_panel()),
                ]),
            ], style={"width": "280px", "flexShrink": "0",
                      "padding": "8px", "borderLeft": "1px solid #2a2a4a",
                      "overflowY": "auto", "maxHeight": "calc(55vh + 80px)"}),
        ], style={"display": "flex", "gap": "0"}),
    ])


def _manip_panel() -> list:
    features = ["brightness", "rms_energy_bass", "danceability",
                "hardness", "female_probability"]
    labels   = ["Brightness", "Bass Energy", "Danceability", "Hardness", "Female Voice"]
    return [
        html.Div([
            html.Label(lbl, style={"fontSize": "10px", "color": "#888",
                                    "width": "100px", "display": "inline-block"}),
            dcc.Slider(id={"type": "manip-slider", "index": feat},
                       min=-2, max=2, step=0.1, value=0.0,
                       marks={-2: "-2", 0: "0", 2: "+2"},
                       tooltip={"always_visible": False},
                       updatemode="drag"),
        ], style={"display": "flex", "alignItems": "center", "gap": "4px",
                  "marginBottom": "4px"})
        for feat, lbl in zip(features, labels)
    ]


def _ctrl(label: str, child) -> html.Div:
    return html.Div([
        html.Label(label, style={"fontSize": "10px", "color": "#888"}),
        child,
    ], style={"marginBottom": "6px"})


# ── Populate crop dropdowns ───────────────────────────────────────────────────
@callback(Output("v-crop-a", "options"), Output("v-crop-a", "value"),
          Input("v-track-a", "value"))
def update_crops_a(track):
    if not track:
        return [], None
    cfg   = get_config()
    crops = scan_latent_dir(cfg["latent_dir"]).get(track, [])
    opts  = [{"label": c, "value": c} for c in crops]
    return opts, crops[0] if crops else None


@callback(Output("v-crop-b", "options"), Output("v-crop-b", "value"),
          Input("v-track-b", "value"))
def update_crops_b(track):
    if not track:
        return [], None
    cfg   = get_config()
    crops = scan_latent_dir(cfg["latent_dir"]).get(track, [])
    opts  = [{"label": c, "value": c} for c in crops]
    return opts, crops[0] if crops else None


# ── Sync viewer from active-track Store on tab switch ─────────────────────────
@callback(
    Output("v-track-a", "value"),
    Output("v-track-b", "value"),
    Input("main-tabs", "value"),
    State("active-track", "data"),
    State("v-track-a", "value"),
    State("v-track-b", "value"),
    prevent_initial_call=True,
)
def sync_viewer_from_store(tab, state, cur_a, cur_b):
    if tab != "viewer" or not state:
        return no_update, no_update
    track   = state.get("track")
    track_b = state.get("track_b")
    new_a = track   if track   else cur_a
    new_b = track_b if track_b else cur_b
    return new_a, new_b


# ── 3D trajectory ─────────────────────────────────────────────────────────────
@callback(
    Output("v-graph-3d", "figure"),
    Input("v-track-a", "value"), Input("v-crop-a", "value"),
    Input("v-track-b", "value"), Input("v-crop-b", "value"),
    Input("v-avg-crops", "value"),
    Input("v-viz-mode", "value"),
    prevent_initial_call=True,
)
def update_3d_trajectory(track_a, crop_a, track_b, crop_b, avg_crops, viz_mode):
    if not track_a:
        return go.Figure()
    cfg = get_config()
    npz = get_analysis()
    d03 = npz.get("d03")

    # ── Load PCA components ─────────────────────────────────────────────────
    pca_path = Path(__file__).parent.parent.parent / "models" / "global_pca_3d.npz"
    if pca_path.exists():
        pca_data   = np.load(str(pca_path))
        components = pca_data["components"]    # [3, 64]
    elif d03 is not None:
        from scipy.linalg import svd
        U, _, _ = svd(d03["xcorr_matrix"])
        components = U[:, :3].T.astype(np.float32)
    else:
        components = np.eye(3, 64, dtype=np.float32)

    # ── Load Track A ────────────────────────────────────────────────────────
    try:
        if "avg" in (avg_crops or []):
            z_a = avg_crops_with_loop_gating(track_a, cfg["latent_dir"],
                                             cfg["source_dir"])
        else:
            if not crop_a:
                return go.Figure()
            z_a = load_latent(track_a, crop_a, cfg["latent_dir"])
    except Exception as e:
        return go.Figure(layout=dict(title=str(e), template="plotly_dark",
                                     paper_bgcolor="#0d0d1a"))

    pts_a = project_latent_pca(z_a, components)
    T     = pts_a.shape[0]
    t_arr = np.linspace(0, 1, T)

    traces = [go.Scatter3d(
        x=pts_a[:, 0], y=pts_a[:, 1], z=pts_a[:, 2],
        mode="lines+markers",
        marker=dict(size=2, color=t_arr, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="t", thickness=10, len=0.5)),
        line=dict(color=t_arr, colorscale="Viridis", width=3),
        name=track_a,
        hovertemplate="t=%{marker.color:.2f}<extra>" + track_a + "</extra>",
    )]

    z_b = None
    if track_b:
        try:
            if "avg" in (avg_crops or []):
                z_b = avg_crops_with_loop_gating(track_b, cfg["latent_dir"],
                                                  cfg["source_dir"])
            elif crop_b:
                z_b = load_latent(track_b, crop_b, cfg["latent_dir"])
            if z_b is not None:
                pts_b = project_latent_pca(z_b, components)
                T_b   = pts_b.shape[0]
                t_b   = np.linspace(0, 1, T_b)
                traces.append(go.Scatter3d(
                    x=pts_b[:, 0], y=pts_b[:, 1], z=pts_b[:, 2],
                    mode="lines+markers",
                    marker=dict(size=2, color=t_b, colorscale="Plasma",
                                showscale=False),
                    line=dict(color=t_b, colorscale="Plasma", width=2),
                    name=track_b,
                    hovertemplate="t=%{marker.color:.2f}<extra>" + track_b + "</extra>",
                ))
        except Exception:
            pass

    # Difference mode
    if viz_mode == "diff" and len(traces) == 2 and z_b is not None:
        T_min = min(z_a.shape[1], z_b.shape[1])
        diff  = z_a[:, :T_min] - z_b[:, :T_min]
        pts_d = project_latent_pca(diff, components)
        t_d   = np.linspace(0, 1, T_min)
        traces = [go.Scatter3d(
            x=pts_d[:, 0], y=pts_d[:, 1], z=pts_d[:, 2],
            mode="lines+markers",
            marker=dict(size=2, color=t_d, colorscale="RdBu"),
            line=dict(color=t_d, colorscale="RdBu", width=3),
            name="A − B",
        )]

    fig = go.Figure(data=traces, layout=_DARK3D)
    title = track_a + (f" / {track_b}" if track_b else "")
    fig.update_layout(title=dict(text=title, font=dict(size=12)))
    return fig


# ── Alignment bar ─────────────────────────────────────────────────────────────
@callback(
    Output("v-alignment-bar", "figure"),
    Input("v-track-a", "value"), Input("v-crop-a", "value"),
    Input("v-track-b", "value"), Input("v-crop-b", "value"),
    prevent_initial_call=True,
)
def update_alignment_bar(track_a, crop_a, track_b, crop_b):
    cfg = get_config()

    def _load_tc(track: str, crop: str) -> dict:
        if not track or not crop:
            return {}
        sidecar = Path(cfg["latent_dir"]) / track / f"{crop}.json"
        if not sidecar.exists():
            return {}
        try:
            with open(sidecar) as f:
                return json.load(f)
        except Exception:
            return {}

    tc_a = _load_tc(track_a, crop_a)
    tc_b = _load_tc(track_b, crop_b) if track_b else {}

    shapes = []; annotations = []

    def _add_markers(tc: dict, y_base: float, y_top: float, prefix: str):
        for t in tc.get("beats", []):
            shapes.append(dict(type="line", x0=t, x1=t, y0=y_base, y1=y_top,
                               line=dict(color="#4cd137", width=0.8)))
        for t in tc.get("downbeats", []):
            shapes.append(dict(type="line", x0=t, x1=t, y0=y_base, y1=y_top,
                               line=dict(color="#e94560", width=1.5)))
        for t in tc.get("onsets", [])[:500]:
            shapes.append(dict(type="line", x0=t, x1=t, y0=y_base, y1=y_top,
                               line=dict(color="#ffd700", width=0.5, dash="dot")))
        annotations.append(dict(x=0, y=(y_base+y_top)/2, text=prefix,
                                showarrow=False,
                                font=dict(size=9, color="#666"), xanchor="left"))

    _add_markers(tc_a, 0.5, 1.0, "A")
    if tc_b:
        _add_markers(tc_b, 0.0, 0.5, "B")

    dur = tc_a.get("duration") or (max(tc_a["beats"]) + 0.5
                                    if tc_a.get("beats") else 30)
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0d1a", plot_bgcolor="#111125",
        margin=dict(l=30, r=10, t=5, b=20), height=80,
        xaxis=dict(range=[0, dur], showticklabels=True, tickfont=dict(size=8),
                   showgrid=False),
        yaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        shapes=shapes, annotations=annotations, showlegend=False,
    )
    return fig


# ── Crossfader panel layout ───────────────────────────────────────────────────
@callback(Output("v-crossfader-panel", "children"),
          Input("v-xf-mode", "value"),
          prevent_initial_call=False)
def update_crossfader_panel(mode):
    npz    = get_analysis()
    d03    = npz.get("d03")
    cl     = d03["cluster_labels"] if d03 else np.zeros(64, dtype=int)
    n_cl   = int(cl.max()) if d03 else 0
    feat_n = list(npz["d01"]["feature_names"]) if npz.get("d01") else []
    r_mat  = npz["d01"]["r_pearson"]          if npz.get("d01") else None

    def cluster_label(c: int) -> str:
        if r_mat is None or not feat_n:
            return f"Cluster {c}"
        dims = np.where(cl == c)[0]
        mr   = r_mat[dims].mean(axis=0)
        top  = int(np.argmax(np.abs(mr)))
        return f"Cl.{c}: {feat_n[top] if top < len(feat_n) else '?'}"

    if mode == "simple":
        return html.Div([
            _ctrl("Mix A→B",
                  dcc.Slider(id="v-mix-alpha", min=0, max=1, step=0.01, value=0.0,
                             marks={0: "A", 1: "B"}, tooltip={"always_visible": False},
                             updatemode="drag")),
            dcc.RadioItems(id="v-interp",
                           options=[{"label": "Slerp", "value": "slerp"},
                                    {"label": "Lerp",  "value": "lerp"}],
                           value="slerp", inline=True,
                           style={"fontSize": "11px", "marginBottom": "6px"}),
            dcc.Checklist(id="v-smart-loop",
                          options=[{"label": "Smart Loop", "value": "smart_loop"}],
                          value=[], inline=True,
                          style={"fontSize": "11px", "marginBottom": "6px"}),
            html.Button("Beat Match", id="v-beat-match", n_clicks=0,
                        style={"fontSize": "11px", "padding": "4px 10px",
                               "background": "#1e2050", "border": "1px solid #4cd137",
                               "color": "#4cd137", "cursor": "pointer"}),
            html.Div(id="v-beat-match-info",
                     style={"fontSize": "10px", "color": "#666", "marginTop": "4px"}),
            html.Button("▶ Play", id="v-play-xf", n_clicks=0,
                        style={"marginTop": "8px", "fontSize": "11px",
                               "padding": "4px 10px",
                               "background": "#e94560", "border": "none",
                               "color": "white", "cursor": "pointer"}),
        ])

    # Advanced
    cluster_sliders = [
        html.Div([
            html.Label(cluster_label(c),
                       style={"fontSize": "10px", "color": "#888",
                               "width": "110px", "flexShrink": "0"}),
            dcc.Slider(id={"type": "cluster-alpha", "index": c},
                       min=0, max=1, step=0.05, value=0.0,
                       marks={0: "A", 1: "B"}, tooltip={"always_visible": False},
                       updatemode="drag"),
        ], style={"display": "flex", "alignItems": "center", "gap": "6px",
                  "marginBottom": "3px"})
        for c in range(1, n_cl + 1)
    ] if n_cl > 0 else [html.P("No cluster data.",
                                style={"color": "#555", "fontSize": "11px"})]

    return html.Div([
        dcc.RadioItems(id="v-dim-mix-mode",
                       options=[{"label": "By cluster",   "value": "cluster"},
                                {"label": "By dim range", "value": "regex"}],
                       value="cluster", inline=True,
                       style={"fontSize": "11px", "marginBottom": "6px"}),
        html.Div(id="v-cluster-sliders", children=cluster_sliders),
        html.Div([
            dcc.Input(id="v-dim-regex", type="text",
                      placeholder="e.g. 0-15,32,48-63",
                      style={"width": "180px", "background": "#0f1535",
                             "color": "#ccc", "border": "1px solid #2a2a4a",
                             "padding": "3px", "fontSize": "11px"}),
            html.Div("Comma-separated dim indices or ranges. "
                     "Only these dims blend A→B; others stay at A.",
                     style={"fontSize": "9px", "color": "#555", "marginTop": "3px"}),
        ], id="v-regex-panel", style={"display": "none"}),
        dcc.RadioItems(id="v-interp",
                       options=[{"label": "Slerp", "value": "slerp"},
                                {"label": "Lerp",  "value": "lerp"}],
                       value="slerp", inline=True,
                       style={"fontSize": "11px", "margin": "6px 0"}),
        dcc.Checklist(id="v-smart-loop",
                      options=[{"label": "Smart Loop", "value": "smart_loop"}],
                      value=[], inline=True,
                      style={"fontSize": "11px", "marginBottom": "6px"}),
        html.Button("Beat Match", id="v-beat-match", n_clicks=0,
                    style={"fontSize": "11px", "padding": "4px 10px",
                           "background": "#1e2050", "border": "1px solid #4cd137",
                           "color": "#4cd137", "cursor": "pointer"}),
        html.Div(id="v-beat-match-info",
                 style={"fontSize": "10px", "color": "#666", "marginTop": "4px"}),
        html.Button("▶ Play", id="v-play-xf", n_clicks=0,
                    style={"marginTop": "8px", "fontSize": "11px",
                           "padding": "4px 10px",
                           "background": "#e94560", "border": "none",
                           "color": "white", "cursor": "pointer"}),
    ])


@callback(
    Output("v-regex-panel",     "style"),
    Output("v-cluster-sliders", "style"),
    Input("v-dim-mix-mode", "value"),
    prevent_initial_call=True,
)
def toggle_dim_mix_mode(mode):
    show = {"display": "block"}; hide = {"display": "none"}
    if mode == "regex":
        return show, hide
    return hide, show


# ── Crossfader play button ────────────────────────────────────────────────────
@callback(
    Output("player-cmd", "data", allow_duplicate=True),
    Input("v-play-xf", "n_clicks"),
    State("v-track-a",   "value"), State("v-crop-a",    "value"),
    State("v-track-b",   "value"), State("v-crop-b",    "value"),
    State("v-mix-alpha", "value"),
    State("v-interp",    "value"),
    State("v-smart-loop","value"),
    State("v-avg-crops", "value"),
    State({"type": "manip-slider", "index": dash.ALL}, "value"),
    State({"type": "manip-slider", "index": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def viewer_play(n, track_a, crop_a, track_b, crop_b, alpha, interp,
                smart_loop, avg_crops, manip_values, manip_ids):
    if not n or not track_a:
        return no_update
    smart = "smart_loop" in (smart_loop or [])
    avg   = "avg" in (avg_crops or [])
    manip = {mid["index"]: float(v or 0)
             for mid, v in zip(manip_ids or [], manip_values or [])
             if abs(float(v or 0)) > 1e-6}
    manip_params = {f"manip_{k}": str(v) for k, v in manip.items()} if manip else None
    if avg:
        url = build_average_url(track_a, track_b or None, float(alpha or 0),
                                interp or "slerp", smart)
    elif track_b and crop_b:
        url = build_crossfade_url(track_a, str(crop_a or "0.5"),
                                  track_b, str(crop_b or "0.5"),
                                  float(alpha or 0), interp or "slerp", smart,
                                  manip=manip_params)
    else:
        url = build_decode_url(track_a, "0.5", smart, manip=manip_params)
    return {"action": "play", "url": url, "loop_start": None, "loop_end": None}


@callback(
    Output("v-beat-match-info", "children"),
    Input("v-beat-match", "n_clicks"),
    State("v-track-a", "value"), State("v-crop-a", "value"),
    State("v-track-b", "value"), State("v-crop-b", "value"),
    prevent_initial_call=True,
)
def beat_match(n, track_a, crop_a, track_b, crop_b):
    if not n or not track_a or not track_b:
        return no_update
    cfg = get_config()

    def _bpm(track, crop):
        if not crop:
            return None
        p = Path(cfg["latent_dir"]) / track / f"{crop}.json"
        if not p.exists():
            return None
        try:
            with open(p) as f:
                return json.load(f).get("bpm")
        except Exception:
            return None

    bpm_a = _bpm(track_a, crop_a)
    bpm_b = _bpm(track_b, crop_b)
    if bpm_a and bpm_b:
        ratio = bpm_a / bpm_b
        return f"A: {bpm_a:.1f} BPM | B: {bpm_b:.1f} BPM | ratio: {ratio:.4f}"
    return "BPM data not available for one or both tracks."
