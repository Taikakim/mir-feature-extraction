"""Tab 1 — Dataset Explorer: 8-mode scatter + sidebar."""
from __future__ import annotations

import json

import dash
from dash import dcc, html, Input, Output, State, callback, no_update, Patch
import numpy as np
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.explorer.data import (
    get_app_data, get_analysis, CURATED_FEATURES, FEATURE_UNITS, FEATURE_GROUPS,
    norm01, corrcoef, parse_class_label,
)
from plots.explorer.audio import build_decode_url

# ── Plot style defaults ───────────────────────────────────────────────────────
_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0d0d1a",
    plot_bgcolor="#111125",
    font=dict(color="#ccc"),
    hoverlabel=dict(bgcolor="#0f1535", font=dict(size=12, color="white")),
    dragmode="lasso",
    margin=dict(t=46, b=48, l=58, r=20),
)

def _dark_layout(**overrides) -> dict:
    """Merge _DARK defaults with mode-specific overrides (overrides win)."""
    return {**_DARK, **overrides}

CLASS_COLORS = [
    '#e94560', '#4cd137', '#00d2ff', '#fbc531', '#9c88ff', '#ff79c6', '#ff9f43',
    '#0097e6', '#bdc581', '#fd7272', '#5f27cd', '#55efc4', '#81ecec', '#ff7675',
    '#a29bfe', '#e1b12c', '#00cec9', '#e84393', '#badc58', '#c8d6e5',
]

_HIDE = {"display": "none"}
_SHOW = {"display": ""}


def _cluster_top_feature(cluster_data: dict, valid_mask: np.ndarray):
    """
    Given a cluster_data dict {"dims": [...], "cluster": N} and a boolean mask
    of valid track rows, return (feature_name, color_values) where feature_name
    is the MIR feature most correlated (by absolute Pearson r) with the cluster's
    latent dims, and color_values is that feature's values for the valid rows.
    Returns (None, None) if analysis data or feature is unavailable.
    """
    try:
        an = get_analysis()
        d01 = an.get("d01") if an else None
        if d01 is None:
            return None, None
        dims = cluster_data.get("dims")
        if not dims:
            return None, None
        r_pearson    = d01["r_pearson"]        # (64, n_features)
        feature_names = d01["feature_names"]   # (n_features,)
        dims_arr = np.array(dims, dtype=int)
        # Clip dims to valid range
        dims_arr = dims_arr[(dims_arr >= 0) & (dims_arr < r_pearson.shape[0])]
        if len(dims_arr) == 0:
            return None, None
        mean_r = np.abs(r_pearson[dims_arr, :]).mean(axis=0)   # (n_features,)
        ranked = np.argsort(mean_r)[::-1]                      # descending
        ad = get_app_data()
        for top_idx in ranked:
            top_feat = str(feature_names[top_idx])
            if top_feat in ad.num_cols:
                cvals = ad.feat_array(top_feat)[valid_mask]
                return top_feat, cvals
        return None, None
    except Exception:
        return None, None


# ── Layout ────────────────────────────────────────────────────────────────────
def layout() -> html.Div:
    ad = get_app_data()
    feat_opts  = [{"label": f, "value": f} for f in sorted(ad.num_cols)]
    class_opts = [{"label": c, "value": c} for c in ad.class_cols]
    track_opts = ad.track_options()
    group_opts = [{"label": g, "value": g}
                  for g in list(FEATURE_GROUPS.keys()) + ["All"]]

    # Default values
    default_x  = "bpm" if "bpm" in ad.num_cols else (ad.num_cols[0] if ad.num_cols else None)
    default_y  = "brightness" if "brightness" in ad.num_cols else (ad.num_cols[1] if len(ad.num_cols) > 1 else default_x)
    default_xp = default_x
    default_xn = "roughness" if "roughness" in ad.num_cols else default_y
    default_yp = "danceability" if "danceability" in ad.num_cols else default_x
    default_yn = "atonality" if "atonality" in ad.num_cols else default_y

    return html.Div([
        # ── Mode radio ──────────────────────────────────────────────────────
        html.Div([
            dcc.RadioItems(
                id="view-mode",
                options=[
                    {"label": "Scatter",    "value": "scatter"},
                    {"label": "Quadrant",   "value": "quadrant"},
                    {"label": "Histogram",  "value": "histogram"},
                    {"label": "Radar",      "value": "radar"},
                    {"label": "Heatmap",    "value": "heatmap"},
                    {"label": "Parallel",   "value": "parallel"},
                    {"label": "Similarity", "value": "similarity"},
                    {"label": "Classes",    "value": "classes"},
                ],
                value="scatter",
                inline=True,
                style={"fontSize": "12px", "gap": "8px"},
            ),
            html.Button("?", id="btn-help", title="Show keyboard shortcuts",
                        style={"marginLeft": "auto", "background": "none",
                               "border": "1px solid #444", "color": "#777",
                               "cursor": "pointer", "borderRadius": "50%",
                               "width": "22px", "height": "22px", "fontSize": "11px",
                               "padding": "0", "lineHeight": "20px", "flexShrink": "0"}),
        ], className="controls-bar", style={"display": "flex", "alignItems": "center"}),

        # ── Help panel (hidden by default) ───────────────────────────────────
        html.Div([
            html.Div([
                html.Span("Shortcuts & Tips", style={"fontWeight": "bold",
                                                      "fontSize": "12px", "color": "#e94560"}),
                html.Button("✕", id="btn-help-close",
                            style={"marginLeft": "auto", "background": "none", "border": "none",
                                   "color": "#777", "cursor": "pointer", "fontSize": "13px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            html.Table([
                html.Tbody([
                    html.Tr([html.Td("Click point", style={"color": "#888", "paddingRight": "12px"}),
                             html.Td("Load in slot A (player)")]),
                    html.Tr([html.Td("Double-click point"),
                             html.Td("Load in slot B (rapid 2nd click)")]),
                    html.Tr([html.Td("B button in sidebar"),
                             html.Td("Load track in slot B (reliable)")]),
                    html.Tr([html.Td("New lasso / click background"),
                             html.Td("Deselect / clear selection")]),
                    html.Tr([html.Td("L"),
                             html.Td("Lasso select tool")]),
                    html.Tr([html.Td("B"),
                             html.Td("Box select tool")]),
                    html.Tr([html.Td("P"),
                             html.Td("Pan tool")]),
                    html.Tr([html.Td("Z / +"),
                             html.Td("Zoom in")]),
                    html.Tr([html.Td("Z + Shift / −"),
                             html.Td("Zoom out")]),
                    html.Tr([html.Td("R or Home"),
                             html.Td("Reset axes")]),
                    html.Tr([html.Td("Position slider release"),
                             html.Td("Prefetches audio for instant playback")]),
                    html.Tr([html.Td("Continue auto"),
                             html.Td("Advances to next crop when current ends")]),
                    html.Tr([html.Td("VAE fade"),
                             html.Td("Smooth latent-space crossfade into next crop")]),
                ]),
            ], style={"fontSize": "11px", "borderCollapse": "collapse",
                      "lineHeight": "1.8"}),
        ], id="help-panel",
           style={"display": "none", "background": "#0f1535",
                  "border": "1px solid #2a2a4a", "padding": "12px 16px",
                  "fontSize": "11px", "color": "#ccc"}),

        # ── ALL mode-specific controls (always in DOM, visibility toggled) ──
        # Scatter controls
        html.Div([
            _label_group("X axis",   dcc.Dropdown(id="sel-x", options=feat_opts,
                                                  value=default_x, clearable=False,
                                                  style={"width": "180px"})),
            _label_group("Y axis",   dcc.Dropdown(id="sel-y", options=feat_opts,
                                                  value=default_y, clearable=False,
                                                  style={"width": "180px"})),
            _label_group("Colour by", dcc.Dropdown(
                id="sel-colour",
                options=[{"label": "None", "value": ""},
                         {"label": "Outliers", "value": "__outliers__"}]
                + feat_opts,
                value="", clearable=True,
                style={"width": "160px"})),
            dcc.Checklist(id="chk-scatter-opts",
                          options=[{"label": "Trend", "value": "trend"},
                                   {"label": "Log X", "value": "logx"},
                                   {"label": "Log Y", "value": "logy"}],
                          value=[], inline=True, style={"fontSize": "11px"}),
            dcc.Checklist(id="chk-show-clusters",
                          options=[{"label": "Show clusters", "value": "clusters"}],
                          value=[], inline=True, style={"fontSize": "11px"}),
        ], id="ctrl-scatter", className="controls-bar", style={"gap": "12px"}),

        # Quadrant controls
        html.Div([
            _label_group("X+", dcc.Dropdown(id="sel-xp", options=feat_opts,
                                            value=default_xp, clearable=False,
                                            style={"width": "160px"})),
            _label_group("X−", dcc.Dropdown(id="sel-xn", options=feat_opts,
                                            value=default_xn, clearable=False,
                                            style={"width": "160px"})),
            _label_group("Y+", dcc.Dropdown(id="sel-yp", options=feat_opts,
                                            value=default_yp, clearable=False,
                                            style={"width": "160px"})),
            _label_group("Y−", dcc.Dropdown(id="sel-yn", options=feat_opts,
                                            value=default_yn, clearable=False,
                                            style={"width": "160px"})),
        ], id="ctrl-quadrant", className="controls-bar", style=_HIDE),

        # Histogram controls
        html.Div([
            _label_group("Feature", dcc.Dropdown(id="sel-hist", options=feat_opts,
                                                  value=default_x, clearable=False,
                                                  style={"width": "200px"})),
            _label_group("Bins", dcc.Input(id="hist-bins", type="number",
                                           value=30, min=4, max=64,
                                           style={"width": "60px"})),
        ], id="ctrl-histogram", className="controls-bar", style=_HIDE),

        # Radar controls
        html.Div([
            _label_group("Track", dcc.Dropdown(
                id="sel-radar-track",
                options=track_opts,
                value=track_opts[0]["value"] if track_opts else None,
                clearable=False, style={"width": "320px"},
                placeholder="Search track…")),
        ], id="ctrl-radar", className="controls-bar", style=_HIDE),

        # Heatmap controls
        html.Div([
            _label_group("Feature group",
                         dcc.Dropdown(id="sel-heatmap-group", options=group_opts,
                                      value="All", clearable=False,
                                      style={"width": "180px"})),
        ], id="ctrl-heatmap", className="controls-bar", style=_HIDE),

        # Parallel controls
        html.Div([
            html.Span("Curated features | Drag axis bands to filter",
                      style={"fontSize": "11px", "color": "#666"}),
        ], id="ctrl-parallel", className="controls-bar", style=_HIDE),

        # Similarity controls
        html.Div([
            _label_group("Reference", dcc.Dropdown(
                id="sel-sim-ref",
                options=track_opts,
                value=track_opts[0]["value"] if track_opts else None,
                clearable=False, style={"width": "280px"},
                placeholder="Search reference…")),
            _label_group("X+", dcc.Dropdown(id="sim-xp", options=feat_opts,
                                            value=default_xp, clearable=False,
                                            style={"width": "140px"})),
            _label_group("X−", dcc.Dropdown(id="sim-xn", options=feat_opts,
                                            value=default_xn, clearable=False,
                                            style={"width": "140px"})),
            _label_group("Y+", dcc.Dropdown(id="sim-yp", options=feat_opts,
                                            value=default_yp, clearable=False,
                                            style={"width": "140px"})),
            _label_group("Y−", dcc.Dropdown(id="sim-yn", options=feat_opts,
                                            value=default_yn, clearable=False,
                                            style={"width": "140px"})),
        ], id="ctrl-similarity", className="controls-bar", style=_HIDE),

        # Classes controls
        html.Div([
            _label_group("X axis",   dcc.Dropdown(id="sel-class-x", options=feat_opts,
                                                   value=default_x, clearable=False,
                                                   style={"width": "160px"})),
            _label_group("Y axis",   dcc.Dropdown(id="sel-class-y", options=feat_opts,
                                                   value=default_y, clearable=False,
                                                   style={"width": "160px"})),
            _label_group("Class by", dcc.Dropdown(
                id="sel-class-by",
                options=class_opts,
                value=class_opts[0]["value"] if class_opts else None,
                clearable=False, style={"width": "180px"})),
            dcc.Checklist(id="chk-class-trend",
                          options=[{"label": "Per-class trend", "value": "trend"}],
                          value=[], inline=True, style={"fontSize": "11px"}),
        ], id="ctrl-classes", className="controls-bar", style=_HIDE),

        # ── Main area: plot + sidebar ────────────────────────────────────────
        html.Div([
            html.Div([
                dcc.Graph(
                    id="dataset-graph",
                    style={"height": "calc(100vh - 260px)"},
                    config={"displayModeBar": True, "scrollZoom": True,
                            "doubleClick": False},
                    figure=go.Figure(layout=go.Layout(**_DARK)),
                ),
                html.Div(id="info-bar",
                         style={"fontSize": "11px", "color": "#666",
                                "padding": "3px 8px", "background": "#0a0a15"}),
            ], style={"flex": "1", "minWidth": 0}),
            html.Div([
                html.Div([
                    html.Span("Tracks", id="tl-title",
                              style={"fontWeight": "bold", "fontSize": "12px"}),
                    dcc.Dropdown(
                        id="tl-service",
                        options=[
                            {"label": "Spotify",     "value": "spotify"},
                            {"label": "Tidal",       "value": "tidal"},
                            {"label": "MusicBrainz", "value": "musicbrainz"},
                        ],
                        value="tidal", clearable=False,
                        style={"width": "120px", "fontSize": "11px"}),
                ], style={"display": "flex", "justifyContent": "space-between",
                          "alignItems": "center", "marginBottom": "4px"}),
                html.Div(id="track-list-body",
                         style={"overflowY": "auto",
                                "maxHeight": "calc(50vh - 120px)"}),
                html.Hr(style={"borderColor": "#2a2a4a"}),
                html.Div(id="nn-panel"),
            ], className="sidebar"),
        ], style={"display": "flex", "gap": "0"}),
    ])


# ── Controls fragments ────────────────────────────────────────────────────────
def _label_group(label: str, child) -> html.Div:
    return html.Div([
        html.Label(label, style={"fontSize": "10px", "color": "#888",
                                 "textTransform": "uppercase"}),
        child,
    ], style={"display": "flex", "flexDirection": "column", "gap": "2px"})


# ── Mode → controls visibility switcher ───────────────────────────────────────
# Instead of replacing children (which destroys component IDs), we toggle display
_CTRL_IDS = [
    "ctrl-scatter", "ctrl-quadrant", "ctrl-histogram", "ctrl-radar",
    "ctrl-heatmap", "ctrl-parallel", "ctrl-similarity", "ctrl-classes",
]
_MODE_TO_CTRL = {
    "scatter":    "ctrl-scatter",
    "quadrant":   "ctrl-quadrant",
    "histogram":  "ctrl-histogram",
    "radar":      "ctrl-radar",
    "heatmap":    "ctrl-heatmap",
    "parallel":   "ctrl-parallel",
    "similarity": "ctrl-similarity",
    "classes":    "ctrl-classes",
}


@callback(
    [Output(cid, "style") for cid in _CTRL_IDS],
    Input("view-mode", "value"),
)
def toggle_mode_controls(mode: str):
    active = _MODE_TO_CTRL.get(mode, "")
    return [_SHOW if cid == active else _HIDE for cid in _CTRL_IDS]


# ── Main graph callback — single dispatcher for all 8 modes ──────────────────
@callback(
    Output("dataset-graph", "figure"),
    Output("info-bar", "children"),
    # mode
    Input("view-mode", "value"),
    # scatter
    Input("sel-x",         "value"),
    Input("sel-y",         "value"),
    Input("sel-colour",    "value"),
    Input("chk-scatter-opts", "value"),
    # quadrant
    Input("sel-xp", "value"),
    Input("sel-xn", "value"),
    Input("sel-yp", "value"),
    Input("sel-yn", "value"),
    # histogram
    Input("sel-hist",  "value"),
    Input("hist-bins", "value"),
    # heatmap
    Input("sel-heatmap-group", "value"),
    # radar
    Input("sel-radar-track", "value"),
    # similarity
    Input("sel-sim-ref", "value"),
    Input("sim-xp", "value"),
    Input("sim-xn", "value"),
    Input("sim-yp", "value"),
    Input("sim-yn", "value"),
    # classes (separate IDs from scatter to avoid conflicts)
    Input("sel-class-x",     "value"),
    Input("sel-class-y",     "value"),
    Input("sel-class-by",    "value"),
    Input("chk-class-trend", "value"),
    # cluster overlay
    Input("cluster-highlight", "data"),
    Input("chk-show-clusters", "value"),
    # radar selection state
    Input("radar-selection", "data"),
    # active track (State only — highlight is patched separately to preserve selectedData)
    State("active-track", "data"),
    prevent_initial_call=False,
)
def update_graph(
    mode,
    # scatter
    kx, ky, colour, scatter_opts,
    # quadrant
    kxp, kxn, kyp, kyn,
    # histogram
    hist_feat, hist_bins,
    # heatmap
    heatmap_group,
    # radar
    radar_track,
    # similarity
    sim_ref, sim_xp, sim_xn, sim_yp, sim_yn,
    # classes
    class_kx, class_ky, class_by, class_trend_opts,
    # cluster
    cluster_data, show_clusters,
    # radar selection
    radar_selected,
    # active track
    active_track_data,
):
    ad = get_app_data()
    empty = go.Figure(layout=dict(**_DARK))

    # ── Scatter ───────────────────────────────────────────────────────────────
    if mode == "scatter":
        if not kx or not ky or kx not in ad.num_cols or ky not in ad.num_cols:
            return empty, "Select X and Y features."
        dx, dy = ad.feat_array(kx), ad.feat_array(ky)
        valid = np.isfinite(dx) & np.isfinite(dy)
        x, y = dx[valid], dy[valid]
        names = [ad.tracks[i] for i in np.where(valid)[0]]
        idxs  = np.where(valid)[0]
        scatter_opts = scatter_opts or []
        r = corrcoef(x, y)
        ux = f" [{FEATURE_UNITS[kx]}]" if kx in FEATURE_UNITS else ""
        uy = f" [{FEATURE_UNITS[ky]}]" if ky in FEATURE_UNITS else ""
        hover = [f"<b>{n}</b><br>{kx}: {x[j]:.4f}<br>{ky}: {y[j]:.4f}"
                 for j, n in enumerate(names)]

        if colour == "__outliers__":
            mx, sx = x.mean(), x.std()
            my, sy = y.mean(), y.std()
            is_out = (np.abs(x - mx) > 2*sx) | (np.abs(y - my) > 2*sy)
            traces = [
                go.Scatter(x=x[~is_out], y=y[~is_out], mode="markers",
                             marker=dict(color="#e94560", size=5, opacity=0.5),
                             hovertext=[hover[j] for j in np.where(~is_out)[0]],
                             hoverinfo="text", name=f"normal ({(~is_out).sum()})",
                             customdata=idxs[~is_out].tolist()),
                go.Scatter(x=x[is_out], y=y[is_out], mode="markers",
                             marker=dict(color="#ffd700", size=9, opacity=0.85),
                             hovertext=[hover[j] for j in np.where(is_out)[0]],
                             hoverinfo="text", name=f"outlier ({is_out.sum()})",
                             customdata=idxs[is_out].tolist()),
            ]
        elif colour and colour in ad.num_cols:
            cvals = ad.feat_array(colour)[valid]
            traces = [go.Scatter(x=x, y=y, mode="markers",
                                   marker=dict(color=cvals, colorscale="Viridis",
                                               showscale=True, size=5, opacity=0.7,
                                               colorbar=dict(title=colour, thickness=14)),
                                   hovertext=hover, hoverinfo="text",
                                   name="tracks", customdata=idxs.tolist())]
        elif cluster_data and "clusters" in (show_clusters or []):
            # Cluster overlay: color scatter by the top MIR feature correlated with
            # the highlighted cluster's latent dims.
            top_feat, cvals = _cluster_top_feature(cluster_data, valid)
            if top_feat is not None and cvals is not None:
                cl_label = cluster_data.get("cluster", "?")
                traces = [go.Scatter(x=x, y=y, mode="markers",
                                       marker=dict(color=cvals, colorscale="Plasma",
                                                   showscale=True, size=5, opacity=0.7,
                                                   colorbar=dict(
                                                       title=dict(text=f"cluster {cl_label}<br>{top_feat}",
                                                                  side="right"),
                                                       thickness=14)),
                                       hovertext=hover, hoverinfo="text",
                                       name=f"cluster {cl_label} → {top_feat}",
                                       customdata=idxs.tolist())]
            else:
                traces = [go.Scatter(x=x, y=y, mode="markers",
                                       marker=dict(color="#e94560", size=5, opacity=0.5),
                                       hovertext=hover, hoverinfo="text",
                                       name="tracks", customdata=idxs.tolist())]
        else:
            traces = [go.Scatter(x=x, y=y, mode="markers",
                                   marker=dict(color="#e94560", size=5, opacity=0.5),
                                   hovertext=hover, hoverinfo="text",
                                   name="tracks", customdata=idxs.tolist())]

        if "trend" in scatter_opts and len(x) > 2:
            mx2, my2 = x.mean(), y.mean()
            num = ((x-mx2)*(y-my2)).sum()
            den = ((x-mx2)**2).sum()
            slope = num/den if den > 1e-12 else 0
            ic = my2 - slope * mx2
            xmn, xmx = x.min(), x.max()
            traces.append(go.Scatter(
                x=[xmn, xmx], y=[slope*xmn+ic, slope*xmx+ic],
                mode="lines", line=dict(color="#00d2ff", width=2, dash="dash"),
                name=f"trend (r={r:.3f})", hoverinfo="skip",
            ))

        # Active track highlight — gold ring (always last trace so Patch callback can update it)
        at = _safe_tidx((active_track_data or {}).get("track_idx"))
        ring_x, ring_y = [], []
        if at is not None:
            pos = np.where(idxs == at)[0]
            if len(pos) > 0:
                p = int(pos[0])
                ring_x, ring_y = [float(x[p])], [float(y[p])]
        traces.append(go.Scatter(
            x=ring_x, y=ring_y, mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=14, opacity=1.0,
                        line=dict(color="#ffd700", width=2)),
            hoverinfo="skip", name="active", showlegend=False,
        ))

        log_x = "logx" in scatter_opts and np.all(x > 0)
        log_y = "logy" in scatter_opts and np.all(y > 0)
        layout = dict(**_DARK,
                      title=dict(text=f"{kx} vs {ky}  (r={r:.3f}, n={len(x)})",
                                 font=dict(size=14)),
                      xaxis=dict(title=kx+ux, type="log" if log_x else "linear"),
                      yaxis=dict(title=ky+uy, type="log" if log_y else "linear"),
                      showlegend=True)
        return go.Figure(data=traces, layout=layout), f"{kx} vs {ky} | r={r:.3f} | n={len(x)}"

    # ── Quadrant ──────────────────────────────────────────────────────────────
    if mode == "quadrant":
        if not all([kxp, kxn, kyp, kyn]):
            return empty, "Select all four quadrant features."
        dxp, dxn = ad.feat_array(kxp), ad.feat_array(kxn)
        dyp, dyn = ad.feat_array(kyp), ad.feat_array(kyn)
        valid = np.isfinite(dxp) & np.isfinite(dxn) & np.isfinite(dyp) & np.isfinite(dyn)
        idxs = np.where(valid)[0]
        nxp = norm01(dxp[valid]); nxn = norm01(dxn[valid])
        nyp = norm01(dyp[valid]); nyn = norm01(dyn[valid])
        x = nxp - nxn; y = nyp - nyn
        hover = [f"<b>{ad.tracks[i]}</b><br>X:{x[j]:.3f} Y:{y[j]:.3f}"
                 for j, i in enumerate(idxs)]
        layout = dict(**_DARK,
                      title=dict(text=f"Quadrant (n={len(idxs)})", font=dict(size=14)),
                      xaxis=dict(title=f"← {kxn}  |  {kxp} →",
                                 range=[-1.1, 1.1], zeroline=True, zerolinecolor="#555577"),
                      yaxis=dict(title=f"← {kyn}  |  {kyp} →",
                                 range=[-1.1, 1.1], zeroline=True, zerolinecolor="#555577",
                                 scaleanchor="x"),
                      showlegend=False,
                      annotations=[
                          dict(x=1.0,  y=0, text=f"<b>{kxp}</b>", showarrow=False,
                               font=dict(size=11, color="#00d2ff"), xanchor="right",
                               bgcolor="#0f3460", bordercolor="#e94560", borderpad=3),
                          dict(x=-1.0, y=0, text=f"<b>{kxn}</b>", showarrow=False,
                               font=dict(size=11, color="#00d2ff"), xanchor="left",
                               bgcolor="#0f3460", bordercolor="#e94560", borderpad=3),
                          dict(x=0, y=1.0,  text=f"<b>{kyp}</b>", showarrow=False,
                               font=dict(size=11, color="#e94560"), yanchor="bottom",
                               bgcolor="#0f3460", bordercolor="#e94560", borderpad=3),
                          dict(x=0, y=-1.0, text=f"<b>{kyn}</b>", showarrow=False,
                               font=dict(size=11, color="#e94560"), yanchor="top",
                               bgcolor="#0f3460", bordercolor="#e94560", borderpad=3),
                      ])
        trace = go.Scatter(x=x, y=y, mode="markers",
                             marker=dict(color="#e94560", size=6, opacity=0.5),
                             hovertext=hover, hoverinfo="text", customdata=idxs.tolist())
        return go.Figure(data=[trace], layout=layout), f"X: {kxp}−{kxn} | Y: {kyp}−{kyn} | n={len(idxs)}"

    # ── Histogram ─────────────────────────────────────────────────────────────
    if mode == "histogram":
        if not hist_feat or hist_feat not in ad.num_cols:
            return empty, "Select a feature."
        vals = ad.feat_array(hist_feat)
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        if n == 0:
            return empty, f"{hist_feat}: no data"
        nbins = max(4, min(64, int(hist_bins or 30)))
        m, s  = vals.mean(), vals.std()
        vmin, vmax = vals.min(), vals.max()
        bsize = (vmax - vmin) / nbins if nbins > 0 else 1
        u = f" [{FEATURE_UNITS[hist_feat]}]" if hist_feat in FEATURE_UNITS else ""
        trace = go.Histogram(
            x=vals, xbins=dict(start=vmin, end=vmax, size=bsize),
            marker=dict(color="#e94560", opacity=0.75,
                        line=dict(color="#c03050", width=0.5)),
            name=hist_feat,
        )
        layout = _dark_layout(
                      title=dict(text=f"{hist_feat}  (n={n}, μ={m:.3f}, σ={s:.3f})",
                                 font=dict(size=14)),
                      xaxis=dict(title=hist_feat+u), yaxis=dict(title="count"),
                      dragmode="select",
                      shapes=[
                          dict(type="line", x0=m, x1=m, y0=0, y1=1, yref="paper",
                               line=dict(color="#00d2ff", width=2, dash="dash")),
                          dict(type="line", x0=m-s, x1=m-s, y0=0, y1=1, yref="paper",
                               line=dict(color="#888", width=1, dash="dot")),
                          dict(type="line", x0=m+s, x1=m+s, y0=0, y1=1, yref="paper",
                               line=dict(color="#888", width=1, dash="dot")),
                      ])
        return go.Figure(data=[trace], layout=layout), f"{hist_feat} | n={n} | μ={m:.3f} | σ={s:.3f} | bins={nbins}"

    # ── Heatmap ───────────────────────────────────────────────────────────────
    if mode == "heatmap":
        feats = (
            [f for g in FEATURE_GROUPS.values() for f in g if f in ad.num_cols]
            if heatmap_group == "All" or not heatmap_group
            else [f for f in FEATURE_GROUPS.get(heatmap_group, []) if f in ad.num_cols]
        )
        feats = feats[:40]
        n = len(feats)
        if n == 0:
            return empty, "No features for this group."
        mat = np.zeros((n, n))
        for i, fi in enumerate(feats):
            for j, fj in enumerate(feats):
                if i == j:
                    mat[i, j] = 1.0
                    continue
                xi, xj = ad.feat_array(fi), ad.feat_array(fj)
                valid = np.isfinite(xi) & np.isfinite(xj)
                mat[i, j] = corrcoef(xi[valid], xj[valid])
        trace = go.Heatmap(
            z=mat, x=feats, y=feats,
            colorscale=[[0, "#00d2ff"], [0.5, "#0d0d1a"], [1, "#e94560"]],
            zmin=-1, zmax=1,
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r=%{z:.3f}<extra></extra>",
        )
        layout = _dark_layout(
                      title=dict(text=f"Pearson r — {n} features", font=dict(size=14)),
                      xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                      yaxis=dict(tickfont=dict(size=9)),
                      margin=dict(t=46, b=80, l=80, r=20))
        return go.Figure(data=[trace], layout=layout), f"Pearson r | {n} features | click cell → Scatter"

    # ── Parallel ──────────────────────────────────────────────────────────────
    if mode == "parallel":
        feats = [f for f in CURATED_FEATURES if f in ad.num_cols]
        dims = []
        for i, f in enumerate(feats):
            arr = ad.feat_array(f).tolist()
            short = _parcoords_label(f)
            label = ("\n" + short) if i % 2 == 1 else short   # alternating rows
            dims.append(dict(label=label, values=arr))
        color_vals = ad.feat_array(feats[0]).tolist() if feats else []
        trace = go.Parcoords(
            line=dict(
                color=color_vals,
                colorscale=[[0, "#0d0520"], [0.33, "#2e0d5c"],
                            [0.66, "#6b2ca0"], [1, "#a855d4"]],
                showscale=True,
                colorbar=dict(title=feats[0] if feats else "", thickness=12),
            ),
            dimensions=dims,
        )
        layout = _dark_layout(
                      font=dict(color="#ccc", size=10),
                      title=dict(text=f"Parallel Coords — {len(feats)} curated features",
                                 font=dict(size=14)),
                      margin=dict(t=60, b=20, l=30, r=60))
        return go.Figure(data=[trace], layout=layout), f"Parallel coords | {len(feats)} features | drag bands to filter"

    # ── Radar ─────────────────────────────────────────────────────────────────
    if mode == "radar":
        if not radar_track or radar_track not in ad.tracks:
            return empty, "Select a track."
        tidx = ad.tracks.index(radar_track)
        feats = [f for f in CURATED_FEATURES if f in ad.num_cols
                 and np.isfinite(ad.feat_array(f)[tidx])]
        if not feats:
            return empty, "No curated features for this track."

        selected: set[str] = set(radar_selected or [])

        r_inner, r_mid, r_outer = [], [], []
        c_inner, c_mid, c_outer = [], [], []
        h_inner, h_mid, h_outer = [], [], []
        labels = [_short_label(f) for f in feats]

        for f in feats:
            all_vals = ad.feat_array(f)
            valid    = all_vals[np.isfinite(all_vals)]
            p01  = float(np.percentile(valid, 1))
            p99  = float(np.percentile(valid, 99))
            mn   = float(valid.mean())
            raw_t = float(ad.feat_array(f)[tidx])
            clamp = lambda v, _p01=p01, _p99=p99: max(
                0.0, min(1.0, (v - _p01) / (_p99 - _p01 + 1e-12)))
            nmt, nmd = clamp(raw_t), clamp(mn)
            v1, v2 = min(nmt, nmd), max(nmt, nmd)
            t_gt_m = raw_t >= mn
            r_inner.append(v1)
            r_mid.append(v2 - v1)
            r_outer.append(1.0 - v2)
            u = f" {FEATURE_UNITS[f]}" if f in FEATURE_UNITS else ""
            fmt = lambda v, _u=u: f"{v:.2f}{_u}"
            if t_gt_m:
                c_inner.append("rgba(233,69,96,0.4)")
                c_mid.append("rgba(233,69,96,0.9)")
                h_inner.append(f"Below mean (<{fmt(mn)})")
                h_mid.append(f"Track>{fmt(mn)}")
            else:
                c_inner.append("rgba(0,210,255,0.9)")
                c_mid.append("rgba(180,180,180,0.3)")
                h_inner.append(f"Track value {fmt(raw_t)}")
                h_mid.append("Above track, below mean")
            c_outer.append("rgba(255,255,255,0.04)")
            h_outer.append(f"Above {fmt(max(raw_t, mn))}")

        # Highlight borders on selected features
        sel_feats = {k.rsplit("_", 1)[0] for k in selected if "_" in k}
        hl_c = ["#ffd700" if f in sel_feats else "rgba(0,0,0,0)" for f in feats]
        hl_w = [2 if f in sel_feats else 0 for f in feats]

        traces = [
            go.Barpolar(r=r_inner, theta=labels, name="Inner",
                        marker=dict(color=c_inner, line=dict(color=hl_c, width=hl_w)),
                        hovertext=h_inner, hoverinfo="text"),
            go.Barpolar(r=r_mid,   theta=labels, name="Mid",
                        marker=dict(color=c_mid,   line=dict(color=hl_c, width=hl_w)),
                        hovertext=h_mid,   hoverinfo="text"),
            go.Barpolar(r=r_outer, theta=labels, name="Outer",
                        marker=dict(color=c_outer, line=dict(color=hl_c, width=hl_w)),
                        hovertext=h_outer, hoverinfo="text"),
        ]
        layout = dict(
            template="plotly_dark", paper_bgcolor="#0d0d1a", font=dict(color="#ccc"),
            title=dict(text=f"Radar: {radar_track}", font=dict(size=13)),
            polar=dict(barmode="stack",
                       radialaxis=dict(visible=True, showticklabels=False,
                                       range=[0, 1.0], gridcolor="#333355"),
                       angularaxis=dict(tickfont=dict(size=10), gridcolor="#333355"),
                       bgcolor="#111125"),
            showlegend=False,
            margin=dict(t=60, b=30, l=30, r=30),
            meta={"radar_selected": list(selected)},
        )
        return go.Figure(data=traces, layout=layout), f"Radar: {radar_track} | click bars to filter"

    # ── Similarity ────────────────────────────────────────────────────────────
    if mode == "similarity":
        if not sim_ref or not all([sim_xp, sim_xn, sim_yp, sim_yn]):
            return empty, "Select reference and four axes."
        if sim_ref not in ad.tracks:
            return empty, "Reference track not found."
        ridx = ad.tracks.index(sim_ref)
        dxp = ad.feat_array(sim_xp); dxn = ad.feat_array(sim_xn)
        dyp = ad.feat_array(sim_yp); dyn = ad.feat_array(sim_yn)
        valid = (np.isfinite(dxp) & np.isfinite(dxn)
                 & np.isfinite(dyp) & np.isfinite(dyn))
        idxs = np.where(valid)[0]
        nxp = norm01(dxp[valid]); nxn = norm01(dxn[valid])
        nyp = norm01(dyp[valid]); nyn = norm01(dyn[valid])
        ax = nxp - nxn; ay = nyp - nyn

        rpos = np.where(idxs == ridx)[0]
        if len(rpos) == 0:
            return empty, "Reference not in valid set."
        rx, ry = ax[rpos[0]], ay[rpos[0]]
        rel_x = ax - rx; rel_y = ay - ry
        dists = np.sqrt(rel_x**2 + rel_y**2)
        max_d = float(dists[idxs != ridx].max()) if (idxs != ridx).any() else 1.0

        mask_other = idxs != ridx
        ox = rel_x[mask_other]; oy = rel_y[mask_other]
        od = dists[mask_other]; oidxs = idxs[mask_other]
        hover = [f"<b>{ad.tracks[i]}</b><br>dist: {od[j]:.3f}"
                 for j, i in enumerate(oidxs)]
        traces = [
            go.Scatter(x=ox, y=oy, mode="markers",
                         marker=dict(color=od,
                                     colorscale=[[0, "#e94560"], [0.5, "#552244"],
                                                 [1, "#111133"]],
                                     cmin=0, cmax=max_d, size=6, opacity=0.75,
                                     colorbar=dict(title="dist", thickness=12)),
                         hovertext=hover, hoverinfo="text",
                         name="tracks", customdata=oidxs.tolist()),
            go.Scatter(x=[0], y=[0], mode="markers+text",
                       marker=dict(symbol="star", size=20, color="#ffd700",
                                   line=dict(color="white", width=1)),
                       text=[sim_ref], textposition="top center",
                       textfont=dict(color="#ffd700", size=10),
                       name=sim_ref, hoverinfo="text",
                       hovertext=[f"<b>{sim_ref}</b> (reference)"],
                       customdata=[ridx]),
        ]
        layout = dict(**_DARK,
                      title=dict(text=f"Similarity — {sim_ref}", font=dict(size=13)),
                      xaxis=dict(title=f"← {sim_xn}  |  {sim_xp} →",
                                 range=[-1.35, 1.35], zeroline=True,
                                 zerolinecolor="#555577"),
                      yaxis=dict(title=f"← {sim_yn}  |  {sim_yp} →",
                                 range=[-1.35, 1.35], zeroline=True,
                                 zerolinecolor="#555577", scaleanchor="x"),
                      showlegend=False,
                      shapes=[
                          dict(type="circle", x0=-0.5, y0=-0.5, x1=0.5, y1=0.5,
                               line=dict(color="#3a3a5a", width=1, dash="dot")),
                          dict(type="circle", x0=-1.0, y0=-1.0, x1=1.0, y1=1.0,
                               line=dict(color="#3a3a5a", width=1, dash="dot")),
                      ])
        return go.Figure(data=traces, layout=layout), f"Similarity: {sim_ref} | {len(oidxs)} tracks"

    # ── Classes ───────────────────────────────────────────────────────────────
    if mode == "classes":
        kx_c = class_kx
        ky_c = class_ky
        if not kx_c or not ky_c or not class_by:
            return empty, "Select X, Y, and class-by."
        if kx_c not in ad.num_cols or ky_c not in ad.num_cols:
            return empty, "Feature not found."
        if class_by not in ad.class_cols:
            return empty, f"{class_by} not available."
        dx = ad.feat_array(kx_c); dy = ad.feat_array(ky_c)
        raw_labels = ad.class_array(class_by)
        groups: dict[str, dict] = {}
        for i, (xi, yi) in enumerate(zip(dx, dy)):
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            lbl = parse_class_label(raw_labels[i]) or "(unknown)"
            g = groups.setdefault(lbl, {"x": [], "y": [], "idxs": []})
            g["x"].append(xi); g["y"].append(yi); g["idxs"].append(i)

        sorted_groups = sorted(groups.items(), key=lambda t: -len(t[1]["x"]))
        if len(sorted_groups) > 20:
            top = sorted_groups[:20]
            other: dict = {"x": [], "y": [], "idxs": []}
            for _, g in sorted_groups[20:]:
                other["x"] += g["x"]; other["y"] += g["y"]
                other["idxs"] += g["idxs"]
            top.append(("(other)", other))
            sorted_groups = top

        ux = f" [{FEATURE_UNITS[kx_c]}]" if kx_c in FEATURE_UNITS else ""
        uy = f" [{FEATURE_UNITS[ky_c]}]" if ky_c in FEATURE_UNITS else ""
        traces = []
        class_trend_opts = class_trend_opts or []
        for ci, (lbl, g) in enumerate(sorted_groups):
            x, y = np.array(g["x"]), np.array(g["y"])
            col = CLASS_COLORS[ci % len(CLASS_COLORS)]
            grp = f"class_{ci}"   # tie scatter + trend so legend toggle hides both
            hover = [f"<b>{ad.tracks[i]}</b><br>{kx_c}: {x[j]:.3f}<br>{ky_c}: {y[j]:.3f}"
                     f"<br>{class_by}: {lbl}"
                     for j, i in enumerate(g["idxs"])]
            traces.append(go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(color=col, size=5, opacity=0.7),
                hovertext=hover, hoverinfo="text",
                name=f"{lbl} ({len(x)})", customdata=g["idxs"],
                legendgroup=grp,
            ))
            if "trend" in class_trend_opts and len(x) >= 3:
                mx, my = x.mean(), y.mean()
                num = ((x-mx)*(y-my)).sum(); den = ((x-mx)**2).sum()
                if den > 1e-12:
                    slope = num/den; ic = my - slope*mx
                    xmn, xmx = x.min(), x.max()
                    traces.append(go.Scatter(
                        x=[xmn, xmx], y=[slope*xmn+ic, slope*xmx+ic],
                        mode="lines", line=dict(color=col, width=2, dash="dash"),
                        name=f"{lbl} trend", hoverinfo="skip", showlegend=False,
                        legendgroup=grp,
                    ))
        layout = _dark_layout(
                      title=dict(text=f"{kx_c} vs {ky_c} by {class_by}",
                                 font=dict(size=14)),
                      xaxis=dict(title=kx_c+ux), yaxis=dict(title=ky_c+uy),
                      showlegend=True,
                      legend=dict(x=1.02, y=1, bgcolor="rgba(15,52,96,0.8)",
                                  font=dict(size=10)),
                      margin=dict(t=46, b=48, l=58, r=180))
        total = sum(len(g["x"]) for _, g in sorted_groups)
        return go.Figure(data=traces, layout=layout), f"Classes: {class_by} | {len(sorted_groups)} groups | {total} tracks"

    return empty, f"Unknown mode: {mode}"


# ── Active track gold-ring overlay (Patch — does not clear selectedData) ─────
@callback(
    Output("dataset-graph", "figure", allow_duplicate=True),
    Input("active-track", "data"),
    State("view-mode", "value"),
    State("sel-x", "value"),
    State("sel-y", "value"),
    prevent_initial_call=True,
)
def highlight_active_track(active_track_data, mode, kx, ky):
    """Patch only the gold-ring overlay trace so lasso selection is preserved.
    Reads feature values directly from get_app_data() to avoid Dash's typed-array
    binary serialisation of figure State (which breaks list-indexing)."""
    if mode != "scatter":
        return no_update

    at = _safe_tidx((active_track_data or {}).get("track_idx"))
    patched = Patch()

    if at is None or not kx or not ky:
        patched["data"][-1]["x"] = []
        patched["data"][-1]["y"] = []
        return patched

    ad = get_app_data()
    if kx not in ad.num_cols or ky not in ad.num_cols:
        return no_update

    vx = ad.feat_array(kx)
    vy = ad.feat_array(ky)
    if at >= len(vx) or not np.isfinite(vx[at]) or not np.isfinite(vy[at]):
        patched["data"][-1]["x"] = []
        patched["data"][-1]["y"] = []
        return patched

    patched["data"][-1]["x"] = [float(vx[at])]
    patched["data"][-1]["y"] = [float(vy[at])]
    return patched


# ── Radar click → selection Store ────────────────────────────────────────────
@callback(
    Output("radar-selection", "data"),
    Input("dataset-graph", "clickData"),
    State("radar-selection", "data"),
    State("view-mode", "value"),
    State("sel-radar-track", "value"),
    prevent_initial_call=True,
)
def update_radar_selection(click_data, current_selected, mode, radar_track):
    if mode != "radar" or not click_data or not click_data.get("points"):
        return no_update
    ad = get_app_data()
    if not radar_track or radar_track not in ad.tracks:
        return no_update
    tidx = ad.tracks.index(radar_track)
    feats = [f for f in CURATED_FEATURES if f in ad.num_cols
             and np.isfinite(ad.feat_array(f)[tidx])]
    selected = set(current_selected or [])
    pt    = click_data["points"][0]
    lbl   = pt.get("theta", "")
    curve = pt.get("curveNumber", -1)
    feat_match = next((f for f in feats if _short_label(f) == lbl), None)
    if feat_match and 0 <= curve <= 2:
        key = f"{feat_match}_{curve}"
        if key in selected:
            selected.discard(key)
        else:
            selected.add(key)
    return list(selected)


# ── Track list sidebar — scatter lasso + histogram bar click ──────────────────
@callback(
    Output("track-list-body", "children"),
    Output("tl-title", "children"),
    Input("dataset-graph", "selectedData"),
    Input("dataset-graph", "clickData"),
    Input("dataset-graph", "restyleData"),
    Input("radar-selection", "data"),
    Input("tl-service", "value"),
    State("view-mode", "value"),
    State("sel-hist", "value"),
    State("hist-bins", "value"),
    State("dataset-graph", "figure"),
    State("sel-radar-track", "value"),
    prevent_initial_call=True,
)
def update_track_list(selected_data, click_data, restyle_data, radar_selected,
                      service, mode, hist_feat, hist_bins, fig, radar_track):
    ctx = dash.callback_context
    ad  = get_app_data()
    trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    _hint = lambda msg: html.Div(msg, style={"color": "#555", "fontSize": "11px",
                                             "padding": "8px"})

    # Histogram bar click
    if "clickData" in trig and mode == "histogram":
        if not click_data or not hist_feat or hist_feat not in ad.num_cols:
            return no_update, no_update
        vals  = ad.feat_array(hist_feat)
        nbins = max(4, min(64, int(hist_bins or 30)))
        fin   = vals[np.isfinite(vals)]
        vmin, vmax = float(fin.min()), float(fin.max())
        bsize = (vmax - vmin) / nbins
        pt    = click_data["points"][0]
        bin_lo = pt["x"] - bsize / 2
        bin_hi = pt["x"] + bsize / 2
        idxs = [i for i, v in enumerate(vals) if np.isfinite(v) and bin_lo <= v < bin_hi]
        return _render_track_list(idxs[:200], ad, service), f"Tracks ({len(idxs)})"

    # Parallel coords drag
    if "restyleData" in trig and mode == "parallel":
        if not restyle_data or not fig or not fig.get("data"):
            return no_update, no_update
        feats = [f for f in CURATED_FEATURES if f in ad.num_cols]
        dims  = fig["data"][0].get("dimensions", [])
        idxs  = []
        for ti in range(len(ad.tracks)):
            passes = True
            for di, dim in enumerate(dims):
                cr = dim.get("constraintrange")
                if cr is None:
                    continue
                feat = feats[di] if di < len(feats) else None
                if feat is None:
                    continue
                v = ad.feat_array(feat)[ti]
                if not np.isfinite(v):
                    passes = False; break
                lo, hi = (cr[0], cr[1]) if isinstance(cr[0], (int, float)) \
                    else (cr[0][0], cr[0][1])
                if not (lo <= float(v) <= hi):
                    passes = False; break
            if passes:
                idxs.append(ti)
        return _render_track_list(idxs[:300], ad, service), f"Tracks ({len(idxs)})"

    # Scatter lasso / box select
    if "selectedData" in trig and mode in ("scatter", "quadrant", "similarity", "classes"):
        if not selected_data or not selected_data.get("points"):
            return _hint("Lasso or box select to filter tracks."), "Tracks"
        idxs = [int(pt["customdata"]) for pt in selected_data["points"]
                if pt.get("customdata") is not None]
        return _render_track_list(idxs[:300], ad, service), f"Tracks ({len(idxs)})"

    # Radar segment click — defer to Store update (handled below)
    if "clickData" in trig and mode == "radar":
        return no_update, no_update

    # Radar selection Store changed — filter tracks similar in selected features
    if "radar-selection" in trig and mode == "radar":
        sel_feats = {k.rsplit("_", 1)[0] for k in (radar_selected or []) if "_" in k}
        if not sel_feats or not radar_track or radar_track not in ad.tracks:
            return _hint("Click radar bars to find similar tracks."), "Tracks"
        tidx = ad.tracks.index(radar_track)
        idxs = []
        for i in range(len(ad.tracks)):
            if i == tidx:
                continue
            match = True
            for f in sel_feats:
                if f not in ad.num_cols:
                    continue
                ref_val = float(ad.feat_array(f)[tidx])
                val = float(ad.feat_array(f)[i])
                if not np.isfinite(val):
                    match = False; break
                all_v = ad.feat_array(f)
                sigma = float(all_v[np.isfinite(all_v)].std())
                if abs(val - ref_val) > sigma:
                    match = False; break
            if match:
                idxs.append(i)
        return _render_track_list(idxs[:300], ad, service), f"Similar ({len(idxs)})"

    # tl-service change — rebuild existing list if any trigger fired
    if "tl-service" in trig:
        return no_update, no_update

    return no_update, no_update


# ── NN panel for Similarity mode ──────────────────────────────────────────────
@callback(
    Output("nn-panel", "children"),
    Input("view-mode", "value"),
    Input("sel-sim-ref", "value"),
    Input("sim-xp", "value"), Input("sim-xn", "value"),
    Input("sim-yp", "value"), Input("sim-yn", "value"),
    Input("dataset-graph", "selectedData"),
    prevent_initial_call=False,
)
def update_nn_panel(mode, ref_name, kxp, kxn, kyp, kyn, selected_data):
    if mode != "similarity":
        return html.Div()
    ad = get_app_data()
    if not all([kxp, kxn, kyp, kyn]):
        return html.Div()

    dxp = ad.feat_array(kxp); dxn = ad.feat_array(kxn)
    dyp = ad.feat_array(kyp); dyn = ad.feat_array(kyn)
    valid = (np.isfinite(dxp) & np.isfinite(dxn)
             & np.isfinite(dyp) & np.isfinite(dyn))
    idxs = np.where(valid)[0]
    idxs_set = set(idxs.tolist())
    nxp = norm01(dxp[valid]); nxn = norm01(dxn[valid])
    nyp = norm01(dyp[valid]); nyn = norm01(dyn[valid])
    ax = nxp - nxn; ay = nyp - nyn

    # Determine reference: selection centroid (≥2 tracks) or single ref track
    sel_tidxs = []
    if selected_data and selected_data.get("points"):
        sel_tidxs = [int(pt["customdata"]) for pt in selected_data["points"]
                     if pt.get("customdata") is not None
                     and int(pt["customdata"]) in idxs_set]

    if len(sel_tidxs) >= 2:
        # Centroid of selected tracks in the 4-axis feature space
        sel_pos = [int(np.where(idxs == s)[0][0])
                   for s in sel_tidxs if len(np.where(idxs == s)[0]) > 0]
        if not sel_pos:
            return html.Div()
        rx = float(np.mean([ax[p] for p in sel_pos]))
        ry = float(np.mean([ay[p] for p in sel_pos]))
        ref_label   = f"{len(sel_tidxs)} selected"
        exclude_set = set(sel_tidxs)
    else:
        # Single reference track
        if not ref_name or ref_name not in ad.tracks:
            return html.Div()
        ridx = ad.tracks.index(ref_name)
        rpos = np.where(idxs == ridx)[0]
        if len(rpos) == 0:
            return html.Div()
        rx, ry = ax[rpos[0]], ay[rpos[0]]
        ref_label   = ref_name
        exclude_set = {ridx}

    dists = np.sqrt((ax - rx)**2 + (ay - ry)**2)
    pairs = [(float(dists[i]), int(idxs[i]))
             for i in range(len(idxs)) if idxs[i] not in exclude_set]
    pairs.sort(key=lambda t: t[0])
    pairs = pairs[:20]

    nn_items = []
    for rank, (dist, tidx) in enumerate(pairs):
        nn_items.append(html.Div([
            html.Span(f"{rank+1}.", style={"fontSize": "9px", "color": "#555",
                                            "flexShrink": "0"}),
            html.Span(ad.tracks[tidx], className="tl-item",
                      style={"flex": "1", "fontSize": "10px", "overflow": "hidden",
                             "textOverflow": "ellipsis", "whiteSpace": "nowrap"},
                      title=ad.tracks[tidx]),
            html.Span(f"{dist:.3f}", style={"fontSize": "9px", "color": "#555",
                                             "flexShrink": "0"}),
        ], className="nn-row", style={"display": "flex", "gap": "4px"},
           id={"type": "tl-item", "index": int(tidx)}))

    title = f"Similar to {ref_label}" if len(sel_tidxs) < 2 else f"Similar to avg({ref_label})"
    return html.Div([
        html.Div(title, style={"fontSize": "11px", "fontWeight": "bold",
                               "color": "#e94560" if len(sel_tidxs) >= 2 else "#ccc",
                               "marginBottom": "4px"}),
        html.Div(nn_items, style={"overflowY": "auto",
                                   "maxHeight": "calc(50vh - 100px)"}),
    ])


# ── Active track Store — click (slot A/B via click-register) + hover ─────────
@callback(
    Output("active-track", "data"),
    Input("click-register", "data"),
    Input("dataset-graph", "hoverData"),
    State("active-track", "data"),
    State("autoplay-hover", "data"),
    prevent_initial_call=True,
)
def update_active_track(click_reg, hover_data, current, autoplay_hover):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update
    ad   = get_app_data()
    trig = ctx.triggered[0]["prop_id"]

    if "click-register" in trig and click_reg:
        tidx = _safe_tidx(click_reg.get("track_idx"))
        slot = click_reg.get("slot", "a")
        if tidx is not None and 0 <= tidx < len(ad.tracks):
            state = dict(current or {})
            if slot == "b":
                state.update({"track_b": ad.tracks[tidx], "track_b_idx": tidx})
            else:
                state.update({"track": ad.tracks[tidx], "track_idx": tidx, "slot": "a"})
            return state

    if "hoverData" in trig and hover_data and autoplay_hover:
        pt = hover_data["points"][0]
        tidx = _safe_tidx(pt.get("customdata"))
        if tidx is not None and 0 <= tidx < len(ad.tracks):
            state = dict(current or {})
            state.update({"track": ad.tracks[tidx], "track_idx": tidx, "slot": "a"})
            return state

    return no_update


@callback(
    Output("active-track", "data", allow_duplicate=True),
    Input({"type": "tl-item", "index": dash.ALL}, "n_clicks"),
    State("active-track", "data"),
    prevent_initial_call=True,
)
def track_list_click(n_clicks_list, current):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n for n in (n_clicks_list or []) if n):
        return no_update
    trig_id = ctx.triggered[0]["prop_id"]
    raw  = trig_id.split(".")[0]
    tidx = json.loads(raw)["index"]
    ad   = get_app_data()
    state = dict(current or {})
    state.update({"track": ad.tracks[tidx], "track_idx": tidx, "slot": "a"})
    return state


@callback(
    Output("active-track", "data", allow_duplicate=True),
    Input({"type": "tl-item-b", "index": dash.ALL}, "n_clicks"),
    State("active-track", "data"),
    prevent_initial_call=True,
)
def track_list_b_click(n_clicks_list, current):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n for n in (n_clicks_list or []) if n):
        return no_update
    trig_id = ctx.triggered[0]["prop_id"]
    raw  = trig_id.split(".")[0]
    tidx = json.loads(raw)["index"]
    ad   = get_app_data()
    state = dict(current or {})
    state.update({"track_b": ad.tracks[tidx], "track_b_idx": tidx})
    return state


# ── Heatmap cell click → switch to Scatter with those two features ────────────
@callback(
    Output("view-mode", "value", allow_duplicate=True),
    Output("sel-x",     "value", allow_duplicate=True),
    Output("sel-y",     "value", allow_duplicate=True),
    Input("dataset-graph", "clickData"),
    State("view-mode", "value"),
    prevent_initial_call=True,
)
def heatmap_to_scatter(click_data, mode):
    if mode != "heatmap" or not click_data or not click_data.get("points"):
        return no_update, no_update, no_update
    pt = click_data["points"][0]
    fx = pt.get("x")
    fy = pt.get("y")
    ad = get_app_data()
    if not fx or not fy or fx == fy:
        return no_update, no_update, no_update
    if fx not in ad.num_cols or fy not in ad.num_cols:
        return no_update, no_update, no_update
    return "scatter", fx, fy


# ── Pattern-search for track dropdowns ───────────────────────────────────────
@callback(
    Output("sel-radar-track", "options"),
    Output("sel-sim-ref",     "options"),
    Input("sel-radar-track", "search_value"),
    Input("sel-sim-ref",     "search_value"),
    prevent_initial_call=False,
)
def filter_track_dropdowns(radar_q, sim_q):
    ad = get_app_data()
    ctx = dash.callback_context
    trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    if "radar" in trig:
        return ad.track_options(radar_q or ""), no_update
    if "sim-ref" in trig:
        return no_update, ad.track_options(sim_q or "")
    opts = ad.track_options()
    return opts, opts


# ── Helpers ───────────────────────────────────────────────────────────────────
# ── Autoplay on hover ─────────────────────────────────────────────────────────
@callback(
    Output("player-cmd", "data", allow_duplicate=True),
    Input("dataset-graph", "hoverData"),
    State("autoplay-hover", "data"),
    State("pos-slider-a",   "value"),
    State("player-options", "value"),
    prevent_initial_call=True,
)
def autoplay_on_hover(hover_data, autoplay_enabled, pos, opts):
    if not autoplay_enabled or not hover_data:
        return no_update
    pt = hover_data["points"][0]
    raw = pt.get("customdata")
    if raw is None:
        return no_update
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if raw is None:
        return no_update
    ad    = get_app_data()
    track = ad.tracks[int(raw)]
    smart_loop = "smart_loop" in (opts or [])
    ls, le = (0, -1) if smart_loop else (None, None)
    url   = build_decode_url(track, str(pos or 0.5), smart_loop=smart_loop)
    return {"action": "play", "url": url, "loop_start": ls, "loop_end": le,
            "from_hover": True}


_SERVICE_URL = {
    "spotify":     ("spotify_id",     "https://open.spotify.com/track/{}"),
    "tidal":       ("tidal_id",       "https://tidal.com/browse/track/{}"),
    "musicbrainz": ("musicbrainz_id", "https://musicbrainz.org/recording/{}"),
}

def _render_track_list(idxs: list[int], ad, service: str = "tidal") -> html.Div:
    if not idxs:
        return html.Div("No tracks match.",
                        style={"color": "#555", "fontSize": "11px", "padding": "8px"})
    meta_col, url_tmpl = _SERVICE_URL.get(service or "tidal", ("tidal_id", ""))
    _btn_b_style = {
        "background": "none", "border": "1px solid #224", "color": "#00d2ff",
        "fontSize": "9px", "cursor": "pointer", "padding": "0 3px",
        "borderRadius": "2px", "flexShrink": "0", "lineHeight": "14px",
        "marginLeft": "2px",
    }
    items = []
    for idx in idxs:
        name = ad.tracks[idx]
        meta = ad.meta_val(meta_col, idx)
        link = None
        if meta and url_tmpl:
            link = html.A("↗", href=url_tmpl.format(meta), target="_blank",
                          title=f"Open in {service}",
                          style={"color": "#555", "fontSize": "10px",
                                 "marginLeft": "4px", "textDecoration": "none",
                                 "flexShrink": "0"})
        # Span (slot A) + optional link + B button — siblings so B click doesn't bubble to A
        row = html.Div([
            html.Span(name,
                      className="tl-item",
                      id={"type": "tl-item", "index": idx},
                      n_clicks=0, title=name,
                      style={"flex": "1", "overflow": "hidden",
                             "textOverflow": "ellipsis", "whiteSpace": "nowrap",
                             "minWidth": 0, "cursor": "pointer"}),
            *(([link]) if link else []),
            html.Button("B", id={"type": "tl-item-b", "index": idx},
                        n_clicks=0, title="Load in slot B",
                        style=_btn_b_style),
        ], style={"display": "flex", "alignItems": "center",
                  "justifyContent": "space-between", "gap": "2px"})
        items.append(row)
    return html.Div(items)


# ── Help panel toggle ─────────────────────────────────────────────────────────
@callback(
    Output("help-panel", "style"),
    Input("btn-help",       "n_clicks"),
    Input("btn-help-close", "n_clicks"),
    State("help-panel", "style"),
    prevent_initial_call=True,
)
def toggle_help(open_n, close_n, style):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update
    trig = ctx.triggered[0]["prop_id"]
    base = {k: v for k, v in (style or {}).items() if k != "display"}
    if "close" in trig:
        return {**base, "display": "none"}
    current = (style or {}).get("display", "none")
    return {**base, "display": "none" if current != "none" else "",
            "background": "#0f1535", "border": "1px solid #2a2a4a",
            "padding": "12px 16px", "fontSize": "11px", "color": "#ccc"}


def _safe_tidx(val):
    """Coerce a track-index value (possibly BigInt-serialized list) to int or None."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _short_label(feat: str) -> str:
    return (feat.replace("rms_energy_", "")
                .replace("spectral_", "spec_")
                .replace("_probability", "_p"))


def _parcoords_label(feat: str) -> str:
    """Short axis label for parallel coords (avoids overlap)."""
    return (feat
            .replace("rms_energy_", "rms_")
            .replace("spectral_",   "sp_")
            .replace("_probability", "_p")
            .replace("production_", "prod_")
            .replace("content_",    "")
            .replace("reverberation", "reverb")
            .replace("danceability",  "dance")
            .replace("instrumental",  "instr")
            .replace("lufs_",         "L_")
            )
