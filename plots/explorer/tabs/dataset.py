"""Tab 1 — Dataset Explorer: 8-mode scatter + sidebar."""
from __future__ import annotations

import json

import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import numpy as np
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.explorer.data import (
    get_app_data, CURATED_FEATURES, FEATURE_UNITS, FEATURE_GROUPS,
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

CLASS_COLORS = [
    '#e94560', '#4cd137', '#00d2ff', '#fbc531', '#9c88ff', '#ff79c6', '#ff9f43',
    '#0097e6', '#bdc581', '#fd7272', '#5f27cd', '#55efc4', '#81ecec', '#ff7675',
    '#a29bfe', '#e1b12c', '#00cec9', '#e84393', '#badc58', '#c8d6e5',
]


# ── Layout ────────────────────────────────────────────────────────────────────
def layout() -> html.Div:
    ad = get_app_data()
    feat_opts  = [{"label": f, "value": f} for f in sorted(ad.num_cols)]
    class_opts = [{"label": c, "value": c} for c in ad.class_cols]
    track_opts = ad.track_options()

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
        ], className="controls-bar"),

        # ── Mode-specific controls ───────────────────────────────────────────
        html.Div(id="mode-controls",
                 children=_scatter_controls(feat_opts, class_opts)),

        # ── Main area: plot + sidebar ────────────────────────────────────────
        html.Div([
            html.Div([
                dcc.Graph(
                    id="dataset-graph",
                    style={"height": "calc(100vh - 260px)"},
                    config={"displayModeBar": True, "scrollZoom": True},
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
def _scatter_controls(feat_opts, class_opts):
    return html.Div([
        _label_group("X axis",   dcc.Dropdown(id="sel-x", options=feat_opts,
                                              value="bpm", clearable=False,
                                              style={"width": "180px"})),
        _label_group("Y axis",   dcc.Dropdown(id="sel-y", options=feat_opts,
                                              value="brightness", clearable=False,
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
    ], className="controls-bar", style={"gap": "12px"})


def _label_group(label: str, child) -> html.Div:
    return html.Div([
        html.Label(label, style={"fontSize": "10px", "color": "#888",
                                 "textTransform": "uppercase"}),
        child,
    ], style={"display": "flex", "flexDirection": "column", "gap": "2px"})


# ── Mode → controls switcher ──────────────────────────────────────────────────
@callback(Output("mode-controls", "children"),
          Input("view-mode", "value"),
          prevent_initial_call=False)
def update_mode_controls(mode: str):
    ad = get_app_data()
    feat_opts  = [{"label": f, "value": f} for f in sorted(ad.num_cols)]
    class_opts = [{"label": c, "value": c} for c in ad.class_cols]
    track_opts = ad.track_options()

    if mode == "scatter":
        return _scatter_controls(feat_opts, class_opts)

    if mode == "quadrant":
        return html.Div([
            _label_group("X+", dcc.Dropdown(id="sel-xp", options=feat_opts,
                                            value="brightness", clearable=False,
                                            style={"width": "160px"})),
            _label_group("X−", dcc.Dropdown(id="sel-xn", options=feat_opts,
                                            value="roughness", clearable=False,
                                            style={"width": "160px"})),
            _label_group("Y+", dcc.Dropdown(id="sel-yp", options=feat_opts,
                                            value="danceability", clearable=False,
                                            style={"width": "160px"})),
            _label_group("Y−", dcc.Dropdown(id="sel-yn", options=feat_opts,
                                            value="atonality", clearable=False,
                                            style={"width": "160px"})),
        ], className="controls-bar")

    if mode == "histogram":
        return html.Div([
            _label_group("Feature", dcc.Dropdown(id="sel-hist", options=feat_opts,
                                                  value="bpm", clearable=False,
                                                  style={"width": "200px"})),
            _label_group("Bins", dcc.Input(id="hist-bins", type="number",
                                           value=30, min=4, max=64,
                                           style={"width": "60px"})),
        ], className="controls-bar")

    if mode == "radar":
        return html.Div([
            _label_group("Track", dcc.Dropdown(
                id="sel-radar-track",
                options=track_opts,
                value=track_opts[0]["value"] if track_opts else None,
                clearable=False, style={"width": "320px"},
                placeholder="Search track…")),
        ], className="controls-bar")

    if mode == "heatmap":
        group_opts = [{"label": g, "value": g}
                      for g in list(FEATURE_GROUPS.keys()) + ["All"]]
        return html.Div([
            _label_group("Feature group",
                         dcc.Dropdown(id="sel-heatmap-group", options=group_opts,
                                      value="All", clearable=False,
                                      style={"width": "180px"})),
        ], className="controls-bar")

    if mode == "parallel":
        return html.Div([
            html.Span("Curated features | Drag axis bands to filter",
                      style={"fontSize": "11px", "color": "#666"}),
        ], className="controls-bar")

    if mode == "similarity":
        return html.Div([
            _label_group("Reference", dcc.Dropdown(
                id="sel-sim-ref",
                options=track_opts,
                value=track_opts[0]["value"] if track_opts else None,
                clearable=False, style={"width": "280px"},
                placeholder="Search reference…")),
            _label_group("X+", dcc.Dropdown(id="sim-xp", options=feat_opts,
                                            value="brightness", clearable=False,
                                            style={"width": "140px"})),
            _label_group("X−", dcc.Dropdown(id="sim-xn", options=feat_opts,
                                            value="roughness", clearable=False,
                                            style={"width": "140px"})),
            _label_group("Y+", dcc.Dropdown(id="sim-yp", options=feat_opts,
                                            value="danceability", clearable=False,
                                            style={"width": "140px"})),
            _label_group("Y−", dcc.Dropdown(id="sim-yn", options=feat_opts,
                                            value="atonality", clearable=False,
                                            style={"width": "140px"})),
        ], className="controls-bar")

    if mode == "classes":
        return html.Div([
            _label_group("X axis",   dcc.Dropdown(id="sel-x", options=feat_opts,
                                                   value="bpm", clearable=False,
                                                   style={"width": "160px"})),
            _label_group("Y axis",   dcc.Dropdown(id="sel-y", options=feat_opts,
                                                   value="brightness", clearable=False,
                                                   style={"width": "160px"})),
            _label_group("Class by", dcc.Dropdown(
                id="sel-class-by",
                options=class_opts,
                value=class_opts[0]["value"] if class_opts else None,
                clearable=False, style={"width": "180px"})),
            dcc.Checklist(id="chk-class-trend",
                          options=[{"label": "Per-class trend", "value": "trend"}],
                          value=[], inline=True, style={"fontSize": "11px"}),
        ], className="controls-bar")

    return html.Div(className="controls-bar")


# ── Main graph callback — single dispatcher for all 8 modes ──────────────────
@callback(
    Output("dataset-graph", "figure"),
    Output("info-bar", "children"),
    # mode
    Input("view-mode", "value"),
    # scatter / classes
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
    # classes
    Input("sel-class-by",   "value"),
    Input("chk-class-trend", "value"),
    # cluster overlay
    Input("cluster-highlight", "data"),
    Input("chk-show-clusters", "value"),
    # radar click state (for segment selection)
    Input("dataset-graph", "clickData"),
    State("dataset-graph", "figure"),
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
    class_by, class_trend_opts,
    # cluster
    cluster_data, show_clusters,
    # radar click
    click_data, current_fig,
):
    ctx = dash.callback_context
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
                go.Scattergl(x=x[~is_out], y=y[~is_out], mode="markers",
                             marker=dict(color="#e94560", size=5, opacity=0.5),
                             hovertext=[hover[j] for j in np.where(~is_out)[0]],
                             hoverinfo="text", name=f"normal ({(~is_out).sum()})",
                             customdata=idxs[~is_out]),
                go.Scattergl(x=x[is_out], y=y[is_out], mode="markers",
                             marker=dict(color="#ffd700", size=9, opacity=0.85),
                             hovertext=[hover[j] for j in np.where(is_out)[0]],
                             hoverinfo="text", name=f"outlier ({is_out.sum()})",
                             customdata=idxs[is_out]),
            ]
        elif colour and colour in ad.num_cols:
            cvals = ad.feat_array(colour)[valid]
            traces = [go.Scattergl(x=x, y=y, mode="markers",
                                   marker=dict(color=cvals, colorscale="Viridis",
                                               showscale=True, size=5, opacity=0.7,
                                               colorbar=dict(title=colour, thickness=14)),
                                   hovertext=hover, hoverinfo="text",
                                   name="tracks", customdata=idxs)]
        else:
            traces = [go.Scattergl(x=x, y=y, mode="markers",
                                   marker=dict(color="#e94560", size=5, opacity=0.5),
                                   hovertext=hover, hoverinfo="text",
                                   name="tracks", customdata=idxs)]

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
        trace = go.Scattergl(x=x, y=y, mode="markers",
                             marker=dict(color="#e94560", size=6, opacity=0.5),
                             hovertext=hover, hoverinfo="text", customdata=idxs)
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
        bsize = (vmax - vmin) / nbins
        u = f" [{FEATURE_UNITS[hist_feat]}]" if hist_feat in FEATURE_UNITS else ""
        trace = go.Histogram(
            x=vals, xbins=dict(start=vmin, end=vmax, size=bsize),
            marker=dict(color="#e94560", opacity=0.75,
                        line=dict(color="#c03050", width=0.5)),
            name=hist_feat,
        )
        layout = dict(**_DARK,
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
        layout = dict(**_DARK,
                      title=dict(text=f"Pearson r — {n} features", font=dict(size=14)),
                      xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                      yaxis=dict(tickfont=dict(size=9)),
                      margin=dict(t=46, b=80, l=80, r=20))
        return go.Figure(data=[trace], layout=layout), f"Pearson r | {n} features | click cell → Scatter"

    # ── Parallel ──────────────────────────────────────────────────────────────
    if mode == "parallel":
        feats = [f for f in CURATED_FEATURES if f in ad.num_cols]
        dims = []
        for f in feats:
            arr = ad.feat_array(f).tolist()
            dims.append(dict(label=f, values=arr))
        color_vals = ad.feat_array(feats[0]).tolist() if feats else []
        trace = go.Parcoords(
            line=dict(
                color=color_vals,
                colorscale=[[0, "#0d0520"], [0.33, "#2e0d5c"],
                            [0.66, "#6b2ca0"], [1, "#a855d4"]],
                showscale=True,
                colorbar=dict(title=feats[0] if feats else "", thickness=12),
                opacity=0.25,
            ),
            dimensions=dims,
        )
        layout = dict(template="plotly_dark", paper_bgcolor="#0d0d1a",
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

        # Recover segment-selection state from current figure meta
        selected: set[str] = set()
        trig_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        if (current_fig and current_fig.get("layout", {}).get("meta")
                and "dataset-graph.clickData" in trig_id):
            selected = set(current_fig["layout"]["meta"].get("radar_selected", []))
            if click_data and click_data["points"]:
                pt = click_data["points"][0]
                lbl   = pt.get("theta", "")
                curve = pt.get("curveNumber", -1)
                feat_match = next(
                    (f for f in feats if _short_label(f) == lbl), None)
                if feat_match and 0 <= curve <= 2:
                    key = f"{feat_match}_{curve}"
                    if key in selected:
                        selected.discard(key)
                    else:
                        selected.add(key)

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

        traces = [
            go.Barpolar(r=r_inner, theta=labels, name="Inner",
                        marker=dict(color=c_inner), hovertext=h_inner, hoverinfo="text"),
            go.Barpolar(r=r_mid,   theta=labels, name="Mid",
                        marker=dict(color=c_mid),   hovertext=h_mid,   hoverinfo="text"),
            go.Barpolar(r=r_outer, theta=labels, name="Outer",
                        marker=dict(color=c_outer), hovertext=h_outer, hoverinfo="text"),
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
            go.Scattergl(x=ox, y=oy, mode="markers",
                         marker=dict(color=od,
                                     colorscale=[[0, "#e94560"], [0.5, "#552244"],
                                                 [1, "#111133"]],
                                     cmin=0, cmax=max_d, size=6, opacity=0.75,
                                     colorbar=dict(title="dist", thickness=12)),
                         hovertext=hover, hoverinfo="text",
                         name="tracks", customdata=oidxs),
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
        if not kx or not ky or not class_by:
            return empty, "Select X, Y, and class-by."
        if kx not in ad.num_cols or ky not in ad.num_cols:
            return empty, "Feature not found."
        if class_by not in ad.class_cols:
            return empty, f"{class_by} not available."
        dx = ad.feat_array(kx); dy = ad.feat_array(ky)
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

        ux = f" [{FEATURE_UNITS[kx]}]" if kx in FEATURE_UNITS else ""
        uy = f" [{FEATURE_UNITS[ky]}]" if ky in FEATURE_UNITS else ""
        traces = []
        class_trend_opts = class_trend_opts or []
        for ci, (lbl, g) in enumerate(sorted_groups):
            x, y = np.array(g["x"]), np.array(g["y"])
            col = CLASS_COLORS[ci % len(CLASS_COLORS)]
            hover = [f"<b>{ad.tracks[i]}</b><br>{kx}: {x[j]:.3f}<br>{ky}: {y[j]:.3f}"
                     f"<br>{class_by}: {lbl}"
                     for j, i in enumerate(g["idxs"])]
            traces.append(go.Scattergl(
                x=x, y=y, mode="markers",
                marker=dict(color=col, size=5, opacity=0.7),
                hovertext=hover, hoverinfo="text",
                name=f"{lbl} ({len(x)})", customdata=g["idxs"],
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
                    ))
        layout = dict(**_DARK,
                      title=dict(text=f"{kx} vs {ky} by {class_by}",
                                 font=dict(size=14)),
                      xaxis=dict(title=kx+ux), yaxis=dict(title=ky+uy),
                      showlegend=True,
                      legend=dict(x=1.02, y=1, bgcolor="rgba(15,52,96,0.8)",
                                  font=dict(size=10)),
                      margin=dict(t=46, b=48, l=58, r=180))
        total = sum(len(g["x"]) for _, g in sorted_groups)
        return go.Figure(data=traces, layout=layout), f"Classes: {class_by} | {len(sorted_groups)} groups | {total} tracks"

    return empty, f"Unknown mode: {mode}"


# ── Track list sidebar — scatter lasso + histogram bar click ──────────────────
@callback(
    Output("track-list-body", "children"),
    Output("tl-title", "children"),
    Input("dataset-graph", "selectedData"),
    Input("dataset-graph", "clickData"),
    Input("dataset-graph", "restyleData"),
    State("view-mode", "value"),
    State("sel-hist", "value"),
    State("hist-bins", "value"),
    State("dataset-graph", "figure"),
    prevent_initial_call=True,
)
def update_track_list(selected_data, click_data, restyle_data, mode,
                      hist_feat, hist_bins, fig):
    ctx = dash.callback_context
    ad  = get_app_data()
    trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

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
        return _render_track_list(idxs[:200], ad), f"Tracks ({len(idxs)})"

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
        return _render_track_list(idxs[:300], ad), f"Tracks ({len(idxs)})"

    # Scatter lasso / box select
    if "selectedData" in trig and mode in ("scatter", "quadrant", "similarity", "classes"):
        if not selected_data or not selected_data.get("points"):
            return (html.Div("Lasso or box select to filter tracks.",
                             style={"color": "#555", "fontSize": "11px",
                                    "padding": "8px"}), "Tracks")
        idxs = [int(pt["customdata"]) for pt in selected_data["points"]
                if pt.get("customdata") is not None]
        return _render_track_list(idxs[:300], ad), f"Tracks ({len(idxs)})"

    # Radar segment filter — track list populated inside update_graph via no_update;
    # The radar no longer updates track list here (it's returned from the main callback
    # via the radar branch when click_data fires). But radar doesn't have a second
    # callback output for track-list. Handle here when radar click fires:
    if "clickData" in trig and mode == "radar":
        return no_update, no_update

    return no_update, no_update


# ── NN panel for Similarity mode ──────────────────────────────────────────────
@callback(
    Output("nn-panel", "children"),
    Input("view-mode", "value"),
    Input("sel-sim-ref", "value"),
    Input("sim-xp", "value"), Input("sim-xn", "value"),
    Input("sim-yp", "value"), Input("sim-yn", "value"),
    prevent_initial_call=False,
)
def update_nn_panel(mode, ref_name, kxp, kxn, kyp, kyn):
    if mode != "similarity":
        return html.Div()
    ad = get_app_data()
    if not ref_name or ref_name not in ad.tracks or not all([kxp, kxn, kyp, kyn]):
        return html.Div()
    ridx = ad.tracks.index(ref_name)
    dxp = ad.feat_array(kxp); dxn = ad.feat_array(kxn)
    dyp = ad.feat_array(kyp); dyn = ad.feat_array(kyn)
    valid = (np.isfinite(dxp) & np.isfinite(dxn)
             & np.isfinite(dyp) & np.isfinite(dyn))
    idxs = np.where(valid)[0]
    nxp = norm01(dxp[valid]); nxn = norm01(dxn[valid])
    nyp = norm01(dyp[valid]); nyn = norm01(dyn[valid])
    ax = nxp - nxn; ay = nyp - nyn
    rpos = np.where(idxs == ridx)[0]
    if len(rpos) == 0:
        return html.Div()
    rx, ry = ax[rpos[0]], ay[rpos[0]]
    dists = np.sqrt((ax - rx)**2 + (ay - ry)**2)
    mask_other = idxs != ridx
    od = dists[mask_other]; oidxs = idxs[mask_other]
    sorted_pairs = sorted(zip(od, oidxs), key=lambda t: t[0])[:20]
    nn_items = []
    for rank, (dist, tidx) in enumerate(sorted_pairs):
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
    return html.Div([
        html.Div("Most Similar", style={"fontSize": "11px", "fontWeight": "bold",
                                         "marginBottom": "4px"}),
        html.Div(nn_items, style={"overflowY": "auto",
                                   "maxHeight": "calc(50vh - 100px)"}),
    ])


# ── Active track Store — click and hover ──────────────────────────────────────
@callback(
    Output("active-track", "data"),
    Input("dataset-graph", "clickData"),
    Input("dataset-graph", "hoverData"),
    State("active-track", "data"),
    State("autoplay-hover", "data"),
    prevent_initial_call=True,
)
def update_active_track(click_data, hover_data, current, autoplay_hover):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update
    ad   = get_app_data()
    trig = ctx.triggered[0]["prop_id"]

    if "clickData" in trig and click_data:
        pt = click_data["points"][0]
        tidx = pt.get("customdata")
        if tidx is not None:
            tidx = int(tidx)
            state = dict(current or {})
            state.update({"track": ad.tracks[tidx], "track_idx": tidx, "slot": "a"})
            return state

    if "hoverData" in trig and hover_data and autoplay_hover:
        pt = hover_data["points"][0]
        tidx = pt.get("customdata")
        if tidx is not None:
            tidx = int(tidx)
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
    Output("view-mode", "value"),
    Input({"type": "tl-item", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def track_list_click_switch_mode(n_clicks_list):
    if not any(n for n in (n_clicks_list or []) if n):
        return no_update
    return "similarity"


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
    tidx = pt.get("customdata")
    if tidx is None:
        return no_update
    ad    = get_app_data()
    track = ad.tracks[int(tidx)]
    smart_loop = "smart_loop" in (opts or [])
    url   = build_decode_url(track, str(pos or 0.5), smart_loop=smart_loop)
    hover_x = hover_data.get("event", {}).get("clientX") if isinstance(hover_data, dict) else None
    hover_y = hover_data.get("event", {}).get("clientY") if isinstance(hover_data, dict) else None
    return {"action": "play", "url": url, "loop_start": None, "loop_end": None,
            "from_hover": True, "hover_x": hover_x, "hover_y": hover_y}


def _render_track_list(idxs: list[int], ad) -> html.Div:
    if not idxs:
        return html.Div("No tracks match.",
                        style={"color": "#555", "fontSize": "11px", "padding": "8px"})
    items = []
    for idx in idxs:
        name = ad.tracks[idx]
        items.append(
            html.Div(name, className="tl-item",
                     id={"type": "tl-item", "index": idx}, title=name)
        )
    return html.Div(items)


def _short_label(feat: str) -> str:
    return (feat.replace("rms_energy_", "")
                .replace("spectral_", "spec_")
                .replace("_probability", "_p"))
