"""Tab 2 — Analysis: ported from plots/latent_analysis/app.py."""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback, no_update

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.explorer.data import get_analysis, get_app_data
from plots.latent_analysis.config import (
    FEATURE_GROUPS, LATENT_DIM, POSTER_CLAMP,
    TEMPORAL_FEATURE_NAMES,
)

_TAB_STYLE        = {"padding": "6px 12px", "color": "#888", "fontSize": "12px"}
_TAB_ACTIVE_STYLE = {"padding": "6px 12px", "backgroundColor": "#7eb8f7",
                     "color": "#0d0d1a", "fontWeight": "700", "fontSize": "12px"}

_HELP_CORR = (
    "Rows = 64 VAE latent dimensions; columns = MIR features. "
    "Colour = Pearson r (red positive, blue negative). "
    "Sort by 'Feature loading' to surface most expressive dims. "
    "Click any cell to see a scatter plot of that dim vs. feature."
)
_HELP_POSTERS = (
    "Each cell is one of the 64 VAE latent dimensions in an 8×8 grid. "
    "Colour = Pearson r with the selected feature. "
    "Red = dim rises when feature rises; blue = inverse."
)
_HELP_PCA = (
    "PCA compresses 64 latent dims into principal components. "
    "Scatter shows every crop projected onto PC axes, coloured by a MIR feature."
)
_HELP_TEMPORAL = (
    "Each latent frame = 2048 audio samples @ 44 100 Hz ≈ 46 ms. "
    "Latent dim values over time alongside frame-level audio features."
)
_HELP_XCORR = (
    "64×64 Pearson r matrix between every pair of latent dims, averaged over crops. "
    "Sorted by Ward cluster. Tight clusters encode related perceptual qualities."
)
_HELP_CLUSTERS = (
    "Ward clusters from the cross-correlation analysis. "
    "Click 'Highlight in Dataset' to overlay cluster dims on the scatter."
)


def layout() -> html.Div:
    npz = get_analysis()
    d01 = npz.get("d01")
    feat_names = list(d01["feature_names"]) if d01 else []
    ALL_GROUPS = list(FEATURE_GROUPS.keys()) + ["All"]
    return html.Div([
        dcc.Tabs(id="analysis-subtabs", value="corr", children=[
            dcc.Tab(label="Correlation Matrix", value="corr",
                    style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
            dcc.Tab(label="Feature Posters",    value="posters",
                    style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
            dcc.Tab(label="PCA Explorer",       value="pca",
                    style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
            dcc.Tab(label="Temporal",           value="temporal",
                    style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
            dcc.Tab(label="Latent Cross-Corr",  value="xcorr",
                    style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
            dcc.Tab(label="Cluster Map",        value="clusters",
                    style=_TAB_STYLE, selected_style=_TAB_ACTIVE_STYLE),
        ]),
        html.Div(id="analysis-tab-content", style={"padding": "10px"}),
    ])


@callback(Output("analysis-tab-content", "children"),
          Input("analysis-subtabs", "value"))
def render_analysis_tab(sub: str):
    npz = get_analysis()
    d01 = npz.get("d01"); d02 = npz.get("d02")
    d03 = npz.get("d03"); d04 = npz.get("d04")
    feat_names = list(d01["feature_names"]) if d01 else []
    ALL_GROUPS = list(FEATURE_GROUPS.keys()) + ["All"]

    if sub == "corr":
        if d01 is None:
            return html.P("Run 01_aggregate_correlation.py first.")
        return html.Div([
            html.P(_HELP_CORR, style={"color": "#888", "fontSize": "0.85em",
                                       "marginBottom": "8px"}),
            html.Div([
                html.Label("Sort dims by:"),
                dcc.RadioItems(id="corr-sort",
                               options=[{"label": "Feature loading", "value": "loading"},
                                        {"label": "Cluster order",   "value": "cluster"},
                                        {"label": "Index",           "value": "index"}],
                               value="loading", inline=True),
                html.Label("Feature group:", style={"marginLeft": "20px"}),
                dcc.Dropdown(id="corr-group",
                             options=[{"label": g, "value": g} for g in ALL_GROUPS],
                             value="All", clearable=False,
                             style={"width": "160px", "display": "inline-block"}),
                html.Label("|r| ≥", style={"marginLeft": "20px"}),
                dcc.Slider(id="corr-thresh", min=0, max=0.35, step=0.05, value=0.0,
                           marks={v: f"{v:.2f}" for v in [0, 0.1, 0.2, 0.35]},
                           tooltip={"always_visible": False}),
                html.Label("Metric:", style={"marginLeft": "20px"}),
                dcc.RadioItems(id="corr-metric",
                               options=[{"label": "Pearson",  "value": "pearson"},
                                        {"label": "Spearman", "value": "spearman"}],
                               value="pearson", inline=True),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px",
                      "alignItems": "center", "marginBottom": "8px"}),
            dcc.Graph(id="corr-heatmap"),
            html.Div(id="corr-scatter-container"),
        ])

    if sub == "posters":
        if d01 is None:
            return html.P("Run 01_aggregate_correlation.py first.")
        options = [{"label": n, "value": n} for n in feat_names]
        return html.Div([
            html.P(_HELP_POSTERS, style={"color": "#888", "fontSize": "0.85em",
                                          "marginBottom": "8px"}),
            dcc.Dropdown(id="poster-feat", options=options,
                         value=feat_names[0] if feat_names else None,
                         clearable=False, style={"width": "300px"}),
            dcc.Graph(id="poster-graph", style={"marginTop": "12px"}),
        ])

    if sub == "pca":
        if d02 is None:
            return html.P("Run 02_pca_analysis.py first.")
        return html.Div([
            html.P(_HELP_PCA, style={"color": "#888", "fontSize": "0.85em",
                                      "marginBottom": "8px"}),
            html.Div([
                html.Label("Colour by feature:"),
                dcc.Dropdown(id="pca-colour",
                             options=[{"label": n, "value": n} for n in feat_names],
                             value=feat_names[0] if feat_names else None,
                             style={"width": "220px", "display": "inline-block"}),
                html.Label("PC axes:", style={"marginLeft": "16px"}),
                dcc.RadioItems(id="pca-axes",
                               options=[{"label": "PC1 vs PC2", "value": "12"},
                                        {"label": "PC1 vs PC3", "value": "13"}],
                               value="12", inline=True),
            ], style={"display": "flex", "gap": "12px", "alignItems": "center",
                      "marginBottom": "8px"}),
            dcc.Graph(id="pca-scatter"),
            html.H4("Cross-PCA alignment", style={"marginTop": "16px"}),
            dcc.Graph(id="pca-cross-heatmap"),
        ])

    if sub == "temporal":
        if d04 is None:
            return html.P("Run 04_temporal_correlation.py first.")
        sample    = d04.get("sample_crops")
        crop_opts = [{"label": f"crop {i}", "value": i}
                     for i in range(len(sample) if sample is not None else 0)]
        return html.Div([
            html.P(_HELP_TEMPORAL, style={"color": "#888", "fontSize": "0.85em",
                                           "marginBottom": "8px"}),
            html.Div([
                html.Label("Crop:"),
                dcc.Dropdown(id="temp-crop", options=crop_opts, value=0,
                             style={"width": "140px", "display": "inline-block"}),
                html.Label("Dims (comma-sep):", style={"marginLeft": "16px"}),
                dcc.Input(id="temp-dims", value="0,1,2,3,4", type="text",
                          style={"width": "180px"}),
            ], style={"display": "flex", "gap": "10px", "alignItems": "center",
                      "marginBottom": "8px"}),
            dcc.Graph(id="temporal-graph"),
        ])

    if sub == "xcorr":
        if d03 is None:
            return html.P("Run 03_latent_xcorr.py first.")
        return html.Div([
            html.P(_HELP_XCORR, style={"color": "#888", "fontSize": "0.85em",
                                        "marginBottom": "8px"}),
            dcc.Graph(id="xcorr-heatmap", figure=_build_xcorr_fig(d03)),
        ])

    if sub == "clusters":
        if d03 is None or d01 is None:
            return html.P("Run scripts 01 and 03 first.")
        return _cluster_layout(d01, d03, feat_names)

    return html.P("Unknown sub-tab.")


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(Output("corr-heatmap", "figure"),
          [Input("corr-sort", "value"), Input("corr-group", "value"),
           Input("corr-thresh", "value"), Input("corr-metric", "value")])
def update_corr_heatmap(sort_by, group, thresh, metric):
    npz = get_analysis(); d01 = npz.get("d01"); d03 = npz.get("d03")
    if d01 is None:
        return go.Figure()
    feat_names = list(d01["feature_names"])
    cluster_labels = d03["cluster_labels"] if d03 else np.zeros(LATENT_DIM, dtype=int)
    r = d01["r_pearson"] if metric == "pearson" else d01["r_spearman"]
    names = feat_names
    if group != "All":
        keep  = [i for i, n in enumerate(names) if n in set(FEATURE_GROUPS.get(group, []))]
        r     = r[:, keep]; names = [names[i] for i in keep]
    r_disp = np.where(np.abs(r) >= thresh, r, 0.0)
    if sort_by == "loading":    order = np.argsort(np.abs(r_disp).max(axis=1))[::-1]
    elif sort_by == "cluster":  order = np.argsort(cluster_labels)
    else:                       order = np.arange(LATENT_DIM)
    r_disp = r_disp[order]
    fig = go.Figure(go.Heatmap(
        z=r_disp, x=names, y=[f"dim {i}" for i in order],
        colorscale="RdBu_r", zmid=0,
        zmin=-POSTER_CLAMP, zmax=POSTER_CLAMP,
        colorbar=dict(title="r")))
    fig.update_layout(template="plotly_dark", height=700,
                      xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                      yaxis=dict(tickfont=dict(size=9)),
                      margin=dict(l=60, r=20, t=30, b=120))
    return fig


@callback(Output("corr-scatter-container", "children"),
          Input("corr-heatmap", "clickData"),
          State("corr-metric", "value"))
def corr_scatter(click_data, metric):
    npz = get_analysis(); d01 = npz.get("d01")
    if click_data is None or d01 is None:
        return html.P("Click a cell to see scatter plot.", style={"color": "#555"})
    pt  = click_data["points"][0]
    feat_name = pt["x"]; dim_label = pt["y"]
    dim = int(dim_label.split(" ")[1])
    feat_names = list(d01["feature_names"])
    fi  = feat_names.index(feat_name) if feat_name in feat_names else None
    if fi is None:
        return html.P(f"Feature {feat_name} not found.")
    r_val = d01["r_pearson"][dim, fi]
    from plots.latent_analysis.config import DATA_DIR
    scatter_d_path = DATA_DIR / "scatter_sample.npz"
    if not scatter_d_path.exists():
        return html.P("scatter_sample.npz not found — run script 01.")
    scatter_d = dict(np.load(str(scatter_d_path), allow_pickle=True))
    lm = scatter_d["latent_means"]; fv = scatter_d["feature_values"]
    fn = list(scatter_d["feature_names"])
    x  = lm[:, dim]
    if feat_name not in fn:
        return html.P(f"Feature {feat_name} not in scatter sample.")
    y  = fv[:, fn.index(feat_name)]
    fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers",
                                  marker=dict(size=3, opacity=0.4)))
    fig.update_layout(template="plotly_dark", height=300,
                      title=f"dim {dim} × {feat_name} — r={r_val:+.3f}",
                      xaxis_title=f"Latent dim {dim} mean", yaxis_title=feat_name)
    return dcc.Graph(figure=fig)


@callback(Output("poster-graph", "figure"), Input("poster-feat", "value"))
def update_poster(feat_name):
    npz = get_analysis(); d01 = npz.get("d01")
    if not feat_name or d01 is None:
        return go.Figure()
    feat_names = list(d01["feature_names"])
    fi = feat_names.index(feat_name) if feat_name in feat_names else None
    if fi is None:
        return go.Figure()
    r_col = d01["r_pearson"][:, fi]; n = int(d01["n_per_feature"][fi])
    grid  = r_col.reshape(8, 8)
    top3p = np.argsort(r_col)[::-1][:3]; top3n = np.argsort(r_col)[:3]
    fig = go.Figure(go.Heatmap(
        z=grid, colorscale="RdBu_r", zmid=0,
        zmin=-POSTER_CLAMP, zmax=POSTER_CLAMP,
        text=[[f"{grid[r,c]:.2f}" for c in range(8)] for r in range(8)],
        texttemplate="%{text}", textfont={"size": 10}))
    fig.update_layout(
        template="plotly_dark", width=500, height=500,
        title=f"{feat_name} — N={n:,} | top+: {list(top3p)} | top-: {list(top3n)}",
        xaxis=dict(title="dim mod 8", tickvals=list(range(8))),
        yaxis=dict(title="dim // 8", tickvals=list(range(8)),
                   ticktext=[str(i*8) for i in range(8)]))
    return fig


@callback(Output("pca-scatter", "figure"),
          [Input("pca-colour", "value"), Input("pca-axes", "value")])
def update_pca_scatter(colour_feat, axes):
    npz = get_analysis(); d02 = npz.get("d02")
    if d02 is None:
        return go.Figure()
    scores = d02["latent_scores"]
    pc1 = int(axes[0]) - 1; pc2 = int(axes[1]) - 1
    colour = None
    from plots.latent_analysis.config import DATA_DIR
    scatter_path = DATA_DIR / "scatter_sample.npz"
    if colour_feat and scatter_path.exists():
        sd = dict(np.load(str(scatter_path), allow_pickle=True))
        fn = list(sd["feature_names"])
        if colour_feat in fn:
            colour = sd["feature_values"][:, fn.index(colour_feat)][:scores.shape[0]]
    ev  = d02["latent_explained_variance_ratio"]
    fig = go.Figure(go.Scattergl(
        x=scores[:, pc1], y=scores[:, pc2],
        mode="markers",
        marker=dict(size=2, opacity=0.3, color=colour)))
    fig.update_layout(template="plotly_dark", height=500,
                      xaxis_title=f"Latent PC{pc1+1} ({ev[pc1]:.1%})",
                      yaxis_title=f"Latent PC{pc2+1} ({ev[pc2]:.1%})",
                      title="Latent PCA scatter")
    return fig


@callback(Output("pca-cross-heatmap", "figure"), Input("analysis-subtabs", "value"))
def update_cross_heatmap(sub):
    npz = get_analysis(); d02 = npz.get("d02")
    if sub != "pca" or d02 is None:
        return go.Figure()
    cc  = d02["cross_corr"]
    fig = go.Figure(go.Heatmap(z=cc, colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                                colorbar=dict(title="r")))
    fig.update_layout(template="plotly_dark", height=400,
                      xaxis_title="Latent PC", yaxis_title="Feature PC",
                      title="Feature PC ↔ Latent PC correlation")
    return fig


@callback(Output("temporal-graph", "figure"),
          [Input("temp-crop", "value"), Input("temp-dims", "value")])
def update_temporal(crop_idx, dims_str):
    npz = get_analysis(); d04 = npz.get("d04")
    if d04 is None:
        return go.Figure()
    sample = d04.get("sample_crops"); feat_sample = d04.get("sample_feat_segs")
    tfeat_names = list(d04.get("temporal_feature_names", TEMPORAL_FEATURE_NAMES))
    if sample is None or crop_idx is None or crop_idx >= len(sample):
        return go.Figure()
    try:
        dims = [int(d.strip()) for d in dims_str.split(",")]
        dims = [d for d in dims if 0 <= d < LATENT_DIM]
    except ValueError:
        dims = [0]
    lat = sample[crop_idx]; t = np.arange(256) * (2048 / 44100)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for dim in dims:
        fig.add_trace(
            go.Scattergl(x=t, y=lat[dim], mode="lines",
                         name=f"dim {dim}", line=dict(width=1)),
            secondary_y=False)
    if feat_sample is not None and crop_idx < len(feat_sample):
        feats = feat_sample[crop_idx]
        for fi in [0, 5]:
            if fi < feats.shape[0]:
                y = feats[fi]; y_n = (y - y.mean()) / (y.std() + 1e-9)
                fig.add_trace(
                    go.Scattergl(x=t, y=y_n, mode="lines",
                                 name=tfeat_names[fi],
                                 line=dict(width=1, dash="dot")),
                    secondary_y=True)
    fig.update_layout(template="plotly_dark", height=500,
                      xaxis_title="Time (s)",
                      title=f"Latent dims + frame features — crop {crop_idx}",
                      legend=dict(orientation="h"))
    fig.update_yaxes(title_text="Latent value", secondary_y=False)
    fig.update_yaxes(title_text="Feature (z-scored)", secondary_y=True)
    return fig


def _build_xcorr_fig(d03) -> go.Figure:
    xcorr = d03["xcorr_matrix"]; cl = d03["cluster_labels"]
    order = np.argsort(cl); xcorr_ord = xcorr[np.ix_(order, order)]
    labels_ord = [f"dim {i}" for i in order]
    fig = go.Figure(go.Heatmap(z=xcorr_ord, x=labels_ord, y=labels_ord,
                                colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                                colorbar=dict(title="r")))
    fig.update_layout(
        template="plotly_dark", height=700,
        title=f"64×64 Latent Cross-Correlation ({int(d03['n_crops_used'])} crops)",
        xaxis=dict(tickfont=dict(size=7)),
        yaxis=dict(tickfont=dict(size=7)),
        margin=dict(l=80, r=20, t=50, b=80))
    return fig


def _cluster_layout(d01, d03, feat_names) -> html.Div:
    cl = d03["cluster_labels"]; r_mat = d01["r_pearson"]
    n_cl = int(cl.max())
    rows = []
    for c in range(1, n_cl + 1):
        dims = list(np.where(cl == c)[0])
        mean_r = r_mat[dims].mean(axis=0)
        top3 = np.argsort(np.abs(mean_r))[::-1][:3]
        summary = ", ".join(
            f"{feat_names[i]} (r={mean_r[i]:+.2f})"
            for i in top3 if i < len(feat_names)
        )
        rows.append(html.Tr([
            html.Td(f"Cluster {c}"),
            html.Td(", ".join(map(str, dims)), style={"fontSize": "0.85em"}),
            html.Td(summary, style={"fontSize": "0.85em"}),
            html.Td(html.Button("Highlight in Dataset", n_clicks=0,
                                id={"type": "cluster-highlight-btn", "index": c},
                                style={"fontSize": "10px", "padding": "2px 6px",
                                       "background": "#1e2050", "border": "1px solid #4cd137",
                                       "color": "#4cd137", "cursor": "pointer"})),
        ]))
    return html.Div([
        html.P(_HELP_CLUSTERS, style={"color": "#888", "fontSize": "0.85em",
                                       "marginBottom": "8px"}),
        html.Table(
            [html.Tr([html.Th("Cluster"), html.Th("Dims"),
                      html.Th("Top features"), html.Th("")])] + rows,
            style={"width": "100%", "borderCollapse": "collapse",
                   "fontSize": "0.9em"},
        ),
    ])


@callback(
    Output("cluster-highlight", "data"),
    Input({"type": "cluster-highlight-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def highlight_cluster(n_clicks_list):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n for n in (n_clicks_list or []) if n):
        return no_update
    raw  = ctx.triggered[0]["prop_id"].split(".")[0]
    c    = json.loads(raw)["index"]
    npz  = get_analysis(); d03 = npz.get("d03")
    if d03 is None:
        return None
    cl   = d03["cluster_labels"]
    dims = [int(i) for i in np.where(cl == c)[0]]
    return {"dims": dims, "cluster": c}
