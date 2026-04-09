# plots/latent_analysis/app.py
"""
Latent Feature Analysis — Plotly Dash Explorer (port 7895)

Run:
    python plots/latent_analysis/app.py [--port 7895] [--debug]

Reads from:
    plots/latent_analysis/data/01_correlations.npz
    plots/latent_analysis/data/02_pca.npz
    plots/latent_analysis/data/03_xcorr.npz
    plots/latent_analysis/data/04_temporal.npz
    plots/latent_analysis/data/posters/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash.exceptions

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plots.latent_analysis.config import (
    DATA_DIR, POSTER_DIR, FEATURE_GROUPS, LATENT_DIM,
    POSTER_CLAMP, EFFECT_WEAK, EFFECT_STRONG,
    TEMPORAL_FEATURE_NAMES,
)

# ---------------------------------------------------------------------------
# Data loading (at startup)
# ---------------------------------------------------------------------------

def _load(npz_path):
    if not Path(npz_path).exists():
        return None
    return dict(np.load(str(npz_path), allow_pickle=True))


d01 = _load(DATA_DIR / "01_correlations.npz")
d02 = _load(DATA_DIR / "02_pca.npz")
d03 = _load(DATA_DIR / "03_xcorr.npz")
d04 = _load(DATA_DIR / "04_temporal.npz")

feat_names = list(d01["feature_names"]) if d01 else []
ALL_GROUPS = list(FEATURE_GROUPS.keys()) + ["All"]
cluster_labels = d03["cluster_labels"] if d03 else np.zeros(LATENT_DIM, dtype=int)

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="Latent Feature Explorer",
                suppress_callback_exceptions=True)

_tab_style        = {"padding": "8px 16px", "color": "#888"}
_tab_active_style = {"padding": "8px 16px", "backgroundColor": "#7eb8f7",
                     "color": "#0d0d1a", "fontWeight": "700"}

app.layout = html.Div([
    html.H2("Latent ↔ Feature Analysis", style={"margin": "12px 16px 4px"}),

    dcc.Tabs(id="tabs", value="corr", children=[
        dcc.Tab(label="Correlation Matrix", value="corr",
                style=_tab_style, selected_style=_tab_active_style),
        dcc.Tab(label="Feature Posters",    value="posters",
                style=_tab_style, selected_style=_tab_active_style),
        dcc.Tab(label="PCA Explorer",       value="pca",
                style=_tab_style, selected_style=_tab_active_style),
        dcc.Tab(label="Temporal",           value="temporal",
                style=_tab_style, selected_style=_tab_active_style),
        dcc.Tab(label="Latent Cross-Corr",  value="xcorr",
                style=_tab_style, selected_style=_tab_active_style),
        dcc.Tab(label="Cluster Map",        value="clusters",
                style=_tab_style, selected_style=_tab_active_style),
    ]),
    html.Div(id="tab-content", style={"padding": "12px"}),
], style={"fontFamily": "monospace", "backgroundColor": "#0d0d1a", "color": "#ccc",
          "minHeight": "100vh"})


# ---------------------------------------------------------------------------
# Tab 1 — Correlation Matrix
# ---------------------------------------------------------------------------

_HELP_CORR = (
    "Rows = 64 VAE latent dimensions; columns = MIR features. "
    "Colour = Pearson r (red positive, blue negative). "
    "Use this to understand which perceptual qualities each latent dim encodes: "
    "a dim with high r for 'brightness' will push audio brighter when nudged upward. "
    "Sort by 'Feature loading' to surface the most expressive dims first. "
    "Click any cell to see a scatter plot of that dim vs. feature across 5 000 crops."
)


def _corr_layout():
    if d01 is None:
        return html.P("Run 01_aggregate_correlation.py first.")
    return html.Div([
        html.P(_HELP_CORR, style={"color": "#888", "fontSize": "0.85em", "marginBottom": "8px"}),
        html.Div([
            html.Label("Sort dims by:"),
            dcc.RadioItems(id="corr-sort",
                           options=[{"label": "Feature loading", "value": "loading"},
                                    {"label": "Cluster order",   "value": "cluster"},
                                    {"label": "Index",           "value": "index"}],
                           value="loading", inline=True),
            html.Label("Feature group:", style={"marginLeft": "20px"}),
            dcc.Dropdown(id="corr-group", options=[{"label": g, "value": g} for g in ALL_GROUPS],
                         value="All", clearable=False, style={"width": "160px", "display": "inline-block"}),
            html.Label("|r| ≥", style={"marginLeft": "20px"}),
            dcc.Slider(id="corr-thresh", min=0, max=0.35, step=0.05, value=0.0,
                       marks={v: f"{v:.2f}" for v in [0, 0.1, 0.2, 0.35]}, tooltip={"always_visible": False}),
            html.Label("Show:", style={"marginLeft": "20px"}),
            dcc.RadioItems(id="corr-metric",
                           options=[{"label": "Pearson", "value": "pearson"},
                                    {"label": "Spearman", "value": "spearman"}],
                           value="pearson", inline=True),
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px",
                  "alignItems": "center", "marginBottom": "8px"}),
        dcc.Graph(id="corr-heatmap"),
        html.Div(id="corr-scatter-container"),
    ])


@app.callback(Output("corr-heatmap", "figure"),
              [Input("corr-sort", "value"), Input("corr-group", "value"),
               Input("corr-thresh", "value"), Input("corr-metric", "value")])
def update_corr_heatmap(sort_by, group, thresh, metric):
    if d01 is None:
        return go.Figure()
    r = d01["r_pearson"] if metric == "pearson" else d01["r_spearman"]  # [64, N_feat]
    names = feat_names

    # Filter by group
    if group != "All":
        group_feats = set(FEATURE_GROUPS.get(group, []))
        keep = [i for i, n in enumerate(names) if n in group_feats]
        r     = r[:, keep]
        names = [names[i] for i in keep]

    # Apply threshold: mask weak correlations
    r_disp = np.where(np.abs(r) >= thresh, r, 0.0)

    # Sort dims
    if sort_by == "loading":
        order = np.argsort(np.abs(r_disp).max(axis=1))[::-1]
    elif sort_by == "cluster":
        order = np.argsort(cluster_labels)
    else:
        order = np.arange(LATENT_DIM)
    r_disp = r_disp[order]

    fig = go.Figure(go.Heatmap(
        z=r_disp, x=names, y=[f"dim {i}" for i in order],
        colorscale="RdBu_r", zmid=0, zmin=-POSTER_CLAMP, zmax=POSTER_CLAMP,
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        template="plotly_dark", height=700,
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        margin=dict(l=60, r=20, t=30, b=120),
    )
    return fig


@app.callback(Output("corr-scatter-container", "children"),
              Input("corr-heatmap", "clickData"),
              State("corr-metric", "value"))
def corr_scatter(click_data, metric):
    if click_data is None or d01 is None:
        return html.P("Click a cell to see scatter plot.", style={"color": "#555"})
    pt  = click_data["points"][0]
    feat_name = pt["x"]
    dim_label = pt["y"]  # "dim N"
    dim = int(dim_label.split(" ")[1])
    fi  = feat_names.index(feat_name) if feat_name in feat_names else None
    if fi is None:
        return html.P(f"Feature {feat_name} not found.")
    r_val = d01["r_pearson"][dim, fi]

    scatter_d = _load(DATA_DIR / "scatter_sample.npz")
    if scatter_d is None:
        return html.P("scatter_sample.npz not found — run script 01.")
    lm        = scatter_d["latent_means"]          # [5000, 64]
    fv        = scatter_d["feature_values"]        # [5000, N_feat]
    fn        = list(scatter_d["feature_names"])   # list of feature name strings
    x = lm[:, dim]
    if feat_name in fn:
        y = fv[:, fn.index(feat_name)]
    else:
        return html.P(f"Feature {feat_name} not in scatter sample.")

    fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers",
                                  marker=dict(size=3, opacity=0.4)))
    fig.update_layout(template="plotly_dark", height=300,
                      title=f"dim {dim} × {feat_name} — Pearson r = {r_val:+.3f}",
                      xaxis_title=f"Latent dim {dim} mean",
                      yaxis_title=feat_name)
    return dcc.Graph(figure=fig)


# ---------------------------------------------------------------------------
# Tab 2 — Feature Posters
# ---------------------------------------------------------------------------

_HELP_POSTERS = (
    "Each cell is one of the 64 VAE latent dimensions arranged in an 8×8 grid. "
    "Colour = Pearson r with the selected MIR feature across all crops. "
    "Red = latent dim rises when the feature rises; blue = inverse. "
    "Strong cells (|r| ≥ 0.20) indicate that the decoder encodes that perceptual quality "
    "in a specific part of latent space — useful for targeted latent manipulation."
)


def _posters_layout():
    options = [{"label": n, "value": n} for n in feat_names]
    default = feat_names[0] if feat_names else None
    return html.Div([
        html.P(_HELP_POSTERS, style={"color": "#888", "fontSize": "0.85em", "marginBottom": "8px"}),
        dcc.Dropdown(id="poster-feat", options=options, value=default,
                     clearable=False, style={"width": "300px"}),
        dcc.Graph(id="poster-graph", style={"marginTop": "12px"}),
    ])


@app.callback(Output("poster-graph", "figure"), Input("poster-feat", "value"))
def update_poster(feat_name):
    if not feat_name or d01 is None:
        return go.Figure()
    fi = feat_names.index(feat_name) if feat_name in feat_names else None
    if fi is None:
        return go.Figure()
    r_col = d01["r_pearson"][:, fi]  # [64]
    n     = int(d01["n_per_feature"][fi])
    grid  = r_col.reshape(8, 8)

    fig = go.Figure(go.Heatmap(
        z=grid, colorscale="RdBu_r", zmid=0, zmin=-POSTER_CLAMP, zmax=POSTER_CLAMP,
        text=[[f"{grid[r,c]:.2f}" for c in range(8)] for r in range(8)],
        texttemplate="%{text}", textfont={"size": 10},
        xaxis="x", yaxis="y",
    ))
    top3_pos = np.argsort(r_col)[::-1][:3]
    top3_neg = np.argsort(r_col)[:3]
    fig.update_layout(
        template="plotly_dark", width=500, height=500,
        title=f"{feat_name} — N={n:,} | top+: dims {list(top3_pos)} | top-: dims {list(top3_neg)}",
        xaxis=dict(title="dim mod 8", tickvals=list(range(8))),
        yaxis=dict(title="dim // 8 × 8", tickvals=list(range(8)),
                   ticktext=[str(i*8) for i in range(8)]),
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 3 — PCA Explorer
# ---------------------------------------------------------------------------

_HELP_PCA = (
    "PCA compresses the 64 latent dims into principal components that capture maximum variance. "
    "The scatter shows every crop projected onto latent PC axes, coloured by any MIR feature. "
    "If a feature cleanly gradient-colours the scatter, the latent manifold encodes that quality "
    "in a structured direction — meaning interpolation in latent space will smoothly vary it. "
    "The cross-correlation heatmap below shows alignment between feature PCA components and "
    "latent PCA components — high |r| means both spaces discovered the same underlying structure."
)


def _pca_layout():
    if d02 is None:
        return html.P("Run 02_pca_analysis.py first.")
    return html.Div([
        html.P(_HELP_PCA, style={"color": "#888", "fontSize": "0.85em", "marginBottom": "8px"}),
        html.Div([
            html.Label("Colour by feature:"),
            dcc.Dropdown(id="pca-colour", options=[{"label": n, "value": n} for n in feat_names],
                         value=feat_names[0] if feat_names else None,
                         style={"width": "220px", "display": "inline-block"}),
            html.Label("PC axes:", style={"marginLeft": "16px"}),
            dcc.RadioItems(id="pca-axes",
                           options=[{"label": "PC1 vs PC2", "value": "12"},
                                    {"label": "PC1 vs PC3", "value": "13"}],
                           value="12", inline=True),
        ], style={"display": "flex", "gap": "12px", "alignItems": "center", "marginBottom": "8px"}),
        dcc.Graph(id="pca-scatter"),
        html.H4("Cross-PCA alignment (Feature PC → Latent PC)", style={"marginTop": "16px"}),
        dcc.Graph(id="pca-cross-heatmap"),
    ])


@app.callback(Output("pca-scatter", "figure"),
              [Input("pca-colour", "value"), Input("pca-axes", "value")])
def update_pca_scatter(colour_feat, axes):
    if d02 is None:
        return go.Figure()
    scores = d02["latent_scores"]  # [N, 20]
    pc1 = int(axes[0]) - 1
    pc2 = int(axes[1]) - 1

    # Colour by feature using scatter_sample.npz (saved by script 01)
    colour = None
    scatter_d = _load(DATA_DIR / "scatter_sample.npz")
    if colour_feat and scatter_d is not None:
        fn = list(scatter_d["feature_names"])
        if colour_feat in fn:
            n_scatter = scatter_d["latent_means"].shape[0]
            colour = scatter_d["feature_values"][:, fn.index(colour_feat)][:n_scatter]

    ev = d02["latent_explained_variance_ratio"]
    fig = go.Figure(go.Scattergl(
        x=scores[:, pc1], y=scores[:, pc2],
        mode="markers", marker=dict(size=2, opacity=0.3, color=colour),
    ))
    fig.update_layout(
        template="plotly_dark", height=500,
        xaxis_title=f"Latent PC{pc1+1} ({ev[pc1]:.1%})",
        yaxis_title=f"Latent PC{pc2+1} ({ev[pc2]:.1%})",
        title="Latent PCA scatter",
    )
    return fig


@app.callback(Output("pca-cross-heatmap", "figure"), Input("tabs", "value"))
def update_cross_heatmap(tab):
    if tab != "pca" or d02 is None:
        return go.Figure()
    cc = d02["cross_corr"]  # [n_feat_pc, n_lat_pc]
    fig = go.Figure(go.Heatmap(
        z=cc, colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        xaxis="x", yaxis="y",
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        template="plotly_dark", height=400,
        xaxis_title="Latent PC", yaxis_title="Feature PC",
        title="Feature PC ↔ Latent PC correlation",
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 4 — Temporal
# ---------------------------------------------------------------------------

_HELP_TEMPORAL = (
    "Each latent frame = 2048 audio samples @ 44 100 Hz ≈ 46 ms. "
    "This tab plots latent dim values over time for a single crop alongside frame-level "
    "audio features (RMS energy, spectral flatness, etc.) at the same resolution. "
    "Use it to see whether a latent dim tracks transients, energy bursts, or spectral changes "
    "within a track — revealing whether a dim encodes short-term dynamics or long-term texture."
)


def _temporal_layout():
    if d04 is None:
        return html.P("Run 04_temporal_correlation.py first.")
    n_crops = int(d04.get("n_crops_used", 0))
    sample  = d04.get("sample_crops")
    crop_opts = [{"label": f"crop {i}", "value": i}
                 for i in range(len(sample) if sample is not None else 0)]
    return html.Div([
        html.P(_HELP_TEMPORAL, style={"color": "#888", "fontSize": "0.85em", "marginBottom": "8px"}),
        html.Div([
            html.Label("Crop:"),
            dcc.Dropdown(id="temp-crop", options=crop_opts, value=0,
                         style={"width": "140px", "display": "inline-block"}),
            html.Label("Dims (comma-sep):", style={"marginLeft": "16px"}),
            dcc.Input(id="temp-dims", value="0,1,2,3,4", type="text",
                      style={"width": "180px"}),
            html.Label("BPM source:", style={"marginLeft": "16px"}),
            dcc.RadioItems(id="temp-bpm",
                           options=[{"label": "Essentia", "value": "essentia"},
                                    {"label": "Madmom",   "value": "madmom"},
                                    {"label": "Average",  "value": "avg"}],
                           value="essentia", inline=True),
        ], style={"display": "flex", "gap": "10px", "alignItems": "center", "marginBottom": "8px"}),
        dcc.Graph(id="temporal-graph"),
    ])


@app.callback(Output("temporal-graph", "figure"),
              [Input("temp-crop", "value"), Input("temp-dims", "value"),
               Input("temp-bpm", "value")])
def update_temporal(crop_idx, dims_str, bpm_src):
    if d04 is None:
        return go.Figure()
    sample      = d04.get("sample_crops")        # [K, 64, 256]
    feat_sample = d04.get("sample_feat_segs")    # [K, N_tfeats, 256]
    tfeat_names = list(d04.get("temporal_feature_names", TEMPORAL_FEATURE_NAMES))

    if sample is None or crop_idx is None or crop_idx >= len(sample):
        return go.Figure()

    try:
        dims = [int(d.strip()) for d in dims_str.split(",")]
        dims = [d for d in dims if 0 <= d < LATENT_DIM]
    except ValueError:
        dims = [0]

    lat = sample[crop_idx]   # [64, 256]
    t   = np.arange(256) * (2048 / 44100)   # time in seconds

    # Two-axis figure: left=latent dims, right=temporal features (normalised)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for dim in dims:
        fig.add_trace(go.Scattergl(x=t, y=lat[dim], mode="lines",
                                    name=f"dim {dim}", line=dict(width=1)),
                      secondary_y=False)

    # Overlay temporal features (z-scored for comparability)
    if feat_sample is not None and crop_idx < len(feat_sample):
        feats = feat_sample[crop_idx]   # [N_tfeats, 256]
        # Show broadband RMS and spectral flatness by default (indices 0 and 5)
        for fi in [0, 5]:
            if fi < feats.shape[0]:
                y = feats[fi]
                y_norm = (y - y.mean()) / (y.std() + 1e-9)
                fig.add_trace(go.Scattergl(x=t, y=y_norm, mode="lines",
                                            name=tfeat_names[fi],
                                            line=dict(width=1, dash="dot")),
                              secondary_y=True)

    fig.update_layout(
        template="plotly_dark", height=500,
        xaxis_title="Time (s)",
        title=f"Latent dims + frame features — sample crop {crop_idx}",
        legend=dict(orientation="h"),
    )
    fig.update_yaxes(title_text="Latent value",        secondary_y=False)
    fig.update_yaxes(title_text="Feature (z-scored)",  secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# Tab 5 — Latent Cross-Corr
# ---------------------------------------------------------------------------

_HELP_XCORR = (
    "64×64 matrix of Pearson r between every pair of latent dims, averaged over 2 000 crops "
    "using Fisher-Z transform. Dims sorted by Ward hierarchical cluster. "
    "Tight clusters (bright red blocks along the diagonal) are dims that move together "
    "— they likely encode related perceptual qualities. "
    "Dims in the same cluster can often be manipulated as a group for stronger perceptual effect."
)


def _xcorr_layout():
    if d03 is None:
        return html.P("Run 03_latent_xcorr.py first.")
    return html.Div([
        html.P(_HELP_XCORR, style={"color": "#888", "fontSize": "0.85em", "marginBottom": "8px"}),
        dcc.Graph(id="xcorr-heatmap", figure=_build_xcorr_fig()),
    ])


def _build_xcorr_fig():
    xcorr = d03["xcorr_matrix"]  # [64, 64]
    cl    = d03["cluster_labels"]
    order = np.argsort(cl)
    xcorr_ord = xcorr[np.ix_(order, order)]
    labels_ord = [f"dim {i}" for i in order]
    fig = go.Figure(go.Heatmap(
        z=xcorr_ord, x=labels_ord, y=labels_ord,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        template="plotly_dark", height=700,
        title=f"64×64 Latent Cross-Correlation (Fisher-Z avg, {int(d03['n_crops_used'])} crops, sorted by cluster)",
        xaxis=dict(tickfont=dict(size=7)), yaxis=dict(tickfont=dict(size=7)),
        margin=dict(l=80, r=20, t=50, b=80),
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 6 — Cluster Map
# ---------------------------------------------------------------------------

_HELP_CLUSTERS = (
    "Summary table of the Ward clusters found in the cross-correlation analysis. "
    "Each row shows which latent dims group together and what MIR features they correlate with most. "
    "This is the most actionable view: it tells you which dims to move together when you want to "
    "push the audio toward a specific quality — e.g. a 'brightness cluster' can be nudged as a "
    "unit in the Shape Explorer's manipulation sliders for a stronger, more coherent effect."
)


def _cluster_layout():
    if d03 is None or d01 is None:
        return html.P("Run scripts 01 and 03 first.")
    cl     = d03["cluster_labels"]
    r_mat  = d01["r_pearson"]
    n_cl   = int(cl.max())

    rows = []
    for c in range(1, n_cl + 1):
        dims = np.where(cl == c)[0]
        mean_r = r_mat[dims].mean(axis=0)     # avg r per feature
        top3   = np.argsort(np.abs(mean_r))[::-1][:3]
        summary = ", ".join(
            f"{feat_names[i]} (r={mean_r[i]:+.2f})"
            for i in top3 if i < len(feat_names)
        )
        rows.append(html.Tr([
            html.Td(f"Cluster {c}"),
            html.Td(", ".join(map(str, dims))),
            html.Td(summary, style={"fontSize": "0.85em"}),
        ]))

    return html.Div([
        html.P(_HELP_CLUSTERS, style={"color": "#888", "fontSize": "0.85em", "marginBottom": "8px"}),
        html.Table(
            [html.Tr([html.Th("Cluster"), html.Th("Dims"), html.Th("Top correlated features")])] + rows,
            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.9em"},
        ),
    ])


# ---------------------------------------------------------------------------
# Main tab router
# ---------------------------------------------------------------------------

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "corr":    return _corr_layout()
    if tab == "posters": return _posters_layout()
    if tab == "pca":     return _pca_layout()
    if tab == "temporal":return _temporal_layout()
    if tab == "xcorr":   return _xcorr_layout()
    if tab == "clusters":return _cluster_layout()
    return html.P("Unknown tab.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int, default=7895)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"Starting Latent Feature Explorer on http://localhost:{args.port}")
    print(f"Data loaded: 01={'✓' if d01 else '✗'} 02={'✓' if d02 else '✗'} "
          f"03={'✓' if d03 else '✗'} 04={'✓' if d04 else '✗'}")
    app.run(debug=args.debug, port=args.port, host="127.0.0.1")
