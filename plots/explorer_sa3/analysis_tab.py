"""Live analysis tab: PCA, dim xcorr, dim<->feature correlation, co-activation."""
from __future__ import annotations
import csv as _csv
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html

# W's precomputed dim x feature correlations (999 crops) — the global dim
# ordering for the co-activation view (mir/stats, 2026-07-08).
_XCORR_CSV = Path(__file__).resolve().parents[2] / "stats" / "latent_dim_feature_xcorr.csv"
_DIM_CORR: dict[str, np.ndarray] = {}


def dim_corr_table() -> dict[str, np.ndarray]:
    """feature name -> [256] per-dim r, lazily parsed from W's csv."""
    if not _DIM_CORR and _XCORR_CSV.exists():
        with open(_XCORR_CSV) as f:
            for row in _csv.DictReader(f):
                _DIM_CORR[row["feature"]] = np.array(
                    [float(row[f"dim{i}"]) for i in range(256)], dtype=np.float32)
    return _DIM_CORR


def coactivation_figure(z: np.ndarray, feat: np.ndarray, feature_name: str,
                        top_n: int = 96) -> go.Figure:
    """Kim's '(latent dim x selected feature) x time' view: pointwise product of
    the standardized dim series and the standardized feature series — each cell
    glows where that dim moves WITH (blue) or AGAINST (red) the feature at that
    moment. Rows sorted by |global corr| from W's precomputed table (top_n dims),
    feature curve on a shared, zoom-linked time axis."""
    T = min(z.shape[-1], len(feat))
    z = z[..., :T].astype(np.float32)
    if z.ndim == 3:
        z = z[0]
    f = np.asarray(feat[:T], dtype=np.float32)
    corr = dim_corr_table().get(feature_name)
    order = (np.argsort(-np.abs(corr))[:top_n] if corr is not None
             else np.arange(min(top_n, z.shape[0])))
    zs = (z[order] - z[order].mean(axis=1, keepdims=True)) / (
        z[order].std(axis=1, keepdims=True) + 1e-8)
    fs = (f - f.mean()) / (f.std() + 1e-8)
    co = zs * fs[None, :]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.22, 0.78],
                        vertical_spacing=0.03)
    fig.add_trace(go.Scatter(y=f, mode="lines", name=feature_name,
                             line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Heatmap(z=co, colorscale="RdBu", zmid=0, zmin=-3, zmax=3,
                             y=[f"d{d}" for d in order], showscale=True), row=2, col=1)
    fig.update_layout(
        title=(f"co-activation: latent dims × {feature_name} over time "
               f"(top {len(order)} dims by |global r|, shared zoom)"),
        height=720, showlegend=False)
    fig.update_yaxes(title_text=feature_name, row=1, col=1)
    fig.update_yaxes(title_text="latent dim (affinity-sorted)", row=2, col=1)
    fig.update_xaxes(title_text="latent frame (10.77 Hz)", row=2, col=1)
    return fig


def xcorr_figure(c: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=c, colorscale="RdBu", zmid=0, zmin=-1, zmax=1))
    fig.update_layout(
        title="Latent-channel cross-correlation (256 × 256 channels)",
        xaxis_title="latent channel index",
        yaxis_title="latent channel index",
        height=560)
    return fig


def feature_corr_figure(corr: np.ndarray, feature: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=np.arange(len(corr)), y=corr))
    fig.update_layout(title=f"latent channel ↔ {feature} correlation",
                      xaxis_title="latent channel index",
                      yaxis_title="Pearson r",
                      height=320)
    return fig


def layout() -> html.Div:
    return html.Div([
        html.Button("Recompute (sampled)", id="sa3-analysis-go"),
        dcc.Loading(dcc.Graph(id="sa3-xcorr-graph"), type="default"),
        dcc.Dropdown(id="sa3-feat-dd",
                     placeholder="feature to correlate against…"),
        dcc.Loading(dcc.Graph(id="sa3-featcorr-graph"), type="default"),
        html.H4("Co-activation — how a feature lives across latent dims (per crop)"),
        html.Div([
            dcc.Dropdown(id="sa3-coact-crop", placeholder="crop…",
                         style={"minWidth": "260px"}),
            dcc.Dropdown(id="sa3-coact-feat", placeholder="feature…",
                         style={"minWidth": "260px"}),
        ], style={"display": "flex", "gap": "8px"}),
        dcc.Loading(dcc.Graph(id="sa3-coact-graph"), type="default"),
    ])


def register_coact(app, index, latent_dir):
    """Callbacks for the co-activation view (self-contained)."""
    from dash import Input, Output, no_update

    @app.callback(Output("sa3-coact-crop", "options"),
                  Output("sa3-coact-feat", "options"),
                  Input("sa3-analysis-go", "n_clicks"))
    def _fill(_n):
        crops = [{"label": c.stem, "value": c.stem} for c in index[:4000]]
        feats = [{"label": k, "value": k} for k in sorted(dim_corr_table())]
        return crops, feats

    @app.callback(Output("sa3-coact-graph", "figure"),
                  Input("sa3-coact-crop", "value"),
                  Input("sa3-coact-feat", "value"))
    def _draw(stem, feat_name):
        if not stem or not feat_name:
            return no_update
        z = np.load(f"{latent_dir}/{stem}.npy")
        with np.load(f"{latent_dir}/{stem}.TIMESERIES.npz") as d:
            if feat_name not in d.files:
                return no_update
            f = np.asarray(d[feat_name], dtype=np.float32)
            if f.ndim > 1:
                f = f.mean(axis=-1)
        return coactivation_figure(z, f, feat_name)
