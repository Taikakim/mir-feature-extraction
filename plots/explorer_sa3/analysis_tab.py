"""Live analysis tab: PCA, dim xcorr, dim<->feature correlation."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html


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
    ])
