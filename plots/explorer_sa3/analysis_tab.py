"""Live analysis tab: PCA, dim xcorr, dim<->feature correlation."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html


def xcorr_figure(c: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=c, colorscale="RdBu", zmid=0, zmin=-1, zmax=1))
    fig.update_layout(title="256×256 dim cross-correlation", height=560)
    return fig


def feature_corr_figure(corr: np.ndarray, feature: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=np.arange(len(corr)), y=corr))
    fig.update_layout(title=f"dim ↔ {feature} correlation",
                      xaxis_title="latent dim", yaxis_title="Pearson r",
                      height=320)
    return fig


def layout() -> html.Div:
    return html.Div([
        html.Button("Recompute (sampled)", id="sa3-analysis-go"),
        dcc.Graph(id="sa3-xcorr-graph"),
        dcc.Dropdown(id="sa3-feat-dd"),
        dcc.Graph(id="sa3-featcorr-graph"),
    ])
