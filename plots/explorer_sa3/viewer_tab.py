"""Per-crop latent + timeseries view."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from .latents import FRAME_RATE_HZ


def latent_figure(z: np.ndarray, content_frames: int) -> go.Figure:
    T = z.shape[1]
    secs = np.arange(T) / FRAME_RATE_HZ
    fig = go.Figure(go.Heatmap(z=z, x=secs, colorscale="RdBu", zmid=0))
    if 0 < content_frames < T:
        cut = content_frames / FRAME_RATE_HZ
        fig.add_vline(x=cut, line_dash="dash", line_color="black")
    fig.update_layout(title="Latent [256 × T]", xaxis_title="seconds",
                      yaxis_title="latent dim", height=480)
    return fig


def timeseries_figure(name: str, ts: np.ndarray) -> go.Figure:
    if ts.ndim == 2:
        fig = go.Figure(go.Heatmap(z=ts.T, colorscale="Viridis"))
    else:
        secs = np.arange(len(ts)) / FRAME_RATE_HZ
        fig = go.Figure(go.Scatter(x=secs, y=ts, mode="lines"))
    fig.update_layout(title=name, height=240, margin=dict(t=30))
    return fig


def placeholder_figure(msg: str, height: int = 480) -> go.Figure:
    """Blank-but-labelled figure so an unpopulated graph isn't a bare plane."""
    fig = go.Figure()
    fig.update_layout(height=height,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      annotations=[dict(text=msg, showarrow=False,
                                        font=dict(color="#888", size=14))])
    return fig


def layout() -> html.Div:
    return html.Div([
        html.Label("Track"),
        dcc.Dropdown(id="sa3-track-dd", placeholder="choose a track…"),
        html.Label("Crop (within track)"),
        dcc.Dropdown(id="sa3-crop-dd", placeholder="choose a crop…"),
        dcc.Graph(id="sa3-latent-graph",
                  figure=placeholder_figure("choose a track above — its first "
                                            "crop's latent renders here")),
        dcc.Dropdown(id="sa3-ts-dd", placeholder="timeseries feature…"),
        dcc.Graph(id="sa3-ts-graph",
                  figure=placeholder_figure("timeseries of the selected crop",
                                            height=240)),
        html.Div(id="sa3-audio-panel"),
    ])
