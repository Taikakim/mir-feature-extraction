"""Dataset-wide scatter/histograms over cached sidecar scalars."""
from __future__ import annotations
import plotly.graph_objects as go
from dash import dcc, html


def scatter_figure(xs, ys, xlabel, ylabel, text=None) -> go.Figure:
    fig = go.Figure(go.Scattergl(x=xs, y=ys, mode="markers", text=text,
                                 marker=dict(size=5, opacity=0.6)))
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, height=520)
    return fig


def layout() -> html.Div:
    return html.Div([
        dcc.Dropdown(id="sa3-ds-x"), dcc.Dropdown(id="sa3-ds-y"),
        dcc.Graph(id="sa3-ds-graph"),
    ])
