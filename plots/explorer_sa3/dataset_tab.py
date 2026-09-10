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
        html.Div(id="sa3-ds-progress", style={"padding": "4px 0",
                                              "color": "#888"}),
        dcc.Interval(id="sa3-ds-interval", interval=1000),
        dcc.Dropdown(id="sa3-ds-x", value="bpm"),
        dcc.Dropdown(id="sa3-ds-y", value="lufs"),
        dcc.Graph(id="sa3-ds-graph"),
    ])
