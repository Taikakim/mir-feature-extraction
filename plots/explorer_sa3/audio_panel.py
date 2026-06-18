"""Audio controls that target the SA3 player over HTTP."""
from __future__ import annotations
from dash import html
from . import player_client as pc


def panel(crop_id: str, alive: bool) -> html.Div:
    if not alive:
        return html.Div([
            html.P("Player offline. Launch:"),
            html.Code("/home/kim/Projects/SAO/stable-audio-3/.venv/bin/"
                      "python scripts/latent_server_sa3.py"),
        ])
    return html.Div([
        html.Audio(src=pc.decode_url(crop_id), controls=True),
        html.Audio(src=pc.source_url(crop_id), controls=True),
    ])
