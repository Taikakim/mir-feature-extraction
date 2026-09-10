"""Audio controls that target the SA3 player over HTTP."""
from __future__ import annotations
from dash import html
from . import player_client as pc


def panel(crop_id: str, alive: bool) -> html.Div:
    if not alive:
        return html.Div([
            html.P("Render server offline (it hosts the player endpoints). Launch:"),
            html.Code("/home/kim/Projects/SAO/.venv/bin/python "
                      "/home/kim/Projects/SAO/eval/explorer_render_server.py"),
        ])
    return html.Div([
        html.Audio(src=pc.decode_url(crop_id), controls=True),
        html.Audio(src=pc.source_url(crop_id), controls=True),
    ])
