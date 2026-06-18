"""SA3 latent explorer — Dash entrypoint (mir venv)."""
from __future__ import annotations
import configparser
from pathlib import Path
import dash
from dash import dcc, html
from .sidecar_index import scan_index
from . import viewer_tab, dataset_tab, analysis_tab

_INI = Path(__file__).parent.parent.parent / "latent_player_sa3.ini"


def _latent_dir() -> Path:
    cfg = configparser.ConfigParser()
    cfg.read(_INI)
    return Path(cfg["server"]["latent_dir"])


def build_layout(index) -> html.Div:
    return html.Div([
        html.H2(f"SA3 Latent Explorer — {len(index)} crops"),
        dcc.Tabs([
            dcc.Tab(label="Viewer", children=viewer_tab.layout()),
            dcc.Tab(label="Dataset", children=dataset_tab.layout()),
            dcc.Tab(label="Analysis", children=analysis_tab.layout()),
        ]),
    ])


app = dash.Dash(__name__)


def main():
    index = scan_index(_latent_dir())
    app.layout = build_layout(index)
    # Callbacks are registered in app_callbacks (wired in Task 9).
    app.run(debug=False, port=8051)


if __name__ == "__main__":
    main()
