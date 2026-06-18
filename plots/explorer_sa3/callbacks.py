"""Dash callback wiring for the SA3 explorer."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from dash import Input, Output, no_update
from . import latents, analysis, viewer_tab, dataset_tab, analysis_tab, audio_panel
from . import player_client as pc
from .sidecar_index import CropMeta


def sample_ids(index: list[CropMeta], n: int, seed: int = 0) -> list[str]:
    ids = [c.id for c in index]
    if len(ids) <= n:
        return ids
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ids), size=n, replace=False)
    return [ids[i] for i in sorted(pick)]


def register(app, index: list[CropMeta], latent_dir: Path):
    crop_opts = [{"label": f"{c.id} — {c.source_track}", "value": c.id}
                 for c in index]

    @app.callback(Output("sa3-crop-dd", "options"),
                  Input("sa3-crop-dd", "id"))
    def _fill(_):
        return crop_opts

    @app.callback(Output("sa3-latent-graph", "figure"),
                  Output("sa3-ts-dd", "options"),
                  Output("sa3-audio-panel", "children"),
                  Input("sa3-crop-dd", "value"))
    def _show(cid):
        if not cid:
            return no_update, no_update, no_update
        import json
        z = latents.load_latent(latent_dir, cid)
        meta = json.loads((latent_dir / f"{cid}.json").read_text())
        ts = latents.load_timeseries(latent_dir, cid)
        fig = viewer_tab.latent_figure(z, latents.content_frames(meta))
        return (fig, [{"label": k, "value": k} for k in ts],
                audio_panel.panel(cid, pc.status()))

    @app.callback(Output("sa3-ts-graph", "figure"),
                  Input("sa3-crop-dd", "value"), Input("sa3-ts-dd", "value"))
    def _ts(cid, name):
        if not cid or not name:
            return no_update
        ts = latents.load_timeseries(latent_dir, cid)[name]
        return viewer_tab.timeseries_figure(name, ts)

    @app.callback(Output("sa3-xcorr-graph", "figure"),
                  Input("sa3-analysis-go", "n_clicks"))
    def _analysis(n):
        if not n:
            return no_update
        ids = sample_ids(index, 400)
        lats = [latents.load_latent(latent_dir, i) for i in ids]
        return analysis_tab.xcorr_figure(analysis.dim_xcorr(lats))
