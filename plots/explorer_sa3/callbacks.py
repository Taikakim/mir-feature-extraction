"""Dash callback wiring for the SA3 explorer."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from dash import Input, Output, no_update
from . import latents, analysis, viewer_tab, dataset_tab, analysis_tab, audio_panel
from . import player_client as pc
from .sidecar_index import CropMeta


SCALAR_FIELDS = ["bpm", "lufs", "rel_pos"]


def oned_feature_names(ts: dict) -> list[str]:
    """TIMESERIES field names that are 1-D per-frame features (excludes 2-D like hpcp_ts)."""
    return [k for k, v in ts.items() if getattr(v, "ndim", 0) == 1]


def scalar_options() -> list[dict]:
    """Dropdown options for the numeric CropMeta scalars usable on the scatter."""
    return [{"label": f, "value": f} for f in SCALAR_FIELDS]


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

    @app.callback(Output("sa3-feat-dd", "options"),
                  Output("sa3-feat-dd", "value"),
                  Input("sa3-analysis-go", "n_clicks"))
    def _feat_dd_fill(n):
        if not n:
            return no_update, no_update
        ids = sample_ids(index, 400)
        ts = latents.load_timeseries(latent_dir, ids[0])
        names = oned_feature_names(ts)
        opts = [{"label": k, "value": k} for k in names]
        default = names[0] if names else None
        return opts, default

    @app.callback(Output("sa3-featcorr-graph", "figure"),
                  Input("sa3-feat-dd", "value"))
    def _featcorr(feat):
        if not feat:
            return no_update
        ids = sample_ids(index, 400)
        lats, feats = [], []
        for cid in ids:
            ts = latents.load_timeseries(latent_dir, cid)
            if feat not in ts:
                continue
            f = ts[feat]
            if getattr(f, "ndim", 0) != 1:
                continue
            lats.append(latents.load_latent(latent_dir, cid))
            feats.append(f)
        if not lats:
            return no_update
        corr = analysis.dim_feature_corr(lats, feats)
        return analysis_tab.feature_corr_figure(corr, feat)

    @app.callback(Output("sa3-ds-x", "options"), Output("sa3-ds-x", "value"),
                  Output("sa3-ds-y", "options"), Output("sa3-ds-y", "value"),
                  Input("sa3-ds-x", "id"))
    def _ds_fill(_):
        opts = scalar_options()
        return opts, "bpm", opts, "lufs"

    @app.callback(Output("sa3-ds-graph", "figure"),
                  Input("sa3-ds-x", "value"), Input("sa3-ds-y", "value"))
    def _ds_scatter(xf, yf):
        if not xf or not yf:
            return no_update
        xs, ys, txt = [], [], []
        for c in index:
            xv, yv = getattr(c, xf), getattr(c, yf)
            if xv is None or yv is None:
                continue
            xs.append(xv); ys.append(yv); txt.append(f"{c.artist} — {c.title}")
        return dataset_tab.scatter_figure(xs, ys, xlabel=xf, ylabel=yf, text=txt)
