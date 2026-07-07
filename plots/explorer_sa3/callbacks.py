"""Dash callback wiring for the SA3 explorer."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from dash import Input, Output, State, no_update
from . import latents, analysis, viewer_tab, dataset_tab, analysis_tab, audio_panel
from . import player_client as pc
from .sidecar_index import CropMeta, group_by_track
from .scalar_cache import ScalarCache
from . import inference_tab, a2a_tab


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
    tracks = group_by_track(index)
    track_opts = [{"label": f"{t or '(unknown track)'} — {len(cs)} crops",
                   "value": t} for t, cs in sorted(tracks.items())]
    cache = ScalarCache(latent_dir, [c.id for c in index])

    @app.callback(Output("sa3-track-dd", "options"),
                  Input("sa3-track-dd", "id"))
    def _track_fill(_):
        return track_opts

    @app.callback(Output("sa3-crop-dd", "options"),
                  Output("sa3-crop-dd", "value"),
                  Input("sa3-track-dd", "value"))
    def _track_pick(track):
        if track is None or track not in tracks:
            return [], None
        crops = sorted(tracks[track], key=lambda c: c.rel_pos)
        opts = [{"label": f"{c.id} — pos {c.rel_pos:.2f}", "value": c.id}
                for c in crops]
        return opts, crops[0].id   # auto-select first crop → plots populate

    @app.callback(Output("sa3-latent-graph", "figure"),
                  Output("sa3-ts-dd", "options"),
                  Output("sa3-ts-dd", "value"),
                  Output("sa3-audio-panel", "children"),
                  Input("sa3-crop-dd", "value"),
                  State("sa3-ts-dd", "value"))
    def _show(cid, ts_sel):
        if not cid:
            return (viewer_tab.placeholder_figure(
                        "choose a track above — its first crop's latent "
                        "renders here"),
                    [], None, no_update)
        import json
        z = latents.load_latent(latent_dir, cid)
        meta = json.loads((latent_dir / f"{cid}.json").read_text())
        ts = latents.load_timeseries(latent_dir, cid)
        fig = viewer_tab.latent_figure(z, latents.content_frames(meta))
        names = list(ts)
        value = ts_sel if ts_sel in ts else (names[0] if names else None)
        return (fig, [{"label": k, "value": k} for k in names], value,
                audio_panel.panel(cid, pc.status()))

    @app.callback(Output("sa3-ts-graph", "figure"),
                  Input("sa3-crop-dd", "value"), Input("sa3-ts-dd", "value"))
    def _ts(cid, name):
        if not cid or not name:
            return viewer_tab.placeholder_figure(
                "timeseries of the selected crop", height=240)
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

    @app.callback(Output("sa3-ds-progress", "children"),
                  Output("sa3-ds-x", "options"), Output("sa3-ds-y", "options"),
                  Output("sa3-ds-interval", "disabled"),
                  Input("sa3-ds-interval", "n_intervals"))
    def _ds_progress(_n):
        cache.ensure_started()   # loads cache npz or spawns ONE bg thread
        st = cache.status()
        base = scalar_options()
        if st["state"] == "ready":
            opts = base + [{"label": f"mean({f})", "value": f}
                           for f in cache.fields()]
            return (f"{len(opts)} features (timeseries-mean cache ready)",
                    opts, opts, True)
        return (f"computing timeseries means… {st['done']}/{st['total']} crops",
                base, base, False)

    def _ds_value(c: CropMeta, field: str):
        if field in SCALAR_FIELDS:
            return getattr(c, field)
        return cache.value(c.id, field)

    @app.callback(Output("sa3-ds-graph", "figure"),
                  Input("sa3-ds-x", "value"), Input("sa3-ds-y", "value"))
    def _ds_scatter(xf, yf):
        if not xf or not yf:
            return no_update
        xs, ys, txt = [], [], []
        for c in index:
            xv, yv = _ds_value(c, xf), _ds_value(c, yf)
            if xv is None or yv is None:
                continue
            xs.append(xv); ys.append(yv); txt.append(f"{c.artist} — {c.title}")
        return dataset_tab.scatter_figure(xs, ys, xlabel=xf, ylabel=yf, text=txt)

    inference_tab.register_callbacks(app)
    a2a_tab.register_callbacks(app, index, latent_dir)
