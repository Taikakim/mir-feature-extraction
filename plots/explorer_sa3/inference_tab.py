"""Inference tab — text-to-audio + init-audio (a2a_track) renders via the
SA3 explorer render server (design B.3).

Routing: non-empty `inf-init-path` -> POST /a2a_track, else POST /generate.
The generic steering panel (LatCH / FiLM / DoRA) is shared with the A2A tab
via `controls` (ns="inf"). Long renders block this callback's worker thread
synchronously; the 2 s interval keeps the status badge live meanwhile.

Guidance controls: `cfg_interval` is native SIGMA semantics — the DiT gates
CFG per step on `cfg_interval[0] <= sigma <= cfg_interval[1]`
(stable_audio_3/models/dit.py:479, limited-interval guidance per
Kynkäänniemi 2024) — so the RangeSlider lives in sigma space and step indices
are only a secondary annotation on the schedule chart. The chart itself is a
plotly port of interface/diffusion_cond.py:create_sigma_chart, fed by the
server's /schedule endpoint so it shows the *real* dist-shift-warped run
schedule. Checkpoint picker is fed by the /ckpts journal endpoint.
"""
from __future__ import annotations
import time

import plotly.graph_objects as go
import dash
from dash import Input, Output, State, dcc, html, no_update

from . import bracket, controls, render_client

_HISTORY_CAP = 50

_TIP_APG = ("Adaptive Projected Guidance (Sadat 2024): guidance direction is "
            "split into components parallel/orthogonal to the conditional "
            "prediction. 1.0 = orthogonal-only (cleaner at high CFG), "
            "0.0 = vanilla CFG, in between blends.")
_TIP_CFG_INTERVAL = ("Limited-interval guidance (Kynkäänniemi 2024): CFG is "
                     "applied ONLY at steps whose sigma lies inside this "
                     "interval — native SIGMA semantics, gated in the DiT as "
                     "cfg_interval[0] <= sigma <= cfg_interval[1]. Outside "
                     "the interval the model runs uncond-free (also faster). "
                     "[0,1] = CFG everywhere (default).")
_TIP_DURPAD = ("Silence padding (s) appended after the requested duration "
               "before generation; paper default 6 s. Trimmed on decode.")
_TIP_LADDER = ("Comma-separated init-noise levels, e.g. 0.35,0.42,0.5 — "
               "renders the same seed once per level (a2a_track only); "
               "overrides the single init-noise value.")
_TIP_DIST_SHIFT = ("Sigma-SCHEDULE WARP (not a model input). Blank = the checkpoint's "
                   "trained default (length-dependent shift, paper Eq 3). Number = constant "
                   "override: 1.0 linear; >1 concentrates steps at HIGH sigma (structure); "
                   "<1 at LOW sigma (detail/polish — the under-trained tail). The sigma "
                   "chart redraws with it. Original tip: "
                   "in build_schedule. Leave blank for the model default. "
                   "Length-dependent — duration changes the warp too.")
_TIP_CKPT = ("Checkpoint journal: server-side recursive scan for *.ckpt / "
             "*.safetensors (default root <eval-drive>/sa3_lora_runs, resolved server-side), "
             "cached by mtime+size. Free-text path overrides the dropdown.")
_TIP_DIST_MODE = ("Schedule-warp mode when the numeric dist-shift is blank: "
                  "'default' = the checkpoint's trained sampling_dist_shift; "
                  "'flux' = the stock length-dependent FluxDistributionShift "
                  "(breathing_v2_blockbuild convention). A non-blank number "
                  "always wins (constant-alpha Flux override).")
_MUT_OPS = ("shuffle", "drift", "blur", "contrast", "tilt", "life")


def layout() -> html.Div:
    return html.Div([
        html.Div(id="inf-server-badge"),
        dcc.Interval(id="inf-interval", interval=2000),
        # A preset is the whole form under a name. The viewer snapshot rides in
        # dcc.Store between the two halves of Load -- see _preset_load below for
        # why loading is two callbacks and not one.
        dcc.Store(id="inf-preset-pending"),
        html.Div([
            html.Span("Preset", style={"minWidth": "60px", "fontWeight": "bold"}),
            dcc.Dropdown(id="inf-preset-dd", options=[], value=None, clearable=True,
                         placeholder="saved presets",
                         style={"width": "300px", "display": "inline-block"}),
            html.Button("Load", id="inf-preset-load", n_clicks=0),
            dcc.Input(id="inf-preset-name", type="text", placeholder="save as\u2026",
                      style={"width": "200px"}),
            html.Button("Save", id="inf-preset-save", n_clicks=0),
            html.Span(id="inf-preset-status",
                      style={"fontSize": "11px", "color": "#888"}),
        ], style={"display": "flex", "gap": "6px", "alignItems": "center",
                  "marginBottom": "6px"}),
        html.Div([
            html.Label("Base prompt (persistent)"),
            dcc.Textarea(
                id="inf-prompt",
                placeholder=("TrackType: Music, VocalType: Instrumental, "
                             "... (prefix vocabulary, then free text)"),
                style={"width": "100%", "height": "60px"}),
            html.Label("Variation (appended with ', ' when both non-empty)"),
            dcc.Textarea(
                id="inf-variation",
                placeholder="per-render variation, e.g. 'darker pads, half-time drop'",
                style={"width": "100%", "height": "36px"}),
            html.Label("Negative prompt"),
            dcc.Input(id="inf-negprompt", type="text", value="",
                      style={"width": "100%"}),
        ]),
        html.Div([
            html.Span("roots", title=(
                "Which checkpoint root folders the picker scans. Configured in "
                "SAO/eval/model_roots.json; unavailable = that removable drive is "
                "not mounted (the cached journal is still served, so the entries "
                "stay visible but cannot load). Leave empty for every available root."),
                style={"minWidth": "60px", "fontWeight": "bold",
                       "cursor": "help", "borderBottom": "1px dotted #888"}),
            dcc.Dropdown(id="inf-root-dd", options=[], value=[], multi=True,
                         placeholder="all available roots",
                         style={"flex": "1", "minWidth": "320px"}),
        ], style={"display": "flex", "gap": "6px", "alignItems": "center",
                  "margin": "2px 0"}),
        html.Div([
            html.Span("checkpoint", title=_TIP_CKPT,
                      style={"fontWeight": "bold"}),
            dcc.Dropdown(id="inf-ckpt-dd", options=[], value=None,
                         clearable=True, placeholder="base model (no ckpt)",
                         style={"width": "420px", "display": "inline-block"}),
            html.Button("Rescan", id="inf-ckpt-rescan"),
            html.Span("or path"),
            dcc.Input(id="inf-ckpt-path", type="text", value="",
                      placeholder="free-text ckpt path (overrides dropdown)",
                      style={"width": "30%"}),
            html.Span(id="inf-ckpt-status",
                      style={"fontSize": "11px", "color": "#666"}),
        ], style={"display": "flex", "gap": "6px", "alignItems": "center",
                  "flexWrap": "wrap", "marginTop": "8px"}),
        html.Div(id="inf-model-info",
                 style={"fontSize": "11px", "margin": "2px 0 6px 0",
                        "padding": "4px 6px", "background": "#f6f6f6",
                        "borderLeft": "3px solid #999"}),
        # ---- A/B slots ---------------------------------------------------------
        # Several adapters held RESIDENT on one base. Switching between them costs
        # milliseconds (measured 1.7 s per 8 s render, rebuilt=False) where the ckpt
        # picker above costs a multi-GB reload per switch. Bounded by VRAM, so the
        # server refuses a set that would cross the floor rather than OOMing.
        html.Details([
            html.Summary("A/B slots — hold several adapters resident and switch instantly"),
            html.Div([
                html.Span("slots", title=(
                    "Adapters to keep resident together. Only adapters can share one "
                    "base — a full-FT checkpoint replaces the backbone, so it cannot "
                    "be a slot. ~0.40 GB each at rank 128 (measured)."),
                    style={"minWidth": "60px", "fontWeight": "bold",
                           "cursor": "help", "borderBottom": "1px dotted #888"}),
                dcc.Dropdown(id="inf-slot-dd", options=[], value=[], multi=True,
                             placeholder="pick up to 3–4 adapters",
                             style={"flex": "1", "minWidth": "360px"}),
                html.Button("Make resident", id="inf-slot-apply", n_clicks=0),
            ], style={"display": "flex", "gap": "6px", "alignItems": "center",
                      "margin": "4px 0"}),
            html.Div(id="inf-slot-status",
                     style={"fontSize": "11px", "color": "#666", "margin": "2px 0"}),
            html.Div([
                html.Span("compare", style={"minWidth": "60px", "fontWeight": "bold"}),
                dcc.Checklist(id="inf-ab-arms", options=[], value=[],
                              inline=True, style={"display": "inline-block"}),
                html.Button("Render A/B", id="inf-ab-go", n_clicks=0),
                html.Span(id="inf-ab-status",
                          style={"fontSize": "11px", "color": "#666"}),
            ], style={"display": "flex", "gap": "8px", "alignItems": "center",
                      "flexWrap": "wrap", "margin": "4px 0"}),
            html.Div(id="inf-ab-results"),
        ], open=False, style={"margin": "4px 0 8px"}),
        html.Div([
            html.Span("duration (s)"),
            dcc.Input(id="inf-duration", type="number", value=47, min=1,
                      max=378, step=1),
            html.Span("steps"),
            dcc.Input(id="inf-steps", type="number", value=24, min=1, max=500),
            html.Span("cfg"),
            dcc.Input(id="inf-cfg", type="number", value=6.0, min=0, max=25,
                      step=0.1),
            html.Span("seed"),
            dcc.Input(id="inf-seed", type="number", value=-1),
            html.Span("batch"),
            dcc.Input(id="inf-batch", type="number", value=1, min=1, max=4),
            html.Span("apg", title=_TIP_APG,
                      style={"textDecoration": "underline dotted"}),
            dcc.Input(id="inf-apg", type="number", value=1.0, min=0, max=1,
                      step=0.1),
            html.Span("dur pad (s)", title=_TIP_DURPAD,
                      style={"textDecoration": "underline dotted"}),
            dcc.Input(id="inf-durpad", type="number", value=6.0, min=0, max=30,
                      step=0.5),
            html.Span("dist shift", title=_TIP_DIST_SHIFT,
                      style={"textDecoration": "underline dotted"}),
            dcc.Input(id="inf-dist-shift", type="number", value=None,
                      placeholder="ckpt default", style={"width": "90px"}),
            html.Span("mode", title=_TIP_DIST_MODE,
                      style={"textDecoration": "underline dotted"}),
            dcc.Dropdown(id="inf-dist-mode", clearable=False, value="default",
                         options=[{"label": m, "value": m}
                                  for m in ("default", "flux")],
                         style={"width": "110px", "display": "inline-block"}),
        ], style={"display": "flex", "gap": "6px", "alignItems": "center",
                  "flexWrap": "wrap", "marginTop": "8px"}),
        html.Div([
            html.Span("init audio path (non-empty → a2a_track)"),
            dcc.Input(id="inf-init-path", type="text", value="",
                      style={"width": "50%"}),
            html.Span("init noise"),
            dcc.Input(id="inf-init-noise", type="number", value=0.4, min=0.05,
                      max=0.95, step=0.01),
            html.Span("noise ladder (e.g. 0.35,0.42,0.5)", title=_TIP_LADDER,
                      style={"textDecoration": "underline dotted"}),
            dcc.Input(id="inf-noise-ladder", type="text", value=""),
        ], style={"display": "flex", "gap": "6px", "alignItems": "center",
                  "flexWrap": "wrap", "marginTop": "8px"}),
        html.Div([
            html.Label("CFG interval (σ)", title=_TIP_CFG_INTERVAL,
                       style={"fontWeight": "bold",
                              "textDecoration": "underline dotted"}),
            dcc.RangeSlider(
                id="inf-cfg-interval", min=0.0, max=1.0, step=0.01,
                value=[0.0, 1.0], allowCross=False,
                marks={0: "σ=0", 0.25: "0.25", 0.5: "0.5",
                       0.75: "0.75", 1: "σ=1"},
                tooltip={"placement": "bottom", "always_visible": False}),
            dcc.Loading(dcc.Graph(
                id="inf-sigma-graph",
                config={"displayModeBar": False},
                style={"height": "280px"})),
        ], style={"marginTop": "8px"}),
        controls.steering_panel("inf", dora_default="none"),
        html.Details([
            html.Summary("Weight garden — DiT weight mutations (databending)"),
            dcc.Checklist(id="inf-mut-on",
                          options=[{"label": " enable (rebuilds model)", "value": "on"}],
                          value=[]),
            html.Span("op (shuffle = the proven-musical one)"),
            dcc.Dropdown(id="inf-mut-op", clearable=False, value="shuffle",
                         options=[{"label": o, "value": o} for o in _MUT_OPS]),
            html.Span("amount (fraction mutated)"),
            dcc.Slider(id="inf-mut-amount", min=0.05, max=1.0, step=0.05, value=0.25,
                       marks={0.05: "0.05", 0.25: "0.25", 0.5: "0.5", 1.0: "1"}),
            html.Span("target"),
            dcc.Dropdown(id="inf-mut-target", clearable=False, value="attn",
                         options=[{"label": t, "value": t}
                                  for t in ("attn", "mlp", "norm", "all")]),
            html.Span("mutation seed"),
            dcc.Input(id="inf-mut-seed", type="number", value=1234),
            html.Span("decay direction"),
            dcc.Dropdown(id="inf-mut-decay", clearable=False, value="late",
                         options=[{"label": d, "value": d} for d in ("late", "early")]),
            html.Span("decay rate"),
            dcc.Slider(id="inf-mut-decay-rate", min=0.0, max=1.0, step=0.1, value=0.5,
                       marks={0: "0", 0.5: "0.5", 1: "1"}),
            html.Span("life quantile (alive |w| threshold; life op only)"),
            dcc.Slider(id="inf-mut-quantile", min=0.05, max=0.99, step=0.01,
                       value=0.75,
                       marks={0.05: "0.05", 0.5: "0.5", 0.75: "0.75",
                              0.99: "0.99"}),
        ], open=False),
        html.Details([
            html.Summary("Rhythm preserve — selection steering (a2a only)"),
            dcc.Checklist(id="inf-pres-on",
                          options=[{"label": " enable (ping-pong sampler, best-of-K "
                                             "renoise scored by LatCH head vs source "
                                             "envelope)", "value": "on"}],
                          value=[]),
            html.Span("scoring head"),
            dcc.Dropdown(id="inf-pres-head", clearable=False, value="onset_envelope",
                         options=[{"label": h, "value": h}
                                  for h in ("onset_envelope", "onset_envelope_drums",
                                            "rms_energy_bass", "rms_drums")]),
            html.Span("K candidates"),
            dcc.Input(id="inf-pres-k", type="number", value=4, min=2, max=16),
            html.Span("active until (fraction of steps)"),
            dcc.Slider(id="inf-pres-until", min=0.1, max=1.0, step=0.05, value=0.5,
                       marks={0.1: "0.1", 0.5: "0.5", 1.0: "1"}),
        ], open=False),
        bracket.layout(),
        html.Button("Render", id="inf-render-btn"),
        html.Pre(id="inf-status", style={"whiteSpace": "pre-wrap"}),
        html.Div(id="inf-result"),
        dcc.Store(id="inf-history", data=[]),
        html.H4("History"),
        html.Div(id="inf-history-div"),
    ])


def _players(urls, files) -> list:
    out = []
    for u, f in zip(urls, files or [None] * len(urls)):
        out.append(html.Div([
            html.Audio(src=render_client.audio_url(u), controls=True),
            html.Div(f or "", style={"fontSize": "11px", "color": "#666"}),
        ]))
    return out


def _parse_ladder(text: str) -> list[float] | None:
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    if not parts:
        return None
    return [float(p) for p in parts]


def _full_prompt(base: str | None, variation: str | None) -> str:
    """Base + ', ' + variation when both non-empty; else whichever is set."""
    b = (base or "").strip()
    v = (variation or "").strip()
    if b and v:
        return f"{b}, {v}"
    return b or v


def resolve_dist_shift_value(dist_shift, dist_mode):
    """Payload/`/schedule` dist_shift from the numeric + mode pair: a non-blank
    number always wins (constant-alpha Flux); else mode 'flux' -> the string
    "flux" (stock length-dependent warp); else None (= ckpt default, key
    omitted from payloads)."""
    if dist_shift not in (None, ""):
        return float(dist_shift)
    if dist_mode == "flux":
        return "flux"
    return None


# ids consumed by build_payload, in order (the bracket panel snapshots the
# same form through form_states(); keep the two in lockstep)
_FORM_IDS = ("inf-prompt", "inf-variation", "inf-negprompt", "inf-duration",
             "inf-steps", "inf-cfg", "inf-cfg-interval", "inf-seed",
             "inf-batch", "inf-apg", "inf-durpad", "inf-dist-shift",
             "inf-dist-mode", "inf-ckpt-dd", "inf-ckpt-path", "inf-init-path",
             "inf-init-noise", "inf-noise-ladder",
             "inf-mut-on", "inf-mut-op", "inf-mut-amount", "inf-mut-target",
             "inf-mut-seed", "inf-mut-decay", "inf-mut-decay-rate",
             "inf-mut-quantile",
             "inf-pres-on", "inf-pres-head", "inf-pres-k", "inf-pres-until")


def form_states() -> list[State]:
    """The full render form as Dash States: the 30 plain fields above followed
    by the frozen 35-value steering contract (consumed as-is — new features
    ride their own State groups, per the contract note in controls.py)."""
    return ([State(i, "value") for i in _FORM_IDS]
            + controls.steering_states("inf"))


# A preset is a RECIPE, so the two fields that describe THIS render are not
# restored: loading a preset must not silently re-pin the seed you were varying.
_VOLATILE_FORM_IDS = ("inf-seed", "inf-batch")

# Loading is split in two because of a callback race. Setting a LatCH slot's head
# fires controls._autofill_defaults, which writes that slot's gain/kind/value from
# the head's own checkpoint defaults -- so anything the preset wrote to those in
# the same response is immediately overwritten. The fix is to make the second half
# DEPEND on the first: _preset_apply_slots is triggered by the slot meter, which is
# an Output of _autofill_defaults, so Dash cannot run it too early.
_SLOT_LATE_SUFFIXES = ("kind", "value", "gain", "start", "end", "loss", "wsec")


def _late_ids() -> list[str]:
    return [f"inf-ctl-latch{i}-{sfx}"
            for i in range(1, controls.LATCH_SLOTS + 1)
            for sfx in _SLOT_LATE_SUFFIXES]


def _early_ids() -> list[str]:
    late = set(_late_ids())
    return [st.component_id for st in form_states() if st.component_id not in late]


def build_payload(vals) -> tuple[str, dict]:
    """form_states() values -> (op, payload) exactly as the Render button sends
    them (op routing: non-empty init path -> a2a_track, else generate). Shared
    by the single-render callback and the bracket fan-out so the two cannot
    drift. Raises ValueError on an empty prompt / bad ladder / steering error."""
    (base_prompt, variation, negprompt, duration, steps, cfg, cfg_interval,
     seed, batch, apg, durpad, dist_shift, dist_mode, ckpt_dd, ckpt_path,
     init_path, init_noise, ladder,
     mut_on, mut_op, mut_amount, mut_target, mut_seed, mut_decay,
     mut_decay_rate, mut_quantile,
     pres_on, pres_head, pres_k, pres_until) = \
        vals[:len(_FORM_IDS)]
    steer = list(vals[len(_FORM_IDS):])
    prompt = _full_prompt(base_prompt, variation)
    if not prompt:
        raise ValueError("error: prompt is empty")
    try:
        steering = controls.steering_payload(steer)
    except Exception as e:
        raise ValueError(f"steering error: {e}") from e
    lo, hi = (cfg_interval or [0.0, 1.0])
    common = {"steps": int(steps or 24), "cfg_scale": float(cfg or 6.0),
              "seed": int(seed if seed is not None else -1),
              "cfg_interval": [float(lo), float(hi)],
              "apg_scale": float(apg if apg is not None else 1.0)}
    ds = resolve_dist_shift_value(dist_shift, dist_mode)
    if ds is not None:
        common["dist_shift"] = ds
    ckpt = (ckpt_path or "").strip() or (ckpt_dd or "").strip()
    if ckpt:
        common["ckpt_path"] = ckpt
    if mut_on and "on" in mut_on:
        common["mutate"] = {"enabled": True,
                            "op": mut_op or "shuffle",
                            "amount": float(mut_amount or 0.25),
                            "target": mut_target or "attn",
                            "seed": int(mut_seed or 1234),
                            "decay_rate": float(mut_decay_rate
                                                if mut_decay_rate is not None else 0.5),
                            "decay_direction": mut_decay or "late",
                            # resolve_mutate reads this for op=life only
                            "quantile": float(mut_quantile
                                              if mut_quantile is not None else 0.75)}
    if (init_path or "").strip():
        op = "a2a_track"
        payload = {"audio_path": init_path.strip(),
                   "prompt": prompt,
                   "noise_level": float(init_noise or 0.4),
                   **common, **steering}
        try:
            levels = _parse_ladder(ladder)
        except ValueError:
            raise ValueError(f"bad noise ladder: {ladder!r}") from None
        if levels:
            payload["noise_levels"] = levels
        if pres_on and "on" in pres_on:
            payload["preserve"] = {"enabled": True,
                                   "head": pres_head or "onset_envelope",
                                   "k": int(pres_k or 4),
                                   "until": float(pres_until
                                                  if pres_until is not None else 0.5)}
    else:
        op = "generate"
        payload = {"prompt": prompt,
                   "negative_prompt": negprompt or "",
                   "duration": float(duration or 47),
                   "batch_size": int(batch or 1),
                   "duration_padding_sec": float(
                       durpad if durpad is not None else 6.0),
                   **common, **steering}
    return op, payload


def _sigma_figure(steps, duration, dist_shift, interval, cfg,
                  latch_windows) -> go.Figure:
    """Plotly port of interface/diffusion_cond.py:create_sigma_chart.

    Curves: sigma (descending) + progress = 1-sigma over step index.
    CFG band: exactly the steps whose sigma satisfies the DiT gate
    lo <= sigma <= hi (sigma semantics; step indices are annotation only).
    Latch windows are fractions of total steps (yellow), with the
    LatCH ∩ CFG overlap in orange.
    """
    steps = max(1, int(steps or 24))
    duration = float(duration or 47)
    ds = dist_shift          # resolved value: float | "flux" | None
    lo, hi = (interval or [0.0, 1.0])
    lo, hi = float(lo), float(hi)

    sig = render_client.schedule(steps, duration, ds)
    src = "server /schedule"
    if sig is None:
        sig = [1.0 - i / steps for i in range(steps + 1)]
        src = "LINEAR FALLBACK — server down, no dist-shift warp"
    x = list(range(len(sig)))

    fig = go.Figure()
    fig.add_scatter(x=x, y=sig, mode="lines+markers", name="sigma",
                    line={"color": "#1f77b4"}, marker={"size": 4})
    fig.add_scatter(x=x, y=[1.0 - s for s in sig], mode="lines",
                    name="progress (1−σ)",
                    line={"color": "#17becf", "dash": "dash"})

    cfg_band = None
    idx = [i for i, s in enumerate(sig) if lo <= s <= hi]
    if idx:
        cfg_band = (min(idx), max(idx))
        fig.add_vrect(
            x0=cfg_band[0], x1=cfg_band[1], fillcolor="green", opacity=0.15,
            line_width=0,
            annotation_text=(f"CFG σ∈[{lo:.2f},{hi:.2f}] "
                             f"(steps {cfg_band[0]}–{cfg_band[1]})"),
            annotation_position="top left")

    for (ls, le) in latch_windows:
        l0, l1 = float(ls) * steps, float(le) * steps
        if l1 <= l0:
            continue
        fig.add_vrect(x0=l0, x1=l1, fillcolor="gold", opacity=0.15,
                      line_width=0, annotation_text="LatCH",
                      annotation_position="bottom left")
        if cfg_band:
            o0, o1 = max(l0, cfg_band[0]), min(l1, cfg_band[1])
            if o0 < o1:
                fig.add_vrect(x0=o0, x1=o1, fillcolor="orange", opacity=0.25,
                              line_width=0)

    fig.update_layout(
        title={"text": (f"σ schedule · steps={steps} · dur={duration:g}s · "
                        f"dist_shift={ds if ds is not None else 'model'} · "
                        f"cfg={cfg} · [{src}]"),
               "font": {"size": 12}},
        xaxis_title="step", yaxis_title="value",
        yaxis_range=[0, 1.05], height=280,
        margin={"l": 40, "r": 10, "t": 40, "b": 30},
        legend={"orientation": "h", "y": 1.12})
    return fig


def register_callbacks(app) -> None:
    # ------------------------------------------------------------- A/B slots
    @app.callback(Output("inf-slot-dd", "options"),
                  Input("inf-ctl-info-poll", "n_intervals"),
                  State("inf-slot-dd", "options"))
    def _slot_options(_n, current):
        if current:
            return no_update
        resp = render_client.models(family="adapter", loadable=True)
        if not resp:
            return []
        # Only adapters, and only ones actually on a mounted drive: offering a
        # checkpoint the server will refuse is the same lie the DoRA picker used to
        # tell before the drive-offline marking.
        return [{"label": f"{m.get('label') or m['path'].rsplit('/', 1)[-1]}"
                          f"{'  · r' + str(m['rank']) if m.get('rank') else ''}"
                          f"  [{m.get('root_id', '?')}]",
                 "value": m["path"]}
                for m in resp.get("models", []) if m.get("path")]

    @app.callback(Output("inf-slot-status", "children"),
                  Output("inf-ab-arms", "options"),
                  Output("inf-ab-arms", "value"),
                  Input("inf-slot-apply", "n_clicks"),
                  Input("inf-ctl-info-poll", "n_intervals"),
                  State("inf-slot-dd", "value"),
                  State("inf-ab-arms", "value"))
    def _slot_apply(n_clicks, _n, paths, current_arms):
        trig = dash.callback_context.triggered_id
        if trig == "inf-slot-apply" and n_clicks:
            try:
                st = render_client.set_slots([{"ckpt_path": p} for p in (paths or [])])
            except render_client.RenderError as e:
                return f"refused: {e}", no_update, no_update
        else:
            st = render_client.slots()
            if st is None:
                return "render server :8056 unreachable — retrying every 5 s", [], []
        slots = st.get("slots", [])
        # `null` first: the control arm. An adapter that "sounds better" against
        # nothing is not a finding.
        opts = [{"label": "base (no adapter)", "value": "base"}] + [
            {"label": f"{s['label'][:40]}", "value": str(s["index"])} for s in slots]
        keep = [v for v in (current_arms or []) if v in {o["value"] for o in opts}]
        if not keep:
            keep = [o["value"] for o in opts]
        msg = (f"{len(slots)} resident · {st.get('free_gb', '?')} GB free · floor "
               f"{st.get('vram_floor_gb', '?')} GB · max {st.get('max_slots', '?')}"
               + (f" · {sum(s['cost_gb'] for s in slots):.2f} GB in slots" if slots else ""))
        return msg, opts, keep

    @app.callback(Output("inf-ab-results", "children"),
                  Output("inf-ab-status", "children"),
                  Input("inf-ab-go", "n_clicks"),
                  State("inf-ab-arms", "value"),
                  *form_states(),
                  prevent_initial_call=True)
    def _ab_go(n, arms, *vals):
        if not n:
            return no_update, no_update
        if not arms or len(arms) < 2:
            return no_update, "pick at least two arms (base counts as one)"
        try:
            op, payload = build_payload(list(vals))
        except ValueError as e:
            return no_update, str(e)
        if op != "generate":
            return no_update, "A/B is the t2a path only — clear the init audio path"
        payload["ab"] = {"slots": [None if a == "base" else int(a) for a in arms]}
        try:
            resp = render_client.ab(payload)
        except render_client.RenderError as e:
            return no_update, f"A/B failed: {e}"
        cells = []
        for a in resp.get("arms", []):
            if a.get("error"):
                cells.append(html.Div([html.B(a["label"]),
                                       html.Div(f"ERROR: {a['error']}",
                                                style={"color": "#b00"})],
                                      style={"minWidth": "260px"}))
                continue
            url = render_client.audio_url(a["urls"][0]) if a.get("urls") else None
            cells.append(html.Div([
                html.B(a["label"]),
                html.Div(f"slot {a['slot']} · strength {a.get('strength')}",
                         style={"fontSize": "11px", "color": "#888"}),
                html.Audio(src=url, controls=True, style={"width": "250px"})
                if url else html.Div("no file", style={"color": "#b00"}),
            ], style={"minWidth": "270px", "padding": "4px 8px",
                      "borderLeft": "3px solid #999"}))
        note = (f"seed {resp.get('seed')} — SAME seed on every arm, so they differ "
                f"only by the model · {resp['timings']['total_sec']}s")
        return (html.Div(cells, style={"display": "flex", "gap": "8px",
                                       "flexWrap": "wrap", "margin": "6px 0"}),
                note)

    # ---------------------------------------------------------------- presets
    @app.callback(Output("inf-preset-dd", "options"),
                  Input("inf-ctl-info-poll", "n_intervals"),
                  Input("inf-preset-save", "n_clicks"))
    def _preset_options(_n, _saved):
        lst = render_client.presets_list()
        if lst is None:
            return []
        return [{"label": f"{d['name']}  \u00b7  {(d.get('prompt') or '')[:60]}",
                 "value": d["name"]} for d in lst]

    @app.callback(Output("inf-preset-status", "children"),
                  Input("inf-preset-save", "n_clicks"),
                  State("inf-preset-name", "value"),
                  *form_states(),
                  prevent_initial_call=True)
    def _preset_save(n, name, *vals):
        if not n:
            return no_update
        name = (name or "").strip()
        if not name:
            return "type a name in the box first"
        try:
            # The SAME builder the Render button uses -- a preset that does not
            # round-trip to the same render is worse than no preset.
            op, payload = build_payload(list(vals))
        except ValueError as e:
            return str(e)
        ids = [st.component_id for st in form_states()]
        form = {i: v for i, v in zip(ids, vals) if i not in _VOLATILE_FORM_IDS}
        resp = render_client.preset_save(name, payload, notes=f"op={op}", form=form)
        if not resp:
            return "save failed — render server :8056 unreachable"
        return f"saved “{name}” ({op}, seed not stored)"

    @app.callback(
        *[Output(i, "value", allow_duplicate=True) for i in _early_ids()],
        Output("inf-preset-pending", "data"),
        Output("inf-preset-status", "children", allow_duplicate=True),
        Input("inf-preset-load", "n_clicks"),
        State("inf-preset-dd", "value"),
        prevent_initial_call=True)
    def _preset_load(n, name):
        early = _early_ids()
        blank = [no_update] * len(early)
        if not n:
            return (*blank, no_update, no_update)
        if not name:
            return (*blank, no_update, "pick a preset from the dropdown first")
        rec = render_client.preset_load(name)
        if not rec:
            return (*blank, no_update, f"could not load “{name}”")
        form = rec.get("form") or {}
        if not form:
            return (*blank, no_update,
                    f"“{name}” has no viewer snapshot (written by a CLI or an "
                    f"older build) — it is still usable from the sweep CLI")
        late = {i: form[i] for i in _late_ids() if i in form}
        return (*[form.get(i, no_update) for i in early], late,
                f"loaded “{name}” (seed/batch left as they are)")

    @app.callback(
        *[Output(i, "value", allow_duplicate=True) for i in _late_ids()],
        Output("inf-preset-pending", "data", allow_duplicate=True),
        *[Input(f"inf-ctl-latch{i}-meter", "children")
          for i in range(1, controls.LATCH_SLOTS + 1)],
        State("inf-preset-pending", "data"),
        prevent_initial_call=True)
    def _preset_apply_slots(*args):
        """Second half of Load. Runs after _autofill_defaults (it is triggered by
        that callback's own meter output), so the preset's per-slot numbers land
        on top of the head defaults instead of under them. The pending store is
        cleared on use, so a later manual head change finds nothing to re-apply."""
        pending = args[-1]
        late = _late_ids()
        if not pending:
            return (*[no_update] * len(late), no_update)
        return (*[pending.get(i, no_update) for i in late], None)

    controls.register(app, "inf")
    bracket.register(app, form_states, build_payload)

    @app.callback(Output("inf-server-badge", "children"),
                  Input("inf-interval", "n_intervals"))
    def _badge(_n):
        st = render_client.status()
        if st is None:
            return html.Div([
                html.B("Render server DOWN — launch with:"),
                html.Pre(render_client.LAUNCH_HINT),
            ], style={"color": "#a00"})
        tail = "\n".join((st.get("log_tail") or [])[-5:])
        label = ("BUSY — " + str(st.get("job_id"))) if st.get("busy") else "idle"
        return html.Div([
            html.B(f"Render server up · {label}"),
            html.Pre(tail, style={"fontSize": "11px", "margin": "2px 0"}),
        ], style={"color": "#070"})

    @app.callback(Output("inf-root-dd", "options"),
                  Input("inf-ctl-info-poll", "n_intervals"),
                  Input("inf-ckpt-rescan", "n_clicks"),
                  State("inf-root-dd", "options"))
    def _root_options(_n, n_clicks, current):
        # Same retry-until-filled shape as the other pickers, plus a forced refresh
        # on Rescan so plugging a drive in is one click, not a page reload.
        if current and dash.callback_context.triggered_id == "inf-ctl-info-poll":
            return no_update
        resp = render_client.roots()
        if resp is None:
            return []
        return [{"label": (f"{r['label']} ({r['count']})"
                           + ("" if r["available"] else "  — DRIVE NOT MOUNTED")),
                 "value": r["id"],
                 "disabled": not (r["available"] or r["count"])}
                for r in resp.get("roots", [])]

    @app.callback(Output("inf-ckpt-dd", "options"),
                  Output("inf-ckpt-status", "children"),
                  Input("inf-ckpt-rescan", "n_clicks"),
                  Input("inf-root-dd", "value"),
                  Input("inf-ctl-info-poll", "n_intervals"),
                  State("inf-ckpt-dd", "options"))
    def _ckpt_journal(n_clicks, root_ids, _n, current):
        # Three triggers: Rescan (force a fresh walk), the root selector (rescope),
        # and the shared 5 s poller. The poller exists because this used to fire once
        # at load and never again — with :8056 down at load the picker stayed empty
        # forever. Once filled, the poller path short-circuits here.
        trig = dash.callback_context.triggered_id
        rescan = trig == "inf-ckpt-rescan" and bool(n_clicks)
        if trig == "inf-ctl-info-poll" and current:
            return no_update, no_update
        resp = render_client.ckpts(rescan=rescan, root_ids=root_ids or None)
        if resp is None:
            return [], "render server :8056 unreachable — retrying every 5 s"
        if not resp.get("ok", True):
            return [], f"ckpt root unavailable: {resp.get('error', 'unknown')}"
        root = (resp.get("root") or "").rstrip("/")
        opts = []
        for c in resp.get("ckpts", []):
            path = c.get("path", "")
            label = path[len(root) + 1:] if root and path.startswith(root + "/") else path
            bits = [b for b in (c.get("family"), c.get("corpus"),
                                f"r{c['rank']}" if c.get("rank") else None,
                                f"ep{c['epoch']}" if c.get("epoch") is not None else None,
                                "verdict" if c.get("verdict") else None) if b]
            if bits:
                label += "   [" + " · ".join(bits) + "]"
            size = c.get("size")
            if size:
                label += f"  ({size / 1e6:.0f} MB)"
            opts.append({"label": label, "value": path})
        stale = resp.get("stale_root_ids") or []
        note = f" · STALE (unmounted): {', '.join(stale)}" if stale else ""
        scope = ", ".join(root_ids) if root_ids else (root or "all available roots")
        return opts, f"{len(opts)} ckpts · {scope}{note}"

    @app.callback(Output("inf-model-info", "children"),
                  Input("inf-ckpt-dd", "value"),
                  Input("inf-ckpt-path", "value"))
    def _model_info(dd_val, free_path):
        """Three-audience standard (eval-tables spec §14): what this model IS, the
        reproducible recipe, and the plain-language why -- at the moment of picking
        it, which is the only moment it changes a decision."""
        path = (free_path or "").strip() or (dd_val or "").strip()
        if not path:
            return "no checkpoint selected — renders use the base model (medium-base)."
        resp = render_client.models(q=path)
        recs = [m for m in ((resp or {}).get("models") or []) if m.get("path") == path]
        if not recs:
            return f"not in the model DB: {path}"
        m = recs[0]
        head = " · ".join(str(x) for x in (
            m.get("family"), m.get("corpus") or "corpus?",
            f"r{m['rank']}/a{m['alpha']}" if m.get("rank") else None,
            m.get("adapter_type"),
            f"ep{m['epoch']} step{m['step']}" if m.get("epoch") is not None else None,
            f"{m['load_cost_gb']} GB resident" if m.get("load_cost_gb") else None) if x)
        rows = [html.B(m.get("label") or path), html.Span("  " + head)]
        for field, prefix in (("recipe", "recipe"), ("purpose", "why"),
                              ("verdict", "verdict"), ("kim_feedback", "Kim")):
            v = m.get(field)
            if not v:
                continue
            if isinstance(v, dict):
                # A recipe arrives as a dict of ~12 fields. Dumping str(dict) puts a
                # 1000-character Python repr on one line -- readable to nobody. One
                # line per field, indented under the heading.
                rows += [html.Br(), html.B(prefix + ":")]
                for k in sorted(v):
                    rows += [html.Br(),
                             html.Span(f"\u2003{k}: {v[k]}",
                                       style={"color": "#444"})]
            else:
                rows += [html.Br(), html.Span(f"{prefix}: {v}")]
        if m.get("run_meta_path"):
            rows += [html.Br(),
                     html.A("run_meta.json", href="file://" + m["run_meta_path"],
                            target="_blank")]
        prov = m.get("provenance") or {}
        if prov:
            rows += [html.Br(),
                     html.Span("provenance: " + ", ".join(f"{k}={v}" for k, v
                                                          in sorted(prov.items())),
                               style={"color": "#777"})]
        return rows

    @app.callback(
        Output("inf-sigma-graph", "figure"),
        Input("inf-steps", "value"),
        Input("inf-duration", "value"),
        Input("inf-dist-shift", "value"),
        Input("inf-dist-mode", "value"),
        Input("inf-cfg-interval", "value"),
        Input("inf-cfg", "value"),
        *[Input(f"inf-ctl-latch{i}-{s}", "value")
          for i in range(1, controls.LATCH_SLOTS + 1)
          for s in ("head", "start", "end")])
    def _sigma_chart(steps, duration, dist_shift, dist_mode, interval, cfg,
                     *latch):
        windows = []
        for i in range(controls.LATCH_SLOTS):
            head, start, end = latch[i * 3:(i + 1) * 3]
            if head in (None, "none", ""):
                continue
            windows.append((start if start is not None else 0.0,
                            end if end is not None else 0.6))
        ds = resolve_dist_shift_value(dist_shift, dist_mode)
        return _sigma_figure(steps, duration, ds, interval, cfg, windows)

    @app.callback(
        Output("inf-result", "children"),
        Output("inf-history", "data"),
        Output("inf-status", "children"),
        Input("inf-render-btn", "n_clicks"),
        *form_states(),
        State("inf-history", "data"),
        prevent_initial_call=True)
    def _render(_n, *vals):
        history = vals[-1]
        try:
            op, payload = build_payload(list(vals[:-1]))
        except ValueError as e:
            return no_update, no_update, str(e)
        try:
            resp = render_client.render(op, payload)
        except render_client.RenderError as e:
            return no_update, no_update, str(e)
        urls = resp.get("urls", [])
        files = resp.get("files", [])
        entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "op": op,
                 "label": payload["prompt"][:60],
                 "urls": urls, "files": files,
                 "meta": resp.get("meta", {})}
        history = ([entry] + (history or []))[:_HISTORY_CAP]
        status_txt = (f"ok · op={op} · seed={resp.get('seed')} · "
                      f"total={resp.get('timings', {}).get('total_sec')}s\n"
                      f"warnings: {resp.get('warnings')}\n"
                      f"meta: {resp.get('meta')}")
        return _players(urls, files), history, status_txt

    @app.callback(Output("inf-history-div", "children"),
                  Input("inf-history", "data"))
    def _history(data):
        rows = []
        for e in (data or []):
            rows.append(html.Div([
                html.B(f"{e.get('ts')} [{e.get('op')}] {e.get('label')}"),
                *_players(e.get("urls", []), e.get("files", [])),
            ], style={"borderBottom": "1px solid #ccc",
                      "padding": "4px 0"}))
        return rows
