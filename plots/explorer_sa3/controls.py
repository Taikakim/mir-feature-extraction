"""Shared steering panel (LatCH slots + FiLM + DoRA) for the SA3 explorer.

Frozen contract (design B.2). Dash ids are app-global, so the panel is
prefix-parameterized: instantiate once per tab with ns="inf" and ns="a2a".
`steering_states(ns)` returns EXACTLY 23 States in a fixed order which
`steering_payload(vals)` consumes to build the shared latch/film/dora JSON
blocks accepted by /generate, /a2a_track and /a2a_mix (design A.4).
"""
from __future__ import annotations
import os
from typing import Sequence

from dash import Input, Output, State, dcc, html, no_update

from . import render_client

LATCH_SLOTS = 3

# fallback when /info is unreachable at payload-build time (server hard-codes
# the same default in film_default)
# The eval drive is REMOVABLE and udisks labels it Mantu or Mantu1 depending on whether the
# label collided at mount time — the same reason explorer_render_server.py resolves it at import
# rather than hardcoding. This fallback hardcoded Mantu1, which does not exist today (the drive
# mounts as Mantu), so the degraded path — used only when /info is unreachable at payload-build
# time — pointed at a file that is not there. Verified 2026-08-23: the same ckpt IS under Mantu.
def _resolve_mantu():
    for _d in ("/run/media/kim/Mantu", "/run/media/kim/Mantu1"):
        if os.path.isdir(os.path.join(_d, "sa3_control_runs")):
            return _d
    return "/run/media/kim/Mantu"


_FILM_CKPT_FALLBACK = os.path.join(
    _resolve_mantu(), "sa3_control_runs", "onset_Fusion_lr1e-4_randomcrop", "riffer_final.pt")

_KIND_OPTIONS = ["constant", "ramp_up", "ramp_down", "beat_grid"]
# loss shaping per head (latch_guided.head_loss); "" = head/server default.
# scalar_pooled = the 2026-07-10 buzz fix for scalar heads; chroma_rung* = T2.
_LOSS_OPTIONS = ["", "mse", "smooth_l1", "l1", "huber", "bce_logits",
                 "cosine", "scalar_pooled", "chroma_rung1", "chroma_rung2"]
_DORA_OPTIONS = ["none", "hof", "newstack", "evr1x"]

_ROW_STYLE = {"display": "flex", "gap": "6px", "alignItems": "center",
              "flexWrap": "wrap", "marginBottom": "4px"}
_NUM_STYLE = {"width": "90px"}


def _latch_slot(ns: str, i: int) -> html.Div:
    p = f"{ns}-ctl-latch{i}"
    return html.Div([
        html.Span(f"LatCH {i}", style={"minWidth": "60px", "fontWeight": "bold"}),
        dcc.Dropdown(id=f"{p}-head", options=[], value=None, clearable=True,
                     placeholder="head (off)",
                     style={"width": "260px", "display": "inline-block"}),
        dcc.Dropdown(id=f"{p}-kind", options=_KIND_OPTIONS, value="constant",
                     clearable=False,
                     style={"width": "130px", "display": "inline-block"}),
        html.Span("value", id=f"{p}-vlabel", style=_HELP_STYLE),
        dcc.Input(id=f"{p}-value", type="number", value=-30.0, step=0.1,
                  style=_NUM_STYLE),
        html.Span("gain"),
        dcc.Input(id=f"{p}-gain", type="number", value=512.0, min=0, step=1,
                  style=_NUM_STYLE),
        html.Span("start", title=_WINDOW_HELP, style=_HELP_STYLE),
        dcc.Input(id=f"{p}-start", type="number", value=0.0, min=0, max=1,
                  step=0.01, style=_NUM_STYLE),
        html.Span("end", title=_WINDOW_HELP, style=_HELP_STYLE),
        dcc.Input(id=f"{p}-end", type="number", value=0.6, min=0, max=1,
                  step=0.01, style=_NUM_STYLE),
        html.Span("loss", id=f"{p}-llabel", style=_HELP_STYLE),
        dcc.Dropdown(id=f"{p}-loss", options=_LOSS_OPTIONS, value="",
                     clearable=False, placeholder="default",
                     style={"width": "130px", "display": "inline-block"}),
        html.Span("w_sec"),
        dcc.Input(id=f"{p}-wsec", type="number", value=None, min=0, step=0.1,
                  placeholder="def", style=_NUM_STYLE),
        # Display-only. NEVER add these to steering_states() -- the payload builder
        # consumes that list positionally.
        html.Span(id=f"{p}-health", style={"fontSize": "11px", "fontWeight": "bold"}),
        html.Div(id=f"{p}-meter", style={"flexBasis": "100%", "fontSize": "11px",
                                         "color": "#888", "marginLeft": "66px"}),
    ], style=_ROW_STYLE)


def steering_panel(ns: str, dora_default: str = "none",
                   collapsed: bool = True) -> html.Div:
    """Steering panel with ids {ns}-ctl-* (see design B.2 table)."""
    body = html.Div([
        *[_latch_slot(ns, i) for i in range(1, LATCH_SLOTS + 1)],
        html.Div([
            dcc.Checklist(id=f"{ns}-ctl-film-enable",
                          options=[{"label": " FiLM density adapter",
                                    "value": "on"}],
                          value=[],
                          style={"display": "inline-block"}),
            html.Span("onsets/s"),
            dcc.Input(id=f"{ns}-ctl-film-value", type="number", value=4.0,
                      step=0.1, style=_NUM_STYLE),
            html.Span("gain"),
            dcc.Input(id=f"{ns}-ctl-film-gain", type="number", value=1.75,
                      min=0, max=5, step=0.05, style=_NUM_STYLE),
        ], style=_ROW_STYLE),
        html.Div([
            html.Span("DoRA", style={"minWidth": "60px", "fontWeight": "bold"}),
            dcc.Dropdown(id=f"{ns}-ctl-dora-dd", options=_DORA_OPTIONS,
                         value=dora_default, clearable=False,
                         style={"width": "160px", "display": "inline-block"}),
            html.Span("strength"),
            dcc.Input(id=f"{ns}-ctl-dora-strength", type="number", value=1.0,
                      min=0, max=10, step=0.01, style=_NUM_STYLE),
            html.Span("σ-interval"),
            dcc.Input(id=f"{ns}-ctl-dora-imin", type="number", value=0.0,
                      min=0, max=1, step=0.05, style=_NUM_STYLE),
            dcc.Input(id=f"{ns}-ctl-dora-imax", type="number", value=1.0,
                      min=0, max=1, step=0.05, style=_NUM_STYLE),
        ], style=_ROW_STYLE),
        html.Div([
            html.Span("LatCH adv", style={"minWidth": "60px",
                                          "fontWeight": "bold"}),
            html.Span("ρ", title=_RHO_HELP, style=_HELP_STYLE),
            dcc.Input(id=f"{ns}-ctl-latch-rho", type="number", value=None,
                      min=0, step=1, placeholder="auto",
                      style=_NUM_STYLE),
            html.Span("μ", title=_MU_HELP, style=_HELP_STYLE),
            dcc.Input(id=f"{ns}-ctl-latch-mu", type="number", value=None,
                      min=0, step=1, placeholder="auto",
                      style=_NUM_STYLE),
            html.Span("γ", title=_GAMMA_HELP, style=_HELP_STYLE),
            dcc.Input(id=f"{ns}-ctl-latch-gamma", type="number", value=0.3,
                      min=0, max=1, step=0.05,
                      style=_NUM_STYLE),
            html.Span("n_iter", title=_NITER_HELP, style=_HELP_STYLE),
            dcc.Input(id=f"{ns}-ctl-latch-niter", type="number", value=4,
                      min=1, max=16, step=1,
                      style=_NUM_STYLE),
            html.Span("(hover each label for how to pick a value)",
                      style={"fontSize": "11px", "color": "#888"}),
        ], style=_ROW_STYLE),
        # The pickers used to fill from Input(<id>, "id"), which Dash fires exactly
        # once per page load: with :8056 down at load time they returned [] and never
        # retried, so the panel stayed empty forever with no error shown. This poller
        # retries until a scan succeeds, then every callback below short-circuits on
        # its own State and the polling costs nothing.
        dcc.Interval(id=f"{ns}-ctl-info-poll", interval=5000, n_intervals=0,
                     max_intervals=-1),
    ])
    return html.Div(html.Details([
        html.Summary("Steering (LatCH / FiLM / DoRA)"),
        body,
    ], open=not collapsed))


def steering_states(ns: str) -> list[State]:
    """EXACTLY this order, length 35 (v2 contract, extended append-only per slot
    2026-07-12: +loss/+wsec per slot, +rho/mu/gamma/n_iter advanced row;
    steering_payload depends on it)."""
    states: list[State] = []
    for i in range(1, LATCH_SLOTS + 1):
        p = f"{ns}-ctl-latch{i}"
        for suffix in ("head", "kind", "value", "gain", "start", "end",
                       "loss", "wsec"):
            states.append(State(f"{p}-{suffix}", "value"))
    states += [State(f"{ns}-ctl-film-enable", "value"),
               State(f"{ns}-ctl-film-value", "value"),
               State(f"{ns}-ctl-film-gain", "value"),
               State(f"{ns}-ctl-dora-dd", "value"),
               State(f"{ns}-ctl-dora-strength", "value"),
               State(f"{ns}-ctl-dora-imin", "value"),
               State(f"{ns}-ctl-dora-imax", "value"),
               State(f"{ns}-ctl-latch-rho", "value"),
               State(f"{ns}-ctl-latch-mu", "value"),
               State(f"{ns}-ctl-latch-gamma", "value"),
               State(f"{ns}-ctl-latch-niter", "value")]
    return states


def steering_payload(vals: Sequence) -> dict:
    """Consume exactly 35 values in steering_states order -> A.4 blocks (v2)."""
    if len(vals) != 35:
        raise ValueError(f"steering_payload expects 35 values, got {len(vals)}")
    latch = []
    for i in range(LATCH_SLOTS):
        head, kind, value, gain, start, end, loss, wsec = vals[i * 8:(i + 1) * 8]
        if head in (None, "none", ""):
            continue
        slot = {
            "head": head,
            "kind": kind or "constant",
            "value": float(value) if value is not None else -30.0,
            "gain": float(gain) if gain is not None else 512.0,
            "start_pct": float(start) if start is not None else 0.0,
            "end_pct": float(end) if end is not None else 0.6,
        }
        if loss:
            slot["loss_type"] = loss
        if wsec is not None:
            slot["w_sec"] = float(wsec)
        latch.append(slot)
    film_enable, film_value, film_gain = vals[24:27]
    film = None
    if film_enable and "on" in film_enable:
        srv = render_client.info() or {}
        ckpt = (srv.get("film_default") or {}).get("ckpt", _FILM_CKPT_FALLBACK)
        film = {"ckpt": ckpt,
                "value": float(film_value) if film_value is not None else 4.0,
                "gain": float(film_gain) if film_gain is not None else 1.75}
    dora_dd, dora_strength, dora_imin, dora_imax = vals[27:31]
    dora = None
    if dora_dd == "none":
        # explicit "none" must reach the server by name: an absent dora block
        # means "use the op default" there (evr1x for /a2a_mix), which would
        # silently override the user's base-model choice. strength rides along
        # because a top-level ckpt_path folds into this same dict server-side
        # (resolve_dora_req) and honors dora.strength — same as
        # bracket._apply_weight; unused when no adapter loads
        dora = {"name": "none",
                "strength": (float(dora_strength)
                             if dora_strength is not None else 1.0)}
    elif dora_dd:
        # The picker now carries real checkpoint PATHS alongside the four legacy
        # registry names (hof/newstack/evr1x). _resolve_dora() on the server takes
        # either: a ckpt_path bypasses DORA_REGISTRY, a bare name must be in it.
        _is_path = str(dora_dd).startswith("/")
        dora = {"name": None if _is_path else dora_dd,
                "ckpt_path": dora_dd if _is_path else None,
                "strength": (float(dora_strength)
                             if dora_strength is not None else 1.0)}
    if dora is not None:
        # sigma-native interval gating (dit.py:466): e.g. (0.25, 1.0) disables
        # the adapter for the final low-noise detail steps
        dora["interval_min"] = float(dora_imin) if dora_imin is not None else 0.0
        dora["interval_max"] = float(dora_imax) if dora_imax is not None else 1.0
    out = {"latch": latch, "film": film, "dora": dora}
    # advanced LatCH hparams ride top-level (server resolve_latch reads them);
    # blank rho/mu = keep the auto slot-1-gain tie
    rho, mu, gamma, niter = vals[31:35]
    if rho is not None:
        out["rho"] = float(rho)
    if mu is not None:
        out["mu"] = float(mu)
    if gamma is not None:
        out["gamma"] = float(gamma)
    if niter is not None:
        out["n_iter"] = int(niter)
    return out


def _dora_rows(models: list, offline: set) -> list:
    """Adapter dicts from /models -> Dropdown options. Pure, so it is testable.

    Adapters whose root is an UNMOUNTED removable drive are kept in the list but
    marked and disabled: silently offering 692 adapters that cannot load (measured
    2026-08-26 with Mantu unplugged) is the same class of lie as the one-size-fits-all
    LatCH slider. Offline entries sort last so the usable ones stay at the top.
    """
    live, dead, seen = [], [], set()
    for m in models:
        path = m.get("path")
        if not path or path in seen:
            continue
        seen.add(path)
        label = m.get("label") or path.rsplit("/", 1)[-1]
        rank = m.get("rank")
        suffix = f" \u00b7 r{rank}" if rank else ""
        rid = m.get("root_id", "?")
        if rid in offline:
            dead.append({"label": f"{label}{suffix}  [{rid}] \u2014 drive offline",
                         "value": path, "disabled": True})
        else:
            live.append({"label": f"{label}{suffix}  [{rid}]", "value": path})
    return live + dead


def _root_availability() -> tuple:
    """(root_id, available) pairs, sorted. Cheap; used as a change signature so the
    picker refreshes when a drive is plugged in, without a page reload."""
    resp = render_client.roots()
    if not resp:
        return ()
    return tuple(sorted((r.get("id"), bool(r.get("available")))
                        for r in resp.get("roots", [])))


def _dora_options(offline: set | None = None) -> list:
    """Registry names first, then every loadable adapter the model database knows.

    Falls back to the four legacy registry names alone when the render server is
    unreachable, so the picker is never empty (it was hardcoded to exactly those
    four until 2026-08-24, which is why no trained adapter was selectable)."""
    opts = [{"label": n, "value": n} for n in _DORA_OPTIONS]
    resp = render_client.models(family="adapter", loadable=True)
    if not resp:
        return opts
    if offline is None:
        offline = {rid for rid, ok in _root_availability() if not ok}
    known = {o["value"] for o in opts}
    return opts + [r for r in _dora_rows(resp.get("models", []), offline)
                   if r["value"] not in known]


_ROOT_SIG: dict = {}          # ns -> last seen root-availability signature
_HEALTH_COLOR = {"ok": "#3a3", "undertrained": "#c60", "unstable": "#c00",
                 "unknown": "#888"}
_HELP_STYLE = {"cursor": "help", "borderBottom": "1px dotted #888"}

# These encode resolve_latch():514-517 in explorer_render_server.py -- the server
# NORMALISES: rho = mu = slot 1's gain, and every slot's weight becomes
# slot_gain / slot-1 gain. Do not paraphrase these into a raw-weight claim; a
# weight copied from another tool is a different scale and will not reproduce.
_RHO_HELP = ("\u03c1 -- guidance step size. LEAVE EMPTY. Empty means 'auto', and auto = slot 1's "
             "gain (512 for the medium heads, 2048 for chroma_other). The server sets "
             "rho = mu = slot-1 gain and then scales every other slot as "
             "slot_gain / slot-1 gain, so gains here are RELATIVE, not absolute -- a raw "
             "weight taken from another tool is a different scale and will not reproduce. "
             "Override only to decouple step size from the slot balance; start at the "
             "auto value and move by factors of 2.")
_MU_HELP = ("\u03bc -- proximal/consistency term, auto-tied to \u03c1 (both default to slot 1's "
            "gain). Raising \u03bc above \u03c1 pulls harder toward the current latent, i.e. "
            "guidance bites less; lowering it lets guidance dominate and is where artefacts "
            "start. Change it only after \u03c1 is settled, and change one at a time.")
_GAMMA_HELP = ("\u03b3 -- NOISE added to the latent before each head evaluation "
               "(z_in = z0 + \u03b3\u00b7\u03b5, latch_guided.py:179), default 0.3. It is the "
               "paper's 'guidance accuracy strength': HIGHER \u03b3 = deliberately LESS precise "
               "guidance = SAFER. Pons et al. (2603.04366) found raising \u03b3 is the way to keep "
               "audio quality when guidance follows the target too aggressively and drifts "
               "off-manifold -- an alternative to weakening \u03c1/\u03bc, which costs adherence. "
               "MEASURED CAVEAT (C, 2026-08-28): that is the paper's advice, but raising \u03b3 0.3 -> 0.6 "
               "on an OVERDRIVEN narrow-window render changed nothing (crest 2.21 -> 2.22, bass "
               "0.293 -> 0.292). \u03b3 blurs the head's INPUT, so it makes guidance less precise "
               "but does not shrink the STEP -- it is the wrong tool when the artefact comes from "
               "gain being too high for the window. Lower the gain for that. Reach for \u03b3 when "
               "guidance tracks the target too literally at a sane gain. Target being "
               "ignored -> lower it toward 0. \u03b3 = 0 also makes the render bit-reproducible; "
               "any \u03b3 > 0 makes head evaluation stochastic, so a fixed seed no longer gives "
               "an identical file.")
_NITER_HELP = ("n_iter -- guidance iterations per sampling step, default 4. NOT purely a cost "
               "knob: each iteration takes a FULL mu_t gradient step on the re-evaluated z0 "
               "(latch_guided.py:176-186), so raising it increases ADHERENCE as well as cost. "
               "2 for a quick look, 4 to work, 8 when a target is being ignored. Above 8 the "
               "returns are small and the artefacts are not. Note the paper's TFG also has an "
               "N_recur; we implement N_iter only.")

_WINDOW_HELP = ("start/end -- the FRACTION OF STEPS this slot is active for, but the delivered "
                "guidance is NOT proportional to the width. Per-step strength is scaled by "
                "signal fraction (1-sigma)/sum(1-sigma) (latch_guided.py:117,138), so rho/mu is a "
                "TOTAL budget spread across the schedule and weighted toward the CLEAN end. "
                "Measured on a real 24-step schedule: the first 20% of steps carry 0.2% of that "
                "budget, the first 60% carry 9.3%, and the last 20% carry ~62%. So the default "
                "0.0-0.6 window delivers under a tenth of the available guidance, and widening "
                "end 0.6 -> 0.8 roughly quadruples it. If a head seems dead, widen the window "
                "toward 1.0 BEFORE raising the gain.")


def _slot_view(h: dict) -> tuple:
    """A /info head dict -> the 17 Outputs of _autofill_defaults.

    Pure on purpose: Dash callbacks are not unit-testable in this repo, this is.
    """
    scalar = h.get("supports_scalar_target", True)
    loss_ok = h.get("supports_loss_select", True)
    lo, hi = h.get("slider_min"), h.get("slider_max")
    units = h.get("units") or ""
    step = (round(max((hi - lo) / 200.0, 1e-4), 6)
            if (lo is not None and hi is not None and hi > lo) else 0.1)
    kinds = h.get("supports_kinds") or _KIND_OPTIONS
    unit_sfx = (" " + units) if units else ""

    if scalar and h.get("std_mean") is not None:
        m = h["std_mean"]
        sg = h.get("std_std") or 0.0
        k = h.get("sigma_k", 2.0)
        meter = (f"dataset {m:.4g} \u00b1 {sg:.4g}{unit_sfx}  \u00b7  "
                 f"slider = mean \u00b1 {k:g}\u03c3  \u00b7  [{lo:.4g}, {hi:.4g}]")
        vtitle = (f"Target value in the head's own units{(' (' + units + ')') if units else ''}. "
                  f"Dataset mean {m:.4g}, \u03c3 {sg:.4g}. The server standardises this for you "
                  f"-- type RAW units. Values far outside \u00b12\u03c3 are out-of-distribution: "
                  f"the head will still push, but toward audio it never saw.")
    elif not scalar:
        meter = (f"{h.get('out_channels')}-channel head -- a single number cannot be a target "
                 f"for it. Use a measured curve (target_raw) instead.")
        vtitle = ("Disabled: this head predicts a vector per frame, so a scalar target means "
                  "'every channel equally', which is not musically meaningful.")
    else:
        meter = "no dataset statistics in this checkpoint -- slider is the generic fallback"
        vtitle = "This checkpoint records no std_mean/std_std; bounds are the generic default."

    if loss_ok:
        ltitle = (f"Loss shaping. This head was TRAINED with '{h.get('loss_type')}' -- leaving "
                  f"the box empty uses it. Overriding is legal and sometimes right "
                  f"(scalar_pooled is the 2026-07-10 buzz fix for constant targets on scalar "
                  f"heads), but any other choice is a departure from how the head was fit.")
    else:
        ltitle = (f"Disabled: '{h.get('loss_type')}' is this head's objective, not a preference. "
                  f"Swapping it on a {h.get('out_channels')}-channel head is a bug, not a knob.")

    hl = h.get("health", "unknown")
    badge = "" if hl == "ok" else f"\u26a0 {hl}"
    if h.get("readout_source") == "not-recorded":
        badge = (badge + "  readout: unrecorded").strip()
    hstyle = {"fontSize": "11px", "fontWeight": "bold",
              "color": _HEALTH_COLOR.get(hl, "#888"), "cursor": "help",
              "marginLeft": "6px"}
    htitle = h.get("health_reason") or ""
    if h.get("readout_source") == "not-recorded":
        htitle = (htitle + "  This head's target readout was not recorded at training time "
                  "(train_latch.py only started saving chroma_key/target_source on "
                  "2026-08-24), so which chroma it was fit against is unknown.").strip()

    return (h.get("default_gain"), h.get("target_kind_default", "constant"), kinds,
            not scalar,
            h.get("value_default"), lo, hi, step, not scalar, vtitle,
            h.get("loss_type") or "", not loss_ok, ltitle,
            meter, badge, hstyle, htitle)


def register(app, ns: str) -> None:
    """Head-options fill + per-head defaults autofill for one namespace."""

    # Every picker below fills off the shared poller rather than Input(<id>,"id"),
    # and short-circuits on its own current value, so it retries only while empty.
    @app.callback(Output(f"{ns}-ctl-dora-dd", "options"),
                  Input(f"{ns}-ctl-info-poll", "n_intervals"),
                  State(f"{ns}-ctl-dora-dd", "options"))
    def _fill_dora_options(_n, current):
        # Rebuild when empty, or when a removable drive came or went -- otherwise
        # short-circuit, so the steady-state cost of the poller is one cheap GET.
        sig = _root_availability()
        filled = bool(current) and len(current) > len(_DORA_OPTIONS)
        if filled and sig == _ROOT_SIG.get(ns):
            return no_update
        _ROOT_SIG[ns] = sig
        opts = _dora_options({rid for rid, ok in sig if not ok})
        return opts if len(opts) > len(_DORA_OPTIONS) or not current else no_update

    for i in range(1, LATCH_SLOTS + 1):
        head_id = f"{ns}-ctl-latch{i}-head"

        @app.callback(Output(head_id, "options"),
                      Output(head_id, "placeholder"),
                      Input(f"{ns}-ctl-info-poll", "n_intervals"),
                      State(head_id, "options"))
        def _fill_options(_n, current):
            if current:
                return no_update, no_update
            srv = render_client.info()
            if not srv:
                return [], "render server :8056 unreachable -- retrying every 5 s"
            return ([{"label": f"{h['name']} [{h.get('family', '?')}]",
                      "value": h["name"]}
                     for h in srv.get("latch_heads", [])], "head (off)")

        @app.callback(Output(f"{ns}-ctl-latch{i}-gain", "value"),
                      Output(f"{ns}-ctl-latch{i}-kind", "value"),
                      Output(f"{ns}-ctl-latch{i}-kind", "options"),
                      Output(f"{ns}-ctl-latch{i}-kind", "disabled"),
                      Output(f"{ns}-ctl-latch{i}-value", "value"),
                      Output(f"{ns}-ctl-latch{i}-value", "min"),
                      Output(f"{ns}-ctl-latch{i}-value", "max"),
                      Output(f"{ns}-ctl-latch{i}-value", "step"),
                      Output(f"{ns}-ctl-latch{i}-value", "disabled"),
                      Output(f"{ns}-ctl-latch{i}-vlabel", "title"),
                      Output(f"{ns}-ctl-latch{i}-loss", "value"),
                      Output(f"{ns}-ctl-latch{i}-loss", "disabled"),
                      Output(f"{ns}-ctl-latch{i}-llabel", "title"),
                      Output(f"{ns}-ctl-latch{i}-meter", "children"),
                      Output(f"{ns}-ctl-latch{i}-health", "children"),
                      Output(f"{ns}-ctl-latch{i}-health", "style"),
                      Output(f"{ns}-ctl-latch{i}-health", "title"),
                      Input(head_id, "value"))
        def _autofill_defaults(head):
            blank = (no_update,) * 17
            if head in (None, "none", ""):
                return blank
            srv = render_client.info()
            if not srv:
                return blank
            h = next((x for x in srv.get("latch_heads", []) if x["name"] == head), None)
            return _slot_view(h) if h is not None else blank
