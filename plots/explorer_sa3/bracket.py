"""Bracket panel — client-side param fan-out for the Inference tab (v1 of the
Gradio_Lab port, docs/bracket-port-design-2026-07-13.md §2).

Minimal axis set: steps / cfg / nl / weight (comma lists; empty = the form's
single value). The panel snapshots the WHOLE inference form through
`inference_tab.form_states()` + `build_payload()` (passed into `register` to
avoid a circular import), enumerates the Cartesian product, then issues
sequential POSTs — one render per `brk-tick` interval so the grid/status
update per item. The server serializes on GPU_LOCK and echoes the merged
request back as `meta.params_echo`, which is the recall record. The 35-state
steering contract is consumed as-is; every id here is `brk-*`.
"""
from __future__ import annotations

import itertools
import json
import random
import time

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from . import render_client

_ROW_STYLE = {"display": "flex", "gap": "6px", "alignItems": "center",
              "flexWrap": "wrap", "marginBottom": "4px"}
_AX_STYLE = {"width": "160px"}

_TIP_AXES = ("Comma lists; an empty axis falls back to the form's single "
             "value. nl (noise level) needs a non-empty init audio path and "
             "an empty noise ladder. weight sweeps the target picked in the "
             "dropdown (DoRA strength needs an adapter/ckpt selected; LatCH-1 "
             "gain needs a head in slot 1 — with blank ρ/μ the server ties "
             "ρ=μ to that gain, so the gain sweep sweeps them too).")
_TIP_SEED = ("A form seed of -1 is resolved ONCE per bracket run to a shared "
             "random seed so combos differ only along the axes.")

# axis order = product order (weight varies fastest); (field-id-suffix,
# payload key, cast). weight is special-cased through the target dropdown.
_AXES = (("steps", "steps", int),
         ("cfg", "cfg_scale", float),
         ("nl", "noise_level", float),
         ("weight", "_weight", float))


def layout() -> html.Div:
    return html.Div(html.Details([
        html.Summary("Bracket — param fan-out (comma lists; empty = form value)",
                     title=_TIP_AXES),
        html.Div([
            html.Span("steps"),
            dcc.Input(id="brk-ax-steps", type="text", value="",
                      placeholder="e.g. 16,24,48", style=_AX_STYLE),
            html.Span("cfg"),
            dcc.Input(id="brk-ax-cfg", type="text", value="",
                      placeholder="e.g. 4,6,8.5", style=_AX_STYLE),
            html.Span("nl (a2a only)"),
            dcc.Input(id="brk-ax-nl", type="text", value="",
                      placeholder="e.g. 0.35,0.5", style=_AX_STYLE),
            html.Span("weight"),
            dcc.Input(id="brk-ax-weight", type="text", value="",
                      placeholder="e.g. 0.5,1,1.5", style=_AX_STYLE),
            dcc.Dropdown(id="brk-ax-weight-target", clearable=False,
                         value="dora_strength",
                         options=[{"label": "DoRA strength",
                                   "value": "dora_strength"},
                                  {"label": "LatCH-1 gain",
                                   "value": "latch1_gain"}],
                         style={"width": "150px", "display": "inline-block"}),
        ], style=_ROW_STYLE),
        html.Div([
            html.Span(id="brk-count", title=_TIP_SEED,
                      style={"fontWeight": "bold"}),
            html.Button("Start bracket", id="brk-start"),
            html.Button("Stop", id="brk-stop"),
        ], style=_ROW_STYLE),
        html.Pre(id="brk-status", style={"whiteSpace": "pre-wrap",
                                         "fontSize": "0.8em"}),
        dcc.Store(id="brk-queue", data=None),
        dcc.Store(id="brk-results", data=[]),
        dcc.Interval(id="brk-tick", interval=400, disabled=True),
        html.Div(id="brk-grid"),
    ], open=False))


def _parse_axis(text: str, cast, name: str) -> list:
    """Comma-list -> typed values; [] = axis inactive. Raises ValueError with
    the axis + offending token (fork behavior, better message)."""
    vals = []
    for tok in (text or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(cast(tok))
        except ValueError:
            raise ValueError(f"axis {name}: bad token {tok!r}") from None
    return vals


def _label(combo: dict, seed) -> str:
    parts = []
    if "steps" in combo:
        parts.append(f"steps {combo['steps']}")
    if "cfg_scale" in combo:
        parts.append(f"cfg {combo['cfg_scale']:g}")
    if "noise_level" in combo:
        parts.append(f"nl {combo['noise_level']:g}")
    if "_weight" in combo:
        parts.append(f"w {combo['_weight']:g}")
    parts.append(f"seed {seed}")
    return " · ".join(parts)


def _apply_weight(payload: dict, target: str, w: float) -> None:
    """Weight axis reaches inside a nested block — copy, never mutate the
    shared base's inner dicts (design §2.5)."""
    if target == "latch1_gain":
        latch = [dict(s) for s in (payload.get("latch") or [])]
        latch[0]["gain"] = float(w)
        payload["latch"] = latch
    else:
        payload["dora"] = {**(payload.get("dora") or {}), "strength": float(w)}


def _merge(base: dict, combo: dict, weight_target: str) -> dict:
    payload = dict(base)
    for k, v in combo.items():
        if k == "_weight":
            _apply_weight(payload, weight_target, v)
        else:
            payload[k] = v
    return payload


def register(app, form_states, build_payload) -> None:
    """`form_states` / `build_payload` come from inference_tab (passed in, not
    imported — inference_tab imports this module for layout())."""

    # 1. live count — mirrors the fork's calculate_generation_count
    @app.callback(Output("brk-count", "children"),
                  Input("brk-ax-steps", "value"),
                  Input("brk-ax-cfg", "value"),
                  Input("brk-ax-nl", "value"),
                  Input("brk-ax-weight", "value"),
                  Input("inf-init-path", "value"),
                  Input("inf-batch", "value"))
    def _count(steps, cfg, nl, weight, init_path, batch):
        try:
            sizes = [len(_parse_axis(steps, int, "steps")) or 1,
                     len(_parse_axis(cfg, float, "cfg")) or 1,
                     len(_parse_axis(nl, float, "nl")) or 1,
                     len(_parse_axis(weight, float, "weight")) or 1]
        except ValueError as e:
            return f"⚠ {e}"
        n = 1
        for s in sizes:
            n *= s
        txt = f"{n} combos ({'×'.join(str(s) for s in sizes)})"
        if (init_path or "").strip():
            txt += f" · op a2a_track × batch {int(batch or 1)}"
        elif _parse_axis(nl, float, "nl"):
            txt += " — ⚠ nl axis needs an init audio path"
        if n > 32:
            txt += " — that's a long night"
        return txt

    # 2. start — snapshot form -> base payload + combo list, arm the ticker
    @app.callback(
        Output("brk-queue", "data"),
        Output("brk-results", "data"),
        Output("brk-tick", "disabled"),
        Output("brk-status", "children"),
        Input("brk-start", "n_clicks"),
        State("brk-ax-steps", "value"),
        State("brk-ax-cfg", "value"),
        State("brk-ax-nl", "value"),
        State("brk-ax-weight", "value"),
        State("brk-ax-weight-target", "value"),
        *form_states(),
        prevent_initial_call=True)
    def _start(_n, ax_steps, ax_cfg, ax_nl, ax_weight, weight_target, *vals):
        try:
            op, base = build_payload(list(vals))
            axes = [(key, _parse_axis(txt, cast, name))
                    for (name, key, cast), txt in
                    zip(_AXES, (ax_steps, ax_cfg, ax_nl, ax_weight))]
        except ValueError as e:
            return no_update, no_update, no_update, str(e)
        active = [(k, v) for k, v in axes if v]
        if not active:
            return no_update, no_update, no_update, \
                "no axis has values — nothing to bracket"
        ax = dict(active)
        if "noise_level" in ax:
            if op != "a2a_track":
                return no_update, no_update, no_update, \
                    "nl axis needs an init audio path (op a2a_track)"
            if base.get("noise_levels"):
                return no_update, no_update, no_update, \
                    "clear the noise ladder when using the nl axis " \
                    "(two ladder mechanisms would multiply)"
        if "_weight" in ax:
            if weight_target == "latch1_gain" and not base.get("latch"):
                return no_update, no_update, no_update, \
                    "weight target LatCH-1 gain: slot 1 has no head"
            if weight_target == "dora_strength" and not (
                    base.get("ckpt_path")
                    or (base.get("dora") or {}).get("name", "none") != "none"):
                return no_update, no_update, no_update, \
                    "weight target DoRA strength: pick an adapter or ckpt first"
        # -1 seed resolved ONCE per run so all combos share it (fork behavior)
        if int(base.get("seed", -1)) == -1:
            base = {**base, "seed": random.randint(0, 2**31 - 1)}
        keys = [k for k, _ in active]
        combos = [dict(zip(keys, prod))
                  for prod in itertools.product(*(v for _, v in active))]
        queue = {"op": op, "base": base, "combos": combos,
                 "total": len(combos), "weight_target": weight_target,
                 "t0": time.time()}
        return queue, [], False, \
            f"queued {len(combos)} combos · op {op} · seed {base['seed']}"

    # 3. ticker — ONE render per tick = sequential fan-out with per-item updates
    @app.callback(
        Output("brk-queue", "data", allow_duplicate=True),
        Output("brk-results", "data", allow_duplicate=True),
        Output("brk-tick", "disabled", allow_duplicate=True),
        Output("brk-status", "children", allow_duplicate=True),
        Input("brk-tick", "n_intervals"),
        State("brk-queue", "data"),
        State("brk-results", "data"),
        prevent_initial_call=True)
    def _tick(_n, queue, results):
        if not queue or not queue.get("combos"):
            return None, no_update, True, no_update
        combo = queue["combos"][0]
        payload = _merge(queue["base"], combo, queue["weight_target"])
        seed = payload.get("seed")
        entry = {"index": len(results or []),
                 "label": _label(combo, seed), "combo": combo,
                 "op": queue["op"], "seed": seed, "error": None,
                 "urls": [], "files": [], "total_sec": None,
                 "params_echo": None}
        t0 = time.time()
        try:
            resp = render_client.render(queue["op"], payload)
            entry.update(urls=resp.get("urls", []),
                         files=resp.get("files", []),
                         seed=resp.get("seed", seed),
                         total_sec=resp.get("timings", {}).get("total_sec"),
                         params_echo=resp.get("meta", {}).get("params_echo"))
        except render_client.RenderError as e:
            entry["error"] = str(e)          # failed combo stays in the grid
        results = (results or []) + [entry]
        queue = {**queue, "combos": queue["combos"][1:]}
        done, total = len(results), queue["total"]
        times = [r["total_sec"] for r in results if r.get("total_sec")]
        eta = ""
        if times and queue["combos"]:
            eta = f" · ETA {sum(times) / len(times) * len(queue['combos']):.0f}s"
        status = (f"[{done}/{total}] last {time.time() - t0:.0f}s{eta}"
                  if queue["combos"] else
                  f"done — {total} combos in {time.time() - queue['t0']:.0f}s")
        return (queue if queue["combos"] else None), results, \
            not queue["combos"], status

    # 4. stop — drain the queue (in-flight POST finishes as the last entry)
    @app.callback(
        Output("brk-queue", "data", allow_duplicate=True),
        Output("brk-tick", "disabled", allow_duplicate=True),
        Output("brk-status", "children", allow_duplicate=True),
        Input("brk-stop", "n_clicks"),
        prevent_initial_call=True)
    def _stop(_n):
        return None, True, "stopped — finished renders kept"

    # 5. grid — pure render of the results store
    @app.callback(Output("brk-grid", "children"),
                  Input("brk-results", "data"))
    def _grid(results):
        cards = []
        for e in (results or []):
            style = {"display": "inline-block", "verticalAlign": "top",
                     "border": "1px solid #ccc", "borderRadius": "4px",
                     "padding": "6px", "margin": "4px", "maxWidth": "340px"}
            body = [html.B(e.get("label", ""),
                           style={"display": "block", "fontSize": "0.85em"})]
            if e.get("error"):
                style["border"] = "1px solid #a00"
                body.append(html.Pre(e["error"][:500],
                                     style={"color": "#a00",
                                            "whiteSpace": "pre-wrap",
                                            "fontSize": "0.7em"}))
            for u, f in zip(e.get("urls", []),
                            e.get("files") or [None] * len(e.get("urls", []))):
                body.append(html.Div([
                    html.Audio(src=render_client.audio_url(u), controls=True,
                               style={"width": "100%"}),
                    html.Div(f or "", style={"fontSize": "0.7em",
                                             "color": "#666"}),
                ]))
            body.append(html.Button(
                "Recall → form",
                id={"type": "brk-recall", "index": e.get("index", 0)}))
            if e.get("params_echo"):
                body.append(html.Details([
                    html.Summary("params"),
                    html.Pre(json.dumps(e["params_echo"], indent=2,
                                        default=str),
                             style={"fontSize": "0.65em", "maxHeight": "240px",
                                    "overflow": "auto"}),
                ]))
            cards.append(html.Div(body, style=style))
        return cards

    # 6. per-clip recall — combo values back INTO the form (index-based, never
    # string matching). latch1-gain is already an Output of the controls
    # autofill, hence allow_duplicate throughout.
    @app.callback(
        Output("inf-steps", "value", allow_duplicate=True),
        Output("inf-cfg", "value", allow_duplicate=True),
        Output("inf-init-noise", "value", allow_duplicate=True),
        Output("inf-ctl-dora-strength", "value", allow_duplicate=True),
        Output("inf-ctl-latch1-gain", "value", allow_duplicate=True),
        Input({"type": "brk-recall", "index": ALL}, "n_clicks"),
        State("brk-results", "data"),
        State("brk-ax-weight-target", "value"),
        prevent_initial_call=True)
    def _recall(clicks, results, weight_target):
        if not any(c for c in (clicks or []) if c):
            return (no_update,) * 5
        idx = ctx.triggered_id["index"]
        entry = next((r for r in (results or []) if r.get("index") == idx),
                     None)
        if entry is None:
            return (no_update,) * 5
        combo = entry.get("combo") or {}
        w = combo.get("_weight")
        return (combo.get("steps", no_update),
                combo.get("cfg_scale", no_update),
                combo.get("noise_level", no_update),
                w if (w is not None and weight_target == "dora_strength")
                else no_update,
                w if (w is not None and weight_target == "latch1_gain")
                else no_update)
