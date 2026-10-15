"""Modular sub-presets for the Inference tab: latch, dora, film, sampler, prompt.

Owner: WINTERMUTE (created 2026-10-14)
Last edited: WINTERMUTE 2026-10-15 — LatCH presets carry guide_schedule / guide_anchor / guide_rescale

The whole-form preset (render server /presets) restores everything at once. These
restore ONE section and leave the rest of the form alone, so a LatCH stack can be
tried against any prompt, DoRA or sampler. A workspace preset holds no values of
its own: it names one sub-preset per section and loads those.

LatCH presets are stored in the render payload's own shape ({"latch": [slot, ...],
"rho"?, "mu"?, "gamma"?, "n_iter"?}), so a slot list copied out of a /generate
payload or an eval presets.json is already a valid preset. Loading one either
REPLACES every slot or STACKS onto the free slots after the ones in use.

Storage is client-side JSON, one file per preset:
    <root>/<kind>/<name>.json   root = $SA3_SUBPRESET_DIR or mir/data/explorer_presets
Pure functions only; the Dash wiring lives in inference_tab.py.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from .controls import LATCH_SLOTS

KINDS = ("latch", "dora", "film", "sampler", "prompt")
WORKSPACE = "workspace"
ALL_KINDS = KINDS + (WORKSPACE,)

# Form fields per section, WITHOUT the "<ns>-" prefix so a file is not tied to a tab.
FIELDS = {
    "prompt": ("prompt", "variation", "negprompt"),
    "sampler": ("duration", "steps", "cfg", "cfg-interval", "apg", "durpad",
                "dist-shift", "dist-mode"),
    "dora": ("ckpt-dd", "ckpt-path", "ctl-dora-dd", "ctl-dora-strength",
             "ctl-dora-imin", "ctl-dora-imax"),
    "film": ("ctl-film-enable", "ctl-film-value", "ctl-film-gain"),
}
# the LatCH advanced row: payload key -> form field
LATCH_ADV = {"rho": "ctl-latch-rho", "mu": "ctl-latch-mu",
             "gamma": "ctl-latch-gamma", "n_iter": "ctl-latch-niter",
             "guide_schedule": "ctl-latch-sched", "guide_anchor": "ctl-latch-anchor",
             "guide_rescale": "ctl-latch-rescale"}
# slot form suffix -> payload key
_SLOT_KEYS = {"head": "head", "kind": "kind", "value": "value", "gain": "gain",
              "start": "start_pct", "end": "end_pct", "loss": "loss_type",
              "wsec": "w_sec"}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ +-]{0,63}$")


def root_dir() -> Path:
    env = os.environ.get("SA3_SUBPRESET_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data" / "explorer_presets"


def _path(kind: str, name: str, root: Path | None) -> Path:
    if kind not in ALL_KINDS:
        raise ValueError(f"unknown preset kind {kind!r}")
    if not _NAME_RE.match(name or ""):
        raise ValueError(f"bad preset name {name!r}: letters, digits, space . _ + - (max 64)")
    return (root or root_dir()) / kind / f"{name}.json"


def names(kind: str, root: Path | None = None) -> list[str]:
    d = (root or root_dir()) / kind
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []


def save(kind: str, name: str, data: dict, root: Path | None = None) -> Path:
    p = _path(kind, name, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"schema": 1, "kind": kind, "name": name,
                               "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "data": data}, indent=1))
    tmp.replace(p)
    return p


def load(kind: str, name: str, root: Path | None = None) -> dict:
    p = _path(kind, name, root)
    if not p.is_file():
        raise ValueError(f"no {kind} preset named {name!r}")
    return json.loads(p.read_text())["data"]


# ------------------------------------------------------------------ LatCH slots
def slot_ids(ns: str, i: int) -> dict[str, str]:
    """1-based slot -> {suffix: dash id}."""
    return {s: f"{ns}-ctl-latch{i}-{s}" for s in _SLOT_KEYS}


def slots_from_form(form: dict, ns: str) -> list[dict | None]:
    """The LATCH_SLOTS slot rows as payload-shaped dicts, None where the head is off."""
    out: list[dict | None] = []
    for i in range(1, LATCH_SLOTS + 1):
        ids = slot_ids(ns, i)
        if form.get(ids["head"]) in (None, "", "none"):
            out.append(None)
            continue
        slot = {}
        for sfx, key in _SLOT_KEYS.items():
            v = form.get(ids[sfx])
            if v not in (None, "") or key in ("head", "kind", "value", "gain",
                                              "start_pct", "end_pct"):
                slot[key] = v
        out.append(slot)
    return out


def latch_from_form(form: dict, ns: str) -> dict:
    data = {"latch": [s for s in slots_from_form(form, ns) if s is not None]}
    for key, fid in LATCH_ADV.items():
        v = form.get(f"{ns}-{fid}")
        if v is not None:
            data[key] = v
    return data


def merge_slots(current: list[dict | None], preset: dict, stack: bool) -> list[dict | None]:
    """REPLACE: the preset's slots, rest empty. STACK: keep the slots in use, add the
    preset's after them. Raises ValueError when the result would not fit."""
    new = [s for s in (preset.get("latch") or []) if s and s.get("head")]
    kept = [s for s in current if s] if stack else []
    if len(kept) + len(new) > LATCH_SLOTS:
        raise ValueError(f"{len(kept)} slots in use + {len(new)} from the preset "
                         f"> {LATCH_SLOTS} slots")
    merged = kept + new
    return merged + [None] * (LATCH_SLOTS - len(merged))


def slot_form_values(slots: list[dict | None], ns: str) -> tuple[dict, dict]:
    """Slot dicts -> (early, late) {dash id: value}. Heads go early: setting a head
    fires controls._autofill_defaults, which overwrites gain/kind/value, so the rest
    must land after it (see inference_tab._preset_apply_slots)."""
    early, late = {}, {}
    for i, slot in enumerate(slots, start=1):
        ids = slot_ids(ns, i)
        if slot is None:
            early[ids["head"]] = None
            continue
        early[ids["head"]] = slot["head"]
        late[ids["kind"]] = slot.get("kind") or "constant"
        late[ids["value"]] = slot.get("value")
        late[ids["gain"]] = slot.get("gain")
        late[ids["start"]] = slot.get("start_pct", 0.0)
        late[ids["end"]] = slot.get("end_pct", 0.6)
        late[ids["loss"]] = slot.get("loss_type") or ""
        late[ids["wsec"]] = slot.get("w_sec")
    return early, late


# ------------------------------------------------------------------ sections
def section_from_form(kind: str, form: dict, ns: str) -> dict:
    if kind == "latch":
        return latch_from_form(form, ns)
    return {f: form.get(f"{ns}-{f}") for f in FIELDS[kind]}


def resolve(kind: str, name: str, form: dict, ns: str, stack: bool = False,
            root: Path | None = None) -> tuple[dict, dict, list[str]]:
    """Apply one preset (or a workspace) to the current form.

    Returns (early, late, loaded): {dash id: value} for the first and second half
    of the load, and the "kind:name" list actually loaded. Only ids of the loaded
    sections appear, so every other field is left as it is."""
    if kind == WORKSPACE:
        refs = load(WORKSPACE, name, root).get("refs") or {}
        early, late, loaded = {}, {}, []
        slots = slots_from_form(form, ns)
        for k in KINDS:
            ref = refs.get(k)
            if not ref:
                continue
            if k == "latch":
                # a workspace may stack several LatCH presets: first replaces, rest stack
                for j, n in enumerate([ref] if isinstance(ref, str) else ref):
                    data = load("latch", n, root)
                    slots = merge_slots(slots, data, stack=j > 0)
                    early.update(_adv(data, ns))
                    loaded.append(f"latch:{n}")
                e, late_ = slot_form_values(slots, ns)
                early.update(e)
                late.update(late_)
            else:
                early.update({f"{ns}-{f}": v for f, v in load(k, ref, root).items()
                              if f in FIELDS[k]})
                loaded.append(f"{k}:{ref}")
        return early, late, loaded
    data = load(kind, name, root)
    if kind == "latch":
        early, late = slot_form_values(merge_slots(slots_from_form(form, ns), data, stack), ns)
        early.update(_adv(data, ns))
        return early, late, [f"latch:{name}"]
    return ({f"{ns}-{f}": v for f, v in data.items() if f in FIELDS[kind]}, {},
            [f"{kind}:{name}"])


def _adv(data: dict, ns: str) -> dict:
    return {f"{ns}-{fid}": data[k] for k, fid in LATCH_ADV.items() if k in data}


def save_workspace(name: str, form: dict, ns: str, root: Path | None = None) -> dict:
    """Save every section as a sub-preset called `name`, and a workspace naming them."""
    refs = {}
    for k in KINDS:
        save(k, name, section_from_form(k, form, ns), root)
        refs[k] = name
    save(WORKSPACE, name, {"refs": refs}, root)
    return refs


# ------------------------------------------------------------------ import
def import_payload(name: str, payload: dict, root: Path | None = None) -> list[str]:
    """A /generate-style payload (e.g. one entry of an eval presets.json) -> latch,
    prompt, sampler and dora sub-presets plus a workspace, all called `name`.
    Only fields the payload actually carries are written."""
    refs = {}
    if payload.get("latch"):
        data = {"latch": payload["latch"]}
        data.update({k: payload[k] for k in LATCH_ADV if k in payload})
        save("latch", name, data, root)
        refs["latch"] = name
    if "prompt" in payload:
        save("prompt", name, {"prompt": payload["prompt"], "variation": "",
                              "negprompt": payload.get("negative_prompt", "")}, root)
        refs["prompt"] = name
    smap = {"duration": "duration", "steps": "steps", "cfg": "cfg_scale",
            "apg": "apg_scale", "dist-shift": "dist_shift"}
    sampler = {f: payload[k] for f, k in smap.items() if k in payload}
    if "cfg_interval" in payload:      # sigma-space; cfg_interval_progress is not
        sampler["cfg-interval"] = payload["cfg_interval"]
    if sampler:
        save("sampler", name, sampler, root)
        refs["sampler"] = name
    dora = payload.get("dora") or {}
    ckpt = dora.get("ckpt_path") or payload.get("ckpt_path") or dora.get("name")
    if ckpt:
        d = {"ckpt-dd": None, "ckpt-path": "", "ctl-dora-dd": ckpt,
             "ctl-dora-strength": dora.get("strength", 1.0)}
        for k, f in (("interval_min", "ctl-dora-imin"), ("interval_max", "ctl-dora-imax")):
            if k in dora:
                d[f] = dora[k]
        save("dora", name, d, root)
        refs["dora"] = name
    if refs:
        save(WORKSPACE, name, {"refs": refs}, root)
    return sorted(refs)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Import payload presets as sub-presets.")
    ap.add_argument("presets_json", help="{name: payload} file, e.g. an eval presets.json")
    ap.add_argument("names", nargs="*", help="only these entries (default: all)")
    ap.add_argument("--prefix", default="", help="prepended to every saved name")
    a = ap.parse_args(argv)
    src = json.loads(Path(a.presets_json).read_text())
    for n, payload in src.items():
        if a.names and n not in a.names:
            continue
        print(f"{a.prefix}{n}: {', '.join(import_payload(a.prefix + n, payload))}")
    print(f"-> {root_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
