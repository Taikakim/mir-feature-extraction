# Owner: WINTERMUTE (created 2026-10-14) -- section presets: scope, stacking, workspace, import
import pytest

from plots.explorer_sa3 import controls, inference_tab as it, subpresets as sp

NS = "inf"


def _form(slots=None):
    """A full form dict with every slot empty except the given {i: (head, gain)}."""
    form = {st.component_id: None for st in it.form_states()}
    form.update({"inf-prompt": "goa", "inf-steps": 30, "inf-seed": 764})
    for i, (head, gain) in (slots or {}).items():
        p = f"inf-ctl-latch{i}"
        form.update({f"{p}-head": head, f"{p}-kind": "constant", f"{p}-value": -10.0,
                     f"{p}-gain": gain, f"{p}-start": 0.0, f"{p}-end": 1.0,
                     f"{p}-loss": "", f"{p}-wsec": None})
    return form


def _slot(head, gain=512.0):
    return {"head": head, "kind": "constant", "value": -5.0, "gain": gain,
            "start_pct": 0.1, "end_pct": 0.9}


def test_slot_cap_is_six_and_the_contract_follows_it():
    assert controls.LATCH_SLOTS == 6
    assert len(controls.steering_states("inf")) == controls.N_STEER == 6 * 8 + 14
    vals = [None] * controls.N_STEER
    vals[5 * 8:5 * 8 + 4] = ["rms_energy_air", "constant", -3.0, 900.0]   # slot 6
    o = 6 * 8
    vals[o + 3] = "none"                                                  # dora
    vals[o + 9] = 0.25                                                    # gamma (rho, mu, gamma, n_iter)
    out = controls.steering_payload(vals)
    assert [s["head"] for s in out["latch"]] == ["rms_energy_air"]
    assert out["dora"]["name"] == "none" and out["gamma"] == 0.25
    # defaults (alpha, anchor/rescale empty) add no keys: the request stays as before
    assert not {"guide_schedule", "guide_anchor", "guide_rescale"} & set(out)


def test_guidance_options_reach_the_payload_and_round_trip_a_latch_preset(tmp_path):
    vals = [None] * controls.N_STEER
    vals[0:4] = ["onset_per_beat_e04", "constant", 2.0, 512.0]
    o = 6 * 8
    vals[o + 11:o + 14] = ["flat", 1.2, 0.0]
    out = controls.steering_payload(vals)
    assert out["guide_schedule"] == "flat" and out["guide_anchor"] == 1.2 and "guide_rescale" not in out
    form = {"inf-ctl-latch1-head": "onset_per_beat_e04", "inf-ctl-latch-sched": "flat",
            "inf-ctl-latch-anchor": 1.2}
    data = sp.latch_from_form(form, "inf")
    assert data["guide_schedule"] == "flat" and data["guide_anchor"] == 1.2
    sp.save("latch", "flat_anchor", data, tmp_path)
    early, _late, _ = sp.resolve("latch", "flat_anchor", {}, "inf", root=tmp_path)
    assert early["inf-ctl-latch-sched"] == "flat" and early["inf-ctl-latch-anchor"] == 1.2


def test_a_latch_load_writes_only_the_latch_section(tmp_path):
    sp.save("latch", "punch", {"latch": [_slot("rms_energy_bass"), _slot("rms_energy_body")],
                               "gamma": 0.25}, tmp_path)
    early, late, loaded = sp.resolve("latch", "punch", _form(), NS, root=tmp_path)
    assert loaded == ["latch:punch"]
    touched = set(early) | set(late)
    assert all(i.startswith("inf-ctl-latch") for i in touched)
    assert early["inf-ctl-latch1-head"] == "rms_energy_bass"
    assert early["inf-ctl-latch3-head"] is None
    assert late["inf-ctl-latch2-start"] == 0.1 and early["inf-ctl-latch-gamma"] == 0.25
    # heads early, per-slot values late -- the autofill race split
    assert set(early) <= set(it._early_ids()) and set(late) <= set(it._late_ids())


def test_stack_appends_after_the_slots_in_use_and_refuses_overflow(tmp_path):
    sp.save("latch", "air", {"latch": [_slot("rms_energy_air", 700.0)]}, tmp_path)
    form = _form({1: ("rms_energy_bass", 8448.0), 2: ("rms_energy_body", 11417.6)})
    early, late, _ = sp.resolve("latch", "air", form, NS, stack=True, root=tmp_path)
    assert [early[f"inf-ctl-latch{i}-head"] for i in (1, 2, 3)] == \
        ["rms_energy_bass", "rms_energy_body", "rms_energy_air"]
    assert late["inf-ctl-latch1-gain"] == 8448.0 and late["inf-ctl-latch3-gain"] == 700.0
    sp.save("latch", "six", {"latch": [_slot(f"h{i}") for i in range(5)]}, tmp_path)
    with pytest.raises(ValueError, match="> 6 slots"):
        sp.resolve("latch", "six", form, NS, stack=True, root=tmp_path)


def test_a_non_latch_section_never_touches_slots_or_seed(tmp_path):
    form = _form({1: ("rms_energy_bass", 512.0)})
    sp.save("sampler", "s30", sp.section_from_form("sampler", form, NS), tmp_path)
    early, late, _ = sp.resolve("sampler", "s30", _form(), NS, root=tmp_path)
    assert late == {} and early["inf-steps"] == 30
    assert set(early) == {f"inf-{f}" for f in sp.FIELDS["sampler"]}
    assert "inf-seed" not in early


def test_workspace_is_thin_and_stacks_several_latch_refs(tmp_path):
    sp.save("latch", "bass", {"latch": [_slot("rms_energy_bass")]}, tmp_path)
    sp.save("latch", "air", {"latch": [_slot("rms_energy_air")]}, tmp_path)
    sp.save("prompt", "goa", {"prompt": "goa", "variation": "", "negprompt": ""}, tmp_path)
    sp.save("workspace", "w", {"refs": {"latch": ["bass", "air"], "prompt": "goa"}}, tmp_path)
    form = _form({1: ("old", 1.0)})
    early, _, loaded = sp.resolve("workspace", "w", form, NS, root=tmp_path)
    assert loaded == ["latch:bass", "latch:air", "prompt:goa"]
    assert [early[f"inf-ctl-latch{i}-head"] for i in (1, 2, 3)] == \
        ["rms_energy_bass", "rms_energy_air", None]       # first ref REPLACES
    assert early["inf-prompt"] == "goa" and "inf-steps" not in early


def test_save_workspace_writes_one_preset_per_section(tmp_path):
    refs = sp.save_workspace("all", _form({1: ("rms_energy_bass", 512.0)}), NS, tmp_path)
    assert refs == {k: "all" for k in sp.KINDS}
    assert sp.load("latch", "all", tmp_path)["latch"][0]["head"] == "rms_energy_bass"
    assert sp.names("workspace", tmp_path) == ["all"]


def test_import_an_eval_payload(tmp_path):
    payload = {"prompt": "goa", "negative_prompt": "", "steps": 30, "cfg_scale": 9,
               "apg_scale": 1, "duration": 48, "seed": 764,
               "latch": [_slot("rms_energy_bass", 8448.0)], "rho": 5913.6, "gamma": 0.25,
               "dora": {"ckpt_path": "/x/epoch=19.ckpt", "strength": 0.51}}
    assert sp.import_payload("cleanest", payload, tmp_path) == \
        ["dora", "latch", "prompt", "sampler"]
    assert sp.load("latch", "cleanest", tmp_path)["rho"] == 5913.6
    assert "seed" not in str(sp.load("sampler", "cleanest", tmp_path))
    assert sp.load("dora", "cleanest", tmp_path)["ctl-dora-dd"] == "/x/epoch=19.ckpt"


def test_names_are_validated(tmp_path):
    with pytest.raises(ValueError):
        sp.save("latch", "../escape", {"latch": []}, tmp_path)
    with pytest.raises(ValueError):
        sp.save("nope", "x", {}, tmp_path)


def test_sub_output_ids_are_all_early():
    assert set(it._sub_output_ids()) <= set(it._early_ids())
