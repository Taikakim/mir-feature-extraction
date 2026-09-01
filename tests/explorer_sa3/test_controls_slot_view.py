"""_slot_view is the pure core of the LatCH panel: a /info head dict in, the 16
Dash Outputs out. Dash callbacks themselves are not testable in this repo; this is.
"""
from plots.explorer_sa3 import controls


def _h(**kw):
    base = {"name": "rms_energy_bass", "family": "medium", "out_channels": 1,
            "loss_type": "smooth_l1", "default_gain": 512.0,
            "target_kind_default": "constant", "slider_min": -50.99,
            "slider_max": 7.141, "value_default": -21.93, "std_mean": -21.93,
            "std_std": 14.53, "sigma_k": 2.0, "units": "dB", "health": "ok",
            "health_reason": "", "supports_scalar_target": True,
            "supports_loss_select": True, "readout_source": "n/a",
            "supports_kinds": ["constant", "ramp_up", "ramp_down", "beat_grid"]}
    base.update(kw)
    return base


# Output order, kept here so a reorder in controls.py breaks a test not a UI.
GAIN, KIND, KINDS, KDIS, VAL, VMIN, VMAX, STEP, VDIS, VTITLE, \
    LOSS, LDIS, LTITLE, METER, BADGE, HSTYLE, HTITLE = range(17)


def test_the_callback_contract_is_seventeen_outputs():
    assert len(controls._slot_view(_h())) == 17


def test_a_scalar_head_gets_its_own_bounds_and_a_meter_naming_the_dataset():
    v = controls._slot_view(_h())
    assert (v[VMIN], v[VMAX], v[VAL]) == (-50.99, 7.141, -21.93)
    assert v[VDIS] is False and v[LDIS] is False and v[KDIS] is False
    assert "-21.93" in v[METER] and "14.53" in v[METER] and "dB" in v[METER]


def test_the_step_is_derived_from_the_range_not_a_fixed_tenth():
    # 0.1 steps across hardness's 14-unit range is fine; across beat_activation's
    # 0.26 range it is three usable positions.
    v = controls._slot_view(_h(slider_min=-0.08735, slider_max=0.1751))
    assert 0 < v[STEP] < 0.01


def test_a_384_channel_cosine_head_disables_both_inputs_and_says_why():
    v = controls._slot_view(_h(name="same_chroma", out_channels=384,
                               loss_type="cosine", supports_scalar_target=False,
                               supports_loss_select=False, supports_kinds=[],
                               readout_source="overrides"))
    assert v[VDIS] is True and v[LDIS] is True
    # the target-KIND dropdown only shapes a scalar target, so it is gated too
    assert v[KDIS] is True
    assert "384-channel" in v[METER]
    assert "not a preference" in v[LTITLE]


def test_an_undertrained_head_shows_an_orange_badge_carrying_its_reason():
    v = controls._slot_view(_h(name="spectral_kurtosis", health="undertrained",
                               health_reason="trained only 3 epochs and sigma is 521.6"))
    assert v[BADGE].endswith("undertrained")
    assert v[HSTYLE]["color"] == controls._HEALTH_COLOR["undertrained"]
    assert "521.6" in v[HTITLE]


def test_a_healthy_head_shows_no_badge():
    assert controls._slot_view(_h())[BADGE] == ""


def test_an_unrecorded_readout_is_surfaced_as_its_own_badge_and_explained():
    v = controls._slot_view(_h(name="hpcp", out_channels=12,
                               supports_scalar_target=False, supports_kinds=[],
                               readout_source="not-recorded"))
    assert "readout: unrecorded" in v[BADGE]
    assert "2026-08-24" in v[HTITLE]


def test_a_checkpoint_without_statistics_says_the_bounds_are_a_fallback():
    v = controls._slot_view(_h(std_mean=None, std_std=None,
                               slider_min=-80.0, slider_max=20.0))
    assert "fallback" in v[METER]


def test_the_advanced_help_is_phrased_in_the_normalised_scale():
    # resolve_latch() sets rho = mu = slot-1 gain and scales the rest relatively.
    # If this help ever claims an absolute weight it is wrong, so pin the words.
    assert "RELATIVE, not absolute" in controls._RHO_HELP
    assert "slot_gain / slot-1 gain" in controls._RHO_HELP
    for txt in (controls._MU_HELP, controls._GAMMA_HELP, controls._NITER_HELP):
        assert len(txt) > 80


def test_the_gamma_help_states_the_MECHANISM_not_a_prescription(controls_mod=controls):
    """gamma is input-noise std (latch_guided.py:179): higher = less precise guidance.

    This help has now been wrong in BOTH directions. First it called gamma "inner-loop
    damping" and told users to LOWER it for unstable heads -- backwards. Then I
    corrected it to the paper's advice, "artefacts -> RAISE gamma (0.4-0.6)", which a
    measurement immediately contradicted: on an overdriven narrow-window render, gamma
    0.3 -> 0.6 changed nothing (crest 2.21 -> 2.22). gamma blurs the head's INPUT; it
    does not shrink the STEP, so it cannot fix an artefact caused by too much gain for
    the window. So the text must now state the mechanism and let the user reason,
    rather than prescribe a knob-turn for a failure it does not address.
    """
    g = controls_mod._GAMMA_HELP
    assert "damping" not in g.lower(), "gamma is noise augmentation, not damping"
    assert "less precise" in g, "must state the direction: higher = less precise"
    assert "does not shrink the STEP" in g, "must say WHY it cannot fix overdrive"
    assert "Lower the gain for that" in g, "must point at the remedy that does work"


# --- DoRA picker: adapters on an unmounted removable drive ---------------------
# (restored 2026-08-28 — an open-ended s[index:] replacement deleted these. Third
# time a slice has eaten trailing code in this session; use anchored replacements.)

_MODELS = [
    {"path": "/uuid/a.ckpt", "label": "arm_a", "rank": 128, "root_id": "lumi_uuid"},
    {"path": "/mantu/b.ckpt", "label": "arm_b", "rank": 16, "root_id": "lumi_mantu"},
    {"path": "/mantu/b.ckpt", "label": "dupe", "root_id": "lumi_mantu"},
    {"path": None, "label": "no path", "root_id": "lumi_uuid"},
]


def test_adapters_on_a_mounted_root_are_plain_selectable_options():
    rows = controls._dora_rows(_MODELS, offline=set())
    assert [r["value"] for r in rows] == ["/uuid/a.ckpt", "/mantu/b.ckpt"]
    assert all("disabled" not in r for r in rows)
    assert "r128" in rows[0]["label"]


def test_adapters_on_an_unmounted_root_are_disabled_labelled_and_sorted_last():
    rows = controls._dora_rows(_MODELS, offline={"lumi_mantu"})
    assert rows[-1]["value"] == "/mantu/b.ckpt"
    assert rows[-1]["disabled"] is True
    assert "drive offline" in rows[-1]["label"]
    assert rows[0]["value"] == "/uuid/a.ckpt" and "disabled" not in rows[0]


def test_a_path_seen_twice_is_offered_once():
    assert len(controls._dora_rows(_MODELS, offline=set())) == 2
