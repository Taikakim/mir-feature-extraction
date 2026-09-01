"""The preset Load is split in two callbacks on purpose; this pins the split.

Setting a LatCH slot's head fires controls._autofill_defaults, which writes that
slot's gain/kind/value from the head's checkpoint defaults. Anything the preset
wrote to those in the SAME response is overwritten. So the per-slot values must
land in a second callback that DEPENDS on the first (it is triggered by the slot
meter, an Output of _autofill_defaults). If someone later moves an id across the
partition, that silent clobber comes back — hence these tests.
"""
from plots.explorer_sa3 import controls, inference_tab as it


def test_the_two_halves_exactly_partition_the_form():
    early, late = it._early_ids(), it._late_ids()
    assert set(early) & set(late) == set()
    assert set(early) | set(late) == {st.component_id for st in it.form_states()}
    assert len(early) + len(late) == len(it.form_states())


def test_every_control_autofill_writes_is_in_the_LATE_half():
    # _autofill_defaults outputs gain/kind/value per slot -- those must be late.
    for i in range(1, controls.LATCH_SLOTS + 1):
        for sfx in ("gain", "kind", "value"):
            assert f"inf-ctl-latch{i}-{sfx}" in it._late_ids()


def test_the_head_dropdowns_are_in_the_EARLY_half():
    # The heads must be set first -- they are what triggers the autofill the late
    # half waits on. A head in the late half would deadlock the chain.
    for i in range(1, controls.LATCH_SLOTS + 1):
        assert f"inf-ctl-latch{i}-head" in it._early_ids()


def test_seed_and_batch_are_never_restored():
    assert it._VOLATILE_FORM_IDS == ("inf-seed", "inf-batch")
    # ...and they are still part of the form, so build_payload keeps working.
    ids = {st.component_id for st in it.form_states()}
    assert set(it._VOLATILE_FORM_IDS) <= ids
