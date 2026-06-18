import json
from pathlib import Path
from plots.explorer_sa3.sidecar_index import scan_index, search, group_by_track


def _write_crop(d: Path, cid: str, **over):
    base = {
        "source_track": "AZukx - Earth Chakra",
        "track_metadata_artist": "AZukx", "track_metadata_title": "Earth Chakra",
        "prompt": "earth chakra, 1996, 123", "bpm_madmom": 123.0, "lufs": -9.5,
        "relative_position_start": 0.1, "relative_position_end": 0.2,
    }
    base.update(over)
    (d / f"{cid}.json").write_text(json.dumps(base))
    (d / f"{cid}.npy").write_bytes(b"")  # presence only


def test_scan_groups_and_searches(tmp_path):
    _write_crop(tmp_path, "000000")
    _write_crop(tmp_path, "000001", source_track="Other - Song",
                track_metadata_artist="Other", track_metadata_title="Song",
                prompt="dark forest")
    idx = scan_index(tmp_path)
    assert len(idx) == 2
    assert idx[0].id == "000000" and abs(idx[0].rel_pos - 0.1) < 1e-9
    assert idx[0].bpm == 123.0
    g = group_by_track(idx)
    assert set(g) == {"AZukx - Earth Chakra", "Other - Song"}
    hits = search(idx, "forest")
    assert len(hits) == 1 and hits[0].id == "000001"


def test_scan_skips_crop_with_unreadable_json(tmp_path):
    _write_crop(tmp_path, "000000")
    (tmp_path / "000001.json").write_text("{not valid")
    (tmp_path / "000001.npy").write_bytes(b"")
    idx = scan_index(tmp_path)
    assert [c.id for c in idx] == ["000000"]
