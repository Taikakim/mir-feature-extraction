"""Unit tests for the per-track feature-table builder's pure logic.

Covers the two things that make the raw data clustering-hostile: ragged top-k
prob dicts (must align to a fixed-dim, zero-filled vector over a canonical vocab)
and messy release-year strings (must degrade to a year_known=False sentinel).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tools.build_feature_table import (
    YEAR_UNKNOWN,
    agg_scalars,
    align_vector,
    build_rows,
    mean_probs,
    parse_year,
    union_vocab,
)


def test_parse_year_variants():
    assert parse_year("1996") == (1996, True)
    assert parse_year(1996) == (1996, True)
    assert parse_year("1994-05-01") == (1994, True)
    assert parse_year(None) == (YEAR_UNKNOWN, False)
    assert parse_year("") == (YEAR_UNKNOWN, False)
    assert parse_year("n/a") == (YEAR_UNKNOWN, False)
    assert parse_year("1850") == (YEAR_UNKNOWN, False)   # out of guard range


def test_union_vocab_sorted_and_deduped():
    dicts = [{"b": 0.5, "a": 0.2}, {"a": 0.1, "c": 0.9}, {}]
    assert union_vocab(dicts) == ["a", "b", "c"]


def test_mean_probs_treats_absent_as_zero():
    # 'a' present in 1 of 2 crops at 0.4 -> mean 0.2 (absent counts as 0)
    out = mean_probs([{"a": 0.4, "b": 1.0}, {"b": 0.0}])
    assert out["a"] == 0.2
    assert out["b"] == 0.5


def test_mean_probs_empty():
    assert mean_probs([]) == {}


def test_align_vector_fixed_dim_zero_fill():
    vocab = ["a", "b", "c"]
    assert align_vector({"a": 0.7, "c": 0.3}, vocab) == [0.7, 0.0, 0.3]
    assert align_vector({}, vocab) == [0.0, 0.0, 0.0]
    # a label outside the vocab is ignored (vocab is authoritative on dimension)
    assert align_vector({"a": 0.7, "z": 9.9}, vocab) == [0.7, 0.0, 0.0]


def test_agg_scalars_mean_std_and_nan_handling():
    s = agg_scalars([2.0, 4.0, None])
    assert s["mean"] == 3.0
    assert s["n"] == 2
    assert math.isclose(s["std"], 1.0)
    empty = agg_scalars([None])
    assert math.isnan(empty["mean"]) and empty["n"] == 0


def test_build_rows_end_to_end_shape_and_join():
    latents = {
        "Artist - A": {
            "genre_dicts": [{"Goa": 0.8}, {"Goa": 0.6, "Psy": 0.2}],
            "bpm": [145.0, 145.0], "onset": [3.0, 5.0],
            "year_raw": "1996", "indices": ["000000", "000001"], "relpaths": [],
        },
        "Artist - B": {  # no crop-info join partner -> mood zero-filled
            "genre_dicts": [{"Techno": 1.0}],
            "bpm": [130.0], "onset": [8.0],
            "year_raw": None, "indices": ["000002"], "relpaths": [],
        },
    }
    crop = {
        "Artist - A": {
            "mood_dicts": [{"melodic": 0.4}, {"energetic": 0.6}],
            "genre_dicts": [{"Electronic---Techno": 0.5}, {"Electronic---Bleep": 0.3}],
            "rms": {"bass": [0.1, 0.3], "body": [], "mid": [], "air": []},
        },
    }
    rows, vocabs = build_rows(latents, crop, source="goa")

    assert vocabs["genre_vocab"] == ["Goa", "Psy", "Techno"]
    assert vocabs["mood_vocab"] == ["energetic", "melodic"]
    # second lane: discogs taxonomy from the crop info, kept separate
    assert vocabs["genre_discogs_vocab"] == ["Electronic---Bleep", "Electronic---Techno"]

    a = next(r for r in rows if r["source_track"] == "Artist - A")
    b = next(r for r in rows if r["source_track"] == "Artist - B")

    assert a["source"] == "goa" and a["n_crops"] == 2
    assert a["latent_indices"] == ["000000", "000001"]
    # genre_vec aligned to ["Goa","Psy","Techno"]: mean Goa=(0.8+0.6)/2=0.7, Psy=0.1
    assert a["genre_vec"] == [0.7, 0.1, 0.0]
    # genre_discogs_vec aligned to ["Bleep","Techno"]: Bleep=(0+0.3)/2, Techno=(0.5+0)/2
    assert a["genre_discogs_vec"] == [0.15, 0.25]
    assert a["mood_vec"] == [0.3, 0.2]      # energetic (0+0.6)/2, melodic (0.4+0)/2
    assert a["mood_present"] is True
    assert a["release_year"] == 1996 and a["year_known"] is True
    assert math.isclose(a["rms_energy_bass_mean"], 0.2)
    assert a["onset_density_mean"] == 4.0

    # B: unjoined -> mood + discogs-genre vectors zero-filled, year unknown, rms NaN
    assert b["mood_vec"] == [0.0, 0.0]
    assert b["genre_discogs_vec"] == [0.0, 0.0]
    assert b["mood_present"] is False
    assert b["year_known"] is False and b["release_year"] == YEAR_UNKNOWN
    assert math.isnan(b["rms_energy_bass_mean"])
    assert b["genre_vec"] == [0.0, 0.0, 1.0]   # Techno (coarse lane, from latents)
