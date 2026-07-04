#!/usr/bin/env python3
"""Per-track feature table builder for the LoRA/DoRA conditioning corpus.

Produces one row per track — the single source of truth that feeds CONTINUITY's
tag-vocabulary curation, clustering, and Flamingo-budget stratification (see
SAO/docs/prompting-conditioning-plan.md).

Why this exists: the raw features are scattered across two incompatible schemas
and stored as ragged top-k probability dicts, so clustering cannot consume them
directly. This tool joins per-TRACK (crop schemes differ, so a per-crop join is
wrong), aligns genre/mood probs to fixed-dim vectors over canonical vocabularies
(zero-filled), and aggregates crop-level scalars.

Sources per track:
  - genre / bpm / onset_density / release_year  <-  <latents-dir>/*.json
    (grouped by ``source_track``; genre = mean of the ``style_genre`` dict over
    that track's crops; falls back to per-crop ``essentia_genre`` if a latents
    dir has no ``style_genre``, so new-corpus emitters work too)
  - mood / rms_energy_{bass,body,mid,air}        <-  <crop-info-root>/<track>/*.INFO
    (the fields the latents companions lack)

Per Kim's data-layout directive (2026-07-04): sources are NEVER pooled — each
corpus is built with its own ``--source`` label and (for new corpora) its own
latents/crop dirs; the original Goa ``latents_sa3`` stays pristine. The table is
one row per track carrying an explicit ``source`` column so it stays queryable
per-source while remaining a single per-track source of truth.

Outputs (into --out-dir):
  feature_table[.parquet|.csv]  one row/track (both join keys: source_track + latent indices)
  vocab_map.json                ordered genre + mood vocabularies (vector column semantics)

Run with the mir venv:  /home/kim/Projects/mir/mir/bin/python
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("build_feature_table")

# Sentinel for an unknown release year. The scalar year FiLM lane drops the
# condition (CFG dropout) when year_known is False, so this value never reaches
# training — it is only a table placeholder.
YEAR_UNKNOWN = -1

RMS_BANDS = ("bass", "body", "mid", "air")


# ─────────────────────────── pure helpers (unit-tested) ───────────────────────

def parse_year(raw: Any) -> Tuple[int, bool]:
    """Return (year, year_known). Unparseable / missing -> (YEAR_UNKNOWN, False)."""
    if raw is None:
        return YEAR_UNKNOWN, False
    try:
        # tolerate "1996", "1996-05-01", 1996, 1996.0
        s = str(raw).strip()
        if not s:
            return YEAR_UNKNOWN, False
        year = int(s[:4])
    except (ValueError, TypeError):
        return YEAR_UNKNOWN, False
    # guard against garbage years
    if 1900 <= year <= 2100:
        return year, True
    return YEAR_UNKNOWN, False


def union_vocab(dicts: Iterable[Dict[str, float]]) -> List[str]:
    """Sorted union of keys across a stream of prob dicts (the canonical vocab)."""
    keys: set = set()
    for d in dicts:
        if d:
            keys.update(d.keys())
    return sorted(keys)


def mean_probs(dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean of per-crop prob dicts over a track. A key absent from a crop counts
    as 0 for that crop (top-k truncation => absent means ~0), so the mean is over
    the full crop count, not just the crops where the key appeared."""
    if not dicts:
        return {}
    acc: Dict[str, float] = defaultdict(float)
    for d in dicts:
        for k, v in (d or {}).items():
            acc[k] += float(v)
    n = len(dicts)
    return {k: v / n for k, v in acc.items()}


def align_vector(prob_dict: Dict[str, float], vocab: Sequence[str]) -> List[float]:
    """Map a (ragged) prob dict onto a fixed-dim vector over ``vocab``, zero-filled."""
    pd = prob_dict or {}
    return [float(pd.get(label, 0.0)) for label in vocab]


def agg_scalars(values: Sequence[Optional[float]]) -> Dict[str, float]:
    """mean/std over non-None numeric values (population std). Empty -> NaNs."""
    xs = [float(v) for v in values if v is not None and not _isnan(v)]
    if not xs:
        return {"mean": math.nan, "std": math.nan, "n": 0}
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return {"mean": mean, "std": math.sqrt(var), "n": len(xs)}


def _isnan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (ValueError, TypeError):
        return False


# ─────────────────────────── collection (integration) ─────────────────────────

def _load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001 — a bad sidecar shouldn't abort the run
        logger.warning("skip unreadable %s: %s", p, exc)
        return None


def collect_latents(latents_dir: Path) -> Dict[str, dict]:
    """Group latents *.json by source_track -> aggregated per-track record.

    Returns {source_track: {genre: dict, bpm: [..], onset: [..], year_raw,
    indices: [..], relpaths: [..]}}.
    """
    per_track: Dict[str, dict] = {}
    files = sorted(latents_dir.glob("*.json"))
    logger.info("latents: %d crop jsons in %s", len(files), latents_dir)
    for jp in files:
        d = _load_json(jp)
        if not d:
            continue
        track = d.get("source_track")
        if not track:
            logger.warning("no source_track in %s — skipped", jp.name)
            continue
        rec = per_track.setdefault(track, {
            "genre_dicts": [], "bpm": [], "onset": [],
            "year_raw": None, "indices": [], "relpaths": [],
        })
        # genre: prefer style_genre (training-native), else per-crop essentia_genre
        g = d.get("style_genre") or d.get("essentia_genre")
        if isinstance(g, dict):
            rec["genre_dicts"].append(g)
        bpm = d.get("bpm_essentia") or d.get("bpm_madmom") or d.get("bpm")
        if bpm is not None:
            rec["bpm"].append(bpm)
        if d.get("onset_density") is not None:
            rec["onset"].append(d["onset_density"])
        if rec["year_raw"] is None:
            rec["year_raw"] = d.get("track_metadata_year") or d.get("release_year")
        rec["indices"].append(jp.stem)  # e.g. "000123"
        if d.get("relpath"):
            rec["relpaths"].append(d["relpath"])
    logger.info("latents: %d unique tracks", len(per_track))
    return per_track


def collect_crop_info(crop_info_root: Path, tracks: Iterable[str]) -> Dict[str, dict]:
    """For each track name, aggregate mood + rms from <root>/<track>/*.INFO.

    Missing folders are simply absent from the result (caller zero-fills).
    """
    out: Dict[str, dict] = {}
    tracks = list(tracks)
    for i, track in enumerate(tracks, 1):
        folder = crop_info_root / track
        if not folder.is_dir():
            continue
        mood_dicts: List[dict] = []
        rms: Dict[str, list] = {b: [] for b in RMS_BANDS}
        for ip in folder.glob("*.INFO"):
            d = _load_json(ip)
            if not d:
                continue
            if isinstance(d.get("essentia_mood"), dict):
                mood_dicts.append(d["essentia_mood"])
            for b in RMS_BANDS:
                v = d.get(f"rms_energy_{b}")
                if v is not None:
                    rms[b].append(v)
        if mood_dicts or any(rms.values()):
            out[track] = {"mood_dicts": mood_dicts, "rms": rms}
        if i % 500 == 0:
            logger.info("crop-info join: %d/%d tracks scanned", i, len(tracks))
    logger.info("crop-info join: matched %d/%d tracks", len(out), len(tracks))
    return out


def build_rows(latents: Dict[str, dict], crop: Dict[str, dict],
               source: str) -> Tuple[List[dict], List[str], List[str]]:
    """Assemble per-track records into rows + return (rows, genre_vocab, mood_vocab)."""
    genre_vocab = union_vocab(
        mean_probs(rec["genre_dicts"]) for rec in latents.values())
    mood_vocab = union_vocab(
        mean_probs(c["mood_dicts"]) for c in crop.values())

    rows: List[dict] = []
    for track, rec in sorted(latents.items()):
        genre_mean = mean_probs(rec["genre_dicts"])
        c = crop.get(track, {})
        mood_mean = mean_probs(c.get("mood_dicts", []))
        year, year_known = parse_year(rec["year_raw"])
        onset = agg_scalars(rec["onset"])
        bpm = agg_scalars(rec["bpm"])
        row = {
            "source": source,
            "source_track": track,
            "latent_indices": rec["indices"],
            "n_crops": len(rec["indices"]),
            "genre_vec": align_vector(genre_mean, genre_vocab),
            "mood_vec": align_vector(mood_mean, mood_vocab),
            "mood_present": bool(c.get("mood_dicts")),
            "bpm_mean": bpm["mean"],
            "onset_density_mean": onset["mean"],
            "onset_density_std": onset["std"],
            "release_year": year,
            "year_known": year_known,
        }
        for b in RMS_BANDS:
            s = agg_scalars(c.get("rms", {}).get(b, []))
            row[f"rms_energy_{b}_mean"] = s["mean"]
            row[f"rms_energy_{b}_std"] = s["std"]
        rows.append(row)
    return rows, genre_vocab, mood_vocab


def write_outputs(rows: List[dict], genre_vocab: List[str], mood_vocab: List[str],
                  out_dir: Path, source: str) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    pq = out_dir / "feature_table.parquet"
    df.to_parquet(pq, index=False)
    # CSV: list columns are JSON-encoded so the file stays a flat, greppable table
    csv_df = df.copy()
    for col in ("latent_indices", "genre_vec", "mood_vec"):
        csv_df[col] = csv_df[col].map(json.dumps)
    csv = out_dir / "feature_table.csv"
    csv_df.to_csv(csv, index=False)
    vocab = out_dir / "vocab_map.json"
    vocab.write_text(json.dumps({
        "source": source,
        "genre_vocab": genre_vocab,     # index i <-> genre_vec[i]
        "mood_vocab": mood_vocab,       # index i <-> mood_vec[i]
        "n_tracks": len(rows),
        "note": "genre_vec/mood_vec are zero-filled prob vectors aligned to these "
                "vocabularies (union of observed top-k labels; tail probs ~0).",
    }, indent=2))
    logger.info("wrote %s (%d rows), %s, %s", pq.name, len(rows), csv.name, vocab.name)
    logger.info("genre vocab: %d labels | mood vocab: %d labels",
                len(genre_vocab), len(mood_vocab))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latents-dir", required=True, type=Path,
                    help="dir of per-crop *.json companions (grouped by source_track)")
    ap.add_argument("--crop-info-root", type=Path, default=None,
                    help="root of <track>/*.INFO crops for mood + rms (optional)")
    ap.add_argument("--source", required=True,
                    help="corpus label for the 'source' column (e.g. goa, prog_psytechno)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")
    logger.setLevel(logging.INFO)

    latents = collect_latents(args.latents_dir)
    if not latents:
        logger.error("no tracks found in %s", args.latents_dir)
        return 1
    crop = collect_crop_info(args.crop_info_root, latents.keys()) \
        if args.crop_info_root else {}
    rows, genre_vocab, mood_vocab = build_rows(latents, crop, args.source)
    write_outputs(rows, genre_vocab, mood_vocab, args.out_dir, args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
