"""Scan latents_sa3 .json sidecars into an in-memory, searchable index."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CropMeta:
    id: str
    source_track: str
    artist: str
    title: str
    prompt: str
    bpm: float | None
    lufs: float | None
    rel_pos: float


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def scan_index(latent_dir: Path) -> list[CropMeta]:
    """Scan every NNNNNN.json beside an NNNNNN.npy. Skip unreadable sidecars."""
    latent_dir = Path(latent_dir)
    out: list[CropMeta] = []
    for jp in sorted(latent_dir.glob("*.json")):
        if jp.name.endswith(".TIMESERIES.json"):
            continue
        if not jp.with_suffix(".npy").exists():
            continue
        try:
            m = json.loads(jp.read_text())
        except Exception:
            continue
        out.append(CropMeta(
            id=jp.stem,
            source_track=str(m.get("source_track", "")),
            artist=str(m.get("track_metadata_artist", "")),
            title=str(m.get("track_metadata_title", "")),
            prompt=str(m.get("prompt", "")),
            bpm=_to_float(m.get("bpm_madmom") or m.get("bpm_essentia")),
            lufs=_to_float(m.get("lufs")),
            rel_pos=_to_float(m.get("relative_position_start")) or 0.0,
        ))
    return out


def search(index: list[CropMeta], query: str) -> list[CropMeta]:
    if not query:
        return list(index)
    q = query.lower()
    return [c for c in index
            if q in c.artist.lower() or q in c.title.lower()
            or q in c.prompt.lower() or q in c.source_track.lower()]


def group_by_track(index: list[CropMeta]) -> dict[str, list[CropMeta]]:
    g: dict[str, list[CropMeta]] = {}
    for c in index:
        g.setdefault(c.source_track, []).append(c)
    return g
