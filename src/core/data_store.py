"""
DataStore: Consolidated dataset cache for HDD-optimised pipeline.

Maintains a ``dataset.jsonl`` file that aggregates all ``.INFO`` file contents
into a single sequential read, eliminating thousands of random seeks at
analysis time.

``dataset.jsonl`` is a *derived artefact* — always rebuilt from the primary
``.INFO`` files.  Pipeline workers keep writing ``.INFO`` for crash safety.
At the end of each phase the main process consolidates everything here.

One JSON object per line, each carrying its own key — this keeps the file
line-greppable / jq-streamable (`grep '"_key": "TrackName_0"' dataset.jsonl`),
unlike the old single-line ``dataset.json`` dict::

    {"_key": "TrackName", "bpm": 128.5, "lufs": -14.2, ...}
    {"_key": "TrackName_0", "bpm": 128.5, "lufs": -14.2, ...}
    ...

Generation metadata lives in a small sidecar next to the jsonl,
``dataset.meta.json``::

    {"generated_at": "ISO-8601", "root": "/abs/path", "count": 5000}

Usage::

    # Build / rebuild from .INFO files (always authoritative)
    store = DataStore.bootstrap(Path("/output/crops"))

    # Load existing cache (fast: single sequential read)
    store = DataStore.load(Path("/output/crops/dataset.jsonl"))

    # Queries
    missing_keys = store.missing("flamingo_brief")
    cov = store.coverage()

    # Export for feature explorer
    store.to_csv(Path("tracks.csv"))

    # Standalone rebuild
    python -c "from core.data_store import DataStore; DataStore.bootstrap('/path/to/output')"
"""

import csv
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _compact_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Replace list values with their length before storing in dataset.jsonl.

    Timeseries arrays (e.g. rms_energy_bass_ts: [256 floats]) and beat arrays
    (downbeats, beat_activations_ts, …) can each be hundreds of floats.  Storing
    full arrays in a consolidated JSON file causes it to balloon to tens of GB.

    The length value is sufficient for coverage tracking (key present = feature
    was extracted; length = expected resolution).  Raw arrays are always available
    in the per-crop .INFO files when needed.
    """
    return {k: len(v) if isinstance(v, list) else v for k, v in entry.items()}


def _meta_path(dataset_path: Path) -> Path:
    """Sidecar path for a dataset.jsonl's generation metadata, e.g.
    ``dataset.jsonl`` -> ``dataset.meta.json``."""
    return dataset_path.with_name(dataset_path.stem + ".meta.json")


def _read_meta(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"DataStore: could not read {meta_path}: {exc}")
        return {}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class DataStore:
    """Consolidated in-memory view of all .INFO files under a directory root."""

    def __init__(self, path: Path, data: Dict[str, Dict], root: Optional[Path] = None):
        self.path = Path(path)
        self.data: Dict[str, Dict] = data
        self._root: Path = root if root is not None else self.path.parent

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def bootstrap(cls, root: Path) -> "DataStore":
        """Scan all *.INFO files under *root*, build the data dict, write dataset.jsonl.

        Always rebuilds from source-of-truth ``.INFO`` files.  Call this at the
        end of each pipeline phase from the main process (not from workers).

        Args:
            root: Directory to scan recursively for ``*.INFO`` files.

        Returns:
            DataStore loaded with all discovered entries.
        """
        root = Path(root)
        dataset_path = root / "dataset.jsonl"
        data: Dict[str, Dict] = {}

        info_files = sorted(root.rglob("*.INFO"))
        logger.info(
            f"DataStore.bootstrap: scanning {len(info_files)} .INFO files under {root}"
        )

        for info_file in info_files:
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                key = info_file.stem
                data[key] = _compact_entry(entry)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"DataStore: skipping {info_file}: {exc}")

        store = cls(path=dataset_path, data=data, root=root)
        store.flush()
        logger.info(
            f"DataStore.bootstrap: wrote {len(data)} entries to {dataset_path}"
        )
        return store

    @classmethod
    def load(cls, path: Path) -> "DataStore":
        """Load an existing dataset.jsonl.  Fast: single sequential read.

        Args:
            path: Path to ``dataset.jsonl``.

        Returns:
            DataStore with the cached entries.

        Raises:
            FileNotFoundError: if *path* does not exist.
            json.JSONDecodeError: if a line is corrupt.
        """
        path = Path(path)
        data: Dict[str, Dict] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec: Dict[str, Any] = json.loads(line)
                key = rec.pop("_key")
                data[key] = rec

        meta = _read_meta(_meta_path(path))
        root_str = meta.get("root")
        root = Path(root_str) if root_str else path.parent

        store = cls(path=path, data=data, root=root)
        logger.debug(f"DataStore.load: loaded {len(data)} entries from {path}")
        return store

    # ------------------------------------------------------------------
    # Queries  (all O(n) dict scan, no file I/O)
    # ------------------------------------------------------------------

    def missing(self, feature: str) -> List[str]:
        """Return the list of entry keys that lack *feature*.

        O(n) dict scan — no file I/O.
        """
        return [key for key, entry in self.data.items() if feature not in entry]

    def coverage(self) -> Dict[str, Dict]:
        """Per-feature summary: count, pct, min, max across all entries.

        Returns:
            ``{feature: {"count": int, "pct": float, "min": ..., "max": ...}}``
        """
        total = len(self.data)
        if total == 0:
            return {}

        feature_values: Dict[str, List] = {}
        for entry in self.data.values():
            for key, val in entry.items():
                feature_values.setdefault(key, []).append(val)

        result: Dict[str, Dict] = {}
        for feat, vals in feature_values.items():
            count = len(vals)
            numeric = [
                v
                for v in vals
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            info: Dict[str, Any] = {
                "count": count,
                "pct": round(count / total * 100, 1),
            }
            if numeric:
                info["min"] = min(numeric)
                info["max"] = max(numeric)
            result[feat] = info

        return result

    def get(self, key: str) -> Dict:
        """Return the feature dict for *key*, or empty dict if absent."""
        return self.data.get(key, {})

    def update(self, key: str, features: Dict) -> None:
        """Merge *features* into the in-memory entry for *key*.  No file I/O."""
        if key in self.data:
            self.data[key].update(features)
        else:
            self.data[key] = dict(features)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Atomically write the current data dict to ``dataset.jsonl`` + its
        ``dataset.meta.json`` sidecar."""
        lines = []
        for key, entry in self.data.items():
            rec = {"_key": key}
            rec.update(entry)
            lines.append(json.dumps(rec, ensure_ascii=False))
        _atomic_write_text(self.path, "\n".join(lines) + ("\n" if lines else ""))

        meta = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "root": str(self._root.resolve()),
            "count": len(self.data),
        }
        _atomic_write_text(_meta_path(self.path), json.dumps(meta, ensure_ascii=False))

    def to_csv(self, out: Path) -> None:
        """Export the dataset to a CSV file for use with the feature explorer.

        Columns: ``stem``, then all feature keys found across any entry (sorted
        alphabetically).  Entries that lack a column get an empty cell.
        """
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)

        all_keys: set = set()
        for entry in self.data.values():
            all_keys.update(entry.keys())
        cols = sorted(all_keys)

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["stem"] + cols)
            for stem, entry in sorted(self.data.items()):
                row = [stem] + [entry.get(col, "") for col in cols]
                writer.writerow(row)

        logger.info(
            f"DataStore.to_csv: wrote {len(self.data)} rows × {len(cols)} cols to {out}"
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:  # pragma: no cover
        return f"DataStore(path={self.path}, entries={len(self.data)})"


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_store.py /path/to/output")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    store = DataStore.bootstrap(Path(sys.argv[1]))
    print(f"Built DataStore with {len(store)} entries -> {store.path}")
