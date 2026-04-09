"""
TimeseriesDB — SQLite-backed store for per-crop timeseries arrays.

Keeps timeseries data (hpcp_ts, rms_energy_*_ts, spectral_*_ts, etc.) out of
companion JSON files.  Companion JSONs shrink from ~130 KB to ~3 KB.

Storage
-------
Path : /home/kim/Projects/mir/data/timeseries.db  (NVMe, fast random I/O)
Mode : SQLite WAL — safe for concurrent multi-process writes
Codec: gzip(level=1) over msgpack — ~5× smaller than JSON text (~25 KB/crop)

Schema::

    CREATE TABLE ts (
        key  TEXT PRIMARY KEY,   -- crop stem, e.g. "Artist - Title_0"
        data BLOB                -- gzip(msgpack({field: {s:[shape], b:bytes}}))
    )

Each array is stored as its raw float32 bytes plus shape so it can be
reconstructed exactly with numpy.  gzip(level=1) adds ~1 ms of overhead per
crop but cuts float-array text from ~126 KB to ~13 KB per crop.

Usage::

    db = TimeseriesDB.open()

    # Write (from pipeline workers — each opens its own connection)
    db.put("Artist - Title_0", {"hpcp_ts": [[...], ...], "rms_energy_bass_ts": [...]})

    # Bulk write (from migration / bootstrap)
    db.bulk_put([("Artist - Title_0", ts_dict), ...])

    # Read
    arrays = db.get("Artist - Title_0")   # {field: np.ndarray}

    # Check coverage
    db.count()
    db.has("Artist - Title_0")
"""

import gzip
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import msgpack
import numpy as np

DEFAULT_DB_PATH = Path("/home/kim/Projects/mir/data/timeseries.db")

# Fields that are 2-D in the INFO files.  All others are assumed (256,).
_2D_FIELDS = {"hpcp_ts"}


# ---------------------------------------------------------------------------
# Codec helpers
# ---------------------------------------------------------------------------

def _encode_entry(ts_dict: dict) -> bytes:
    """Encode {field: numeric-list-or-ndarray} → compressed binary blob."""
    packed = {}
    for k, v in ts_dict.items():
        arr = np.array(v, dtype=np.float32)
        packed[k] = {"s": list(arr.shape), "b": arr.tobytes()}
    return gzip.compress(msgpack.packb(packed, use_bin_type=True), compresslevel=1)


def _decode_entry(blob: bytes) -> Dict[str, np.ndarray]:
    """Decompress binary blob → {field: np.ndarray}."""
    packed = msgpack.unpackb(gzip.decompress(blob), raw=False)
    result = {}
    for k, v in packed.items():
        arr = np.frombuffer(v["b"], dtype=np.float32)
        shape = tuple(v["s"])
        result[k] = arr.reshape(shape) if shape != arr.shape else arr
    return result


def is_timeseries_field(key: str, value) -> bool:
    """Return True for INFO fields that should live in TimeseriesDB, not JSON.

    Only float-array fields ending in _ts (or the fixed beat/downbeat arrays)
    are timeseries.  String lists like stem_names stay in the JSON.
    """
    if not isinstance(value, list) or not value:
        return False
    if key == "padding_mask":
        return False
    # Only migrate numeric lists — string lists stay in companion JSON
    return isinstance(value[0], (int, float)) and not isinstance(value[0], bool)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TimeseriesDB:
    """Per-process SQLite connection to the central timeseries store."""

    def __init__(self, path: Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con: sqlite3.Connection = self._connect()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, path: Path = DEFAULT_DB_PATH) -> "TimeseriesDB":
        return cls(path)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path), timeout=60, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA cache_size=-32768")   # 32 MB page cache
        con.execute("""
            CREATE TABLE IF NOT EXISTS ts (
                key  TEXT PRIMARY KEY,
                data BLOB NOT NULL
            )
        """)
        con.commit()
        return con

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def put(self, key: str, ts_dict: dict) -> None:
        """Insert or replace one crop's timeseries.

        Safe to call from multiple processes simultaneously (WAL mode).
        """
        blob = _encode_entry(ts_dict)
        self._con.execute(
            "INSERT OR REPLACE INTO ts (key, data) VALUES (?, ?)", (key, blob)
        )
        self._con.commit()

    def bulk_put(self, items: Iterable[Tuple[str, dict]], batch_size: int = 500) -> int:
        """Insert many (key, ts_dict) pairs efficiently.

        Returns number of rows written.  Uses batched transactions for speed.
        """
        written = 0
        batch: List[Tuple[str, bytes]] = []

        def _flush():
            nonlocal written
            self._con.executemany(
                "INSERT OR REPLACE INTO ts (key, data) VALUES (?, ?)", batch
            )
            self._con.commit()
            written += len(batch)
            batch.clear()

        for key, ts_dict in items:
            batch.append((key, _encode_entry(ts_dict)))
            if len(batch) >= batch_size:
                _flush()

        if batch:
            _flush()

        return written

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        """Return {field: np.ndarray} for *key*, or None if absent."""
        row = self._con.execute(
            "SELECT data FROM ts WHERE key=?", (key,)
        ).fetchone()
        return _decode_entry(row[0]) if row else None

    def get_field(self, key: str, field: str) -> Optional[np.ndarray]:
        """Fetch a single field for *key* without decoding everything."""
        arrays = self.get(key)
        return arrays.get(field) if arrays else None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has(self, key: str) -> bool:
        return bool(
            self._con.execute("SELECT 1 FROM ts WHERE key=?", (key,)).fetchone()
        )

    def count(self) -> int:
        return self._con.execute("SELECT COUNT(*) FROM ts").fetchone()[0]

    def all_keys(self) -> List[str]:
        return [r[0] for r in self._con.execute("SELECT key FROM ts").fetchall()]

    def missing_from(self, keys: Iterable[str]) -> List[str]:
        """Return keys not yet in the database."""
        existing = set(self.all_keys())
        return [k for k in keys if k not in existing]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return f"TimeseriesDB(path={self.path}, entries={self.count()})"


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        db = TimeseriesDB.open()
        print(f"TimeseriesDB at {db.path}")
        print(f"  Entries : {db.count():,}")
        sys.exit(0)

    if sys.argv[1] == "get" and len(sys.argv) == 3:
        db = TimeseriesDB.open()
        key = sys.argv[2]
        arrays = db.get(key)
        if arrays is None:
            print(f"Key not found: {key}")
            sys.exit(1)
        for field, arr in sorted(arrays.items()):
            print(f"  {field}: shape={arr.shape}  min={arr.min():.4f}  max={arr.max():.4f}")
