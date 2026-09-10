"""Lazy per-crop scalar cache: mean of every 1-D TIMESERIES field.

Computed once in a background thread on first Dataset-tab visit (numpy only,
mir venv), saved as a single .npz beside the index
(``latent_dir/_ts_scalar_cache.npz``) and reloaded on later runs while its
crop-id list still matches the scanned index. 2-D fields (hpcp_ts) are skipped.
"""
from __future__ import annotations
import threading
from pathlib import Path
import numpy as np

CACHE_NAME = "_ts_scalar_cache.npz"


class ScalarCache:
    """field -> per-crop mean of the crop's 1-D TIMESERIES arrays."""

    def __init__(self, latent_dir: Path, ids: list[str]):
        self.latent_dir = Path(latent_dir)
        self.ids = list(ids)
        self._idx = {cid: i for i, cid in enumerate(self.ids)}
        self._cols: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._state = "idle"          # idle | computing | ready
        self._done = 0

    @property
    def path(self) -> Path:
        return self.latent_dir / CACHE_NAME

    def status(self) -> dict:
        with self._lock:
            return {"state": self._state, "done": self._done,
                    "total": len(self.ids)}

    def fields(self) -> list[str]:
        with self._lock:
            return sorted(self._cols) if self._state == "ready" else []

    def value(self, crop_id: str, field: str) -> float | None:
        with self._lock:
            col = self._cols.get(field)
        i = self._idx.get(crop_id)
        if col is None or i is None:
            return None
        v = float(col[i])
        return None if np.isnan(v) else v

    def ensure_started(self) -> None:
        """Load the on-disk cache if it matches the index, else spawn ONE
        background compute thread. Idempotent; safe to call every poll."""
        with self._lock:
            if self._state in ("computing", "ready"):
                return
            if self._load_locked():
                self._state = "ready"
                self._done = len(self.ids)
                return
            self._state = "computing"
            self._done = 0
            threading.Thread(target=self._compute, daemon=True).start()

    def _load_locked(self) -> bool:
        if not self.path.exists():
            return False
        try:
            with np.load(self.path) as z:
                if [str(s) for s in z["_ids"]] != self.ids:
                    return False
                self._cols = {k: z[k] for k in z.files if k != "_ids"}
            return True
        except Exception:
            return False

    def _compute(self) -> None:
        n = len(self.ids)
        cols: dict[str, np.ndarray] = {}
        for i, cid in enumerate(self.ids):
            p = self.latent_dir / f"{cid}.TIMESERIES.npz"
            try:
                with np.load(p) as z:
                    for k in z.files:
                        a = z[k]
                        if a.ndim != 1 or a.size == 0:
                            continue
                        if k not in cols:
                            cols[k] = np.full(n, np.nan, dtype=np.float64)
                        cols[k][i] = float(np.nanmean(a))
            except Exception:
                pass  # missing/corrupt sidecar -> NaN row, crop just drops out
            if (i + 1) % 25 == 0 or i + 1 == n:
                with self._lock:
                    self._done = i + 1
        out = {k: v.astype(np.float32) for k, v in cols.items()}
        try:
            np.savez(self.path, _ids=np.array(self.ids), **out)
        except Exception:
            pass  # unwritable dir -> keep in-memory only
        with self._lock:
            self._cols = out
            self._done = n
            self._state = "ready"
