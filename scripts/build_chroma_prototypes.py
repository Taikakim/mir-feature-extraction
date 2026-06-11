"""
Dataset-derived chroma prototype palette for SAME/SA3 steering targets.

Computes time-averaged SAME-format chroma profiles over a crop dataset,
root-normalizes them using per-crop tonic (so clusters capture MODE/VOICING
flavor, not key), and clusters bass (band 0) and melody (band 1, mid)
SEPARATELY with spherical k-means -> a selectable palette of "bass flavors" x
"melody colors" for the steering UI. At selection time, prototypes are
re-rotated to the user's chosen root (fractional roll — a semitone is
128/12 ~= 10.667 bins, NOT an integer).

numpy/scipy only — runs anywhere, including the dev laptop (--selftest).

Usage:
    python build_chroma_prototypes.py --audio_dir ./crops --out prototypes.npz
    python build_chroma_prototypes.py --selftest
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "harmonic"))
import same_chroma as sc  # noqa: E402

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


# ── Root normalization (fractional circular shift) ───────────────────────────

def fractional_roll(profile: np.ndarray, shift_bins: float) -> np.ndarray:
    """Circularly shift a (..., n_bins) profile by a FRACTIONAL number of bins
    (linear interpolation, periodic)."""
    n = profile.shape[-1]
    idx = (np.arange(n) - shift_bins) % n
    lo = np.floor(idx).astype(int) % n
    hi = (lo + 1) % n
    w = (idx - np.floor(idx)).astype(profile.dtype)
    return profile[..., lo] * (1 - w) + profile[..., hi] * w


def root_normalize(profile: np.ndarray, tonic: int, n_chroma: int = 128) -> np.ndarray:
    """Shift a (3, 128) profile so its tonic pitch class lands where C sits.
    tonic: pitch class 0-11 (C=0)."""
    return fractional_roll(profile, -float(tonic) * n_chroma / 12.0)


def rotate_to_root(prototype: np.ndarray, root: int, n_chroma: int = 128) -> np.ndarray:
    """Inverse of root_normalize: place a C-normalized prototype at `root`."""
    return fractional_roll(prototype, float(root) * n_chroma / 12.0)


# ── Spherical k-means (cosine), numpy ─────────────────────────────────────────

def spherical_kmeans(X: np.ndarray, k: int, iters: int = 100, n_init: int = 8,
                     seed: int = 0):
    """
    Cluster unit-normalized rows of X (N, D) by cosine similarity.
    Returns (centroids (k, D) unit rows, labels (N,), mean_sim).
    """
    rng = np.random.default_rng(seed)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    best = None
    for trial in range(n_init):
        # k-means++ style init on cosine distance
        centroids = np.empty((k, X.shape[1]))
        centroids[0] = Xn[rng.integers(len(Xn))]
        for j in range(1, k):
            sim = Xn @ centroids[:j].T            # (N, j)
            d = 1.0 - sim.max(axis=1)
            d = np.maximum(d, 0) ** 2
            p = d / max(d.sum(), 1e-12)
            centroids[j] = Xn[rng.choice(len(Xn), p=p)]
        labels = np.zeros(len(Xn), dtype=int)
        for _ in range(iters):
            sim = Xn @ centroids.T
            new_labels = sim.argmax(axis=1)
            if (new_labels == labels).all() and _ > 0:
                break
            labels = new_labels
            for j in range(k):
                members = Xn[labels == j]
                if len(members) == 0:             # reseed empty cluster
                    centroids[j] = Xn[rng.integers(len(Xn))]
                else:
                    c = members.mean(axis=0)
                    centroids[j] = c / (np.linalg.norm(c) + 1e-12)
        mean_sim = float((Xn * centroids[labels]).sum(axis=1).mean())
        if best is None or mean_sim > best[2]:
            best = (centroids.copy(), labels.copy(), mean_sim)
    return best


def farthest_point_medoids(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Pick k maximally-spread unit rows (cosine farthest-point sampling)."""
    rng = np.random.default_rng(seed)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    chosen = [int(rng.integers(len(Xn)))]
    for _ in range(k - 1):
        sim = Xn @ Xn[chosen].T                   # (N, |chosen|)
        chosen.append(int((1.0 - sim.max(axis=1)).argmax()))
    return Xn[chosen]


# ── Profile extraction ─────────────────────────────────────────────────────────

def read_tonic(audio_path: Path, tonic_key: str) -> int | None:
    """Look for a sidecar JSON (.INFO or .json, same stem or in folder) with a
    tonic field (pitch class 0-11)."""
    candidates = [
        audio_path.with_suffix(".INFO"),
        audio_path.with_suffix(".json"),
        audio_path.parent / ".INFO",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            try:
                data = json.loads(c.read_text())
                if tonic_key in data:
                    return int(round(float(data[tonic_key]))) % 12
            except Exception as e:
                logger.debug(f"sidecar parse failed {c}: {e}")
    return None


def extract_profiles(audio_dir: Path, tonic_key: str, max_files: int | None):
    """Returns (profiles (N, 3, 128), tonics (N,), paths)."""
    import soundfile as sf
    files = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if max_files:
        files = files[:max_files]
    profiles, tonics, kept = [], [], []
    n_argmax = 0
    for p in files:
        try:
            audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
        except Exception as e:
            logger.warning(f"skip {p.name}: {e}")
            continue
        chroma = sc.compute_same_chroma(audio, sr, align_to_latent=False)
        if chroma.shape[-1] == 0:
            continue
        prof = chroma.mean(axis=-1)               # (3, 128) time-average
        tonic = read_tonic(p, tonic_key)
        if tonic is None:
            # crude fallback: folded mid-band argmax as root proxy
            tonic = int(np.argmax(sc.fold_to_12(prof[1])))
            n_argmax += 1
        profiles.append(prof)
        tonics.append(tonic)
        kept.append(str(p))
    if n_argmax:
        logger.warning(f"{n_argmax}/{len(kept)} files had no sidecar tonic - "
                       f"used folded-argmax proxy (less reliable)")
    return np.array(profiles, dtype=np.float64), np.array(tonics), kept


def build_palette(profiles: np.ndarray, tonics: np.ndarray, k: int,
                  medoids: bool, seed: int):
    """Root-normalize, then cluster bass and melody bands separately."""
    normed = np.stack([root_normalize(p, t) for p, t in zip(profiles, tonics)])
    out = {}
    for name, band in (("bass", 0), ("melody", 1)):
        X = normed[:, band, :]                    # (N, 128)
        if medoids:
            protos = farthest_point_medoids(X, k, seed)
            labels, sim = None, None
        else:
            protos, labels, sim = spherical_kmeans(X, k, seed=seed)
            logger.info(f"{name}: k={k} mean within-cluster cosine {sim:.3f}, "
                        f"sizes {np.bincount(labels, minlength=k).tolist()}")
        out[f"{name}_prototypes"] = protos.astype(np.float32)
        if labels is not None:
            out[f"{name}_labels"] = labels
    # keep per-melody-cluster air-band means for completeness
    if "melody_labels" in out:
        k_m = out["melody_prototypes"].shape[0]
        air = np.stack([
            normed[out["melody_labels"] == j, 2, :].mean(axis=0)
            if (out["melody_labels"] == j).any() else np.zeros(128)
            for j in range(k_m)
        ])
        air /= (np.linalg.norm(air, axis=1, keepdims=True) + 1e-12)
        out["air_means"] = air.astype(np.float32)
    return out


# ── Self-test (synthetic, no audio needed) ────────────────────────────────────

def selftest() -> bool:
    rng = np.random.default_rng(1)
    modes = {                                     # pitch-class sets
        "major":    [0, 2, 4, 5, 7, 9, 11],
        "minor":    [0, 2, 3, 5, 7, 8, 10],
        "phrygian": [0, 1, 3, 5, 7, 8, 10],
    }
    X, labels_true, tonics = [], [], []
    for mi, (name, pcs) in enumerate(modes.items()):
        for _ in range(60):
            w = np.zeros(12)
            w[pcs] = 0.5 + rng.random(len(pcs))   # varied voicing weights
            prof = sc.expand_semitone_weights(w).astype(np.float64)
            root = int(rng.integers(12))
            prof = fractional_roll(prof, root * 128 / 12.0)   # transpose
            prof *= 0.5 + 2.0 * rng.random()                  # loudness
            prof += 0.02 * rng.random(128)                    # noise floor
            X.append(prof)
            labels_true.append(mi)
            tonics.append(root)
    X, labels_true, tonics = np.array(X), np.array(labels_true), np.array(tonics)

    normed = np.stack([root_normalize(p[None, :], t)[0] for p, t in zip(X, tonics)])
    protos, labels, sim = spherical_kmeans(normed, k=3, seed=0)
    purity = sum(np.bincount(labels_true[labels == j]).max()
                 for j in range(3) if (labels == j).any()) / len(labels)
    rot = rotate_to_root(root_normalize(X[0][None, :], tonics[0])[0], tonics[0])
    roundtrip = np.abs(rot - X[0]).max() / X[0].max()

    print(f"  [{'PASS' if purity > 0.9 else 'FAIL'}] cluster purity vs mode "
          f"labels across random keys: {purity:.3f}")
    print(f"  [{'PASS' if roundtrip < 0.02 else 'FAIL'}] root-normalize/"
          f"rotate round-trip rel err: {roundtrip:.4f}")
    print(f"  [INFO] mean within-cluster cosine: {sim:.3f}")
    ok = purity > 0.9 and roundtrip < 0.02
    print(f"\nself-test {'PASSED' if ok else 'FAILED'}")
    return ok


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--audio_dir", type=str, help="Directory of crop audio files")
    parser.add_argument("--out", type=str, default="chroma_prototypes.npz")
    parser.add_argument("--k", type=int, default=10, help="Prototypes per band group")
    parser.add_argument("--tonic-key", type=str, default="tonic",
                        help="JSON sidecar field holding pitch class 0-11")
    parser.add_argument("--medoids", action="store_true",
                        help="Farthest-point medoids instead of k-means (max distinctness)")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.audio_dir:
        parser.error("provide --audio_dir or --selftest")

    profiles, tonics, paths = extract_profiles(
        Path(args.audio_dir), args.tonic_key, args.max_files)
    if len(profiles) < args.k * 3:
        logger.warning(f"only {len(profiles)} profiles for k={args.k} - "
                       f"consider smaller k")
    palette = build_palette(profiles, tonics, args.k, args.medoids, args.seed)
    np.savez(args.out, **palette,
             n_files=len(profiles), k=args.k, n_chroma=sc.SAME_N_CHROMA,
             note="prototypes are root-normalized to C; use rotate_to_root() "
                  "at selection time")
    print(f"saved {args.k} bass + {args.k} melody prototypes "
          f"from {len(profiles)} crops -> {args.out}")
