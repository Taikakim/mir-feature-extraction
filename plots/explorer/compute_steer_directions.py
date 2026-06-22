"""Precompute training-free latent-steering directions for the SAO-Small VAE.

For each per-frame MIR feature, ridge-regress feature[t] <- latent_frame[t] (64-d) over a
sample of crops, giving a unit direction beta[64] in latent space. Editing a latent by

    z += strength * sigma_proj * beta_dir        (broadcast over all 256 frames)

moves the decoded audio's feature along that axis with NO trained head — training-free
steering (validated on SA3 SAME-L; SAO-Small's conv-VAE exposes acoustics even more
linearly, e.g. bass-RMS corr 0.965, so directions are clean edit axes here).

`strength` is the GUI slider value in natural-std units: +1 = +1 sigma along the axis.

Output: models/latch/steer_directions.npz  (consumed by plots/explorer/latch.py).
Run:    mir/bin/python plots/explorer/compute_steer_directions.py [--n 400]
"""
import sys, argparse, sqlite3, random
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
from core.timeseries_db import TimeseriesDB

LAT = Path("/run/media/kim/Lehto/latents")
DB_PATH = "data/timeseries.db"
OUT = Path("models/latch/steer_directions.npz")
# slider/feature name -> per-frame TS field used as the regression target
FEATURES = {
    "rms_energy_bass": "rms_energy_bass_ts",
    "rms_energy_body": "rms_energy_body_ts",
    "rms_energy_mid":  "rms_energy_mid_ts",
    "rms_energy_air":  "rms_energy_air_ts",
    "spectral_flux":   "spectral_flux_ts",
    "spectral_flatness": "spectral_flatness_ts",
    "spectral_skewness": "spectral_skewness_ts",
    "spectral_kurtosis": "spectral_kurtosis_ts",
}
ALPHA = 30.0


def main(n):
    db = TimeseriesDB.open()
    con = sqlite3.connect(DB_PATH)
    keys = [r[0] for r in con.execute("SELECT key FROM ts")]
    random.Random(0).shuffle(keys)
    Xs = []
    Ys = {f: [] for f in FEATURES}
    bright = []
    used = 0
    for k in keys:
        if used >= n:
            break
        track = k.rsplit("_", 1)[0]
        p = LAT / track / f"{k}.npy"
        if not p.exists():
            continue
        try:
            arrs = db.get(k)
            if arrs is None or any(tf not in arrs for tf in FEATURES.values()):
                continue
            lat = np.load(p).astype(np.float32)            # (64, 256)
            if lat.shape != (64, 256):
                continue
        except Exception:
            continue
        Xs.append(lat.T)                                   # (256, 64)
        for f, tf in FEATURES.items():
            Ys[f].append(arrs[tf].astype(np.float32))
        # brightness proxy: treble-minus-bass spectral tilt (centroid-like, dB)
        bright.append(arrs["rms_energy_air_ts"].astype(np.float32)
                      - arrs["rms_energy_bass_ts"].astype(np.float32))
        used += 1

    X = np.concatenate(Xs)                                 # (used*256, 64)
    print(f"used {used} crops -> X {X.shape}")
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xn = (X - mu) / sd
    XtX = Xn.T @ Xn + ALPHA * np.eye(64, dtype=np.float32)

    targets = dict(Ys); targets["brightness"] = bright
    out = {}
    print(f"\n{'feature':<20s} {'R2':>7s} {'sigma':>8s}")
    for f, ylist in targets.items():
        y = np.concatenate(ylist)
        ym, ys_ = y.mean(), y.std() + 1e-6
        yn = (y - ym) / ys_
        beta = np.linalg.solve(XtX, Xn.T @ yn)             # standardized space
        r2 = float(1 - ((yn - Xn @ beta) ** 2).mean())
        beta_raw = beta / sd                               # back to latent units
        beta_dir = (beta_raw / (np.linalg.norm(beta_raw) + 1e-12)).astype(np.float32)
        sigma_proj = float((X @ beta_dir).std())
        out[f"{f}__dir"] = beta_dir
        out[f"{f}__sigma"] = np.float32(sigma_proj)
        out[f"{f}__r2"] = np.float32(r2)
        print(f"{f:<20s} {r2:>7.3f} {sigma_proj:>8.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)
    print(f"\nsaved {OUT}  ({len(targets)} directions)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    main(ap.parse_args().n)
