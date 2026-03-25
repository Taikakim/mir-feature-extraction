#!/usr/bin/env python3
"""
latent_shape_server.py - API server for 3D Latent Shape Visualizer.

Serves the frontend static files and provides API endpoints to get
PCA-projected 3D trajectories for audio latents.
"""

import argparse
import configparser
import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from sklearn.decomposition import PCA
import csv

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
MIR_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INI = MIR_DIR / "latent_player.ini"
STATIC_ROOT = MIR_DIR / "plots" / "latent_shape_explorer"
PCA_MODEL_PATH = MIR_DIR / "models" / "global_pca_3d.npz"
CORRELATIONS_PATH = MIR_DIR / "models" / "latent_correlations.json"

STEM_SUFFIXES = {"_bass", "_drums", "_other", "_vocals"}
STEMS = ["drums", "bass", "other", "vocals"]

_latent_dir = None
_stem_dir = None
_raw_audio_dir = None
_pca_mean = None
_pca_components = None

_metadata_cache = None
_source_dir = None

# ---------------------------------------------------------------------------
# PCA Fitting / Loading
# ---------------------------------------------------------------------------
def load_or_fit_pca():
    global _pca_mean, _pca_components
    if PCA_MODEL_PATH.exists():
        print(f"Loading PCA model from {PCA_MODEL_PATH}...")
        data = np.load(str(PCA_MODEL_PATH))
        _pca_mean = data['mean']
        _pca_components = data['components']
        return

    print("Fitting global PCA model (this may take a minute)...")
    tracks = [d for d in _latent_dir.iterdir() if d.is_dir()]
    sampled_frames = []
    
    # Sub-sample up to 2000 tracks, 10 frames each
    np.random.seed(42)
    sample_tracks = np.random.choice(tracks, min(2000, len(tracks)), replace=False)
    
    for i, t in enumerate(sample_tracks):
        npys = list(t.glob("*.npy"))
        full_npys = [n for n in npys if not any(n.stem.endswith(s) for s in STEM_SUFFIXES)]
        if not full_npys: continue
        
        # Load the first crop
        try:
            data = np.load(str(full_npys[0])).astype(np.float32) # [64, T]
            T = data.shape[1]
            if T > 10:
                idx = np.linspace(0, T - 1, 10).astype(int)
                sampled_frames.append(data[:, idx].T)
        except Exception as e:
            pass
            
        if (i+1) % 100 == 0:
            print(f"  Loaded {i+1} tracks for PCA...")

    X = np.vstack(sampled_frames)
    
    # Drop any rows containing NaNs
    valid_rows = ~np.isnan(X).any(axis=1)
    X_clean = X[valid_rows]
    
    print(f"Fitting PCA on feature matrix of shape {X_clean.shape} (dropped {X.shape[0] - X_clean.shape[0]} NaN rows)...")
    pca = PCA(n_components=3)
    pca.fit(X_clean)
    
    _pca_mean = pca.mean_
    _pca_components = pca.components_
    
    os.makedirs(PCA_MODEL_PATH.parent, exist_ok=True)
    np.savez(PCA_MODEL_PATH, mean=_pca_mean, components=_pca_components)
    print(f"PCA fit complete! Variance ratio: {pca.explained_variance_ratio_}")

def project_3d(latent_data):
    # latent_data is [64, T]
    # return [T, 3]
    X = latent_data.T
    return np.dot(X - _pca_mean, _pca_components.T)


# ---------------------------------------------------------------------------
# Timecode helpers
# ---------------------------------------------------------------------------

def _read_sidecar_times(path: Path) -> list:
    """Read a sidecar file (one float per line) → list of floats."""
    if not path.exists():
        return None          # None signals "file missing"
    times = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                times.append(float(line))
            except ValueError:
                pass
    return times


def _read_timecodes(track: str, crop_id: str) -> dict | None:
    """Return timecodes for a crop window, offset to crop-relative seconds.

    Returns None if _source_dir is unconfigured or the crop JSON is absent.
    Returns a dict with keys: beats, downbeats, onsets, duration, bpm, missing.
    """
    if _source_dir is None:
        return None

    crop_json = _latent_dir / track / f"{crop_id}.json"
    if not crop_json.exists():
        return None

    try:
        meta = json.loads(crop_json.read_text())
    except Exception:
        return None

    start_t = float(meta.get("start_time", 0.0))
    end_t   = float(meta.get("end_time",   meta.get("duration", 0.0) + start_t))
    duration = end_t - start_t

    sidecar_map = {
        "beats":     _source_dir / track / f"{track}.BEATS_GRID",
        "downbeats": _source_dir / track / f"{track}.DOWNBEATS",
        "onsets":    _source_dir / track / f"{track}.ONSETS",
    }

    result = {"duration": round(duration, 4), "missing": []}
    for key, path in sidecar_map.items():
        raw = _read_sidecar_times(path)
        if raw is None:
            result[key] = []
            result["missing"].append(key)
        else:
            filtered = [round(t - start_t, 5)
                        for t in raw if start_t - 1e-4 <= t <= end_t + 1e-4]
            # For downbeats: prepend the last downbeat before start_t as a
            # phase anchor at 0.0 when no downbeat falls exactly at start_t.
            if key == "downbeats" and (not filtered or filtered[0] > 1e-4):
                before = [t for t in raw if t < start_t - 1e-4]
                if before:
                    filtered = [0.0] + filtered
            result[key] = filtered

    # BPM from median beat interval
    beats = result["beats"]
    if len(beats) >= 2:
        intervals = [beats[i+1] - beats[i] for i in range(len(beats)-1)]
        intervals.sort()
        med = intervals[len(intervals)//2]
        result["bpm"] = round(60.0 / med, 2) if med > 0 else None
    else:
        result["bpm"] = None

    if not result["missing"]:
        del result["missing"]

    return result


def _compute_average_shape(track: str) -> list | None:
    """Load all full-mix latent files for a track, average, project 3D.

    Shorter files are zero-padded to the length of the longest file.
    Returns [[x,y,z], ...] or None if no latent files exist.
    """
    track_dir = _latent_dir / track
    npys = [p for p in sorted(track_dir.glob("*.npy"))
            if not any(p.stem.endswith(s) for s in STEM_SUFFIXES)]
    if not npys:
        return None

    arrays = []
    max_T  = 0
    for npy in npys:
        try:
            arr = np.load(str(npy)).astype(np.float32)  # [64, T]
            arrays.append(arr)
            max_T = max(max_T, arr.shape[1])
        except Exception:
            pass

    if not arrays:
        return None

    # Zero-pad all arrays to max_T, then average
    padded = np.zeros((len(arrays), 64, max_T), dtype=np.float32)
    for i, arr in enumerate(arrays):
        padded[i, :, :arr.shape[1]] = arr
    mean_latent = padded.mean(axis=0)   # [64, max_T]

    return project_3d(mean_latent).tolist()


# ---------------------------------------------------------------------------
# API Handlers
# ---------------------------------------------------------------------------
class LatentShapeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def log_message(self, format, *args):
        # Mute normal GET logs
        pass

    def send_error_json(self, status, msg):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"error": msg}).encode("utf-8"))

    def send_json(self, data):
        body = json.dumps(data)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/status":
            self.send_json({"ok": True})
            return

        if path == "/api/tracks":
            tracks = sorted([d.name for d in _latent_dir.iterdir() if d.is_dir()])
            self.send_json({"tracks": tracks})
            return

        if path == "/api/pca":
            if _pca_components is not None:
                self.send_json({"components": _pca_components.tolist()})
            else:
                self.send_error_json(500, "PCA model not loaded")
            return

        if path == "/api/tracks.csv":
            csv_path = MIR_DIR / "plots" / "tracks.csv"
            if csv_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(csv_path.stat().st_size))
                self.end_headers()
                with open(csv_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error_json(404, "tracks.csv not found")
            return

        if path == "/api/correlations":
            if CORRELATIONS_PATH.exists():
                with open(CORRELATIONS_PATH) as f:
                    self.send_json(json.load(f))
            else:
                self.send_error_json(404, "Correlations file not found")
            return

        if path == "/api/crops":
            track = query.get("track", [""])[0]
            if not track:
                return self.send_error_json(400, "Missing track")
            
            track_path = _latent_dir / track
            if not track_path.exists():
                return self.send_error_json(404, "Track not found")
                
            crops = []
            for npy in sorted(track_path.glob("*.npy")):
                if any(npy.stem.endswith(s) for s in STEM_SUFFIXES):
                    continue
                json_path = npy.with_suffix(".json")
                pos = 0.0
                if json_path.exists():
                    try:
                        data = json.loads(json_path.read_text())
                        pos = float(data.get("position", 0.0))
                    except Exception:
                        pass
                crops.append({"id": npy.stem, "position": pos})
            crops.sort(key=lambda x: x["position"])
            self.send_json({"crops": crops})
            return

        if path == "/api/shape":
            track = query.get("track", [""])[0]
            crop_id = query.get("crop", [""])[0]
            if not track or not crop_id:
                return self.send_error_json(400, "Missing track or crop")

            stem_root = _stem_dir or _latent_dir
            fm_path = _latent_dir / track / f"{crop_id}.npy"
            if not fm_path.exists():
                return self.send_error_json(404, "Full mix crop not found")

            shapes = {}
            # Full mix
            try:
                fm_data = np.load(str(fm_path)).astype(np.float32)
                shapes['fullmix'] = project_3d(fm_data).tolist()
            except Exception as e:
                return self.send_error_json(500, f"Error processing full mix: {e}")

            # Stems
            for stem in STEMS:
                stem_npy = stem_root / track / f"{crop_id}_{stem}.npy"
                if stem_npy.exists():
                    try:
                        stem_data = np.load(str(stem_npy)).astype(np.float32)
                        shapes[stem] = project_3d(stem_data).tolist()
                    except Exception as e:
                        pass
            
            self.send_json(shapes)
            return

        if path == "/api/timecodes":
            track   = query.get("track", [""])[0]
            crop_id = query.get("crop",  [""])[0]
            if not track or not crop_id:
                return self.send_error_json(400, "Missing track or crop")
            result = _read_timecodes(track, crop_id)
            if result is None:
                return self.send_error_json(404,
                    "Crop not found or source_dir not configured")
            self.send_json(result)
            return

        if path == "/api/average-shape":
            track = query.get("track", [""])[0]
            if not track:
                return self.send_error_json(400, "Missing track")
            if not (_latent_dir / track).is_dir():
                return self.send_error_json(404, "Track not found")
            points = _compute_average_shape(track)
            if points is None:
                return self.send_error_json(404, "No latent files for track")
            self.send_json({"points": points})
            return

        # Fallback to serving static files
        super().do_GET()


# ---------------------------------------------------------------------------
# Setup and Run
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_INI))
    parser.add_argument("--port", type=int, default=7892)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    global _latent_dir, _stem_dir, _raw_audio_dir, _source_dir
    _latent_dir = Path(config.get("server", "latent_dir", fallback="/tmp"))
    if not _latent_dir.exists():
        print(f"Error: latent_dir not found at {_latent_dir}", file=sys.stderr)
        sys.exit(1)

    sd = config.get("server", "stem_dir", fallback="")
    _stem_dir = Path(sd) if sd else _latent_dir

    rad = config.get("server", "raw_audio_dir", fallback="")
    _raw_audio_dir = Path(rad) if rad else _latent_dir

    sd_src = config.get("server", "source_dir", fallback="")
    _source_dir = Path(sd_src) if sd_src else None
    if _source_dir:
        print(f"  Source dir : {_source_dir}")

    print(f"Loading PCA projection space...")
    load_or_fit_pca()

    print(f"Starting latent shape server on port {args.port}...")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), LatentShapeHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
