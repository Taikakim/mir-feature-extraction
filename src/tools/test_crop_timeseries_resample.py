"""Demonstrate the corrected resampler against the one it replaces, on a real sidecar."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/kim/Projects/mir/src/tools")
sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-tools/scripts")
from crop_timeseries_resample import build_crop_timeseries
from whole_track_target_source import resample_axis0

p = Path("/run/media/kim/Lehto/timeseries/Ayahuasca - Propella.TIMESERIES.npz")
z = np.load(p, allow_pickle=True)
meta = json.loads(str(z["__meta__"]))
arrays = {k: z[k] for k in z.files if k != "__meta__"}
N = 4096

def old(start, end):
    """the existing single-rate loop"""
    fr = float(meta.get("frame_rate", 100))
    out = {}
    for k, a in arrays.items():
        s = max(0, int(round(start*fr))); e = min(a.shape[0], int(round(end*fr)))
        if e <= s: return None
        out[k] = resample_axis0(a[s:e], N)
    return out

print("1) LATE CROP (300-347s) -- the drop")
print("   old:", "DROPPED (returned None)" if old(300, 347) is None else "ok")
new = build_crop_timeseries(arrays, meta, 300, 347, N, strict=False)
print(f"   new: {len(new)} fields emitted, none dropped")

print("\n2) EARLY CROP (10-57s) -- the wrong region")
o = old(10, 57)
print(f"   old: {'DROPPED' if o is None else 'ok'} -- and this is EARLY, not late")
# how early does the old code start dropping? bounded by the coarsest field
worst = min((a.shape[0] / 100.0, k) for k, a in arrays.items() if a.shape[0] > 0)
print(f"   coarsest field under the 100 Hz assumption: {worst[1]} -> covers only "
      f"{worst[0]:.2f}s, so ANY crop starting after {worst[0]:.2f}s is dropped")
surv = sum(1 for st in range(0, 380, 10) if old(st, st+47) is not None)
print(f"   crops surviving, start=0,10,...,370s: {surv}/38")
n = build_crop_timeseries(arrays, meta, 10, 57, N, strict=False)
print(f"   new: {len(n)} fields at 10-57s, sliced per field rate")

print("\n3) SENTINEL + CATEGORICAL, isolated from the rate bug")
# old() cannot reach a mid-track window at all, so compare the POOLING rules directly on a
# window that actually has voiced content.
from crop_timeseries_resample import _mean_pool, _mode_pool
st, en = 120.0, 167.0
sl = slice(int(st*100), int(en*100))
fb, mb = arrays["f0_bass_ts"][sl].astype(np.float32), arrays["f0_bass_voiced_ts"][sl].astype(np.float32)
naive = _mean_pool(fb, N)
num, den = _mean_pool(fb*mb, N), _mean_pool(mb, N)
good = np.where(den > 0, num/np.maximum(den,1e-9), 0.0)
ok = (naive > 0) & (good > 0)
err = 12*np.log2(good[ok]/naive[ok])
print(f"   window {st:.0f}-{en:.0f}s, voiced {mb.mean()*100:.0f}%")
print(f"   f0_bass mean-pooled vs masked: {np.mean(np.abs(err)>1)*100:5.1f}% of frames >1 st off, "
      f"max {np.abs(err).max():.1f} st")
ci = arrays["chords_idx_ts"][sl]
cn, co = _mode_pool(ci, N), _mean_pool(ci.astype(np.float32), N)
print(f"   chords_idx mean-pooled: {np.mean(np.isclose(co, np.round(co)))*100:5.1f}% are real chord indices, e.g. {np.round(co[:3],2)}")
print(f"   chords_idx mode-pooled: {np.mean(np.isclose(cn, np.round(cn)))*100:5.1f}% are real chord indices, e.g. {cn[:3]}")

print("\n4) STRICT MODE fails loudly instead of dropping")
try:
    build_crop_timeseries(arrays, meta, 300, 347, N, strict=True)
    print("   NO RAISE -- bad")
except ValueError as e:
    print(f"   raised: {str(e)[:88]}...")
