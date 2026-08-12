#!/usr/bin/env python3
"""Verify the f0 melody-target backfill: FULL coverage, non-destructive merge, and stem-format
parity. Written before the pass finished, because the failure it guards against is the one that
reports success.

WHY NOT A SAMPLE. The bug this pass already survived was a silent coverage gap: stems were
looked up as .flac only, so the 818 of 4461 folders carrying .mp3 stems got no f0, logged a
warning, and would have been retried forever while the run reported success. A sampled "did the
fields land" check is blind to exactly that -- it passes as long as the tracks it happens to
draw are fine. So this checks EVERY track.

THREE CHECKS, because coverage alone is not enough:
  1. COVERAGE   -- every track folder's sidecar carries all four f0 fields.
  2. INTEGRITY  -- no sidecar LOST a pre-existing field. The pass rewrites 4461 files in place;
                   a lost legacy field is far worse than a missing f0 and must stop everything.
  3. PARITY     -- the .mp3-stem group and the .flac-stem group produce comparable f0. If mp3
                   stems track measurably worse, the head would train on two populations as if
                   they were one. A coverage sweep cannot see this: the fields are all present,
                   they are just worse on one group.

EXPECTED, NOT A FAILURE: 4 folders have no stems in any format and correctly get no f0. The
verdict is 4457/4461, not 4461/4461 -- check for the round number and you file a false failure.

    mir/bin/python src/tools/verify_f0_coverage.py
"""
import json
import sys
from pathlib import Path

import numpy as np

CORPUS = Path("/run/media/kim/Mantu/ai-music/Goa_Separated")
TS = Path("/run/media/kim/Lehto/timeseries")
F0_FIELDS = ["f0_other_ts", "f0_other_voiced_ts", "f0_bass_ts", "f0_bass_voiced_ts"]
LEGACY_MIN = 46          # field count before the melody backfill


STEM_EXTS = (".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aiff")


def stem_format(d: Path):
    for ext in STEM_EXTS:
        if (d / f"other{ext}").exists() and (d / f"bass{ext}").exists():
            return ext
    return None


def is_track(d: Path) -> bool:
    """A track folder is one with a full_mix, NOT merely a directory.

    THE DENOMINATOR IS ITSELF A TRAP, and this checker walked into it. Counting every
    subdirectory gives 4463; the producer processes 4461. The two extra are directories literally
    NAMED ".flac" and ".mp3" -- junk entries that contain no full_mix -- so a dirs-based check
    reports 2 phantom missing tracks. Use the producer's own criterion (a full_mix in any of its
    six extensions) or the verifier disagrees with the pass for reasons that have nothing to do
    with f0.
    """
    return any((d / f"full_mix{e}").exists() for e in STEM_EXTS)


def main() -> int:
    tracks = sorted(p for p in CORPUS.iterdir() if p.is_dir() and is_track(p))
    missing, stemless, shrunk, absent_npz = [], [], [], []
    voiced = {".flac": [], ".mp3": []}

    for d in tracks:
        fmt = stem_format(d)
        npz = TS / f"{d.name}.TIMESERIES.npz"
        if not npz.exists():
            absent_npz.append(d.name)
            continue
        try:
            with np.load(npz, allow_pickle=True) as z:
                have = set(z.files) - {"__meta__"}
                if fmt is None:
                    stemless.append(d.name)
                elif not set(F0_FIELDS) <= have:
                    missing.append((d.name, fmt, sorted(set(F0_FIELDS) - have)))
                else:
                    for v in ("other", "bass"):
                        voiced[fmt].append((v, float(z[f"f0_{v}_voiced_ts"].mean())))
                if len(have) < LEGACY_MIN:
                    shrunk.append((d.name, len(have)))
        except Exception as e:
            missing.append((d.name, fmt, f"UNREADABLE: {type(e).__name__}"))

    n = len(tracks)
    covered = n - len(missing) - len(stemless) - len(absent_npz)
    print(f"tracks {n} | f0 present {covered} | legitimately stemless {len(stemless)} "
          f"| MISSING {len(missing)} | no sidecar {len(absent_npz)}")
    for name, fmt, what in missing[:10]:
        print(f"  MISSING {name[:48]:48s} stems={fmt} {what}")
    if shrunk:
        print(f"  *** INTEGRITY FAILURE: {len(shrunk)} sidecars have < {LEGACY_MIN} fields "
              f"-- the merge LOST data. STOP. e.g. {shrunk[:3]}")

    print("\nPARITY (voiced fraction by stem format -- a coverage sweep cannot see this):")
    for v in ("other", "bass"):
        row = {}
        for fmt in (".flac", ".mp3"):
            vals = [x for vv, x in voiced[fmt] if vv == v]
            row[fmt] = (len(vals), float(np.median(vals)) if vals else float("nan"))
        d_med = row[".flac"][1] - row[".mp3"][1]
        flag = "  <-- CHECK: groups differ by >10 points" if abs(d_med) > 0.10 else ""
        print(f"  f0_{v:5s} flac n={row['.flac'][0]:4d} median {row['.flac'][1]:.3f} | "
              f"mp3 n={row['.mp3'][0]:4d} median {row['.mp3'][1]:.3f} | "
              f"diff {d_med:+.3f}{flag}")

    ok = not missing and not shrunk
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} "
          f"({covered}/{n - len(stemless)} non-stemless tracks carry all four f0 fields)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
