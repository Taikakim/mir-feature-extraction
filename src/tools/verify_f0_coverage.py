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
  2. INTEGRITY  -- no sidecar lost one of the 36 always-present fields. The pass rewrites 4461
                   files in place; a lost legacy field is far worse than a missing f0 and must
                   stop everything. This is a floor DERIVED from the corpus, not an assumed
                   field count -- see UNIVERSAL for why the assumed version false-alarmed -- and
                   it is the second line of defence behind a per-track before/after diff.
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
# INTEGRITY FLOOR, DERIVED FROM THE CORPUS RATHER THAN ASSUMED. The first version of this
# checker asserted every sidecar has >= 46 fields and printed "INTEGRITY FAILURE ... STOP" below
# that. Measured field counts are 38 (x4), 44 (x1), 46 (x3756), 48 (x1), 50 (x699): the 4
# stemless tracks legitimately lack the 8 per-stem fields (46-8=38), and one track is an
# irregular 44/48. So the assumed floor fires the LOUDEST alarm in the tool on five healthy
# tracks -- a false positive that would have stopped the whole pipeline. THE_FINN found the same
# variation independently from a before-snapshot; this is the version that survives it.
#
# The 36 fields below are present in all 4461 goa sidecars. Note the weakness honestly: a
# membership floor computed post-hoc cannot detect a field the pass destroyed everywhere (it
# would simply not be in the universal set). The authoritative integrity test is a per-track
# before/after diff -- THE_FINN holds that one. This is the second line of defence, not the first.
UNIVERSAL = {
    "attack_logattacktime_ts", "attack_maxratio_ts", "attack_strongdecay_ts",
    "attack_tctototal_ts", "bark_bands_ts", "bass_chroma_linmap_ts", "chords_idx_ts",
    "chords_strength_ts", "chroma_linmap_ts", "dissonance_ts", "dyncomplexity_loudness_ts",
    "dyncomplexity_ts", "effnet_genre400_ts", "effnet_instrument_ts", "effnet_moodtheme_ts",
    "erb_bands_ts", "hpcp_ts", "inharmonicity_ts", "loudness_ebu_momentary_ts",
    "loudness_ebu_shortterm_ts", "maest_embed_ts", "novelty_curve_ts", "onset_envelope_ts",
    "pitch_salience_ts", "rms_energy_air_ts", "rms_energy_bass_ts", "rms_energy_body_ts",
    "rms_energy_mid_ts", "spectral_flatness_ts", "spectral_flux_ts", "spectral_kurtosis_ts",
    "spectral_skewness_ts", "stereo_corr_ts", "stereo_width_ts", "va_deam_ts", "va_emomusic_ts",
}
# Parity DECISION is effect size, not significance: at n=4457 a KS test calls a musically
# meaningless 0.02 median gap "divergent". Aligned with THE_FINN's threshold so our two
# independent checkers decide on the same criterion rather than two different ones.
PARITY_MAX_GAP = 0.05


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
                lost = UNIVERSAL - have
                if lost:
                    shrunk.append((d.name, sorted(lost)[:4]))
        except Exception as e:
            missing.append((d.name, fmt, f"UNREADABLE: {type(e).__name__}"))

    n = len(tracks)
    covered = n - len(missing) - len(stemless) - len(absent_npz)
    print(f"tracks {n} | f0 present {covered} | legitimately stemless {len(stemless)} "
          f"| MISSING {len(missing)} | no sidecar {len(absent_npz)}")
    for name, fmt, what in missing[:10]:
        print(f"  MISSING {name[:48]:48s} stems={fmt} {what}")
    if shrunk:
        print(f"  *** INTEGRITY FAILURE: {len(shrunk)} sidecars are missing always-present "
              f"fields -- the merge LOST data. STOP. e.g. {shrunk[:3]}")

    print("\nPARITY (voiced fraction by stem format -- a coverage sweep cannot see this):")
    for v in ("other", "bass"):
        row = {}
        for fmt in (".flac", ".mp3"):
            vals = [x for vv, x in voiced[fmt] if vv == v]
            row[fmt] = (len(vals), float(np.median(vals)) if vals else float("nan"))
        d_med = row[".flac"][1] - row[".mp3"][1]
        flag = (f"  <-- CHECK: groups differ by more than {PARITY_MAX_GAP}"
                if abs(d_med) > PARITY_MAX_GAP else "")
        print(f"  f0_{v:5s} flac n={row['.flac'][0]:4d} median {row['.flac'][1]:.3f} | "
              f"mp3 n={row['.mp3'][0]:4d} median {row['.mp3'][1]:.3f} | "
              f"diff {d_med:+.3f}{flag}")

    ok = not missing and not shrunk
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} "
          f"({covered}/{n - len(stemless)} non-stemless tracks carry all four f0 fields)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
