# Quick-Start: Production Box Setup & Test Sequence

*Tonight's script. Full rationale in `PRODUCTION_MIGRATION.md`.*

---

## 0. Prerequisites

```bash
# Activate the SA3 venv (whichever you use on the production box)
source ~/venvs/sa3/bin/activate   # or conda activate sa3

# Pull the latest branch
git clone https://github.com/Taikakim/mir-feature-extraction
# — or, if already cloned:
git fetch && git checkout same-chroma && git pull

# Install this repo's only non-torch dependency (for build_chroma_prototypes)
pip install soundfile
```

Python path for all scripts is handled automatically via `sys.path.insert` at
the top of each file (points to `src/harmonic/`). No `pip install -e .` needed.

---

## 1. Sanity check (5 min, no audio needed)

```bash
# Step 1a: numpy/scipy selftest — runs on any machine
python src/harmonic/same_chroma.py --selftest
# Expected: 9/9 PASS, rel error ~4e-07, "~300x realtime"

# Step 1b: full environment check — torch + stable_audio_3 + real encoder
python scripts/sa3_refit_readout.py --sanity --model same-l
# Expected output:
#   [PASS] same_chroma --selftest
#   [PASS] torch path matches numpy path (rel err ...)
#   [PASS] GPU path matches numpy path (rel err ...)       ← ROCm
#   [PASS] 10 s encode -> 108 latent frames (n_latent_frames says 108; ...)
#   [INFO] latent shape (1, 256, 108), std ~1.0
#   sanity PASSED

# If frame count is off by one: look at same_chroma.n_latent_frames() and fix.
# If GPU rel err > 1e-3: check ROCm torchaudio install; numpy path is still usable.
```

**Gate:** all PASS before continuing.

---

## 2. Refit the chroma readout heads (15-60 min depending on dataset size)

You need a directory of audio crops (wav/flac). The longer and more varied, the
better. Even 30-60 minutes of music at varied tempos/keys is enough for a rough
fit.

```bash
python scripts/sa3_refit_readout.py \
    --audio_dir /path/to/your/audio/crops \
    --model same-l \
    --out chroma_heads.npz \
    --max-seconds 60 \
    --holdout-every 8

# Typical progress output:
#   50/N files, XXXXX frames
#   solving: XXXXX frames, latent dim 256, N holdout files
#   bass_oct1:    held-out R2=0.XXX  frame-cosine=0.XXX  [OK / rough]
#   mid_oct5:     held-out R2=0.XXX  frame-cosine=0.XXX  [OK]
#   treble_oct9:  held-out R2=0.XXX  frame-cosine=0.XXX  [OK]
#   saved heads -> chroma_heads.npz
```

**Gate thresholds (from PRODUCTION_MIGRATION.md §1):**
- `R2 > 0.3` = OK, proceed
- `R2 0.1-0.3` = rough but try it
- `R2 < 0.1` = check frame alignment (off-by-one is the only common cause);
  not a sign the method is wrong

Bass band (oct1) typically has the strongest linear readout. Mid (oct5) is the
most important for melody steering.

---

## 3. THE steering gate (45 min including listening, 2 presets)

This is the key experiment. It generates baseline + steered audio with the
same seed and measures whether pitch content moves.

```bash
# Preset A: "only C" — blunt test, all pitch content toward C
python scripts/sa3_steer_chroma.py \
    --heads chroma_heads.npz \
    --model medium-base \
    --test only-c \
    --out_dir steering_results

# Preset B: "goa-e" — E phrygian melody + E bass (root5 profile)
python scripts/sa3_steer_chroma.py \
    --heads chroma_heads.npz \
    --model medium-base \
    --test goa-e \
    --prompt "melodic goa trance, hypnotic arpeggiated lead, driving rolling bassline" \
    --out_dir steering_results
```

Each run takes ~2x the generation time (baseline + steered). At 50 steps on
`medium-base` that's roughly 5-10 min per preset on a good GPU.

**What to look for in the log:**
```
guided euler: 50 steps, guidance on first 10 steps, mu=0.03 (const) gamma=0.3 N_iter=4
  step 0: band cosines 0.XXX 0.XXX 0.XXX     ← should increase across steps
  ...
baseline: alignment [0.XXX, 0.XXX, None]  top PCs [('C', 0.34), ...]
steered:  alignment [0.XXX, 0.XXX, None]  top PCs [('C', 0.51), ...]
alignment delta (steered - baseline) per band: [0.0X, 0.0X, None]
```

**LISTEN** to `steering_results/only-c_*/baseline.wav` vs `steered.wav`.
The metric (cosine alignment) is a sanity check. Your ear is the gate.

**Outcome interpretation:**
| Observation | Meaning |
|---|---|
| Pitches audibly shift toward target + delta > 0.05 | ✅ Guidance works — build GUI on this |
| Delta > 0 but no audible change | Metric and audio are decoupled; try `--mu 0.1` |
| Audible pitch shift but artifacts | Reduce `--mu`, try `--frac 0.1`, check `--n-iter 2` |
| No change at any mu | Refit R2 too low, or try `--mu-schedule alpha` + `--mu 0.3` |
| Generation sounds degraded with no pitch shift | Wrong checkpoint (use -base only) |

**Tuning knobs (in order of impact):**
```bash
--mu 0.03          # step size per guidance iteration (try 0.01–0.1)
--n-iter 4         # gradient steps per sampling step (try 2–8)
--frac 0.2         # fraction of steps with guidance (0.1–0.4)
--band-weights 1.0 1.0 0.3   # bass / mid / air loss weights
--mu-schedule const  # or 'alpha' (TFG-faithful but weaker in early window)
```

---

## 4. ZeroSep-lite (10 min, optional but fast)

Only needs a mixed audio file. Pick something with a clear bass/melody separation.

```bash
python scripts/sa3_zerosep_lite.py \
    --input /path/to/mix.wav \
    --model medium-base \
    --prompts "the bassline" "the lead melody" "drums and percussion" \
    --noise-levels 0.4 0.55 0.7 \
    --out_dir zerosep_results

# Output: one wav per (prompt, noise_level) combination + params.json
```

**LISTEN** for: does any cell isolate the named source while suppressing
others? Low noise (0.4) preserves the mixture; high (0.7) regenerates freely.
The separation sweet spot, if it exists, is 0.5-0.6.

This is a 10-minute go/no-go. Only build the full DDPM-inversion wrapper if
something interesting happens here.

---

## 5. Prototype palette (after gate #2 passes, numpy-only)

```bash
python scripts/build_chroma_prototypes.py --selftest
# Expected: cluster purity 1.000, round-trip err ~0.000, PASSED

python scripts/build_chroma_prototypes.py \
    --audio_dir /path/to/your/audio/crops \
    --out prototypes.npz \
    --k 10

# Output: 10 bass + 10 melody prototypes, root-normalized to C
# Use rotate_to_root(prototype, root) at selection time to re-key
```

---

## Test sequence summary

| # | Script | Runtime | Gate |
|---|---|---|---|
| 0a | `same_chroma --selftest` | <1 min | 9/9 PASS |
| 0b | `sa3_refit_readout --sanity` | 2 min | all PASS |
| 1 | `sa3_refit_readout --audio_dir crops` | 15-60 min | R2 > 0.1 |
| 5 | `sa3_zerosep_lite -i mix.wav` | 10 min | listen |
| 2a | `sa3_steer_chroma --test only-c` | 10 min | **LISTEN** |
| 2b | `sa3_steer_chroma --test goa-e` | 10 min | **LISTEN** |
| 4 | `build_chroma_prototypes --audio_dir crops` | ~2 min | selftest PASS |

Step 2 is the decision gate for the entire project. Everything else is prep or
optional. If `--test only-c` produces audibly more C-heavy output, the approach
is valid and the UI work begins.

---

## Notes

- **Checkpoint names** passed to `--model` are whatever `StableAudioModel.from_pretrained()`
  accepts on your box (HuggingFace repo ID or local path). `medium-base` =
  `stabilityai/stable-audio-3-medium-base` or however you've got it cached.
- **ROCm**: scripts use `torch.cuda.is_available()` which returns `True` under ROCm.
  No code changes needed.
- **Memory**: encoding a full crop dataset may need 8-16 GB VRAM. If OOM,
  add `--max-seconds 30` to `sa3_refit_readout` to use shorter crops.
- **same_chroma.py path**: each script does `sys.path.insert(0, ".../src/harmonic")`
  relative to its own location — run scripts from anywhere.
- Full context: `PRODUCTION_MIGRATION.md` (complete rationale, resolved/open
  items, LatCH paper recipe, TADA assessment, ZeroSep port plan).
