# AVP Open Dataset — Release Design

**Status:** Design (brainstorm captured 2026-07-09; not yet approved for implementation)
**Owner:** Kim (aavepyörä / Summamutikka)
**Author of this draft:** WINTERMUTE, from the 2026-07-09 design conversation
**Goal:** Publish Kim's own ~30-year electronic-music catalogue as a distributable,
radically-reproducible **SA3 + MIR dataset** — the "keys included" showcase of the MIR
feature pipeline and the SA3 scripts — under a training-permissive license, timed to land
with a model release + write-up.

> This is forward-planning. Nothing here is built yet. The canonical build (§7) is
> explicitly gated on the feature set, SA3 checkpoint, and captions being frozen.

---

## 1. Vision / positioning

The strategic thesis (Kim's, sharpened in discussion): the value is not the audio — there is
infinite scraped audio — it is **legitimacy**. In a field that is currently defensive
(artists opting out, labs scraping in the dark and being sued), a **consented, creator-authored,
richly-annotated, rights-clean** electronic-music dataset is scarce *precisely because* everyone
else is hoarding or litigating. The moat is the clean provenance chain, not the music.

This mirrors Kim's 2001 free-download move: the edge then was **distribution** (findability when
others hoarded → the music reached scenes that scarcity never could); the edge now is
**provenance + consent**. The play is legibility, not monetization — the open release + write-up
is the portfolio artifact that makes Kim findable to opportunities he can't foresee, the same way
free CDrs made him findable to the Ukrainian/Russian underground. Give it away → reach →
opportunity: a pattern that already paid off once.

**End-to-end clean chain as the headline asset:** SA3 trained on ethical material only →
fine-tuned on Kim's own consented catalogue → released with a datasheet documenting consent top to
bottom. A *fully-consented generative-music stack* is a position almost no one else can claim.

---

## 2. Governing policy: the Invertibility Rule

The single technical-legal principle that governs what is safe to release. **Release a
representation in inverse proportion to how well the source audio can be reconstructed from it.**

| Representation | Reconstructable? | Policy |
|---|---|---|
| **Full-mix audio** | is the audio, but a *licensed finished work* | **Release** — this is the permitted form under sample-library EULAs (a finished track incorporating samples is what the license grants) |
| **Hand-crafted MIR features / timeseries** (multiband RMS, spectral, onsets, chroma/HPCP, per-stem envelopes; ~20 fields @ 100 Hz) | No — wildly lossy, non-invertible; a fact *about* the sound, not the sound | **Release for all tracks/stems**, sample-contaminated or not |
| **Neural embeddings** (MERT, essentia EffNet, Audiobox) | Partially — audio-from-embedding inversion research exists | **Gray zone** — gate to the certified-clean subset; keep out of the contaminated-stem layer |
| **SA3 latents** | Yes — the decoder inverts them to audio at high fidelity | **Audio-equivalent** → treat exactly like the stem/mix audio; **reproduce-only**, and never for contaminated stems |

Why it matters: the finished mix **launders** sample-library content into a licensed work; a stem
**un-launders** it (an isolated library sound is the "cleanly separated" content EULAs forbid
redistributing). Latents un-launder it too (they decode back to the sample). Hand-crafted
timeseries never contain the sample at all.

---

## 3. Licensing & provenance

- **Core (Kim's works + facts): CC-BY 4.0.** Full mixes (finished works), numeric features
  (uncopyrightable facts), Kim's creator captions (his text), MIDI (his composition). Attribution
  required; commercial + ML training permitted — the choice that makes the dataset *usable* by labs
  rather than merely admired. (NC was explicitly rejected: it blocks ML-training use in practice.)
- **Model-generated layers** (Flamingo/Granite captions, essentia labels, Demucs/BS-RoFormer stems,
  Audiobox scores): ride their **upstream licenses**. Ship with a `LICENSES.md` per-layer map, mark
  clearly as machine-generated, keep distinct from creator-authored content.
- **Provenance / consent:** the ethical moat. It is Kim's own music — consent is trivially satisfied
  and, crucially, *documented* (the opposite of scraped datasets). Per-track attestations carry the
  rights facts (see §4 stems).
- **Public vs private separation (do not blur):** the *public* face is Kim's own music only —
  unimpeachable. Training on **owned third-party goa** (≈2/3 purchased; the *Bartz v. Anthropic*
  ruling turned on lawful acquisition) stays **private** — Kim's own models, his own use. Legality ≠
  community goodwill; the public artifact must be spotless.

---

## 4. Content model (the layers)

**Base — v0.x (ships now, decoupled from any deadline; all clean):**
- Full mixes (FLAC) — the whole catalogue.
- Numeric features (`.INFO` JSON) — facts, CC-BY. Model-caption fields split OUT into the captions layer.
- Whole-track timeseries (`.TIMESERIES.npz`, 100 Hz, 20 fields).
- Model-generated captions (Flamingo/Granite) — *with* upstream-license note + machine-generated flag.
  (Alternative if a spotless-license v0.x is preferred: hold all captions until creator captions land.)

**Premium — v1.x (as ready; does not gate v0.x):**
- **Per-stem timeseries/features — ALL ~40 stem tracks.** Clean under the Invertibility Rule
  regardless of sample content (no reconstructable audio). This is the *most valuable stem artifact for
  the conditioning use case* (it is literally the LatCH/control-adapter target signal) with zero
  liability — already produced by the pipeline (per-stem RMS + onset envelopes in the npz;
  `per_stem_rhythm`/`per_stem_harmonic` scalars).
- **Certified-clean stem AUDIO — all-synth subset only.** Ship stem WAVs only for tracks Kim can attest
  are 100% his own synthesis/recording (no third-party samples — drums are the usual contamination
  point). Per-track provenance line: *"all sounds synthesized/performed by the artist; no third-party
  samples."* Smaller, legally bulletproof.
- **Creator captions** — the crown jewel; artist-authored, cleanly separated from model captions.
- **MIDI** — legally spotless (his composition, no sample audio); acknowledged thin on timbre for
  electronic music, so positioned as a clean supplementary layer (note/harmony/rhythm/transcription),
  not a stems substitute.
- **Detached project files** (`.bwproject` etc., sample-content stripped) — clean (Kim's authorship),
  a fascinating sound-design window, but DAW-locked + non-rendering + proprietary → **archival "process"
  bonus**, not core ML data.
- *(Future, optional)* **own synth-patch parameter sets** — clean AND novel (the parameter↔audio data
  the reverse-synthesis line wants; creator-authored patches are rarer than creator captions).
  Third-party presets excluded.

**Excluded / reproduce-only:**
- SA3 latents (checkpoint-specific *and* audio-invertible) → shipped as the encode *script*, not files,
  so latents regenerate per checkpoint.
- Contaminated stem audio; neural embeddings for contaminated stems.

---

## 5. Structure / format

Mirrors the pipeline output so minimal new tooling is needed; premium layers slot into the same tree.

```
dataset/
  README.md              # HF dataset card: summary, license, usage, citation
  DATASHEET.md           # Gebru datasheet (see §8)
  LICENSES.md            # per-layer license map (CC-BY core; upstream for model layers)
  CITATION.cff           # so people cite it → attribution flows back
  schema/features.json   # data dictionary = FEATURE_RANGES + a description per field
  schema/timeseries.md   # the 20 whole-track fields, 100 Hz grid, npz layout
  manifest.parquet/.jsonl# one row/track: id, artist, year, dur, bpm, key, paths, flags
  audio/<track>/full_mix.flac
  features/<track>.json          # numeric features (facts → CC-BY)
  captions/<track>.model.json    # Flamingo/Granite — provenance + upstream license
  captions/<track>.creator.json  # (v1.x) artist-authored
  timeseries/<track>.TIMESERIES.npz
  stems_ts/<track>/<stem>.*      # (v1.x) per-stem timeseries — ALL 40
  stems/<track>/*.flac           # (v1.x) certified all-synth subset ONLY
  midi/<track>/*                 # (v1.x)
  projects/<track>.*             # (v1.x) sample-stripped, archival
  reproduce/                     # THE KEYS (see §7)
```

Distribution surfaces: **HuggingFace Hub** (usable — `load_dataset`, streaming, the card) **+ Zenodo**
(DOI, citable, permanent archive). HF sharding + Zenodo deposit sizing depend on catalogue scale (§9).

---

## 6. Reproducibility as a first-class pillar

The existing features are a **patchwork** — accreted across months of pipeline runs, varying code
states, configs, and model versions. Each number is correct, but the dataset can't point at one commit
and say "this produced all of it." That is the gap the canonical build closes.

**Canonical build = clean-room regeneration from the original WAVs, one pinned configuration, on LUMI.**
Replaces fuzzy accreted provenance with the sentence a trustworthy datasheet needs: *"every feature and
latent in v1.0 was produced by mir@`<commit>` with config X and model weights Y, in environment Z, from
these WAVs."*

**What "reproducible" honestly means here:** *not* bit-identical (madmom's nets, ROCm kernels, and
LUMI's gfx90a vs local RDNA4 all introduce nondeterminism; cross-hardware determinism is unattainable).
It means **documented, single-config, re-runnable, equivalent-within-tolerance, from the raw audio** —
already lightyears past any published music dataset.

**The complete "keys" bundle** (scripts alone reproduce nothing in five years):
`code (pinned commit) + exact config + model weights + archived environment (container/SIF) + seeds +
the literal command`. Archive the container *alongside* the data.

**Execution shape (from the LUMI scaffolding, `SAO/lumi/`):**
1. Freeze inputs first (feature set, SA3 checkpoint, captions) — this is the canonical build; don't chase
   a moving target.
2. Stage to LUMI: original WAVs + pinned mir pipeline snapshot + model weights (essentia ONNX, Flamingo
   GGUF, madmom, Audiobox, Demucs/BS-RoFormer) + the SA3 encoder, in a cotainr SIF.
3. **Parity gate on gfx90a first:** CPU features (spectral/RMS/chroma/rhythm/timbral, the whole
   timeseries) port cleanly and are the bulk; GPU steps (essentia embeddings, Audiobox, Flamingo, SA3
   encode) get a few-track parity check vs the RDNA4 output *before* the full corpus run.
4. Full regeneration → the released artifacts are a *cached build* of the documented pipeline.

**Positioning:** this is publishable on its own — a fully-regenerable MIR + latent dataset speaks to ML's
reproducibility crisis independent of the music. Most "open datasets" ship the outputs and a paragraph;
almost none ship the *generator* in executable form.

---

## 7. Phasing / versioning

- **v0.x** — base: full mixes + numeric features + whole-track timeseries + model captions. Ships with the
  model release; gated on nothing.
- **v1.0** — adds per-stem timeseries (all 40) + certified-clean stem audio (subset) + creator captions +
  MIDI, once ready. Ideally coincides with the canonical LUMI regeneration so v1.0 carries single-commit
  provenance.
- **v1.x** — detached projects, synth-patch parameter sets, additional stems as recorded.

Semantic versioning; the pipeline is the source of truth, released artifacts are cached builds.

---

## 8. Datasheet (Gebru et al.) — sections to fill

Motivation · Composition · Collection process (incl. **consent** — Kim is the artist) · Preprocessing/
cleaning/labeling (the pinned pipeline, §6) · Uses · Distribution (CC-BY + upstream layers) · Maintenance.
The datasheet is what earns researcher trust; ship it as a first-class file, not an afterthought.

---

## 9. Open decisions (pending Kim)

1. **Catalogue scale** — total tracks / GB across the 30 years (drives HF sharding + Zenodo deposit sizing).
2. **Which stem tracks pass the all-synth attestation** — determines whether certified-clean stem *audio*
   is a headline feature or a handful (per-stem *timeseries* ships for all 40 regardless).
3. **Model captions in v0.x** — include-with-disclosure (default) vs hold-for-spotless-CC-BY.
4. **Embeddings policy** — if per-stem neural embeddings are ever added, confirm gating to the clean subset.
5. **SA3 latents** — confirmed reproduce-only (script, not files); revisit only if plug-and-play convenience
   is wanted for a frozen checkpoint.

---

## 10. Related assets already in place

- MIR pipeline + `FEATURE_RANGES` (→ the data dictionary), whole-track timeseries producer
  (`src/spectral/whole_track_timeseries.py`), per-stem feature extractors.
- `src/tools/augment_tracks.py` (the aug *recipe* — shipped, not the aug outputs).
- LUMI bundle `SAO/lumi/` (staging, SIF build, parity-gate pattern) for the canonical regeneration.
- SA3 encode scripts (`stable-audio-3/scripts/sa3_encode_from_manifest.py`) for the latent layer.
