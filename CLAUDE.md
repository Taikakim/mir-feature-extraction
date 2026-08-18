# CLAUDE.md

Guidance for Claude Code when working in this repository.

> **Cross-project coordination (read first).** This repo is one of three in the
> mir + Stable Audio pipeline. Shared facts — data paths, which venv for which
> task, gotchas that span repos — live in `/home/kim/Projects/SAO/MASTER.md`.
> Read it before cross-cutting work, and append to
> `/home/kim/Projects/SAO/WORKLOG.md` when you finish something another repo's
> agent would want to know.
>
> ⚠️ **`WORKLOG.md` and the `AGENT_DIALOGUE.md` cross-instance channel are PUBLIC** (the
> dialogue log auto-mirrors to a public URL for remote review). **Never write secrets** —
> passwords, API keys/tokens, SSH creds, `.netrc` contents, or credential-revealing paths —
> into either; keep secrets in the shell/env. (See MASTER §4.)
@/home/kim/Projects/SAO/MASTER.md

## Project Overview

MIR feature extraction pipeline for conditioning Stable Audio Tools. Extracts 97+ numeric features, 496 classification labels, and 5 AI text descriptions from audio. Processes full mixes and separated stems.

**Hardware:** AMD RX 9070 XT (RDNA4, 16GB VRAM) + Ryzen 9 9900X, ROCm 7.2, PyTorch ROCm nightly.

See **[TOOLS.md](TOOLS.md)** for the Pitch Shifter GUI and Feature Explorer / Latent Player.

**SA3 control-eval specs (cross-repo):** the control-response evaluator `avp_sa3/sa3_control/onset_eval.py` (gain×density grid → `onset_eval.json`) and the eval-site GUIs in `~/riffer-evals/` (`onset_eval.html` browses every `onset_eval.json`; `disentangle.html` = the onset_per_beat tempo-shortcut page; `mp.html`, `traj.html`) — **full spec in `SAO/MASTER.md` §4.** The eval *measurement* (BPM/onset/groove via essentia/librosa) runs in **mir's venv** (`mir/bin/python`).

## Commands

```bash
# Full pipeline (config-driven)
python src/master_pipeline.py --config config/master_pipeline.yaml

# Test all features on one file
python src/test_all_features.py "/path/to/audio.flac"
python src/test_all_features.py "/path/to/audio.flac" --skip-flamingo --skip-demucs

# Music Flamingo GGUF (fast, recommended)
python src/classification/music_flamingo.py "/path/to/audio.flac" --model Q6_K

# Music Flamingo Transformers (slower, native Python)
python src/classification/music_flamingo_transformers.py "/path/to/audio.flac" --flash-attention

# Stem separation
python src/preprocessing/demucs_sep_optimized.py /path/to/audio/ --batch
python src/preprocessing/bs_roformer_sep.py /path/to/audio/ --batch

# Audio captioning benchmark
python tests/poc_lmm_revise.py "/path/to/audio.flac" --genre "Goa Trance, Psytrance" -v
```

## ROCm Environment

**Central module:** `src/core/rocm_env.py` -- single source of truth for all GPU env vars.
**Config reference:** `config/master_pipeline.yaml` `rocm:` section.

Every GPU-using entry point must call `setup_rocm_env()` BEFORE `import torch`. Shell exports override defaults via `setdefault`.

Key settings:
- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` -- Triton FA2 for AMD
- `PYTORCH_TUNABLEOP_ENABLED=1`, `TUNING=0` -- use pre-tuned GEMM kernels
- `PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512` (PYTORCH_HIP_ALLOC_CONF deprecated in ROCm 7.2)
- `HIP_FORCE_DEV_KERNARG=1` -- prevent CPU/GPU desync
- `TORCH_COMPILE=0` -- buggy with Flash Attention on RDNA
- `MIOPEN_FIND_MODE` NOT set by default (causes freezes on some workloads)
- **Native CK Flash-Attention — 30–100% faster, use it on the SA3/SAT venvs.** On the torch-2.10/2.12
  ROCm venvs (`sat-venv`, `stable-audio-3/.venv`, `sa3-rocm7.13-test`) `flash_attn` is the **CK build**,
  and you MUST `export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE` (before `import torch`) to activate it —
  the `TRUE` above is the slower Triton-AMD path; without `FALSE` you get `No module named 'aiter'` → no
  FA. mir's own rocm-7.2 venv runs Triton FA2 (`TRUE`) and gains CK only after moving to the unified
  rocm-7.13 venv (`SAO/MASTER.md` §3). Full story: MASTER §5 + `SAO/docs/flash-attn-ck-rdna4.md`.

## Architecture

```
src/
  core/             # rocm_env, json_handler (safe_update), file_utils (read_audio), text_utils, batch_utils
  preprocessing/    # file_organizer, demucs_sep_optimized, bs_roformer_sep
  rhythm/           # beat_grid, bpm, onsets, syncopation, complexity, per_stem_rhythm
  spectral/         # spectral_features, multiband_rms
  harmonic/         # chroma, per_stem_harmonic
  timbral/          # audio_commons (8 features), audiobox_aesthetics (4 features), loudness
  classification/   # essentia_features, music_flamingo, music_flamingo_transformers
  transcription/    # drums/adtof.py, drums/drumsep.py
  tools/            # track_metadata_lookup, create_training_crops, statistical_analysis
  crops/            # pipeline.py, feature_extractor.py
tests/              # poc_lmm_revise.py (audio captioning benchmark)
config/             # master_pipeline.yaml (all settings)
models/             # GGUF models (Qwen3-14B, GPT-OSS-20B, Granite-tiny, Music Flamingo)
repos/              # External repos (timbral_models, llama.cpp, ADTOF-pytorch, Qwen2.5-Omni, etc.)
```

### Output Structure

```
Track Name/
  full_mix.flac           # Original audio
  drums.mp3               # Separated stems
  bass.mp3
  other.mp3
  vocals.mp3
  Track Name.INFO         # All features (JSON, append-only)
  Track Name.BEATS_GRID   # Beat timestamps
  Track Name.ONSETS       # Onset timestamps (required for syncopation/complexity)
  Track Name.DOWNBEATS    # Downbeat timestamps
```

### Key Subsystems

**Audio I/O:** `core.file_utils.read_audio()` handles all formats including m4a/AAC via pydub/ffmpeg fallback. Use this instead of `sf.read()` directly.

**Essentia Classification:** `classification/essentia_features.py`. EffNet embeddings via `effnet_onnx.py` (ONNX+MIGraphX). Genre/mood/instrument heads via `gmi_onnx.py` (ONNX+MIGraphX, JIT compiles ~28s on first run). TF `TensorflowPredict2D` is a fallback only. VGGish classifiers also on ONNX+MIGraphX via `vggish_onnx.py`.

**Music Flamingo:** `music_flamingo.py` (GGUF via llama-mtmd-cli, ~4s/track) or `music_flamingo_transformers.py` (native Python, ~28s/prompt). `music_flamingo_llama_cpp.py` is DEPRECATED. Prompt types configurable; `{metadata}` placeholder injects ID3 year/label/genres. Supports `flamingo_sample_probability` to annotate a fraction of crops per run. Unsupported audio formats (m4a, ogg) are auto-converted to WAV via ffmpeg before passing to llama-mtmd-cli. Output normalized via `core.text_utils.normalize_music_flamingo_text()`.

**Granite Revision:** `classification/granite_revision.py`. PASS 4b in `pipeline.py` runs Granite-tiny (llama-cpp-python) to condense Flamingo descriptions into short summaries. Runs independently of `--skip-flamingo` — set `flamingo_revision.enabled: false` in config to disable. Scans the entire crops directory each run so interrupted runs are caught up automatically. `reset()` called before every inference to avoid KV cache cascade failures.

**Metadata Lookup:** `tools/track_metadata_lookup.py` searches Spotify → MusicBrainz → Tidal. Candidates scored by duration match (0.5), year match (0.3), artist similarity (0.2). Tidal looked up via ISRC obtained from Spotify result. Saves: `release_year`, `artists`, `label`, `genres`, `popularity`, `album`, `spotify_id`, `musicbrainz_id`, `isrc`, `tidal_id`, `tidal_url`. Controlled by `metadata.use_spotify`, `metadata.use_musicbrainz`. Per-source skip logic: a track missing `spotify_id` is retried even if `musicbrainz_id` exists (handles Spotify rate limits). AcoustID fingerprinting prefers original file from `paths.input` over output `full_mix`. Tidal session cached as module-level singleton (`_TIDAL_UNAVAILABLE` sentinel prevents re-auth storms). Spotify `/v1/audio-features/` removed Nov 2024 — endpoint disabled. Spotify 429 rate-limit: `search_spotify()` re-raises HTTP 429; `_run_metadata_lookup()` catches this, disables Spotify for the rest of the session (`sp=None`), and logs a warning — affected tracks are retried on next run. Spotify client uses `retries=0` so 429s surface immediately. Verbose Spotify HTTP logging (available_markets) suppressed at WARNING level. Lookups run on **source track folders only** — never on crops. `_migrate_track_features_to_crops()` propagates all `TRANSFERRABLE_FEATURES` (including Tidal/ISRC) to crop INFOs.

**Captioning Benchmark:** `tests/poc_lmm_revise.py` -- 5-phase comparison (Flamingo baseline, genre-hint, LLM revision, Qwen2.5-Omni, ensemble). GGUF models in `models/LMM/`. Chat templates: Qwen3=ChatML, GPT-OSS=Harmony, Granite=start_of_role/end_of_role. llama-cpp-python `type_k`/`type_v` must be integers not strings.

**Onset Detection:** `rhythm/onsets.py`. Combined beat+onset worker `process_folder_rhythm()` in `pipeline_workers.py` handles both `.BEATS_GRID` and `.ONSETS` in one subprocess, loading audio once. Controlled by `rhythm.onsets: true` in config — runs as a catch-up pass independently of `track_analysis: false`. Required by `analyze_syncopation()` and `rhythmic_complexity`.

**Stage 2 Catch-up:** After any Stage 2 branch (skip-to-crop, already-complete, disabled, or never-run), the pipeline runs a unified catch-up: 2b onset detection (`rhythm.onsets`), 2c metadata lookup (`metadata.enabled`), 2d first-stage features. PASS 1 timbral timeout: uses `cf_wait(timeout=300)` instead of `as_completed()` — hung `timbral_models.timbral_reverb()` workers are abandoned after 5 minutes without blocking the pipeline.

**Statistical Analysis:** `tools/statistical_analysis.py`. Scans `.INFO` files recursively. Basic stats + outlier detection per feature. `--per-track` aggregates crops to one value per track. `--top N --key bpm` queries ranked values. Feature selection: `--vif`, `--pca`, `--cluster`, `--mi`, `--feature-select` (all). Plots (heatmap, dendrogram, scree, VIF bar) via `--plots-dir`.

## TimeseriesDB

Time-coded feature arrays (one value per analysis step across the crop duration) are stored in a SQLite database at `data/timeseries.db`, **not** in `.INFO` sidecar files. This keeps companion JSONs compact (~3 KB instead of ~130 KB).

**Fields stored in DB** (all `*_ts` numeric arrays):
- `rms_energy_{bass,body,mid,air}_ts` — per-band energy over time (shape `(n_steps,)`)
- `spectral_{flatness,flux,skewness,kurtosis}_ts`
- `beat_activations_ts`, `downbeat_activations_ts`, `onsets_activations_ts`
- `hpcp_ts` — chroma over time (shape `(n_steps, 12)`)
- `tonic_ts`, `tonic_strength_ts`
- Optional timbral `_ts` fields when `timeseries_timbral: true`

**API:**
```python
from core.timeseries_db import TimeseriesDB

db = TimeseriesDB.open()             # opens data/timeseries.db
arrays = db.get("Artist - Title_0")  # {field: np.ndarray} or None
db.has("Artist - Title_0")           # bool
db.count()                           # total entries
```

**Pipeline integration:** `pipeline.py` writes to TimeseriesDB instead of `results` dict when `skip_timeseries: false`. The `existing_keys` / sentinel pattern is bypassed — the DB `has()` check is the source of truth for resume logic.

**Encoding:** `encode_dataset.py` already strips all list fields from companion JSONs (the `padding_mask` exception is for SAT training). TimeseriesDB is for analysis tools and future conditioning only — the encoder does not read from it.

## Whole-Track Timeseries (per-track npz, for variable-offset crops)

The per-crop TimeseriesDB above is keyed by `<track>_<crop>` and only works when the crop-to-track mapping is fixed at MIR time. For workflows that need **variable training-crop offsets** (e.g. SA3 LoRA fine-tuning with beat-aligned crops chosen at encode time, or future LatCH training against arbitrary windows), a parallel **whole-track** store exists:

- **Producer:** `src/spectral/whole_track_timeseries.py`
  ```bash
  python src/spectral/whole_track_timeseries.py <Goa_Separated_root> --workers 4
  ```
  Walks each track folder, extracts 20 fields at **100 Hz over the whole track** (`madmom` beat/downbeat activations, `librosa` onset envelopes per stem, multiband/per-stem RMS, spectral, HPCP), writes one `<track>.TIMESERIES.npz` per source track. Resumable (skips existing). Driven by chunked-fresh-pool workers (see `--chunk-size`, `--chunk-timeout`).

- **Expected input layout — one folder per track, not a flat directory:**
  ```
  <track_dir>/full_mix.<ext>                     REQUIRED (flac/wav/mp3/ogg/m4a/aiff)
  <track_dir>/{drums,bass,other,vocals}.<ext>     OPTIONAL — separated stems, same folder
  ```
  `find_full_mix()`/`find_stem_files()` (both in `whole_track_timeseries.py`) look for exactly
  this — a flat directory of audio files (`--expanded` reports "Found 0 track folders" against
  one) or a per-track folder missing `full_mix.*` both fail silently/loudly depending on mode.
  To build this layout from a flat corpus, **hardlink** (`os.link`) `full_mix.<ext>` into a
  per-track folder rather than symlinking (mixes things up for some readers) or copying (wastes
  disk on a large corpus) — `master_pipeline.py`'s Stage 1 (Organization, legacy/pre-timeseries)
  established this same convention and Stage 2a (stem separation) already writes stems into that
  identical per-track folder, so a `master_pipeline.py`-organized corpus needs no reshaping.
  Stems are optional but drive real fields: the base extractor's per-stem
  `onset_envelope_{stem}_ts`/`rms_{stem}_ts` (8 fields, all 4 stems) and the expanded
  extractor's melody-height `f0_{other,bass}_ts` (2026-08-12, below) both come from stems, not
  the mix. A missing stem SKIPS its fields rather than faking them from the mix — check
  `field_rates`/`fields` in the sidecar `__meta__`, never assume a field is present.
  **Gotcha (2026-08-17): `--add-fields` only backfills the EXPANDED field set
  (`missing_expanded_fields()`, scoped to `whole_track_expanded.py`'s `EXPANDED_FIELDS`) — it
  does NOT re-run the base extractor. So if a track was first extracted with only `full_mix`
  present and stems land later, `--add-fields` will correctly add melody-height but will
  silently never add the base per-stem onset/RMS fields (they're not in its "missing" list at
  all, not even reported). If stems arrive after the initial pass, re-run the full `--expanded
  --overwrite` pass instead of `--add-fields`, or you end up with a permanent two-tier field
  set and no signal that it happened.** Also: the melody-height stem lookup
  (`whole_track_expanded.py`) only recognized `.flac/.mp3/.wav` until 2026-08-17, when `.m4a`
  was added — BS-RoFormer via `goa_sep_task.py` outputs `.m4a`, the same blind-spot class as an
  earlier goa `.mp3` fix documented in the same function (818/4461 goa folders were silently
  skipped forever before that fix; check the extension list before trusting a new stem source).

- **Expanded fields:** `src/spectral/whole_track_expanded.py` adds 26 model/DSP fields at their
  own **native rates** (0.2–100 Hz) — read `field_rates` from the sidecar meta, never assume 100 Hz.
  Incremental backfill: `whole_track_timeseries.py --add-fields` (recomputes only what is missing).
  **`field_rates` covers ONLY the 30 expanded fields (26 + the 4 melody) — the 20 base fields have
  no entry at all** (verified across all 5035 Lehto sidecars), so `field_rates[f]` on a base field
  is a `KeyError`: read it as `field_rates.get(f, meta["frame_rate"])`. Note also that
  `expanded_version` lives at `meta["expanded"]["expanded_version"]`, **not** top level.
  `chords_idx_ts` is a **class index** into the 24-triad `CHORD_VOCAB` (`-1` = unknown) — it must
  be mode-pooled, never mean-pooled (averaging C=3 and G=10 gives a different chord).
  **Per-track silent failure (2 goa sidecars, found 2026-08-18):** a madmom beat/downbeat
  activation failure is logged as a warning and the track is written anyway, **48 fields instead of
  50, with `beat_activation_ts`/`downbeat_activation_ts` simply absent** — nothing downstream
  announces it. Check the field count, not just the file's existence.
  **Stated-rate bug, fixed 2026-08-18:** `va_deam_ts`/`va_emomusic_ts` were stamped at
  `16000/(96*160)` = 1.041667 Hz — 96 is VGGish's patch *size*; essentia's default patch *hop* is
  93, so the true rate is 1.075269 Hz. 3.1% off, which sits under `crop_timeseries_resample`'s 5%
  warn threshold and was therefore accepted silently (~19 s of tail drift on a 600 s track). The
  producer now asks the algorithm for `patchHopSize`; **sidecars written earlier still carry the
  wrong stated rate** — repair with `src/tools/repair_timeseries_meta.py --fix-vggish-rate`
  (that tool also rebuilds a sidecar's `fields`/`field_rates`/`expanded` from the arrays actually
  present, dry-run by default; it exists because `avp_f0_augment_transform.py` wrote the four f0
  arrays into 1346 avp variants without ever announcing them in `__meta__`).

- **Melody height (4 fields, 2026-08-12) — the pitch/melody control-head target.**
  `f0_other_ts`, `f0_other_voiced_ts`, `f0_bass_ts`, `f0_bass_voiced_ts`, all 100 Hz,
  `PredominantPitchMelodia` + `EqualLoudness` on the **separated stems** (other 55–1760 Hz,
  bass 30–350 Hz). These are the only pitch fields that are **not octave-folded** — hpcp,
  chroma_linmap, bass_chroma_linmap and chords are all pitch *class*, in which a rising line
  and its inversion are identical, so "make the lead go up" is unexpressible from them.
  - **Unvoiced frames are `0.0` Hz. MASK with the `_voiced_ts` field; never regress on the raw
    values — 0 Hz is not a low note.** Voiced *fraction* is itself meaningful (a low value means
    that stem has little predominant melody, not that tracking failed).
  - **RESAMPLING: never mean-pool f0 in Hz.** (`crop_timeseries_resample.py` already does the
    right thing here — this is why to use it.) The older consumer
    (`stable-audio-tools/scripts/whole_track_target_source.py::resample_axis0`) downsamples by
    fractional-bin mean pooling — correct for density/energy envelopes, wrong here, because it
    averages real pitches with the 0.0 sentinel and drags each window toward silence by its
    unvoiced fraction. Measured over 120 tracks at the SA3 grid (100 → 10.767 Hz, ~9.3 source
    frames per target frame): **median error +0.00 st but p95 +15.86 st, with 17.2% of frames
    wrong by more than a semitone** — sparse, severe, and concentrated at note boundaries where
    the melody actually is. The median being zero is why a spot-check passes it. Pool the voiced
    frames only:
    ```python
    num = resample_axis0(f0 * mask, n)
    den = resample_axis0(mask, n)
    f0_ds = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)   # and keep den as the weight
    ```
    (Or convert to semitones first and pool there.) The same caution applies to **any** future
    field with a sentinel value — `resample_axis0` cannot know that `0.0` means "absent".
  - Two voices because a rolling bassline is a melodic voice in its own right. Any mapping onto
    SA3's 3-band chroma conditioning (bass→low, other→mid+high) belongs at the **conditioning**
    stage, not in extraction.
  - Skipped, not faked from the mix, where a stem is missing (4 goa folders).
  - **Open:** `f0_bass_ts` may sit one octave above the true fundamental — melodia and YIN
    disagree by ~12 semitones on every bass stem tested and three tests failed to settle it.
    Melodia was chosen for contour stability (YIN flips octaves *within* a track), not because
    its octave is known right. Re-open if absolute bass register ever matters.

- **Output:** `/run/media/kim/Lehto/timeseries/<track>.TIMESERIES.npz` — **5035 npz, ~37 GiB**
  (audited 2026-08-18): **4461 goa + 574 genre-corpus sidecars, and ZERO avp** — **do not glob the
  directory as a goa denominator**. The avp sidecars are NOT here; they live in place under
  `<UUID drive>/avp-analyzed/<track>/` (and `.../augmentations/<variant>/` for the 1346 augmented
  variants). A third, separate store — undocumented until now — is
  `<UUID drive>/suomisoundi_timeseries/` (1260 npz, 6.7 GB).
  **"50 fields" describes 4455 of those sidecars, not the store:** 574 have **46** (melody was
  never run on the genre corpora), 4 have **38** (no stems → the 8 per-stem + 4 melody fields are
  skipped), and 2 have **48** (the madmom activation failure noted above). Read `fields` from the
  sidecar; do not assume the full set.
  Each npz contains:
  - 1-D fields shape `(N_frames,)` where `N_frames ≈ duration_sec × 100`
  - `hpcp_ts` shape `(N_frames, 12)`
  - `__meta__` JSON string with `frame_rate`, `n_frames`, `duration`, `fields`, `field_rates`, etc.

- **Consumer (cropper/resampler) — use `src/tools/crop_timeseries_resample.py`:**
  ```python
  from crop_timeseries_resample import build_crop_timeseries
  out = build_crop_timeseries(arrays, meta, start_sec, end_sec, n_frames)
  ```
  It lives in mir because how a field may legally be downsampled is a property of the
  measurement, not of the consumer. It derives each field's rate from `n_frames / duration`
  rather than trusting `field_rates` (one stored rate — `maest_embed_ts` — is wrong by exactly
  2×), masked-mean-pools the sentinel fields (`f0_*_ts`, where `0.0` means unvoiced), mode-pools
  the categorical ones (`chords_idx_ts`), and fails the crop loudly on partial coverage.
  The older `SAO/stable-audio-tools/scripts/whole_track_target_source.py` is **superseded for
  expanded fields**: its `get()` applies the single top-level `meta["frame_rate"]` (100 Hz) to
  *every* field, which is right for the 20 base fields and wrong for the 26 expanded ones at
  0.2–100 Hz — a coarse field then gets sliced at the wrong offset, silently, or comes back
  empty and the crop is dropped. It is still fine for base-field LatCH dataloading
  (`WholeTrackTargetSource.get(crop_key, feature, start_time, end_time, n_frames)`).

- **Use cases**: LatCH-head training against arbitrary crop windows (the SAT trainer reads via the consumer above); per-crop timeseries companions for SA3 LoRA latents (sliced to T=4096 alongside each `.npy`, see `/tmp/sa3_encode_from_manifest.py`).

When to choose which store:
- **TimeseriesDB (per crop)**: fixed crop-to-track mapping, queried by crop key — used by the legacy LatCH dataset.
- **Whole-track npz (per track)**: arbitrary windows extracted at consumer time — used by anything that needs to slice a specific `[start_sec, end_sec]` and resample to a target frame count.

## Development Rules

### JSON handling

**Always** use `safe_update()` for atomic merge. **Never** use `write_info(merge=False)`.

```python
from core.json_handler import safe_update, get_info_path
safe_update(get_info_path(audio_path), {'feature': value})
```

### New features

1. Add to `FEATURE_RANGES` in `src/core/common.py`
2. Use `safe_update()` to save
3. Test with `test_all_features.py`

### ROCm env in new scripts

Any script that imports torch must start with:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # adjust to reach src/
from core.rocm_env import setup_rocm_env
setup_rocm_env()
import torch  # now safe
```

### Other rules

- Load GPU models once for batch processing, not per-file
- Never move original files -- copy to output dir
- Never delete `.INFO` files -- only merge
- Use `FileLock` for concurrent batch processing

## Known Issues

- **INT8/INT4 quantization:** Non-functional on ROCm. Use bfloat16 + Flash Attention 2.
- **torch.compile:** Fails with Demucs (complex FFT), Music Flamingo (Dynamo+accelerate), and FA on RDNA. Keep `TORCH_COMPILE=0`.
- **GGUF POOL_1D warning:** Cosmetic on RDNA4 (gfx1201).
- **numba/numpy:** Pin numpy <2.4.
- **Qwen2.5-Omni AWQ:** Requires patched modeling file (RoPE fix) in `repos/Qwen2.5-Omni/low-VRAM-mode/`.
- **Madmom:** CPU-only (NumPy/Cython neural networks, no GPU support).
- **Audiobox Aesthetics ONNX:** Not exportable — WavLM attention uses non-contiguous tensor views that break both dynamo and legacy tracer. Runs as PyTorch ROCm.
- **GMI ONNX first-run JIT:** Genre model takes ~29s to JIT compile kernels on first inference per process. Mood/instrument ~0.4-0.6s. Subsequent calls are <1ms.
- **Spotify audio features:** `/v1/audio-features/` returns 403 for all standard API apps since Nov 2024. Disabled in pipeline (`fetch_audio_features_flag=False`). The endpoint is gone permanently.
- **timbral_models hang:** `timbral_reverb()` can loop forever on pathological audio. PASS 1 uses `cf_wait(timeout=300)` — hung crops are skipped and retried next run.
- **`mir/bin/python` is an ffmpeg9-SONAME compat wrapper, not a plain symlink (2026-08-16).** System ffmpeg was upgraded 8.1.2→9.0.1 (pacman) on 2026-08-16, which removed `libavdevice.so.62`/`libavcodec.so.62`/etc from `/usr/lib`. `torchaudio`'s audio-load chain delegates to **torchcodec** (`torchcodec-0.10.0a0`, installed here), whose bundled `libtorchcodec_core8.so` `DT_NEEDS` those exact SONAMEs — the crash zeroed out **every** Audiobox CE score (0/162 on a test batch, no partial results; `audiobox_aesthetics.py` never imports torchcodec directly, the break is transitive through the librosa/torchaudio load path). Investigated and ruled out: no ffmpeg8-compat package exists (official repos or AUR); rebuilding torchcodec against ffmpeg9 needs torchcodec≥0.16, which needs **torch≥2.11** (mir has 2.9.1 — a much bigger, separate call than this fix warrants); a blanket ffmpeg downgrade + `IgnorePkg` would hold the *system-wide* package back indefinitely (silent security-patch rot) just for this one venv. Fix: extracted the exact ffmpeg8 SONAMEs (avcodec/avdevice/avformat/avfilter/avutil/swscale/swresample + the matching `libx265.so.216`) from the still-cached `/var/cache/pacman/pkg/ffmpeg-2:8.1.2-10-*.pkg.tar.zst` into `mir/lib/ffmpeg8-compat/` (gitignored, venv-internal — regenerate via `bsdtar -xf` on that cached package if it's ever gone; check `archive.archlinux.org/packages/f/ffmpeg/` as a fallback source). `mir/bin/python` was a symlink to the shared uv-managed interpreter; it's now a wrapper script that prepends `LD_LIBRARY_PATH` and `exec -a`'s the real interpreter (**`exec -a` is required** — a plain `exec` changes `argv[0]` to the real interpreter's own path, which breaks CPython's venv detection and silently drops the whole venv, torch/numpy included; caught live building the first version of this wrapper). Setting `os.environ['LD_LIBRARY_PATH']` from *inside* an already-running Python does **not** work here — verified live — glibc resolves a loaded `.so`'s `DT_NEEDED` entries against the `LD_LIBRARY_PATH` snapshot taken at process start, not re-read per `dlopen()`; the fix has to happen in the launcher, before the interpreter starts. `python3`/`python3.12` are unaffected (they're relative symlinks to `python`, so they inherit the wrapper transparently). Verified end-to-end: `torchaudio.load()` on a real clip, then the real `control/sa3_control/clip_metrics_audiobox.py --pattern base__base` scoring 30/30 clips with real non-null CE/PQ values in `clip_metrics.db`.
