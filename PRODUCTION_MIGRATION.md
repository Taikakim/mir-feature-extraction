# Production-Box Migration: Pending Tests & Findings

*Companion to `SAME_CHROMA_FINDINGS.md`. Everything here needs the production
desktop (ROCm GPU, venvs, audio dataset, SAME/SA3 checkpoints) — none of it can
run on the dev laptop. Ordered by dependency: each gate decides what comes next.*

---

## Checkpoint selection (release provides three families)

| Key | Family | Use here for |
|---|---|---|
| small-music / small-sfx / medium | Post-trained (APT, few-step) | final generation only |
| small-music-base / small-sfx-base / medium-base | Base (pre-APT flow matching) | **steering experiments, ZeroSep inversion, LoRA training** |
| same-s / same-l | SAME autoencoder standalone | **readout refit (encode-only, no DiT)** |

Rules: run the steering gate (#2) and ZeroSep (#5) on **-base** checkpoints —
clean continuous velocity field (faithful inversion, room for selective TFG's
first-20%-of-steps window, which on a 4-8-step APT model is ~1 step). Only
after steering is proven, test whether it survives on the post-trained
checkpoint.

RESOLVED from the local stable-audio-3 repo (docs/guides/model-overview.md,
docs/workflows/inference.md, docs/workflows/autoencoder.md):
- Encoder pairing: small-music/small-sfx -> **SAME-S**; medium (and large) ->
  **SAME-L**. Refit readouts on same-l latents to steer medium-base.
- cfg_scale: free knob **on -base checkpoints only** ("no effect on
  post-trained checkpoints"); default is 1.0 — ZeroSep's exact omega.
  negative_prompt likewise base-only. All control experiments => -base.
- LoRA: trained on -base, "can be applied to the post-trained model and will
  work as expected" — a trained chroma conditioner deploys on the fast model.
- `model.generate(init_audio=..., init_noise_level=...)` is built-in
  audio-to-audio (SDEdit-style noise-then-denoise), and inpainting supports
  multi-region masks. scripts/pre_encode_dataset.py pre-encodes a dataset to
  .npy latents — the refit data prep is a stock script.

**LATENT RATE — RESOLVED from the HF checkpoint configs (ALL varieties):**
`downsampling_ratio: 4096` (patch_size 256 x strides [16]) verified in
medium, medium-base, AND small-music-base (i.e. both SAME-L and SAME-S) —
the papers were right; the repo doc's "216 frames per 10 s" is a
documentation bug for every model. 10 s -> 108 frames at 10.766 Hz; chroma
hop 4096 is natively 1:1 with latents; `same_chroma.n_latent_frames` =
ceil(T/4096) is CORRECT as implemented for any checkpoint. LATCH SAO-format
bridge: a 524288-sample crop = 128 SAME frames vs 256 SAO frames — a clean
2x `interpolate_linear(c, 256)`. Note for TADA localization: small DiT is
depth 20 / embed 1024 (medium: 24 / 1536) — bottleneck indices will not
transfer between sizes.

**More facts from model_config.json (stable-audio-3-medium):**
- `cfg_dropout_prob: 0.1` — the unconditional branch was trained; the
  cfg_scale/omega knob is real in the released weights.
- Conditioning topology: prompt (T5Gemma) + seconds_total via CROSS-ATTENTION;
  seconds_total also global adaLN; inpainting via **local_add_cond**
  (frame-aligned additive, dim 257 = 256 latent + 1 mask). For the LoRA
  conditioner path: time-aligned chroma fits the local_add_cond pattern
  exactly (project 3x128 -> add channels) — reuse the mechanism the model
  already knows for time-aligned signals, do not invent a new one.
- DiT: depth 24 (same as SAO, where TADA's bottleneck was {12,13}), embed
  1536, differential attention, 64 memory tokens. ARC discriminator taps DiT
  hidden layer 18 — Stability's own pointer to semantically rich depth.
- Objectives differ by checkpoint (both configs inspected): **-base =
  "rectified_flow"** (pure RF velocity — the target for inversion/steering
  math; predicted-clean is z0|t = z_t - t*v), post-trained = "rf_denoiser".
- Base demo settings confirm the regime: demo_steps 50 + demo_cfg_scales
  [2,4,7] (vs 8 steps + cfg [1] post-trained) — many-step sampling and a
  live, working unconditional branch on -base. Selective TFG's first-20%
  window = ~10 steps of room at 50 steps.
- Conditioning topology is IDENTICAL across base/post-trained (why LoRA
  transfer works; the local_add_cond chroma template applies to both).
- Encoder mask_noise 0.001 at inference (decoder 0.1), softnorm bottleneck
  with noise_regularize — tiny latent noise floor; the refit readout sees it.
- sample_size 2^24 = ~380 s max; latent length capped at 4096 frames.

## 0. Environment sanity (5 min)

- [ ] `python src/harmonic/same_chroma.py --selftest` — should pass 9/9 (numpy/scipy only).
- [ ] Cross-check torch path vs NumPy path (torch unavailable on laptop, never run):
      random 10 s stereo signal -> `compute_same_chroma` vs `compute_same_chroma_torch`
      (CPU and `cuda`/ROCm) -> max rel err should be ~1e-5 or better.
- [ ] Verify `n_latent_frames` = ceil(T/4096) against the *real* SAME encoder output
      length for several durations (edge/padding could differ by one frame).
      If it differs, fix `n_latent_frames()` and re-run selftest.

## 1. Readout refit (the LatCH probe)

Goal: reconstruct the (lost) linear chroma readout heads from public SAME latents.
Offline information confirms this is the intended route (the correlations
are re-extractable from the latents, even for a different setup).

- [ ] Encode a few hours of varied audio (use the crop dataset) with the public
      SAME encoder -> latents (B, 256, T) at ~10.766 Hz.
- [ ] Targets: `compute_same_chroma(audio, 44100)` -> (3, 128, T), natively
      frame-aligned (hop 4096 = latent hop).
- [ ] Per band: least-squares fit `latent -> 128 bins` WITH bias (the original
      heads were Conv1d k=1 bias=True). Minutes on CPU; no training loop needed.
- [ ] GATE: held-out R^2 / correlation per band. A rough correlation is
      expected — bass band likely strongest. If near-zero, check frame
      alignment first (off-by-one), not the method.

## 2. The steering experiment (decides the whole architecture)

Question (untested by anyone, incl. the authors): does pushing the latent along
the readout direction audibly move generated pitch — or only move the meter?

- [ ] Implement guidance: at each sampling step, chroma_hat = refit_head(z);
      score = cosine(chroma_hat, target) per frame; nudge z by grad ascent
      scaled per-band. Targets from `make_steering_target()`.
- [ ] Test 1 — "only C": all-zero melody weights except C, bass_root=0,
      high gains. Listen: do pitches collapse toward C?
- [ ] Test 2 — "goa preset": bass_root=E (profile 'root5'), melody = E phrygian,
      air_gain 0.3. Listen: static E bass, melody constrained to scale?
- [ ] Try coarse-to-fine: Gaussian-blur the target (time sigma 2-4 frames,
      chroma sigma 2-3 bins) at high-noise steps, sharpen toward the end.

**LatCH paper recipe (arXiv 2603.04366 — Novack + the SAME/SA3 team; this IS
the published LatCH method, and the fork already has its scripts per
LATCH_CONTROL.md):**
- Their working TFG hyperparameters on SAO: rho = mu = 0.03, gamma = 0.3,
  **selective guidance on only the FIRST ~20% of sampling steps**, N_iter = 4,
  N_recur = 1, 100-step DDIM, CFG 7, variance-preserving form. Start there.
- Apply the clean-latent refit head via MEAN guidance on the predicted-clean
  z0|t (their Readouts baseline lost precisely because it cannot do mean
  guidance; heads on z0|t can).
- CAUTION, and why we may still win: they tried 12-bin chroma + CREPE pitch
  guidance on SAO with "mixed results" — but (a) SAO's latent was never
  chroma-regularized, SAME's is (linear decodability was a training
  objective); (b) their failure mode was rapid note-contours with sparse
  one-hot targets, ours is near-static soft palette targets (their
  "gradual/low-frequency controls work well" regime); (c) their own
  hypothesis favors dense/smooth targets — SAME's 3x128 soft chroma is that.
- Upgrade path if clean-head guidance underwhelms: small noise-conditioned
  head trained LatCH-B-style (on generated-trajectory intermediates — their
  best variant), NOT a paradigm change.

- [ ] GATE: pitches move cleanly -> build the GUI on guidance alone (no
      training). Meter moves but audio doesn't / artifacts dominate ->
      Plan B (below) + trained conditioner path.

## 3. TADA additions (arXiv 2602.11910 — assessed 2026-06-11)

Paper: localized activation steering via a 2-layer cross-attention "semantic
bottleneck" ({12,13} in Stable Audio Open, {7,8} in Ace-Step) beats prompt/
weight/score-space steering. IMPORTANT: their score-space baseline
(FreeSliders, contrastive-prompt noise differencing) is NOT our readout-gradient
guidance — Plan A is untested by them, and their models lack SA3's
chroma-regularized latent. Their vectors are text-derived and time-averaged:
no pitch concepts, no temporal control. Three things transfer:

- [ ] **Localization on SA3's DiT** (do this BEFORE any LoRA work): replicate
      their K/V activation-patching protocol (their Eq. 2: patch one layer's
      cross-attn K/V from a concept run into a counterfactual run, score concept
      presence) to find SA3's functional layers. Their ablation shows steering
      outside the bottleneck does ~nothing -> attach LATCH/LoRA conditioning AT
      the bottleneck, not everywhere. Note SA3 also conditions via AdaLN +
      memory embeddings; adapt sites accordingly. Layer indices will NOT match
      SAO's {12,13}.
- [ ] **Evaluation protocol** for experiment #2: alignment-preservation AUC.
      Preservation = LPAPS distance from unsteered baseline; alignment =
      chroma-similarity of the OUTPUT audio (via compute_same_chroma) to the
      steering target; sweep guidance gain; report AUC + Audiobox Aesthetics.
      Turns "listen to it" into a curve.
- [ ] **Plan B if gate #2 fails**: harvest cross-attn activations at SA3's
      functional layers from generations/inpainting runs over known-key audio,
      build per-pitch-class steering vectors (CAA mean-difference or SAE),
      steer there. Per-frame injection for temporal control is unexplored in
      the paper — would be novel work.

## 4. Prototype palette (dataset-derived targets)

- [ ] Batch `compute_same_chroma` over the crop dataset; store per-crop
      time-averaged (3, 128) profiles + tonic (already in .INFO via hpcp_tiv).
- [ ] Root-normalize: fractional circular roll by tonic x 128/12 bins
      (np.interp — a semitone is 10.667 bins, NOT integer).
- [ ] L2-normalize per band; spherical k-means, bands clustered SEPARATELY
      (bass flavors x melody colors); k~10 or farthest-point medoids for
      max distinctness -> prototypes.npy.
- [ ] UI: prototype selector + root selector (de-rotation at selection time =
      the fundamental conditioner). Manual 12-weight EQ stays as advanced mode.
- [ ] Also: measure one genre-typical bass patch per root -> empirical
      band-0 template (replaces theoretical BASS_PROFILES weights).

## 5. ZeroSep on SA3: open-set text-queried separation (optional)

ZeroSep (NeurIPS 2025, OpenReview IIjiNTR1cV): zero-training separation =
invert the mixture's latent to noise, re-denoise with the TARGET source's
text description at CFG **omega = 1** (drops the unconditional term; omega=0
reconstructs the mixture, omega>1 generates new content). Validated on
mel-VAE backbones (AudioLDM-class); their "does not apply to SAO" footnote
scopes the mel+vocoder pipeline, not the method. Adaptations for SA3:

- [ ] Inversion: SA3 is flow matching -> reverse-ODE inversion is natural.
      RESOLVED RISK: pre-APT flow-matching weights ARE released (the -base
      checkpoints) — use those; DDPM-style (edit-friendly) inversion remains
      the more robust default mode regardless.
- [ ] Verify cfg_scale is a free knob (near-certain on -base checkpoints;
      still check the post-trained ones if ever used here). ZeroSep needs
      omega = 1 exactly.
- [ ] Honest unknown: ZeroSep's evidence is from near-linear mel-VAE latents;
      SAME's 4096x semantic latent does not superpose sources linearly.
      Untested either way — run "the bassline" / "drums only" on a few known
      crops and listen before investing further.
- [ ] **ZeroSep-lite, ZERO porting (run this FIRST):** SA3's built-in
      audio-to-audio is the SDEdit approximation of the same idea:
      `model.generate(init_audio=mixture, init_noise_level~0.5-0.7,
      prompt="the bassline", cfg_scale=1)` on medium-base. Not exact
      inversion (loses some mixture fidelity), but answers the
      semantic-latent separability question in 10 minutes with the stock
      repo. Only if promising, build the full DDPM-inversion wrapper.
- Use cases if it works: open-set stem queries beyond Demucs's taxonomy
  ("the acid line", "the 909 hats") for dataset curation; isolating basslines
  to build empirical band-0 chroma templates; and the inversion plumbing
  doubles for Plan-B activation harvesting (TADA) and audio-to-audio editing.

**Port plan (repo reviewed locally at py/ZeroSep, 2026-06-11):** the codebase
is cleanly abstracted — all separation logic (code/ddm_inversion/, ~21 KB) is
model-agnostic behind a PipelineWrapper base class; existing wrappers are
Tango/AudioLDM/AudioLDM2 (code/models.py). Port = one StableAudio3Wrapper:
- Trivial: vae_encode/decode (SAME), encode_text (T5 + SA3's required timing
  conditioning via the setup_extra_inputs hook), unet_forward + CFG.
- Small surgery: the interface assumes mel-STFT front-end (get_fn_STFT,
  decode_to_mel, load_audio stft=True) — SA3 is waveform-native; identity
  STFT + branch load_audio + drop vocoder path.
- The real work: six DDPM-parameterized scheduler methods
  (sample_xts_from_x0, get_zs_from_xts, reverse_step_with_custom_noise,
  get_variance, get_alpha_prod_t_prev, get_sigma) need rectified-flow
  equivalents (alpha_t = 1-t, sigma_t = t; RF-Inversion / FlowEdit math).
- Code facts: omega=1 is the shipped default for BOTH guidance scales;
  DDPM-inversion is the default mode (the APT-friendly one); tstart < steps
  gives partial inversion (fidelity/separation dial); 50-step default.
- Estimate: 1-2 focused days; run the two go/no-go checks (cfg knob, a quick
  semantic-latent separation listen) in the first hour before writing the
  full wrapper.

## 6. Open / unresolved

- The Gaussian-blur-for-pitch-steering paper is still unidentified (it is NOT
  TADA). Mechanism is sound regardless (gradient basin, over-constraint relief,
  coarse-to-fine) — find the citation for the exact blur schedule, or just
  sweep sigma in experiment #2.
- SAO-era LatCH rate mismatch: CONFIRMED real (SAME 10.766 Hz vs SAO/LATCH
  21.533 Hz) but trivial — exactly 2x; bridge with `interpolate_linear(c, 256)`
  per 524288-sample crop.
- Whether the SAME release includes the trained semantic-regressor weights
  (probably not, per offline information); check the same-s / same-l
  checkpoint keys once, before refitting from scratch.
- ~~Which autoencoder pairs with which diffusion checkpoint~~ RESOLVED:
  small -> SAME-S, medium/large -> SAME-L (model-overview.md).

## References

- `SAME_CHROMA_FINDINGS.md` — full decoded recipe + verification log.
- SAME: arXiv 2605.18613. SA3 report: arXiv 2605.17991.
- TADA: arXiv 2602.11910 (v2, May 2026).
- LatCH: arXiv 2603.04366 — "Low-Resource Guidance for Controllable Latent
  Audio Diffusion" (Novack et al., Stability AI; selective TFG + LatCH heads).
- ZeroSep: OpenReview IIjiNTR1cV (NeurIPS 2025).
- stable-audio-tools @ 3241adb: training/autoencoders.py:567-597.
