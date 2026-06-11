# Production-Box Migration: Pending Tests & Findings

*Companion to `SAME_CHROMA_FINDINGS.md`. Everything here needs the production
desktop (ROCm GPU, venvs, audio dataset, SAME/SA3 checkpoints) — none of it can
run on the dev laptop. Ordered by dependency: each gate decides what comes next.*

---

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
Zach Evans confirmed this is the intended route ("the latents have the
correlations, you can pull them out again").

- [ ] Encode a few hours of varied audio (use the crop dataset) with the public
      SAME encoder -> latents (B, 256, T) at ~10.766 Hz.
- [ ] Targets: `compute_same_chroma(audio, 44100)` -> (3, 128, T), natively
      frame-aligned (hop 4096 = latent hop).
- [ ] Per band: least-squares fit `latent -> 128 bins` WITH bias (the original
      heads were Conv1d k=1 bias=True). Minutes on CPU; no training loop needed.
- [ ] GATE: held-out R^2 / correlation per band. Rough is expected (Zach: "rough
      correlation") — bass band likely strongest. If near-zero, check frame
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

- [ ] Inversion: SA3 is flow matching -> reverse-ODE inversion is natural,
      BUT released checkpoints are adversarially post-trained few-step models
      (distilled velocity fields invert poorly). Check the repo for pre-APT
      flow-matching weights; otherwise use DDPM-style (edit-friendly)
      inversion — per-step noise recording makes reconstruction exact by
      construction even on imperfect models.
- [ ] Verify cfg_scale is a free knob in the released inference code (APT
      models sometimes bake guidance in); ZeroSep needs omega = 1 exactly.
- [ ] Honest unknown: ZeroSep's evidence is from near-linear mel-VAE latents;
      SAME's 4096x semantic latent does not superpose sources linearly.
      Untested either way — run "the bassline" / "drums only" on a few known
      crops and listen before investing further.
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
- SAO-era LatCH rate mismatch: existing LATCH_CONTROL.md pipeline expects
  21.533 Hz / 256 frames per crop (SAO, 2048x); SAME/SA3 is 10.766 Hz / 128
  frames (4096x). Decide one rate per pipeline; `interpolate_linear(c, 256)`
  bridges if needed.
- Whether the SAME release includes the trained semantic-regressor weights
  (probably not — Zach: "not sure we still have those"); check the checkpoint
  keys once, before refitting from scratch.

## References

- `SAME_CHROMA_FINDINGS.md` — full decoded recipe + verification log.
- SAME: arXiv 2605.18613. SA3 report: arXiv 2605.17991.
- TADA: arXiv 2602.11910 (v2, May 2026).
- LatCH: arXiv 2603.04366 — "Low-Resource Guidance for Controllable Latent
  Audio Diffusion" (Novack et al., Stability AI; selective TFG + LatCH heads).
- ZeroSep: OpenReview IIjiNTR1cV (NeurIPS 2025).
- stable-audio-tools @ 3241adb: training/autoencoders.py:567-597.
