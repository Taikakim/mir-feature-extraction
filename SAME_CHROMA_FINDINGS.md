# SAME / Stable Audio 3 Chroma Conditioning — Findings Log

*Last updated: 2026-06-10. Status: extraction implemented & verified; steering experiment pending (production box).*

Goal: a 12-semitone-weight GUI conditioner ("pitch-class EQ" + bass lock) for the
Stable Audio 3 fork, steering generation through the chroma structure that the
SAME autoencoder bakes into its latents — via a refit linear readout (LatCH-style),
ideally with no base-model finetuning.

---

## 1. The exact SAME chroma recipe (decoded from code)

Source: `stable-audio-tools/training/autoencoders.py:567-597` and
`models/transforms.py`, commit `3241adba4fc2a85cf5b29d9eb68d42f40a28e820`
(pointer provided by Zach Evans). Paper: SAME, arXiv 2605.18613 §3.3.2.

```
TightSpectrogram(n_fft=8192, normalized=True, power=1.0)   # hop FORCED to 4096
  -> per-channel |STFT|, mean over channels                # NOT mono-mix-first
  -> log1p
  -> torchaudio.prototype.transforms.ChromaScale(
       sample_rate=44100, n_chroma=128, n_freqs=4097,
       ctroct=center, octwidth=width, norm=1, base_c=True)
  for (center, width) in [(1.0, 1.0), (5.0, 1.5), (9.0, 1.0)]
  -> F.interpolate(target, size=n_latent_frames, mode='linear')
```

* Readout heads: `Conv1d(latent_dim=256 -> 128, kernel_size=1, bias=True)` per band
  (affine, not pure linear). L1 loss, per-band weights **[0.035, 0.05, 0.2]**
  (treble weighted ~6x bass). 10k-step warmup on `latents.detach()` before
  gradients shape the encoder.
* ILD head: 32-band mel L/R log-magnitude difference, weight x0.1, stereo only.
* Window: the "tight power-sine" window at hop = N/2 reduces to exactly
  `sin(pi*n/N)` (amplitude 1). `demodulate` only flips complex signs — irrelevant
  after `.abs()`.
* Band centers in Hz (octave ref A0=27.5): octave 1/5/9 = **55 / 880 / 14080 Hz**
  = bass / mid / brilliance, register-resolved chroma.

## 2. Critical subtleties (each would have silently broken compatibility)

1. **C is at bin 2.0, not bin 0.** `base_c=True` rolls by `-3*(n_chroma//12)` =
   -30 bins, but the exact A->C offset at n_chroma=128 is 32 bins. Each semitone
   spans 128/12 ≈ 10.667 bins; pitch class p sits at bin
   `(128*(p+3)/12 - 30) mod 128` (C=2.0, A=98.0). Never assume bin 0 = C.
2. **No per-frame normalization.** `norm=1` only L1-normalizes filterbank rows.
   Targets are unbounded log1p-energy values that scale with signal level
   -> steering objectives should be *relative* (dot products / cosine), not
   absolute target matching.
3. **`torch.stft(normalized=True)` = division by sqrt(n_fft)** ("frame_length"
   mode; ATen `fft_norm_mode::by_root_n`). The sqrt(sum(win^2)) "window" variant
   — exactly sqrt(2) different here — belongs to `torchaudio.transforms.Spectrogram`,
   which SAME does NOT use. Verified to the ATen source.
4. **Hermitian isometry:** one-sided bins scaled by sqrt(2) except DC and Nyquist.
5. **Frame alignment:** STFT hop 4096 = SAME latent hop (patch 256 x TRB stride 16)
   -> chroma frames are natively 1:1 with latent frames (~10.766 Hz at 44.1k);
   training interpolates the leftover off-by-one (center=False trims one frame).
6. **`torchaudio.prototype` is deprecated** -> the filterbank is vendored
   (NumPy port, verified bit-exact vs a literal transcription in float64).
7. **Interpolation coords in float32** to match ATen's accumulation for float32
   tensors (float64 coords drift from training behavior on long sequences).

## 3. Author confirmations (Zach Evans / "Fauno15", Discord, June 2026)

* The original trained readout heads are likely **not kept** ("not sure we still
  have those").
* The heads were a **semantic regularizer** ("rough correlation with chroma in
  the latents, to increase their semantic content"), *not* built for control.
* The correlations are **re-extractable**: "the latents have the correlations,
  you can pull them out again, even for a different setup" — i.e. refitting a
  linear probe on public-encoder latents is endorsed ("basically a LatCH").
* **Open question (the make-or-break):** whether *steering* the latent along the
  readout direction audibly moves generated pitch content, or only moves the
  meter. Untested by the authors. Test: refit heads, guide toward "only C"
  during sampling, listen.

## 4. Paper findings

* **SAME** (arXiv 2605.18613): §3.3.2 specifies the three regressors (above).
  No rationale is stated for *three* bands or centers 1/5/9 — in contrast, the
  neighbouring patched-waveform discriminators DO get a reason (prime sizes
  "to avoid harmonic aliasing"). The same 1/5/9 banding recurs in the transformer
  chroma discriminators (§3.2.2) and a 48-bin chromagram in the conv
  discriminator (§3.2.1). Ablation Table 2 (C->D): adding Lsem+Lcon gives the
  best generation scores (MuQEval 3.340 -> 3.870, FAD-CLAP 0.593 -> 0.576) at a
  small reconstruction cost — but never isolates chroma from ILD/contrastive,
  and never ablates band count.
* **SA3 technical report** (arXiv 2605.17991): three diffusion models
  (small 459M / medium 1.4B / large 2.7B), small+medium open-weights; Dt=4096,
  d=256; flow-matching -> ODE warmup distillation -> adversarial post-training.
* **Why three bands (our reconstruction, not paper-stated):** a folded chromagram
  is octave-invariant (bass C == treble C), so three octave-windowed bands
  restore a coarse bass/mid/brilliance register axis; the mid band is widened
  (1.5 oct) over the densest musical region; n_fft=8192 is forced by bass
  resolution (semitone step ~3 Hz near A1 vs 5.4 Hz bins); 128 bins/octave buys
  sub-semitone (tuning/vibrato) resolution and a smooth L1 target; the linear
  (1x1-conv) heads force chroma to be *linearly decodable* from the latent.

## 5. Implementation: `src/harmonic/same_chroma.py`

Self-contained (numpy+scipy core; torch optional). Adversarially verified
against upstream sources (5/5 aspects correct; filterbank bit-exact in float64
across 7 parameterizations incl. all three SAME bands). Self-test `--selftest`
passes 9/9 with synthetic signals only.

* `compute_same_chroma(audio, sr)` -> `(3, 128, T)` float32 at the SAME latent
  rate. ~300x realtime single-process on a laptop CPU.
* `compute_same_chroma_torch(...)` -> batched `(B, C, T)` GPU path via
  `torch.stft` (cuFFT/rocFFT; ROCm-ready). **TODO: numerical cross-check vs the
  NumPy path on the production box** (torch unavailable on the dev laptop).
* GUI/steering helpers: `semitone_bin_centers()`, `expand_semitone_weights()`,
  `fold_to_12()`, `make_steering_target()`.
* Performance: zero-copy framing, batched pocketfft rfft(workers=-1), all three
  bands fused into one (384 x 4097) float32 GEMM (BLAS dispatches AVX-512).
* **nnAudio evaluated and rejected for this op:** its conv1d STFT at n_fft=8192
  costs ~600x the FLOPs of an FFT (O(F*N) vs O(N log N)) plus a ~270 MB kernel
  tensor. nnAudio's value is small-n_fft / trainable frontends, not this.
  (The local nnAudio fork itself is ROCm-clean: pure torch, modern torch.fft,
  no custom kernels — fine for other uses.)

## 6. UI design: two groups, three bands

Decision: UI folds mid+air into one "melody" group; **bass stays separate**
(genre use case: goa/techno static bass + free melody).

* Backend always emits 3 x 128 profiles; the fold is presentation-only.
* **Asymmetric bleed favors this split:** melody at octave 4 reaches the bass
  band at weight ~0.011 (negligible — bass lock is clean); bass harmonics do
  reach the mid band (~0.4 weight at octave 3) but land on root/5th/b7 pitch
  classes, which the scale palette usually allows anyway.
* `BASS_PROFILES`: 'root' (h1/h2), 'root5' (+h3 fifth at ~0.3, octave-window
  attenuated), 'root5b7' (+h7 trace). Better: measure a real genre bass patch's
  band-0 profile once and use it as the template.
* **Lock-vs-prefer = per-band guidance gain:** bass high (constraint), melody
  moderate (palette preference — motion stays free), air ~0.3 or 0
  (>8 kHz chroma is hats/noise in these genres).
* Bump width: sigma=0.25 semitones matches real through-filterbank peak widths
  and gives ~18:1 on/off-scale contrast after folding (0.35 -> 5:1, 0.15 ->
  760:1 but over-penalizes vibrato/detuning).
* Caveat: chroma is pitch-class; band 0 pins *register* (octave window), not the
  exact octave.

## 7. Rate compatibility: SAO-era LatCH vs SAME/SA3

`LATCH_CONTROL.md` targets the **SAO** VAE: 2048x downsampling -> 21.5332 Hz,
256 frames per 11.89 s crop. **SAME/SA3 is 4096x -> ~10.766 Hz, 128 frames per
524288-sample crop.** `compute_same_chroma(align_to_latent=True)` aligns to
SAME (ceil(T/4096)). To feed the existing SAO-era LatCH dataset format instead,
use `interpolate_linear(chroma, 256)` (or `align_to_latent=False` + interpolate).
Don't mix the two rates silently.

## 8. Next steps (production box)

1. Refit the three linear readouts: encode varied audio with the public SAME
   encoder -> latents; compute targets with `compute_same_chroma`; least-squares
   `latents -> target` per band (with bias). Sanity-check on held-out audio.
2. Cross-check `compute_same_chroma_torch` vs NumPy path (torch available there).
3. Verify `n_latent_frames` = ceil(T/4096) against the real encoder's output
   length (edge/padding could differ by one frame).
4. **The steering experiment** (decides everything): guide sampling toward a
   `make_steering_target` profile through the refit head ("only C", then
   "bass=E + phrygian palette") and listen. If pitches move -> build the GUI on
   guidance alone. If not -> fall back to a trained conditioner / LoRA.
5. Optional: measure a genre bass patch -> empirical band-0 template.

## References

* SAME: arXiv 2605.18613 (autoencoder; §3.3.2 semantic regression).
* Stable Audio 3 technical report: arXiv 2605.17991.
* Code: github.com/Stability-AI/stable-audio-tools @ `3241adb`,
  `stable_audio_tools/training/autoencoders.py:567-597`,
  `stable_audio_tools/models/transforms.py` (TightSpectrogram, ILDTransform,
  MeanChannelLog1pTransform).
* torchaudio v2.5.0 `src/torchaudio/prototype/functional/functional.py`
  (chroma_filterbank), `.../transforms/_transforms.py` (ChromaScale.forward).
* PyTorch `torch/functional.py` + ATen `SpectralOps.cpp` (stft normalization).
