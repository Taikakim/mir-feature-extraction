"""
ZeroSep-lite: zero-port separability test on SA3 (runbook section 5, run FIRST).

Uses SA3's built-in audio-to-audio (SDEdit-style noise-then-denoise) as the
approximation of ZeroSep's invert-then-redenoise: encode the mixture, noise to
init_noise_level, denoise with a prompt describing the TARGET SOURCE at
cfg_scale=1 (ZeroSep's omega=1 — also SA3's default). Not exact inversion,
but answers in minutes whether prompt-selective re-denoising can isolate
sources on SAME's 4096x semantic latent at all. Only if promising, build the
full DDPM-inversion PipelineWrapper port.

Use a -base checkpoint: cfg_scale has no effect on post-trained models.

Example:
    python sa3_zerosep_lite.py --input mix.wav \
        --prompts "the bassline" "drums only" "the lead melody" \
        --noise-levels 0.4 0.55 0.7
"""

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--prompts", "-p", type=str, nargs="+",
                        default=["the bassline", "drums only"])
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.4, 0.55, 0.7])
    parser.add_argument("--model", type=str, default="medium-base",
                        help="MUST be -base: cfg_scale is inert on post-trained")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.0,
                        help="ZeroSep separation regime: omega=1 (0=reconstruct, "
                             ">1 generates new content)")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out_dir", type=str, default="zerosep_lite_results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if "base" not in args.model:
        logger.warning(f"'{args.model}' is not a -base checkpoint - the cfg "
                       f"knob (the whole mechanism here) is likely inert.")

    import torch
    import torchaudio
    from stable_audio_3 import StableAudioModel

    wav, sr = torchaudio.load(args.input)                     # (C, T)
    max_samples = int(args.max_seconds * sr)
    if wav.shape[-1] > max_samples:
        wav = wav[..., :max_samples]
    duration = wav.shape[-1] / sr
    logger.info(f"input: {wav.shape} @ {sr} Hz ({duration:.1f} s)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StableAudioModel.from_pretrained(args.model, device=device)

    out_dir = Path(args.out_dir) / f"{Path(args.input).stem}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "original.wav"), wav, sr)

    results = []
    for prompt in args.prompts:
        for nl in args.noise_levels:
            tag = f"{prompt.replace(' ', '_')}_nl{nl:.2f}"
            logger.info(f"--- {tag} ---")
            audio = model.generate(
                # _encode_audio_input unpacks (sample_rate, audio) — the
                # README example passing torchaudio.load() output directly
                # is a docs bug (that tuple is (audio, sr)).
                init_audio=(sr, wav),
                init_noise_level=nl,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                duration=duration,
                steps=args.steps,
                cfg_scale=args.cfg,
                seed=args.seed,
            )
            path = out_dir / f"{tag}.wav"
            out_sr = getattr(model.model, "sample_rate", None) \
                or model.model_config.get("sample_rate", 44100)
            torchaudio.save(str(path), audio[0].float().cpu(), out_sr)
            results.append({"prompt": prompt, "noise_level": nl,
                            "file": path.name})

    (out_dir / "params.json").write_text(json.dumps(
        {"args": vars(args), "results": results}, indent=2))
    logger.info(f"done -> {out_dir}")
    logger.info("LISTEN: does any (prompt, noise_level) cell isolate the "
                "named source while suppressing the rest? Low noise levels "
                "preserve the mixture; high ones regenerate freely. The "
                "separation sweet spot, if it exists, is in between.")


if __name__ == "__main__":
    main()
