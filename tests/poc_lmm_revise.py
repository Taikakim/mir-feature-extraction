"""
Audio Captioning Benchmark

Compares 4 approaches to generating training-quality audio captions:
  Phase 1: Baseline Music Flamingo (GGUF) — 5 prompts from YAML config
  Phase 2: Genre-Hint Music Flamingo (GGUF) — same prompts with genre override
  Phase 3: LLM Revision (GGUF via llama-cpp-python) — revise Phase 1 captions
  Phase 4: Qwen2.5-Omni-7B-AWQ (transformers) — direct audio captioning

Usage:
    python tests/poc_lmm_revise.py /path/to/audio.flac --genre "Goa Trance, Psytrance"
    python tests/poc_lmm_revise.py /path/to/audio.flac --skip-phase3 --skip-phase4
    python tests/poc_lmm_revise.py /path/to/audio.flac --genre "Techno" -v
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)


# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.rocm_env import setup_rocm_env
setup_rocm_env()

import torch
import yaml

from core.text_utils import normalize_music_flamingo_text
from classification.music_flamingo import MusicFlamingoGGUF

# LLM revision models (Phase 3)
LLM_MODELS = {
    "qwen3_14b": {
        "path": "models/LMM/Qwen3-14B-GGUF/Qwen3-14B-Q6_K.gguf",
        "template": "qwen3",
        "label": "Qwen3-14B",
    },
    "gpt_oss_20b": {
        "path": "models/LMM/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf",
        "template": "gpt_oss",
        "label": "GPT-OSS-20B",
    },
    "granite_tiny": {
        "path": "models/LMM/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q8_0.gguf",
        "template": "granite",
        "label": "Granite-tiny",
    },
}

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PromptResult:
    prompt_name: str
    caption: str
    wall_time_s: float
    tokens_generated: int
    tokens_per_sec: float
    raw_caption: str = "" # Raw LLM output before cleaning


@dataclass
class PhaseResult:
    phase_name: str
    model_name: str
    results: Dict[str, PromptResult] = field(default_factory=dict)
    total_wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Genre extraction
# ---------------------------------------------------------------------------

def _genres_from_info_file(audio_path: Path, info_path_override: Optional[Path] = None) -> List[str]:
    """Try to load genres from a .INFO JSON file."""
    from core.json_handler import get_info_path
    candidates = []
    if info_path_override:
        candidates.append(info_path_override)
    # Try get_info_path (folder-based: parent_name.INFO)
    candidates.append(get_info_path(audio_path))
    # Also try sibling .INFO with same stem as the audio file
    candidates.append(audio_path.with_suffix(".INFO"))

    for info_path in candidates:
        if not info_path.exists():
            continue
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
            # Merge genres + musicbrainz_genres, dedupe
            genres = list(info.get("genres", []))
            genres.extend(info.get("musicbrainz_genres", []))
            genres = list(dict.fromkeys(genres))  # dedupe, preserve order
            if genres:
                return genres
        except Exception:
            continue

    # Show what we tried so the user can debug
    if info_path_override:
        print(f"  WARNING: --info file not found: {info_path_override}")
    return []


def extract_genres(audio_path: Path, cli_genre: Optional[str] = None, info_path: Optional[Path] = None) -> List[str]:
    """Extract genre tags. Priority: CLI > .INFO file > ID3 tags > fallback."""

    # 1. CLI Override
    if cli_genre:
        # return [g.strip() for g in cli_genre.split(",")]
        # old method
        res = [g.strip() for g in cli_genre.split(",")]
        print(f"Genres used from CLI override: {res}")
        return res

    # 2. Try .INFO file (pipeline-generated metadata)
    info_genres = _genres_from_info_file(audio_path, info_path)
    if info_genres:
        print(f"Genres found in file .INFO keys: {info_genres}")
        return info_genres

    # 3. Try ID3 / vorbis tags via mutagen
    try:
        import mutagen
        audio = mutagen.File(str(audio_path), easy=True)
        if audio and "genre" in audio:
            raw = audio["genre"]
            genres = []
            for g in raw:
                genres.extend(part.strip() for part in g.split(","))
            if genres:
                print(f"Genres found in ID3 tags: {genres}")
                return genres
    except Exception:
        pass

    # 4. Try ID3 / vorbis tags via mutagen (MANUAL FALLBACK)
    try:
        import mutagen
        audio = mutagen.File(str(audio_path))
        if audio and hasattr(audio, "tags") and audio.tags:

            # A. FLAC / Vorbis comments
            genre_tag = audio.tags.get("GENRE") or audio.tags.get("genre")
            if genre_tag:
                if isinstance(genre_tag, list):
                    # return [g.strip() for g in genre_tag]
                    # Old method
                    res = [g.strip() for g in genre_tag]
                else:
                    res = [str(genre_tag).strip()]
                print(f"Genres found in FLAC/Vorbis tags: {res}")
                return res

            # B. MP3 TCON Frame
            from mutagen.id3 import TCON
            tcon = audio.tags.get("TCON")
            if tcon:
                # return [g.strip() for g in tcon.text]
                # Old method
                res = [g.strip() for g in tcon.text]
                print(f"Genres found in ID3 TCON frame: {res}")
                return res
    except Exception:
        pass

    return ["Unknown"]


# ---------------------------------------------------------------------------
# Phase 1 & 2: Music Flamingo (GGUF)
# ---------------------------------------------------------------------------

def run_music_flamingo_phase(
    audio_path: Path,
    prompts: Dict[str, str],
    flamingo_model: str,
    context_size: int,
    phase_name: str,
    genres: Optional[List[str]] = None,
    verbose: bool = False,
) -> PhaseResult:
    """Run Music Flamingo on all prompts. If genres provided, append genre hint."""
    phase = PhaseResult(phase_name=phase_name, model_name=f"MusicFlamingo-{flamingo_model}")
    t0 = time.monotonic()

    analyzer = MusicFlamingoGGUF(model=flamingo_model, context_size=context_size)

    for prompt_name, prompt_text in prompts.items():
        if genres:
            genre_str = ", ".join(genres)
            prompt_text = (
                f"{prompt_text} Assume that what you understand about the genre "
                f"of the audio is wrong, actually it is/are {genre_str}"
            )

        if verbose:
            print(f"  [{prompt_name}] Running...")

        pt0 = time.monotonic()
        caption = analyzer.analyze(audio_path, prompt=prompt_text, prompt_type=prompt_name)
        pt1 = time.monotonic()

        wall_s = pt1 - pt0
        stats = analyzer.last_stats
        tok = stats.eval_tokens if stats else 0
        tps = stats.eval_tokens_per_sec if stats else 0.0

        phase.results[prompt_name] = PromptResult(
            prompt_name=prompt_name,
            caption=caption,
            wall_time_s=round(wall_s, 2),
            tokens_generated=tok,
            tokens_per_sec=round(tps, 1),
        )

        if verbose:
            print(f"    {tok} tok, {tps:.1f} t/s, {wall_s:.1f}s")

    phase.total_wall_time_s = round(time.monotonic() - t0, 2)
    return phase


# ---------------------------------------------------------------------------
# Phase 3: LLM Revision (llama-cpp-python)
# ---------------------------------------------------------------------------

def strip_thinking_traces(text: str) -> str:
    """Remove thinking/reasoning traces from LLM output."""
    # Strict splitting on </think> if present (Qwen3/DeepSeek style)
    if "</think>" in text:
        parts = text.split("</think>")
        if len(parts) > 1 and len(parts[-1].strip()) > 5:
            text = parts[-1].strip()
        else:
            # If nothing after </think>, maybe the model didn't finish or put the answer inside (unlikely but possible)
            # For now, if empty after think, we might just have to return empty or try to find a caption inside
            pass 

    # Remove <think>...</think> blocks if they remain
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</think>", "", text)
    text = re.sub(r"<think>", "", text)
    
    # Strip GPT-OSS special tokens
    text = text.replace("<|end|>", "").replace("<|return|>", "").replace("<|message|>", "").replace("<|channel|>", "")

    # GPT-OSS / CoT style extraction
    # Look for "Let's write:" "Here is the revised caption:" etc.
    match = re.search(r"(?:Here is|Let's write|Let's craft|Revised description|Revised caption)[:\.]\s*(?:\"|')?(.*?)(?:\"|')?$", text, re.DOTALL | re.IGNORECASE)
    if match:
        potential_caption = match.group(1).strip()
        if len(potential_caption) > 10:
            text = potential_caption

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        low = line.strip().lower()
        if any(
            low.startswith(p)
            for p in [
                "we need to", "we have to", "we must", "we should",
                "the content", "the original", "the user", "the task",
                "revise this", "the caption", "the revised",
                "</analysis>", "<analysis>", "so we", "that matches",
                "(must be", "(keep it", "sure, here", "here is the",
                "the prompt", "the problem", "energetic ...?", "deep ... ...?",
            ]
        ):
            continue
        if line.strip().startswith("(") and line.strip().endswith(")"): # Remove parenthetical instructions
             continue
        
        # specific fix for GPT-OSS-20B artifacts like "The track ... ?" or "The prompt says:"
        if re.match(r"^The (track|prompt|problem|user).*\?$", line.strip()):
            continue

        # Skip lines that are mostly dots, question marks, or ellipsis
        stripped = line.strip()
        noise_chars = stripped.replace(".", "").replace("?", "").replace("!", "").replace(" ", "").replace("-", "").replace(",", "")
        if not noise_chars or (len(noise_chars) < len(stripped) * 0.3 and len(stripped) > 5):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned).strip()

    # Try to extract caption if text starts with common starts but has garbage before
    # Only if we haven't found a clean start already
    if not text.startswith(("This track", "The track", "A ", "An ", "New", "Techno",
                            "The music", "Electronic", "Dark", "High", "Driving")):
        match = re.search(r"((?:This|The|A|An|High|Dark|Driving|Electronic)\s[^.]*\.)", text)
        if match:
            text = match.group(1)

    return text.strip()


def format_revision_prompt(template: str, original_caption: str, genres: List[str], field_name: str) -> str:
    """Format prompt for LLM caption revision."""
    genres_str = ", ".join(genres)
    instruction = (
        f"The real genre of this cropped clip presenting a ten second part of the trace is/are {genres_str}, the genres in the caption to be revised are probably wrong.\n"
        f"Revise this music caption to use the correct genres.\n"
        f"The original caption may have incorrect genre labels. "
        f"Keep the revised caption between 10-30 words.\n"
        f"Do not reference numerical values directly. "
        f"Do NOT output any thinking, reasoning, or preamble. "
        f"Output ONLY the final revised caption.\n\n"
        f"Original caption ({field_name}):\n{original_caption}\n\n"
        f"Revised caption:"
    )

    if template == "qwen3":
        # Use ChatML for Qwen3 as well
        return (
            f"<|im_start|>system\n"
            f"You are a helpful assistant. Output only the revised caption.<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    elif template == "granite":
        return (
            "<|start_of_role|>system<|end_of_role|>"
            "You are a helpful assistant. Please ensure responses are professional, accurate, and safe. "
            "You revise music captions with correct genres. Output only the revised caption."
            "<|end_of_text|>\n"
            f"<|start_of_role|>user<|end_of_role|>{instruction}<|end_of_text|>\n"
            "<|start_of_role|>assistant<|end_of_role|>"
        )
    elif template == "gpt_oss":
        return (
            "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
            "Knowledge cutoff: 2024-06\nCurrent date: 2025-08-05\n\nReasoning: low\n\n"
            "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
            f"<|start|>user<|message|>{instruction}<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
        )
    elif template == "chatml":
        return (
            f"<|im_start|>system\n"
            f"You revise music captions with correct genres. "
            f"Output only the revised caption.<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return instruction


def cleanup_vram():
    """Release GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_llm_revision_phase(
    baseline_results: Dict[str, PromptResult],
    genres: List[str],
    verbose: bool = False,
) -> List[PhaseResult]:
    """Run LLM revision on baseline captions. Returns one PhaseResult per model."""
    from llama_cpp import Llama

    phases = []

    for model_key, config in LLM_MODELS.items():
        model_path = PROJECT_ROOT / config["path"]
        template = config["template"]
        label = config["label"]

        print(f"\n  Loading {label}...")

        if not model_path.exists():
            print(f"    SKIP — model not found: {model_path}")
            continue

        phase = PhaseResult(phase_name=f"Phase 3: LLM Revision ({label})", model_name=label)
        t0 = time.monotonic()

        # Stop tokens per template
        stop_tokens = {
            "granite": ["<|end_of_text|>", "<|start_of_role|>"],
            "chatml": ["<|im_end|>", "<|im_start|>"],
            "qwen3": ["</s>", "<|im_end|>"], # Qwen3 might use either
            "gpt_oss": ["<|end|>", "<|return|>"],
        }.get(template, ["</s>"])

        try:
            # Granite-tiny seems to struggle with large context/batch on some setups
            # Qwen3/GPT-OSS have large context windows and can use Q4 KV cache
            is_large_model = template in ["qwen3", "gpt_oss"]
            
            n_ctx = 32768 if is_large_model else (4096 if template == "granite" else 16384)
            n_batch = 2048 if is_large_model else (256 if template == "granite" else 512)
            
            init_kwargs = {
                "model_path": str(model_path),
                "n_gpu_layers": -1,
                "n_ctx": n_ctx,
                "n_batch": n_batch,
                "flash_attn": True,
                "verbose": False,
            }

            # Enable 4-bit KV cache for large models to save VRAM/RAM
            if is_large_model:
                from llama_cpp import GGML_TYPE_Q4_0
                init_kwargs["type_k"] = GGML_TYPE_Q4_0
                init_kwargs["type_v"] = GGML_TYPE_Q4_0
            
            llm = Llama(**init_kwargs)

            # Granite has tiny context — truncate more aggressively
            max_caption_chars = 300 if template == "granite" else 2000 # Increased for larger context

            for prompt_name, baseline in baseline_results.items():
                if not baseline.caption:
                    continue
                
                # Reset context for each prompt
                llm.reset()

                # Truncate long captions to avoid context overflow
                caption_text = baseline.caption
                if len(caption_text) > max_caption_chars:
                    caption_text = caption_text[:max_caption_chars] + "..."
                prompt = format_revision_prompt(template, caption_text, genres, prompt_name)

                if verbose:
                    print(f"    [{prompt_name}] Revising...")
                    # print(f"--- Prompt ({template}) ---\n{prompt}\n-----------------------")

                try:
                    pt0 = time.monotonic()
                    # Qwen3 non-thinking mode recommended params
                    gen_kwargs = {
                        "max_tokens": 512, # Increased to allow for thinking/CoT
                        "stop": stop_tokens,
                        "echo": False,
                        "temperature": 0.7,
                    }
                    if template == "qwen3":
                        gen_kwargs.update({"top_p": 0.8, "top_k": 20, "min_p": 0.0})
                    elif template == "gpt_oss":
                        # OpenAI/GPT-OSS recommended settings
                        gen_kwargs.update({
                            "max_tokens": 1024, # Larger context for reasoning models
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "top_k": 0,
                        })
                    
                    output = llm(prompt, **gen_kwargs)
                except Exception as e:
                    print(f"      [{prompt_name}] inference error: {e}")
                    continue
                pt1 = time.monotonic()

                wall_s = pt1 - pt0
                raw = output["choices"][0]["text"].strip()
                if verbose:
                    print(f"      RAW: {raw[:100]}..." if len(raw) > 100 else f"      RAW: {raw}")

                cleaned = strip_thinking_traces(raw)
                cleaned = normalize_music_flamingo_text(cleaned)

                usage = output.get("usage", {})
                tok = usage.get("completion_tokens", 0)
                tps = tok / wall_s if wall_s > 0 else 0.0

                phase.results[prompt_name] = PromptResult(
                    prompt_name=prompt_name,
                    caption=cleaned,
                    wall_time_s=round(wall_s, 2),
                    tokens_generated=tok,
                    tokens_per_sec=round(tps, 1),
                    raw_caption=raw,
                )

                if verbose:
                    print(f"      {tok} tok, {tps:.1f} t/s, {wall_s:.1f}s")

            del llm

        except Exception as e:
            print(f"    ERROR: {e}")
            phase.results["error"] = PromptResult(
                prompt_name="error", caption=str(e),
                wall_time_s=0, tokens_generated=0, tokens_per_sec=0,
            )

        cleanup_vram()
        phase.total_wall_time_s = round(time.monotonic() - t0, 2)
        phases.append(phase)

    return phases


# ---------------------------------------------------------------------------
# Phase 4: Qwen2.5-Omni-7B-AWQ (low-VRAM mode via autoawq)
# ---------------------------------------------------------------------------

# Path to the patched low-VRAM modeling file (with RoPE fix for transformers 5.x)
QWEN_OMNI_LOW_VRAM_DIR = PROJECT_ROOT / "repos" / "Qwen2.5-Omni" / "low-VRAM-mode"


def _patch_qwen_omni_module():
    """Monkey-patch transformers with the low-VRAM Qwen2.5-Omni modeling file."""
    import importlib.util

    mod_name = "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni"
    mod_path = str(QWEN_OMNI_LOW_VRAM_DIR / "modeling_qwen2_5_omni_low_VRAM_mode.py")

    if mod_name in sys.modules:
        del sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    new_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new_mod)
    sys.modules[mod_name] = new_mod


def _load_qwen_omni_awq(device: str = "cuda"):
    """Load Qwen2.5-Omni-7B-AWQ using the official low-VRAM approach."""
    from awq.models.base import BaseAWQForCausalLM

    # Import AFTER patching
    sys.path.insert(0, str(QWEN_OMNI_LOW_VRAM_DIR))
    from modeling_qwen2_5_omni_low_VRAM_mode import (
        Qwen2_5OmniDecoderLayer,
        Qwen2_5OmniForConditionalGeneration,
    )

    class Qwen2_5_OmniAWQForConditionalGeneration(BaseAWQForCausalLM):
        layer_type = "Qwen2_5OmniDecoderLayer"
        max_seq_len_key = "max_position_embeddings"
        modules_to_not_convert = ["visual"]

        @staticmethod
        def get_model_layers(model):
            return model.thinker.model.layers

        @staticmethod
        def get_act_for_scaling(module):
            return dict(is_scalable=False)

        @staticmethod
        def move_embed(model, device):
            model.thinker.model.embed_tokens = model.thinker.model.embed_tokens.to(device)
            model.thinker.visual = model.thinker.visual.to(device)
            model.thinker.audio_tower = model.thinker.audio_tower.to(device)
            model.thinker.visual.rotary_pos_emb = model.thinker.visual.rotary_pos_emb.to(device)
            model.thinker.model.rotary_emb = model.thinker.model.rotary_emb.to(device)
            for layer in model.thinker.model.layers:
                layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

        @staticmethod
        def get_layers_for_scaling(module, input_feat, module_kwargs):
            layers = []
            layers.append(dict(
                prev_op=module.input_layernorm,
                layers=[module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            ))
            if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                layers.append(dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                ))
            layers.append(dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            ))
            layers.append(dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            ))
            return layers

    model_id = "Qwen/Qwen2.5-Omni-7B-AWQ"

    model = Qwen2_5_OmniAWQForConditionalGeneration.from_quantized(
        model_id,
        model_type="qwen2_5_omni",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )

    # Move embeddings + rotary to GPU (weights stay on GPU via AWQ)
    model.model.thinker.model.embed_tokens = model.model.thinker.model.embed_tokens.to(device)
    model.model.thinker.visual = model.model.thinker.visual.to(device)
    model.model.thinker.audio_tower = model.model.thinker.audio_tower.to(device)
    model.model.thinker.visual.rotary_pos_emb = model.model.thinker.visual.rotary_pos_emb.to(device)
    model.model.thinker.model.rotary_emb = model.model.thinker.model.rotary_emb.to(device)
    for layer in model.model.thinker.model.layers:
        layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

    return model, model_id



def run_qwen_omni_phase(
    audio_path: Path,
    prompts: Dict[str, str],
    baseline_flamingo_results: Optional[Dict[str, PromptResult]] = None,
    genres: Optional[List[str]] = None,
    verbose: bool = False,
) -> List[PhaseResult]:
    """Run Qwen2.5-Omni-7B-AWQ in 3 sub-phases: Baseline, Genre-Hint, Revision."""
    # Check dependencies
    try:
        from awq.models.base import BaseAWQForCausalLM  # noqa: F401
        from transformers import Qwen2_5OmniProcessor
        from qwen_omni_utils import process_mm_info
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print(f"\n  SKIP Phase 4 — missing dependency: {e}")
        return []

    if not QWEN_OMNI_LOW_VRAM_DIR.exists():
        print(f"\n  SKIP Phase 4 — low-VRAM mode not found: {QWEN_OMNI_LOW_VRAM_DIR}")
        return []

    # Prepare phases
    phases_to_run = []
    
    # 4a: Baseline
    phases_to_run.append({
        "name": "Phase 4a: Qwen Omni (Baseline)",
        "model": "Qwen2.5-Omni-7B-AWQ",
        "prompts": prompts,
        "type": "baseline"
    })

    # 4b: Genre-Hint
    if genres:
        genres_str = ", ".join(genres)
        hint_prompts = {}
        for k, v in prompts.items():
            hint_prompts[k] = (
                f"The real genre of this cropped clip presenting a ten second part of the trace is/are {genres_str}, "
                f"the genres in the audio might be subtle.\n{v}"
            )
        phases_to_run.append({
            "name": "Phase 4b: Qwen Omni (Genre-Hint)",
            "model": "Qwen2.5-Omni-7B-AWQ",
            "prompts": hint_prompts,
            "type": "genre-hint"
        })

    # 4c: Revision (Audio + Flamingo Caption + Genre)
    if genres and baseline_flamingo_results:
        rev_prompts = {}
        for k, v in prompts.items():
            if k in baseline_flamingo_results and baseline_flamingo_results[k].caption:
                orig = baseline_flamingo_results[k].caption
                rev_prompts[k] = (
                    f"The real genre of this cropped clip is {genres_str}.\n"
                    f"A previous analysis described it as: '{orig}'\n"
                    f"Listen to the audio yourself and revise the description to be more accurate, "
                    f"combining what you hear with the provided genre info.\n{v}"
                )
        if rev_prompts:
            phases_to_run.append({
                "name": "Phase 4c: Qwen Omni (Revision)",
                "model": "Qwen2.5-Omni-7B-AWQ",
                "prompts": rev_prompts,
                "type": "revision"
            })

    results = []
    
    if not phases_to_run:
        return []

    try:
        # Patch transformers with low-VRAM modeling
        _patch_qwen_omni_module()

        print("  Loading Qwen2.5-Omni-7B-AWQ (low-VRAM mode)...")
        model, model_id = _load_qwen_omni_awq()
        
        try:
            spk_path = hf_hub_download(repo_id=model_id, filename='spk_dict.pt')
            model.model.load_speakers(spk_path)
        except Exception as e:
            print(f"    WARNING: Failed to load speaker map: {e}")

        processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        audio_url = str(audio_path.resolve())

        # Run all prepared phases with the loaded model
        for phase_cfg in phases_to_run:
            print(f"\n  Running {phase_cfg['name']}...")
            phase_res = PhaseResult(phase_name=phase_cfg["name"], model_name=phase_cfg["model"])
            t0_phase = time.monotonic()

            for prompt_name, prompt_text in phase_cfg["prompts"].items():
                if verbose:
                    print(f"    [{prompt_name}] Generating...")
                    # print(f"--- Prompt ---\n{prompt_text}\n--------------")

                messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are a music analysis assistant."}]},
                    {"role": "user", "content": [
                        {"type": "audio", "audio_url": audio_url},
                        {"type": "text", "text": prompt_text},
                    ]},
                ]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
                inputs = processor(
                    text=text, audio=audios, images=images, videos=videos,
                    return_tensors="pt", padding=True,
                ).to("cuda")

                input_len = inputs["input_ids"].shape[1]
                
                pt0 = time.monotonic()
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        use_audio_in_video=True,
                        return_audio=False,
                        max_new_tokens=256,
                    )
                pt1 = time.monotonic()

                text_ids = output[0] if isinstance(output, tuple) else output
                new_ids = text_ids[0][input_len:]
                caption = processor.decode(new_ids, skip_special_tokens=True).strip()
                caption = normalize_music_flamingo_text(caption)

                wall_s = pt1 - pt0
                tok = len(new_ids)
                tps = tok / wall_s if wall_s > 0 else 0.0

                phase_res.results[prompt_name] = PromptResult(
                    prompt_name=prompt_name,
                    caption=caption,
                    wall_time_s=round(wall_s, 2),
                    tokens_generated=tok,
                    tokens_per_sec=round(tps, 1),
                )

                if verbose:
                    print(f"      {tok} tok, {tps:.1f} t/s, {wall_s:.1f}s")
            
            phase_res.total_wall_time_s = round(time.monotonic() - t0_phase, 2)
            results.append(phase_res)

        del model, processor
        cleanup_vram()
        return results

    except torch.cuda.OutOfMemoryError:
        print("    OOM — skipping Qwen2.5-Omni (not enough VRAM)")
        cleanup_vram()
        return []
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        cleanup_vram()
        return []



# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_comparison_table(all_phases: List[PhaseResult], prompt_names: List[str]):
    """Print side-by-side comparison grouped by prompt."""
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)

    for pname in prompt_names:
        print(f"\nPrompt: {pname}")
        print("-" * 100)
        print(f"  {'Phase':<30} | {'Time':>6} | {'Tok':>5} | {'t/s':>8} | Caption")
        print(f"  {'':-<30}-+-{'':-<6}-+-{'':-<5}-+-{'':-<8}-+-{'':-<40}")

        for phase in all_phases:
            result = phase.results.get(pname)
            if not result or result.prompt_name == "error":
                continue

            caption_preview = result.caption[:60].replace("\n", " ")
            if len(result.caption) > 60:
                caption_preview += "..."

            print(
                f"  {phase.phase_name:<30} | "
                f"{result.wall_time_s:5.1f}s | "
                f"{result.tokens_generated:5d} | "
                f"{result.tokens_per_sec:7.1f} | "
                f'"{caption_preview}"'
            )

    print("\n" + "=" * 100)
    print("PHASE TOTALS")
    print("=" * 100)
    for phase in all_phases:
        print(f"  {phase.phase_name:<40} {phase.total_wall_time_s:6.1f}s  ({phase.model_name})")
    total = sum(p.total_wall_time_s for p in all_phases)
    print(f"  {'TOTAL':<40} {total:6.1f}s")


def build_output_json(
    audio_path: Path,
    genres: List[str],
    all_phases: List[PhaseResult],
) -> dict:
    """Build full JSON output with metadata and all results."""
    out = {
        "audio_file": str(audio_path),
        "audio_filename": audio_path.name,
        "genres": genres,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phases": [],
    }

    for phase in all_phases:
        phase_dict = {
            "phase_name": phase.phase_name,
            "model_name": phase.model_name,
            "total_wall_time_s": phase.total_wall_time_s,
            "results": {},
        }
        for pname, result in phase.results.items():
            if result.prompt_name == "error":
                phase_dict["results"]["error"] = result.caption
                continue
            phase_dict["results"][pname] = asdict(result)
        out["phases"].append(phase_dict)

    return out



def format_ensemble_prompt(template: str, flamingo_caption: str, qwen_caption: str, genres: List[str]) -> str:
    """Format prompt for Phase 5 ensemble rewrite."""
    genres_str = ", ".join(genres)
    instruction = (
        f"The real genre of this audio clip is {genres_str}.\n"
        f"Here are two descriptions of the audio from different AI models:\n"
        f"1. {flamingo_caption}\n"
        f"2. {qwen_caption}\n\n"
        f"Combine the salient points from both into a single, accurate, and descriptive caption. "
        f"Prioritize details that align with the known genre ({genres_str}). "
        f"Keep the result between 20-50 words. Do not hallucinate instruments not mentioned.\n\n"
        f"Revised Description:"
    )

    if template == "qwen3":
        return (
            "<|im_start|>system\n"
            "You are an expert music editor. Synthesize multiple descriptions into one. "
            "Output only the final description.<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    elif template == "granite":
        return (
            "<|start_of_role|>system<|end_of_role|>"
            "You are a helpful assistant. Please ensure responses are professional, accurate, and safe. "
            "You are an expert music editor. Synthesize multiple descriptions into one."
            "<|end_of_text|>\n"
            f"<|start_of_role|>user<|end_of_role|>{instruction}<|end_of_text|>\n"
            "<|start_of_role|>assistant<|end_of_role|>"
        )
    elif template == "gpt_oss":
        return (
            "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
            "Knowledge cutoff: 2024-06\nCurrent date: 2025-08-05\n\nReasoning: low\n\n"
            "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
            f"<|start|>user<|message|>{instruction}<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
        )
    elif template == "chatml":
        return (
            f"<|im_start|>system\n"
            f"You are an expert music editor. Synthesize multiple descriptions into one. "
            f"Output only the final description.<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return instruction


def run_ensemble_rewrite_phase(
    phase2_results: Dict[str, PromptResult], # From Flamingo Genre-Hint
    phase4b_results: Dict[str, PromptResult], # From Qwen Omni Genre-Hint
    genres: List[str],
    verbose: bool = False,
) -> List[PhaseResult]:
    """Run Phase 5: Ensemble Rewrite using LLMs (Qwen3-14B / Granite)."""
    from llama_cpp import Llama

    start_t0 = time.monotonic()
    phases = []
    
    # Identify common prompts
    common_prompts = set(phase2_results.keys()) & set(phase4b_results.keys())
    if not common_prompts:
        print("  SKIP Phase 5 — no common prompts between Flamingo and Qwen Omni phases.")
        return []

    for model_key, config in LLM_MODELS.items():
        # if model_key == "gpt_oss_20b": continue # Skip the broken one for now? Or keep it? Let's keep it.

        model_path = PROJECT_ROOT / config["path"]
        template = config["template"]
        label = config["label"]

        print(f"\n  Loading {label} (Ensemble)...")
        if not model_path.exists():
            print(f"    SKIP — model not found: {model_path}")
            continue

        phase = PhaseResult(phase_name=f"Phase 5: Ensemble ({label})", model_name=label)
        t0 = time.monotonic()

        stop_tokens = {
            "granite": ["<|end_of_text|>", "<|start_of_role|>"],
            "chatml": ["<|im_end|>", "<|im_start|>"],
            "qwen3": ["</s>", "<|im_end|>"],
            "gpt_oss": ["<|end|>", "<|return|>"],
        }.get(template, ["</s>"])

        try:
            is_large_model = template in ["qwen3", "gpt_oss"]
            n_ctx = 4096 if template == "granite" else 12288
            n_batch = 256 if template == "granite" else 512

            init_kwargs = {
                "model_path": str(model_path),
                "n_gpu_layers": -1,
                "n_ctx": n_ctx,
                "n_batch": n_batch,
                "flash_attn": True,
                "verbose": False,
            }

            if is_large_model:
                from llama_cpp import GGML_TYPE_Q4_0
                init_kwargs["type_k"] = GGML_TYPE_Q4_0
                init_kwargs["type_v"] = GGML_TYPE_Q4_0

            llm = Llama(**init_kwargs)

            for prompt_name in common_prompts:
                flam_cap = phase2_results[prompt_name].caption
                qwen_cap = phase4b_results[prompt_name].caption
                
                if not flam_cap or not qwen_cap:
                    continue
                
                # Reset context
                llm.reset()

                prompt = format_ensemble_prompt(template, flam_cap, qwen_cap, genres)

                if verbose:
                    print(f"    [{prompt_name}] synthesizing...")
                    # print(f"--- Prompt ({template}) ---\n{prompt}\n-----------------------")

                pt0 = time.monotonic()
                gen_kwargs = {
                    "max_tokens": 512,
                    "stop": stop_tokens,
                    "echo": False,
                    "temperature": 0.7,
                }
                if template == "qwen3":
                    gen_kwargs.update({"top_p": 0.8, "top_k": 20, "min_p": 0.0})
                elif template == "gpt_oss":
                    gen_kwargs.update({
                        "max_tokens": 1024,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": 0,
                    })

                output = llm(prompt, **gen_kwargs)
                pt1 = time.monotonic()

                wall_s = pt1 - pt0
                raw = output["choices"][0]["text"].strip()
                if verbose:
                    print(f"      RAW: {raw[:100]}..." if len(raw) > 100 else f"      RAW: {raw}")

                cleaned = strip_thinking_traces(raw)
                cleaned = normalize_music_flamingo_text(cleaned)

                usage = output.get("usage", {})
                tok = usage.get("completion_tokens", 0)
                tps = tok / wall_s if wall_s > 0 else 0.0

                phase.results[prompt_name] = PromptResult(
                    prompt_name=prompt_name,
                    caption=cleaned,
                    wall_time_s=round(wall_s, 2),
                    tokens_generated=tok,
                    tokens_per_sec=round(tps, 1),
                    raw_caption=raw,
                )

                if verbose:
                    print(f"      {tok} tok, {tps:.1f} t/s, {wall_s:.1f}s")

            del llm
            cleanup_vram()
        
        except Exception as e:
            print(f"    ERROR: {e}")
            cleanup_vram()
            continue

        phase.total_wall_time_s = round(time.monotonic() - t0, 2)
        phases.append(phase)

    return phases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict:
    """Load YAML config and extract music_flamingo section."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("music_flamingo", {})


def main():
    parser = argparse.ArgumentParser(
        description="Audio Captioning Benchmark — compare Music Flamingo, LLM revision, and Qwen-Omni"
    )
    parser.add_argument("audio", help="Path to audio file (WAV, FLAC, MP3)")
    parser.add_argument("--config", default="config/master_pipeline.yaml", help="YAML config path")
    parser.add_argument("--output", default="poc_benchmark_results.json", help="Output JSON path")
    parser.add_argument("--genre", default=None, help='Genre override, comma-separated (e.g. "Goa Trance, Psytrance")')
    parser.add_argument("--info", default=None, help="Path to .INFO file for genre lookup (if not co-located with audio)")
    parser.add_argument("--flamingo-model", default="Q8_0", choices=["IQ3_M", "Q6_K", "Q8_0"],
                        help="Music Flamingo quantization (default: Q8_0)")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip baseline Music Flamingo")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip genre-hint Music Flamingo")
    parser.add_argument("--skip-phase3", action="store_true", help="Skip LLM revision")
    parser.add_argument("--skip-phase4", action="store_true", help="Skip Qwen2.5-Omni")
    parser.add_argument("--skip-phase5", action="store_true", help="Skip Ensemble Rewrite")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    mf_config = load_config(config_path)

    # Load prompts from config (fall back to defaults in music_flamingo.py)
    prompts = mf_config.get("prompts", {})
    if not prompts:
        from classification.music_flamingo import DEFAULT_PROMPTS
        prompts = DEFAULT_PROMPTS.copy()

    context_size = mf_config.get("context_size", 2048)
    prompt_names = list(prompts.keys())

    # Genre extraction
    info_path = Path(args.info).resolve() if args.info else None
    genres = extract_genres(audio_path, args.genre, info_path)

    print("=" * 100)
    print("AUDIO CAPTIONING BENCHMARK")
    print("=" * 100)
    print(f"  Audio:   {audio_path.name}")
    print(f"  Genres:  {', '.join(genres)}")
    print(f"  Prompts: {', '.join(prompt_names)}")
    print(f"  Model:   MusicFlamingo {args.flamingo_model}, ctx={context_size}")
    skips = []
    if args.skip_phase1: skips.append("P1")
    if args.skip_phase2: skips.append("P2")
    if args.skip_phase3: skips.append("P3")
    if args.skip_phase4: skips.append("P4")
    if args.skip_phase5: skips.append("P5")
    if skips:
        print(f"  Skip:    {', '.join(skips)}")
    print("=" * 100)

    all_phases: List[PhaseResult] = []

    # Phase 1: Baseline Music Flamingo
    baseline_flamingo_results = None
    if not args.skip_phase1:
        print("\n>> Phase 1: Baseline Music Flamingo")
        phase1 = run_music_flamingo_phase(
            audio_path, prompts, args.flamingo_model, context_size,
            phase_name="Phase 1 (Flamingo)", verbose=args.verbose,
        )
        all_phases.append(phase1)
        baseline_flamingo_results = phase1.results
        print(f"   Done in {phase1.total_wall_time_s:.1f}s")
    else:
        print("\n>> Phase 1: SKIPPED")
        phase1 = None

    # Phase 2: Genre-Hint Music Flamingo
    phase2_results = None
    if not args.skip_phase2:
        print("\n>> Phase 2: Genre-Hint Music Flamingo")
        phase2 = run_music_flamingo_phase(
            audio_path, prompts, args.flamingo_model, context_size,
            phase_name="Phase 2 (Genre-Hint)", genres=genres, verbose=args.verbose,
        )
        all_phases.append(phase2)
        phase2_results = phase2.results
        print(f"   Done in {phase2.total_wall_time_s:.1f}s")
    else:
        print("\n>> Phase 2: SKIPPED")

    # Phase 3: LLM Revision
    if not args.skip_phase3:
        if phase1 is None:
            print("\n>> Phase 3: SKIPPED (no baseline captions — Phase 1 was skipped)")
        else:
            print("\n>> Phase 3: LLM Revision")
            revision_phases = run_llm_revision_phase(
                phase1.results, genres, verbose=args.verbose,
            )
            all_phases.extend(revision_phases)
            for rp in revision_phases:
                print(f"   {rp.model_name}: {rp.total_wall_time_s:.1f}s")
    else:
        print("\n>> Phase 3: SKIPPED")

    # Phase 4: Qwen2.5-Omni (Sub-phases 4a, 4b, 4c)
    phase4b_results = None
    if not args.skip_phase4:
        print("\n>> Phase 4: Qwen2.5-Omni-7B-AWQ (Three Sub-Phases)")
        qwen_phases = run_qwen_omni_phase(
            audio_path, prompts, 
            baseline_flamingo_results=baseline_flamingo_results, 
            genres=genres, 
            verbose=args.verbose
        )
        if qwen_phases:
            all_phases.extend(qwen_phases)
            for qp in qwen_phases:
                print(f"   {qp.phase_name}: {qp.total_wall_time_s:.1f}s")
                if qp.phase_name == "Phase 4b: Qwen Omni (Genre-Hint)":
                    phase4b_results = qp.results
        else:
            print("   (No results from Phase 4)")
    else:
        print("\n>> Phase 4: SKIPPED")

    # Phase 5: Ensemble Rewrite
    if not args.skip_phase5 and phase2_results and phase4b_results:
        print("\n>> Phase 5: Ensemble Rewrite (Combining Flamingo & Qwen Omni)")
        ensemble_phases = run_ensemble_rewrite_phase(
            phase2_results, phase4b_results, genres, verbose=args.verbose
        )
        all_phases.extend(ensemble_phases)
        for ep in ensemble_phases:
            print(f"   {ep.model_name}: {ep.total_wall_time_s:.1f}s")
    elif not args.skip_phase5:
        print("\n>> Phase 5: SKIPPED (Requires Phase 2 and Phase 4b results)")

    # Output
    if all_phases:
        print_comparison_table(all_phases, prompt_names)

        # Save JSON
        output_data = build_output_json(audio_path, genres, all_phases)
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\nNo phases were run.")


if __name__ == "__main__":
    main()
