# Session Summary - 2026-02-09 / 2026-02-10

## What Was Accomplished

### 1. Audio Captioning Benchmark (Full Rewrite)

`tests/poc_lmm_revise.py` was rewritten from a 282-line hardcoded prototype into a ~1200-line multi-phase audio captioning benchmark with CLI interface, YAML config loading, and JSON output.

**5 Phases:**
- **Phase 1:** Baseline Music Flamingo (GGUF via subprocess) -- 5 prompts from YAML config
- **Phase 2:** Genre-Hint Music Flamingo -- same prompts with genre context appended
- **Phase 3:** LLM Revision (llama-cpp-python) -- Qwen3-14B, GPT-OSS-20B, Granite-tiny revise Phase 1 captions
- **Phase 4a/b/c:** Qwen2.5-Omni-7B-AWQ -- direct audio captioning (baseline, genre-hint, revision sub-phases)
- **Phase 5:** Ensemble Rewrite -- LLMs synthesize Flamingo + Qwen Omni outputs into final captions

**CLI:**
```bash
python tests/poc_lmm_revise.py /path/to/audio.flac --genre "Goa Trance, Psytrance" -v
python tests/poc_lmm_revise.py /path/to/audio.flac --skip-phase3 --skip-phase4
python tests/poc_lmm_revise.py /path/to/audio.flac --info /path/to/file.INFO
```

### 2. Genre Extraction Chain

Implemented multi-source genre lookup with priority:
1. `--genre` CLI flag (comma-separated)
2. `--info` or co-located `.INFO` file (`genres` + `musicbrainz_genres` keys)
3. ID3/Vorbis tags via mutagen (easy mode + manual TCON fallback)
4. Fallback: `["Unknown"]`

### 3. Qwen2.5-Omni-7B-AWQ Integration

Integrated as Phase 4 using the official low-VRAM approach:
- Monkey-patches `transformers.models.qwen2_5_omni.modeling_qwen2_5_omni` with RoPE-fixed modeling file
- Loads via `BaseAWQForCausalLM.from_quantized()` (autoawq)
- Manual device placement for embeddings, rotary, visual, audio_tower
- Uses `Qwen2_5OmniProcessor` + `process_mm_info()` + `apply_chat_template()`
- Text-only output (`return_audio=False`), ~6GB VRAM

**Requirements:** autoawq, qwen-omni-utils, low-VRAM modeling file in `repos/Qwen2.5-Omni/low-VRAM-mode/`

### 4. Model-Specific Chat Templates (Fixed)

Applied each model's documented chat format instead of relying on ad-hoc output filtering:

**GPT-OSS-20B (Harmony format):**
- `<|start|>role<|message|>content<|end|>` token structure
- `<|channel|>final` to bypass reasoning and get direct answers
- `Reasoning: low` in system prompt
- Stop tokens: `<|end|>`, `<|return|>`

**Granite 4.0 H Tiny:**
- `<|start_of_role|>role<|end_of_role|>content<|end_of_text|>` format
- Documented default system prompt prefix
- Context limited to 4096, batch 256 (small model)

**Qwen3-14B (ChatML):**
- `<|im_start|>role\ncontent<|im_end|>` format
- Proper ChatML for ensemble (replaced broken `/no_think` prefix)
- Sampling: `top_p=0.8`, `top_k=20`, `min_p=0.0`

### 5. Inference Optimizations

- **4-bit KV Cache:** `GGML_TYPE_Q4_0` for Qwen3-14B and GPT-OSS-20B in both Phase 3 and Phase 5 (reduces VRAM/bandwidth)
- **32k Context Window:** For large models with 2048 batch size
- **Context Reset:** `llm.reset()` between prompts to prevent KV cache pollution and `llama_decode` crashes
- **Per-prompt try/except:** Granite failures on individual prompts don't kill the whole model's run
- **Caption truncation:** 300 chars for Granite, 2000 for larger models

### 6. Output Cleaning

- `strip_thinking_traces()` handles `<think>`/`</think>` (Qwen3), GPT-OSS Harmony tokens, reasoning preambles
- GPT-OSS artifacts: "Let's write:", "The problem is...", parenthetical instructions
- Noise filter: Lines that are mostly dots/question marks/ellipsis
- All captions normalized via `normalize_music_flamingo_text()` for T5 compatibility

## Key Bugs Fixed

| Bug | Cause | Fix |
|-----|-------|-----|
| Phase 3 crash: `'str' object cannot be interpreted as an integer` | `type_k="q4_0"` (string) | Changed to `type_k=GGML_TYPE_Q4_0` (integer 2) |
| Phase 3 Granite `llama_decode returned -1` | Context overflow | Reduced n_ctx to 4096, caption truncation, `llm.reset()` per prompt |
| Phase 5 Qwen3 blank captions for "technical"/"brief" | `strip_thinking_traces` filtered lines starting with "the track is a" | Removed over-aggressive prefix filters |
| GPT-OSS template `\\n` literal | f-string escaping | String concatenation without f-prefix for static parts |
| Phase 4 garbage output (initial attempt) | Wrong processor/loading method | Rewrote to use official AWQ low-VRAM approach |

## Key Files Modified

- `tests/poc_lmm_revise.py` -- Complete rewrite + iterative fixes
- (Templates, data structures, CLI, genre extraction, 5 phases all in this file)

## VRAM Sequence (16GB card)

```
Phase 1 (MF subprocess ~9GB) -> exits cleanly
Phase 2 (MF subprocess ~9GB) -> exits cleanly
Phase 3a (Qwen3-14B ~10GB)   -> del + cleanup
Phase 3b (GPT-OSS-20B ~8GB)  -> del + cleanup
Phase 3c (Granite-tiny ~3GB)  -> del + cleanup
Phase 4  (Qwen-Omni AWQ ~6GB, shared across 4a/b/c) -> del + cleanup
Phase 5a (Qwen3-14B ~10GB)   -> del + cleanup
Phase 5b (GPT-OSS-20B ~8GB)  -> del + cleanup
Phase 5c (Granite-tiny ~3GB)  -> del + cleanup
```

## Known Limitations

- **GPT-OSS-20B MXFP4:** Aggressive quantization degrades instruction-following; may still produce reasoning artifacts despite Harmony template
- **Granite-tiny:** Very small context and model size; truncates input captions and produces abbreviated output
- **Qwen2.5-Omni deps:** Requires specific autoawq + qwen-omni-utils + patched modeling file; not yet integrated into main requirements.txt
