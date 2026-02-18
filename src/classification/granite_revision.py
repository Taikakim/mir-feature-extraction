"""
Granite-based revision of Music Flamingo descriptions.

Condenses verbose MF output into focused summaries using Granite-tiny (~130MB).
Uses a single batched LLM call with XML-tagged output for all revision prompts.

Usage:
    from classification.granite_revision import GraniteReviser
    r = GraniteReviser('models/LMM/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q8_0.gguf')
    results = r.revise(
        {'music_flamingo_full': 'A dense atmospheric trance track...'},
        {'short_mood': 'The mood and emotional character (10-20 words)'}
    )
    r.close()
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Granite stop tokens
GRANITE_STOP = ["<|end_of_text|>", "<|start_of_role|>"]


class GraniteReviser:
    """Condenses Music Flamingo descriptions using Granite-tiny."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_batch: int = 256,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        from llama_cpp import Llama

        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(f"Loading Granite revision model: {Path(model_path).name}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_gpu_layers=-1,
            verbose=False,
        )
        logger.info("Granite revision model loaded")

    def revise(
        self,
        mf_results: Dict[str, str],
        revision_prompts: Dict[str, str],
    ) -> Dict[str, str]:
        """Run all revision prompts in a single LLM call.

        Args:
            mf_results: Dict of MF output keys to text (e.g. {'music_flamingo_full': '...'})
            revision_prompts: Dict of revision key to instruction (e.g. {'short_mood': '...'})

        Returns:
            Dict of revision key to condensed text (e.g. {'short_mood': '...'})
        """
        if not mf_results or not revision_prompts:
            return {}

        # Build source descriptions block
        source_lines = []
        for key, text in mf_results.items():
            if text and isinstance(text, str):
                source_lines.append(f"[{key}]: {text}")

        if not source_lines:
            return {}

        sources = "\n".join(source_lines)

        # Build revision instructions with XML tags
        instructions = []
        for key, instruction in revision_prompts.items():
            instructions.append(f"<{key}>{instruction}</{key}>")
        instructions_text = "\n".join(instructions)

        # Granite chat template
        prompt = (
            f"<|start_of_role|>system<|end_of_role|>"
            f"You condense music descriptions into focused summaries. "
            f"Output each summary inside its XML tag. Output ONLY the tags with content."
            f"<|end_of_text|>"
            f"<|start_of_role|>user<|end_of_role|>"
            f"Source descriptions:\n{sources}\n\n"
            f"Create these summaries:\n{instructions_text}"
            f"<|end_of_text|>"
            f"<|start_of_role|>assistant<|end_of_role|>"
        )

        t0 = time.monotonic()
        output = self.llm.create_completion(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=GRANITE_STOP,
        )
        wall_s = time.monotonic() - t0

        raw = output["choices"][0]["text"]
        usage = output.get("usage", {})
        tok = usage.get("completion_tokens", 0)
        tps = tok / wall_s if wall_s > 0 else 0

        logger.info(f"  Granite revision: {tok} tok, {tps:.1f} t/s, {wall_s:.1f}s")

        # Parse XML-tagged output
        from core.text_utils import strip_structural_tokens, normalize_music_flamingo_text

        results = {}
        for key in revision_prompts:
            match = re.search(rf"<{re.escape(key)}>(.*?)</{re.escape(key)}>", raw, re.DOTALL)
            if match:
                text = match.group(1).strip()
                text = strip_structural_tokens(text)
                text = normalize_music_flamingo_text(text)
                if text:
                    results[key] = text

        return results

    def close(self):
        """Free the model."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            logger.info("Granite revision model freed")
