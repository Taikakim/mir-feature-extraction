import json
import os
import gc
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_cpp import Llama
from src.core.text_utils import normalize_music_flamingo_text

# Try to import torch for VRAM cleanup (optional - may fail on some ROCm setups)
try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

# 1. Define Models with their chat template info
MODELS = {
    "qwen": {
        "path": "/home/kim/Projects/mir/models/LMM/Qwen3-14B-GGUF/Qwen3-14B-Q6_K.gguf",
        "template": "qwen3",
    },
    "gpt_oss": {
        "path": "/home/kim/Projects/mir/models/LMM/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf",
        "template": "chatml",
    },
    "granite": {
        "path": "/home/kim/Projects/mir/models/LMM/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q8_0.gguf",
        "template": "granite",
    }
}

# Caption fields to revise
CAPTION_FIELDS = [
    "music_flamingo_full",
    "music_flamingo_technical",
    "music_flamingo_genre_mood",
    "music_flamingo_instrumentation",
    "music_flamingo_structure",
]

# 2. Define Input Data (Mock)
INPUT_DATA = {
  "position": 0.680267711124854,
  "bpm": 123.0,
  "beat_count": 92,
  "downbeats": 24,
  "release_year": 1990,
  "release_date": "null",
  "artists": ["Fatal Morgana"],
  "label": "Antler-Subway",
  "genres": ["new beat", "techno"],
  "popularity": 68,
  "spotify_id": "5cF0dROlMOK5uNZtivgu50",
  "lufs": -12.42329965596635,
  "lra": 3.8750562720109265,
  "lufs_drums": -16.709359312386415,
  "lufs_bass": -24.583092998138394,
  "lufs_other": -18.55552710506464,
  "lufs_vocals": -21.803445089736304,
  "spectral_flatness": 0.008952255360782146,
  "spectral_flux": 3.0,
  "spectral_skewness": 1.3965900462482845,
  "spectral_kurtosis": 5.575034917704278,
  "rms_energy_bass": -22.211019225098507,
  "rms_energy_body": -20.213741349828066,
  "rms_energy_mid": -23.51870479346629,
  "rms_energy_air": -23.837356854584584,
  "chroma_0": 0.040063709020614624,
  "chroma_1": 0.23388366401195526,
  "chroma_2": 0.029513441026210785,
  "chroma_3": 0.010784460231661797,
  "chroma_4": 0.11189345270395279,
  "chroma_5": 0.036441121250391006,
  "chroma_6": 0.23806947469711304,
  "chroma_7": 0.029021088033914566,
  "chroma_8": 0.04346969351172447,
  "chroma_9": 0.07327371090650558,
  "chroma_10": 0.02888210490345955,
  "chroma_11": 0.12470410019159317,
  "brightness": 70.08593952268102,
  "roughness": 65.80986523307459,
  "hardness": 69.15425985714043,
  "depth": 50.44467089316521,
  "booming": 28.083507135400176,
  "reverberation": 85.32240939720653,
  "sharpness": 59.12628795613269,
  "warmth": 41.50808384499096,
  "content_enjoyment": 7.213399887084961,
  "content_usefulness": 7.892210006713867,
  "production_complexity": 6.462733268737793,
  "production_quality": 8.014187812805176,
  "danceability": 0.9876527190208435,
  "atonality": 0.039693862199783325,
  "music_flamingo_full": "This track is an energetic, hypnotic blend of Tech House and Tribal House, merging the steady four-on-the-floor drive of tech-house with the organic, percussive flair of tribal rhythms. Production is polished and high-fidelity, built around a crisp electronic drum kit that delivers a relentless kick, off-beat hi-hats, and syncopated tribal percussion elements that are panned across a wide stereo field for depth and movement. A deep, resonant synth bassline anchors the groove, while atmospheric pads and subtle melodic synth stabs add texture without detracting from the rhythmic focus. The mix emphasizes tight low-end compression on the drums and bass, with bright, clean highs on the percussive accents, creating a club-ready, danceable soundscape. The duration of the piece is 30.00 seconds.\nThere are no vocal parts, so the track relies entirely on its instrumental momentum. The structure is loop-based, typical of club music: an introductory groove establishes the main rhythm, followed by incremental layering of percussion and synth elements that build tension, then periodic subtraction of layers that release that tension before the cycle repeats. This additive/subtractive arrangement keeps the energy flowing and sustains a hypnot",
  "music_flamingo_technical": "This track is a high-energy Hardstyle piece that fuses the pounding, distorted kick-driven foundation of classic Hardstyle with the soaring, detuned lead synths typical of Trance.  It sits at 120 BPM and is rooted in E flat minor, delivering a relentless, driving pulse throughout. The duration of the piece is 30.00 seconds.\nInstrumentation & production: The arrangement is built around a massive, over-compressed four-on-the-floor kick that dominates the low end, paired with a deep, tightly side-chained synth bass that locks",
  "music_flamingo_genre_mood": "This track is an energetic Eurodance piece that leans heavily into trance-style synth work, creating a high-energy, euphoric club atmosphere. The production is polished and electronic, with a driving four-on-the-floor beat that propels the dancefloor forward. The duration of the piece is 30.00 seconds.\nTempo & Key - The song runs at 120 BPM and is centered in E flat minor.\nInstrumentation & Production - The arrangement is built around layered synthesizers: bright lead synths carry the melodic hooks, while lush pads and a punchy",
  "music_flamingo_instrumentation": "The track features a driving four-on-the-floor beat, classic drum machine sounds, deep synth basslines, and atmospheric synth pads, creating a vibrant electronic dance music soundscape.",
  "music_flamingo_structure": "The track is built on a repetitive, four-on-the-floor electronic beat that maintains a constant high energy throughout its duration. It opens with a short intro that establishes the driving rhythm before moving into a series of looping sections that feature the same chord progressions and melodic motifs. These sections are connected by subtle dynamic shifts and occasional percussive accents, creating a sense of forward momentum without dramatic breaks or contrasting bridges. The arrangement relies on layering and filtering of synth and drum elements to add texture and variation, culminating in a seamless, club-ready finale.",
  "music_flamingo_model": "gguf_Q8_0",
  "musicbrainz_genres": ["New Beat", "Techno", "EBM"],
  "electronic_main_genre_code": "NEW_BEAT"
}

def load_statistics(filepath):
    """Load feature statistics from file."""
    try:
        with open(filepath, 'r') as f:
            stats = json.load(f)
        summary = "Feature Statistics (Mean / Median):\n"
        for feature in ['bpm', 'lufs', 'danceability', 'brightness', 'roughness', 'warmth', 'sharpness', 'booming', 'depth', 'reverberation', 'hardness', 'atonality']:
            if feature in stats:
                data = stats[feature]
                if 'mean' in data and 'median' in data:
                    summary += f"- {feature}: Mean={data['mean']:.2f}, Median={data['median']:.2f}\n"
        return summary
    except Exception as e:
        print(f"Warning: Could not load statistics: {e}")
        return ""

def format_single_caption_prompt(template_type, original_caption, genres, field_name, stats_summary):
    """Format prompt to revise a single caption."""
    genres_str = ", ".join(genres)
    
    instruction = f"""Revise this music caption to use the correct genres: {genres_str}
The original caption may have incorrect genre labels. Keep the revised caption between 10-30 words.
Do not reference numerical values directly. Output ONLY the revised caption, nothing else.

For context, here are the dataset statistics to help interpret audio features:
{stats_summary}

Original caption ({field_name}):
{original_caption}

Revised caption:"""
    
    if template_type == "qwen3":
        # Qwen3: /no_think disables chain-of-thought reasoning
        return f"/no_think\n{instruction}"
    elif template_type == "granite":
        # Granite 4.0: Standard instruction model, no thinking mode
        return f"<|start_of_role|>system<|end_of_role|>You revise music captions with correct genres. Output only the revised caption.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{instruction}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
    elif template_type == "chatml":
        # GPT-OSS: Use "Reasoning: low" in system prompt to minimize reasoning output
        return f"<|im_start|>system\nReasoning: low\nYou revise music captions with correct genres. Output only the revised caption, no explanations.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    else:
        return instruction

def strip_thinking_traces(text):
    """Remove thinking/reasoning traces from LLM output."""
    import re
    
    # Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove </think> tags that might be orphaned
    text = re.sub(r'</think>', '', text)
    
    # Remove lines that look like reasoning (start with "We need to", "The content", etc.)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line_lower = line.strip().lower()
        # Skip reasoning lines
        if any(line_lower.startswith(p) for p in [
            'we need to', 'we have to', 'we must', 'we should',
            'the content', 'the original', 'the user', 'the task',
            '</analysis>', '<analysis>', '...', 'so we', 'that matches'
        ]):
            continue
        # Skip lines that are mostly dots or ellipsis
        if line.strip().replace('.', '').replace(' ', '') == '':
            continue
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines).strip()
    
    # If text starts with common prefixes, try to extract just the caption
    # Look for patterns like "New Beat, ..." at the start
    if text.startswith(('New', 'Techno', 'This track', 'The track', 'A ', 'An ')):
        # Good, starts with expected content
        pass
    else:
        # Try to find the actual caption after noise
        match = re.search(r'(This track[^.]*\.|The track[^.]*\.)', text)
        if match:
            text = match.group(1)
    
    return text.strip()

def cleanup_vram():
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("  [VRAM cleaned]")

def run_inference():
    results = INPUT_DATA.copy()
    genres = INPUT_DATA.get("genres", []) + INPUT_DATA.get("musicbrainz_genres", [])
    genres = list(set(genres))  # Dedupe
    
    # Load statistics for context
    stats_summary = load_statistics("/home/kim/Projects/mir/statistics.txt")
    
    for model_name, config in MODELS.items():
        path = config["path"]
        template = config["template"]
        
        print(f"\n{'='*40}\nModel: {model_name} (template: {template})\n{'='*40}")
        
        if not os.path.exists(path):
            print(f"Error: Model not found at {path}")
            continue
            
        try:
            llm = Llama(
                model_path=path,
                n_gpu_layers=-1,
                n_ctx=4096,
                flash_attn=True,
                verbose=False
            )
            
            # Determine stop tokens
            if template == "granite":
                stop_tokens = ["<|end_of_text|>", "<|start_of_role|>", "\n\n"]
            elif template == "chatml":
                stop_tokens = ["<|im_end|>", "<|im_start|>", "\n\n"]
            elif template == "qwen3":
                stop_tokens = ["</" + "s>", "\n\n"]
            else:
                stop_tokens = ["</" + "s>", "\n\n"]
            
            # Process each caption field
            for field in CAPTION_FIELDS:
                if field not in INPUT_DATA:
                    continue
                    
                original_caption = INPUT_DATA[field]
                prompt = format_single_caption_prompt(template, original_caption, genres, field, stats_summary)
                
                output = llm(
                    prompt,
                    max_tokens=100,
                    stop=stop_tokens,
                    echo=False,
                    temperature=0.7
                )
                
                raw_output = output['choices'][0]['text'].strip()
                # Clean thinking traces and normalize text
                cleaned_output = strip_thinking_traces(raw_output)
                cleaned_output = normalize_music_flamingo_text(cleaned_output)
                
                # Store with structured key: {model}_{field}
                result_key = f"{model_name}_{field}"
                results[result_key] = cleaned_output
                
                print(f"  {field}: {cleaned_output[:80]}...")
            
            del llm
            cleanup_vram()
            
        except Exception as e:
            print(f"Error running {model_name}: {e}")
            results[f"{model_name}_error"] = str(e)

    print("\n\n" + "="*40)
    print("FINAL RESULTS:")
    print("="*40)
    print(json.dumps(results, indent=2))
    
    with open("poc_lmm_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_inference()

