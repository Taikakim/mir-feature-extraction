import json
import numpy as np
import os
from pathlib import Path

latent_dir = Path("/run/media/kim/Lehto/goa-small")
source_dir = Path("/run/media/kim/Mantu/ai-music/Goa_Separated_crops")

# Find a track that has INFO files
tracks = [d.name for d in latent_dir.iterdir() if d.is_dir()]
for t in tracks:
    i_files = list((source_dir / t).glob("*.INFO"))
    if i_files:
        print(f"Found INFO in track: {t}")
        print(f"File: {i_files[0]}")
        with open(i_files[0]) as f:
            data = json.load(f)
            print("Keys:", list(data.keys()))
            if 'original_features' in data:
                 print("Features length:", len(data['original_features']))
        break
