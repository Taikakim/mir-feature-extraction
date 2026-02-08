
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing.bs_roformer_sep import discover_model_path

MODELS_DIR = Path("/home/kim/Projects/mir/models/bs-roformer")
MODEL_NAME = "jarredou-BS-ROFO-SW-Fixed-drums"

def test_discovery():
    print(f"Testing model discovery for: {MODEL_NAME}")
    print(f"Models dir: {MODELS_DIR}")
    
    result = discover_model_path(MODELS_DIR, MODEL_NAME)
    
    if result:
        print("\nSUCCESS: Model found!")
        print(f"  Path: {result['path']}")
        print(f"  Config: {result['config']}")
        print(f"  Checkpoint: {result['checkpoint']}")
    else:
        print("\nFAILURE: Model not found.")
        
        # Debug info
        search_path = MODELS_DIR / MODEL_NAME
        print(f"\nDebug info for {search_path}:")
        if search_path.exists():
            print("  Directory exists.")
            print("  Files:")
            for f in search_path.iterdir():
                print(f"   - {f.name}")
        else:
            print("  Directory does not exist.")

if __name__ == "__main__":
    test_discovery()
