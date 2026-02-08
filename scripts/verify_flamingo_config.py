
import sys
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.master_pipeline import MasterPipelineConfig
from classification.music_flamingo import MusicFlamingoGGUF

def test_config_loading():
    print("Testing config loading...")
    config_path = PROJECT_ROOT / "config" / "master_pipeline.yaml"
    
    try:
        config = MasterPipelineConfig.from_yaml(config_path)
        print("✓ Config loaded successfully")
        
        prompts = config.flamingo_prompts
        print(f"Prompts type: {type(prompts)}")
        
        if not isinstance(prompts, dict):
            print("❌ Error: flamingo_prompts is not a dict")
            return False
            
        if not prompts:
             print("❌ Error: flamingo_prompts is empty")
             return False

        print(f"✓ Prompts found: {list(prompts.keys())}")
        
        # Check first prompt value is string
        first_key = list(prompts.keys())[0]
        if not isinstance(prompts[first_key], str):
            print(f"❌ Error: prompt value is not string: {type(prompts[first_key])}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analyzer_interface():
    print("\nTesting analyzer interface...")
    try:
        # Just check if we can instantiate and call the method (mocking the actual inference)
        # We don't want to actually load the massive model or run inference
        
        # Mock the cli path and model dir check to avoid FileNotFoundError if models aren't present
        # But we can't easily mock __init__ without import patching. 
        # Instead, let's just inspect the method signature or docstring? 
        # Or better, just rely on valid python syntax check which we passed if we imported it.
        
        import inspect
        sig = inspect.signature(MusicFlamingoGGUF.analyze_all_prompts)
        print(f"analyze_all_prompts signature: {sig}")
        
        # Check if 'prompts' argument exists
        if 'prompts' not in sig.parameters:
            print("❌ Error: 'prompts' argument missing in analyze_all_prompts")
            return False
            
        print("✓ analyze_all_prompts has 'prompts' argument")
        return True
        
    except Exception as e:
        print(f"❌ Failed to inspect analyzer: {e}")
        return False

if __name__ == "__main__":
    if test_config_loading() and test_analyzer_interface():
        print("\n✅ VERIFICATION PASSED")
        sys.exit(0)
    else:
        print("\n❌ VERIFICATION FAILED")
        sys.exit(1)
