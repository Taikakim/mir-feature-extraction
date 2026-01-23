#!/usr/bin/env python3
"""
Extract All Features Script

Wrapper for the master batch processing pipeline.
Ensures all features, including metadata, are extracted.

Usage:
    python scripts/extract_all_features.py /path/to/audio [options]
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    # Pass all arguments through to batch_process.py
    cmd = [sys.executable, str(PROJECT_ROOT / "src" / "batch_process.py")] + sys.argv[1:]
    
    # Ensure metadata is included if features are specified, or default to all
    # Actually batch_process.py handles defaults relative to config.
    # We just act as a convenient alias.
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(130)

if __name__ == "__main__":
    main()
