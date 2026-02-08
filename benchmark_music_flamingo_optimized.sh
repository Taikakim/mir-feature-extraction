#!/bin/bash
# Music Flamingo Benchmark with Full ROCm Optimizations
# ROCm env vars are set by src/core/rocm_env.py automatically.
# Shell exports here override those defaults if needed.

echo "========================================================"
echo "Music Flamingo Benchmark - AMD ROCm"
echo "========================================================"
echo "ROCm env: handled by src/core/rocm_env.py"
echo "========================================================"
echo ""

python3 src/benchmark_music_flamingo.py "$@"
