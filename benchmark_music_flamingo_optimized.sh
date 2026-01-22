#!/bin/bash
# Music Flamingo Benchmark with Full ROCm Optimizations
# Adapted from ComfyUI settings for RDNA 4 / RX 9070 XT

# --- Flash Attention 2 (Triton Backend) ---
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

# --- Performance Tuning (GEMMs & Kernels) ---
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_VERBOSE=1

export OMP_NUM_THREADS=8
# General CPU parallelism

# --- Memory Management ---
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export PYTORCH_HIP_FREE_MEMORY_THRESHOLD_MB=256

# --- Prevent AMD performance bug (CPU/GPU sync) ---
export HIP_FORCE_DEV_KERNARG=1

# --- Device Selection ---
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

echo "========================================================"
echo "Music Flamingo Benchmark - Optimized for AMD ROCm"
echo "========================================================"
echo "Flash Attention 2:     $FLASH_ATTENTION_TRITON_AMD_ENABLE"
echo "Tunable Ops:           $PYTORCH_TUNABLEOP_ENABLED"
echo "OMP Threads:           $OMP_NUM_THREADS"
echo "Memory Config:         $PYTORCH_HIP_ALLOC_CONF"
echo "========================================================"
echo ""

python3 src/benchmark_music_flamingo.py "$@" --modes flash
