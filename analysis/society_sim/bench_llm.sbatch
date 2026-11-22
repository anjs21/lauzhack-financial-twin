#!/bin/bash
#SBATCH --job-name=twin-llm-bench
#SBATCH --partition=frida
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
# One B200, one small model at a time. Usage: sbatch bench_llm.sbatch Qwen/Qwen3-1.7B 4096
set -euo pipefail
export HF_HUB_OFFLINE=1 TMPDIR=$HOME/.cache/tmp VLLM_LOGGING_LEVEL=WARNING
# FlashInfer JIT-compiles kernels on first use: it needs ninja and nvcc from the venv on PATH
VENV=$HOME/envs/lauzhack-sim
export CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0   # use the PyTorch sampler instead of a JIT-built one
cd "$SLURM_SUBMIT_DIR"   # submit from analysis/society_sim
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
for model in "$@"; do
  ~/envs/lauzhack-sim/bin/python bench_llm.py "$model" 4096
done
