#!/bin/bash
#SBATCH --job-name=argmin_100
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=25GB
# SBATCH --constraint=a100

nvidia-smi

set -euo pipefail

MODEL_ID="Qwen/Qwen3-4B" # Qwen/Qwen3-8B Qwen/Qwen3-0.6B
SMOKE_TEST=0
N=5

uv run main.py \
    --model-id "$MODEL_ID" \
    --smoke-test "$SMOKE_TEST" \
    --n "$N"
