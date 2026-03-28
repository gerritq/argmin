#!/bin/bash
#SBATCH --job-name=argmin_100
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=25GB
#SBATCH --constraint=h200|b200

nvidia-smi

set -euo pipefail

MODEL_ID="Qwen/Qwen3-4B" # Qwen/Qwen3-8B Qwen/Qwen3-0.6B Qwen/Qwen3-4B
SMOKE_TEST=0
N=3000
START_N=2500
MODE="single"

uv run main.py \
    --model-id "$MODEL_ID" \
    --smoke-test "$SMOKE_TEST" \
    --n "$N" \
    --start-n "$START_N" \
    --low-level-mode "$MODE"
