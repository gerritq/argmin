#!/bin/bash
#SBATCH --job-name=coe_horizontal
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15GB
# SBATCH --constraint=a100

set -euo pipefail

MODEL_ID="Qwen/Qwen3-0.6B" # Qwen/Qwen3-8B
SMOKE_TEST=1
N=100

uv run main.py \
    --model-id "$MODEL_ID" \
    --smoke-test "$SMOKE_TEST" \
    --n "$N"
