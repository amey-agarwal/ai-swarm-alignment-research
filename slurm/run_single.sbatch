#!/usr/bin/env bash
# One swarmguard episode (or one judge pass) on one A100. Submit from the repo root:
#   SWARM_REPO=$PWD SWARM_SIF=... SWARM_WEIGHTS=... SWARM_RUNS=... \
#     sbatch slurm/run_single.sbatch --impostor scripted --claim write --task logic-grid:L4 --seed 0
#   ACTION=judge MODEL_KEY=gemma-3-27b sbatch --gres=gpu:A100:2 slurm/run_single.sbatch <run_id>
#     (Gemma-3-27B bf16 needs ~54 GB: on Phoenix's 40 GB A100s request two GPUs; vLLM uses
#      tensor parallelism across all GPUs Slurm grants. --gres on the command line overrides the header.)
#SBATCH --job-name=swarmguard
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out
# PACE charge account / QOS: Slurm reads them from the environment, so nothing personal is
# committed here. Before submitting:
#   export SBATCH_ACCOUNT=gts-<pi> SBATCH_QOS=inferno      # or SBATCH_QOS=embers (free, preemptible)
set -euo pipefail
source "${SWARM_REPO:?set SWARM_REPO}/slurm/common.sh"

MODEL_KEY="${MODEL_KEY:-llama-3.2-3b}"
ACTION="${ACTION:-run}"
swarm_launch "${MODEL_KEY}" "${ACTION}" "$@"
