#!/usr/bin/env bash
# One swarmguard episode (or one judge pass) on one A100. Submit from the repo root:
#   SWARM_REPO=$PWD SWARM_SIF=... SWARM_WEIGHTS=... SWARM_RUNS=... \
#     sbatch slurm/run_single.sbatch --impostor scripted --claim write --task logic-grid:L4 --seed 0
#   ACTION=judge MODEL_KEY=gemma-3-27b sbatch slurm/run_single.sbatch <run_id>   (judge needs an 80 GB A100)
#SBATCH --job-name=swarmguard
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out
# PACE: set your charge account and QOS, e.g.
##SBATCH --account=gts-<pi>
##SBATCH --qos=inferno
set -euo pipefail
source "${SWARM_REPO:?set SWARM_REPO}/slurm/common.sh"

MODEL_KEY="${MODEL_KEY:-llama-3.2-3b}"
ACTION="${ACTION:-run}"
swarm_launch "${MODEL_KEY}" "${ACTION}" "$@"
