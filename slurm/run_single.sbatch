#!/usr/bin/env bash
# One swarmguard episode (or one judge pass) on one A100. Submit from OUTSIDE the repo (the
# Slurm log lands in the submit dir and would break the manifest check), e.g. from $SWARM_RUNS:
#   SWARM_REPO=$PWD SWARM_SIF=... SWARM_WEIGHTS=... SWARM_RUNS=... \
#     sbatch "$SWARM_REPO/slurm/run_single.sbatch" --impostor scripted --claim write --task logic-grid:L4 --seed 0
#   ACTION=judge MODEL_KEY=gemma-3-27b sbatch --gres=gpu:A100:2 "$SWARM_REPO/slurm/run_single.sbatch" <run_id>
#     (Gemma-3-27B bf16 needs ~54 GB: on Phoenix's 40 GB A100s request two GPUs; vLLM uses
#      tensor parallelism across all GPUs Slurm grants. --gres on the command line overrides the header.)
#SBATCH --job-name=swarmguard
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out
# Mail goes to the submitting user unless --mail-user is given (slurm/pipeline.sh sets it).
#SBATCH --mail-type=END,FAIL
# PACE charge account / QOS: Slurm reads them from the environment, so nothing personal is
# committed here. Before submitting:
#   export SBATCH_ACCOUNT=gts-<pi> SBATCH_QOS=inferno      # or SBATCH_QOS=embers (free, preemptible)
set -euo pipefail
source "${SWARM_REPO:?set SWARM_REPO}/slurm/submit_guard.sh" && refuse_submit_inside_repo
source "${SWARM_REPO:?set SWARM_REPO}/slurm/common.sh"

MODEL_KEY="${MODEL_KEY:-llama-3.2-3b}"
ACTION="${ACTION:-run}"
swarm_launch "${MODEL_KEY}" "${ACTION}" "$@"
