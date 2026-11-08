#!/usr/bin/env bash
# Job array over a grid: each array task starts one vLLM server and runs TASKS_PER_JOB
# consecutive grid tasks (cell x seed). Size the array from the grid first:
#   N=$(swarm sweep configs/grid_core.yaml --count)            # 200 for grid_core.yaml
#   sbatch --array=0-$(( (N + 9) / 10 - 1 ))%4 slurm/sweep_array.sbatch configs/grid_core.yaml
# (the default --array below matches grid_core.yaml with TASKS_PER_JOB=10, at most 4 at a time)
#SBATCH --job-name=swarmguard-sweep
#SBATCH --array=0-19%4
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%A_%a.out
# PACE charge account / QOS: Slurm reads them from the environment, so nothing personal is
# committed here. Before submitting:
#   export SBATCH_ACCOUNT=gts-<pi> SBATCH_QOS=inferno      # or SBATCH_QOS=embers (free, preemptible)
set -euo pipefail
source "${SWARM_REPO:?set SWARM_REPO}/slurm/common.sh"

GRID="${1:-configs/grid_core.yaml}"
PER_JOB="${TASKS_PER_JOB:-10}"
START=$(( ${SLURM_ARRAY_TASK_ID:?must run as an array task} * PER_JOB ))
swarm_launch auto sweep "${GRID}" "${START}" "${PER_JOB}"
