#!/usr/bin/env bash
# Interactive GPU session for the live impostor: allocates one A100, starts vLLM inside the
# container, and opens a shell there. Run from a login node:
#   SWARM_REPO=$PWD SWARM_SIF=... SWARM_WEIGHTS=... SWARM_RUNS=... bash slurm/interactive.sh [model_key]
# Then, inside the container shell:
#   swarm run --impostor live --mode cluster --backend vllm -m llama-3.2-3b --out /runs
# The run prints pass 1 and pass 2 and waits for you to type CONFIRM.
#
# PACE charge account / QOS come from the environment (salloc reads them):
#   export SALLOC_ACCOUNT=gts-<pi> SALLOC_QOS=inferno
# For the judge (gemma-3-27b, ~54 GB in bf16) ask for two 40 GB A100s:
#   SWARM_GPUS=2 bash slurm/interactive.sh gemma-3-27b
# (This also works from a terminal in PACE's Open OnDemand portal.)
set -euo pipefail
: "${SWARM_REPO:?set SWARM_REPO}" "${SWARM_SIF:?}" "${SWARM_WEIGHTS:?}" "${SWARM_RUNS:?}"
MODEL_KEY="${1:-llama-3.2-3b}"

exec salloc --gres="gpu:A100:${SWARM_GPUS:-1}" --time=02:00:00 --signal=B:USR1@120 --cpus-per-task=8 --mem=64G \
    srun --pty bash -c 'source "${SWARM_REPO}/slurm/common.sh" && swarm_launch "$0" shell' "${MODEL_KEY}"
