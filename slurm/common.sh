#!/usr/bin/env bash
# Shared launcher for swarmguard Slurm jobs (sourced by run_single.sbatch, sweep_array.sbatch,
# interactive.sh). Runs ON THE COMPUTE NODE, inside a Slurm allocation, never on a login node.
#
# Required environment (set before sbatch/salloc):
#   SWARM_REPO     path to this repository (bound read-only at /repo)
#   SWARM_SIF      path to the built container image (apptainer build swarm.sif container/swarm.def)
#   SWARM_WEIGHTS  directory holding pre-downloaded weights as <hf_id>/ (bound read-only at /weights)
#   SWARM_RUNS     output root for run directories (bound read-write at /runs)
set -euo pipefail

: "${SLURM_JOB_ID:?must run inside a Slurm job (never on a login node)}"
: "${SWARM_REPO:?set SWARM_REPO to the swarmguard repo}"
: "${SWARM_SIF:?set SWARM_SIF to the container image}"
: "${SWARM_WEIGHTS:?set SWARM_WEIGHTS to the weights directory}"
: "${SWARM_RUNS:?set SWARM_RUNS to the run output root}"

# OS-level network isolation for the container (loopback only). If the cluster does not allow
# unprivileged network namespaces, apptainer refuses to start: do NOT silently drop this flag.
# The only allowed fallback is sandbox.network_isolation: unavailable plus a written
# justification in the config (recorded in every safety report); the pass-2 network probes
# still run and must not reach the outside world.
NET_FLAGS=(--net --network none)

swarm_launch() {
    # usage: swarm_launch <model_key|auto> <run|judge|sweep|shell> [args...]
    local job_tmp port key app rc
    job_tmp="${TMPDIR:-/tmp}/swarmguard-${SLURM_JOB_ID}"
    mkdir -p "${job_tmp}"/{hf,vllm,cache,tmp}
    mkdir -p "${SWARM_RUNS}"
    trap 'rm -rf "${job_tmp}"' EXIT

    # Random free port on 127.0.0.1 and a random per-job API key (never enters a prompt).
    port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
    key="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"

    local cmd=(apptainer exec --nv --containall --cleanenv --no-home "${NET_FLAGS[@]}"
        --bind "${SWARM_REPO}:/repo:ro"
        --bind "${SWARM_WEIGHTS}:/weights:ro"
        --bind "${SWARM_RUNS}:/runs:rw"
        --bind "${job_tmp}:/jobtmp:rw"
        --pwd /repo
        --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
        --env "SWARM_VLLM_PORT=${port}"
        --env "SWARM_VLLM_API_KEY=${key}"
        --env "HF_HOME=/jobtmp/hf"
        --env "VLLM_CACHE_ROOT=/jobtmp/vllm"
        --env "XDG_CACHE_HOME=/jobtmp/cache"
        --env "TRITON_CACHE_DIR=/jobtmp/cache/triton"
        --env "TORCHINDUCTOR_CACHE_DIR=/jobtmp/cache/inductor"
        --env "TMPDIR=/jobtmp/tmp"
        --env "HF_HUB_OFFLINE=1"
        --env "TRANSFORMERS_OFFLINE=1"
        --env "PYTHONDONTWRITEBYTECODE=1"
        --env "PYTHONPATH=/repo/src"
        "${SWARM_SIF}" bash /repo/slurm/container_entry.sh "$@")

    if [[ "${2:-}" == "shell" ]]; then
        # Interactive: the container shell needs the terminal, so run it in the foreground.
        "${cmd[@]}"
        return
    fi
    "${cmd[@]}" &
    app=$!

    # Slurm sends USR1 120 s before the time limit (--signal=B:USR1@120): forward it so the
    # episode checkpoints at the next round boundary and exits cleanly.
    trap 'kill -USR1 "${app}" 2>/dev/null || true' USR1
    trap 'kill -TERM "${app}" 2>/dev/null || true' TERM
    set +e
    wait "${app}"; rc=$?
    while kill -0 "${app}" 2>/dev/null; do wait "${app}"; rc=$?; done
    set -e
    return "${rc}"
}
