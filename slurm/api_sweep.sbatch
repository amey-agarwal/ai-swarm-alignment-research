#!/usr/bin/env bash
# API track sweep (OpenRouter): CPU-only job array, no GPU, no container (the container has no
# network). Each array task runs TASKS_PER_JOB consecutive grid tasks; before each one it checks
# the key's authoritative spend:
#   >= api.usd_warn ($25): Slurm emails you once (a mail-only job), runs continue
#   >= api.usd_hard ($50): this task stops starting episodes (the backend also refuses requests)
# Size the array from the grid (20 tasks / 10 per job = 2):
#   N=$(swarm sweep configs/grid_openrouter.yaml --count)
#   sbatch --array=0-$(( (N + 9) / 10 - 1 ))%2 slurm/api_sweep.sbatch configs/grid_openrouter.yaml
# Pilot first (one episode):  TASKS_PER_JOB=1 sbatch --array=0 slurm/api_sweep.sbatch configs/grid_openrouter.yaml
#
# Required environment (set before sbatch):
#   SWARM_REPO          path to this repository
#   SWARM_VENV          venv with: pip install -e "$SWARM_REPO[api]"
#   SWARM_RUNS_API      output root for API-track run directories (keep separate from cluster runs)
#   SWARM_NOTIFY_EMAIL  where the $25 / $50 spend emails go
#SBATCH --job-name=swarmguard-api
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm-%x-%A_%a.out
# PACE charge account / QOS: Slurm reads them from the environment, so nothing personal is
# committed here. Before submitting:
#   export SBATCH_ACCOUNT=gts-<pi> SBATCH_QOS=inferno      # or SBATCH_QOS=embers (free, preemptible)
set -euo pipefail
: "${SLURM_JOB_ID:?must run inside a Slurm job (never on a login node)}"
: "${SWARM_REPO:?set SWARM_REPO}" "${SWARM_VENV:?set SWARM_VENV}" "${SWARM_RUNS_API:?set SWARM_RUNS_API}"
: "${SWARM_NOTIFY_EMAIL:?set SWARM_NOTIFY_EMAIL for spend alerts}"
GRID="${1:-configs/grid_openrouter.yaml}"
CONFIG="${SWARM_API_CONFIG:-configs/api_openrouter.yaml}"
PER_JOB="${TASKS_PER_JOB:-10}"
START=$(( ${SLURM_ARRAY_TASK_ID:?must run as an array task} * PER_JOB ))
cd "${SWARM_REPO}"
mkdir -p "${SWARM_RUNS_API}"

run_swarm() {
    env -i PATH="${SWARM_VENV}/bin:/usr/bin:/bin" HOME="${HOME}" PYTHONDONTWRITEBYTECODE=1 \
        SLURM_JOB_ID="${SLURM_JOB_ID}" MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}" \
        "${SWARM_VENV}/bin/swarm" "$@"
}

# One email per threshold across all array tasks: mkdir is atomic, so only one task sends it.
notify_once() {  # $1 = marker name, $2 = subject
    if mkdir "${SWARM_RUNS_API}/.${1}" 2>/dev/null; then
        sbatch --quiet -J "$2" -t 00:01:00 --mem=100M -o /dev/null \
            --mail-type=BEGIN --mail-user="${SWARM_NOTIFY_EMAIL}" --wrap "true" \
            || echo "WARNING: could not submit the spend-alert mail job"
    fi
}

STOP=0
trap 'STOP=1' USR1 TERM
TOTAL=$(run_swarm sweep "${GRID}" --count)
for (( i = START; i < START + PER_JOB && i < TOTAL; i++ )); do
    [[ ${STOP} -eq 1 ]] && { echo "signal received: stopping before task ${i}"; break; }
    set +e
    msg=$(run_swarm spend --config "${CONFIG}")
    rc=$?
    set -e
    echo "${msg}"
    if [[ ${rc} -eq 10 ]]; then
        notify_once spend_warned "swarmguard: OpenRouter spend reached warning level"
    elif [[ ${rc} -ne 0 ]]; then
        notify_once spend_capped "swarmguard: OpenRouter spend HARD CAP reached - runs stopped"
        echo "spend at the hard cap or unknown (exit ${rc}): not starting task ${i}"
        break
    fi
    set +e
    run_swarm sweep "${GRID}" --index "${i}" --mode api --backend openrouter --config "${CONFIG}" \
        --out "${SWARM_RUNS_API}"
    echo "task ${i}: exit $?"
    set -e
done
