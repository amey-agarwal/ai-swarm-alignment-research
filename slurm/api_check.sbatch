#!/usr/bin/env bash
# API track readiness check (CPU, 5 min, no model call): outbound HTTPS to openrouter.ai from a
# compute node, the key file is private, safety pass 1 + pass 2 in api mode, and current spend.
#   jid=$(sbatch --parsable slurm/api_check.sbatch)
#   sbatch --dependency=afterok:${jid} --array=0-1 slurm/api_sweep.sbatch configs/grid_openrouter.yaml
#
# Required environment (set before sbatch):
#   SWARM_REPO   path to this repository
#   SWARM_VENV   venv with: pip install -e "$SWARM_REPO[api]"
# The key lives in the file named by api.key_file in configs/api_openrouter.yaml
# (default ~/.config/swarmguard/openrouter.key, chmod 600). Never export it as a variable.
#SBATCH --job-name=swarmguard-api-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%x-%j.out
# PACE charge account / QOS: Slurm reads them from the environment, so nothing personal is
# committed here. Before submitting:
#   export SBATCH_ACCOUNT=gts-<pi> SBATCH_QOS=inferno      # or SBATCH_QOS=embers (free, preemptible)
set -euo pipefail
: "${SLURM_JOB_ID:?must run inside a Slurm job (never on a login node)}"
: "${SWARM_REPO:?set SWARM_REPO}" "${SWARM_VENV:?set SWARM_VENV}"
CONFIG="${SWARM_API_CONFIG:-configs/api_openrouter.yaml}"
cd "${SWARM_REPO}"

echo "node: $(hostname)"
getent hosts openrouter.ai >/dev/null || { echo "FAIL: DNS for openrouter.ai"; exit 1; }
code=$(curl -sS -o /dev/null --max-time 10 -w "%{http_code}" https://openrouter.ai/api/v1/models) \
    || { echo "FAIL: no outbound HTTPS from this node"; exit 1; }
echo "outbound HTTPS to openrouter.ai: HTTP ${code}"

# Minimal environment for every swarm command: pass 1 refuses secret-like variable names.
run_swarm() {
    env -i PATH="${SWARM_VENV}/bin:/usr/bin:/bin" HOME="${HOME}" PYTHONDONTWRITEBYTECODE=1 \
        SLURM_JOB_ID="${SLURM_JOB_ID}" MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}" \
        "${SWARM_VENV}/bin/swarm" "$@"
}
run_swarm doctor --mode api --config "${CONFIG}"     # includes key-file and spend checks
set +e
run_swarm spend --config "${CONFIG}"
rc=$?
set -e
case ${rc} in
    0) echo "spend OK" ;;
    10) echo "spend at/above the warning level (runs can still start)" ;;
    *) echo "FAIL: spend at the hard cap or unknown (exit ${rc})"; exit 1 ;;
esac
echo "API CHECK: PASS"
