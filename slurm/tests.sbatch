#!/usr/bin/env bash
# Stage 1 of slurm/pipeline.sh: the repo's own tests and the dev-mode safety doctor on a CPU node
# (no GPU, no model, no network). A failure here stops the pipeline before any GPU time is used.
#
# Required environment (set before sbatch):
#   SWARM_REPO   path to this repository
#   SWARM_VENV   venv with: pip install -e "$SWARM_REPO[dev,analysis]"
#SBATCH --job-name=swarmguard-tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%x-%j.out
# PACE charge account / QOS: Slurm reads them from the environment, so nothing personal is
# committed here. Before submitting:
#   export SBATCH_ACCOUNT=gts-<pi> SBATCH_QOS=inferno      # or SBATCH_QOS=embers (free, preemptible)
set -euo pipefail
: "${SLURM_JOB_ID:?must run inside a Slurm job (never on a login node)}"
: "${SWARM_REPO:?set SWARM_REPO}" "${SWARM_VENV:?set SWARM_VENV}"
source "${SWARM_REPO:?set SWARM_REPO}/slurm/submit_guard.sh" && refuse_submit_inside_repo
cd "${SWARM_REPO}"

# Minimal environment: pass 1 refuses secret-like variable names.
run() {
    env -i PATH="${SWARM_VENV}/bin:/usr/bin:/bin" HOME="${HOME}" PYTHONDONTWRITEBYTECODE=1 \
        TMPDIR="${TMPDIR:-/tmp}" MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}" "$@"
}
run "${SWARM_VENV}/bin/python" -m pytest -q
run "${SWARM_VENV}/bin/swarm" doctor --mode dev
echo "TESTS: PASS"
