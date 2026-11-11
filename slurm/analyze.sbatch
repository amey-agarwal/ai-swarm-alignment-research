#!/usr/bin/env bash
# Report figures from finished runs: CPU only, no GPU, no model, no container (non-inference;
# `swarm report` runs safety pass 1 first). Chain it after the run jobs, e.g.:
#   RUNS=$(sbatch --parsable slurm/sweep_array.sbatch configs/grid_core.yaml)
#   sbatch --dependency=afterany:${RUNS} slurm/analyze.sbatch
# Download ${SWARM_REPORT}/report_figures.zip afterwards (Open OnDemand > Files).
#
# Required environment (set before sbatch):
#   SWARM_REPO     path to this repository
#   SWARM_RUNS     run output root (the same one the run jobs wrote to)
#   SWARM_REPORT   output directory for the report, OUTSIDE the repo (e.g. project storage)
#   SWARM_VENV     a Python venv with: pip install -e "$SWARM_REPO[analysis]"
# Optional:
#   SWARM_AGREEMENT  metrics.json from `swarm judge --validate` (enables judge-based figures)
#   SWARM_TRACK      cluster (default: open-weight PACE runs) | api (OpenRouter runs; point SWARM_RUNS
#                    at SWARM_RUNS_API). The two tracks are never pooled: run this job once per track
#                    with a different SWARM_REPORT.
#SBATCH --job-name=swarmguard-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
# Mail goes to the submitting user unless --mail-user is given (slurm/pipeline.sh sets it).
#SBATCH --mail-type=END,FAIL
# PACE charge account / QOS: Slurm reads them from the environment, so nothing personal is
# committed here. Before submitting:
#   export SBATCH_ACCOUNT=gts-<pi> SBATCH_QOS=inferno      # or SBATCH_QOS=embers (free, preemptible)
set -euo pipefail
: "${SLURM_JOB_ID:?must run inside a Slurm job (never on a login node)}"
: "${SWARM_REPO:?set SWARM_REPO}" "${SWARM_RUNS:?set SWARM_RUNS}" "${SWARM_REPORT:?set SWARM_REPORT}"
: "${SWARM_VENV:?set SWARM_VENV to a venv with the analysis extra installed}"
source "${SWARM_REPO:?set SWARM_REPO}/slurm/submit_guard.sh" && refuse_submit_inside_repo

TRACK="${SWARM_TRACK:-cluster}"
[[ "${TRACK}" == "cluster" || "${TRACK}" == "api" ]] || { echo "SWARM_TRACK must be cluster or api"; exit 1; }
agreement=()
if [[ -n "${SWARM_AGREEMENT:-}" ]]; then
    agreement=(--agreement "${SWARM_AGREEMENT}")
fi

# Minimal environment: pass 1 refuses secret-like variable names, and nothing else is needed.
cd "${SWARM_REPO}"
env -i PATH="${SWARM_VENV}/bin:/usr/bin:/bin" HOME="${HOME}" PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}" \
    "${SWARM_VENV}/bin/swarm" report "${SWARM_RUNS}" --out "${SWARM_REPORT}" --track "${TRACK}" \
    ${agreement[@]+"${agreement[@]}"}
# Quick summary table next to the figures (same track).
env -i PATH="${SWARM_VENV}/bin:/usr/bin:/bin" HOME="${HOME}" PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}" \
    "${SWARM_VENV}/bin/swarm" analyze "${SWARM_RUNS}" --track "$([[ ${TRACK} == api ]] && echo api || echo main)" \
    --out "${SWARM_REPORT}/analyze" ${agreement[@]+"${agreement[@]}"} || true
