#!/usr/bin/env bash
# Submit the whole cluster pipeline in one go, as a chain of dependent Slurm jobs:
#   1 tests   CPU   pytest + dev doctor                       (slurm/tests.sbatch)
#   2 smoke   GPU   one Llama episode; must reach safety 1.0  (slurm/run_single.sbatch)   after 1 succeeds
#   3 grid    GPU   job array over the grid                   (slurm/sweep_array.sbatch)  after 2 succeeds
#   4 report  CPU   figures, tables, zip                      (slurm/analyze.sbatch)      after 3 ends
# Optional stage (SWARM_STAGES=calib,...): calibration pilot (clean runs at every task level).
# Every job emails at END and on FAIL (array: per task). If a stage fails, the stages that need it are
# cancelled at once (--kill-on-invalid-dep) instead of waiting forever. Run on a login node / OnDemand
# shell, from OUTSIDE the repo (Slurm writes its logs into the current directory):
#   cd "$SWARM_RUNS" && bash "$SWARM_REPO/slurm/pipeline.sh" [grid.yaml]
#
# Required environment:
#   SWARM_REPO SWARM_SIF SWARM_WEIGHTS SWARM_RUNS   (as for the other scripts)
#   SWARM_VENV          venv with: pip install -e "$SWARM_REPO[dev,analysis]"
#   SWARM_REPORT        report output directory, outside the repo
#   SWARM_NOTIFY_EMAIL  where Slurm sends the END / FAIL emails
#   SBATCH_ACCOUNT SBATCH_QOS
# Optional:
#   SWARM_ARRAY   array spec for the grid (default 0-11%6: grid_core tasks 0-119, the claim x policy x
#                 contact block; use 0-19%6 for the full 200-task grid)
#   SWARM_STAGES  comma list, default tests,smoke,grid,report (add calib to run the calibration pilot first)
set -euo pipefail
: "${SWARM_REPO:?set SWARM_REPO}" "${SWARM_SIF:?set SWARM_SIF}" "${SWARM_WEIGHTS:?set SWARM_WEIGHTS}"
: "${SWARM_RUNS:?set SWARM_RUNS}" "${SWARM_VENV:?set SWARM_VENV}" "${SWARM_REPORT:?set SWARM_REPORT}"
: "${SWARM_NOTIFY_EMAIL:?set SWARM_NOTIFY_EMAIL}" "${SBATCH_ACCOUNT:?set SBATCH_ACCOUNT}" "${SBATCH_QOS:?set SBATCH_QOS}"
source "${SWARM_REPO}/slurm/submit_guard.sh" && refuse_submit_inside_repo
export SWARM_REPO SWARM_SIF SWARM_WEIGHTS SWARM_RUNS SWARM_VENV SWARM_REPORT

GRID="${1:-configs/grid_core.yaml}"
ARRAY="${SWARM_ARRAY:-0-11%6}"
STAGES=",${SWARM_STAGES:-tests,smoke,grid,report},"
S="${SWARM_REPO}/slurm"
mkdir -p "${SWARM_RUNS}"

MAIL=(--mail-user="${SWARM_NOTIFY_EMAIL}" --mail-type=END,FAIL)
last=""
submit() {  # submit <after-type> <args...>; prints the job id, chained on the previous stage
    local after="$1"; shift
    local dep=()
    [[ -n "${last}" ]] && dep=(--dependency="${after}:${last}" --kill-on-invalid-dep=yes)
    local out
    out=$(sbatch --parsable "${MAIL[@]}" ${dep[@]+"${dep[@]}"} "$@")
    echo "${out%%;*}"
}

ids=()
if [[ "${STAGES}" == *,tests,* ]]; then
    last=$(submit afterok "${S}/tests.sbatch"); ids+=("tests=${last}")
fi
if [[ "${STAGES}" == *,smoke,* ]]; then
    last=$(submit afterok --time=01:00:00 "${S}/run_single.sbatch" \
        --impostor scripted --claim write --task logic-grid:L4 --seed 0); ids+=("smoke=${last}")
fi
if [[ "${STAGES}" == *,calib,* ]]; then
    last=$(submit afterok --array=0-4%5 --mail-type=END,FAIL,ARRAY_TASKS \
        "${S}/sweep_array.sbatch" configs/pilot_calibration.yaml); ids+=("calib=${last}")
fi
if [[ "${STAGES}" == *,grid,* ]]; then
    last=$(submit afterok --array="${ARRAY}" --mail-type=END,FAIL,ARRAY_TASKS \
        "${S}/sweep_array.sbatch" "${GRID}"); ids+=("grid=${last}")
fi
if [[ "${STAGES}" == *,report,* ]]; then
    # afterany: the report runs even if some grid tasks failed (they are excluded and listed in it)
    last=$(submit afterany "${S}/analyze.sbatch"); ids+=("report=${last}")
fi

echo "submitted: ${ids[*]}"
echo "watch:  squeue -u \$USER        cancel all:  scancel ${ids[*]//*=/}"
echo "logs:   ${PWD}/slurm-*.out      report: ${SWARM_REPORT}/report_figures.zip"
