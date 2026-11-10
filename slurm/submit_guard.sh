#!/usr/bin/env bash
# Sourced by every sbatch script. Slurm writes the job log (slurm-*.out) into the directory the job
# was SUBMITTED from. If that is inside the repo, the new file fails the MANIFEST.json integrity check
# (safety pass 1) and the job would abort anyway, so stop at once with a clear message.
# Submit from outside the repo instead, e.g.:
#   cd "$SWARM_RUNS" && sbatch "$SWARM_REPO/slurm/run_single.sbatch" ...

refuse_submit_inside_repo() {
    local sub repo
    sub="$(cd "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
    repo="$(cd "${SWARM_REPO:?set SWARM_REPO}" && pwd -P)"
    if [[ "${sub}/" == "${repo}/"* ]]; then
        echo "ERROR: this job was submitted from inside the repo (${sub})." >&2
        echo "Its Slurm log file there breaks the MANIFEST.json integrity check, so no job can pass." >&2
        echo "Fix: rm ${sub}/slurm-*.out, then resubmit from outside the repo:" >&2
        echo "  cd \"\$SWARM_RUNS\" && sbatch \"\$SWARM_REPO/slurm/<script>\" ..." >&2
        exit 1
    fi
}
