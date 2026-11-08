#!/usr/bin/env bash
# Runs INSIDE the Apptainer container (started by slurm/common.sh). Starts one vLLM server on
# 127.0.0.1 serving one model, runs the requested swarm command against it, and always stops
# the server afterwards. Every swarm command runs its own safety pass 1 + pass 2 and postflight.
#
# usage: container_entry.sh <model_key|auto> run   [swarm run options...]
#        container_entry.sh <model_key>      judge <run_id>
#        container_entry.sh auto             sweep <grid.yaml> <start_index> <count>
#        container_entry.sh <model_key>      shell
set -euo pipefail

MODEL_KEY="${1:?model key}"; shift
ACTION="${1:?action}"; shift
: "${SWARM_VLLM_PORT:?}" "${SWARM_VLLM_API_KEY:?}" "${SLURM_JOB_ID:?}"

# The code that runs must be the read-only, manifest-checked repo, not a copy in the image.
python -c 'import sys, swarmguard; sys.exit(0 if swarmguard.__file__.startswith("/repo/src/") else "swarmguard must be imported from /repo/src")'

if [[ "${ACTION}" == "sweep" && "${MODEL_KEY}" == "auto" ]]; then
    MODEL_KEY="$(python - "$1" <<'EOF'
import sys
from swarmguard.sweep import cells, load_grid
models = {c.get("model", "llama-3.2-3b") for c in cells(load_grid(sys.argv[1]))}
if len(models) != 1:
    sys.exit(f"one vLLM server serves one model; grid uses {sorted(models)}")
print(models.pop())
EOF
)"
fi
HF_ID="$(python -c 'import sys; from swarmguard.config import load_models; print(load_models()[sys.argv[1]].hf_id)' "${MODEL_KEY}")"
WEIGHTS="/weights/${HF_ID}"
[[ -d "${WEIGHTS}" ]] || { echo "weights not found: ${WEIGHTS} (download them before the job)" >&2; exit 1; }

vllm serve "${WEIGHTS}" \
    --host 127.0.0.1 --port "${SWARM_VLLM_PORT}" --api-key "${SWARM_VLLM_API_KEY}" \
    --served-model-name "${HF_ID}" --dtype bfloat16 \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-32}" --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
    --enable-prefix-caching --generation-config vllm --disable-log-requests &
VLLM_PID=$!
cleanup() { kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT

python - <<'EOF'
import os, sys, time, urllib.request
url = f"http://127.0.0.1:{os.environ['SWARM_VLLM_PORT']}/health"
deadline = time.time() + 900
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status == 200:
                sys.exit(0)
    except OSError:
        pass
    time.sleep(5)
sys.exit("vLLM did not become healthy within 15 minutes")
EOF

STOPPING=0
run_forwarding() {
    "$@" &
    local child=$! rc
    trap 'STOPPING=1; kill -USR1 "${child}" 2>/dev/null || true' USR1
    trap 'STOPPING=1; kill -TERM "${child}" 2>/dev/null || true' TERM
    set +e
    wait "${child}"; rc=$?
    while kill -0 "${child}" 2>/dev/null; do wait "${child}"; rc=$?; done
    set -e
    return "${rc}"
}

case "${ACTION}" in
    run)
        run_forwarding swarm run --mode cluster --backend vllm -m "${MODEL_KEY}" --out /runs "$@" ;;
    judge)
        run_forwarding swarm judge "${1:?run_id}" --mode cluster --backend vllm --out /runs ;;
    sweep)
        GRID="${1:?grid}"; START="${2:?start}"; COUNT="${3:?count}"
        TOTAL="$(swarm sweep "${GRID}" --count)"
        rc=0
        for ((i = START; i < START + COUNT && i < TOTAL; i++)); do
            run_forwarding swarm sweep "${GRID}" --index "${i}" --mode cluster --backend vllm --out /runs || rc=1
            # A Slurm USR1/TERM stops the current episode (checkpointed); do not start another.
            if (( STOPPING )); then echo "stop signal received: ending sweep after task ${i}" >&2; rc=1; break; fi
        done
        exit "${rc}" ;;
    shell)
        echo "vLLM serving ${HF_ID} on 127.0.0.1:${SWARM_VLLM_PORT}. Example live session:"
        echo "  swarm run --impostor live --mode cluster --backend vllm -m ${MODEL_KEY} --out /runs"
        bash -i || true ;;
    *)
        echo "unknown action ${ACTION}" >&2; exit 2 ;;
esac
