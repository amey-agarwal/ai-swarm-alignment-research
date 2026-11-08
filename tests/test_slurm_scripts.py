"""Static checks of the Slurm scripts and container recipe (they are never run here)."""

import re
import shutil
import subprocess

import pytest

from swarmguard.config import REPO_ROOT

SLURM = REPO_ROOT / "slurm"
SBATCH = ["run_single.sbatch", "sweep_array.sbatch"]
ALL = [*SBATCH, "interactive.sh", "common.sh", "container_entry.sh"]
DEF = REPO_ROOT / "container" / "swarm.def"


def _text(name):
    return (SLURM / name).read_text()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("name", ALL)
def test_bash_syntax(name):
    subprocess.run(["bash", "-n", str(SLURM / name)], check=True, timeout=10)


@pytest.mark.parametrize("name", SBATCH)
def test_sbatch_limits(name):
    t = _text(name)
    assert "#SBATCH --time=02:00:00" in t
    assert "#SBATCH --signal=B:USR1@120" in t
    assert "#SBATCH --gres=gpu:A100:1" in t
    assert 'source "${SWARM_REPO:?set SWARM_REPO}/slurm/common.sh"' in t
    # charge account / QOS come from Slurm's own env vars; nothing personal is hard-coded
    assert "SBATCH_ACCOUNT" in t and "SBATCH_QOS" in t
    assert not re.search(r"^#SBATCH --(account|qos)=", t, re.MULTILINE)


def test_interactive_limits():
    t = _text("interactive.sh")
    for flag in ('--gres="gpu:A100:${SWARM_GPUS:-1}"', "--time=02:00:00", "--signal=B:USR1@120", "salloc",
                 "srun --pty", "SALLOC_ACCOUNT", "SALLOC_QOS"):
        assert flag in t


def test_common_sandbox_flags():
    t = _text("common.sh")
    for flag in ("--containall", "--cleanenv", "--no-home", "--nv", "NET_FLAGS=(--net --network none)",
                 ":/repo:ro", ":/weights:ro", ":/runs:rw", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
                 "PYTHONDONTWRITEBYTECODE=1", "HF_HOME=/jobtmp/hf", "VLLM_CACHE_ROOT=/jobtmp/vllm",
                 "TMPDIR=/jobtmp/tmp", "SWARM_VLLM_API_KEY=", "kill -USR1"):
        assert flag in t, flag
    assert '"127.0.0.1", 0' in t and "secrets.token_hex(32)" in t
    assert ': "${SLURM_JOB_ID:?' in t  # refuses to run outside a Slurm job
    # nothing from the host environment leaks in except what is passed explicitly
    assert "--env-file" not in t and "APPTAINERENV_" not in t


def test_entry_vllm_flags():
    t = _text("container_entry.sh")
    assert "--host 127.0.0.1" in t and '--api-key "${SWARM_VLLM_API_KEY}"' in t
    assert '--tensor-parallel-size "${TP}"' in t and "torch.cuda.device_count()" in t
    assert "--enable-prefix-caching" in t and "--max-num-seqs" in t and "--dtype bfloat16" in t
    assert "--disable-log-requests" in t  # prompts are not echoed to the Slurm log
    assert 'startswith("/repo/src/")' in t  # runs the manifest-checked repo code only
    for cmd in ("swarm run --mode cluster --backend vllm", "swarm judge", "swarm sweep"):
        assert cmd in t
    assert "trap cleanup EXIT" in t


def test_no_forbidden_models_or_tools():
    text = "\n".join(_text(n) for n in ALL)
    assert not re.search(r"qwen|deepseek", text, re.IGNORECASE)
    assert not re.search(r"\b(curl|wget|ssh|scp|nc)\b ", text)
    assert "tools" not in text


@pytest.mark.skipif(not DEF.exists(), reason="container/swarm.def not copied in yet")
def test_container_recipe():
    t = DEF.read_text()
    assert "From: python:3.11" in t
    assert re.search(r'"vllm==\d+\.\d+\.\d+(\.post\d+)?"', t)  # pinned exactly
    assert '"/opt/swarmguard[cluster]"' in t
    for env in ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "PYTHONDONTWRITEBYTECODE=1", "VLLM_NO_USAGE_STATS=1",
                "PYTHONPATH=/repo/src"):
        assert env in t
    assert "network tool present" in t  # build fails if curl/wget/ssh/... are present
    assert not re.search(r"qwen|deepseek", t, re.IGNORECASE)
