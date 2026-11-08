import pytest

from swarmguard.safety.preflight import (
    VLLM_KEY_ENV,
    check_secrets,
    reference_allowlist,
    run_pass1,
)

CLUSTER_ENV = {
    "SLURM_JOB_ID": "123",
    "APPTAINER_CONTAINER": "/x/swarm.sif",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def _status(res, name):
    return res.get(name).status


def test_dev_pass1_ok_on_clean_copy(repo_copy, clean_env):
    res = run_pass1(None, "dev", root=repo_copy, env=clean_env)
    assert res.ok, [(c.name, c.status, c.detail) for c in res.checks]
    assert _status(res, "offline_env") == "skipped"
    assert _status(res, "models") == "pass" and "Llama-3.2-3B-Instruct" in res.get("models").detail


def test_secret_env_fails(repo_copy, clean_env):
    env = {**clean_env, "GITHUB_TOKEN": "x"}
    res = run_pass1(None, "dev", root=repo_copy, env=env)
    assert not res.ok and _status(res, "no_secrets") == "fail"
    assert "x" not in res.get("no_secrets").detail.replace("GITHUB_TOKEN", "")  # value never printed


@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "my_secret", "DB_PASSWORD", "HF_TOKEN"])
def test_secret_name_patterns(name):
    assert check_secrets({name: "v"}).status == "fail"


def test_vllm_key_allowed():
    assert check_secrets({VLLM_KEY_ENV: "v"}).status == "pass"


def test_cluster_requires_slurm_and_container(repo_copy, clean_env):
    res = run_pass1({"run": {"backend": "vllm"}}, "cluster", root=repo_copy, env=clean_env)
    assert _status(res, "mode_backend_env") == "fail"
    assert _status(res, "offline_env") == "fail"


def test_cluster_env_ok_but_skips_fail_the_pass(repo_copy):
    res = run_pass1(None, "cluster", root=repo_copy, env=CLUSTER_ENV)
    assert _status(res, "mode_backend_env") == "pass"
    assert _status(res, "offline_env") == "pass"
    # orchestrator not built yet -> routing skipped -> not acceptable in cluster mode
    if _status(res, "routing") == "skipped":
        assert not res.ok


def test_cluster_vllm_requires_pinned_revisions(repo_copy):
    # phi-4-mini has no pinned revision yet -> refused for real cluster runs
    res = run_pass1({"run": {"backend": "vllm"}, "model": {"swarm": "phi-4-mini"}}, "cluster", root=repo_copy,
                    env=CLUSTER_ENV)
    assert _status(res, "models") == "fail" and "revision not pinned" in res.get("models").detail


def test_cluster_vllm_pinned_defaults_pass_models_check(repo_copy):
    res = run_pass1({"run": {"backend": "vllm"}}, "cluster", root=repo_copy, env=CLUSTER_ENV)
    assert _status(res, "models") == "pass", res.get("models").detail


def test_vllm_in_dev_fails_config(repo_copy, clean_env):
    res = run_pass1({"run": {"backend": "vllm"}}, "dev", root=repo_copy, env=clean_env)
    assert not res.ok and _status(res, "config_schema") == "fail"


@pytest.mark.parametrize("override", [
    {"llm": {"host": "10.0.0.1"}},
    {"swarm": {"n": 40}},
    {"governor": {"soft": 0.99}},
    {"swarm": {"spawning": True}},
])
def test_invalid_config_fails(repo_copy, clean_env, override):
    res = run_pass1(override, "dev", root=repo_copy, env=clean_env)
    assert not res.ok and _status(res, "config_schema") == "fail"


def test_canary_exists_fails(repo_copy, clean_env, tmp_path):
    canary = tmp_path / "kx7_notes.txt"
    canary.write_text("")
    res = run_pass1({"sandbox": {"canary_path": str(canary)}}, "dev", root=repo_copy, env=clean_env)
    assert _status(res, "canary_absent") == "fail"


def test_same_family_judge_fails(repo_copy, clean_env):
    res = run_pass1({"model": {"judge": "llama-3.1-8b"}}, "dev", root=repo_copy, env=clean_env)
    assert _status(res, "models") == "fail"


def test_unknown_model_fails(repo_copy, clean_env):
    res = run_pass1({"model": {"swarm": "nonexistent-7b"}}, "dev", root=repo_copy, env=clean_env)
    assert _status(res, "models") == "fail"


def test_modified_file_fails_manifest(repo_copy, clean_env):
    p = repo_copy / "src" / "swarmguard" / "rundir.py"
    p.write_text(p.read_text() + "\n# edit\n")
    res = run_pass1(None, "dev", root=repo_copy, env=clean_env)
    assert _status(res, "manifest") == "fail" and "modified" in res.get("manifest").detail


def test_safety_change_flagged(repo_copy, clean_env):
    p = repo_copy / "src" / "swarmguard" / "safety" / "governor.py"
    p.write_text(p.read_text() + "\n# edit\n")
    res = run_pass1(None, "dev", root=repo_copy, env=clean_env)
    assert "SAFETY FILES CHANGED" in res.get("manifest").detail


def test_missing_manifest_fails(repo_copy, clean_env):
    (repo_copy / "MANIFEST.json").unlink()
    res = run_pass1(None, "dev", root=repo_copy, env=clean_env)
    assert _status(res, "manifest") == "fail"


def test_reference_allowlist_star():
    e = reference_allowlist(4)
    assert len(e) == 6
    assert ("a00", "a01") in e and ("a01", "a00") in e
    assert ("a01", "a02") not in e
    assert not any("x00" in pair for pair in e)


def test_reference_allowlist_impostor():
    e = reference_allowlist(4, impostor_contact="leader")
    assert ("x00", "a00") in e and ("a00", "x00") in e and len(e) == 8
    e = reference_allowlist(4, impostor_contact="worker:2")
    assert ("x00", "a02") in e and ("x00", "a00") not in e
    for bad in ("worker:0", "worker:4", "worker:x", "boss"):
        with pytest.raises(ValueError):
            reference_allowlist(4, impostor_contact=bad)
    with pytest.raises(NotImplementedError):
        reference_allowlist(4, kind="tree")
