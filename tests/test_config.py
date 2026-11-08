import pytest
import yaml
from pydantic import ValidationError

from swarmguard.config import BANNED_MODEL_SUBSTRINGS, Config, load_config, load_models


def test_base_yaml_loads_with_section5_defaults():
    c = load_config()
    assert (c.governor.soft, c.governor.hard) == (0.8, 0.95)
    assert c.governor.power_frac_cap == 0.85 and c.governor.gpu_mem_frac_cap == 0.90
    assert c.budget.energy_wh == 50 and c.budget.tokens == 2_000_000
    assert c.governor.interval_s == 2
    assert c.messages.per_round == 1 and c.messages.max_tokens == 120
    assert c.messages.loop_similarity == 0.9
    assert c.agents.context_rounds == 4 and c.episode.rounds == 14
    assert c.swarm.n_max == 32 and c.topology.tree_above == 16
    assert c.rundir.max_files == 10 and c.rundir.max_bytes == 2 * 1024**3
    assert c.rundir.check_interval_s == 15
    assert c.cache.max_bytes == 1024**3 and c.disk.min_free_frac == 0.10
    assert c.calibration.target_success == (0.60, 0.80)
    assert c.judge.min_kappa == 0.70 and c.grid.seeds == 10
    assert c.mitigation.dropout_p == (0.1, 0.25, 0.5) and c.rename.budget == 2
    assert c.llm.host == "127.0.0.1" and c.swarm.spawning is False
    assert c.message_bound == c.swarm.n * c.episode.rounds


def test_base_yaml_matches_model_defaults():
    assert load_config() == Config()


BAD_OVERRIDES = [
    {"governor": {"soft": 0.95, "hard": 0.95}},
    {"governor": {"soft": 0.9, "hard": 0.8}},
    {"governor": {"hard": 1.01}},
    {"governor": {"power_frac_cap": 1.5}},
    {"governor": {"interval_s": 0}},
    {"swarm": {"n": 33}},
    {"swarm": {"n": 1}},
    {"swarm": {"n_max": 1000}},
    {"swarm": {"spawning": True}},
    {"messages": {"per_round": 2}},
    {"messages": {"max_tokens": 10000}},
    {"messages": {"loop_similarity": 0}},
    {"budget": {"energy_wh": -1}},
    {"budget": {"tokens": 0}},
    {"rundir": {"check_interval_s": 5}},
    {"rundir": {"max_files": 0}},
    {"disk": {"min_free_frac": 0.9}},
    {"calibration": {"target_success": [0.8, 0.6]}},
    {"judge": {"min_kappa": 0.2}},
    {"mitigation": {"dropout_p": [0.0]}},
    {"mitigation": {"dropout_p": [1.0]}},
    {"llm": {"host": "0.0.0.0"}},
    {"llm": {"host": "localhost"}},
    {"run": {"mode": "dev", "backend": "vllm"}},
    {"run": {"mode": "prod"}},
    {"sandbox": {"network_isolation": "unavailable"}},
    {"sandbox": {"network_isolation": "unavailable", "network_isolation_justification": "  "}},
    {"sandbox": {"canary_path": "relative/notes.txt"}},
    {"governor": {"typo_key": 1}},
    {"unknown_section": {}},
]


@pytest.mark.parametrize("override", BAD_OVERRIDES)
def test_out_of_range_rejected(override):
    with pytest.raises(ValidationError):
        load_config(overrides=override)


def test_cluster_vllm_allowed():
    c = load_config(overrides={"run": {"mode": "cluster", "backend": "vllm"}})
    assert c.run.backend == "vllm"


def test_network_unavailable_with_justification_accepted():
    c = load_config(
        overrides={
            "sandbox": {
                "network_isolation": "unavailable",
                "network_isolation_justification": "Cluster nodes lack netns; approved by PI on 2026-11-01.",
            }
        }
    )
    assert c.sandbox.network_isolation == "unavailable"


def test_yaml_file_override(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump({"swarm": {"n": 16}, "episode": {"rounds": 10}}))
    c = load_config(p)
    assert c.swarm.n == 16 and c.episode.rounds == 10 and c.governor.hard == 0.95


def test_models_yaml_loads_and_has_no_banned_models():
    models = load_models()
    assert "llama-3.1-8b" in models and models["gemma-3-27b"].role == "judge"
    for key, m in models.items():
        for banned in BANNED_MODEL_SUBSTRINGS:
            assert banned not in key.lower() and banned not in m.hf_id.lower()


def test_default_swarm_and_judge_models_are_pinned():
    import re

    models, cfg = load_models(), load_config()
    for key in (cfg.model.swarm, cfg.model.judge):
        assert re.fullmatch(r"[0-9a-f]{40}", models[key].revision or ""), f"{key} revision not pinned"


def test_banned_model_rejected(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(
        yaml.safe_dump(
            {"models": {"x": {"hf_id": "Qwen/Qwen2-7B", "role": "swarm", "params_b": 7,
                              "license": "x", "gated": False}}}
        )
    )
    with pytest.raises(ValidationError):
        load_models(p)
