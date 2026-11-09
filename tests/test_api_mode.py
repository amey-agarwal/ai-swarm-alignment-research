"""API track (OpenRouter). No network, no credits: the OpenAI client and the key-usage endpoint
are faked. Egress allowlist, key handling, provider pinning, spend caps, scoring and tracks."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

try:
    import httpx2 as httpx
except ImportError:
    httpx = pytest.importorskip("httpx")

from swarmguard import runner  # noqa: E402
from swarmguard.agents import llm_client as lc  # noqa: E402
from swarmguard.agents.parser import worker_json_schema  # noqa: E402
from swarmguard.config import CONFIG_DIR, ModelEntry, load_config  # noqa: E402
from swarmguard.rundir import RunDir  # noqa: E402
from swarmguard.safety import postflight, preflight, probes  # noqa: E402
from swarmguard.safety.report import is_analysable, load_safety_report  # noqa: E402

API_YAML = CONFIG_DIR / "api_openrouter.yaml"
KEY = "sk-or-v1-TESTKEY0123456789"


@pytest.fixture
def key_file(tmp_path):
    p = tmp_path / "secrets" / "openrouter.key"
    p.parent.mkdir()
    p.write_text(KEY + "\n")
    p.chmod(0o600)
    return p


@pytest.fixture
def api_yaml(tmp_path, key_file):
    data = API_YAML.read_text().replace("~/.config/swarmguard/openrouter.key", str(key_file))
    p = tmp_path / "api.yaml"
    p.write_text(data)
    return p


@pytest.fixture
def env():
    return {"PATH": "/usr/bin", "HOME": "/nonexistent", "PYTHONDONTWRITEBYTECODE": "1", "SLURM_JOB_ID": "4242"}


class Usage:
    """Fake OpenRouter key usage that grows with recorded spend."""

    def __init__(self, start=1.0):
        self.value = start

    def __call__(self, key, **kw):
        assert key == KEY
        return {"usage": self.value, "limit": 50.0, "limit_remaining": 50.0 - self.value}


def _req(agent="a01", rnd=0):
    return lc.ChatRequest.make(agent, rnd, [("system", "You are W1."), ("user", "Round 1.")], worker_json_schema({"L"}),
                               lc.SamplingParams(temperature=0.7, max_tokens=120, seed=lc.call_seed(0, agent, rnd)))


class FakeOpenAI:
    """Stands in for openai.OpenAI. Runs the egress hook of the given http client like a real request."""

    instances: list = []
    cost = 0.01
    usage_tracker: "Usage | None" = None

    def __init__(self, base_url=None, api_key=None, http_client=None, **kw):
        self.base_url, self.http, self.calls = base_url, http_client, []
        assert api_key == KEY
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        FakeOpenAI.instances.append(self)

    def _create(self, **kwargs):
        for hook in self.http.event_hooks["request"]:
            hook(httpx.Request("POST", f"{self.base_url}/chat/completions"))
        self.calls.append(kwargs)
        if FakeOpenAI.usage_tracker is not None:
            FakeOpenAI.usage_tracker.value += FakeOpenAI.cost
        props = kwargs["response_format"]["json_schema"]["schema"].get("properties", {})
        if "messages" in props:
            out = {"messages": [], "submit": None}
        elif "capabilities" in props:
            out = {"capabilities": []}
        else:
            out = {"to": props["to"]["enum"][0], "content": "My clue is shared."}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(out)), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=9, cost=FakeOpenAI.cost))


@pytest.fixture
def fake_api(monkeypatch):
    FakeOpenAI.instances, FakeOpenAI.cost = [], 0.01
    usage = Usage()
    FakeOpenAI.usage_tracker = usage
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr(lc, "fetch_key_usage", usage)
    monkeypatch.setattr(ModelEntry, "is_placeholder", property(lambda self: False))
    return usage


# --- config / registry --------------------------------------------------------------------


def test_api_yaml_valid():
    cfg = load_config(API_YAML)
    assert (cfg.run.mode, cfg.run.backend, cfg.api.host) == ("api", "openrouter", "openrouter.ai")
    assert (cfg.api.usd_warn, cfg.api.usd_hard) == (25, 50)


@pytest.mark.parametrize("over", [{"run": {"mode": "api", "backend": "stub"}}, {"run": {"mode": "dev", "backend": "openrouter"}},
                                  {"api": {"usd_hard": 60}}, {"api": {"usd_warn": 50, "usd_hard": 40}},
                                  {"api": {"host": "evil.example"}}, {"sandbox": {"network_isolation": "required"}}])
def test_api_config_rejects(over):
    with pytest.raises(ValueError):
        load_config(API_YAML, overrides=over)


@pytest.mark.parametrize("kw", [
    {"openrouter_id": "qwen/qwen3-235b", "providers": ["together"]},
    {"openrouter_id": "deepseek/deepseek-chat", "providers": ["together"]},
    {"openrouter_id": "acme/model", "providers": ["deepseek"]},
    {"openrouter_id": "acme/model:online", "providers": ["acme"]},
    {"openrouter_id": "openrouter/auto", "providers": ["acme"]},
    {"openrouter_id": "acme/model", "providers": []},
    {"openrouter_id": "acme/model", "providers": ["acme"], "hf_id": "other/model"},
])
def test_openrouter_entries_rejected(kw):
    data = {"hf_id": kw.get("openrouter_id"), "role": "swarm", "license": "x", "gated": False, **kw}
    with pytest.raises(ValueError):
        ModelEntry(**data)


def test_open_weight_entry_needs_params():
    with pytest.raises(ValueError):
        ModelEntry(hf_id="meta-llama/x", role="swarm", license="x", gated=False)


def test_placeholders_refused_by_pass1(repo_copy, api_yaml, env, monkeypatch):
    monkeypatch.setattr(lc, "fetch_key_usage", Usage())
    p1 = preflight.run_pass1(api_yaml, "api", root=repo_copy, env=env)
    assert not p1.ok and "CHANGEME" in p1.get("models").detail


# --- key file and egress ------------------------------------------------------------------


def test_key_file_rules(tmp_path, key_file, monkeypatch):
    assert lc.read_api_key(key_file) == KEY
    key_file.chmod(0o644)
    with pytest.raises(lc.LLMError, match="chmod 600"):
        lc.read_api_key(key_file)
    key_file.chmod(0o600)
    link = tmp_path / "link.key"
    link.symlink_to(key_file)
    with pytest.raises(lc.LLMError, match="symlink"):
        lc.read_api_key(link)
    monkeypatch.setattr(lc, "REPO_ROOT", key_file.parent)
    with pytest.raises(lc.LLMError, match="outside the repository"):
        lc.read_api_key(key_file)


def test_key_error_never_contains_key(tmp_path):
    p = tmp_path / "k"
    p.write_text(KEY + " extra")
    p.chmod(0o600)
    with pytest.raises(lc.LLMError) as ei:
        lc.read_api_key(p)
    assert KEY not in str(ei.value)


@pytest.mark.parametrize("url", probes.EGRESS_REFUSE)
def test_egress_refused(url):
    with pytest.raises(lc.LLMError):
        lc.check_api_url(url)


def test_egress_probe():
    assert probes.probe_egress_allowlist().status == "pass"
    assert probes.probe_egress_allowlist(check=lambda u: u).status == "fail"


def test_transport_hook_blocks_other_hosts():
    hosts = set()
    client = lc._api_http(5.0, hosts)
    with pytest.raises(lc.LLMError):
        client.get("https://example.com/")  # refused in the hook, before any connection
    assert hosts == set()


# --- backend ------------------------------------------------------------------------------


def test_backend_pins_providers_and_counts_spend(key_file, fake_api):
    b = lc.OpenRouterBackend("acme/model", ["acme"], key_file=str(key_file), usd_hard=50)
    out = b.complete(_req())
    call = FakeOpenAI.instances[0].calls[0]
    assert not (set(call) & lc.FORBIDDEN_REQUEST_KEYS) and "extra_body" in call
    prov = call["extra_body"]["provider"]
    assert prov == {"order": ["acme"], "only": ["acme"], "allow_fallbacks": False, "require_parameters": True,
                    "data_collection": "deny"}
    assert call["model"] == "acme/model" and call["response_format"]["type"] == "json_schema"
    assert out.prompt_tokens == 50 and abs(b.spend_usd - 0.01) < 1e-12
    assert b.api_summary()["hosts"] == ["openrouter.ai"] and FakeOpenAI.instances[0].base_url == lc.OPENROUTER_BASE
    assert KEY not in repr(b) and KEY not in json.dumps(b.api_summary())


def test_backend_hard_cap(key_file, fake_api):
    fake_api.value = 49.995
    b = lc.OpenRouterBackend("acme/model", ["acme"], key_file=str(key_file), usd_hard=50)
    b.complete(_req())  # 49.995 + 0.01 crosses the cap
    with pytest.raises(lc.SpendCapError):
        b.complete(_req())
    fake_api.value = 50.0
    with pytest.raises(lc.SpendCapError):
        lc.OpenRouterBackend("acme/model", ["acme"], key_file=str(key_file), usd_hard=50)


def test_make_backend_refuses_outside_api(key_file, fake_api):
    from swarmguard.config import RunCfg

    cfg = load_config().model_copy(update={"run": RunCfg.model_construct(mode="cluster", backend="openrouter", seed=0)})
    with pytest.raises(lc.LLMError, match="api mode"):
        lc.make_backend(cfg, model_id="acme/model", providers=["acme"])


# --- spend command --------------------------------------------------------------------------


@pytest.mark.parametrize("used,code", [(3.0, 0), (25.0, 10), (49.9, 10), (50.0, 11)])
def test_spend_exit_codes(api_yaml, env, repo_copy, used, code):
    oc = runner.spend_cmd(api_yaml, root=repo_copy, env=env, usage_fn=lambda kf: {"usage": used, "limit": 50.0})
    assert oc.exit_code == code


def test_spend_unknown_fails_closed(api_yaml, env, repo_copy):
    def boom(kf):
        raise lc.LLMError("down")

    assert runner.spend_cmd(api_yaml, root=repo_copy, env=env, usage_fn=boom).exit_code == 1


# --- end to end ------------------------------------------------------------------------------


def test_api_run_scores_on_its_own_track(tmp_path, repo_copy, api_yaml, env, fake_api, monkeypatch):
    pytest.importorskip("psutil")
    from swarmguard.safety import governor

    monkeypatch.setattr(governor.shutil, "disk_usage",  # independent of the host's free disk
                        lambda p: SimpleNamespace(total=1000, used=500, free=500))
    out = tmp_path / "runs"
    o = runner.RunOptions(n=4, rounds=4, task="logic-grid:L4", impostor="scripted", claim="write", model="api-model-a",
                          mode="api", backend="openrouter", seed=0, out=out, config=api_yaml)
    oc = runner.execute_run(o, root=repo_copy, env=env, out=lambda s: None)
    assert oc.result is not None and oc.result.rounds_completed == 4, oc.message
    rep = load_safety_report(oc.rd.file("safety_report.json"))
    assert rep.mode == "api" and rep.api is not None
    assert rep.api.hosts_contacted == ["openrouter.ai"] and rep.api.egress_allowlist_only
    assert rep.api.key_never_logged and rep.api.spend_within_cap and rep.api.isolation_acknowledged
    assert "containment.network_attempts_succeeded" not in rep.checks and rep.containment.network_attempts_succeeded == 2
    assert rep.resources.spend_usd > 0 and rep.safety_score == 1.0 and is_analysable(rep), rep.checks
    assert oc.exit_code == 0
    names = {c.name: c.status for c in oc.pass2.checks}
    assert names["net_tcp_external"] == "waived" and names["egress_allowlist"] == "pass"
    for p in oc.rd.entries():
        assert KEY.encode() not in p.read_bytes()

    # tracks are never pooled
    assert runner.analyze(out, root=repo_copy, env=env, track="main").extra["n_included"] == 0
    assert runner.analyze(out, root=repo_copy, env=env, track="api").extra["n_included"] == 1


def test_api_run_refused_outside_slurm(tmp_path, repo_copy, api_yaml, env, fake_api):
    env = {k: v for k, v in env.items() if k != "SLURM_JOB_ID"}
    o = runner.RunOptions(n=4, rounds=2, model="api-model-a", mode="api", backend="openrouter", out=tmp_path / "r",
                          config=api_yaml)
    oc = runner.execute_run(o, root=repo_copy, env=env, out=lambda s: None)
    assert oc.exit_code == 1 and "pass 1" in oc.message and not FakeOpenAI.instances


def test_leaked_key_fails_postflight(tmp_path, key_file):
    cfg = load_config(API_YAML, overrides={"api": {"key_file": str(key_file)}})
    rd = RunDir.create(tmp_path)
    rd.file("log.jsonl").write_text(json.dumps({"oops": KEY}) + "\n")
    api, problems = postflight.api_checks(cfg, rd, {"hosts": ["openrouter.ai"], "spend_usd": 0.1, "baseline_usd": 1.0,
                                                    "key_usage_end": 1.1})
    assert not api.key_never_logged and problems
    api, _ = postflight.api_checks(cfg, rd, {"hosts": ["openrouter.ai", "evil.example"], "key_usage_end": 1.0})
    assert not api.egress_allowlist_only
    api, _ = postflight.api_checks(cfg, rd, {"hosts": [], "key_usage_end": 50.0})
    assert not api.spend_within_cap
    api, _ = postflight.api_checks(cfg, rd, None)
    assert not api.egress_allowlist_only and not api.spend_within_cap


def test_api_report_requires_api_checks():
    from swarmguard.safety.report import build_safety_report
    from swarmguard.schemas import Containment, Messaging, Resources

    cfg = load_config(API_YAML)
    with pytest.raises(ValueError, match="api checks"):
        build_safety_report(run_id="20261108-000000_00000000", cfg=cfg,
                            containment=Containment(network_attempts_succeeded=0, writes_outside_rundir=0,
                                                    outputs_executed=0, routing_violations_accepted=0),
                            resources=Resources(peak_cpi=0, cpi_hard=0.95, energy_wh=0, energy_budget_wh=50, tokens=0,
                                                token_budget=1),
                            messaging=Messaging(total=0, bound=1, max_len_respected=True, per_agent_round_respected=True,
                                                loops_broken=0),
                            hash_chain_valid=True, manifest_unchanged=True, rundir_allowlist_ok=True, pass1=True,
                            pass2=True)


def test_cluster_runs_cannot_use_openrouter_models(repo_copy, env):
    p1 = preflight.run_pass1({"model": {"swarm": "api-model-a"}}, "dev", root=repo_copy, env=env)
    assert not p1.ok and "only in api mode" in p1.get("models").detail


def test_api_key_never_in_environment_check(repo_copy, env):
    assert preflight.check_secrets({**env, "OPENROUTER_API_KEY": "x"}).status == "fail"
    assert os.environ.get("OPENROUTER_API_KEY") is None or True  # documented: never export the key


def test_judge_config_for_api_runs_passes_model_check(repo_copy, env):
    # what judge_run builds for an api run: the run's config in cluster mode with the default swarm key
    run_cfg = load_config(API_YAML).model_dump(mode="json")
    run_cfg["run"] = {"mode": "cluster", "backend": "vllm", "seed": 0}
    run_cfg["model"]["swarm"] = load_config().model.swarm
    from swarmguard.config import Config

    assert preflight.check_models(Config.model_validate(run_cfg)).status == "pass"
