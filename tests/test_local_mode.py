"""Local exploratory mode (Ollama on 127.0.0.1). No Ollama, no network: the OpenAI client
is replaced by a fake module, and network/write probes must never be attempted."""

import json
import sys
from types import SimpleNamespace

import pytest

from swarmguard import runner
from swarmguard.agents.llm_client import (
    FORBIDDEN_REQUEST_KEYS,
    ChatRequest,
    LLMError,
    OllamaBackend,
    SamplingParams,
    call_seed,
    make_backend,
)
from swarmguard.agents.parser import worker_json_schema
from swarmguard.config import CONFIG_DIR, Config, ModelEntry, RunCfg, load_config, load_models
from swarmguard.rundir import RunDir
from swarmguard.safety.preflight import run_pass1
from swarmguard.safety.probes import probe_dns, probe_tcp, probe_write_outside
from swarmguard.safety.report import is_analysable, load_safety_report

LOCAL_YAML = CONFIG_DIR / "local_ollama.yaml"
ENV = {"PATH": "/usr/bin", "HOME": "/nonexistent", "PYTHONDONTWRITEBYTECODE": "1"}


def _local_cfg() -> Config:
    return load_config(LOCAL_YAML)


def _boom(*a, **k):
    raise AssertionError("probe must not be attempted in local mode")


# --- config ---------------------------------------------------------------------------


def test_local_yaml_is_valid():
    cfg = _local_cfg()
    assert (cfg.run.mode, cfg.run.backend, cfg.llm.host, cfg.llm.port) == ("local", "ollama", "127.0.0.1", 11434)
    assert cfg.sandbox.network_isolation == "unavailable"


@pytest.mark.parametrize("run", [{"mode": "dev", "backend": "ollama"}, {"mode": "cluster", "backend": "ollama"},
                                 {"mode": "local", "backend": "stub"}, {"mode": "local", "backend": "vllm"}])
def test_ollama_and_local_go_together(run):
    with pytest.raises(ValueError):
        load_config(LOCAL_YAML, overrides={"run": run})


def test_local_requires_waived_isolation():
    with pytest.raises(ValueError, match="network_isolation"):
        load_config(overrides={"run": {"mode": "local", "backend": "ollama"}})


def test_local_host_still_exactly_localhost():
    with pytest.raises(ValueError):
        load_config(LOCAL_YAML, overrides={"llm": {"host": "localhost"}})


@pytest.mark.parametrize("tag", ["qwen3:latest", "deepseek-r1:7b", "gemma4:31b-cloud", "x; rm -rf /", "../x"])
def test_bad_ollama_tags_rejected(tag):
    with pytest.raises(ValueError):
        ModelEntry(hf_id="meta-llama/x", role="swarm", params_b=1, license="l", gated=False, ollama_tag=tag)


def test_registry_has_llama_tag():
    assert load_models()["llama-3.2-3b"].ollama_tag == "llama3.2:3b"


# --- pass 1 / pass 2 -------------------------------------------------------------------


def test_pass1_local_ok(repo_copy):
    p1 = run_pass1(LOCAL_YAML, "local", root=repo_copy, env=ENV)
    assert p1.ok, [(c.name, c.status, c.detail) for c in p1.checks]
    assert "NO OS isolation" in p1.get("mode_backend_env").detail
    assert "llama3.2:3b" in p1.get("models").detail


def test_pass1_local_needs_dontwritebytecode(repo_copy):
    env = {k: v for k, v in ENV.items() if k != "PYTHONDONTWRITEBYTECODE"}
    p1 = run_pass1(LOCAL_YAML, "local", root=repo_copy, env=env)
    assert not p1.ok and p1.get("offline_env").status == "fail"


def test_network_probes_waived_not_attempted():
    cfg = _local_cfg()
    for res in (probe_tcp(cfg, "local", connect=_boom), probe_dns(cfg, "local", resolve=_boom)):
        assert res.status == "waived" and "not attempted" in res.detail


def test_write_probe_waived_not_attempted(tmp_path):
    res = probe_write_outside("local", targets=[tmp_path], cfg=_local_cfg())
    assert res.status == "waived" and list(tmp_path.iterdir()) == []
    assert probe_write_outside("local", targets=[tmp_path]).status == "fail"  # no cfg -> fail closed


# --- backend ---------------------------------------------------------------------------


class FakeChat:
    """Stands in for openai.OpenAI. Answers from the requested JSON schema; never touches the network."""

    instances: list = []

    def __init__(self, base_url=None, api_key=None, **kw):
        self.base_url, self.api_key, self.kwargs = base_url, api_key, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        FakeChat.instances.append(self)

    def _create(self, **kwargs):
        self.kwargs.append(kwargs)
        props = kwargs["response_format"]["json_schema"]["schema"].get("properties", {})
        if "messages" in props:
            out = {"messages": [], "submit": None}
        elif "capabilities" in props:
            out = {"capabilities": []}
        else:
            out = {"to": props["to"]["enum"][0], "content": "My clue is shared."}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(out)), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=40, completion_tokens=8),
        )


@pytest.fixture
def fake_openai(monkeypatch):
    FakeChat.instances = []
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeChat))
    return FakeChat


def test_make_backend_local(fake_openai):
    b = make_backend(_local_cfg(), model_id="llama3.2:3b")
    assert isinstance(b, OllamaBackend)
    assert (b.model_id, b.model_revision) == ("llama3.2:3b", "ollama:llama3.2:3b")
    assert fake_openai.instances[0].base_url == "http://127.0.0.1:11434/v1"


def test_ollama_refused_outside_local(fake_openai):
    cfg = load_config().model_copy(update={"run": RunCfg.model_construct(mode="dev", backend="ollama", seed=0)})
    with pytest.raises(LLMError, match="local"):
        make_backend(cfg, model_id="llama3.2:3b")


def test_ollama_request_has_schema_and_no_tools(fake_openai):
    b = OllamaBackend("llama3.2:3b", host="127.0.0.1", port=11434)
    req = ChatRequest.make("a01", 0, [("system", "You are W1."), ("user", "Round 1.")], worker_json_schema({"L"}),
                           SamplingParams(temperature=0.7, max_tokens=120, seed=call_seed(0, "a01", 0)))
    b.complete(req)
    kw = fake_openai.instances[0].kwargs[0]
    assert not (set(kw) & FORBIDDEN_REQUEST_KEYS)
    assert kw["model"] == "llama3.2:3b" and kw["response_format"]["type"] == "json_schema"
    assert kw["seed"] == call_seed(0, "a01", 0)


@pytest.mark.parametrize("host", ["localhost", "0.0.0.0", "ollama.com"])
def test_ollama_host_must_be_localhost(host, fake_openai):
    with pytest.raises(LLMError):
        OllamaBackend("llama3.2:3b", host=host, port=11434)


# --- end to end ------------------------------------------------------------------------


def _disk(free_frac):
    return lambda path: SimpleNamespace(total=1000, used=int(1000 * (1 - free_frac)), free=int(1000 * free_frac))


def test_local_governor_stops_on_low_disk(monkeypatch):
    pytest.importorskip("psutil")
    from swarmguard.safety import governor

    monkeypatch.setattr(governor.shutil, "disk_usage", _disk(0.02))
    st = governor.Governor(_local_cfg(), governor.make_provider("local", ".")).tick()
    assert st.action == "stop" and st.reason.startswith("disk:free")
    assert st.last_sample.power_w is None and "energy not measured" in governor.LocalMetricsProvider.name


def test_local_run_is_never_analysable(tmp_path, repo_copy, fake_openai, monkeypatch):
    pytest.importorskip("psutil")
    from swarmguard.safety import governor

    monkeypatch.setattr(governor.shutil, "disk_usage", _disk(0.5))  # independent of the host's disk
    out = tmp_path / "runs"
    o = runner.RunOptions(n=4, rounds=4, task="logic-grid:L4", impostor="scripted", claim="write",
                          mode="local", backend="ollama", seed=0, out=out, config=LOCAL_YAML)
    oc = runner.execute_run(o, root=repo_copy, env=ENV, out=lambda s: None)
    assert oc.rd is not None and oc.result is not None, oc.message
    assert oc.result.rounds_completed > 0 and oc.result.stop_reason != "governor", oc.message
    assert oc.exit_code == 1  # score < 1.0 by design
    rep = load_safety_report(oc.rd.file("safety_report.json"))
    assert rep.mode == "local" and rep.network_isolation == "unavailable"
    assert rep.containment.network_attempts_succeeded == 2 and rep.containment.writes_outside_rundir == 1
    assert rep.safety_score < 1.0 and not is_analysable(rep)
    assert rep.integrity.pass1 and rep.integrity.pass2 and rep.integrity.hash_chain_valid
    start = json.loads(oc.rd.file("log.jsonl").read_text().splitlines()[0])
    assert start["model_id"] == "llama3.2:3b"
    assert all(not (set(k) & FORBIDDEN_REQUEST_KEYS) for c in fake_openai.instances for k in c.kwargs)

    # replay needs no backend and reproduces the log
    assert runner.replay_run(oc.rd.run_id, out_root=out, root=repo_copy, env=ENV).extra["identical"]
    # analysis excludes it
    an = runner.analyze(out, root=repo_copy, env=ENV)
    assert an.extra["n_included"] == 0 and oc.rd.run_id in [e["run_id"] for e in an.extra["excluded"]]


def test_judge_refused_in_local_mode(tmp_path):
    oc = runner.judge_run("x", out_root=tmp_path, mode="local", backend="ollama", env=ENV)
    assert oc.exit_code == 2 and "local mode" in oc.message
