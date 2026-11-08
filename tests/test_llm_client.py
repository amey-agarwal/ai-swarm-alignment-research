import json
from types import SimpleNamespace

import pytest

from swarmguard.agents.cache import CacheFullError, CachedResponse, ResponseCache
from swarmguard.agents.llm_client import (
    FORBIDDEN_REQUEST_KEYS,
    CacheMiss,
    ChatRequest,
    LLMClient,
    LLMError,
    SamplingParams,
    StubBackend,
    VLLMBackend,
    build_request,
    cache_key,
    call_seed,
    check_host,
    make_backend,
)
from swarmguard.agents.parser import leader_json_schema, parse_leader_output, parse_worker_output, worker_json_schema
from swarmguard.config import load_config
from swarmguard.rundir import RunDir


def _req(agent="a01", rnd=0, content="Round 1. Share your clue.", schema=None, temp=0.7):
    schema = schema or worker_json_schema({"L"})
    return ChatRequest.make(agent, rnd, [("system", "You are W1."), ("user", content)], schema,
                            SamplingParams(temperature=temp, max_tokens=120, seed=call_seed(0, agent, rnd)))


# --- seeds + keys -------------------------------------------------------------------


def test_call_seed_deterministic_and_distinct():
    assert call_seed(0, "a01", 3) == call_seed(0, "a01", 3)
    seeds = {call_seed(s, a, r) for s in range(3) for a in ("a00", "a01") for r in range(5)}
    assert len(seeds) == 30 and all(0 <= x < 2**31 for x in seeds)


def test_cache_key_sensitivity():
    base = cache_key("m", "rev", _req())
    assert base == cache_key("m", "rev", _req())
    assert base != cache_key("m2", "rev", _req())
    assert base != cache_key("m", "rev2", _req())
    assert base != cache_key("m", "rev", _req(temp=0.5))
    assert base != cache_key("m", "rev", _req(content="different"))
    assert base != cache_key("m", "rev", _req(rnd=1))  # seed differs
    assert base != cache_key("m", "rev", _req(schema=worker_json_schema({"L", "X"})))


def test_invalid_role_rejected():
    with pytest.raises(LLMError):
        ChatRequest.make("a01", 0, [("tool", "x")], {}, SamplingParams(0.7, 120, 1))


# --- stub ------------------------------------------------------------------------------


def test_stub_default_output_parses():
    b = StubBackend()
    w = b.complete(_req())
    assert parse_worker_output(w.text, allowed_recipients={"L"}, max_tokens=120).to == "L"
    l = b.complete(_req(agent="a00", schema=leader_json_schema({"W1"})))
    assert parse_leader_output(l.text, allowed_recipients={"W1"}, max_tokens=120).message.submit is None
    assert b.complete(_req()).text == w.text


def test_stub_custom_responder():
    b = StubBackend(responder=lambda r: json.dumps({"to": "L", "content": f"{r.agent_id}:{r.round}"}))
    assert json.loads(b.complete(_req(rnd=4)).text)["content"] == "a01:4"


def test_make_backend_modes():
    assert isinstance(make_backend(load_config()), StubBackend)
    with pytest.raises(LLMError):
        make_backend(load_config().model_copy(update={"run": load_config().run.model_copy(update={"backend": "vllm"})}),
                     model_id="m")


# --- vLLM ------------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["localhost", "0.0.0.0", "127.0.0.2", "::1", "10.0.0.5", "127.0.0.1 ", "example.com"])
def test_host_must_be_exact_localhost(host):
    with pytest.raises(LLMError):
        check_host(host)
    with pytest.raises(LLMError):
        VLLMBackend("m", "r", host=host, port=8000, client=object())


def test_bad_port_rejected():
    for port in (0, 80, 70000, "8000"):
        with pytest.raises(LLMError):
            VLLMBackend("m", "r", host="127.0.0.1", port=port, client=object())


def test_missing_key_rejected(monkeypatch):
    monkeypatch.delenv("SWARM_VLLM_API_KEY", raising=False)
    with pytest.raises(LLMError):
        VLLMBackend("m", "r", host="127.0.0.1", port=8000)


def test_build_request_has_schema_and_no_tools():
    kw = build_request("meta-llama/Llama-3.2-3B-Instruct", _req())
    assert not (set(kw) & FORBIDDEN_REQUEST_KEYS)
    assert kw["response_format"]["type"] == "json_schema"
    assert kw["response_format"]["json_schema"]["schema"]["properties"]["to"]["enum"] == ["L"]
    assert kw["seed"] == call_seed(0, "a01", 0) and kw["max_tokens"] == 120 and kw["n"] == 1
    assert [m["role"] for m in kw["messages"]] == ["system", "user"]


class FakeOpenAI:
    """Records create() kwargs; returns a canned completion. No network."""

    def __init__(self, text):
        self.kwargs = []
        self.text = text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=9),
        )


def test_vllm_backend_with_fake_client():
    fake = FakeOpenAI('{"to": "L", "content": "My clue: Ana drinks tea."}')
    b = VLLMBackend("meta-llama/Llama-3.2-3B-Instruct", "abc123", host="127.0.0.1", port=8123, client=fake)
    out = b.complete(_req())
    assert out.prompt_tokens == 50 and out.completion_tokens == 9 and out.finish_reason == "stop"
    assert not (set(fake.kwargs[0]) & FORBIDDEN_REQUEST_KEYS)
    assert fake.kwargs[0]["model"] == "meta-llama/Llama-3.2-3B-Instruct"


# --- cache ------------------------------------------------------------------------------


def test_client_caches_and_replays(tmp_path):
    rd = RunDir.create(tmp_path)
    stub = StubBackend()
    with ResponseCache(rd, 10**7) as cache:
        client = LLMClient(stub, cache)
        a = client.complete(_req())
        b = client.complete(_req())
        assert not a.cached and b.cached and a.text == b.text and len(stub.calls) == 1
    with ResponseCache(rd, 10**7, read_only=True) as cache:
        replay = LLMClient(None, cache, cache_only=True, model_id="stub", model_revision="stub-v1")
        assert replay.complete(_req()).text == a.text
        with pytest.raises(CacheMiss):
            replay.complete(_req(rnd=9))
        with pytest.raises(PermissionError):
            cache.put("k", "m", CachedResponse("x", 1, 1, "stop"))


def test_cache_creates_no_extra_files(tmp_path):
    rd = RunDir.create(tmp_path)
    with ResponseCache(rd, 10**7) as cache:
        client = LLMClient(StubBackend(), cache)
        for r in range(20):
            client.complete(_req(rnd=r))
        assert [p.name for p in rd.entries()] == ["cache.sqlite"]
    assert rd.unexpected() == []


def test_cache_size_cap(tmp_path):
    rd = RunDir.create(tmp_path)
    with ResponseCache(rd, 20_000) as cache:
        with pytest.raises(CacheFullError):
            for i in range(10_000):
                cache.put(f"k{i}", "m", CachedResponse("x" * 500, 1, 1, "stop"))
        assert cache.size_bytes() <= 20_000 + 8192  # within a page or two of the cap


def test_cache_stores_hostile_text_literally(tmp_path):
    rd = RunDir.create(tmp_path)
    evil = "'); DROP TABLE responses; --"
    with ResponseCache(rd, 10**7) as cache:
        cache.put(evil, "m", CachedResponse(evil, 1, 1, "stop"))
        assert cache.get(evil).text == evil and len(cache) == 1


def test_cache_path_is_inside_rundir(tmp_path):
    rd = RunDir.create(tmp_path)
    with ResponseCache(rd, 10**7) as cache:
        assert cache.path.parent == rd.path


def test_complete_many_preserves_order(tmp_path):
    stub = StubBackend(responder=lambda r: json.dumps({"to": "L", "content": r.agent_id}))
    client = LLMClient(stub, None)
    reqs = [_req(agent=f"a{i:02d}") for i in range(1, 9)]
    outs = client.complete_many(reqs, concurrency=4)
    assert [json.loads(o.text)["content"] for o in outs] == [f"a{i:02d}" for i in range(1, 9)]


def test_client_requires_backend_or_cache():
    with pytest.raises(LLMError):
        LLMClient(None, None)
