"""LLM backends: deterministic stub (dev/tests), localhost vLLM (cluster) and
localhost Ollama (local exploratory mode).

This is the only module allowed to import a network client (enforced by the
static scan in pass 1). Hard rules enforced here:
- The vLLM / Ollama host must be exactly 127.0.0.1.
- Requests never carry tool / function-calling parameters (checked at runtime
  in `build_request` and statically by the scan).
- The vLLM backend is refused in dev mode; the Ollama backend only runs in local mode.
- Per-call seed = hash(run_seed, agent_internal_id, round).
- Cache key = SHA-256 of (model id, model revision, sampling params, messages, output schema).

Models are text-in -> text-out: the request is a list of chat messages plus a
JSON schema for constrained decoding; the response is text, which the caller
passes to agents/parser.py. Nothing in the response is executed.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from swarmguard.agents.cache import CachedResponse, ResponseCache
from swarmguard.agents.parser import count_tokens
from swarmguard.config import LOCALHOST, OPENROUTER_HOST, REPO_ROOT, Config

VLLM_KEY_ENV = "SWARM_VLLM_API_KEY"
# Request keys that must never be sent (tool / function calling, OpenRouter plugins such as web search).
FORBIDDEN_REQUEST_KEYS = frozenset({"tools", "tool_choice", "functions", "function_call", "parallel_tool_calls",
                                    "plugins", "web_search_options"})
ALLOWED_REQUEST_KEYS = frozenset(
    {"model", "messages", "temperature", "top_p", "max_tokens", "seed", "response_format", "n"}
)


class LLMError(RuntimeError):
    pass


class CacheMiss(LLMError):
    pass


def call_seed(run_seed: int, agent_id: str, round_idx: int) -> int:
    h = hashlib.sha256(f"{run_seed}|{agent_id}|{round_idx}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


@dataclass(frozen=True)
class SamplingParams:
    temperature: float
    max_tokens: int
    seed: int
    top_p: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens, "seed": self.seed, "top_p": self.top_p}


@dataclass(frozen=True)
class ChatRequest:
    agent_id: str
    round: int
    messages: tuple[tuple[str, str], ...]  # (role, content)
    json_schema: str  # canonical JSON of the output schema
    sampling: SamplingParams
    schema_name: str = "output"

    @staticmethod
    def make(agent_id: str, round_idx: int, messages: Sequence[tuple[str, str]], schema: dict[str, Any],
             sampling: SamplingParams, schema_name: str = "output") -> "ChatRequest":
        for role, _ in messages:
            if role not in ("system", "user", "assistant"):
                raise LLMError(f"invalid role {role!r}")
        return ChatRequest(agent_id, round_idx, tuple((r, c) for r, c in messages),
                           json.dumps(schema, sort_keys=True, separators=(",", ":")), sampling, schema_name)

    def schema(self) -> dict[str, Any]:
        return json.loads(self.json_schema)


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    cached: bool = False


def cache_key(model_id: str, model_revision: str | None, req: ChatRequest) -> str:
    payload = {
        "model_id": model_id,
        "model_revision": model_revision,
        "sampling": req.sampling.as_dict(),
        "messages": [list(m) for m in req.messages],
        "schema": req.json_schema,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Backend(Protocol):
    model_id: str
    model_revision: str | None

    def complete(self, req: ChatRequest) -> Completion: ...


# ----------------------------------------------------------------------------- stub


Responder = Callable[[ChatRequest], str]


def default_responder(req: ChatRequest) -> str:
    """Valid JSON for the requested schema, deterministic in the request."""
    schema = req.schema()
    props = schema.get("properties", {})
    if "messages" in props:
        return json.dumps({"messages": [], "submit": None})
    if "to" in props:
        recipients = props["to"].get("enum") or ["?"]
        return json.dumps({"to": recipients[0], "content": f"stub {req.agent_id} round {req.round}"})
    return json.dumps({k: None for k in props})


@dataclass
class StubBackend:
    """Deterministic backend for dev mode and tests. Never touches the network."""

    responder: Responder = default_responder
    model_id: str = "stub"
    model_revision: str | None = "stub-v1"
    calls: list[ChatRequest] = field(default_factory=list)

    def complete(self, req: ChatRequest) -> Completion:
        self.calls.append(req)
        text = self.responder(req)
        if not isinstance(text, str):
            raise LLMError("stub responder must return text")
        prompt_tokens = sum(count_tokens(c) for _, c in req.messages)
        return Completion(text, prompt_tokens, count_tokens(text), "stop")


# ----------------------------------------------------------------------------- vLLM


def check_host(host: str) -> str:
    if host != LOCALHOST:
        raise LLMError(f"LLM host must be exactly {LOCALHOST}, got {host!r}")
    return host


def build_request(model_id: str, req: ChatRequest) -> dict[str, Any]:
    """OpenAI-compatible chat request with structured output and no tools."""
    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": r, "content": c} for r, c in req.messages],
        **req.sampling.as_dict(),
        "n": 1,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": req.schema_name, "schema": req.schema(), "strict": True},
        },
    }
    bad = set(kwargs) & FORBIDDEN_REQUEST_KEYS
    unknown = set(kwargs) - ALLOWED_REQUEST_KEYS
    if bad or unknown:
        raise LLMError(f"refusing request with keys {sorted(bad | unknown)}")
    return kwargs


class VLLMBackend:
    """OpenAI-compatible client pointed at the per-job vLLM server on 127.0.0.1."""

    def __init__(self, model_id: str, model_revision: str | None, *, host: str, port: int,
                 api_key: str | None = None, timeout_s: float = 120.0, client: Any = None):
        check_host(host)
        if not isinstance(port, int) or not 1024 <= port <= 65535:
            raise LLMError(f"invalid port {port!r}")
        self.model_id = model_id
        self.model_revision = model_revision
        if client is None:
            key = api_key or os.environ.get(VLLM_KEY_ENV)
            if not key:
                raise LLMError(f"{VLLM_KEY_ENV} not set")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMError("openai client not installed (cluster extra)") from exc
            client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key=key, timeout=timeout_s, max_retries=0)
        self._client = client

    def complete(self, req: ChatRequest) -> Completion:
        resp = self._client.chat.completions.create(**build_request(self.model_id, req))
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        return Completion(
            text,
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or count_tokens(text)),
            str(choice.finish_reason or "unknown"),
        )


class OllamaBackend(VLLMBackend):
    """OpenAI-compatible client pointed at a local Ollama server on 127.0.0.1 (local mode only).

    Same request as vLLM (chat messages, sampling incl. seed, JSON-schema response_format,
    never tools). Ollama ignores the API key; the client library needs a non-empty one, so a
    fixed placeholder is used (it is not a secret and never enters a prompt).
    """

    PLACEHOLDER_KEY = "ollama-local"

    def __init__(self, model_tag: str, *, host: str, port: int, timeout_s: float = 300.0, client: Any = None):
        check_host(host)
        if not isinstance(port, int) or not 1024 <= port <= 65535:
            raise LLMError(f"invalid port {port!r}")
        self.model_id = model_tag
        self.model_revision = f"ollama:{model_tag}"
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMError("openai client not installed (pip install -e '.[local]')") from exc
            client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key=self.PLACEHOLDER_KEY,
                            timeout=timeout_s, max_retries=0)
        self._client = client


# ----------------------------------------------------------------------------- OpenRouter (api track)

OPENROUTER_BASE = f"https://{OPENROUTER_HOST}/api/v1"
MAX_KEY_FILE_BYTES = 4096


class SpendCapError(LLMError):
    """The API spend cap is reached: no further requests are sent."""


def check_api_url(url: str) -> str:
    """Egress allowlist: only https://openrouter.ai (default port, no credentials in the URL)."""
    from urllib.parse import urlsplit

    u = urlsplit(str(url))
    if u.scheme != "https" or u.hostname != OPENROUTER_HOST or u.port not in (None, 443) or u.username or u.password:
        raise LLMError(f"egress refused: only https://{OPENROUTER_HOST} is allowed")
    return str(url)


def read_api_key(path: str | os.PathLike) -> str:
    """Read the OpenRouter key from a private file. The key never comes from the environment and
    is never logged; errors never include it."""
    from pathlib import Path

    p = Path(os.path.expanduser(str(path)))
    try:
        st = p.lstat()
    except OSError as exc:
        raise LLMError(f"API key file {p} not readable: {type(exc).__name__}") from exc
    import stat as _stat

    if _stat.S_ISLNK(st.st_mode) or not _stat.S_ISREG(st.st_mode):
        raise LLMError(f"API key file {p} must be a regular file (not a symlink)")
    if st.st_mode & 0o077:
        raise LLMError(f"API key file {p} must be private: chmod 600 {p}")
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise LLMError(f"API key file {p} must be owned by you")
    resolved = p.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise LLMError("API key file must be outside the repository")
    if st.st_size == 0 or st.st_size > MAX_KEY_FILE_BYTES:
        raise LLMError(f"API key file {p} is empty or too large")
    key = p.read_text(encoding="utf-8").strip()
    if not key or any(c.isspace() for c in key):
        raise LLMError(f"API key file {p} must contain exactly one key")
    return key


def _api_http(timeout_s: float, hosts: set[str] | None = None, http: Any = None):
    """HTTP client whose every request is checked against the egress allowlist before sending.
    Uses httpx2 (what current openai releases depend on) or, if absent, httpx."""
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx

    def guard(request):
        check_api_url(str(request.url))
        if hosts is not None:
            hosts.add(request.url.host)

    if http is not None:
        http.event_hooks.setdefault("request", []).insert(0, guard)
        return http
    return httpx.Client(timeout=timeout_s, follow_redirects=False, trust_env=False,
                        event_hooks={"request": [guard]})


def fetch_key_usage(key: str, *, timeout_s: float = 30.0, http: Any = None) -> dict[str, float | None]:
    """Authoritative spend for this key from OpenRouter (GET /api/v1/key): credits = USD."""
    client = _api_http(timeout_s, http=http)
    url = check_api_url(f"{OPENROUTER_BASE}/key")
    try:
        resp = client.get(url, headers={"Authorization": f"Bearer {key}"})
    except Exception as exc:  # noqa: BLE001 - never echo request details (they carry the key)
        raise LLMError(f"OpenRouter key-usage request failed: {type(exc).__name__}") from None
    if resp.status_code != 200:
        raise LLMError(f"OpenRouter key-usage request failed: HTTP {resp.status_code}")
    data = (resp.json() or {}).get("data") or {}

    def num(k):
        v = data.get(k)
        return float(v) if isinstance(v, (int, float)) else None

    usage = num("usage")
    if usage is None:
        raise LLMError("OpenRouter key-usage response has no numeric data.usage")
    return {"usage": usage, "limit": num("limit"), "limit_remaining": num("limit_remaining")}


def key_usage_from_file(key_file: str, *, timeout_s: float = 30.0, http: Any = None) -> dict[str, float | None]:
    return fetch_key_usage(read_api_key(key_file), timeout_s=timeout_s, http=http)


class OpenRouterBackend:
    """OpenAI-compatible client for https://openrouter.ai/api/v1 (api mode only).

    - Egress: an httpx request hook refuses any URL other than https://openrouter.ai; no redirects;
      proxies from the environment are ignored (trust_env=False).
    - Providers are pinned: provider.order = provider.only = the model's list, allow_fallbacks=False,
      require_parameters=True (seed + JSON schema must be honoured), data_collection="deny".
    - Spend: usage.cost of every response is added up; before each request the key's authoritative
      usage at start plus this process's spend must be below `usd_hard`, else SpendCapError.
    - Never tools or plugins (build_request + FORBIDDEN_REQUEST_KEYS).
    """

    def __init__(self, model_id: str, providers: Sequence[str], *, key_file: str, usd_hard: float,
                 timeout_s: float = 120.0, client: Any = None, http: Any = None, baseline_usd: float | None = None):
        if not providers:
            raise LLMError("openrouter backend needs pinned providers")
        self.model_id = model_id
        self.providers = list(providers)
        self.model_revision = "openrouter:" + ",".join(self.providers)
        self.usd_hard = float(usd_hard)
        self.hosts: set[str] = set()
        self.spend_usd = 0.0
        self.requests = 0
        key = read_api_key(key_file)
        self._http = _api_http(timeout_s, self.hosts, http)
        self.baseline_usd = (baseline_usd if baseline_usd is not None
                             else float(fetch_key_usage(key, timeout_s=timeout_s, http=self._http)["usage"] or 0.0))
        if self.baseline_usd >= self.usd_hard:
            raise SpendCapError(f"API spend cap reached before start: ${self.baseline_usd:.2f} >= ${self.usd_hard:.2f}")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMError("openai client not installed (pip install -e '.[api]')") from exc
            client = OpenAI(base_url=OPENROUTER_BASE, api_key=key, http_client=self._http, timeout=timeout_s,
                            max_retries=0)
        self._key = key  # in memory only: needed for GET /key; never logged, cached or returned
        self._timeout_s = timeout_s
        self._client = client

    def key_usage_now(self) -> float | None:
        """Authoritative total usage of the key right now (None if it cannot be read)."""
        try:
            return float(fetch_key_usage(self._key, timeout_s=self._timeout_s, http=self._http)["usage"] or 0.0)
        except LLMError:
            return None

    def __repr__(self) -> str:  # never show the key
        return f"OpenRouterBackend(model_id={self.model_id!r}, providers={self.providers!r})"

    def provider_block(self) -> dict[str, Any]:
        return {"order": list(self.providers), "only": list(self.providers), "allow_fallbacks": False,
                "require_parameters": True, "data_collection": "deny"}

    def complete(self, req: ChatRequest) -> Completion:
        if self.baseline_usd + self.spend_usd >= self.usd_hard:
            raise SpendCapError(f"API spend cap reached: ${self.baseline_usd + self.spend_usd:.2f} >= "
                                f"${self.usd_hard:.2f}")
        kwargs = build_request(self.model_id, req)
        resp = self._client.chat.completions.create(**kwargs, extra_body={"provider": self.provider_block()})
        self.requests += 1
        usage = getattr(resp, "usage", None)
        cost = getattr(usage, "cost", None)
        if cost is None and usage is not None and getattr(usage, "model_extra", None):
            cost = usage.model_extra.get("cost")
        if isinstance(cost, (int, float)) and cost > 0:
            self.spend_usd += float(cost)
        choice = resp.choices[0]
        text = choice.message.content or ""
        return Completion(
            text,
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or count_tokens(text)),
            str(choice.finish_reason or "unknown"),
        )

    def api_summary(self) -> dict[str, Any]:
        return {"hosts": sorted(self.hosts), "spend_usd": self.spend_usd, "baseline_usd": self.baseline_usd,
                "usd_hard": self.usd_hard, "providers": list(self.providers), "requests": self.requests}


OLLAMA_DEFAULT_PORT = 11434


def make_backend(cfg: Config, *, model_id: str | None = None, model_revision: str | None = None,
                 port: int | None = None, stub_responder: Responder | None = None,
                 providers: Sequence[str] | None = None) -> Backend:
    if cfg.run.backend == "stub":
        return StubBackend(responder=stub_responder or default_responder)
    if cfg.run.backend == "openrouter":
        if cfg.run.mode != "api":
            raise LLMError("openrouter backend is refused outside api mode")
        if not model_id or not providers:
            raise LLMError("openrouter backend needs a model id and pinned providers")
        return OpenRouterBackend(model_id, providers, key_file=cfg.api.key_file, usd_hard=cfg.api.usd_hard,
                                 timeout_s=cfg.api.timeout_s)
    if cfg.run.backend == "ollama":
        if cfg.run.mode != "local":
            raise LLMError("ollama backend is refused outside local mode")
        if not model_id:
            raise LLMError("an ollama tag is required for the ollama backend")
        return OllamaBackend(model_id, host=cfg.llm.host, port=port or cfg.llm.port or OLLAMA_DEFAULT_PORT)
    if cfg.run.mode != "cluster":
        raise LLMError("vllm backend is refused outside cluster mode")
    if not model_id:
        raise LLMError("model_id required for vllm backend")
    return VLLMBackend(model_id, model_revision, host=cfg.llm.host, port=port or cfg.llm.port or 0)


# ----------------------------------------------------------------------------- client with cache


class LLMClient:
    """Backend + response cache. `cache_only=True` (replay) never calls the backend."""

    def __init__(self, backend: Backend | None, cache: ResponseCache | None, *, cache_only: bool = False,
                 model_id: str | None = None, model_revision: str | None = None):
        if backend is None and not cache_only:
            raise LLMError("a backend is required unless cache_only")
        if cache_only and cache is None:
            raise LLMError("cache_only requires a cache")
        self.backend = backend
        self.cache = cache
        self.cache_only = cache_only
        self.model_id = model_id or (backend.model_id if backend else "")
        self.model_revision = model_revision if model_id else (backend.model_revision if backend else None)

    def complete(self, req: ChatRequest) -> Completion:
        key = cache_key(self.model_id, self.model_revision, req)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return Completion(hit.text, hit.prompt_tokens, hit.completion_tokens, hit.finish_reason, cached=True)
        if self.cache_only:
            raise CacheMiss(f"no cached response for {req.agent_id} round {req.round}")
        assert self.backend is not None
        out = self.backend.complete(req)
        if self.cache is not None:
            self.cache.put(key, self.model_id,
                           CachedResponse(out.text, out.prompt_tokens, out.completion_tokens, out.finish_reason))
        return out

    def complete_many(self, reqs: Sequence[ChatRequest], concurrency: int = 1) -> list[Completion]:
        """Batch one round. Results are returned in request order regardless of concurrency."""
        if concurrency <= 1 or len(reqs) <= 1:
            return [self.complete(r) for r in reqs]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(self.complete, reqs))
