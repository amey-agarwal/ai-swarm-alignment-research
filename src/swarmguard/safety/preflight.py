"""Pass 1: config assertions and static checks (CLAUDE.md §4.3).

Every check returns a CheckResult with status "pass", "fail", or "skipped".
"skipped" is only acceptable in dev mode; in cluster mode any skipped check
fails the pass. There is no flag, env var, or code path that disables a check.

Contract with the orchestrator (Step 6):
    swarmguard.orchestrator.topology.build_allowlist(n, kind, impostor_contact=None)
        -> set[tuple[str, str]] of (sender_id, recipient_id)
    Agent ids: leader "a00", workers "a01".."a{n-1:02d}", impostor "x00".
    The result must equal `reference_allowlist(...)` below.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import ValidationError

from swarmguard.config import LOCALHOST, REPO_ROOT, Config, load_config, load_models
from swarmguard.safety import integrity

Status = Literal["pass", "fail", "skipped", "waived"]

LEADER_ID = "a00"
IMPOSTOR_ID = "x00"
VLLM_KEY_ENV = "SWARM_VLLM_API_KEY"  # per-job vLLM key; never enters a prompt
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str = ""


@dataclass
class PassResult:
    name: str
    mode: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        for c in self.checks:
            if c.status == "fail":
                return False
            if c.status == "skipped" and self.mode != "dev":
                return False
        return bool(self.checks)

    def get(self, name: str) -> CheckResult:
        for c in self.checks:
            if c.name == name:
                return c
        raise KeyError(name)


def _r(name: str, ok: bool, detail: str = "") -> CheckResult:
    return CheckResult(name, "pass" if ok else "fail", detail)


# ----------------------------------------------------------------------------- routing reference


def agent_id(i: int) -> str:
    return f"a{i:02d}"


def contact_id(n: int, impostor_contact: str | None) -> str | None:
    if impostor_contact is None or impostor_contact == "none":
        return None
    if impostor_contact == "leader":
        return LEADER_ID
    m = re.fullmatch(r"worker:(\d+)", impostor_contact)
    if not m or not (1 <= int(m.group(1)) <= n - 1):
        raise ValueError(f"invalid impostor contact {impostor_contact!r} for n={n}")
    return agent_id(int(m.group(1)))


def reference_allowlist(n: int, kind: str = "star", impostor_contact: str | None = None) -> set[tuple[str, str]]:
    """Independent reference of who may message whom. Star: workers <-> leader only."""
    if kind != "star":
        raise NotImplementedError(f"topology {kind!r} has no reference allowlist yet")
    edges: set[tuple[str, str]] = set()
    for i in range(1, n):
        edges.add((LEADER_ID, agent_id(i)))
        edges.add((agent_id(i), LEADER_ID))
    c = contact_id(n, impostor_contact)
    if c is not None:
        edges.add((IMPOSTOR_ID, c))
        edges.add((c, IMPOSTOR_ID))
    return edges


# ----------------------------------------------------------------------------- static scan

FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile", "__import__"})
FORBIDDEN_MODULES = frozenset(
    {"subprocess", "pickle", "_pickle", "cPickle", "shelve", "marshal", "pty", "ctypes", "code", "codeop"}
)
SOCKET_MODULES = frozenset({"socket", "ssl", "socketserver"})
NETWORK_MODULES = frozenset(
    {"requests", "urllib", "urllib3", "httpx", "http", "aiohttp", "openai", "ftplib", "smtplib",
     "telnetlib", "xmlrpc", "websocket", "websockets", "paramiko"}
)
FORBIDDEN_OS_ATTRS = frozenset(
    {"system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp", "startfile", "execv",
     "execve", "execl", "execle", "execlp", "execlpe", "execvp", "execvpe", "spawnl", "spawnle",
     "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"}
)
TOOL_PARAM_NAMES = frozenset({"tools", "tool_choice", "functions", "function_call", "parallel_tool_calls"})

SOCKET_EXEMPT = frozenset({"safety/probes.py"})
NETWORK_EXEMPT = frozenset({"agents/llm_client.py"})


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    rule: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.rule}"


class _Scanner(ast.NodeVisitor):
    def __init__(self, rel: str):
        self.rel = rel
        self.out: list[Violation] = []

    def _v(self, node: ast.AST, rule: str) -> None:
        self.out.append(Violation(self.rel, getattr(node, "lineno", 0), rule))

    def _check_module(self, node: ast.AST, mod: str) -> None:
        top = mod.split(".")[0]
        if top in FORBIDDEN_MODULES:
            self._v(node, f"forbidden import {mod}")
        if top in SOCKET_MODULES and self.rel not in SOCKET_EXEMPT:
            self._v(node, f"socket import {mod} outside safety/probes.py")
        if top in NETWORK_MODULES and self.rel not in NETWORK_EXEMPT:
            self._v(node, f"network import {mod} outside agents/llm_client.py")

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self._check_module(node, a.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module:
            self._check_module(node, node.module)
            if node.module == "os":
                for a in node.names:
                    if a.name in FORBIDDEN_OS_ATTRS:
                        self._v(node, f"forbidden import os.{a.name}")
            if node.module == "builtins":
                for a in node.names:
                    if a.name in FORBIDDEN_CALLS:
                        self._v(node, f"forbidden import builtins.{a.name}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_CALLS:
            self._v(node, f"forbidden builtin {node.id}")
        if node.id in ("__builtins__", "builtins"):
            self._v(node, f"access to {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "os" and node.attr in FORBIDDEN_OS_ATTRS:
            self._v(node, f"forbidden os.{node.attr}")
        # Note: `.compile` etc. on other objects (e.g. re.compile) is fine; builtins access
        # is caught in visit_Name via the `builtins` / `__builtins__` names.
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        for kw in node.keywords:
            if kw.arg in TOOL_PARAM_NAMES:
                self._v(node, f"tool/function-calling parameter {kw.arg}=")
        if isinstance(node.func, ast.Name) and node.func.id in ("getattr", "setattr", "hasattr"):
            for arg in node.args[1:2]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value in FORBIDDEN_CALLS | FORBIDDEN_OS_ATTRS:
                        self._v(node, f"{node.func.id}(..., {arg.value!r})")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for k in node.keys:
            if isinstance(k, ast.Constant) and k.value in TOOL_PARAM_NAMES:
                self._v(node, f"tool/function-calling key {k.value!r}")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        s = node.slice
        if isinstance(s, ast.Constant) and s.value in TOOL_PARAM_NAMES:
            self._v(node, f"tool/function-calling key [{s.value!r}]")
        self.generic_visit(node)


def scan_source(source: str, rel: str) -> list[Violation]:
    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as exc:
        return [Violation(rel, exc.lineno or 0, f"unparseable: {exc.msg}")]
    s = _Scanner(rel)
    s.visit(tree)
    return s.out


def scan_package(pkg_root: Path) -> list[Violation]:
    pkg_root = Path(pkg_root)
    if not pkg_root.is_dir():
        return [Violation(str(pkg_root), 0, "package directory missing")]
    out: list[Violation] = []
    for p in sorted(pkg_root.rglob("*")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(pkg_root).as_posix()
        if p.is_symlink():
            out.append(Violation(rel, 0, "symlink in package"))
        elif p.suffix in (".pyc", ".pyo", ".so", ".pyd"):
            out.append(Violation(rel, 0, "compiled/binary module in package"))
        elif p.suffix == ".py":
            out.extend(scan_source(p.read_text(encoding="utf-8"), rel))
    return out


# ----------------------------------------------------------------------------- individual checks


def check_mode_env(cfg: Config, env: Mapping[str, str]) -> CheckResult:
    if cfg.run.mode == "dev":
        if cfg.run.backend != "stub":
            return _r("mode_backend_env", False, "dev mode requires the stub backend")
        return _r("mode_backend_env", True, "dev + stub")
    problems = []
    if not env.get("SLURM_JOB_ID"):
        problems.append("SLURM_JOB_ID unset (not inside a Slurm job)")
    if not (env.get("APPTAINER_CONTAINER") or env.get("SINGULARITY_CONTAINER")):
        problems.append("not inside an Apptainer container")
    return _r("mode_backend_env", not problems, "; ".join(problems) or f"cluster + {cfg.run.backend}, Slurm job + container")


def check_offline_env(cfg: Config, env: Mapping[str, str]) -> CheckResult:
    if cfg.run.mode == "dev":
        return CheckResult("offline_env", "skipped", "skipped (dev)")
    need = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    bad = [k for k, v in need.items() if env.get(k) != v]
    return _r("offline_env", not bad, f"not set to 1: {bad}" if bad else "offline env vars set")


def check_llm_host(cfg: Config) -> CheckResult:
    return _r("llm_host", cfg.llm.host == LOCALHOST, f"host={cfg.llm.host}")


def check_swarm_caps(cfg: Config) -> CheckResult:
    ok = cfg.swarm.n <= cfg.swarm.n_max and cfg.swarm.spawning is False and cfg.messages.per_round == 1
    return _r("swarm_caps", ok, f"n={cfg.swarm.n} <= n_max={cfg.swarm.n_max}, spawning={cfg.swarm.spawning}")


def check_models(cfg: Config) -> CheckResult:
    try:
        models = load_models()
    except (OSError, ValueError) as exc:
        return _r("models", False, f"models.yaml invalid: {exc}")
    sw, jd = models.get(cfg.model.swarm), models.get(cfg.model.judge)
    problems = []
    if sw is None or sw.role != "swarm":
        problems.append(f"swarm model {cfg.model.swarm!r} not a swarm entry in models.yaml")
    if jd is None or jd.role != "judge":
        problems.append(f"judge model {cfg.model.judge!r} not a judge entry in models.yaml")
    if sw and jd and sw.hf_id.split("/")[0].lower() == jd.hf_id.split("/")[0].lower():
        problems.append("judge must be a different model family from the swarm model")
    if cfg.run.mode == "cluster" and cfg.run.backend == "vllm":
        for m in (sw, jd):
            if m is not None and not m.revision:
                problems.append(f"{m.hf_id}: revision not pinned (required in cluster mode)")
    detail = "; ".join(problems) or f"swarm={sw.hf_id}, judge={jd.hf_id}"  # type: ignore[union-attr]
    return _r("models", not problems, detail)


def check_routing(cfg: Config) -> CheckResult:
    try:
        ref = reference_allowlist(cfg.swarm.n, cfg.topology.kind)
    except NotImplementedError as exc:
        return _r("routing", False, str(exc))
    try:
        from swarmguard.orchestrator import topology  # Step 6
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("swarmguard.orchestrator"):
            return CheckResult("routing", "skipped", "orchestrator not built yet (Step 6)")
        return _r("routing", False, f"import error: {exc}")
    try:
        got = set(topology.build_allowlist(cfg.swarm.n, cfg.topology.kind))
        got_imp = set(topology.build_allowlist(cfg.swarm.n, cfg.topology.kind, impostor_contact="leader"))
    except Exception as exc:  # noqa: BLE001 - any failure fails closed
        return _r("routing", False, f"build_allowlist failed: {exc!r}")
    ref_imp = reference_allowlist(cfg.swarm.n, cfg.topology.kind, impostor_contact="leader")
    ok = got == ref and got_imp == ref_imp
    detail = f"{len(got)} edges match reference" if ok else (
        f"mismatch: extra={sorted(got - ref)[:5]} missing={sorted(ref - got)[:5]}"
    )
    return _r("routing", ok, detail)


def check_canary(cfg: Config) -> CheckResult:
    exists = os.path.lexists(cfg.sandbox.canary_path)
    return _r("canary_absent", not exists, f"{cfg.sandbox.canary_path} {'EXISTS' if exists else 'absent'}")


def secret_like_env_names(env: Mapping[str, str]) -> list[str]:
    return sorted(k for k in env if k != VLLM_KEY_ENV and any(m in k.upper() for m in SECRET_MARKERS))


def check_secrets(env: Mapping[str, str]) -> CheckResult:
    bad = secret_like_env_names(env)
    return _r("no_secrets", not bad, f"secret-like env var names: {bad}" if bad else "none")


def check_static(root: Path) -> CheckResult:
    v = scan_package(Path(root) / "src" / "swarmguard")
    return _r("static_scan", not v, "; ".join(map(str, v[:5])) + (" ..." if len(v) > 5 else "") if v else "clean")


def check_manifest(root: Path) -> CheckResult:
    d = integrity.check_manifest(Path(root))
    return _r("manifest", d.ok, d.summary())


# ----------------------------------------------------------------------------- pass 1


def resolve_config(
    source: Config | Mapping[str, Any] | str | Path | None, mode: str | None
) -> tuple[Config | None, str]:
    """Validate the config. Returns (config or None, detail)."""
    overrides = {"run": {"mode": mode}} if mode else None
    try:
        if isinstance(source, Config):
            data = source.model_dump(mode="json")
            if mode:
                data["run"]["mode"] = mode
            return Config.model_validate(data), "valid"
        if isinstance(source, Mapping):
            merged = dict(source)
            if mode:
                merged = {**merged, "run": {**dict(merged.get("run", {})), "mode": mode}}
            return load_config(overrides=merged), "valid"
        return load_config(source, overrides=overrides), "valid"
    except (ValidationError, ValueError, OSError) as exc:
        msg = str(exc).replace("\n", " ")
        return None, msg[:300]


def run_pass1(
    source: Config | Mapping[str, Any] | str | Path | None = None,
    mode: str | None = None,
    *,
    root: Path = REPO_ROOT,
    env: Mapping[str, str] | None = None,
) -> PassResult:
    env = os.environ if env is None else env
    cfg, detail = resolve_config(source, mode)
    effective_mode = cfg.run.mode if cfg else (mode or "cluster")  # unknown -> strictest
    res = PassResult("pass1", effective_mode)
    res.checks.append(_r("config_schema", cfg is not None, detail))
    if cfg is not None:
        res.checks += [
            check_mode_env(cfg, env),
            check_offline_env(cfg, env),
            check_llm_host(cfg),
            check_swarm_caps(cfg),
            check_models(cfg),
            check_routing(cfg),
            check_canary(cfg),
        ]
    else:
        for name in ("mode_backend_env", "offline_env", "llm_host", "swarm_caps", "models", "routing", "canary_absent"):
            res.checks.append(_r(name, False, "config invalid"))
    res.checks += [check_secrets(env), check_static(root), check_manifest(root)]
    return res
