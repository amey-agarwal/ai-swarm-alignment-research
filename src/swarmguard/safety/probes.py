"""Pass 2: live probes of actions that must FAIL (CLAUDE.md §4.3).

dev mode: container/Slurm probes (network, writes outside the run dir) are reported
as "skipped (dev)" and are not attempted, so dev runs make no network requests.
cluster mode: every probe runs inside a Slurm job; a skipped probe fails the pass. Outside a
Slurm job, cluster-mode network/write probes are failed without being attempted.

If the platform cannot isolate the network, the network probes fail. The only
fallback is `sandbox.network_isolation: unavailable` plus a written justification,
which turns a succeeded network probe into "waived" (recorded, never silent).

Contracts with later steps (probes fail closed if violated; skipped in dev until built):
    Step 5  swarmguard.agents.parser.parse_worker_output(raw: str, *, allowed_recipients,
                max_tokens: int) -> WorkerMessage, raising swarmguard.agents.parser.ParseError
    Step 6  swarmguard.orchestrator.bus.Bus(allowlist: set[tuple[str, str]])
                .send(sender: str, recipient: str, content: str), raising
                swarmguard.orchestrator.bus.RoutingError for a disallowed edge
"""

from __future__ import annotations

import os
import secrets
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from swarmguard.config import REPO_ROOT, Config
from swarmguard.rundir import RunDir
from swarmguard.safety.governor import Governor, GovernorError, make_provider
from swarmguard.safety.killswitch import KillSwitch, request_stop
from swarmguard.safety.preflight import CheckResult, PassResult, reference_allowlist

EXTERNAL_TCP = ("1.1.1.1", 443)
EXTERNAL_HOST = "example.com"
NETWORK_PROBES = ("net_tcp_external", "net_dns_external")
WRITE_PROBE = "write_outside_rundir"
SKIP_DEV = "skipped (dev)"


def _pass(name: str, detail: str) -> CheckResult:
    return CheckResult(name, "pass", detail)


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name, "fail", detail)


def _not_built(name: str, mode: str, what: str) -> CheckResult:
    if mode == "dev":
        return CheckResult(name, "skipped", f"{what} not built yet")
    return _fail(name, f"{what} not available")


def _outside_job(env: Any = None) -> bool:
    """True when not inside a Slurm job. Cluster-mode probes are then failed without being
    attempted, so a mistaken `--mode cluster` on a laptop never touches the network."""
    env = os.environ if env is None else env
    return not env.get("SLURM_JOB_ID")


NOT_IN_JOB = "not attempted: not inside a Slurm job (fail closed)"


def _network_outcome(name: str, cfg: Config, succeeded: bool, detail: str) -> CheckResult:
    if not succeeded:
        return _pass(name, detail)
    if cfg.sandbox.network_isolation == "unavailable":
        return CheckResult(name, "waived", f"{detail}; WAIVED: {cfg.sandbox.network_isolation_justification}")
    return _fail(name, detail)


# ----------------------------------------------------------------------------- probes


def probe_tcp(cfg: Config, mode: str, connect: Callable[..., Any] = socket.create_connection,
              target: tuple[str, int] = EXTERNAL_TCP, timeout: float = 2.0, env: Any = None) -> CheckResult:
    name = "net_tcp_external"
    if mode == "dev":
        return CheckResult(name, "skipped", SKIP_DEV)
    if _outside_job(env):
        return _fail(name, NOT_IN_JOB)
    try:
        conn = connect(target, timeout=timeout)
    except OSError as exc:
        return _network_outcome(name, cfg, False, f"blocked: {type(exc).__name__}")
    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass
    return _network_outcome(name, cfg, True, f"CONNECTED to {target[0]}:{target[1]}")


def probe_dns(cfg: Config, mode: str, resolve: Callable[..., Any] = socket.getaddrinfo,
              host: str = EXTERNAL_HOST, env: Any = None) -> CheckResult:
    name = "net_dns_external"
    if mode == "dev":
        return CheckResult(name, "skipped", SKIP_DEV)
    if _outside_job(env):
        return _fail(name, NOT_IN_JOB)
    try:
        resolve(host, 443)
    except OSError as exc:
        return _network_outcome(name, cfg, False, f"blocked: {type(exc).__name__}")
    return _network_outcome(name, cfg, True, f"RESOLVED {host}")


def default_write_targets() -> list[Path]:
    targets = [REPO_ROOT]
    home = os.environ.get("HOME")
    if home and Path(home).is_dir():
        targets.append(Path(home))
    return targets


def probe_write_outside(mode: str, targets: list[Path] | None = None, env: Any = None) -> CheckResult:
    name = WRITE_PROBE
    if mode == "dev":
        return CheckResult(name, "skipped", SKIP_DEV)
    if _outside_job(env):
        return _fail(name, NOT_IN_JOB)
    targets = default_write_targets() if targets is None else targets
    if not targets:
        return _fail(name, "no targets to probe")
    writable = []
    for d in targets:
        p = Path(d) / f".swarmguard_probe_{secrets.token_hex(6)}"
        try:
            with open(p, "x", encoding="utf-8") as fh:
                fh.write("probe\n")
        except OSError:
            continue
        writable.append(str(d))
        try:
            p.unlink()
        except OSError:
            pass
    if writable:
        return _fail(name, f"WRITABLE: {writable}")
    return _pass(name, f"read-only: {[str(t) for t in targets]}")


def probe_bus(mode: str, bus_cls: Any = None, error_cls: Any = None) -> CheckResult:
    name = "bus_disallowed_recipient"
    if bus_cls is None:
        try:
            from swarmguard.orchestrator import bus as bus_mod  # Step 6
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("swarmguard.orchestrator"):
                return _not_built(name, mode, "orchestrator bus (Step 6)")
            return _fail(name, f"import error: {exc}")
        bus_cls = getattr(bus_mod, "Bus", None)
        error_cls = getattr(bus_mod, "RoutingError", None)
        if bus_cls is None or error_cls is None:
            return _fail(name, "bus module lacks Bus/RoutingError")
    try:
        bus = bus_cls(reference_allowlist(3, "star"))
        bus.send("a01", "a00", "control message")
    except Exception as exc:  # noqa: BLE001
        return _fail(name, f"allowed edge a01->a00 rejected: {exc!r}")
    for sender, recipient in (("a01", "a02"), ("a01", "x00"), ("x00", "a00"), ("a01", "a01")):
        try:
            bus.send(sender, recipient, "probe")
        except error_cls:
            continue
        except Exception as exc:  # noqa: BLE001
            return _fail(name, f"{sender}->{recipient} raised {exc!r}, not RoutingError")
        return _fail(name, f"disallowed edge {sender}->{recipient} ACCEPTED")
    return _pass(name, "disallowed recipients rejected")


PARSER_BAD_INPUTS = {
    "malformed": '{"to": "a00", "content": ',
    "oversized": '{"to": "a00", "content": "' + "x" * 1_000_000 + '"}',
    "extra_field": '{"to": "a00", "content": "hi", "tools": [{"type": "function"}]}',
    "wrong_recipient": '{"to": "a02", "content": "hi"}',
    "non_object": '["a00", "hi"]',
    "wrong_type": '{"to": "a00", "content": 42}',
}


def probe_parser(mode: str, parse: Callable[..., Any] | None = None, error_cls: Any = None,
                 max_tokens: int = 120) -> CheckResult:
    name = "parser_rejects_bad"
    if parse is None:
        try:
            from swarmguard.agents import parser as parser_mod  # Step 5
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("swarmguard.agents"):
                return _not_built(name, mode, "parser (Step 5)")
            return _fail(name, f"import error: {exc}")
        parse = getattr(parser_mod, "parse_worker_output", None)
        error_cls = getattr(parser_mod, "ParseError", None)
        if parse is None or error_cls is None:
            return _fail(name, "parser module lacks parse_worker_output/ParseError")
    allowed = {"a00"}
    try:
        parse('{"to": "a00", "content": "My clue: Dev drinks water."}',
              allowed_recipients=allowed, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001
        return _fail(name, f"valid control message rejected: {exc!r}")
    for label, raw in PARSER_BAD_INPUTS.items():
        try:
            parse(raw, allowed_recipients=allowed, max_tokens=max_tokens)
        except error_cls:
            continue
        except Exception as exc:  # noqa: BLE001
            return _fail(name, f"{label}: raised {exc!r}, not ParseError")
        return _fail(name, f"{label} input ACCEPTED")
    return _pass(name, f"{len(PARSER_BAD_INPUTS)} bad inputs rejected")


def probe_governor(cfg: Config, mode: str, provider: Any = None, timeout_s: float = 3.0) -> CheckResult:
    name = "governor_heartbeat"
    try:
        provider = provider if provider is not None else make_provider(mode, tempfile.gettempdir())
    except GovernorError as exc:
        return _fail(name, str(exc))
    gov = Governor(cfg, provider, interval_s=0.1)
    gov.start()
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            st = gov.state()
            if st.heartbeats >= 2 and st.last_sample is not None and gov.alive():
                return _pass(name, f"{st.heartbeats} heartbeats, provider={provider.name}, cpi={st.cpi:.3f}")
            time.sleep(0.05)
        return _fail(name, f"no heartbeat within {timeout_s}s")
    finally:
        gov.stop()


def probe_stop_file(max_rounds: int = 5) -> CheckResult:
    name = "stop_file_shutdown"
    with tempfile.TemporaryDirectory(prefix="swarmguard_probe_") as d:
        rd = RunDir.create(d)
        ks = KillSwitch(rd)
        if ks.check() is not None:
            return _fail(name, "stopped before STOP file existed")
        stop_requested_at = 1
        for rnd in range(max_rounds):
            if rnd == stop_requested_at:
                request_stop(rd, "probe")
            if rnd > stop_requested_at + 1:
                return _fail(name, "STOP file did not stop the loop within one round")
            reason = ks.check()
            if reason:
                return _pass(name, f"stopped at round {rnd} ({reason})")
        return _fail(name, "STOP file never detected")


# ----------------------------------------------------------------------------- pass 2


def run_pass2(cfg: Config, mode: str | None = None) -> PassResult:
    mode = mode or cfg.run.mode
    res = PassResult("pass2", mode)
    res.checks += [
        probe_tcp(cfg, mode),
        probe_dns(cfg, mode),
        probe_write_outside(mode),
        probe_bus(mode),
        probe_parser(mode, max_tokens=cfg.messages.max_tokens),
        probe_governor(cfg, mode),
        probe_stop_file(),
    ]
    return res
