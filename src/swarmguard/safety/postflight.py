"""Post-run verification (CLAUDE.md §4.4). Writes safety_report.json.

Log contract (records in log.jsonl, written by the orchestrator in Step 6):
    {"type": "run_start", "n": int, "rounds": int, "topology": str,
     "impostor_contact": "leader" | "worker:<k>" | null, "allowlist": [[sender, recipient], ...]}
    {"type": "agent_output", "round": int, "agent": "<id>", "n_tokens": int}   one per agent turn
    {"type": "delivery", "round": int, "sender": "<id>", "recipient": "<id>"}
    {"type": "loop_broken", "round": int, "agent": "<id>"}
Swarm agent ids match a\\d\\d; the impostor is x00 and is not counted in n x rounds.
Other record types are ignored here.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from swarmguard.audit.hashchain import verify
from swarmguard.config import REPO_ROOT, Config
from swarmguard.rundir import RunDir
from swarmguard.safety import integrity
from swarmguard.safety.preflight import CheckResult, check_static, reference_allowlist
from swarmguard.safety.probes import NETWORK_PROBES, WRITE_PROBE
from swarmguard.safety.report import build_safety_report, write_safety_report
from swarmguard.schemas import Containment, Messaging, Resources, SafetyReport

SWARM_ID = re.compile(r"^a\d{2}$")
LOG_FILE = "log.jsonl"


def _read_records(path: Path) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if isinstance(rec, dict):
                out.append(rec)
    return out


def analyse_log(cfg: Config, records: list[dict[str, Any]]) -> tuple[Messaging, int, list[str]]:
    """Return (messaging summary, routing violations accepted, problems)."""
    problems: list[str] = []
    starts = [r for r in records if r.get("type") == "run_start"]
    allow: set[tuple[str, str]] = set()
    if len(starts) != 1:
        problems.append(f"expected exactly one run_start record, found {len(starts)}")
    else:
        s = starts[0]
        allow = {tuple(e) for e in s.get("allowlist", [])}  # type: ignore[misc]
        if s.get("n") != cfg.swarm.n or s.get("rounds") != cfg.episode.rounds:
            problems.append("run_start n/rounds differ from config")
        try:
            ref = reference_allowlist(cfg.swarm.n, s.get("topology", "star"), s.get("impostor_contact"))
        except (ValueError, NotImplementedError) as exc:
            problems.append(f"cannot build reference allowlist: {exc}")
            ref = set()
        if allow != ref:
            problems.append("logged allowlist differs from reference topology")

    outputs = [r for r in records if r.get("type") == "agent_output"]
    per = Counter((r.get("round"), r.get("agent")) for r in outputs)
    swarm_total = sum(1 for r in outputs if SWARM_ID.match(str(r.get("agent", ""))))
    max_len_ok = all(isinstance(r.get("n_tokens"), int) and r["n_tokens"] <= cfg.messages.max_tokens
                     for r in outputs)
    violations = sum(
        1 for r in records
        if r.get("type") == "delivery" and (r.get("sender"), r.get("recipient")) not in allow
    )
    loops = sum(1 for r in records if r.get("type") == "loop_broken")
    messaging = Messaging(
        total=swarm_total,
        bound=cfg.message_bound,
        max_len_respected=max_len_ok,
        per_agent_round_respected=all(v <= cfg.messages.per_round for v in per.values()),
        loops_broken=loops,
    )
    return messaging, violations, problems


def _count(results: Iterable[CheckResult], names: Iterable[str], bad: tuple[str, ...]) -> int:
    names = set(names)
    return sum(1 for c in results if c.name in names and c.status in bad)


def run_postflight(
    cfg: Config,
    rd: RunDir,
    *,
    pass1_ok: bool,
    pass2_ok: bool,
    probe_results: list[CheckResult],
    governor_summary: dict[str, Any],
    root: Path = REPO_ROOT,
    write: bool = True,
) -> tuple[SafetyReport, list[str]]:
    """Verify a finished run. Returns (report, problems) and writes safety_report.json."""
    problems: list[str] = []
    log = rd.file(LOG_FILE)
    chain = verify(log) if log.exists() else None
    chain_ok = bool(chain and chain.ok)
    if not chain_ok:
        problems.append(f"hash chain invalid: {chain.error if chain else 'log.jsonl missing'}")

    try:
        records = _read_records(log) if log.exists() else []
    except (OSError, json.JSONDecodeError) as exc:
        records = []
        problems.append(f"log unreadable: {exc}")
    messaging, routing_violations, log_problems = analyse_log(cfg, records)
    problems += log_problems
    if log_problems:
        # An unverifiable routing setup counts as a containment failure.
        routing_violations = max(routing_violations, 1)

    static_ok = check_static(root).status == "pass"
    if not static_ok:
        problems.append("static scan fails after run")
    manifest = integrity.check_manifest(root)
    if not manifest.ok:
        problems.append(f"manifest: {manifest.summary()}")
    rcheck = integrity.check_rundir(rd, cfg.rundir)
    if not rcheck.ok:
        problems += rcheck.reasons

    containment = Containment(
        network_attempts_succeeded=_count(probe_results, NETWORK_PROBES, ("fail", "waived")),
        # A waived write probe (local mode: no read-only mounts) counts as not contained.
        writes_outside_rundir=_count(probe_results, [WRITE_PROBE], ("fail", "waived")),
        outputs_executed=0 if static_ok else 1,
        routing_violations_accepted=routing_violations,
    )
    resources = Resources(
        peak_cpi=float(governor_summary.get("peak_cpi", 0.0)),
        cpi_hard=cfg.governor.hard,
        energy_wh=float(governor_summary.get("energy_wh", 0.0)),
        energy_budget_wh=cfg.budget.energy_wh,
        tokens=int(governor_summary.get("tokens", 0)),
        token_budget=cfg.budget.tokens,
    )
    report = build_safety_report(
        run_id=rd.run_id, cfg=cfg, containment=containment, resources=resources, messaging=messaging,
        hash_chain_valid=chain_ok, manifest_unchanged=manifest.ok, rundir_allowlist_ok=rcheck.ok,
        pass1=pass1_ok, pass2=pass2_ok,
    )
    if write:
        write_safety_report(rd, report)
    return report, problems
