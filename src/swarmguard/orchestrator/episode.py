"""Run one episode end to end (CLAUDE.md §6).

Writes into the run directory only:
  config.yaml  resolved config + run metadata (git SHA, package versions, model, task)
  log.jsonl    hash-chained records; contains no run id or wall-clock time, so two
               runs with the same seed produce byte-identical logs
  checkpoint.json  when stopped by kill switch / signal / governor

Every round: kill switch (STOP file, signals, governor hard stop) is checked first,
then the governor's throttle advice is applied. Safety passes 1/2 before and
postflight after are run by the CLI (Step 10), not here.

Log records written here (in addition to scheduler.py):
    run_start {n, rounds, topology, impostor_contact, allowlist, task, seed, leader, policy,
               clue_assignment, model_id, impostor, claim, impostor_clue}
    comprehension {agent, status, capabilities, registered}   (post-episode probe; never delivered)
    run_end   {stop_reason, detail, rounds_completed, submitted, success, n_outputs, tokens, trace}
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Literal

import yaml

from swarmguard import __version__
from swarmguard.agents.agent import Agent
from swarmguard.agents.llm_client import LLMClient
from swarmguard.agents.scripted_leader import ScriptedLeader
from swarmguard.impostor import MODES as IMPOSTOR_MODES
from swarmguard.impostor import make_impostor
from swarmguard.impostor.replay import ImpostorSession, Turn, load_session, write_session
from swarmguard.mitigations import parse_mitigation
from swarmguard.mitigations.controller import MitigationController
from swarmguard.audit.hashchain import HashChainWriter
from swarmguard.config import REPO_ROOT, Config
from swarmguard.orchestrator.bus import Bus
from swarmguard.orchestrator.scheduler import RoundScheduler
from swarmguard.orchestrator.topology import IMPOSTOR_ID, LEADER_ID, Topology, agent_id
from swarmguard.rundir import RunDir
from swarmguard.safety.governor import Governor
from swarmguard.safety.killswitch import KillSwitch, write_checkpoint
from swarmguard.schemas import RunReport
from swarmguard.tasks.base import TaskInstance, make_instance
from swarmguard.tasks.distribute import distribute, impostor_clue
from swarmguard.verifiers.comprehension import run_probe
from swarmguard.verifiers.tracer import Tracer

StopReason = Literal["submitted", "rounds_exhausted", "killswitch", "governor", "signal", "error"]
PACKAGES = ("swarmguard", "pydantic", "pyyaml", "typer", "openai", "vllm", "pynvml", "psutil")


@dataclass
class EpisodeSpec:
    task: str = "logic-grid:L4"
    seed: int = 0
    leader: Literal["llm", "scripted"] = "llm"
    policy: Literal["explicit", "none"] = "none"
    model_id: str = "stub"
    model_revision: str | None = None
    task_instance: TaskInstance | None = None  # override (e.g. the reference L4)
    assignment: dict[int, list[str]] | None = None  # override: worker k -> clue ids
    concurrency: int = 8
    impostor: str = "none"  # none | scripted | live | replay
    impostor_contact: str = "leader"  # leader | worker:<k>
    claim: Literal["write", "irrelevant", "none"] = "write"
    replay_from: RunDir | None = None  # source run for impostor=replay
    live_io: tuple | None = None  # (input_fn, output_fn) for impostor=live (tests)
    comprehension: bool = True  # run the post-episode comprehension probe on LLM agents
    mitigation: str = "none"  # none | dropout:<p> | leader-rotate | handoff:structured | handoff:freetext
    model_key: str = "stub"  # key in configs/models.yaml (recorded for replay/judge)
    replay_as: str | None = None  # replay a recorded impostor but log it under its original mode


@dataclass
class EpisodeResult:
    stop_reason: StopReason
    detail: str | None
    rounds_completed: int
    submitted: Any
    success: bool
    n_outputs: int
    tokens: int
    loops_broken: int = 0
    dropped: int = 0
    agents: dict[str, Agent] = field(default_factory=dict)


# ----------------------------------------------------------------------------- metadata


def read_git_sha(root: Path = REPO_ROOT) -> str | None:
    """Read HEAD from .git without running git."""
    git = root / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        p = git / ref
        if p.exists():
            return p.read_text().strip()
        packed = git / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
    except OSError:
        pass
    return None


def package_versions() -> dict[str, str]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def write_run_config(rd: RunDir, cfg: Config, spec: EpisodeSpec, task: TaskInstance) -> None:
    meta = {
        "run_id": rd.run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "swarmguard_version": __version__,
        "git_sha": read_git_sha(),
        "package_versions": package_versions(),
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "task": task.spec,
        "seed": spec.seed,
        "leader": spec.leader,
        "policy": spec.policy,
        "model_key": spec.model_key,
        "impostor": spec.replay_as or spec.impostor,
        "impostor_contact": spec.impostor_contact,
        "claim": spec.claim,
        "mitigation": spec.mitigation,
        "comprehension": spec.comprehension,
        "replay_from": spec.replay_from.run_id if spec.replay_from else None,
    }
    with rd.open("config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump({"meta": meta, "config": cfg.model_dump(mode="json")}, fh, sort_keys=True)


# ----------------------------------------------------------------------------- setup


def build_swarm(cfg: Config, task: TaskInstance, topo: Topology,
                assignment: dict[int, list[str]]) -> dict[str, Agent]:
    agents = {LEADER_ID: Agent(LEADER_ID, topo.names[LEADER_ID], "leader", [], topo.contacts(LEADER_ID))}
    for k in range(1, cfg.swarm.n):
        aid = agent_id(k)
        clues = [task.clue(c) for c in assignment.get(k, [])]
        agents[aid] = Agent(aid, topo.names[aid], "worker", clues, topo.contacts(aid))
    return agents


def _stop_kind(reason: str) -> StopReason:
    if reason.startswith("killswitch"):
        return "killswitch"
    if reason.startswith("signal"):
        return "signal"
    return "governor"


# ----------------------------------------------------------------------------- run


def run_episode(
    cfg: Config,
    spec: EpisodeSpec,
    rd: RunDir,
    client: LLMClient | None,
    *,
    governor: Governor | None = None,
    killswitch: KillSwitch | None = None,
    on_log: Callable[[HashChainWriter], None] | None = None,
) -> EpisodeResult:
    n, rounds = cfg.swarm.n, cfg.episode.rounds
    task = spec.task_instance or make_instance(spec.task, spec.seed)
    assignment = spec.assignment or distribute(task.clue_ids, n - 1, spec.seed)
    if set(assignment) - set(range(1, n)):
        raise ValueError("assignment refers to workers outside 1..n-1")
    if spec.impostor not in IMPOSTOR_MODES:
        raise ValueError(f"unknown impostor mode {spec.impostor!r}")
    contact = spec.impostor_contact if spec.impostor != "none" else None
    topo = Topology.star(n, contact)
    agents = build_swarm(cfg, task, topo, assignment)
    if len(agents) != n:
        raise AssertionError("population must be exactly n")
    imp_clue = None
    if contact is not None:
        imp_clue = task.clue(impostor_clue(assignment, spec.seed))
        agents[IMPOSTOR_ID] = Agent(IMPOSTOR_ID, topo.names[IMPOSTOR_ID], "worker", [imp_clue],
                                    topo.contacts(IMPOSTOR_ID))
    source_session = load_session(spec.replay_from) if spec.impostor == "replay" and spec.replay_from else None
    claim = source_session.claim if source_session else ("live" if spec.impostor == "live" else spec.claim)
    ks = killswitch or KillSwitch(rd)
    ms = parse_mitigation(spec.mitigation)

    def save_session(turns: list[dict]) -> None:
        write_session(rd, ImpostorSession(
            mode=spec.replay_as or spec.impostor, claim=claim, contact=spec.impostor_contact,
            replayed_from=spec.replay_from.run_id if spec.replay_from else None,
            turns=[Turn(**t) for t in turns]))

    write_run_config(rd, cfg, spec, task)
    log = HashChainWriter(rd.file("log.jsonl"))
    if on_log is not None:
        on_log(log)  # lets the governor read the logger's own byte counter
    try:
        log.append({
            "type": "run_start", "n": n, "rounds": rounds, "topology": topo.kind, "impostor_contact": contact,
            "allowlist": sorted(map(list, topo.allowlist)), "task": task.spec, "seed": spec.seed,
            "leader": spec.leader, "policy": spec.policy, "model_id": spec.model_id,
            "clue_assignment": {agent_id(k): v for k, v in sorted(assignment.items())},
            "impostor": spec.replay_as or spec.impostor, "claim": claim,
            "impostor_clue": imp_clue.id if imp_clue else None,
            "mitigation": ms.label,
        })
        scripted = {}
        if spec.leader == "scripted":
            sl = ScriptedLeader(task, rounds)
            scripted[LEADER_ID] = sl.respond
        impostor = None
        if contact is not None:
            impostor = make_impostor(spec.impostor, claim=spec.claim, clue=imp_clue, log=log,
                                     replay_from=spec.replay_from, live_io=spec.live_io, on_turn=save_session,
                                     replay_phase=spec.replay_as or "replay")
            scripted[IMPOSTOR_ID] = impostor.respond
        tracer = Tracer()
        sched = RoundScheduler(cfg, task, topo, agents, Bus(topo.allowlist), log, client, spec.policy, scripted,
                               observer=tracer.observe)
        mit = MitigationController(ms, cfg, task, topo, sched, client, log, spec.seed,
                                   make_scripted_leader=lambda: ScriptedLeader(task, rounds))

        stop: StopReason = "rounds_exhausted"
        detail: str | None = None
        submitted, completed, tokens, outputs, loops, dropped = None, 0, 0, 0, 0, 0
        for r in range(1, rounds + 1):
            reason = ks.check(governor)
            if reason:
                stop, detail = _stop_kind(reason), reason
                break
            conc, pause = governor.advice() if governor else (spec.concurrency, 0.0)
            if conc <= 0:
                stop, detail = "governor", governor.should_stop() if governor else "no capacity"
                break
            mtok = mit.before_round(r)
            active_swarm = sum(1 for a in sched.agents if a in topo.swarm_ids)
            if active_swarm > n:
                raise AssertionError("population exceeded n")
            res = sched.step(r, min(conc, spec.concurrency))
            mit.after_round(r, tracer.summary)
            completed, outputs = r, outputs + active_swarm
            tokens += res.tokens + mtok
            loops += res.loops_broken
            dropped += res.dropped
            if governor:
                governor.record_tokens(res.tokens + mtok)
                governor.record_messages(res.delivered)
            if res.submitted is not None:
                submitted, stop = res.submitted, "submitted"
                break
            if pause:
                time.sleep(pause)

        success = submitted is not None and task.check(submitted)
        agents = sched.agents
        probe_agents = [agents[a] for a in sorted(agents) if a in topo.swarm_ids and a not in sched.scripted]
        if spec.comprehension and client is not None and probe_agents and stop != "governor":
            recs, ptok = run_probe(probe_agents, sched.system_prompt, client, claim=claim, run_seed=cfg.run.seed,
                                   rounds=rounds, concurrency=spec.concurrency)
            for rec in recs:
                log.append(rec)
            tokens += ptok
            if governor:
                governor.record_tokens(ptok)
        if impostor is not None:
            save_session(impostor.turns)
        if stop in ("killswitch", "governor", "signal"):
            write_checkpoint(rd, {"round": completed, "reason": detail, "stop_reason": stop})
        log.append({"type": "run_end", "stop_reason": stop, "detail": detail, "rounds_completed": completed,
                    "submitted": submitted, "success": success, "n_outputs": outputs, "tokens": tokens,
                    "trace": tracer.summary.to_dict()})
    finally:
        log.close()
    return EpisodeResult(stop, detail, completed, submitted, success, outputs, tokens, loops, dropped, agents)


def run_report(rd: RunDir, cfg: Config, spec: EpisodeSpec, result: EpisodeResult,
               energy_wh: float = 0.0) -> RunReport:
    return RunReport(
        run_id=rd.run_id, git_sha=read_git_sha(), package_versions=package_versions(),
        model_id=spec.model_id, model_revision=spec.model_revision, seed=spec.seed,
        task=(spec.task_instance.spec if spec.task_instance else spec.task), n=cfg.swarm.n,
        rounds_planned=cfg.episode.rounds, rounds_completed=result.rounds_completed,
        stop_reason=result.stop_reason, submitted=result.submitted, success=result.success,
        n_messages=result.n_outputs, tokens=result.tokens, energy_wh=energy_wh,
    )
