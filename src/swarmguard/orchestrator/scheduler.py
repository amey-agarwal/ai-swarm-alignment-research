"""Round-synchronous scheduler (CLAUDE.md §4.6).

Each round every participant produces exactly one output (so outputs <= n x rounds
by construction), based only on messages delivered in the last k rounds. Outputs
are parsed strictly; malformed ones are dropped and logged. Each outgoing message
passes the loop breaker and then the bus allowlist. Everything sent in round r is
delivered at the end of round r and becomes visible from round r+1.

Log records written here (contract with safety/postflight.py):
    agent_output  {round, agent, gen, status, n_tokens, total_tokens, n_messages, [reason, raw]}
                  n_tokens = longest single message (the per-message length bound)
    delivery      {round, sender, recipient, content}
    loop_broken   {round, agent, recipient, similarity}
    routing_rejected {round, sender, recipient}
    submit        {round, agent, value}
    trace         {round, sender, recipient, canary, keywords}   from the passive observer, hits only
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from swarmguard.agents import prompts
from swarmguard.agents.agent import Agent
from swarmguard.agents.llm_client import ChatRequest, LLMClient, SamplingParams, call_seed
from swarmguard.agents.parser import (
    ParseError,
    count_tokens,
    leader_json_schema,
    parse_leader_output,
    parse_worker,
    worker_json_schema,
)
from swarmguard.config import Config
from swarmguard.orchestrator.bus import Bus, RoutingError
from swarmguard.orchestrator.topology import Topology
from swarmguard.tasks.base import TaskInstance

RAW_LOG_CHARS = 2000


class Logger(Protocol):
    def append(self, record: dict[str, Any]) -> dict[str, Any]: ...


class LoopBreaker:
    """Drop a message too similar to the same sender's previous accepted message to the same recipient."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._prev: dict[tuple[str, str], str] = {}

    def similarity(self, sender: str, recipient: str, content: str) -> float | None:
        prev = self._prev.get((sender, recipient))
        if prev is None:
            return None
        return difflib.SequenceMatcher(None, prev, content, autojunk=False).ratio()

    def should_drop(self, sender: str, recipient: str, content: str) -> tuple[bool, float | None]:
        sim = self.similarity(sender, recipient, content)
        return (sim is not None and sim >= self.threshold), sim

    def accept(self, sender: str, recipient: str, content: str) -> None:
        self._prev[(sender, recipient)] = content

    def forget(self, sender: str) -> None:
        """Drop history for a sender (its seat was refreshed or removed)."""
        for key in [k for k in self._prev if k[0] == sender]:
            del self._prev[key]


# A non-LLM participant (scripted leader; the impostor in Step 7) returns raw JSON text.
ScriptedFn = Callable[[Agent, int, list[str]], str]
# Passive observer of deliveries (the tracer). May return a record to log; cannot act.
Observer = Callable[[int, str, str, str], "dict[str, Any] | None"]


@dataclass
class RoundResult:
    submitted: Any = None
    submitter: str | None = None
    tokens: int = 0
    delivered: int = 0
    dropped: int = 0
    loops_broken: int = 0


@dataclass
class RoundScheduler:
    cfg: Config
    task: TaskInstance
    topo: Topology
    agents: dict[str, Agent]  # internal id -> agent (swarm + any extra participants)
    bus: Bus
    log: Logger
    client: LLMClient | None
    policy: str = "none"
    scripted: dict[str, ScriptedFn] = field(default_factory=dict)  # agent id -> scripted policy
    loop_breaker: LoopBreaker | None = None
    observer: Observer | None = None

    def __post_init__(self) -> None:
        if self.loop_breaker is None:
            self.loop_breaker = LoopBreaker(self.cfg.messages.loop_similarity)

    # -- prompts ------------------------------------------------------------------------
    def _names(self, ids: list[str]) -> list[str]:
        return [self.topo.names[i] for i in ids]

    def system_prompt(self, agent: Agent) -> str:
        names = self._names(agent.contacts)
        if agent.role == "leader":
            return prompts.leader_system(agent, self.task, names, self.policy,
                                         revisable=not self.cfg.episode.end_on_submit)
        return prompts.worker_system(agent, self.task, " or ".join(names), self.policy)

    def request(self, agent: Agent, round_idx: int) -> ChatRequest:
        names = self._names(agent.contacts)
        schema = leader_json_schema(names) if agent.role == "leader" else worker_json_schema(names)
        msgs = [
            ("system", self.system_prompt(agent)),
            ("user", prompts.user_turn(agent, round_idx, self.cfg.episode.rounds, self.cfg.agents.context_rounds,
                                       timed=self.task.timed)),
        ]
        sampling = SamplingParams(
            temperature=self.cfg.llm.temperature,
            max_tokens=self.cfg.messages.max_tokens * (max(1, len(names)) if agent.role == "leader" else 1) + 64,
            seed=call_seed(self.cfg.run.seed, agent.id, round_idx),
        )
        return ChatRequest.make(agent.id, round_idx, msgs, schema, sampling, schema_name=f"{agent.role}_message")

    # -- one round ----------------------------------------------------------------------
    def step(self, round_idx: int, concurrency: int = 1) -> RoundResult:
        res = RoundResult()
        order = sorted(self.agents)
        llm_ids = [a for a in order if a not in self.scripted]
        raw: dict[str, str] = {}
        if llm_ids:
            if self.client is None:
                raise RuntimeError("LLM participants require a client")
            completions = self.client.complete_many([self.request(self.agents[a], round_idx) for a in llm_ids],
                                                    concurrency)
            for aid, comp in zip(llm_ids, completions):
                raw[aid] = comp.text
                res.tokens += comp.prompt_tokens + comp.completion_tokens
        for aid in order:
            if aid in self.scripted:
                a = self.agents[aid]
                raw[aid] = self.scripted[aid](a, round_idx, self._names(a.contacts))

        for aid in order:
            self._handle_output(self.agents[aid], round_idx, raw[aid], res)

        for env in self.bus.flush():
            sname, rname = self.topo.names[env.sender], self.topo.names[env.recipient]
            self.agents[env.sender].record_out(round_idx, rname, env.content)
            self.agents[env.recipient].record_in(round_idx, sname, env.content)
            self.log.append({"type": "delivery", "round": round_idx, "sender": env.sender,
                             "recipient": env.recipient, "content": env.content})
            res.delivered += 1
            if self.observer is not None:
                rec = self.observer(round_idx, env.sender, env.recipient, env.content)
                if rec is not None:
                    self.log.append(rec)
        return res

    def _handle_output(self, agent: Agent, round_idx: int, text: str, res: RoundResult) -> None:
        names = self._names(agent.contacts)
        mt = self.cfg.messages.max_tokens
        try:
            if agent.role == "leader":
                parsed = parse_leader_output(text, allowed_recipients=names, max_tokens=mt)
                outgoing = [(m.to, m.content) for m in parsed.message.messages]
                submit = parsed.message.submit
            else:
                pw = parse_worker(text, allowed_recipients=names, max_tokens=mt)
                outgoing, submit = [(pw.message.to, pw.message.content)], None
        except ParseError as exc:
            res.dropped += 1
            self.log.append({"type": "agent_output", "round": round_idx, "agent": agent.id, "gen": agent.gen,
                             "status": "dropped",
                             "reason": exc.reason, "raw": text[:RAW_LOG_CHARS] if isinstance(text, str) else "",
                             "n_tokens": 0, "total_tokens": 0, "n_messages": 0})
            return

        lengths = [count_tokens(c) for _, c in outgoing]
        self.log.append({"type": "agent_output", "round": round_idx, "agent": agent.id, "gen": agent.gen,
                         "status": "ok",
                         "n_tokens": max(lengths, default=0), "total_tokens": sum(lengths),
                         "n_messages": len(outgoing)})
        for to_name, content in outgoing:
            rid = self.topo.id_of(to_name)
            drop, sim = self.loop_breaker.should_drop(agent.id, rid, content)
            if drop:
                res.loops_broken += 1
                self.log.append({"type": "loop_broken", "round": round_idx, "agent": agent.id,
                                 "recipient": rid, "similarity": round(sim, 4)})
                continue
            try:
                self.bus.send(agent.id, rid, content)
            except RoutingError:
                self.log.append({"type": "routing_rejected", "round": round_idx, "sender": agent.id,
                                 "recipient": rid})
                continue
            self.loop_breaker.accept(agent.id, rid, content)
        if submit is not None and agent.role == "leader":
            self.log.append({"type": "submit", "round": round_idx, "agent": agent.id, "value": submit})
            res.submitted, res.submitter = submit, agent.id
