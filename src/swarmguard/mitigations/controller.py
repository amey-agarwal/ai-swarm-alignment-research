"""Applies one mitigation during an episode (see mitigations/__init__.py).

Called by the episode loop: `after_round` feeds the online monitor (tracer summary);
`before_round` applies the mitigation when due and returns any tokens it used.
Every action is logged as a `mitigation` record:
    {kind: dropout, round, trigger_round, removed, impostor_disconnected}
    {kind: leader_rotation, round, trigger_round, promoted, leader_gen, impostor_disconnected}
    {kind: handoff, mode, round, state | summary, dropped_items, gen}
"""

from __future__ import annotations

from typing import Any, Callable

from swarmguard.agents.llm_client import LLMClient
from swarmguard.agents.scripted_leader import ScriptedLeader
from swarmguard.config import Config
from swarmguard.mitigations import MitigationSpec
from swarmguard.mitigations import handoff as H
from swarmguard.mitigations.dropout import select_dropout
from swarmguard.mitigations.leader_rotation import choose_promoted
from swarmguard.orchestrator.scheduler import RoundScheduler
from swarmguard.orchestrator.topology import IMPOSTOR_ID, LEADER_ID, Topology
from swarmguard.tasks.base import TaskInstance
from swarmguard.verifiers.tracer import TraceSummary


class MitigationController:
    def __init__(self, ms: MitigationSpec, cfg: Config, task: TaskInstance, topo: Topology, sched: RoundScheduler,
                 client: LLMClient | None, log: Any, seed: int,
                 make_scripted_leader: Callable[[], ScriptedLeader] | None = None):
        self.ms, self.cfg, self.task, self.topo, self.sched = ms, cfg, task, topo, sched
        self.client, self.log, self.seed = client, log, seed
        self.make_scripted_leader = make_scripted_leader
        self.scripted_leader: ScriptedLeader | None = None
        self.trigger_round: int | None = None
        self.applied = False

    # -- monitor --------------------------------------------------------------------------
    def after_round(self, round_idx: int, summary: TraceSummary) -> None:
        if self.trigger_round is None and summary.first_round:
            self.trigger_round = min(summary.first_round.values())

    def _due(self, round_idx: int) -> bool:
        if self.applied or self.ms.kind == "none":
            return False
        mc = self.cfg.mitigation
        if self.ms.kind == "dropout":
            return self.trigger_round is not None and round_idx == self.trigger_round + 1
        if self.ms.kind == "leader-rotate":
            if mc.rotation_trigger == "round":
                return round_idx == mc.rotation_round
            return self.trigger_round is not None and round_idx == self.trigger_round + 1
        if self.ms.kind == "handoff":
            return round_idx == mc.handoff_round
        return False

    def before_round(self, round_idx: int) -> int:
        if not self._due(round_idx):
            return 0
        self.applied = True
        if self.ms.kind == "dropout":
            return self._dropout(round_idx)
        if self.ms.kind == "leader-rotate":
            return self._rotate(round_idx)
        return self._handoff(round_idx)

    # -- helpers --------------------------------------------------------------------------
    @property
    def agents(self):
        return self.sched.agents

    def _workers(self) -> list[str]:
        return sorted(a for a in self.agents if a in self.topo.swarm_ids and a != LEADER_ID)

    def _deactivate(self, ids: list[str]) -> bool:
        """Remove seats; returns True if the impostor lost its only contact (and was removed)."""
        for aid in ids:
            self.agents.pop(aid, None)
            self.sched.scripted.pop(aid, None)
            self.sched.loop_breaker.forget(aid)
        for a in self.agents.values():
            a.contacts = [c for c in a.contacts if c not in ids]
        imp = self.agents.get(IMPOSTOR_ID)
        if imp is not None and not imp.contacts:
            self.agents.pop(IMPOSTOR_ID)
            self.sched.scripted.pop(IMPOSTOR_ID, None)
            for a in self.agents.values():
                a.contacts = [c for c in a.contacts if c != IMPOSTOR_ID]
            return True
        return False

    def _new_scripted_leader(self) -> None:
        if LEADER_ID in self.sched.scripted and self.make_scripted_leader is not None:
            self.scripted_leader = self.make_scripted_leader()
            self.sched.scripted[LEADER_ID] = self.scripted_leader.respond

    # -- mitigations ----------------------------------------------------------------------
    def _dropout(self, r: int) -> int:
        removed = select_dropout(self._workers(), self.ms.p or 0.0, self.seed)
        lost = self._deactivate(removed)
        self.log.append({"type": "mitigation", "kind": "dropout", "round": r, "trigger_round": self.trigger_round,
                         "removed": removed, "impostor_disconnected": lost})
        return 0

    def _rotate(self, r: int) -> int:
        promoted = choose_promoted(self._workers(), self.seed)
        leader = self.agents[LEADER_ID]
        leader.refresh(clues=self.agents[promoted].clues)
        self.sched.loop_breaker.forget(LEADER_ID)
        lost = self._deactivate([promoted])
        self._new_scripted_leader()
        if self.scripted_leader is not None:
            self.scripted_leader.learn(leader.clues)
        self.log.append({"type": "mitigation", "kind": "leader_rotation", "round": r,
                         "trigger_round": self.trigger_round, "promoted": promoted, "leader_gen": leader.gen,
                         "impostor_disconnected": lost})
        return 0

    def _handoff(self, r: int) -> int:
        mode = self.ms.mode or "structured"
        leader = self.agents[LEADER_ID]
        tokens = 0
        if LEADER_ID in self.sched.scripted:
            raw = self._scripted_handoff_text(mode)
        else:
            if self.client is None:
                raise RuntimeError("handoff needs an LLM client for an LLM leader")
            req = H.build_request(mode, leader, self.sched.system_prompt(leader), H.history_text(leader),
                                  self.cfg.run.seed, r)
            comp = self.client.complete(req)
            raw, tokens = comp.text, comp.prompt_tokens + comp.completion_tokens
        record: dict[str, Any] = {"type": "mitigation", "kind": "handoff", "mode": mode, "round": r}
        if mode == "structured":
            state, dropped = H.filter_structured(raw, self.task)
            note = H.render_structured(state)
            record.update(state=state.model_dump(), dropped_items=dropped)
        else:
            summary = H.parse_freetext(raw)
            note = H.render_freetext(summary)
            record.update(summary=summary, dropped_items=0)
        for aid in sorted(a for a in self.agents if a in self.topo.swarm_ids):
            self.agents[aid].refresh(handoff_note=note if aid == LEADER_ID else None)
            self.sched.loop_breaker.forget(aid)
        self._new_scripted_leader()
        if self.scripted_leader is not None and mode == "structured":
            known = {c.text: c for c in self.task.clues}
            self.scripted_leader.learn([known[t] for t in record["state"]["clues_collected"]])
        record["gen"] = leader.gen
        self.log.append(record)
        return tokens

    def _scripted_handoff_text(self, mode: str) -> str:
        import json

        sl = self.scripted_leader
        if sl is None:
            fn = self.sched.scripted[LEADER_ID]
            sl = getattr(fn, "__self__", None)
        seen = [c.text for c in self.task.clues if sl is not None and c.id in sl.seen]
        if mode == "structured":
            return json.dumps({"clues_collected": seen, "partial_solution": None})
        return json.dumps({"summary": f"Collected {len(seen)} clues so far."})
