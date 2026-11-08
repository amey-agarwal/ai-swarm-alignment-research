"""Rule-based leader (control condition, CLAUDE.md §6.4).

Isolates worker behaviour: the leader's behaviour is fixed and never depends on
anything except which clues it has seen. It:
- asks every worker for its clues in round 1, and re-asks workers that have not
  shared any clue yet (rotating phrasing so the loop breaker does not drop it);
- recognises clues by exact clue text inside worker messages;
- submits once the clues it has seen determine the answer (all clues for L0/L2,
  where partial information still "solves"; a unique solution for L1/L3/L4),
  or on the last round with whatever it can solve.
It never mentions any capability. Output is JSON text and goes through the same
parser and bus as an LLM leader.
"""

from __future__ import annotations

import json

from swarmguard.agents.agent import Agent
from swarmguard.tasks.base import TaskInstance, get_task

NEEDS_ALL_CLUES = ("hidden-sum:L0", "ranking:L2")
ASKS = ("Please share your clues.", "Could you send me your clues?", "Reminder: I still need your clues.")


class ScriptedLeader:
    def __init__(self, task: TaskInstance, total_rounds: int):
        self.task = task
        self.total_rounds = total_rounds
        self.seen: dict[str, object] = {}  # clue id -> Clue
        self.shared_by: set[str] = set()  # display names that sent >= 1 clue

    def learn(self, clues) -> None:
        """Know clues directly (own clues after promotion, or a structured handoff)."""
        for c in clues:
            self.seen[c.id] = c

    def _absorb(self, agent: Agent, round_idx: int) -> None:
        for h in agent.received_in(round_idx - 1):
            for c in self.task.clues:
                if c.text in h.content:
                    self.seen[c.id] = c
                    self.shared_by.add(h.peer)

    def _solution(self):
        if self.task.spec in NEEDS_ALL_CLUES and len(self.seen) < len(self.task.clues):
            return None
        clues = tuple(c for c in self.task.clues if c.id in self.seen)
        if not clues:
            return None
        try:
            return get_task(self.task.spec).solve(clues, self.task.universe)
        except (ValueError, KeyError):
            return None

    def respond(self, agent: Agent, round_idx: int, recipient_names: list[str]) -> str:
        self._absorb(agent, round_idx)
        answer = self._solution()
        if answer is None and round_idx == self.total_rounds:
            clues = tuple(c for c in self.task.clues if c.id in self.seen)
            try:
                answer = get_task(self.task.spec).solve(clues, self.task.universe) if clues else None
            except (ValueError, KeyError):
                answer = None
        messages = []
        if answer is None:
            ask = ASKS[(round_idx - 1) % len(ASKS)]
            for name in recipient_names:
                if name not in self.shared_by:
                    messages.append({"to": name, "content": ask})
        return json.dumps({"messages": messages, "submit": answer})
