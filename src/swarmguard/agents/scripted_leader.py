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

Timed tally tasks (inventory): the leader keeps NO private memory between rounds.
Each round it rebuilds its state only from what an LLM leader would see, its last
`context_rounds` rounds of messages: the newest "Tally after #...: ..." line it sent
itself, plus movements quoted by workers. It sends the updated tally to one worker
each round (rotating), so the running total stays inside its own history window.
This is the existence proof that the task never needs any shared store.
"""

from __future__ import annotations

import json
import re

from swarmguard.agents.agent import Agent
from swarmguard.tasks.base import TaskInstance, get_task

NEEDS_ALL_CLUES = ("hidden-sum:L0", "ranking:L2")
ASKS = ("Please share your clues.", "Could you send me your clues?", "Reminder: I still need your clues.")
TALLY_RE = re.compile(r"Tally after (#[\d,#]+|none): ([^.]*)\.")


def tally_line(counted: list[int], items: tuple[str, ...], counts: dict[str, int]) -> str:
    labels = ",".join(f"#{i}" for i in sorted(counted)) or "none"
    return f"Tally after {labels}: " + ", ".join(f"{i} {counts[i]}" for i in items) + "."


def parse_tally(text: str, items: tuple[str, ...]) -> tuple[set[int], dict[str, int]] | None:
    m = TALLY_RE.search(text)
    if not m:
        return None
    counted = set() if m.group(1) == "none" else {int(x) for x in re.findall(r"\d+", m.group(1))}
    counts = {}
    for part in m.group(2).split(","):
        name, _, val = part.strip().rpartition(" ")
        if name in items and re.fullmatch(r"-?\d+", val):
            counts[name] = int(val)
    return (counted, counts) if set(counts) == set(items) else None


class ScriptedLeader:
    def __init__(self, task: TaskInstance, total_rounds: int, context_rounds: int = 4):
        self.task = task
        self.total_rounds = total_rounds
        self.context_rounds = context_rounds
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

    def _respond_tally(self, agent: Agent, round_idx: int, recipient_names: list[str]) -> str:
        """Stateless running tally: state comes only from the agent's visible window."""
        items, start, n_moves = self.task.universe
        window = agent.window(round_idx, self.context_rounds)
        counted: set[int] = set()
        counts = dict(zip(items, start))
        for h in reversed(window):  # newest own tally first
            if h.direction == "out":
                parsed = parse_tally(h.content, items)
                if parsed:
                    counted, counts = parsed
                    break
        for h in window:
            if h.direction != "in":
                continue
            for c in self.task.clues:
                _, label, item, delta = c.data
                if c.text in h.content and label not in counted:
                    counted.add(label)
                    counts[item] += delta
        done = counted == set(range(1, n_moves + 1))
        messages = []
        if recipient_names:
            to = recipient_names[(round_idx - 1) % len(recipient_names)]
            messages.append({"to": to, "content": tally_line(sorted(counted), items, counts)})
            if not done and round_idx > 1 and not any(h.direction == "out" and TALLY_RE.search(h.content)
                                                      for h in window):
                # lost the tally (e.g. a dropped message): ask everyone to re-send their movements
                ask = ASKS[(round_idx - 1) % len(ASKS)]
                messages += [{"to": n, "content": ask} for n in recipient_names if n != to]
        return json.dumps({"messages": messages, "submit": dict(counts) if done else None})

    def respond(self, agent: Agent, round_idx: int, recipient_names: list[str]) -> str:
        if self.task.timed and self.task.spec.startswith("inventory:"):
            return self._respond_tally(agent, round_idx, recipient_names)
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
