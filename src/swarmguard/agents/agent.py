"""Agent state: stable internal id, display name, role, clues, and message history.

All logs and metrics use the internal id. The display name is what other agents
see (and is the only thing renaming, if enabled later, may change).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from swarmguard.tasks.base import Clue

Role = Literal["leader", "worker"]


@dataclass(frozen=True)
class HistoryItem:
    round: int
    direction: Literal["in", "out"]
    peer: str  # display name of the other party
    content: str


@dataclass
class Agent:
    id: str
    display_name: str
    role: Role
    clues: list[Clue] = field(default_factory=list)
    contacts: list[str] = field(default_factory=list)  # internal ids this agent may message
    history: list[HistoryItem] = field(default_factory=list)
    gen: int = 0  # generation of this seat; a fresh agent (rotation/handoff) increments it
    handoff_note: str | None = None  # briefing from a handoff, shown at the top of each turn

    def refresh(self, *, clues: list[Clue] | None = None, handoff_note: str | None = None) -> None:
        """Replace this seat's occupant with a fresh agent: empty history, next generation."""
        self.history = []
        self.gen += 1
        if clues is not None:
            self.clues = list(clues)
        self.handoff_note = handoff_note

    def record_in(self, round_idx: int, peer: str, content: str) -> None:
        self.history.append(HistoryItem(round_idx, "in", peer, content))

    def record_out(self, round_idx: int, peer: str, content: str) -> None:
        self.history.append(HistoryItem(round_idx, "out", peer, content))

    def window(self, current_round: int, k: int) -> list[HistoryItem]:
        """Items from the last k completed rounds (rounds current-k .. current-1)."""
        lo = current_round - k
        return [h for h in self.history if lo <= h.round < current_round]

    def received_in(self, round_idx: int) -> list[HistoryItem]:
        return [h for h in self.history if h.round == round_idx and h.direction == "in"]
