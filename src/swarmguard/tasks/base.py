"""Task base classes and registry.

A task generator produces a TaskInstance from a seed: a set of clues (text for
agents, structured `data` for the reference solver), the hidden answer, and a
one-line description of the `submit` format for the leader prompt.

`Task.solve(clues)` derives the answer from clue data alone; tests use it to
show every instance is solvable from the union of clues.
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Clue:
    id: str  # "C1", "C2", ...
    text: str  # what an agent sees
    data: tuple  # structured form for the reference solver (never shown to agents)
    release_round: int = 1  # the holder learns this clue at this round (timed tasks); 1 = from the start


@dataclass(frozen=True)
class TaskInstance:
    spec: str  # e.g. "logic-grid:L4"
    seed: int
    clues: tuple[Clue, ...]
    answer: Any  # canonical JSON-compatible answer
    answer_format: str  # for the leader prompt
    preamble: str = ""  # common knowledge shown to every agent (e.g. who/what exists)
    universe: tuple | None = None  # structured common knowledge for the solver

    def solve(self) -> Any:
        """Reference solution derived from clues + common knowledge only."""
        return get_task(self.spec).solve(self.clues, self.universe)

    @property
    def timed(self) -> bool:
        """True when some clue is released after round 1 (clues then appear in each round's turn)."""
        return any(c.release_round > 1 for c in self.clues)

    @property
    def clue_ids(self) -> list[str]:
        return [c.id for c in self.clues]

    def clue(self, clue_id: str) -> Clue:
        for c in self.clues:
            if c.id == clue_id:
                return c
        raise KeyError(clue_id)

    def check(self, submitted: Any) -> bool:
        return get_task(self.spec).check(self, submitted)


@dataclass(frozen=True)
class Generated:
    clues: tuple[Clue, ...]
    answer: Any
    preamble: str = ""
    universe: tuple | None = None


def make_clues(items: list[tuple[str, tuple]]) -> tuple[Clue, ...]:
    return tuple(Clue(f"C{i}", text, data) for i, (text, data) in enumerate(items, start=1))


class Task(ABC):
    spec: str
    answer_format: str
    default_clues: int

    @abstractmethod
    def generate(self, rng: random.Random, n_clues: int) -> Generated:
        """Return clues, canonical answer, and common knowledge."""

    @abstractmethod
    def solve(self, clues: tuple[Clue, ...], universe: tuple | None = None) -> Any:
        """Derive the answer from clue data + common knowledge (None if not determined)."""

    @abstractmethod
    def normalize(self, value: Any) -> Any:
        """Map a submitted value to canonical form (None if unparseable)."""

    def instance(self, seed: int, n_clues: int | None = None) -> TaskInstance:
        rng = random.Random(f"{self.spec}:{seed}")
        g = self.generate(rng, n_clues or self.default_clues)
        return TaskInstance(self.spec, seed, g.clues, g.answer, self.answer_format, g.preamble, g.universe)

    def check(self, inst: TaskInstance, submitted: Any) -> bool:
        if submitted is None:
            return False
        try:
            got = self.normalize(submitted)
        except (TypeError, ValueError, KeyError, AttributeError):
            return False
        return got is not None and got == self.normalize(inst.answer)


# ----------------------------------------------------------------------------- normalisers


def norm_word(v: Any) -> str:
    if not isinstance(v, (str, int)) or isinstance(v, bool):
        raise ValueError("not a word")
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def norm_int(v: Any) -> int:
    if isinstance(v, bool):
        raise ValueError("bool")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*-?\d+\s*", v):
        return int(v)
    raise ValueError(f"not an integer: {v!r}")


def split_list(v: Any) -> list[Any]:
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        return [p for p in re.split(r"\s*(?:,|>|;|\n)\s*", v.strip()) if p]
    raise ValueError("not a list")


# ----------------------------------------------------------------------------- registry

TASK_SPECS = ("hidden-sum:L0", "lookup:L1", "ranking:L2", "scheduler:L3", "logic-grid:L4",
              "inventory:I1", "inventory:I2")
_ALIASES = {"L0": "hidden-sum:L0", "L1": "lookup:L1", "L2": "ranking:L2", "L3": "scheduler:L3",
            "L4": "logic-grid:L4", "I1": "inventory:I1", "I2": "inventory:I2"}
INVENTORY_LEVELS = {"inventory:I1": (6, 6), "inventory:I2": (10, 10)}  # (movements, last release round)
# `--task <number>` shortcuts. To add one, register the task above and add a line here.
TASK_NUMBERS = {"1": "logic-grid:L4", "2": "inventory:I1"}


def resolve_task(value: str) -> str:
    """Turn a `--task` number into its full task name; anything else is returned unchanged."""
    return TASK_NUMBERS.get(value, value)


def get_task(spec: str) -> Task:
    spec = _ALIASES.get(spec, spec)
    if spec == "hidden-sum:L0":
        from swarmguard.tasks.hidden_sum import HiddenSum
        return HiddenSum()
    if spec == "lookup:L1":
        from swarmguard.tasks.lookup import Lookup
        return Lookup()
    if spec == "ranking:L2":
        from swarmguard.tasks.ranking import Ranking
        return Ranking()
    if spec == "scheduler:L3":
        from swarmguard.tasks.scheduler import MeetingScheduler
        return MeetingScheduler()
    if spec == "logic-grid:L4":
        from swarmguard.tasks.logic_grid import LogicGridL4
        return LogicGridL4()
    if spec in INVENTORY_LEVELS:
        from swarmguard.tasks.inventory import Inventory
        n_moves, last = INVENTORY_LEVELS[spec]
        return Inventory(spec, n_moves, last)
    raise ValueError(f"unknown task {spec!r}; choose from {TASK_SPECS}")


def make_instance(spec: str, seed: int, n_clues: int | None = None) -> TaskInstance:
    return get_task(spec).instance(seed, n_clues)
