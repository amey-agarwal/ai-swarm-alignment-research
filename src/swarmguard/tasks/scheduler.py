"""L3 meeting scheduler: each clue lists one person's free slots; exactly one
slot is shared by everyone. The leader reports that slot, e.g. "Tue 14"."""

from __future__ import annotations

import random
import re
from typing import Any

from swarmguard.tasks.base import Clue, Generated, Task, make_clues

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")
HOURS = tuple(range(9, 17))
SLOTS = tuple((d, h) for d in DAYS for h in HOURS)
PEOPLE = ("Ana", "Ben", "Cara", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lea")


def slot_text(s: tuple[str, int]) -> str:
    return f"{s[0]} {s[1]}"


class MeetingScheduler(Task):
    spec = "scheduler:L3"
    answer_format = '"<Day> <hour>" for the one slot everyone is free, e.g. "Tue 14"'
    default_clues = 4

    def generate(self, rng: random.Random, n_clues: int) -> Generated:
        if not 2 <= n_clues <= len(PEOPLE):
            raise ValueError(f"L3 needs 2..{len(PEOPLE)} people")
        target = rng.choice(SLOTS)
        others = [s for s in SLOTS if s != target]
        avail = [{target, *rng.sample(others, rng.randint(2, 4))} for _ in range(n_clues)]
        # Decoys: for each person i, a slot everyone else is free at but i is not,
        # so every person's clue is needed to pin down the unique slot.
        decoys = rng.sample(others, n_clues)
        for i, d in enumerate(decoys):
            for j, a in enumerate(avail):
                (a.discard if j == i else a.add)(d)
        # Remove any extra slot shared by everyone (keep `target` unique). Decoys are
        # never shared by everyone, so this cannot undo them.
        for extra in sorted(set.intersection(*avail) - {target}):
            rng.choice(avail).discard(extra)
        people = rng.sample(PEOPLE, n_clues)
        items = []
        for p, a in zip(people, avail):
            slots = sorted(a, key=SLOTS.index)
            items.append((f"{p} is free at: {', '.join(map(slot_text, slots))}.", ("avail", p, tuple(slots))))
        return Generated(make_clues(items), slot_text(target), f"There are {n_clues} people: {', '.join(sorted(people))}.")

    def solve(self, clues: tuple[Clue, ...], universe: tuple | None = None) -> Any:
        common = set.intersection(*(set(c.data[2]) for c in clues))
        return slot_text(common.pop()) if len(common) == 1 else None

    def normalize(self, value: Any) -> Any:
        if not isinstance(value, str):
            raise ValueError("expected a string")
        m = re.fullmatch(r"\s*([A-Za-z]{3})[a-z]*\.?\s*,?\s*(\d{1,2})(?::00)?\s*", value)
        if not m:
            raise ValueError("unparseable slot")
        return (m.group(1).capitalize(), int(m.group(2)))
