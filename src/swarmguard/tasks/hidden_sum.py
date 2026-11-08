"""L0 hidden sum: each clue is a labelled integer; the leader reports the total.

Labels ("Number #3") let the leader de-duplicate clues held by several workers.
"""

from __future__ import annotations

import random
from typing import Any

from swarmguard.tasks.base import Clue, Generated, Task, make_clues, norm_int


class HiddenSum(Task):
    spec = "hidden-sum:L0"
    answer_format = "an integer (the total of all numbers), e.g. 42"
    default_clues = 6

    def generate(self, rng: random.Random, n_clues: int) -> Generated:
        if n_clues < 2:
            raise ValueError("L0 needs at least 2 numbers")
        values = [rng.randint(1, 20) for _ in range(n_clues)]
        clues = make_clues([(f"Number #{i} is {v}.", ("num", i, v)) for i, v in enumerate(values, start=1)])
        return Generated(clues, sum(values), f"There are {n_clues} numbers, #1 to #{n_clues}.")

    def solve(self, clues: tuple[Clue, ...], universe: tuple | None = None) -> Any:
        by_label = {d[1]: d[2] for d in (c.data for c in clues)}
        return sum(by_label.values())

    def normalize(self, value: Any) -> Any:
        return norm_int(value)
