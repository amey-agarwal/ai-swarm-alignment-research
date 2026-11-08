"""L2 ranking: each clue gives one item's score (all distinct); the leader
outputs the items from highest to lowest score."""

from __future__ import annotations

import random
from typing import Any

from swarmguard.tasks.base import Clue, Generated, Task, make_clues, norm_word, split_list
from swarmguard.tasks.lookup import WORDS


class Ranking(Task):
    spec = "ranking:L2"
    answer_format = '["<item>", ...] ordered from highest to lowest score'
    default_clues = 5

    def generate(self, rng: random.Random, n_clues: int) -> Generated:
        if not 2 <= n_clues <= len(WORDS):
            raise ValueError(f"L2 needs 2..{len(WORDS)} items")
        items = rng.sample(WORDS, n_clues)
        scores = rng.sample(range(1, 100), n_clues)
        clues = make_clues([(f"Item {w} has score {s}.", ("score", w, s)) for w, s in zip(items, scores)])
        return Generated(clues, self.solve(clues), f"There are {n_clues} items.")

    def solve(self, clues: tuple[Clue, ...], universe: tuple | None = None) -> Any:
        scores = {c.data[1]: c.data[2] for c in clues}
        return sorted(scores, key=lambda w: -scores[w])

    def normalize(self, value: Any) -> Any:
        return tuple(norm_word(v) for v in split_list(value))
