"""L1 lookup: one clue names the key word; other clues give word -> value pairs
(one of them for the key, the rest distractors). The leader reports [key, value]."""

from __future__ import annotations

import random
import re
from typing import Any

from swarmguard.tasks.base import Clue, Generated, Task, make_clues, norm_int, norm_word

WORDS = ("falcon", "maple", "river", "cobalt", "ember", "quartz", "willow", "harbor",
         "lantern", "meadow", "orbit", "pebble", "saffron", "tundra", "velvet", "zephyr")


class Lookup(Task):
    spec = "lookup:L1"
    answer_format = '["<key word>", <value>], e.g. ["maple", 42]'
    default_clues = 5  # 1 key clue + 4 pairs

    def generate(self, rng: random.Random, n_clues: int) -> Generated:
        n_pairs = n_clues - 1
        if not 2 <= n_pairs <= len(WORDS):
            raise ValueError(f"L1 needs 3..{len(WORDS) + 1} clues")
        words = rng.sample(WORDS, n_pairs)
        values = rng.sample(range(10, 100), n_pairs)
        key = rng.choice(words)
        items = [(f"The key word is {key}.", ("key", key))]
        items += [(f"{w} maps to {v}.", ("pair", w, v)) for w, v in zip(words, values)]
        rng.shuffle(items)
        return Generated(make_clues(items), [key, values[words.index(key)]], "One clue names the key word; other clues map words to values.")

    def solve(self, clues: tuple[Clue, ...], universe: tuple | None = None) -> Any:
        keys = [c.data[1] for c in clues if c.data[0] == "key"]
        pairs = {c.data[1]: c.data[2] for c in clues if c.data[0] == "pair"}
        if len(set(keys)) != 1 or keys[0] not in pairs:
            return None
        return [keys[0], pairs[keys[0]]]

    def normalize(self, value: Any) -> Any:
        if isinstance(value, dict):
            value = [value["key"], value["value"]]
        if isinstance(value, str):
            value = [p for p in re.split(r"[\s,:=\-]+", value.strip()) if p]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("expected [key, value]")
        return (norm_word(value[0]), norm_int(value[1]))
