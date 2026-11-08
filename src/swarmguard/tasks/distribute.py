"""Split clues across workers with redundancy (CLAUDE.md §6.2).

- n_workers <= n_clues: shuffle clues and deal them round-robin; every worker gets
  >= 1 clue and every clue is held exactly once.
- n_workers > n_clues: every worker gets exactly one clue, dealt cyclically over a
  shuffled clue order, so every clue is held by floor(n_workers / n_clues) or more
  workers (>= 2 whenever n_workers >= 2 * n_clues). Dropout then cannot easily
  make the task unsolvable.

The impostor receives a duplicate of a clue already held by a worker, so it can
contribute legitimately without the task depending on it.
"""

from __future__ import annotations

import random
from collections import Counter

Assignment = dict[int, list[str]]  # worker index (1-based) -> clue ids


def distribute(clue_ids: list[str], n_workers: int, seed: int | str) -> Assignment:
    if n_workers < 1:
        raise ValueError("need at least one worker")
    if not clue_ids or len(set(clue_ids)) != len(clue_ids):
        raise ValueError("clue ids must be non-empty and unique")
    rng = random.Random(f"distribute:{seed}")
    order = list(clue_ids)
    rng.shuffle(order)
    out: Assignment = {k: [] for k in range(1, n_workers + 1)}
    if n_workers <= len(order):
        for i, cid in enumerate(order):
            out[i % n_workers + 1].append(cid)
    else:
        for k in range(1, n_workers + 1):
            out[k].append(order[(k - 1) % len(order)])
    for k in out:
        out[k].sort(key=clue_ids.index)
    return out


def coverage(assignment: Assignment) -> Counter:
    """How many workers hold each clue."""
    return Counter(cid for clues in assignment.values() for cid in clues)


def covers_all(assignment: Assignment, clue_ids: list[str]) -> bool:
    return set(coverage(assignment)) == set(clue_ids)


def impostor_clue(assignment: Assignment, seed: int | str) -> str:
    """Pick a clue for the impostor: one already held by at least one worker."""
    held = sorted(coverage(assignment))
    if not held:
        raise ValueError("no clues assigned")
    return random.Random(f"impostor_clue:{seed}").choice(held)
