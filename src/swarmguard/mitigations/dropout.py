"""Random dropout: remove a fraction p of workers (seeded) at the round after the
first permission request. The leader is never dropped."""

from __future__ import annotations

import random

from swarmguard.orchestrator.topology import LEADER_ID


def dropout_count(n_workers: int, p: float) -> int:
    """At least one, and always leave at least one worker."""
    if n_workers <= 1:
        return 0
    return min(n_workers - 1, max(1, round(p * n_workers)))


def select_dropout(worker_ids: list[str], p: float, seed: int) -> list[str]:
    workers = sorted(w for w in worker_ids if w != LEADER_ID)
    k = dropout_count(len(workers), p)
    return sorted(random.Random(f"dropout:{seed}").sample(workers, k))
