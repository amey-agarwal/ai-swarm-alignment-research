"""Leader rotation: promote a worker to leader with fresh context.

The leader seat (a00, the star hub) is taken over by a fresh agent that carries
the promoted worker's clues but none of anyone's history. The promoted worker's
seat is vacated, so the population drops to n-1 (never exceeds n).
"""

from __future__ import annotations

import random

from swarmguard.orchestrator.topology import LEADER_ID


def choose_promoted(worker_ids: list[str], seed: int) -> str:
    workers = sorted(w for w in worker_ids if w != LEADER_ID)
    if not workers:
        raise ValueError("no worker to promote")
    return random.Random(f"rotate:{seed}").choice(workers)
