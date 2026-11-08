"""Topologies and the routing allowlist (CLAUDE.md §6.1).

Star (default): workers <-> leader only; workers cannot message each other.
Tree is P1 and not implemented.

The impostor is an extra node "x00" (not counted in n) attached to exactly one
contact: the leader, or worker k.

Contract checked by safety/preflight.py: `build_allowlist(n, kind, impostor_contact)`
must equal the independent `reference_allowlist` there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LEADER_ID = "a00"
IMPOSTOR_ID = "x00"
LEADER_NAME = "Leader"


def agent_id(i: int) -> str:
    return f"a{i:02d}"


def worker_index(aid: str) -> int:
    m = re.fullmatch(r"a(\d{2})", aid)
    if not m or int(m.group(1)) == 0:
        raise ValueError(f"not a worker id: {aid!r}")
    return int(m.group(1))


def resolve_contact(n: int, impostor_contact: str | None) -> str | None:
    """'leader' -> a00, 'worker:k' -> a0k (1 <= k <= n-1), None/'none' -> None."""
    if impostor_contact in (None, "none"):
        return None
    if impostor_contact == "leader":
        return LEADER_ID
    m = re.fullmatch(r"worker:(\d+)", impostor_contact)
    if m and 1 <= int(m.group(1)) <= n - 1:
        return agent_id(int(m.group(1)))
    raise ValueError(f"invalid impostor contact {impostor_contact!r} for n={n}")


def build_allowlist(n: int, kind: str = "star", impostor_contact: str | None = None) -> set[tuple[str, str]]:
    if n < 2:
        raise ValueError("n must be >= 2")
    if kind == "tree":
        raise NotImplementedError("tree topology is P1")
    if kind != "star":
        raise ValueError(f"unknown topology {kind!r}")
    edges = {(LEADER_ID, agent_id(i)) for i in range(1, n)} | {(agent_id(i), LEADER_ID) for i in range(1, n)}
    contact = resolve_contact(n, impostor_contact)
    if contact is not None:
        edges |= {(IMPOSTOR_ID, contact), (contact, IMPOSTOR_ID)}
    return edges


@dataclass(frozen=True)
class Topology:
    n: int
    kind: str
    allowlist: frozenset[tuple[str, str]]
    names: dict[str, str]  # internal id -> display name

    @staticmethod
    def star(n: int, impostor_contact: str | None = None, impostor_name: str | None = None) -> "Topology":
        edges = build_allowlist(n, "star", impostor_contact)
        names = {LEADER_ID: LEADER_NAME, **{agent_id(i): f"W{i}" for i in range(1, n)}}
        if resolve_contact(n, impostor_contact) is not None:
            names[IMPOSTOR_ID] = impostor_name or f"W{n}"
        if len(set(names.values())) != len(names):
            raise ValueError("display names must be unique")
        return Topology(n, "star", frozenset(edges), names)

    @property
    def swarm_ids(self) -> list[str]:
        return [agent_id(i) for i in range(self.n)]

    def contacts(self, aid: str) -> list[str]:
        return sorted(r for s, r in self.allowlist if s == aid)

    def id_of(self, name: str) -> str:
        for aid, nm in self.names.items():
            if nm == name:
                return aid
        raise KeyError(name)
