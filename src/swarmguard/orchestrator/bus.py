"""In-memory message bus with a routing allowlist (CLAUDE.md §4.6).

In-process only: no sockets, no files. Messages sent during a round are queued
and delivered together at the end of the round (`flush`). A send on an edge that
is not in the allowlist raises RoutingError and is counted.

Contract checked by safety/probes.py: `Bus(allowlist).send(sender, recipient, content)`
raises `RoutingError` for a disallowed edge.
"""

from __future__ import annotations

from dataclasses import dataclass


class RoutingError(Exception):
    pass


@dataclass(frozen=True)
class Envelope:
    sender: str
    recipient: str
    content: str


class Bus:
    def __init__(self, allowlist: set[tuple[str, str]] | frozenset[tuple[str, str]]):
        self._allow = frozenset((str(s), str(r)) for s, r in allowlist)
        self._pending: list[Envelope] = []
        self.rejected = 0
        self.delivered = 0

    @property
    def allowlist(self) -> frozenset[tuple[str, str]]:
        return self._allow

    def allowed(self, sender: str, recipient: str) -> bool:
        return (sender, recipient) in self._allow

    def send(self, sender: str, recipient: str, content: str) -> None:
        if not isinstance(content, str):
            raise TypeError("content must be text")
        if sender == recipient or (sender, recipient) not in self._allow:
            self.rejected += 1
            raise RoutingError(f"{sender} -> {recipient} not allowed")
        self._pending.append(Envelope(sender, recipient, content))

    def flush(self) -> list[Envelope]:
        out, self._pending = self._pending, []
        self.delivered += len(out)
        return out
