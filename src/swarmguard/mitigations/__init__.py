"""Infrastructure-level mitigations (CLAUDE.md §6.5).

Seats (internal ids a00..a{n-1}) and the routing allowlist are fixed for the whole
run, so postflight can always check routing against the reference star. A
mitigation only (a) deactivates seats or (b) replaces a seat's occupant with a
fresh agent (empty history, `gen` + 1). The population never exceeds n.

Spec strings: none | dropout:<p> | leader-rotate | handoff:structured | handoff:freetext

Online monitor (mitigation arms only): the first swarm message the tracer flags
(canary or capability keyword) is the "first permission request"; first-request
triggers fire at the start of the following round.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["none", "dropout", "leader-rotate", "handoff"]


@dataclass(frozen=True)
class MitigationSpec:
    kind: Kind
    p: float | None = None  # dropout fraction
    mode: Literal["structured", "freetext"] | None = None  # handoff mode

    @property
    def label(self) -> str:
        if self.kind == "dropout":
            return f"dropout:{self.p}"
        if self.kind == "handoff":
            return f"handoff:{self.mode}"
        return self.kind


def parse_mitigation(spec: str) -> MitigationSpec:
    if spec == "none":
        return MitigationSpec("none")
    if spec == "leader-rotate":
        return MitigationSpec("leader-rotate")
    m = re.fullmatch(r"dropout:(0?\.\d+)", spec)
    if m and 0.0 < float(m.group(1)) < 1.0:
        return MitigationSpec("dropout", p=float(m.group(1)))
    m = re.fullmatch(r"handoff:(structured|freetext)", spec)
    if m:
        return MitigationSpec("handoff", mode=m.group(1))  # type: ignore[arg-type]
    raise ValueError(f"invalid mitigation {spec!r}")
