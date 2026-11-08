"""Impostor sessions (impostor_session.json) and deterministic replay.

Every impostor mode records what the impostor said, by round. `replay:<run_id>`
re-plays a recorded session verbatim, so a live session becomes a reproducible
script. Session content is untrusted text and is only ever sent as a message.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from swarmguard.agents.agent import Agent
from swarmguard.rundir import RunDir
from swarmguard.schemas import MAX_CONTENT_CHARS

SESSION_FILE = "impostor_session.json"
FALLBACK = "Ready when you are."


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round: int = Field(ge=1)
    content: str = Field(max_length=MAX_CONTENT_CHARS)


class ImpostorSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    mode: Literal["scripted", "live", "replay"]
    claim: str
    contact: str
    replayed_from: str | None = None
    turns: list[Turn] = Field(default_factory=list)


def write_session(rd: RunDir, session: ImpostorSession) -> None:
    with rd.open(SESSION_FILE, "w", encoding="utf-8") as fh:
        fh.write(session.model_dump_json(indent=2))
        fh.write("\n")


def load_session(rd: RunDir) -> ImpostorSession:
    with rd.open(SESSION_FILE, "r", encoding="utf-8") as fh:
        return ImpostorSession.model_validate(json.load(fh))


class ReplayImpostor:
    """Says exactly what the recorded session said in each round (fallback line if absent)."""

    def __init__(self, session: ImpostorSession, log: Any = None, phase: str = "replay"):
        rounds = [t.round for t in session.turns]
        if len(rounds) != len(set(rounds)):
            raise ValueError("session has duplicate rounds")
        self.source = session
        self.by_round = {t.round: t.content for t in session.turns}
        self.log = log
        self.phase = phase  # "live" when reproducing a live run byte-identically
        self.turns: list[dict[str, Any]] = []

    def respond(self, agent: Agent, round_idx: int, recipient_names: list[str]) -> str:
        content = self.by_round.get(round_idx, FALLBACK)
        self.turns.append({"round": round_idx, "content": content})
        if self.log is not None:
            self.log.append({"type": "impostor", "round": round_idx, "phase": self.phase,
                             "ladder_step": None, "request_seen": False})
        return json.dumps({"to": recipient_names[0], "content": content})
