"""Impostor node: scripted, live (stdin REPL), or replay of a recorded session."""

from __future__ import annotations

from typing import Any

from swarmguard.impostor.live import LiveImpostor
from swarmguard.impostor.replay import ReplayImpostor, load_session
from swarmguard.impostor.scripted import ScriptedImpostor
from swarmguard.rundir import RunDir
from swarmguard.tasks.base import Clue

MODES = ("none", "scripted", "live", "replay")


def make_impostor(mode: str, *, claim: str, clue: Clue, log: Any = None, replay_from: RunDir | None = None,
                  live_io: tuple | None = None, on_turn=None, replay_phase: str = "replay"):
    if mode == "scripted":
        return ScriptedImpostor(claim, clue, log)  # type: ignore[arg-type]
    if mode == "live":
        inp, out = live_io or (input, print)
        return LiveImpostor(inp, out, log, on_turn=on_turn)
    if mode == "replay":
        if replay_from is None:
            raise ValueError("replay mode needs the source run")
        return ReplayImpostor(load_session(replay_from), log, phase=replay_phase)
    raise ValueError(f"unknown impostor mode {mode!r}")
