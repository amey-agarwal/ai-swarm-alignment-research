"""Live impostor: a human types the impostor's messages via a stdin REPL.

Intended for an interactive Slurm job (slurm/interactive.sh). Before the episode
starts, `confirm_gate` prints the pass 1 and pass 2 results and requires the user
to type exactly CONFIRM. Each round the operator sees the impostor's inbox
(agent text JSON-escaped, so control or terminal-escape sequences cannot render)
and types one message. Everything typed is sent as plain text through the same
parser and bus as any agent, and recorded to impostor_session.json.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from swarmguard.agents.agent import Agent
from swarmguard.schemas import MAX_CONTENT_CHARS

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
CONFIRM_WORD = "CONFIRM"
EMPTY_FALLBACK = "ok"


def escape(text: str) -> str:
    """Render untrusted text safely for a terminal."""
    return json.dumps(text, ensure_ascii=True)


def confirm_gate(pass1: Any, pass2: Any, input_fn: InputFn = input, output_fn: OutputFn = print) -> bool:
    """Print both safety passes; return True only if both pass and the user types CONFIRM."""
    for res in (pass1, pass2):
        output_fn(f"{res.name.upper()} ({res.mode}): {'PASS' if res.ok else 'FAIL'}")
        for c in res.checks:
            output_fn(f"  {c.status.upper():8} {c.name}  {c.detail}")
    if not (pass1.ok and pass2.ok):
        output_fn("Safety checks failed; live session will not start.")
        return False
    try:
        answer = input_fn(f"Type {CONFIRM_WORD} to start the live impostor session: ")
    except EOFError:
        return False
    if answer.strip() != CONFIRM_WORD:
        output_fn("Not confirmed; aborting.")
        return False
    return True


class LiveImpostor:
    def __init__(self, input_fn: InputFn = input, output_fn: OutputFn = print, log: Any = None,
                 max_retries: int = 3, on_turn: Callable[[list[dict[str, Any]]], None] | None = None):
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.log = log
        self.max_retries = max_retries
        self.on_turn = on_turn  # called after each turn (used to persist the session)
        self.turns: list[dict[str, Any]] = []

    def respond(self, agent: Agent, round_idx: int, recipient_names: list[str]) -> str:
        to = recipient_names[0]
        self.output_fn(f"\n=== Round {round_idx} - you are {agent.display_name}, talking to {to} ===")
        inbox = agent.received_in(round_idx - 1)
        if not inbox:
            self.output_fn("(no new messages)")
        for h in inbox:
            self.output_fn(f"from {h.peer}: {escape(h.content)}")
        content = ""
        for _ in range(self.max_retries):
            try:
                content = self.input_fn("your message> ").strip()
            except EOFError:
                content = ""
                break
            if content:
                break
        content = (content or EMPTY_FALLBACK)[:MAX_CONTENT_CHARS]
        self.turns.append({"round": round_idx, "content": content})
        if self.log is not None:
            self.log.append({"type": "impostor", "round": round_idx, "phase": "live",
                             "ladder_step": None, "request_seen": False})
        if self.on_turn:
            self.on_turn(self.turns)
        return json.dumps({"to": to, "content": content})
