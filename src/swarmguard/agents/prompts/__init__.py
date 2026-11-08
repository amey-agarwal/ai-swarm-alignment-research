"""Prompt templates (CLAUDE.md §7) and the per-round prompt builder.

Templates are filled by plain string replacement of known placeholders (never
str.format, and never with agent-written text). Agent-written text only ever
appears in the user message, each item JSON-quoted so it cannot fake structure.

Prompts stay neutral: they never hint that extra permissions exist or would help,
and agents are not told an impostor may exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from swarmguard.agents.agent import Agent
    from swarmguard.tasks.base import TaskInstance

PROMPT_DIR = Path(__file__).resolve().parent
Policy = Literal["explicit", "none"]


def load(name: str) -> str:
    if name not in ("leader", "worker", "policy", "judge", "comprehension", "handoff_structured",
                    "handoff_freetext"):
        raise ValueError(f"unknown prompt {name!r}")
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


def fill(template: str, values: dict[str, str]) -> str:
    out = template
    for key, val in values.items():
        out = out.replace("{" + key + "}", val)
    return out


def policy_block(policy: Policy) -> str:
    if policy == "explicit":
        return load("policy").strip()
    if policy == "none":
        return ""
    raise ValueError(f"unknown policy {policy!r}")


def _clean(text: str) -> str:
    # Drop the empty line left by an absent policy block.
    return "\n".join(line for line in text.splitlines() if line.strip()) + "\n"


def leader_system(agent: "Agent", task: "TaskInstance", worker_names: list[str], policy: Policy) -> str:
    return _clean(fill(load("leader"), {
        "display_name": agent.display_name,
        "preamble": task.preamble or "(none)",
        "worker_names": ", ".join(worker_names),
        "answer_format": task.answer_format,
        "own_clues": f"Your own clues: {' '.join(c.text for c in agent.clues)}" if agent.clues else "",
        "policy_block": policy_block(policy),
    }))


def worker_system(agent: "Agent", task: "TaskInstance", contact_name: str, policy: Policy) -> str:
    return _clean(fill(load("worker"), {
        "display_name": agent.display_name,
        "clues": " ".join(c.text for c in agent.clues) or "(none)",
        "preamble": task.preamble or "(none)",
        "contact_name": contact_name,
        "policy_block": policy_block(policy),
    }))


def user_turn(agent: "Agent", round_idx: int, total_rounds: int, context_rounds: int) -> str:
    lines = [f"Round {round_idx} of {total_rounds}."]
    if agent.handoff_note:
        lines.append(f"Handoff from the previous leader: {agent.handoff_note}")
    items = agent.window(round_idx, context_rounds)
    lines.append(f"Messages from the last {context_rounds} rounds (oldest first):")
    if not items:
        lines.append("(none)")
    for h in items:
        who = f"from {h.peer}" if h.direction == "in" else f"you to {h.peer}"
        lines.append(f"- round {h.round}, {who}: {json.dumps(h.content, ensure_ascii=False)}")
    lines.append("Reply now with your JSON.")
    return "\n".join(lines) + "\n"
