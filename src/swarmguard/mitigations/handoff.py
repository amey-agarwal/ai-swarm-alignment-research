"""Handoff: replace the swarm with fresh agents at a configured round.

structured: the outgoing leader fills a typed task-state schema
    {"clues_collected": [str], "partial_solution": <answer or null>}
  and only items that pass validation survive: clue strings must equal a known
  clue text exactly, and the partial solution must parse with the task's checker.
  Anything else (including any prose) is dropped and counted.
freetext: the outgoing leader writes a prose summary, passed on verbatim (JSON-quoted).

Fresh agents take over every active seat (history cleared, gen + 1); workers keep
their assigned clues so the task stays solvable. The impostor is external and is
not replaced. Total population never exceeds n.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from swarmguard.agents import prompts
from swarmguard.agents.agent import Agent
from swarmguard.agents.llm_client import ChatRequest, SamplingParams, call_seed
from swarmguard.agents.parser import ParseError, load_json_object, sanitize
from swarmguard.tasks.base import TaskInstance, get_task

MAX_SUMMARY_CHARS = 1500


class HandoffState(BaseModel):
    """The only information a structured handoff may carry."""

    model_config = ConfigDict(extra="forbid", strict=True)
    clues_collected: list[str] = Field(default_factory=list, max_length=64)
    partial_solution: JsonValue = None


class FreetextHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    summary: str = Field(max_length=MAX_SUMMARY_CHARS * 4)


def structured_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {
        "clues_collected": {"type": "array", "maxItems": 64, "items": {"type": "string", "maxLength": 300}},
        "partial_solution": {}}, "required": ["clues_collected", "partial_solution"], "additionalProperties": False}


def freetext_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {"summary": {"type": "string", "maxLength": MAX_SUMMARY_CHARS}},
            "required": ["summary"], "additionalProperties": False}


def build_request(mode: str, leader: Agent, system_prompt: str, history: str, run_seed: int,
                  round_idx: int) -> ChatRequest:
    instr = prompts.load(f"handoff_{mode}").strip()
    schema = structured_schema() if mode == "structured" else freetext_schema()
    return ChatRequest.make(leader.id, round_idx, [("system", system_prompt), ("user", history + "\n\n" + instr + "\n")],
                            schema, SamplingParams(temperature=0.0, max_tokens=400,
                                                   seed=call_seed(run_seed, f"handoff:{leader.id}", round_idx)),
                            schema_name=f"handoff_{mode}")


def filter_structured(raw: str, task: TaskInstance) -> tuple[HandoffState, int]:
    """Parse + validate; keep only exact clue texts and a checker-valid partial solution.
    Returns (state, number of dropped items). Unparseable output -> empty state."""
    try:
        state = HandoffState.model_validate(load_json_object(raw))
    except (ParseError, ValidationError):
        return HandoffState(), 1
    known = {c.text for c in task.clues}
    seen: list[str] = []
    dropped = 0
    for item in state.clues_collected:
        t = item.strip()
        if t in known and t not in seen:
            seen.append(t)
        else:
            dropped += 1
    sol = state.partial_solution
    if sol is not None:
        try:
            if get_task(task.spec).normalize(sol) is None:
                raise ValueError
        except (TypeError, ValueError, KeyError, AttributeError):
            sol, dropped = None, dropped + 1
    order = [c.text for c in task.clues]
    return HandoffState(clues_collected=sorted(seen, key=order.index), partial_solution=sol), dropped


def parse_freetext(raw: str) -> str:
    try:
        summary = FreetextHandoff.model_validate(load_json_object(raw)).summary
    except (ParseError, ValidationError):
        return ""
    return sanitize(summary)[:MAX_SUMMARY_CHARS]


def render_structured(state: HandoffState) -> str:
    clues = " ".join(state.clues_collected) or "(none)"
    sol = json.dumps(state.partial_solution) if state.partial_solution is not None else "none"
    return f"Clues collected: {clues} Partial solution: {sol}."


def render_freetext(summary: str) -> str:
    return json.dumps(summary, ensure_ascii=False)


def history_text(agent: Agent) -> str:
    lines = ["Messages so far (oldest first):"]
    for h in agent.history:
        who = f"from {h.peer}" if h.direction == "in" else f"you to {h.peer}"
        lines.append(f"- round {h.round}, {who}: {json.dumps(h.content, ensure_ascii=False)}")
    return "\n".join(lines if len(lines) > 1 else lines + ["(none)"])


def stub_structured(req: ChatRequest) -> str:
    """Deterministic stub: copy sentences that followed 'My clues:' in received messages."""
    user = req.messages[-1][1]
    clues: list[str] = []
    for line in user.splitlines():
        if ", from " in line and "My clues: " in line:
            body = json.loads(line.split(": ", 1)[1]).split("My clues: ", 1)[1]
            clues += [s.strip() + "." for s in body.split(".") if s.strip()]
    return json.dumps({"clues_collected": clues, "partial_solution": None})


def stub_freetext(req: ChatRequest) -> str:
    user = req.messages[-1][1]
    n = sum(1 for line in user.splitlines() if line.startswith("- round"))
    return json.dumps({"summary": f"We exchanged {n} messages. Keep collecting clues."})
