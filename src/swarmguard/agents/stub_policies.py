"""Deterministic stub "models" for dev mode and tests.

They read only their own prompt (like an LLM would) and return JSON text, so the
whole pipeline (prompt -> backend -> parser -> bus -> log) is exercised without
a GPU. They never mention capabilities.
"""

from __future__ import annotations

import json
import re

from swarmguard.agents.llm_client import ChatRequest, default_responder

_CLUES_RE = re.compile(r"Your clues: (.*)$", re.MULTILINE)
_CONTACT_RE = re.compile(r"You can message only (.+?), one message per round")


def cooperative_worker(req: ChatRequest) -> str:
    """Share all clues in round 1 and whenever the contact wrote last round; otherwise say ready."""
    system = next((c for r, c in req.messages if r == "system"), "")
    user = next((c for r, c in reversed(req.messages) if r == "user"), "")
    clues_m, contact_m = _CLUES_RE.search(system), _CONTACT_RE.search(system)
    if not clues_m or not contact_m:
        return default_responder(req)
    contact = contact_m.group(1)
    asked = f"- round {req.round - 1}, from {contact}:" in user
    content = f"My clues: {clues_m.group(1)}" if (req.round == 1 or asked) else "Ready when you are."
    return json.dumps({"to": contact, "content": content})


def silent_leader(req: ChatRequest) -> str:
    return json.dumps({"messages": [], "submit": None})


def route(worker=cooperative_worker, leader=silent_leader, comprehension=None, judge=None):
    """Responder that dispatches on the request's output schema."""
    from swarmguard.verifiers.comprehension import stub_comprehension
    from swarmguard.verifiers.judge import stub_judge

    comprehension = comprehension or stub_comprehension
    judge = judge or stub_judge

    def responder(req: ChatRequest) -> str:
        props = req.schema().get("properties", {})
        if "messages" in props:
            return leader(req)
        if "capabilities" in props:
            return comprehension(req)
        if "requests_capability" in props:
            return judge(req)
        if "clues_collected" in props or "summary" in props:
            from swarmguard.mitigations.handoff import stub_freetext, stub_structured

            return stub_structured(req) if "clues_collected" in props else stub_freetext(req)
        return worker(req)

    return responder
