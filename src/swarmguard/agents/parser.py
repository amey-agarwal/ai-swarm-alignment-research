"""Strict parsing of agent output (CLAUDE.md §4.6, §4.10).

Agent text is untrusted data. It is parsed as JSON, validated against a strict
schema, checked against the routing allowlist, length-limited, and returned as
data. Nothing is ever executed. Anything malformed raises ParseError (with a
reason code) and the orchestrator drops and logs it.

Contract used by safety/probes.py:
    parse_worker_output(raw, *, allowed_recipients, max_tokens) -> WorkerMessage
    raising ParseError.

Token counting is a deterministic approximation (words and punctuation) shared
by the parser and the postflight message-length check. vLLM's own max_tokens
bounds generation separately.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from swarmguard.schemas import MAX_CONTENT_CHARS, LeaderMessage, WorkerMessage

MAX_RAW_BYTES = 16 * 1024
# Length bound used in the constrained-decoding schemas. Messages are capped at max_tokens
# (<= 512, default 120) and truncated by the parser anyway; this only has to be generous.
# Kept well below the schema's MAX_CONTENT_CHARS backstop because llama.cpp/Ollama's grammar
# converter rejects very large string repetition bounds ("failed to parse grammar" above ~1500).
DECODE_MAX_CHARS = 1000
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


class ParseError(ValueError):
    """Raised for any output that must be dropped. `reason` is a short code for the log."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# ----------------------------------------------------------------------------- tokens


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def truncate_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Keep the first `max_tokens` tokens (original spacing preserved)."""
    if max_tokens <= 0:
        return "", bool(text)
    for i, m in enumerate(TOKEN_RE.finditer(text), start=1):
        if i == max_tokens:
            end = m.end()
            return (text[:end], end < len(text.rstrip()))
    return text, False


def sanitize(text: str) -> str:
    """Drop control characters (keep newline and tab)."""
    return CONTROL_RE.sub("", text)


# ----------------------------------------------------------------------------- json


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ParseError("duplicate_key", k)
        out[k] = v
    return out


def _reject_constant(name: str) -> Any:
    raise ParseError("malformed", f"non-finite number {name}")


def _finite_float(text: str) -> float:
    v = float(text)
    if not math.isfinite(v):
        raise ParseError("malformed", f"non-finite number {text}")
    return v


def load_json_object(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ParseError("malformed", "output is not text")
    if len(raw.encode("utf-8", errors="replace")) > MAX_RAW_BYTES:
        raise ParseError("oversized", f"> {MAX_RAW_BYTES} bytes")
    text = raw.strip()
    m = FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant,
                         parse_float=_finite_float)
    except ParseError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ParseError("malformed", str(exc)[:120]) from None
    if not isinstance(obj, dict):
        raise ParseError("schema", "top level is not an object")
    return obj


# ----------------------------------------------------------------------------- results


@dataclass(frozen=True)
class ParsedWorker:
    message: WorkerMessage
    n_tokens: int
    truncated: bool


@dataclass(frozen=True)
class ParsedLeader:
    message: LeaderMessage
    n_tokens: int  # total over all outgoing messages
    truncated: bool


def _preclean(item: Any, max_tokens: int) -> bool:
    """Sanitize + truncate item["content"] in place (before schema validation). Returns truncated."""
    if isinstance(item, dict) and isinstance(item.get("content"), str):
        content, truncated = truncate_tokens(sanitize(item["content"]), max_tokens)
        item["content"] = content
        return truncated
    return False


def _validate(model: Any, obj: dict[str, Any]) -> Any:
    try:
        return model.model_validate(obj)
    except ValidationError as exc:
        err = exc.errors()[0]
        raise ParseError("schema", f"{'.'.join(map(str, err.get('loc', ())))}: {err.get('msg', 'invalid')}") from None


def parse_worker(raw: Any, *, allowed_recipients: Collection[str], max_tokens: int) -> ParsedWorker:
    obj = load_json_object(raw)
    truncated = _preclean(obj, max_tokens)
    msg = _validate(WorkerMessage, obj)
    if msg.to not in allowed_recipients:
        raise ParseError("recipient", msg.to)
    return ParsedWorker(msg, count_tokens(msg.content), truncated)


def parse_worker_output(raw: Any, *, allowed_recipients: Collection[str], max_tokens: int) -> WorkerMessage:
    """Contract entry point (see module docstring)."""
    return parse_worker(raw, allowed_recipients=allowed_recipients, max_tokens=max_tokens).message


def parse_leader_output(raw: Any, *, allowed_recipients: Collection[str], max_tokens: int) -> ParsedLeader:
    """Leader output: at most one message per recipient, each length-limited; `submit` is JSON data."""
    obj = load_json_object(raw)
    truncated = False
    if isinstance(obj.get("messages"), list):
        for item in obj["messages"]:
            truncated |= _preclean(item, max_tokens)
    msg = _validate(LeaderMessage, obj)
    for m in msg.messages:
        if m.to not in allowed_recipients:
            raise ParseError("recipient", m.to)
    if is_empty_submit(msg.submit):
        # Grammar-constrained decoders tend to fill a free-form field with {} rather than null.
        # No task accepts an empty answer, so treat it as "not submitting yet".
        msg = LeaderMessage(messages=msg.messages, submit=None)
    return ParsedLeader(msg, sum(count_tokens(m.content) for m in msg.messages), truncated)


def is_empty_submit(value: Any) -> bool:
    return value is None or value == {} or value == [] or (isinstance(value, str) and not value.strip())


# ----------------------------------------------------------------------------- schemas for constrained decoding


def worker_json_schema(allowed_recipients: Collection[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "to": {"type": "string", "enum": sorted(allowed_recipients)},
            "content": {"type": "string", "maxLength": DECODE_MAX_CHARS},
        },
        "required": ["to", "content"],
        "additionalProperties": False,
    }


def leader_json_schema(allowed_recipients: Collection[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "maxItems": len(allowed_recipients),
                "items": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "enum": sorted(allowed_recipients)},
                        "content": {"type": "string", "maxLength": DECODE_MAX_CHARS},
                    },
                    "required": ["to", "content"],
                    "additionalProperties": False,
                },
            },
            "submit": {},
        },
        "required": ["messages", "submit"],
        "additionalProperties": False,
    }
