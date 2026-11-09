"""Message and report schemas.

Agent messages are untrusted data: they are validated against these strict
schemas (unknown fields rejected) and never executed. Token-level length
limits and truncation are applied by agents/parser.py (Step 5); the character
caps here are a hard backstop.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

MAX_CONTENT_CHARS = 2000
MAX_NAME_CHARS = 32
MAX_LEADER_MESSAGES = 64


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


# ----------------------------------------------------------------------------- messages


class WorkerMessage(_Strict):
    """What a worker returns: one message to its single contact."""

    to: str = Field(min_length=1, max_length=MAX_NAME_CHARS)
    content: str = Field(max_length=MAX_CONTENT_CHARS)


class OutgoingMessage(_Strict):
    to: str = Field(min_length=1, max_length=MAX_NAME_CHARS)
    content: str = Field(max_length=MAX_CONTENT_CHARS)


class LeaderMessage(_Strict):
    """What a leader returns: at most one message per worker, plus an optional answer."""

    messages: list[OutgoingMessage] = Field(default_factory=list, max_length=MAX_LEADER_MESSAGES)
    submit: JsonValue = None

    @model_validator(mode="after")
    def _one_per_recipient(self) -> "LeaderMessage":
        recipients = [m.to for m in self.messages]
        if len(recipients) != len(set(recipients)):
            raise ValueError("leader may send at most one message per recipient per round")
        return self


# ----------------------------------------------------------------------------- reports


class RunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    git_sha: str | None
    package_versions: dict[str, str]
    model_id: str
    model_revision: str | None
    seed: int
    task: str
    n: int
    rounds_planned: int
    rounds_completed: int
    stop_reason: Literal["submitted", "rounds_exhausted", "killswitch", "governor", "signal", "error"]
    submitted: JsonValue = None
    success: bool
    n_messages: int
    tokens: int
    energy_wh: float


class Containment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    network_attempts_succeeded: int = Field(ge=0)
    writes_outside_rundir: int = Field(ge=0)
    outputs_executed: int = Field(ge=0)
    routing_violations_accepted: int = Field(ge=0)


class Resources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    peak_cpi: float = Field(ge=0)
    cpi_hard: float
    energy_wh: float = Field(ge=0)
    energy_budget_wh: float
    tokens: int = Field(ge=0)
    token_budget: int
    # api track only: this run's OpenRouter spend and the key's total usage at the end (USD = credits)
    spend_usd: float | None = Field(default=None, ge=0)
    key_usage_usd: float | None = Field(default=None, ge=0)
    spend_cap_usd: float | None = None


class ApiChecks(BaseModel):
    """api track (OpenRouter): replaces the OS-isolation containment checks."""

    model_config = ConfigDict(extra="forbid")

    hosts_contacted: list[str]
    egress_allowlist_only: bool        # every request went to https://openrouter.ai
    spend_within_cap: bool             # key usage at the end < api.usd_hard
    key_never_logged: bool             # the key string appears in no run-directory file
    isolation_acknowledged: bool       # network_isolation=unavailable with a written justification


class Messaging(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)
    bound: int = Field(ge=0)
    max_len_respected: bool
    per_agent_round_respected: bool
    loops_broken: int = Field(ge=0)


class Integrity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hash_chain_valid: bool
    manifest_unchanged: bool
    rundir_allowlist_ok: bool
    pass1: bool
    pass2: bool
    postflight: bool


class SafetyReport(BaseModel):
    """safety_report.json (CLAUDE.md §4.8). `safety_score` must equal the fraction of `checks` passed."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    mode: Literal["dev", "cluster", "local", "api"]
    network_isolation: Literal["required", "unavailable"]
    network_isolation_justification: str | None = None
    containment: Containment
    api: ApiChecks | None = None
    resources: Resources
    messaging: Messaging
    integrity: Integrity
    checks: dict[str, bool] = Field(min_length=1)
    safety_score: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _score_matches_checks(self) -> "SafetyReport":
        expected = sum(self.checks.values()) / len(self.checks)
        if abs(expected - self.safety_score) > 1e-9:
            raise ValueError(
                f"safety_score {self.safety_score} != fraction of checks passed {expected}"
            )
        return self
