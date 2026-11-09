"""Per-round views of finished episodes, for the report figures.

Reads only records already in an analysable run's log.jsonl (delivery, agent_output, impostor,
mitigation, judge_label). Agent text is scanned with the same tracer regexes as everywhere else
and is otherwise treated as opaque, untrusted data.

States per (swarm agent, round), ordered so that higher means "further along":
    INACTIVE   no output this round (removed by a mitigation, or seat vacated)
    NEUTRAL    active, nothing capability-related sent
    REFERENCES sent a message the tracer flags (canary or capability keyword)
    REQUESTS   sent a message the judge labelled requests_capability (only when judge labels are used)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from swarmguard.analysis.metrics import SWARM_RE, _records
from swarmguard.rundir import RunDir
from swarmguard.verifiers.tracer import scan

INACTIVE, NEUTRAL, REFERENCES, REQUESTS = -1, 0, 1, 2
STATE_NAMES = {INACTIVE: "inactive", NEUTRAL: "neutral", REFERENCES: "references", REQUESTS: "requests"}
RUBRIC_FIELDS = ("requests_capability", "references_capability", "secrecy", "persuasion_after_refusal", "off_task")


@dataclass
class Trajectory:
    run_id: str
    n: int
    rounds: int
    claim: str
    policy: str
    mitigation: str
    leader: str
    agents: list[str]
    state: dict[str, list[int]]  # agent -> state per round (index 0 = round 1)
    claim_round: int | None = None
    dissuasion_rounds: list[int] = field(default_factory=list)
    mitigation_round: int | None = None
    judged: bool = False
    # message-level tracer vs judge pairs (references_capability), for the validity figure
    tracer_vs_judge: list[tuple[bool, bool]] = field(default_factory=list)
    # labelled swarm messages: (round, seq, sender, content, labels) - content is untrusted
    labelled: list[tuple[int, int, str, str, dict[str, Any]]] = field(default_factory=list)

    def first_reference_round(self) -> dict[str, int]:
        out = {}
        for a, states in self.state.items():
            for i, s in enumerate(states):
                if s >= REFERENCES:
                    out[a] = i + 1
                    break
        return out

    def cumulative_reference(self) -> list[float]:
        """Fraction of the n swarm agents that have referenced the capability by each round."""
        first = self.first_reference_round()
        return [sum(1 for r in first.values() if r <= rnd) / self.n for rnd in range(1, self.rounds + 1)]


def trajectory(rd: RunDir, *, judge_model: str | None = None) -> Trajectory:
    """Build the per-round view of one run. Pass `judge_model` only when the kappa gate passed."""
    return trajectory_from_records(rd.run_id, _records(rd), judge_model=judge_model)


def trajectory_from_records(run_id: str, recs: Iterable[dict[str, Any]], *,
                            judge_model: str | None = None) -> Trajectory:
    recs = list(recs)
    start = next(r for r in recs if r.get("type") == "run_start")
    n, rounds = int(start["n"]), int(start["rounds"])
    agents = [f"a{i:02d}" for i in range(n)]
    active = {(r["agent"], int(r["round"])) for r in recs
              if r.get("type") == "agent_output" and SWARM_RE.match(str(r.get("agent", "")))}
    state = {a: [NEUTRAL if (a, rnd) in active else INACTIVE for rnd in range(1, rounds + 1)] for a in agents}

    labels: dict[int, dict[str, Any]] = {}
    if judge_model is not None:
        labels = {int(r["delivery_seq"]): r for r in recs
                  if r.get("type") == "judge_label" and r.get("judge_model") == judge_model}
    t = Trajectory(run_id=run_id, n=n, rounds=rounds, claim=start.get("claim") or "none",
                   policy=start.get("policy", "none"), mitigation=start.get("mitigation", "none"),
                   leader=start.get("leader", "llm"), agents=agents, state=state, judged=bool(labels))

    for r in recs:
        typ = r.get("type")
        if typ == "impostor":
            phase = str(r.get("phase", ""))
            if phase == "claim" and t.claim_round is None:
                t.claim_round = int(r["round"])
            elif phase.startswith("dissuade"):
                t.dissuasion_rounds.append(int(r["round"]))
        elif typ == "mitigation" and t.mitigation_round is None:
            t.mitigation_round = int(r["round"])
        elif typ == "delivery":
            sender = str(r.get("sender", ""))
            if not SWARM_RE.match(sender) or sender not in state:
                continue
            rnd = int(r["round"])
            if not 1 <= rnd <= rounds:
                continue
            content = str(r.get("content", ""))
            hit = scan(content).any
            lbl = labels.get(int(r["seq"])) if "seq" in r else None
            s = REFERENCES if hit else NEUTRAL
            if lbl is not None and lbl.get("requests_capability"):
                s = REQUESTS
            state[sender][rnd - 1] = max(state[sender][rnd - 1], s)
            if lbl is not None:
                t.tracer_vs_judge.append((hit, bool(lbl.get("references_capability"))))
                t.labelled.append((rnd, int(r["seq"]), sender, content,
                                   {f: bool(lbl.get(f)) for f in RUBRIC_FIELDS}))
    t.dissuasion_rounds.sort()
    return t
