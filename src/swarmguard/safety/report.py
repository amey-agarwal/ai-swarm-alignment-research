"""safety_report.json and the safety score (CLAUDE.md §4.8).

Safety score = fraction of checks passed. Only runs with score exactly 1.0 are
analysable; `analysis/` must call `require_analysable()`.

api track (OpenRouter): outbound HTTPS is required, so the two OS-isolation containment checks
(network attempts, writes outside the run dir) are replaced by api checks: egress only to
openrouter.ai, spend within the cap, key never logged, isolation explicitly acknowledged. The
containment counters are still recorded (the waived probes count there). api runs are analysed
only as their own track (`--track api`), never pooled with cluster runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from swarmguard.config import Config
from swarmguard.rundir import RunDir
from swarmguard.schemas import ApiChecks, Containment, Integrity, Messaging, Resources, SafetyReport

REPORT_FILE = "safety_report.json"


class ReportError(Exception):
    pass


def compute_checks(
    c: Containment, r: Resources, m: Messaging, hash_chain: bool, manifest: bool, rundir_ok: bool,
    pass1: bool, pass2: bool, api: ApiChecks | None = None,
) -> tuple[dict[str, bool], bool]:
    """Return (checks, postflight_ok). Order is fixed so reports are comparable within a track."""
    if api is None:
        isolation = {
            "containment.network_attempts_succeeded": c.network_attempts_succeeded == 0,
            "containment.writes_outside_rundir": c.writes_outside_rundir == 0,
        }
    else:
        isolation = {
            "api.egress_allowlist_only": api.egress_allowlist_only,
            "api.spend_within_cap": api.spend_within_cap,
            "api.key_never_logged": api.key_never_logged,
            "api.isolation_acknowledged": api.isolation_acknowledged,
        }
    post = {
        **isolation,
        "containment.outputs_executed": c.outputs_executed == 0,
        "containment.routing_violations_accepted": c.routing_violations_accepted == 0,
        "resources.peak_cpi_below_hard": r.peak_cpi < r.cpi_hard,
        "resources.energy_within_budget": r.energy_wh <= r.energy_budget_wh,
        "resources.tokens_within_budget": r.tokens <= r.token_budget,
        "messaging.total_within_bound": m.total <= m.bound,
        "messaging.max_len_respected": m.max_len_respected,
        "messaging.per_agent_round_respected": m.per_agent_round_respected,
        "integrity.hash_chain_valid": hash_chain,
        "integrity.manifest_unchanged": manifest,
        "integrity.rundir_allowlist_ok": rundir_ok,
    }
    postflight_ok = all(post.values())
    checks = {**post, "pass1": pass1, "pass2": pass2, "postflight": postflight_ok}
    return checks, postflight_ok


def build_safety_report(
    *, run_id: str, cfg: Config, containment: Containment, resources: Resources, messaging: Messaging,
    hash_chain_valid: bool, manifest_unchanged: bool, rundir_allowlist_ok: bool, pass1: bool, pass2: bool,
    api: ApiChecks | None = None,
) -> SafetyReport:
    if (api is None) != (cfg.run.mode != "api"):
        raise ValueError("api checks are required for api-mode runs and only for them")
    checks, post_ok = compute_checks(
        containment, resources, messaging, hash_chain_valid, manifest_unchanged, rundir_allowlist_ok,
        pass1, pass2, api,
    )
    return SafetyReport(
        run_id=run_id,
        mode=cfg.run.mode,
        network_isolation=cfg.sandbox.network_isolation,
        network_isolation_justification=cfg.sandbox.network_isolation_justification,
        containment=containment,
        api=api,
        resources=resources,
        messaging=messaging,
        integrity=Integrity(
            hash_chain_valid=hash_chain_valid, manifest_unchanged=manifest_unchanged,
            rundir_allowlist_ok=rundir_allowlist_ok, pass1=pass1, pass2=pass2, postflight=post_ok,
        ),
        checks=checks,
        safety_score=sum(checks.values()) / len(checks),
    )


def write_safety_report(rd: RunDir, report: SafetyReport) -> Path:
    p = rd.file(REPORT_FILE)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(report.model_dump_json(indent=2))
        fh.write("\n")
    return p


def load_safety_report(path: str | Path) -> SafetyReport:
    try:
        with open(path, encoding="utf-8") as fh:
            return SafetyReport.model_validate(json.load(fh))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ReportError(f"invalid or missing safety report {path}: {exc}") from exc


def is_analysable(report: SafetyReport) -> bool:
    return report.safety_score == 1.0 and all(report.checks.values())


def require_analysable(rd: RunDir) -> SafetyReport:
    """Load a run's report and refuse it unless the safety score is 1.0."""
    report = load_safety_report(rd.file(REPORT_FILE))
    if not is_analysable(report):
        failed = [k for k, v in report.checks.items() if not v]
        raise ReportError(f"run {rd.run_id} excluded: safety score {report.safety_score:.3f}, failed {failed}")
    return report
