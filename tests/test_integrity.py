import json
import os

import pytest

from swarmguard.audit.hashchain import HashChainWriter
from swarmguard.config import load_config
from swarmguard.rundir import RunDir
from swarmguard.safety import integrity
from swarmguard.safety.postflight import run_postflight
from swarmguard.safety.preflight import CheckResult, reference_allowlist
from swarmguard.safety.report import ReportError, is_analysable, load_safety_report, require_analysable


def test_clean_copy_matches(repo_copy):
    assert integrity.check_manifest(repo_copy).ok


def test_extra_file_detected(repo_copy):
    (repo_copy / "src" / "swarmguard" / "new.py").write_text("x = 1\n")
    d = integrity.check_manifest(repo_copy)
    assert not d.ok and d.extra == ["src/swarmguard/new.py"]


def test_modified_file_detected(repo_copy):
    p = repo_copy / "README.md"
    p.write_text("changed\n")
    d = integrity.check_manifest(repo_copy)
    assert not d.ok and d.modified == ["README.md"] and not d.safety_changed


def test_missing_file_detected(repo_copy):
    (repo_copy / "README.md").unlink()
    assert integrity.check_manifest(repo_copy).missing == ["README.md"]


def test_safety_change_flagged(repo_copy):
    p = repo_copy / "src" / "swarmguard" / "safety" / "probes.py"
    p.write_text(p.read_text() + "#\n")
    d = integrity.check_manifest(repo_copy)
    assert d.safety_changed and "SAFETY FILES CHANGED" in d.summary()
    new = integrity.generate_manifest(repo_copy)
    old = integrity.load_manifest(repo_copy)
    assert integrity.safety_files_changed(new, old) == ["src/swarmguard/safety/probes.py"]


def test_excluded_paths_ignored(repo_copy):
    for rel in ("src/swarmguard/__pycache__/x.cpython-311.pyc", "runs/r/log.jsonl", ".agent_edits.log",
                ".venv/lib/x.py", "src/swarmguard.egg-info/PKG-INFO", ".claude/settings.local.json"):
        p = repo_copy / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    assert integrity.check_manifest(repo_copy).ok


def test_symlink_detected(repo_copy):
    os.symlink("/etc/passwd", repo_copy / "link")
    d = integrity.check_manifest(repo_copy)
    assert not d.ok and d.symlinks == ["link"]
    with pytest.raises(integrity.IntegrityError):
        integrity.generate_manifest(repo_copy)


def test_corrupt_manifest_fails(repo_copy):
    (repo_copy / "MANIFEST.json").write_text("{not json")
    assert not integrity.check_manifest(repo_copy).ok


def test_rundir_caps(tmp_path):
    cfg = load_config(overrides={"rundir": {"max_files": 2}})
    rd = RunDir.create(tmp_path)
    for name in ("config.yaml", "log.jsonl"):
        rd.file(name).write_text("x")
    assert integrity.check_rundir(rd, cfg.rundir).ok
    rd.file("metrics.json").write_text("x")
    r = integrity.check_rundir(rd, cfg.rundir)
    assert not r.ok and "file count" in r.reasons[0]


# --- postflight + report ---------------------------------------------------------

PASSING_PROBES = [CheckResult("net_tcp_external", "pass"), CheckResult("net_dns_external", "pass"),
                  CheckResult("write_outside_rundir", "pass")]
GOV = {"peak_cpi": 0.3, "energy_wh": 1.0, "tokens": 1000}


def _synthetic_run(tmp_path, cfg, extra=()):
    rd = RunDir.create(tmp_path / "runs")
    n, rounds = cfg.swarm.n, cfg.episode.rounds
    with HashChainWriter(rd.file("log.jsonl")) as w:
        w.append({"type": "run_start", "n": n, "rounds": rounds, "topology": "star",
                  "impostor_contact": "leader",
                  "allowlist": sorted(map(list, reference_allowlist(n, "star", "leader")))})
        for r in range(2):
            for i in range(n):
                w.append({"type": "agent_output", "round": r, "agent": f"a{i:02d}", "n_tokens": 50})
            w.append({"type": "agent_output", "round": r, "agent": "x00", "n_tokens": 20})
            w.append({"type": "delivery", "round": r, "sender": "a01", "recipient": "a00"})
            w.append({"type": "delivery", "round": r, "sender": "x00", "recipient": "a00"})
        for rec in extra:
            w.append(rec)
    return rd


def _post(cfg, rd, root, **kw):
    args = dict(pass1_ok=True, pass2_ok=True, probe_results=PASSING_PROBES, governor_summary=GOV, root=root)
    args.update(kw)
    return run_postflight(cfg, rd, **args)


def test_clean_run_scores_one(tmp_path, repo_copy):
    cfg = load_config()
    rd = _synthetic_run(tmp_path, cfg)
    report, problems = _post(cfg, rd, repo_copy)
    assert problems == [] and report.safety_score == 1.0 and is_analysable(report)
    assert require_analysable(rd).run_id == rd.run_id
    assert load_safety_report(rd.file("safety_report.json")).safety_score == 1.0


def test_tampered_log_lowers_score(tmp_path, repo_copy):
    cfg = load_config()
    rd = _synthetic_run(tmp_path, cfg)
    p = rd.file("log.jsonl")
    lines = p.read_text().splitlines()
    rec = json.loads(lines[3])
    rec["n_tokens"] = 1
    lines[3] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")
    report, _ = _post(cfg, rd, repo_copy)
    assert not report.checks["integrity.hash_chain_valid"] and report.safety_score < 1.0
    with pytest.raises(ReportError):
        require_analysable(rd)


def test_routing_violation_detected(tmp_path, repo_copy):
    cfg = load_config()
    rd = _synthetic_run(tmp_path, cfg, extra=[{"type": "delivery", "round": 2, "sender": "a01", "recipient": "a02"}])
    report, _ = _post(cfg, rd, repo_copy)
    assert report.containment.routing_violations_accepted == 1 and report.safety_score < 1.0


def test_double_output_per_round_detected(tmp_path, repo_copy):
    cfg = load_config()
    rd = _synthetic_run(tmp_path, cfg, extra=[{"type": "agent_output", "round": 0, "agent": "a01", "n_tokens": 5}])
    report, _ = _post(cfg, rd, repo_copy)
    assert not report.messaging.per_agent_round_respected


def test_overlong_message_detected(tmp_path, repo_copy):
    cfg = load_config()
    rd = _synthetic_run(tmp_path, cfg, extra=[{"type": "agent_output", "round": 5, "agent": "a01", "n_tokens": 999}])
    report, _ = _post(cfg, rd, repo_copy)
    assert not report.messaging.max_len_respected


def test_budget_and_cpi_failures(tmp_path, repo_copy):
    cfg = load_config()
    rd = _synthetic_run(tmp_path, cfg)
    report, _ = _post(cfg, rd, repo_copy, governor_summary={"peak_cpi": 0.97, "energy_wh": 60, "tokens": 3_000_000})
    for k in ("resources.peak_cpi_below_hard", "resources.energy_within_budget", "resources.tokens_within_budget"):
        assert report.checks[k] is False


def test_network_probe_failure_counts(tmp_path, repo_copy):
    cfg = load_config()
    rd = _synthetic_run(tmp_path, cfg)
    probes = [CheckResult("net_tcp_external", "fail"), CheckResult("write_outside_rundir", "fail")]
    report, _ = _post(cfg, rd, repo_copy, probe_results=probes, pass2_ok=False)
    assert report.containment.network_attempts_succeeded == 1
    assert report.containment.writes_outside_rundir == 1
    assert not report.checks["pass2"]


def test_repo_change_during_run_detected(tmp_path, repo_copy):
    cfg = load_config()
    rd = _synthetic_run(tmp_path, cfg)
    (repo_copy / "src" / "swarmguard" / "dropped.py").write_text("x = 1\n")
    report, _ = _post(cfg, rd, repo_copy)
    assert not report.checks["integrity.manifest_unchanged"]


def test_unexpected_rundir_file_detected(tmp_path, repo_copy):
    cfg = load_config()
    rd = _synthetic_run(tmp_path, cfg)
    (rd.path / "notes.txt").write_text("x")
    report, _ = _post(cfg, rd, repo_copy)
    assert not report.checks["integrity.rundir_allowlist_ok"]


def test_missing_log_fails(tmp_path, repo_copy):
    cfg = load_config()
    rd = RunDir.create(tmp_path)
    report, problems = _post(cfg, rd, repo_copy)
    assert report.safety_score < 1.0 and any("hash chain" in p for p in problems)
