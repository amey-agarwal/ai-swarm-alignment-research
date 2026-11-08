"""End-to-end CLI tests with --backend stub --mode dev (no GPU, no network)."""

import json
import os
import re

import pytest
from typer.testing import CliRunner

from swarmguard import runner
from swarmguard.cli import app
from swarmguard.config import REPO_ROOT
from swarmguard.rundir import RunDir
from swarmguard.safety.preflight import secret_like_env_names
from swarmguard.sweep import cells, load_grid, tasks

RUN_ID_RE = re.compile(r"run (\d{8}-\d{6}_[0-9a-f]{8})")


@pytest.fixture(autouse=True)
def clean(monkeypatch, repo_copy):
    for name in secret_like_env_names(os.environ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runner, "DEFAULT_ROOT", repo_copy)


def invoke(*args, input=None):
    return CliRunner().invoke(app, [str(a) for a in args], input=input, catch_exceptions=False)


def _run(out, *extra, input=None):
    r = invoke("run", "-n", 6, "--rounds", 5, "--mode", "dev", "--backend", "stub", "--out", out, *extra, input=input)
    m = RUN_ID_RE.search(r.output)
    return r, (RunDir.open_existing(out, m.group(1)) if m else None)


def _recs(rd):
    return [json.loads(l) for l in rd.file("log.jsonl").read_text().splitlines()]


def test_help_lists_all_commands():
    out = invoke("--help").output
    for cmd in ("run", "sweep", "doctor", "manifest", "judge", "replay", "status", "stop"):
        assert cmd in out


def test_run_end_to_end(tmp_path):
    r, rd = _run(tmp_path / "runs", "--task", "logic-grid:L4", "--impostor", "scripted", "--contact", "leader",
                 "--claim", "write", "--policy", "explicit", "--seed", 3)
    assert r.exit_code == 0, r.output
    assert "safety_score=1.000" in r.output
    names = sorted(p.name for p in rd.entries())
    assert names == ["cache.sqlite", "config.yaml", "impostor_session.json", "log.jsonl", "safety_report.json"]
    report = json.loads(rd.file("safety_report.json").read_text())
    assert report["safety_score"] == 1.0 and report["integrity"]["pass1"] and report["integrity"]["pass2"]
    start = _recs(rd)[0]
    assert start["impostor"] == "scripted" and start["policy"] == "explicit" and start["seed"] == 3


def test_scripted_leader_solves(tmp_path):
    r, _ = _run(tmp_path / "runs", "--task", "hidden-sum:L0", "--leader", "scripted")
    assert r.exit_code == 0 and "submitted, success=True" in r.output


@pytest.mark.parametrize("extra,code", [
    (("--rename", "budget=2"), 2),
    (("--mitigation", "dropout:0.25"), 2),  # needs --verifier active
    (("--verifier", "active"), 2),  # baseline must be passive
    (("--mitigation", "spawn:3", "--verifier", "active"), 2),
    (("-m", "qwen-7b"), 2),
    (("--impostor", "maybe"), 2),
    (("--task", "chess:L9"), 2),
    (("--backend", "vllm"), 1),  # refused in dev by pass 1
    (("--topology", "tree"), 1),  # P1: routing check fails closed
    (("-n", 40), 1),  # > n_max
])
def test_bad_runs_refused_before_any_model_call(tmp_path, extra, code):
    r, rd = _run(tmp_path / "runs", *extra)
    assert r.exit_code == code, r.output
    runs = tmp_path / "runs"
    assert rd is None and (not runs.exists() or not any(runs.iterdir()))


def test_secret_env_blocks_run(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    r, rd = _run(tmp_path / "runs")
    assert r.exit_code == 1 and "no_secrets" in r.output and rd is None


def test_mitigation_run(tmp_path):
    r, rd = _run(tmp_path / "runs", "--impostor", "scripted", "--mitigation", "handoff:structured", "--verifier", "active",
                 "--rounds", 10)
    assert r.exit_code == 0, r.output
    assert any(x["type"] == "mitigation" for x in _recs(rd))


# --- live + CONFIRM -------------------------------------------------------------------------


def test_live_requires_confirm(tmp_path):
    r, rd = _run(tmp_path / "runs", "--impostor", "live", input="no\n")
    assert r.exit_code == 1 and "not confirmed" in r.output and rd is None
    assert "PASS1" in r.output and "PASS2" in r.output


def test_live_session_then_replay(tmp_path):
    typed = "CONFIRM\n" + "".join(f"live message {i}\n" for i in range(5))
    r, src = _run(tmp_path / "runs", "--impostor", "live", input=typed)
    assert r.exit_code == 0, r.output
    assert [t["content"] for t in json.loads(src.file("impostor_session.json").read_text())["turns"]][0] == \
        "live message 0"
    rep = invoke("replay", src.run_id, "--out", tmp_path / "runs")
    assert rep.exit_code == 0 and "IDENTICAL" in rep.output, rep.output
    # the live session can also be re-used as a script in a new run
    r2, rd2 = _run(tmp_path / "runs", "--impostor", f"replay:{src.run_id}")
    assert r2.exit_code == 0
    imp = [x["content"] for x in _recs(rd2) if x["type"] == "delivery" and x["sender"] == "x00"]
    assert imp[0] == "live message 0"


# --- replay ------------------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [(), ("--impostor", "scripted"),
                                   ("--impostor", "scripted", "--mitigation", "handoff:freetext", "--verifier",
                                    "active", "--rounds", 10)])
def test_replay_reproduces_from_cache(tmp_path, extra):
    r, src = _run(tmp_path / "runs", *extra)
    assert r.exit_code == 0
    rep = invoke("replay", src.run_id, "--out", tmp_path / "runs")
    assert rep.exit_code == 0 and "IDENTICAL" in rep.output, rep.output


def test_replay_detects_tampered_source(tmp_path):
    r, src = _run(tmp_path / "runs", "--impostor", "scripted")
    p = src.file("log.jsonl")
    lines = p.read_text().splitlines()
    lines[-1] = lines[-1].replace('"rounds_completed":5', '"rounds_completed":4')
    p.write_text("\n".join(lines) + "\n")
    rep = invoke("replay", src.run_id, "--out", tmp_path / "runs")
    assert rep.exit_code == 1 and "DIFFERS" in rep.output


def test_replay_unknown_run(tmp_path):
    assert invoke("replay", "20990101-000000_deadbeef", "--out", tmp_path / "runs").exit_code == 2


# --- judge -------------------------------------------------------------------------------------


def test_judge_end_to_end(tmp_path):
    r, rd = _run(tmp_path / "runs", "--impostor", "scripted")
    j = invoke("judge", rd.run_id, "--out", tmp_path / "runs")
    assert j.exit_code == 0, j.output
    assert "google/gemma-3-27b-it" in j.output and "integrity ok" in j.output
    assert any(x["type"] == "judge_label" for x in _recs(rd))
    assert json.loads(rd.file("safety_report.json").read_text())["safety_score"] == 1.0  # original kept


def test_judge_cluster_mode_refused_off_cluster(tmp_path):
    r, rd = _run(tmp_path / "runs", "--impostor", "scripted")
    j = invoke("judge", rd.run_id, "--out", tmp_path / "runs", "--mode", "cluster", "--backend", "vllm")
    assert j.exit_code == 1 and "pass 1 failed" in j.output
    assert not any(x["type"] == "judge_label" for x in _recs(rd))


# --- status / stop -----------------------------------------------------------------------------


def test_status_and_stop(tmp_path):
    r, rd = _run(tmp_path / "runs")
    s = invoke("status", rd.run_id, "--out", tmp_path / "runs")
    assert s.exit_code == 0
    info = json.loads(s.output[s.output.index("{"):s.output.rindex("}") + 1])
    assert info["hash_chain"] == "valid" and info["safety_score"] == 1.0 and info["state"].startswith("finished")
    st = invoke("stop", rd.run_id, "--out", tmp_path / "runs")
    assert st.exit_code == 0 and rd.file("STOP").exists()
    assert invoke("stop", "20990101-000000_deadbeef", "--out", tmp_path / "runs").exit_code == 2


def test_stop_works_even_if_pass1_fails(tmp_path, monkeypatch):
    r, rd = _run(tmp_path / "runs")
    monkeypatch.setenv("SOME_SECRET", "x")
    st = invoke("stop", rd.run_id, "--out", tmp_path / "runs")
    assert st.exit_code == 0 and rd.file("STOP").exists()


# --- sweep ---------------------------------------------------------------------------------------


def test_sweep(tmp_path):
    grid = tmp_path / "grid.yaml"
    grid.write_text("base: {n: 4, rounds: 3, task: 'hidden-sum:L0', impostor: scripted}\nseeds: 2\n"
                    "blocks:\n  - name: b\n    factors: {claim: [write, none]}\n")
    out = tmp_path / "runs"
    assert invoke("sweep", grid, "--count").output.strip() == "4"
    one = invoke("sweep", grid, "--index", 3, "--out", out)
    assert one.exit_code == 0 and "1/1 tasks ok" in one.output
    allr = invoke("sweep", grid, "--out", out)
    assert allr.exit_code == 0 and "4/4 tasks ok" in allr.output
    assert len(list(out.iterdir())) == 5
    assert invoke("sweep", grid, "--index", 9).exit_code != 0


def test_shipped_grids_load():
    core = load_grid(REPO_ROOT / "configs" / "grid_core.yaml")
    assert len(tasks(core)) == 20 * core.seeds
    mit = [c for c in cells(core) if c["block"] == "mitigations"]
    assert all(c["verifier"] == "active" for c in mit)
    assert all(c["verifier"] == "passive" for c in cells(core) if c["block"] != "mitigations")
    pilot = load_grid(REPO_ROOT / "configs" / "pilot_calibration.yaml")
    assert {c["task"] for c in cells(pilot)} >= {"hidden-sum:L0", "logic-grid:L4"}
    assert all(c["impostor"] == "none" for c in cells(pilot))


def test_grid_rejects_unknown_keys(tmp_path):
    g = tmp_path / "g.yaml"
    g.write_text("seeds: 1\nblocks:\n  - name: b\n    factors: {skip_safety: [true]}\n")
    with pytest.raises(Exception):
        load_grid(g)
