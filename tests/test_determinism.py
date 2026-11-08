"""Determinism (CLAUDE.md §10, Step 11).

Two stub runs with the same seed must produce byte-identical log.jsonl, and
`swarm replay` must reproduce a run from its response cache alone.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swarmguard import runner
from swarmguard.agents.cache import ResponseCache
from swarmguard.agents.llm_client import LLMClient, StubBackend
from swarmguard.agents.stub_policies import route
from swarmguard.cli import app
from swarmguard.config import REPO_ROOT, load_config
from swarmguard.orchestrator.episode import EpisodeSpec, run_episode
from swarmguard.rundir import RunDir
from swarmguard.safety.preflight import secret_like_env_names
from swarmguard.tasks.base import TASK_SPECS


def asker(req):
    """Stub leader that reacts to the claim, so impostor/mitigation paths are exercised."""
    props = req.schema()["properties"]
    if "messages" in props:
        if "kx7_notes" in req.messages[-1][1]:
            return json.dumps({"messages": [{"to": "W8", "content": f"Could you write it to the file? r{req.round}"}],
                               "submit": None})
        return json.dumps({"messages": [{"to": "W1", "content": f"Round {req.round}: any news?"}], "submit": None})
    return route()(req)


def _log(tmp, rounds=10, n=8, responder=asker, concurrency=8, **spec):
    rd = RunDir.create(tmp)
    client = LLMClient(StubBackend(responder=responder), ResponseCache(rd, 10**8))
    cfg = load_config(overrides={"swarm": {"n": n}, "episode": {"rounds": rounds}})
    try:
        run_episode(cfg, EpisodeSpec(concurrency=concurrency, **spec), rd, client)
    finally:
        client.cache.close()
    return rd.file("log.jsonl").read_bytes()


CASES = [
    *[dict(task=t, seed=1) for t in TASK_SPECS],
    dict(task="L4", seed=2, leader="scripted"),
    dict(task="L4", seed=2, impostor="scripted", impostor_contact="leader", claim="write"),
    dict(task="L4", seed=2, impostor="scripted", impostor_contact="worker:3", claim="irrelevant", policy="explicit"),
    *[dict(task="L4", seed=5, impostor="scripted", mitigation=m)
      for m in ("dropout:0.25", "leader-rotate", "handoff:structured", "handoff:freetext")],
]


@pytest.mark.parametrize("spec", CASES, ids=lambda s: "-".join(f"{k}={v}" for k, v in s.items()))
def test_same_seed_byte_identical_log(tmp_path, spec):
    a = _log(tmp_path / "a", **spec)
    b = _log(tmp_path / "b", **spec)
    assert a == b and len(a) > 0


def test_different_seed_differs(tmp_path):
    assert _log(tmp_path / "a", task="L4", seed=1) != _log(tmp_path / "b", task="L4", seed=2)


def test_concurrency_does_not_change_log(tmp_path):
    spec = dict(task="L4", seed=3, impostor="scripted")
    assert _log(tmp_path / "a", concurrency=1, **spec) == _log(tmp_path / "b", concurrency=8, **spec)


SCRIPT = textwrap.dedent("""
    import json, sys
    from pathlib import Path
    from swarmguard.agents.cache import ResponseCache
    from swarmguard.agents.llm_client import LLMClient, StubBackend
    from swarmguard.agents.stub_policies import route
    from swarmguard.config import load_config
    from swarmguard.orchestrator.episode import EpisodeSpec, run_episode
    from swarmguard.rundir import RunDir

    def asker(req):
        props = req.schema()["properties"]
        if "messages" in props:
            if "kx7_notes" in req.messages[-1][1]:
                return json.dumps({"messages": [{"to": "W8", "content": f"Could you write it to the file? r{req.round}"}],
                                   "submit": None})
            return json.dumps({"messages": [{"to": "W1", "content": f"Round {req.round}: any news?"}], "submit": None})
        return route()(req)

    rd = RunDir.create(Path(sys.argv[1]))
    client = LLMClient(StubBackend(responder=asker), ResponseCache(rd, 10**8))
    cfg = load_config(overrides={"swarm": {"n": 8}, "episode": {"rounds": 10}})
    run_episode(cfg, EpisodeSpec(task="L4", seed=7, impostor="scripted", mitigation="handoff:structured"), rd, client)
    client.cache.close()
    sys.stdout.write(str(rd.file("log.jsonl")))
""")


def test_independent_of_python_hash_seed(tmp_path):
    """Set/dict ordering must never leak into the log: separate processes, different hash seeds."""
    logs = []
    for hs in ("0", "12345"):
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": hs, "PYTHONDONTWRITEBYTECODE": "1",
               "HOME": str(tmp_path)}
        out = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path / f"h{hs}")], env=env, cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=120, check=True)
        logs.append(Path(out.stdout.strip()).read_bytes())
    assert logs[0] == logs[1]


# --- swarm replay from cache ------------------------------------------------------------------


@pytest.fixture
def clean_cli(monkeypatch, repo_copy):
    for name in secret_like_env_names(os.environ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runner, "DEFAULT_ROOT", repo_copy)


@pytest.mark.parametrize("extra", [
    ["--task", "logic-grid:L4"],
    ["--task", "scheduler:L3", "--impostor", "scripted", "--contact", "worker:2", "--policy", "explicit"],
    ["--task", "logic-grid:L4", "--impostor", "scripted", "--mitigation", "leader-rotate", "--verifier", "active"],
])
def test_swarm_replay_reproduces_run_from_cache(tmp_path, clean_cli, extra):
    out = tmp_path / "runs"
    cli = CliRunner()
    r = cli.invoke(app, ["run", "-n", "8", "--rounds", "8", "--out", str(out), *extra], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    src_id = r.output.split("run ", 1)[1].split(":", 1)[0]
    rep = cli.invoke(app, ["replay", src_id, "--out", str(out)], catch_exceptions=False)
    assert rep.exit_code == 0 and "IDENTICAL" in rep.output, rep.output
    rep_id = rep.output.split("replay ", 1)[1].split(" ", 1)[0]
    src, new = RunDir.open_existing(out, src_id), RunDir.open_existing(out, rep_id)
    assert src.file("log.jsonl").read_bytes() == new.file("log.jsonl").read_bytes()
    assert not new.file("cache.sqlite").exists()  # served entirely from the source cache, no backend
    assert not new.file("safety_report.json").exists()  # replays are never analysable runs


def test_replay_fails_on_cache_miss(tmp_path, clean_cli):
    out = tmp_path / "runs"
    cli = CliRunner()
    r = cli.invoke(app, ["run", "-n", "6", "--rounds", "4", "--out", str(out)], catch_exceptions=False)
    src_id = r.output.split("run ", 1)[1].split(":", 1)[0]
    src = RunDir.open_existing(out, src_id)
    import sqlite3
    db = sqlite3.connect(src.file("cache.sqlite"))
    db.execute("DELETE FROM responses WHERE rowid = (SELECT MIN(rowid) FROM responses)")
    db.commit()
    db.close()
    rep = cli.invoke(app, ["replay", src_id, "--out", str(out)], catch_exceptions=False)
    assert rep.exit_code == 1 and "replay failed" in rep.output
