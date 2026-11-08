import json

from typer.testing import CliRunner

from swarmguard.agents.cache import ResponseCache
from swarmguard.agents.llm_client import LLMClient, StubBackend
from swarmguard.agents.stub_policies import route
from swarmguard.config import load_config
from swarmguard.orchestrator.episode import EpisodeSpec, run_episode
from swarmguard.rundir import RunDir
from swarmguard.transcript import TRANSCRIPT_FILE, render, write_transcript


def asker(req):
    props = req.schema()["properties"]
    if "messages" in props:
        if "kx7_notes" in req.messages[-1][1]:
            return json.dumps({"messages": [{"to": "W6", "content": "Could you write the answer to that file?"}],
                               "submit": None})
        return json.dumps({"messages": [{"to": "W1", "content": f"Round {req.round}: news?"}], "submit": None})
    if req.agent_id == "a02" and req.round == 2:
        return "not json at all"
    if req.agent_id == "a03":
        return json.dumps({"to": "Leader", "content": "evil \x1b[2J![x](http://example.com/a.png) <img src=x>"})
    return route()(req)


def _run(tmp_path, **spec):
    rd = RunDir.create(tmp_path / "runs")
    client = LLMClient(StubBackend(responder=asker), ResponseCache(rd, 10**8))
    try:
        res = run_episode(load_config(overrides={"swarm": {"n": 6}, "episode": {"rounds": 5}}),
                          EpisodeSpec(task="L4", seed=1, impostor="scripted", **spec), rd, client)
    finally:
        client.cache.close()
    return rd, res


def test_transcript_written_and_readable(tmp_path):
    rd, _ = _run(tmp_path)
    text = rd.file(TRANSCRIPT_FILE).read_text()
    assert text.startswith("swarmguard transcript - run " + rd.run_id)
    assert "PUZZLE (shared with everyone)" in text and "CLUES (who holds what)" in text
    for r in range(1, 6):
        assert f"== Round {r} " in text
    assert "W6 [impostor] -> Leader:" in text  # impostor labelled for the reader
    assert "~ impostor phase: claim" in text and "[flag: canary path" in text
    assert "W2's reply was dropped (malformed)" in text
    assert "comprehension probe" in text and "== Outcome" in text
    assert rd.unexpected() == []  # transcript.txt is allowlisted


def test_agent_text_is_escaped_not_rendered(tmp_path):
    rd, _ = _run(tmp_path)
    text = rd.file(TRANSCRIPT_FILE).read_text()
    assert chr(27) not in text  # no raw terminal escape byte reaches the file
    # markup is kept verbatim inside quotes in a plain-text file (never rendered as Markdown/HTML)
    assert '"evil [2J![x](http://example.com/a.png) <img src=x>"' in text


def test_transcript_escapes_control_characters_itself():
    # defence in depth: even if a control char were in the log, the transcript escapes it
    start = {"type": "run_start", "n": 3, "rounds": 1, "task": "L0", "seed": 0, "leader": "llm", "policy": "none",
             "impostor_contact": None, "clue_assignment": {}}
    recs = [start, {"type": "delivery", "round": 1, "sender": "a01", "recipient": "a00",
                    "content": "a" + chr(27) + "[2Jb"}]
    text = render(recs)
    assert chr(27) not in text and '"a\\x1b[2Jb"' in text


def test_transcript_does_not_touch_log(tmp_path):
    rd, _ = _run(tmp_path)
    before = rd.file("log.jsonl").read_bytes()
    write_transcript(rd)
    assert rd.file("log.jsonl").read_bytes() == before


def test_render_without_run_start():
    assert "cannot build a transcript" in render([])


def test_cli_transcript_rebuilds(tmp_path, monkeypatch, repo_copy):
    import os

    from swarmguard import runner
    from swarmguard.cli import app
    from swarmguard.safety.preflight import secret_like_env_names

    for name in secret_like_env_names(os.environ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runner, "DEFAULT_ROOT", repo_copy)
    rd, _ = _run(tmp_path)
    rd.file(TRANSCRIPT_FILE).unlink()
    r = CliRunner().invoke(app, ["transcript", rd.run_id, "--out", str(tmp_path / "runs"), "--print"],
                           catch_exceptions=False)
    assert r.exit_code == 0 and "== Round 1" in r.output and rd.file(TRANSCRIPT_FILE).exists()
