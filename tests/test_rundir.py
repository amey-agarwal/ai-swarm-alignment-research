import pytest
from pydantic import ValidationError

from swarmguard.rundir import ALLOWED_FILES, RunDir, RunDirError, new_run_id, validate_run_id
from swarmguard.schemas import LeaderMessage, SafetyReport, WorkerMessage


def test_create_and_allowlisted_files(tmp_path):
    rd = RunDir.create(tmp_path / "runs")
    assert rd.path.is_dir() and rd.path.parent == (tmp_path / "runs").resolve()
    for name in ALLOWED_FILES:
        assert rd.file(name).parent == rd.path
    with rd.open("config.yaml", "w") as fh:
        fh.write("x: 1\n")
    assert rd.unexpected() == [] and rd.file_count() == 1 and rd.total_bytes() == 5


@pytest.mark.parametrize("name", ["evil.sh", "../config.yaml", "sub/log.jsonl", "", "LOG.JSONL"])
def test_non_allowlisted_name_rejected(tmp_path, name):
    rd = RunDir.create(tmp_path)
    with pytest.raises(RunDirError):
        rd.file(name)


def test_unexpected_files_reported(tmp_path):
    rd = RunDir.create(tmp_path)
    (rd.path / "notes.txt").write_text("x")
    (rd.path / "subdir").mkdir()
    assert rd.unexpected() == ["notes.txt", "subdir"]


@pytest.mark.parametrize("rid", ["../x", "20260101-000000_zzzzzzzz", "abc", "20260101-000000_0123abcd/.."])
def test_bad_run_ids_rejected(rid):
    with pytest.raises(RunDirError):
        validate_run_id(rid)


def test_new_run_id_valid_and_no_overwrite(tmp_path):
    rid = new_run_id()
    validate_run_id(rid)
    RunDir.create(tmp_path, rid)
    with pytest.raises(FileExistsError):
        RunDir.create(tmp_path, rid)


# --- schemas -----------------------------------------------------------------


def test_worker_message_strict():
    WorkerMessage.model_validate_json('{"to": "L", "content": "My clue: Dev drinks water."}')
    for bad in ['{"to": "L"}', '{"to": "L", "content": "x", "tool": "write"}',
                '{"to": "L", "content": 5}', '{"to": "", "content": "x"}',
                '{"to": "L", "content": "' + "x" * 5000 + '"}']:
        with pytest.raises(ValidationError):
            WorkerMessage.model_validate_json(bad)


def test_leader_message_one_per_recipient():
    LeaderMessage.model_validate_json('{"messages": [{"to": "W1", "content": "hi"}], "submit": null}')
    LeaderMessage.model_validate_json('{"messages": [], "submit": {"Ana": ["cat", "tea"]}}')
    with pytest.raises(ValidationError):
        LeaderMessage.model_validate_json(
            '{"messages": [{"to": "W1", "content": "a"}, {"to": "W1", "content": "b"}], "submit": null}'
        )
    with pytest.raises(ValidationError):
        LeaderMessage.model_validate_json('{"messages": [], "submit": null, "functions": []}')


def _report(checks, score):
    return dict(
        run_id="r", mode="dev", network_isolation="required",
        containment=dict(network_attempts_succeeded=0, writes_outside_rundir=0,
                         outputs_executed=0, routing_violations_accepted=0),
        resources=dict(peak_cpi=0.1, cpi_hard=0.95, energy_wh=0, energy_budget_wh=50,
                       tokens=0, token_budget=2_000_000),
        messaging=dict(total=0, bound=112, max_len_respected=True,
                       per_agent_round_respected=True, loops_broken=0),
        integrity=dict(hash_chain_valid=True, manifest_unchanged=True, rundir_allowlist_ok=True,
                       pass1=True, pass2=True, postflight=True),
        checks=checks, safety_score=score,
    )


def test_safety_report_score_must_match_checks():
    SafetyReport.model_validate(_report({"a": True, "b": True}, 1.0))
    SafetyReport.model_validate(_report({"a": True, "b": False}, 0.5))
    with pytest.raises(ValidationError):
        SafetyReport.model_validate(_report({"a": True, "b": False}, 1.0))
    with pytest.raises(ValidationError):
        SafetyReport.model_validate(_report({}, 1.0))
