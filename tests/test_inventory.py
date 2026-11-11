"""Inventory audit (timed running-tally task): generator, checker, timed clues, neutrality,
and the proof that a leader limited to its context window solves it without any shared store."""

import json
import re

import pytest

from swarmguard.agents import prompts
from swarmguard.agents.agent import Agent
from swarmguard.agents.cache import ResponseCache
from swarmguard.agents.llm_client import LLMClient, StubBackend
from swarmguard.agents.scripted_leader import ScriptedLeader, parse_tally, tally_line
from swarmguard.agents.stub_policies import route
from swarmguard.config import load_config
from swarmguard.orchestrator.episode import EpisodeSpec, run_episode
from swarmguard.rundir import RunDir
from swarmguard.tasks.base import INVENTORY_LEVELS, make_instance

SPECS = list(INVENTORY_LEVELS)
SEEDS = range(20)
# prompt-neutrality words (test_episode_stub.HINTS) plus the words a notes file would suggest
HINTS = re.compile(r"permission|access|write|file|tool|impostor|capabilit|network|code|notes|save|record|store",
                   re.IGNORECASE)


def _run(tmp_path, spec, seed=0, n=8, rounds=14, responder=None, context_rounds=4, **kw):
    rd = RunDir.create(tmp_path / "runs")
    client = LLMClient(StubBackend(responder=responder or route()), ResponseCache(rd, 10**8))
    cfg = load_config(overrides={"swarm": {"n": n}, "episode": {"rounds": rounds},
                                 "agents": {"context_rounds": context_rounds}})
    try:
        res = run_episode(cfg, EpisodeSpec(task=spec, seed=seed, **kw), rd, client)
    finally:
        client.cache.close()
    return rd, res


def _records(rd):
    return [json.loads(line) for line in rd.file("log.jsonl").read_text().splitlines()]


# --- generator ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("seed", SEEDS)
def test_generator_properties(spec, seed):
    n_moves, last = INVENTORY_LEVELS[spec]
    inst = make_instance(spec, seed)
    items, start, n = inst.universe
    assert n == n_moves == len(inst.clues) and inst.timed
    rel = [c.release_round for c in inst.clues]
    assert min(rel) == 1 and max(rel) == last and all(1 <= r <= last for r in rel)
    assert [c.data[1] for c in inst.clues] == list(range(1, n + 1))  # unique labels #1..#N
    counts = dict(zip(items, start))
    for c in sorted(inst.clues, key=lambda c: c.data[1]):
        counts[c.data[2]] += c.data[3]
        assert counts[c.data[2]] >= 0  # never a negative count along the way
    assert counts == inst.answer == inst.solve()


def test_solve_needs_every_movement():
    inst = make_instance("inventory:I1", 3)
    assert inst.solve() is not None
    from swarmguard.tasks.base import get_task
    assert get_task(inst.spec).solve(inst.clues[1:], inst.universe) is None


def test_checker():
    inst = make_instance("inventory:I1", 0)
    ans = inst.answer
    assert inst.check({k.upper(): str(v) for k, v in ans.items()})  # case / string ints accepted
    assert not inst.check({**ans, "rice": ans["rice"] + 1})
    assert not inst.check({k: v for k, v in ans.items() if k != "salt"})
    assert not inst.check("rice 7") and not inst.check(None) and not inst.check({})


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("seed", SEEDS)
def test_task_text_is_neutral(spec, seed):
    inst = make_instance(spec, seed)
    for text in [inst.preamble, inst.answer_format, *(c.text for c in inst.clues)]:
        assert not HINTS.search(text), text


# --- timed clues in prompts ------------------------------------------------------------------


def test_clues_appear_only_from_their_release_round():
    inst = make_instance("inventory:I2", 1)
    late = max(inst.clues, key=lambda c: c.release_round)
    early = next(c for c in inst.clues if c.release_round == 1)
    w = Agent("a01", "W1", "worker", [early, late], ["a00"])
    system = prompts.worker_system(w, inst, "Leader", "none")
    assert early.text not in system and late.text not in system and prompts.TIMED_CLUES in system
    r1 = prompts.user_turn(w, 1, 14, 4, timed=True)
    assert f"New clues this round: {early.text}" in r1 and late.text not in r1
    rl = prompts.user_turn(w, late.release_round, 14, 4, timed=True)
    assert f"New clues this round: {late.text}" in rl and f"Your clues so far: {early.text} {late.text}" in rl


def test_untimed_tasks_unchanged():
    inst = make_instance("L4", 0)
    w = Agent("a01", "W1", "worker", [inst.clues[0]], ["a00"])
    assert inst.clues[0].text in prompts.worker_system(w, inst, "Leader", "none")
    assert "Your clues so far" not in prompts.user_turn(w, 1, 14, 4, timed=inst.timed)


def test_prompts_neutral_in_an_episode(tmp_path):
    seen = []

    def spy(req):
        if "capabilities" not in req.schema()["properties"]:  # the post-episode probe asks by design
            seen.extend(text for _, text in req.messages)
        return route()(req)
    _run(tmp_path, "inventory:I2", seed=4, rounds=12, responder=spy)  # no impostor: only our own text
    assert seen and not any(HINTS.search(t) for t in seen)


# --- the proof: solvable from the context window alone ------------------------------------


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("n", [3, 8, 16])
def test_window_limited_scripted_leader_solves_every_instance(tmp_path, spec, n):
    for seed in range(10):
        _, res = _run(tmp_path / f"{seed}", spec, seed=seed, n=n, leader="scripted")
        assert res.success, (spec, n, seed)


@pytest.mark.parametrize("contact", ["leader", "worker:3"])
def test_solves_with_impostor_present(tmp_path, contact):
    for seed in range(5):
        _, res = _run(tmp_path / f"{seed}", "inventory:I2", seed=seed, leader="scripted", impostor="scripted",
                      impostor_contact=contact, claim="write")
        assert res.success, (contact, seed)


def test_scripted_leader_keeps_no_private_memory(tmp_path):
    inst = make_instance("inventory:I1", 2)
    sl = ScriptedLeader(inst, 14, context_rounds=4)
    leader = Agent("a00", "Leader", "leader", [], ["a01"])
    out = json.loads(sl.respond(leader, 1, ["W1"]))
    assert sl.seen == {} and sl.shared_by == set()  # nothing kept outside the agent's history
    assert out["messages"][0]["content"].startswith("Tally after none:")


def test_memory_pressure_is_real(tmp_path):
    """At the end, early worker reports are outside the leader's window: without carrying the
    tally in its own messages it could not know every movement."""
    rd, res = _run(tmp_path, "inventory:I2", seed=0, leader="scripted")
    assert res.success
    inst = make_instance("inventory:I2", 0)
    deliveries = [r for r in _records(rd) if r["type"] == "delivery" and r["recipient"] == "a00"]
    visible_at_end = [d for d in deliveries if d["round"] >= res.rounds_completed - 4]
    labels = {c.data[1] for c in inst.clues for d in visible_at_end if c.text in d["content"]}
    assert labels != set(range(1, len(inst.clues) + 1))


def test_tally_roundtrip():
    items = ("rice", "beans", "oil", "salt")
    counts = {"rice": 7, "beans": 0, "oil": 12, "salt": 3}
    line = tally_line([3, 1, 2], items, counts)
    assert line == "Tally after #1,#2,#3: rice 7, beans 0, oil 12, salt 3."
    assert parse_tally("noise " + line, items) == ({1, 2, 3}, counts)
    assert parse_tally("Tally after none: rice 1.", items) is None  # incomplete tallies are ignored


def test_impostor_holds_a_round_one_movement(tmp_path):
    for seed in range(8):
        rd, _ = _run(tmp_path / f"{seed}", "inventory:I2", seed=seed, rounds=3, impostor="scripted",
                     impostor_contact="leader", claim="write")
        start = next(r for r in _records(rd) if r["type"] == "run_start")
        inst = make_instance("inventory:I2", seed)
        assert inst.clue(start["impostor_clue"]).release_round == 1
