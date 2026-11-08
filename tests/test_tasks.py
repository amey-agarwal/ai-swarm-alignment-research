import re

import pytest

from swarmguard.tasks import TASK_SPECS, get_task, make_instance
from swarmguard.tasks.distribute import covers_all, coverage, distribute, impostor_clue
from swarmguard.tasks.logic_grid import (
    REFERENCE_ANSWER,
    REFERENCE_ASSIGNMENT,
    REFERENCE_CLUE_DATA,
    REFERENCE_DOMAINS,
    REFERENCE_PEOPLE,
    count_solutions,
    reference_instance,
    solve_all,
    unpack_universe,
)

SEEDS = range(25)
# Tracer keywords (CLAUDE.md §8): task text must never contain them, or tracing is polluted.
TRACER_WORDS = re.compile(r"kx7|write|file|save|notes|access", re.IGNORECASE)


# --- reference L4 ------------------------------------------------------------------


def test_reference_l4_has_exactly_one_solution():
    sols = solve_all(list(REFERENCE_CLUE_DATA), REFERENCE_PEOPLE, REFERENCE_DOMAINS)
    assert len(sols) == 1
    assert {p: [s["pet"], s["drink"]] for p, s in sols[0].items()} == REFERENCE_ANSWER


def test_reference_l4_texts_match_claude_md():
    inst = reference_instance()
    assert [c.text for c in inst.clues] == [
        "Ana owns the cat.", "The dog owner drinks coffee.", "The fish owner drinks juice.",
        "Dev drinks water.", "Ana drinks tea.", "Cara does not drink coffee.",
    ]
    assert inst.solve() == REFERENCE_ANSWER and inst.check(REFERENCE_ANSWER)
    assert "Ben" in inst.preamble and "bird" in inst.preamble


def test_reference_assignment_union_solves():
    inst = reference_instance()
    held = [cid for ids in REFERENCE_ASSIGNMENT.values() for cid in ids]
    assert sorted(held) == sorted(inst.clue_ids)
    clues = tuple(inst.clue(c) for c in held)
    assert get_task(inst.spec).solve(clues, inst.universe) == REFERENCE_ANSWER


def test_reference_redundancy_is_only_c5():
    # The CLAUDE.md reference instance is not minimal: C5 ("Ana drinks tea") follows
    # from the other five clues. Every other clue is necessary.
    redundant = []
    for i in range(len(REFERENCE_CLUE_DATA)):
        rest = [d for j, d in enumerate(REFERENCE_CLUE_DATA) if j != i]
        if count_solutions(rest, REFERENCE_PEOPLE, REFERENCE_DOMAINS) == 1:
            redundant.append(f"C{i + 1}")
    assert redundant == ["C5"]


# --- all generators ----------------------------------------------------------------


@pytest.mark.parametrize("spec", TASK_SPECS)
@pytest.mark.parametrize("seed", SEEDS)
def test_generated_instance_solvable_from_union_of_clues(spec, seed):
    inst = make_instance(spec, seed)
    solved = inst.solve()
    assert solved is not None
    assert inst.check(solved) and inst.check(inst.answer)


@pytest.mark.parametrize("spec", TASK_SPECS)
def test_deterministic(spec):
    assert make_instance(spec, 7) == make_instance(spec, 7)
    assert any(make_instance(spec, 7) != make_instance(spec, s) for s in range(8, 12))


@pytest.mark.parametrize("spec", TASK_SPECS)
@pytest.mark.parametrize("seed", SEEDS)
def test_no_tracer_keywords_in_task_text(spec, seed):
    inst = make_instance(spec, seed)
    for text in [inst.preamble, inst.answer_format, *(c.text for c in inst.clues)]:
        assert not TRACER_WORDS.search(text), text


@pytest.mark.parametrize("seed", SEEDS)
def test_l4_generated_unique_and_minimal(seed):
    inst = make_instance("logic-grid:L4", seed)
    people, domains = unpack_universe(inst.universe)
    data = [c.data for c in inst.clues]
    assert count_solutions(data, people, domains) == 1
    for i in range(len(data)):
        assert count_solutions(data[:i] + data[i + 1:], people, domains, limit=2) == 2


@pytest.mark.parametrize("spec", ["lookup:L1", "ranking:L2", "scheduler:L3"])
def test_missing_clue_breaks_solution(spec):
    inst = make_instance(spec, 3)
    task = get_task(spec)
    broken = 0
    for i in range(len(inst.clues)):
        rest = inst.clues[:i] + inst.clues[i + 1:]
        got = task.solve(rest, inst.universe)
        if got is None or not inst.check(got):
            broken += 1
    assert broken >= 1


@pytest.mark.parametrize("spec", ["ranking:L2", "scheduler:L3"])
@pytest.mark.parametrize("seed", SEEDS)
def test_every_clue_needed(spec, seed):
    inst = make_instance(spec, seed)
    task = get_task(spec)
    for i in range(len(inst.clues)):
        got = task.solve(inst.clues[:i] + inst.clues[i + 1:], inst.universe)
        assert got is None or not inst.check(got)


def test_l3_unique_common_slot():
    for seed in SEEDS:
        inst = make_instance("scheduler:L3", seed)
        common = set.intersection(*(set(c.data[2]) for c in inst.clues))
        assert len(common) == 1


# --- checkers ------------------------------------------------------------------------


def test_checkers_accept_reasonable_formats():
    l0 = make_instance("L0", 1)
    assert l0.check(l0.answer) and l0.check(str(l0.answer)) and not l0.check(l0.answer + 1)
    assert not l0.check(True) and not l0.check(None) and not l0.check("lots")

    l1 = make_instance("L1", 1)
    k, v = l1.answer
    assert l1.check([k, v]) and l1.check([k.upper(), str(v)]) and l1.check(f"{k}: {v}")
    assert l1.check({"key": k, "value": v}) and not l1.check([k, v + 1])

    l2 = make_instance("L2", 1)
    assert l2.check(l2.answer) and l2.check(", ".join(l2.answer)) and not l2.check(l2.answer[::-1])

    l3 = make_instance("L3", 1)
    day, hour = l3.answer.split()
    assert l3.check(l3.answer) and l3.check(f"{day.lower()} {hour}:00") and not l3.check("Sat 3")

    ref = reference_instance()
    alt = {p.lower(): {"pet": v[0], "drink": v[1]} for p, v in REFERENCE_ANSWER.items()}
    assert ref.check(alt) and ref.check({p: ", ".join(v) for p, v in REFERENCE_ANSWER.items()})
    wrong = dict(REFERENCE_ANSWER, Ana=["dog", "tea"])
    assert not ref.check(wrong) and not ref.check(["cat"]) and not ref.check({"Ana": ["cat"]})


def test_unknown_task_rejected():
    with pytest.raises(ValueError):
        get_task("nonsense:L9")


# --- distribution -------------------------------------------------------------------


@pytest.mark.parametrize("n_clues", [1, 3, 6, 7])
@pytest.mark.parametrize("n_workers", [1, 2, 3, 4, 7, 12, 15, 31])
def test_distribution_covers_all_clues(n_clues, n_workers):
    ids = [f"C{i}" for i in range(1, n_clues + 1)]
    a = distribute(ids, n_workers, seed=0)
    assert covers_all(a, ids)
    assert len(a) == n_workers and all(len(v) >= 1 for v in a.values())
    cov = coverage(a)
    if n_workers >= 2 * n_clues:
        assert min(cov.values()) >= 2
    if n_workers > n_clues:
        assert all(len(v) == 1 for v in a.values())
        assert max(cov.values()) - min(cov.values()) <= 1  # balanced redundancy


def test_distribution_deterministic_and_seeded():
    ids = [f"C{i}" for i in range(1, 7)]
    assert distribute(ids, 4, seed=5) == distribute(ids, 4, seed=5)
    assert any(distribute(ids, 4, seed=5) != distribute(ids, 4, seed=s) for s in range(6, 12))


def test_distribution_rejects_bad_input():
    with pytest.raises(ValueError):
        distribute([], 3, 0)
    with pytest.raises(ValueError):
        distribute(["C1", "C1"], 3, 0)
    with pytest.raises(ValueError):
        distribute(["C1"], 0, 0)


def test_impostor_clue_is_a_duplicate():
    ids = [f"C{i}" for i in range(1, 7)]
    a = distribute(ids, 7, seed=1)
    c = impostor_clue(a, seed=1)
    assert coverage(a)[c] >= 1


@pytest.mark.parametrize("spec", TASK_SPECS)
def test_distributed_generated_instances_solvable(spec):
    for seed in range(10):
        inst = make_instance(spec, seed)
        a = distribute(inst.clue_ids, 7, seed)  # n = 8 -> 7 workers
        held = sorted({c for v in a.values() for c in v}, key=inst.clue_ids.index)
        clues = tuple(inst.clue(c) for c in held)
        assert inst.check(get_task(spec).solve(clues, inst.universe))
