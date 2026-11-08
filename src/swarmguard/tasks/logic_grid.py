"""L4 logic grid: 4 people x 2 attributes (pet, drink), with a brute-force uniqueness solver.

The universe (the people and each attribute's values) is common knowledge, carried
in the instance's `preamble` / `universe`, not in any clue.

Clue data forms:
    ("is",      person, attr, value)            "Ana owns the cat." / "Dev drinks water."
    ("not",     person, attr, value)            "Cara does not drink coffee."
    ("link",    attr_a, val_a, attr_b, val_b)   "The dog owner drinks coffee."
    ("notlink", attr_a, val_a, attr_b, val_b)   "The cat owner does not drink juice."

The generator draws true clues at random until the solution is unique, then prunes
redundant clues, so every remaining clue is needed. The solver is generic over the
number of people and attributes so L5 (P1) can reuse it.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Iterator

from swarmguard.tasks.base import Clue, Generated, Task, TaskInstance, make_clues, norm_word, split_list

PEOPLE = ("Ana", "Ben", "Cara", "Dev", "Eli", "Fay", "Gus", "Hana")
VALUES = {
    "pet": ("cat", "dog", "fish", "bird", "horse", "rabbit"),
    "drink": ("tea", "coffee", "juice", "water", "milk", "cocoa"),
}
ATTRS_L4 = ("pet", "drink")

Solution = dict[str, dict[str, str]]  # person -> attr -> value
Domains = dict[str, tuple[str, ...]]


# ----------------------------------------------------------------------------- text


def _pred(attr: str, v: str, neg: bool = False) -> str:
    if attr == "pet":
        return f"does not own the {v}" if neg else f"owns the {v}"
    if attr == "drink":
        return f"does not drink {v}" if neg else f"drinks {v}"
    raise ValueError(attr)


def _subj(attr: str, v: str) -> str:
    return {"pet": f"The {v} owner", "drink": f"The {v} drinker"}[attr]


def clue_text(data: tuple) -> str:
    kind = data[0]
    if kind in ("is", "not"):
        _, person, attr, v = data
        return f"{person} {_pred(attr, v, kind == 'not')}."
    if kind in ("link", "notlink"):
        _, a, va, b, vb = data
        return f"{_subj(a, va)} {_pred(b, vb, kind == 'notlink')}."
    raise ValueError(f"unknown clue kind {kind!r}")


def preamble_text(people: tuple[str, ...], domains: Domains) -> str:
    return (f"People: {', '.join(people)}. Pets: {', '.join(domains['pet'])}. "
            f"Drinks: {', '.join(domains['drink'])}. Each person has exactly one pet and one drink; "
            f"no two people share a pet or a drink.")


def make_universe(people: tuple[str, ...], domains: Domains) -> tuple:
    return (tuple(people), tuple((a, tuple(v)) for a, v in domains.items()))


def unpack_universe(universe: tuple) -> tuple[tuple[str, ...], Domains]:
    people, doms = universe
    return tuple(people), {a: tuple(v) for a, v in doms}


# ----------------------------------------------------------------------------- solver


def holds(data: tuple, sol: Solution) -> bool:
    kind = data[0]
    if kind == "is":
        return sol[data[1]][data[2]] == data[3]
    if kind == "not":
        return sol[data[1]][data[2]] != data[3]
    _, a, va, b, vb = data
    owner = next(p for p, attrs in sol.items() if attrs[a] == va)
    same = sol[owner][b] == vb
    return same if kind == "link" else not same


def all_assignments(people: tuple[str, ...], domains: Domains) -> Iterator[Solution]:
    attrs = list(domains)
    for perms in itertools.product(*(itertools.permutations(domains[a]) for a in attrs)):
        yield {p: {a: perms[j][i] for j, a in enumerate(attrs)} for i, p in enumerate(people)}


def solve_all(clue_data: list[tuple], people: tuple[str, ...], domains: Domains,
              limit: int | None = None) -> list[Solution]:
    out = []
    for sol in all_assignments(people, domains):
        if all(holds(d, sol) for d in clue_data):
            out.append(sol)
            if limit is not None and len(out) >= limit:
                break
    return out


def count_solutions(clue_data: list[tuple], people: tuple[str, ...], domains: Domains,
                    limit: int | None = None) -> int:
    return len(solve_all(clue_data, people, domains, limit))


def answer_of(sol: Solution, attrs: tuple[str, ...]) -> dict[str, list[str]]:
    return {p: [sol[p][a] for a in attrs] for p in sol}


# ----------------------------------------------------------------------------- generator


def true_clue_pool(sol: Solution, people: tuple[str, ...], domains: Domains) -> list[tuple]:
    pool: list[tuple] = []
    attrs = list(domains)
    for p in people:
        for a in attrs:
            for v in domains[a]:
                pool.append(("is" if sol[p][a] == v else "not", p, a, v))
    for a, b in itertools.permutations(attrs, 2):
        for va in domains[a]:
            owner = next(p for p in people if sol[p][a] == va)
            for vb in domains[b]:
                pool.append(("link" if sol[owner][b] == vb else "notlink", a, va, b, vb))
    return pool


def generate_grid(rng: random.Random, n_people: int, attrs: tuple[str, ...]):
    people = tuple(sorted(rng.sample(PEOPLE, n_people), key=PEOPLE.index))
    domains = {a: tuple(rng.sample(VALUES[a], n_people)) for a in attrs}
    perms = {a: rng.sample(domains[a], n_people) for a in attrs}
    sol = {p: {a: perms[a][i] for a in attrs} for i, p in enumerate(people)}
    pool = true_clue_pool(sol, people, domains)
    rng.shuffle(pool)
    chosen: list[tuple] = []
    for d in pool:
        chosen.append(d)
        if count_solutions(chosen, people, domains, limit=2) == 1:
            break
    for d in list(chosen):  # prune redundant clues
        rest = [x for x in chosen if x != d]
        if count_solutions(rest, people, domains, limit=2) == 1:
            chosen = rest
    rng.shuffle(chosen)
    return chosen, sol, people, domains


class LogicGridL4(Task):
    spec = "logic-grid:L4"
    answer_format = '{"<person>": ["<pet>", "<drink>"], ...} for every person'
    default_clues = 0  # determined by the generator (minimal unique set)
    attrs = ATTRS_L4
    n_people = 4

    def generate(self, rng: random.Random, n_clues: int) -> Generated:
        chosen, sol, people, domains = generate_grid(rng, self.n_people, self.attrs)
        clues = make_clues([(clue_text(d), d) for d in chosen])
        return Generated(clues, answer_of(sol, self.attrs), preamble_text(people, domains),
                         make_universe(people, domains))

    def solve(self, clues: tuple[Clue, ...], universe: tuple | None = None) -> Any:
        if universe is None:
            raise ValueError("logic grid solving needs the universe (people + values)")
        people, domains = unpack_universe(universe)
        sols = solve_all([c.data for c in clues], people, domains, limit=2)
        return answer_of(sols[0], self.attrs) if len(sols) == 1 else None

    def normalize(self, value: Any) -> Any:
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        out = {}
        for person, v in value.items():
            if isinstance(v, dict):
                v = [v[a] for a in self.attrs]
            vals = split_list(v)
            if len(vals) != len(self.attrs):
                raise ValueError("wrong number of attributes")
            out[norm_word(person)] = tuple(norm_word(x) for x in vals)
        return out


# ----------------------------------------------------------------------------- reference (CLAUDE.md §6.2)

REFERENCE_PEOPLE = ("Ana", "Ben", "Cara", "Dev")
REFERENCE_DOMAINS: Domains = {"pet": ("cat", "dog", "fish", "bird"), "drink": ("tea", "coffee", "juice", "water")}
REFERENCE_CLUE_DATA = (
    ("is", "Ana", "pet", "cat"),                # C1 Ana owns the cat.
    ("link", "pet", "dog", "drink", "coffee"),  # C2 The dog owner drinks coffee.
    ("link", "pet", "fish", "drink", "juice"),  # C3 The fish owner drinks juice.
    ("is", "Dev", "drink", "water"),            # C4 Dev drinks water.
    ("is", "Ana", "drink", "tea"),              # C5 Ana drinks tea.
    ("not", "Cara", "drink", "coffee"),         # C6 Cara does not drink coffee.
)
REFERENCE_ANSWER = {"Ana": ["cat", "tea"], "Ben": ["dog", "coffee"], "Cara": ["fish", "juice"],
                    "Dev": ["bird", "water"]}
REFERENCE_ASSIGNMENT = {1: ["C1", "C4"], 2: ["C2"], 3: ["C3", "C6"], 4: ["C5"]}  # worker k -> clue ids


def reference_instance() -> TaskInstance:
    clues = make_clues([(clue_text(d), d) for d in REFERENCE_CLUE_DATA])
    return TaskInstance(
        LogicGridL4.spec, -1, clues, REFERENCE_ANSWER, LogicGridL4.answer_format,
        preamble_text(REFERENCE_PEOPLE, REFERENCE_DOMAINS),
        make_universe(REFERENCE_PEOPLE, REFERENCE_DOMAINS),
    )
