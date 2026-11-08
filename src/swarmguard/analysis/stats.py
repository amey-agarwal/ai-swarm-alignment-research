"""Statistics (CLAUDE.md §15): Wilson intervals, PSR tables, exact contrasts, logistic regression.

Wilson CIs and Fisher's exact test are pure Python. Logistic regression uses statsmodels
(install the `analysis` extra).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence

Row = dict[str, Any]


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ----------------------------------------------------------------------------- conditional subsets


def conditional(rows: Iterable[Row], *, functioning: bool = True, comprehension: bool = True) -> list[Row]:
    """PSR is reported conditional on the swarm functioning and on comprehension
    (at least one agent registered the claim). Claim=none cells have nothing to register,
    so the comprehension condition does not apply to them."""
    out = []
    for r in rows:
        if functioning and not r["swarm_functioning"]:
            continue
        if comprehension and r["claim"] != "none" and not r["registered_agents"]:
            continue
        out.append(r)
    return out


def rate_table(rows: Iterable[Row], by: Sequence[str], outcome: str = "permission_seeking") -> list[Row]:
    """Per-cell k/n, rate and 95% Wilson CI. Rows whose outcome is None are skipped and counted."""
    cells: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        cells[tuple(r[k] for k in by)].append(r)
    out = []
    for key in sorted(cells, key=lambda t: tuple(str(x) for x in t)):
        vals = [r[outcome] for r in cells[key]]
        known = [bool(v) for v in vals if v is not None]
        k, n = sum(known), len(known)
        lo, hi = wilson_ci(k, n)
        out.append({**dict(zip(by, key)), "k": k, "n": n, "rate": (k / n) if n else None, "ci_low": lo,
                    "ci_high": hi, "n_missing": len(vals) - n})
    return out


def psr_table(rows: Iterable[Row], by: Sequence[str]) -> list[Row]:
    return rate_table(rows, by, "permission_seeking")


# ----------------------------------------------------------------------------- exact contrast


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for [[a, b], [c, d]]."""
    n1, n2, m1 = a + b, c + d, a + c
    n = n1 + n2

    def pmf(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, m1 - x) / math.comb(n, m1)

    lo, hi = max(0, m1 - n2), min(m1, n1)
    p_obs = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-7)))


def contrast(rows: Iterable[Row], factor: str, a: Any, b: Any, outcome: str = "permission_seeking") -> Row:
    """Compare the outcome rate between factor == a and factor == b (Fisher exact)."""
    rows = list(rows)

    def kn(level):
        vals = [bool(r[outcome]) for r in rows if r[factor] == level and r[outcome] is not None]
        return sum(vals), len(vals)

    ka, na = kn(a)
    kb, nb = kn(b)
    return {"factor": factor, "a": a, "b": b, "k_a": ka, "n_a": na, "k_b": kb, "n_b": nb,
            "rate_a": ka / na if na else None, "rate_b": kb / nb if nb else None,
            "p_fisher": fisher_exact(ka, na - ka, kb, nb - kb) if na and nb else None}


# ----------------------------------------------------------------------------- regression


DEFAULT_FORMULA = "permission_seeking ~ np.log(params_b) + n + C(policy) + C(claim) + C(mitigation)"


def logistic_regression(rows: Iterable[Row], formula: str = DEFAULT_FORMULA) -> Row:
    """Logit of PSR on log(parameters), n, policy, claim, mitigation (statsmodels).
    Terms with a single level in the data are dropped. Returns coefficients, SEs, p-values;
    on (quasi-)separation or a singular design returns {"error": ...} instead of guessing."""
    try:
        import numpy as np  # noqa: F401  (used inside the formula)
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError("install the analysis extra: pip install -e '.[analysis]'") from exc

    df = pd.DataFrame([r for r in rows if r.get("permission_seeking") is not None])
    if df.empty:
        return {"error": "no rows with judge-based permission_seeking"}
    df["permission_seeking"] = df["permission_seeking"].astype(int)
    lhs, rhs = formula.split("~")
    terms = [t.strip() for t in rhs.split("+")]

    def varies(term: str) -> bool:
        col = term[term.find("(") + 1:term.rfind(")")] if "(" in term else term
        return col in df and df[col].nunique() > 1

    kept = [t for t in terms if varies(t)]
    f = f"{lhs.strip()} ~ {' + '.join(kept) if kept else '1'}"
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = smf.logit(f, data=df).fit(disp=False, maxiter=200)
        except Exception as exc:  # noqa: BLE001 - separation / singular matrix
            return {"formula": f, "error": f"{type(exc).__name__}: {exc}"}
    bad = sorted({type(w.message).__name__ for w in caught
                  if type(w.message).__name__ in ("PerfectSeparationWarning", "ConvergenceWarning")})
    if bad or not res.mle_retvals.get("converged", False):
        # Coefficients are not identified: never report them as if they were.
        return {"formula": f, "error": f"not identified ({', '.join(bad) or 'did not converge'}); "
                                       f"report exact per-cell tests (contrast/fisher_exact) instead"}
    return {"formula": f, "n": int(res.nobs), "coef": res.params.to_dict(), "se": res.bse.to_dict(),
            "p": res.pvalues.to_dict(), "llf": float(res.llf), "converged": True}


def summarize(rows: Sequence[Row], by: Sequence[str],
              outcomes: Sequence[str] = ("permission_seeking", "success")) -> dict[str, list[Row]]:
    """PSR next to task accuracy (§6.6: always report accuracy and compliance alongside)."""
    return {o: rate_table(rows, by, o) for o in outcomes}


def mean_by(rows: Iterable[Row], by: Sequence[str], value: str, keep: Callable[[Row], bool] = lambda r: True
            ) -> list[Row]:
    cells: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if keep(r) and r.get(value) is not None:
            cells[tuple(r[k] for k in by)].append(float(r[value]))
    return [{**dict(zip(by, k)), "mean": sum(v) / len(v), "n": len(v)} for k, v in sorted(cells.items(), key=str)]
