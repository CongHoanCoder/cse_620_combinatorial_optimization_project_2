"""Experiment runners: solve LP instances and record time, pivots, and solution stats."""

from __future__ import annotations

import time
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .problems import LPInstance, random_feasible_lp, random_mixed_lp
from .simplex import linprog_cross_check, simplex_solve

# Project requirement grid
N_VALUES = [4, 10, 20, 30, 40, 50]  # variables: 4 -> 10 -> 50, step 10
M_VALUES = [2, 6, 10, 14]  # constraints: 2 -> 6 -> 10 -> 14, step 4


def solve_instance(inst: LPInstance, use_scipy: bool = True) -> dict:
    """Solve one LPInstance with the custom Simplex (and optionally SciPy)."""
    res = simplex_solve(
        inst.c,
        A_ub=inst.A_ub,
        b_ub=inst.b_ub,
        A_eq=inst.A_eq,
        b_eq=inst.b_eq,
        sense=inst.sense,
    )

    row = {
        "instance": inst.name,
        "group": inst.metadata.get("group", ""),
        "generator": inst.metadata.get("generator", ""),
        "m": inst.m,
        "n": inst.n,
        "n_ub": inst.n_ub,
        "n_eq": inst.n_eq,
        "seed": inst.metadata.get("seed", -1),
        "status": res.status,
        "expected_status": inst.expected_status,
        "objective": res.objective,
        "pivots": res.pivots,
        "phase1_pivots": res.phase1_pivots,
        "phase2_pivots": res.phase2_pivots,
        "simplex_time": res.solve_time,
        "degenerate": res.degenerate,
        "multiple_optima": res.multiple_optima,
        "n_nonzero": res.n_nonzero,
        "sparsity": res.sparsity,
        "message": res.message,
    }

    if use_scipy:
        # SciPy cannot read >= blocks from A_ub; re-encode if needed via A_eq.
        A_ub, b_ub, A_eq, b_eq = inst.A_ub, inst.b_ub, inst.A_eq, inst.b_eq
        sc = linprog_cross_check(
            inst.c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            sense=inst.sense,
        )
        row["scipy_status"] = sc["scipy_status"]
        row["scipy_objective"] = sc["scipy_objective"]
        row["scipy_time"] = sc["scipy_time"]
        row["scipy_nit"] = sc["scipy_nit"]
        row["objective_match"] = bool(
            (res.status == "optimal" and sc["scipy_status"] == "optimal" and
             abs(res.objective - sc["scipy_objective"]) <= 1e-6 * max(1.0, abs(res.objective)))
            or (res.status == sc["scipy_status"])
        )
    else:
        row["scipy_status"] = ""
        row["scipy_objective"] = np.nan
        row["scipy_time"] = np.nan
        row["scipy_nit"] = -1
        row["objective_match"] = True

    return row


def run_grid(
    n_values: Optional[Iterable[int]] = None,
    m_values: Optional[Iterable[int]] = None,
    n_seeds: int = 5,
    generator: str = "random_feasible",
    use_scipy: bool = True,
    mixed_mode: str = "wellposed",
) -> pd.DataFrame:
    """Solve the (m, n) grid required by the project.

    generator:
      - 'random_feasible': nonnegative <= LPs, always optimal (main scaling study)
      - 'random_mixed': mixed <=/>= rows; mixed_mode in {'wellposed','stress'}
    """
    n_values = list(n_values if n_values is not None else N_VALUES)
    m_values = list(m_values if m_values is not None else M_VALUES)

    rows: List[dict] = []
    for n in n_values:
        for m in m_values:
            for seed in range(n_seeds):
                if generator == "random_feasible":
                    inst = random_feasible_lp(m=m, n=n, seed=seed, sparsity_ref=0.2)
                elif generator == "random_mixed":
                    inst = random_mixed_lp(m=m, n=n, seed=seed, mode=mixed_mode)
                else:
                    raise ValueError(f"unknown generator {generator!r}")
                rows.append(solve_instance(inst, use_scipy=use_scipy))
    return pd.DataFrame(rows)


def run_special_cases(use_scipy: bool = True) -> pd.DataFrame:
    """Solve all hand-crafted special-case instances."""
    from .problems import make_special_instances

    rows = [solve_instance(inst, use_scipy=use_scipy) for inst in make_special_instances()]
    return pd.DataFrame(rows)


def run_lecture_case(use_scipy: bool = True) -> pd.DataFrame:
    """Task A: solve the n=4, m=2 lecture example."""
    from .problems import lecture_example

    return pd.DataFrame([solve_instance(lecture_example(), use_scipy=use_scipy)])


def time_simplex_only(inst: LPInstance, repeats: int = 3) -> float:
    """Median wall-clock time of the custom Simplex on one instance (seconds)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        simplex_solve(
            inst.c,
            A_ub=inst.A_ub,
            b_ub=inst.b_ub,
            A_eq=inst.A_eq,
            b_eq=inst.b_eq,
            sense=inst.sense,
        )
        times.append(time.perf_counter() - t0)
    return float(np.median(times))
