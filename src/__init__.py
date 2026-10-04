"""Simplex project package: custom two-phase Simplex, LP generators, experiments."""

from .simplex import simplex_solve, SimplexResult, linprog_cross_check
from .problems import (
    LPInstance,
    lecture_example,
    random_feasible_lp,
    random_mixed_lp,
    infeasible_example,
    unbounded_example,
    degenerate_example,
    multiple_optima_example,
    sparse_optimum_example,
    unbounded_after_removing_constraint,
)

__all__ = [
    "simplex_solve",
    "SimplexResult",
    "linprog_cross_check",
    "LPInstance",
    "lecture_example",
    "random_feasible_lp",
    "random_mixed_lp",
    "infeasible_example",
    "unbounded_example",
    "degenerate_example",
    "multiple_optima_example",
    "sparse_optimum_example",
    "unbounded_after_removing_constraint",
]
