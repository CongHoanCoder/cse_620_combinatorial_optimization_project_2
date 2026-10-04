"""LP problem generators and special-case instances for the Simplex project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LPInstance:
    """A single LP instance with metadata."""

    name: str
    c: np.ndarray
    sense: str = "max"  # 'max' or 'min'
    A_ub: Optional[np.ndarray] = None
    b_ub: Optional[np.ndarray] = None
    A_eq: Optional[np.ndarray] = None
    b_eq: Optional[np.ndarray] = None
    description: str = ""
    expected_status: str = "optimal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(np.asarray(self.c).ravel().size)

    @property
    def m(self) -> int:
        total = 0
        if self.A_ub is not None:
            total += np.atleast_2d(self.A_ub).shape[0]
        if self.A_eq is not None:
            total += np.atleast_2d(self.A_eq).shape[0]
        return total

    @property
    def n_ub(self) -> int:
        return 0 if self.A_ub is None else int(np.atleast_2d(self.A_ub).shape[0])

    @property
    def n_eq(self) -> int:
        return 0 if self.A_eq is None else int(np.atleast_2d(self.A_eq).shape[0])


def lecture_example() -> LPInstance:
    """Representative Simplex lecture instance with n = 4 variables, m = 2 constraints.

        maximize  z = 3x1 + 5x2 + 4x3 + 2x4
        subject to   x1 + 2x2 +  x3 +  x4 <= 12
                     2x1 +  x2 + 3x3 + 2x4 <= 18
                     x1, x2, x3, x4 >= 0

    Replace this function with your exact lecture LP if it differs; every
    runner in this project only reads the ``LPInstance`` fields.
    """
    c = np.array([3.0, 5.0, 4.0, 2.0])
    A_ub = np.array(
        [
            [1.0, 2.0, 1.0, 1.0],
            [2.0, 1.0, 3.0, 2.0],
        ]
    )
    b_ub = np.array([12.0, 18.0])
    return LPInstance(
        name="lecture_example_n4_m2",
        c=c,
        sense="max",
        A_ub=A_ub,
        b_ub=b_ub,
        description="Lecture-style LP, n=4, m=2 (swap in your course example if different).",
        expected_status="optimal",
        metadata={"group": "task_A"},
    )


def random_feasible_lp(
    m: int,
    n: int,
    seed: int = 0,
    sparsity_ref: float = 0.0,
    margin: float = 0.5,
) -> LPInstance:
    """Random nonnegative LP that is always feasible (x = 0 works).

        maximize  c^T x
        subject to A x <= b,  x >= 0

    with A_ij ~ U(0,1), c_j ~ U(0,1), and b = A @ x_ref where x_ref is a
    nonnegative reference point (optionally sparse).  With nonnegative A and c
    the feasible region is bounded (the recession cone is trivial when no
    column of A is identically zero), so these instances are always optimal.
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.0, 1.0, size=(m, n))
    c = rng.uniform(0.0, 1.0, size=n)
    x_ref = rng.uniform(0.1, 1.0, size=n)
    if sparsity_ref > 0:
        mask = rng.random(n) < sparsity_ref
        x_ref[mask] = 0.0
        # Ensure at least one nonzero so the instance is not trivially x=0-only.
        if np.all(x_ref == 0):
            x_ref[rng.integers(0, n)] = 1.0
    b = A @ x_ref + margin * rng.uniform(0.0, 0.5, size=m)
    return LPInstance(
        name=f"random_feasible_m{m}_n{n}_s{seed}",
        c=c,
        sense="max",
        A_ub=A,
        b_ub=b,
        description="Random nonnegative <= LP (feasible by construction).",
        expected_status="optimal",
        metadata={"group": "grid", "m": m, "n": n, "seed": seed, "generator": "random_feasible"},
    )


def random_mixed_lp(
    m: int,
    n: int,
    seed: int = 0,
    ge_fraction: float = 0.3,
    mode: str = "wellposed",
) -> LPInstance:
    """Random LP with a mix of <= and >= constraints (signed coefficients).

    mode='wellposed':
        x_ref >= 0 is strictly feasible by construction
        (<= rows: b = A x_ref + margin;  >= rows: b = A x_ref - margin).
        The LP is feasible; with signed c it may still be unbounded.

    mode='stress':
        b signs are random, so instances may be infeasible, unbounded, or
        optimal — used to study solver behavior when problems have no
        solution or no finite optimum.

    >= rows are encoded as negated <= rows so the instance stays in
    scipy-style A_ub / b_ub form.
    """
    if mode not in ("wellposed", "stress"):
        raise ValueError("mode must be 'wellposed' or 'stress'")

    rng = np.random.default_rng(20_000 + seed + 3 * m + n + (0 if mode == "wellposed" else 500))
    A = rng.normal(0.0, 1.0, size=(m, n))
    c = rng.normal(0.0, 1.0, size=n)

    n_ge = int(round(ge_fraction * m))
    idx = rng.permutation(m)
    ge_mask = np.zeros(m, dtype=bool)
    ge_mask[idx[:n_ge]] = True

    if mode == "wellposed":
        x_ref = rng.uniform(0.2, 1.5, size=n)
        margin = rng.uniform(0.05, 0.4, size=m)
        b = np.empty(m)
        proj = A @ x_ref
        for i in range(m):
            b[i] = proj[i] - margin[i] if ge_mask[i] else proj[i] + margin[i]
    else:
        b = rng.uniform(0.1, 2.0, size=m) * rng.choice([-1.0, 1.0], size=m)

    A_rows = A.copy()
    b_rows = b.copy()
    for i in range(m):
        if ge_mask[i]:
            A_rows[i] = -A_rows[i]
            b_rows[i] = -b_rows[i]

    return LPInstance(
        name=f"random_mixed_{mode}_m{m}_n{n}_s{seed}",
        c=c,
        sense="max",
        A_ub=A_rows,
        b_ub=b_rows,
        description=f"Random LP with mixed <= / >= rows (mode={mode}).",
        expected_status="unknown",
        metadata={
            "group": "grid_mixed",
            "m": m,
            "n": n,
            "seed": seed,
            "generator": "random_mixed",
            "mode": mode,
            "n_ge": int(ge_mask.sum()),
        },
    )


def infeasible_example() -> LPInstance:
    """Classic infeasible pair: x1 + x2 <= 4 and x1 + x2 >= 6.

        maximize  x1 + x2
        subject to   x1 + x2 <= 4
                     x1 + x2 >= 6
                     x >= 0

    The two half-planes are parallel and disjoint, so no x >= 0 can satisfy
    both.  Phase I ends with a positive artificial sum -> infeasible.
    """
    c = np.array([1.0, 1.0])
    A_ub = np.array([[1.0, 1.0], [-1.0, -1.0]])
    b_ub = np.array([4.0, -6.0])  # second row: -x1-x2 <= -6  <=>  x1+x2 >= 6
    return LPInstance(
        name="infeasible_parallel_constraints",
        c=c,
        sense="max",
        A_ub=A_ub,
        b_ub=b_ub,
        description="Disjoint parallel constraints -> empty feasible region.",
        expected_status="infeasible",
        metadata={"group": "special"},
    )


def unbounded_example() -> LPInstance:
    """Unbounded LP: the objective ray x1 -> infinity stays feasible.

        maximize  x1 + 0.5*x2
        subject to   x1 -  x2 <= 2
                     -x1 + x2 <= 2   (redundant, keeps region closed on one side)
                     x >= 0

    Along the ray (t, 0) with t -> infinity, x1 - x2 = t <= 2 fails...
    Use instead:
        maximize  x1
        subject to   x1 - x2 <= 2
                     -x2 <= 5        (keeps x2 bounded below is automatic)
    The direction d = (1, 0) satisfies A d <= 0?  Row1: 1 <= 0 false.
    Correct recession condition: d >= 0, A d <= 0, c'd > 0.
        row1: 1*d1 - 1*d2 <= 0
        pick d = (1, 1): row1: 0 <= 0 OK.  c'd = 1 > 0 if c = (1, 0).
    So: max x1  s.t.  x1 - x2 <= 2, x >= 0  is unbounded (x1 = 2 + x2, x2 -> inf,
    x1 -> inf).  We keep a second constraint -x1 + x2 <= 2 as well, which is
    still unbounded via d=(1,1).
    """
    c = np.array([1.0, 0.0])
    A_ub = np.array([[1.0, -1.0], [-1.0, 1.0]])
    b_ub = np.array([2.0, 2.0])
    return LPInstance(
        name="unbounded_ray_d11",
        c=c,
        sense="max",
        A_ub=A_ub,
        b_ub=b_ub,
        description="Recession direction d=(1,1) keeps c'd = 1 > 0 -> unbounded.",
        expected_status="unbounded",
        metadata={"group": "special"},
    )


def degenerate_example() -> LPInstance:
    """Degenerate vertex: three constraints active at a 2-variable optimum.

        maximize  3x1 + 9x2
        subject to   x1 + 4x2 <= 8
                     x1 + 2x2 <= 4
                         x2 <= 2
                     x >= 0

    Optimum is at (0, 2): all three constraints are active, so more than n
    constraints meet at the vertex -> at least one basic variable equals 0
    (degeneracy).  Simplex may revisit the same vertex.
    """
    c = np.array([3.0, 9.0])
    A_ub = np.array([[1.0, 4.0], [1.0, 2.0], [0.0, 1.0]])
    b_ub = np.array([8.0, 4.0, 2.0])
    return LPInstance(
        name="degenerate_vertex_3active",
        c=c,
        sense="max",
        A_ub=A_ub,
        b_ub=b_ub,
        description="Optimal vertex (0,2) has 3 active constraints for n=2 -> degenerate.",
        expected_status="optimal",
        metadata={"group": "special", "degenerate": True},
    )


def multiple_optima_example() -> LPInstance:
    """Alternative optima: objective parallel to an active constraint face.

        maximize  x1 + x2
        subject to   x1 + x2 <= 4
                     x1 <= 4
                     x2 <= 4
                     x >= 0

    Every point on the segment x1 + x2 = 4 (0 <= x1 <= 4) is optimal at
    z* = 4, so the optimum is not unique.  The final reduced cost of a
    nonbasic variable along the face is ~0.
    """
    c = np.array([1.0, 1.0])
    A_ub = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    b_ub = np.array([4.0, 4.0, 4.0])
    return LPInstance(
        name="multiple_optima_face",
        c=c,
        sense="max",
        A_ub=A_ub,
        b_ub=b_ub,
        description="Optimal face x1+x2=4 -> infinitely many optima.",
        expected_status="optimal",
        metadata={"group": "special", "multiple_optima": True},
    )


def sparse_optimum_example(m: int = 4, n: int = 20, seed: int = 7) -> LPInstance:
    """LP whose optimal vertex is sparse: most decision variables are zero.

    A budget-style packing problem: each variable j consumes a_ij resources
    and earns profit c_j.  Only a few items are worth selecting, so the
    optimal basic feasible solution has at most m nonzeros among the n
    original variables — exactly the sparsity phenomenon from the project
    requirements.
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.2, 1.0, size=(m, n))
    c = rng.uniform(0.0, 1.0, size=n)
    # Make only the first few variables clearly profitable.
    k = max(2, m // 2)
    c[k:] *= 0.05
    # Tight-ish budgets.
    b = A[:, :k].sum(axis=1) * rng.uniform(0.6, 1.0, size=m)
    return LPInstance(
        name=f"sparse_optimum_m{m}_n{n}_s{seed}",
        c=c,
        sense="max",
        A_ub=A,
        b_ub=b,
        description="Budget packing LP; optimal BFS has at most m nonzeros.",
        expected_status="optimal",
        metadata={"group": "special", "generator": "sparse_optimum", "n_profitable": k},
    )


def unbounded_after_removing_constraint() -> List[LPInstance]:
    """Bounded LP and the same LP with the bounding constraint removed.

    Base (bounded):
        maximize  x1 + x2
        subject to   x1 + x2 <= 4      <-- bounds the objective
                     x2 <= 3
    Remove the first constraint and only x2 <= 3 remains; the ray
    (t, 3) with t -> infinity is feasible with objective -> infinity,
    so the LP is unbounded.
    """
    base = LPInstance(
        name="bounded_base_with_constraint",
        c=np.array([1.0, 1.0]),
        sense="max",
        A_ub=np.array([[1.0, 1.0], [0.0, 1.0]]),
        b_ub=np.array([4.0, 3.0]),
        description="Bounded LP: x1+x2 <= 4 caps the objective at z*=4.",
        expected_status="optimal",
        metadata={"group": "special", "variant": "with_constraint"},
    )
    removed = LPInstance(
        name="unbounded_after_removing_constraint",
        c=np.array([1.0, 1.0]),
        sense="max",
        A_ub=np.array([[0.0, 1.0]]),
        b_ub=np.array([3.0]),
        description="Same LP with x1+x2<=4 removed -> only x2<=3 remains -> unbounded.",
        expected_status="unbounded",
        metadata={"group": "special", "variant": "constraint_removed"},
    )
    return [base, removed]


def make_special_instances() -> List[LPInstance]:
    """All hand-crafted special cases used in the experiments."""
    insts: List[LPInstance] = [
        lecture_example(),
        infeasible_example(),
        unbounded_example(),
        degenerate_example(),
        multiple_optima_example(),
        sparse_optimum_example(m=4, n=20, seed=7),
        sparse_optimum_example(m=6, n=50, seed=11),
    ]
    insts.extend(unbounded_after_removing_constraint())
    return insts
