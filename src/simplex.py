"""Two-phase tableau Simplex for linear programs with pivot counting.

Solves LPs of the form:

    maximize (or minimize)  c^T x
    subject to              A_ub x <= b_ub
                            A_eq x  = b_eq
                            x >= 0

The LP is converted to standard form (equalities with slacks, surplus
variables and, where needed, artificial variables).  Phase I maximizes
w = -sum(artificials) to find a basic feasible solution or prove
infeasibility; Phase II then optimizes the true objective from that basis.

Pivot rule: Dantzig (largest positive reduced cost).  If the iteration count
exceeds a safeguard, Bland's rule is used to guarantee finite termination
on degenerate problems.

Recorded diagnostics: total pivots (Phase I + Phase II), status
(optimal / infeasible / unbounded), objective value, optimal x, whether the
vertex is degenerate, whether alternative optima exist, and a sparsity
measure on the original variables.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

TOL = 1e-9
BLAND_SWITCH_PIVOTS = 1000


@dataclass
class SimplexResult:
    """Outcome of a Simplex solve."""

    status: str  # 'optimal' | 'infeasible' | 'unbounded'
    x: np.ndarray  # primal solution for original variables (nan if not optimal)
    objective: float  # optimal objective value (nan unless status == 'optimal')
    pivots: int  # total pivots (Phase I + Phase II)
    phase1_pivots: int
    phase2_pivots: int
    solve_time: float  # wall-clock seconds for the Simplex run
    degenerate: bool = False
    multiple_optima: bool = False
    n_nonzero: int = 0  # nonzeros in optimal original x
    sparsity: float = 0.0  # 1 - n_nonzero / n
    basis: List[int] = field(default_factory=list)
    message: str = ""


def _build_standard_form(
    c: np.ndarray,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[np.ndarray],
    b_eq: Optional[np.ndarray],
):
    """Assemble equality-constrained standard form with column bookkeeping.

    Column order is blocked:
        [ x (n) | slacks for <= (n_ub) | surpluses for >= (n_ge) | artificials (n_ge+n_eq) ]

    Returns A_std, b_vec, c_std, col_names, basis_init, artificial_cols,
    n, n_total, m, c_orig_sign.
    """
    c = np.asarray(c, dtype=float).ravel()
    n = c.size

    rows: List[Tuple[np.ndarray, float, str]] = []

    def _consume(A_block, b_vec, s):
        if A_block is None or b_vec is None:
            return
        A_block = np.atleast_2d(np.asarray(A_block, dtype=float))
        b_vec = np.asarray(b_vec, dtype=float).ravel()
        if A_block.shape[0] != b_vec.size:
            raise ValueError("A and b size mismatch")
        if A_block.shape[1] != n:
            raise ValueError("A column count != len(c)")
        for i in range(b_vec.size):
            rows.append((A_block[i].copy(), float(b_vec[i]), s))

    _consume(A_ub, b_ub, "<=")
    _consume(A_eq, b_eq, "=")
    if not rows:
        raise ValueError("no constraints provided")

    A_rows: List[np.ndarray] = []
    b_list: List[float] = []
    sense_list: List[str] = []
    for A_row, b_val, s in rows:
        if b_val < 0:
            A_rows.append(-A_row)
            b_list.append(-b_val)
            if s == "<=":
                sense_list.append(">=")
            elif s == ">=":
                sense_list.append("<=")
            else:
                sense_list.append("=")
        else:
            A_rows.append(A_row)
            b_list.append(b_val)
            sense_list.append(s)

    A_mat = np.vstack(A_rows)
    b_vec = np.array(b_list, dtype=float)
    sense_arr = np.array(sense_list, dtype=object)
    m = A_mat.shape[0]

    n_ub = int(np.sum(sense_arr == "<="))
    n_ge = int(np.sum(sense_arr == ">="))
    n_eq = int(np.sum(sense_arr == "="))
    n_art = n_ge + n_eq

    # Blocked column indices.
    slack_start = n
    surplus_start = slack_start + n_ub
    art_start = surplus_start + n_ge
    n_total = art_start + n_art

    col_names: List[str] = [f"x{j + 1}" for j in range(n)]
    col_names += [f"s{i + 1}" for i in range(n_ub)]
    col_names += [f"u{i + 1}" for i in range(n_ge)]
    col_names += [f"a{i + 1}" for i in range(n_art)]

    A_std = np.zeros((m, n_total))
    A_std[:, :n] = A_mat

    basis_init: List[int] = [0] * m
    si = gi = ai = 0
    for i, s in enumerate(sense_arr):
        if s == "<=":
            A_std[i, slack_start + si] = 1.0
            basis_init[i] = slack_start + si
            si += 1
        elif s == ">=":
            A_std[i, surplus_start + gi] = -1.0  # surplus u: a x - u = b
            A_std[i, art_start + ai] = 1.0  # artificial a: for Phase I
            basis_init[i] = art_start + ai  # artificial is initial basic var
            gi += 1
            ai += 1
        else:  # '='
            A_std[i, art_start + ai] = 1.0
            basis_init[i] = art_start + ai
            ai += 1

    artificial_cols = set(range(art_start, n_total)) if n_art else set()
    c_std = np.zeros(n_total)
    c_std[:n] = c
    c_orig_sign = 1.0  # placeholder; sense applied by caller via c_std
    return A_std, b_vec, c_std, col_names, basis_init, artificial_cols, n, n_total, m, c_orig_sign


def _make_tableau(c_vec: np.ndarray, A: np.ndarray, b: np.ndarray, bas: Sequence[int]) -> np.ndarray:
    """Build the simplex tableau.

    Constraint rows: A x = b expressed in the current basis.
    Objective row (last):  [ -reduced_costs | z ]  with
    reduced_j = c_j - c_B B^{-1} A_j  (positive means improving for max),
    so the row stores  T[-1, j] = -reduced_j  and  T[-1, -1] = z.

    With this convention Gaussian elimination on the full tableau keeps
    T[-1, -1] equal to the true objective value of the current basis.
    """
    m = A.shape[0]
    T = np.zeros((m + 1, A.shape[1] + 1))
    T[:-1, :-1] = A
    T[:-1, -1] = b
    c_B = c_vec[list(bas)]
    T[-1, :-1] = c_B @ A - c_vec  # = -reduced costs
    T[-1, -1] = float(c_B @ b)  # z
    return T


def _run_phase(
    T: np.ndarray,
    bas: List[int],
    c_vec: np.ndarray,
    max_pivots: int,
) -> Tuple[str, int, List[int]]:
    """Optimize the tableau in place (maximization).  Returns (status, pivots, basis).

    Objective row convention: T[-1, j] = -reduced_j, T[-1, -1] = z.
    Optimality (max): all T[-1, :-1] >= -tol.
    Entering variable: most negative T[-1, j].
    Unbounded: entering column has no positive entry.
    """
    m_r = T.shape[0] - 1
    pivots = 0
    use_bland = False
    _ = c_vec

    while True:
        obj_row = T[-1, :-1]
        # Optimal when all stored (-reduced) entries are >= -tol.
        if np.all(obj_row >= -TOL):
            return "optimal", pivots, bas
        if not use_bland and pivots >= BLAND_SWITCH_PIVOTS:
            use_bland = True

        if use_bland:
            # Bland: smallest index j with -reduced_j < -tol  (i.e. reduced > tol).
            cand = np.nonzero(obj_row < -TOL)[0]
            if cand.size == 0:
                return "optimal", pivots, bas
            e = int(cand[0])
        else:
            e = int(np.argmin(obj_row))
            if obj_row[e] >= -TOL:
                return "optimal", pivots, bas

        col = T[:-1, e]
        pos = np.nonzero(col > TOL)[0]
        if pos.size == 0:
            return "unbounded", pivots, bas

        ratios = T[:-1, -1][pos] / col[pos]
        if use_bland:
            min_ratio = float(np.min(ratios))
            tied = pos[np.abs(ratios - min_ratio) <= 1e-12]
            r = int(min(tied, key=lambda i: bas[i]))
        else:
            r = int(pos[int(np.argmin(ratios))])

        piv_val = T[r, e]
        T[r, :] /= piv_val
        for i in range(m_r + 1):
            if i != r:
                T[i, :] -= T[i, e] * T[r, :]
        bas[r] = e
        pivots += 1
        if pivots > max_pivots:
            return "unbounded", pivots, bas


def simplex_solve(
    c: np.ndarray,
    A_ub: Optional[np.ndarray] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[np.ndarray] = None,
    b_eq: Optional[np.ndarray] = None,
    sense: str = "max",
) -> SimplexResult:
    """Solve an LP with the two-phase tableau Simplex.

    Parameters mirror ``scipy.optimize.linprog`` (A_ub x <= b_ub, A_eq x = b_eq,
    x >= 0) with an extra ``sense`` flag in {'max', 'min'}.
    """
    t0 = time.perf_counter()

    if sense not in ("max", "min"):
        raise ValueError("sense must be 'max' or 'min'")

    (A_std, b_vec, c_std_raw, col_names, basis, artificial_cols, n, n_total, m, _sign) = (
        _build_standard_form(c, A_ub, b_ub, A_eq, b_eq)
    )
    c_orig_sign = 1.0 if sense == "max" else -1.0
    c_std = c_std_raw * c_orig_sign  # maximize internally
    needs_phase1 = len(artificial_cols) > 0

    max_pivots = 20_000 + 50 * (m + n_total)
    pivots_total = 0
    phase1_pivots = 0
    phase2_pivots = 0
    msg = ""

    def _nan(status: str, pivots: int = 0, p1: int = 0, p2: int = 0, msg: str = "") -> SimplexResult:
        return SimplexResult(
            status=status,
            x=np.full(n, np.nan),
            objective=float("nan"),
            pivots=pivots,
            phase1_pivots=p1,
            phase2_pivots=p2,
            solve_time=time.perf_counter() - t0,
            basis=list(basis),
            message=msg,
        )

    # ---- Phase I -----------------------------------------------------------
    if needs_phase1:
        c_phase1 = np.zeros(n_total)
        for j in artificial_cols:
            c_phase1[j] = -1.0  # maximize w = -sum(artificials)
        T = _make_tableau(c_phase1, A_std, b_vec, basis)
        st, phase1_pivots, basis = _run_phase(T, basis, c_phase1, max_pivots)
        pivots_total += phase1_pivots
        w_opt = float(T[-1, -1])  # value of w = -sum(artificials)

        if st == "unbounded" or w_opt < -TOL:
            return _nan(
                "infeasible",
                pivots_total,
                phase1_pivots,
                0,
                msg=f"Phase I: status={st}, w={w_opt:.3e} => no feasible solution",
            )

        # Pivot artificials out of the basis; drop redundant rows.
        drop_rows = []
        for r in range(m):
            if basis[r] not in artificial_cols:
                continue
            e_out = None
            for j in range(n_total):
                if j not in artificial_cols and abs(T[r, j]) > TOL:
                    e_out = j
                    break
            if e_out is None:
                drop_rows.append(r)
                continue
            piv_val = T[r, e_out]
            T[r, :] /= piv_val
            for i in range(m + 1):
                if i != r:
                    T[i, :] -= T[i, e_out] * T[r, :]
            basis[r] = e_out
            phase1_pivots += 1
            pivots_total += 1

        if drop_rows:
            keep_row = [i for i in range(m) if i not in drop_rows]
            T = T[np.ix_(keep_row, list(range(n_total + 1)))]
            basis = [basis[i] for i in keep_row]
            m = len(keep_row)

        # Delete artificial columns and rebuild Phase II data.
        keep_cols = [j for j in range(n_total) if j not in artificial_cols]
        old_to_new = {old: new for new, old in enumerate(keep_cols)}
        A_std = T[:-1, :-1][:, keep_cols].copy()
        b_vec = T[:-1, -1].copy()
        basis = [old_to_new[j] for j in basis]
        c_std = c_std[keep_cols].copy()
        col_names = [col_names[j] for j in keep_cols]
        n_total = len(keep_cols)

    # ---- Phase II ----------------------------------------------------------
    T = _make_tableau(c_std, A_std, b_vec, basis)
    st2, phase2_pivots, basis = _run_phase(T, basis, c_std, max_pivots)
    pivots_total += phase2_pivots

    if st2 == "unbounded":
        return _nan(
            "unbounded",
            pivots_total,
            phase1_pivots,
            phase2_pivots,
            msg="Phase II: no finite optimum (unbounded ray exists)",
        )

    # ---- Extract solution --------------------------------------------------
    x_std = np.zeros(n_total)
    for r, j in enumerate(basis):
        x_std[j] = T[r, -1]
    x = x_std[:n]
    z_work = float(T[-1, -1])  # objective value of the current basis (internal max sense)
    objective = c_orig_sign * z_work

    x[np.abs(x) < 1e-12] = 0.0
    n_nonzero = int(np.count_nonzero(np.abs(x) > 1e-9))
    sparsity = 1.0 - (n_nonzero / n if n else 0.0)

    # Degenerate vertex: at least one BASIC variable equals zero.
    basic_values = [T[r, -1] for r in range(len(basis))]
    degenerate = any(abs(v) < 1e-9 for v in basic_values)

    # Alternative optima: a nonbasic variable with ~zero reduced cost.
    # Objective row stores -reduced_j, so reduced ~ 0  <=>  T[-1, j] ~ 0.
    obj_row = T[-1, :-1]
    nonbasic = set(range(n_total)) - set(basis)
    multiple_optima = any(abs(obj_row[j]) < 1e-7 for j in nonbasic)

    return SimplexResult(
        status="optimal",
        x=x,
        objective=objective,
        pivots=pivots_total,
        phase1_pivots=phase1_pivots,
        phase2_pivots=phase2_pivots,
        solve_time=time.perf_counter() - t0,
        degenerate=degenerate,
        multiple_optima=multiple_optima,
        n_nonzero=n_nonzero,
        sparsity=sparsity,
        basis=list(basis),
        message="optimal solution found",
    )


def linprog_cross_check(
    c: np.ndarray,
    A_ub: Optional[np.ndarray] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[np.ndarray] = None,
    b_eq: Optional[np.ndarray] = None,
    sense: str = "max",
) -> dict:
    """Cross-check the custom Simplex against ``scipy.optimize.linprog`` (HiGHS).

    SciPy always *minimizes*, so a maximization problem c^T x is solved as
    the minimization of (-c)^T x and the objective sign is flipped back.
    """
    from scipy.optimize import linprog

    c = np.asarray(c, dtype=float).ravel()
    c_work = -c if sense == "max" else c
    bounds = [(0, None)] * c.size
    t0 = time.perf_counter()
    res = linprog(
        c_work,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    elapsed = time.perf_counter() - t0
    status_map = {0: "optimal", 2: "infeasible", 3: "unbounded"}
    objective = float(res.fun) if res.fun is not None else float("nan")
    if sense == "max" and np.isfinite(objective):
        objective = -objective  # recover the maximization objective
    return {
        "scipy_status": status_map.get(res.status, str(res.status)),
        "scipy_objective": objective,
        "scipy_x": np.asarray(res.x, dtype=float) if res.x is not None else np.full(c.size, np.nan),
        "scipy_time": elapsed,
        "scipy_nit": int(getattr(res, "nit", -1)),
        "scipy_message": res.message,
    }
