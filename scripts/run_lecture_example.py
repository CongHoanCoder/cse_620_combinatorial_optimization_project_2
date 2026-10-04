"""Task A: solve the lecture LP instance (n=4, m=2) with the custom Simplex
and cross-check against SciPy/HiGHS."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.experiment import run_lecture_case
from src.problems import lecture_example
from src.simplex import simplex_solve


def main() -> None:
    inst = lecture_example()
    print("=" * 72)
    print("TASK A — Lecture example LP")
    print("=" * 72)
    print(inst.description)
    print()
    print(f"  maximize  z = c^T x ,  c = {inst.c.tolist()}")
    print(f"  subject to A x <= b :")
    for i in range(inst.A_ub.shape[0]):
        coeffs = " + ".join(f"{inst.A_ub[i, j]:g}x{j + 1}" for j in range(inst.n))
        print(f"              {coeffs} <= {inst.b_ub[i]:g}")
    print(f"              x >= 0")
    print(f"  n = {inst.n}, m = {inst.m}")
    print()

    res = simplex_solve(
        inst.c, A_ub=inst.A_ub, b_ub=inst.b_ub, sense=inst.sense
    )
    print("Custom two-phase Simplex")
    print(f"  status           : {res.status}")
    print(f"  objective z*     : {res.objective:.6f}")
    print(f"  x*               : {np.round(res.x, 6).tolist()}")
    print(f"  pivots (total)   : {res.pivots}  (Phase I: {res.phase1_pivots}, Phase II: {res.phase2_pivots})")
    print(f"  solve time       : {res.solve_time * 1e3:.3f} ms")
    print(f"  degenerate       : {res.degenerate}")
    print(f"  multiple optima  : {res.multiple_optima}")
    print(f"  n_nonzero / n    : {res.n_nonzero}/{inst.n}  (sparsity {res.sparsity:.1%})")
    print(f"  message          : {res.message}")
    print()

    # Verify constraint satisfaction and objective.
    Ax = inst.A_ub @ res.x
    slack = inst.b_ub - Ax
    print("Constraint check (A x <= b):")
    for i in range(inst.A_ub.shape[0]):
        ok = "OK" if slack[i] >= -1e-7 else "VIOLATED"
        print(f"  row {i + 1}: A x = {Ax[i]:.6f}  <=  b = {inst.b_ub[i]:.6f}  slack = {slack[i]:.6f}  [{ok}]")
    print(f"  z = c x = {inst.c @ res.x:.6f}")
    print()

    from src.simplex import linprog_cross_check

    sc = linprog_cross_check(inst.c, A_ub=inst.A_ub, b_ub=inst.b_ub, sense=inst.sense)
    print("SciPy / HiGHS cross-check")
    print(f"  status           : {sc['scipy_status']}")
    print(f"  objective        : {sc['scipy_objective']:.6f}")
    print(f"  iterations (nit) : {sc['scipy_nit']}")
    print(f"  time             : {sc['scipy_time'] * 1e3:.3f} ms")
    match = abs(res.objective - sc["scipy_objective"]) <= 1e-6 * max(1.0, abs(res.objective))
    print(f"  objective match  : {match}")
    print()

    df = run_lecture_case(use_scipy=True)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "lecture_example.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    summary = df.iloc[0].to_dict()
    (out_dir / "lecture_example.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved: {out_dir / 'lecture_example.json'}")


if __name__ == "__main__":
    main()
