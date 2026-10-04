"""Task B + C: run the (m, n) grid experiments and save results to CSV.

Main grid (project requirement):
  - Fix m = 2, increase n from 4 to 50 in steps of 10  (n = 4,10,20,30,40,50)
  - For each n, increase m from 2 to 14 in steps of 4  (m = 2,6,10,14)

Also runs a mixed-constraint grid (feasibility stress test) and the
hand-crafted special cases (infeasible / unbounded / degenerate / ...).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.experiment import M_VALUES, N_VALUES, run_grid, run_special_cases


def main() -> None:
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    print("Grid sizes (project requirement)")
    print(f"  n (variables) : {N_VALUES}")
    print(f"  m (constraints): {M_VALUES}")
    print(f"  seeds per cell : 10")
    print()

    print("[1/4] Main grid — random nonnegative <= LPs (always feasible/bounded)")
    df_main = run_grid(N_VALUES, M_VALUES, n_seeds=10, generator="random_feasible", use_scipy=True)
    p1 = out_dir / "grid_main.csv"
    df_main.to_csv(p1, index=False)
    print(f"  {len(df_main)} solves -> {p1}")
    print(f"  status counts: {df_main['status'].value_counts().to_dict()}")
    print()

    print("[2/4] Mixed-constraint grid, well-posed mode (x_ref feasible; signed coeffs)")
    df_mixed_wp = run_grid(N_VALUES, M_VALUES, n_seeds=5, generator="random_mixed",
                           use_scipy=True, mixed_mode="wellposed")
    p2 = out_dir / "grid_mixed_wellposed.csv"
    df_mixed_wp.to_csv(p2, index=False)
    print(f"  {len(df_mixed_wp)} solves -> {p2}")
    print(f"  status counts: {df_mixed_wp['status'].value_counts().to_dict()}")
    print()

    print("[3/4] Mixed-constraint grid, stress mode (infeasible / unbounded phenomena)")
    df_mixed_st = run_grid(N_VALUES, M_VALUES, n_seeds=5, generator="random_mixed",
                           use_scipy=True, mixed_mode="stress")
    p3 = out_dir / "grid_mixed_stress.csv"
    df_mixed_st.to_csv(p3, index=False)
    print(f"  {len(df_mixed_st)} solves -> {p3}")
    print(f"  status counts: {df_mixed_st['status'].value_counts().to_dict()}")
    print()

    print("[4/4] Special cases (lecture, infeasible, unbounded, degenerate, ...)")
    df_sp = run_special_cases(use_scipy=True)
    p4 = out_dir / "special_cases.csv"
    df_sp.to_csv(p4, index=False)
    print(f"  {len(df_sp)} solves -> {p4}")
    print(df_sp[["instance", "status", "objective", "pivots", "degenerate", "multiple_optima", "sparsity"]].to_string(index=False))
    print()

    print("Done. Next: python analysis/analyze_results.py")


if __name__ == "__main__":
    main()
