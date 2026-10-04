# CSE 620 — Project 2: Linear Programming with the Simplex Method

Implementation for the combinatorial optimization project: solve a series of
increasingly larger LP problems with the Simplex method, measure solution time
and pivot counts, and analyze how the number of variables `n` and constraints
`m` affect the outcome.

## Project mapping

| Requirement | Where it lives |
|---|---|
| **A.** Solve the lecture LP (`n=4`, `m=2`) | `src/problems.py::lecture_example`, `scripts/run_lecture_example.py`, notebook Part A |
| **B.** Scale `n` = 4→50 (step 10) and `m` = 2→14 (step 4) | `src/experiment.py::run_grid`, `scripts/run_grid_experiment.py`, notebook Part B |
| **C.** Record time + pivots vs problem size | `results/grid_main.csv`, `results/tables/summary_main.csv`, `results/plots/*` |
| **D.** Analyze results / phenomena | `analysis/analyze_results.py`, `results/ANALYSIS.md`, notebook Part D |
| Simplex implementation | `src/simplex.py` — custom two-phase tableau Simplex with pivot counting |
| SciPy cross-check | `scipy.optimize.linprog` (HiGHS) via `src/simplex.py::linprog_cross_check` |

## Setup

```bash
cd <this folder>
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
# Task A — lecture example (n=4, m=2)
.venv/bin/python scripts/run_lecture_example.py

# Tasks B + C — full (m, n) grids + special cases
.venv/bin/python scripts/run_grid_experiment.py

# Task D — tables, plots, and report-ready analysis
.venv/bin/python analysis/analyze_results.py
```

Interactive walkthrough: `simplex_project.ipynb`
(with Jupyter: `.venv/bin/jupyter notebook simplex_project.ipynb`).

## Problem sizes (project requirement)

- Variables `n`: **4, 10, 20, 30, 40, 50**
- Constraints `m`: **2, 6, 10, 14**
- 10 random seeds per `(m, n)` cell on the main grid → 240 solves
- Additional mixed-constraint grids (well-posed + stress) and 9 hand-crafted
  special cases (infeasible, unbounded, degenerate, multiple optima, sparse)

## Using your own lecture LP

Replace `lecture_example()` in `src/problems.py` with your course instance
(same `LPInstance` fields: `c`, `A_ub`, `b_ub`, `sense`). Every script and the
notebook read only those fields.

## Repository layout

```
report/
  main.tex            Overleaf-ready LaTeX report (8 pp main + appendix)
  main.pdf            Compiled report
  figures/            All report figures (upload these to Overleaf too)
src/
  simplex.py       Two-phase tableau Simplex (pivot counting, Dantzig + Bland safeguard)
  problems.py      LPInstance + generators + special cases
  experiment.py    Experiment runners (grids, special cases, SciPy cross-check)
scripts/
  run_lecture_example.py   Task A
  run_grid_experiment.py   Tasks B + C (all grids + special cases)
analysis/
  analyze_results.py       Task D — tables, plots, ANALYSIS.md
results/
  lecture_example.csv      Task A output
  grid_main.csv            Main scaling grid (240 solves)
  grid_mixed_wellposed.csv Mixed-constraint grid, x_ref feasible
  grid_mixed_stress.csv    Mixed-constraint grid (infeasible/unbounded stress)
  special_cases.csv        Hand-crafted phenomena
  tables/summary_*.csv     Aggregated (m, n) summaries
  plots/*.png              Figures for the report
  ANALYSIS.md              Report-ready measured tables + discussion prompts
simplex_project.ipynb      Step-by-step walkthrough
```

## Report (Overleaf)

`report/main.tex` is the project report in Overleaf format.

**To compile on Overleaf:**
1. Create a new Overleaf project (pdfLaTeX).
2. Upload `report/main.tex` as `main.tex`.
3. Upload every file in `report/figures/` into the project root
   (the preamble sets `\graphicspath{{figures/}{./}}`, so either layout works).
4. Fill in team member names / video link in the title block.
5. Compile (two passes) — main text is 8 pages, appendix follows.

**Locally:** `cd report && latexmk -pdf main.tex`

The report covers: Introduction (LP equations), Methods (two-phase Simplex
pseudocode + SciPy cross-check + usage), Results (tables/plots of pivots,
time, cost vs $m,n$), Experimental Analysis (scaling, infeasibility,
unboundedness, degeneracy, sparsity), Conclusion, and an Appendix with extra
figures/tables and code excerpts.

## Solver notes

- **Standard form:** `A_ub x ≤ b_ub`, `A_eq x = b_eq`, `x ≥ 0`. Rows with
  negative `b` are flipped; `≥` rows get surplus + artificial variables.
- **Phase I:** maximizes `w = −Σ artificials`. `w* < 0` ⇒ **infeasible**.
- **Phase II:** optimizes the true objective from the Phase I basis.
  A nonbasic column with positive reduced cost and no positive entry ⇒
  **unbounded**.
- **Pivot rule:** Dantzig (most negative entry in the objective row, which
  stores `−reduced costs | z`); switches to Bland's rule after a safeguard
  number of pivots to guarantee termination on degenerate problems.
- **Diagnostics recorded per solve:** total/Phase I/Phase II pivots, wall-clock
  time, objective, `x*`, degeneracy (a *basic* variable at zero), alternative
  optima (nonbasic variable with ~zero reduced cost), sparsity `1 − nnz(x)/n`.
- **Cross-check:** SciPy/HiGHS solves the same LP; objectives agree to
  numerical precision on every optimal instance in the main grid. HiGHS does
  not expose simplex pivot counts, so pivots are measured only by the custom
  solver.
