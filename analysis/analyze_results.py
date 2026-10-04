"""Task D: analyze experiment results — tables, plots, and written observations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
TABLES = RESULTS / "tables"
PLOTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def _load(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/run_grid_experiment.py first")
    return pd.read_csv(path)


def summarize_grid(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Mean/std of pivots, time, cost, sparsity grouped by (m, n)."""
    ok = df[df["status"] == "optimal"].copy()
    agg = (
        df.groupby(["m", "n"])
        .agg(
            n_solves=("status", "size"),
            optimal_frac=("status", lambda s: (s == "optimal").mean()),
            infeasible_n=("status", lambda s: (s == "infeasible").sum()),
            unbounded_n=("status", lambda s: (s == "unbounded").sum()),
            pivots_mean=("pivots", "mean"),
            pivots_std=("pivots", "std"),
            pivots_max=("pivots", "max"),
            time_mean_ms=("simplex_time", lambda s: s.mean() * 1e3),
            time_std_ms=("simplex_time", lambda s: s.std() * 1e3),
            scipy_time_mean_ms=("scipy_time", lambda s: s.mean() * 1e3),
            scipy_nit_mean=("scipy_nit", "mean"),
            objective_mean=("objective", "mean"),
            sparsity_mean=("sparsity", "mean"),
            degenerate_frac=("degenerate", lambda s: s.astype(bool).mean()),
            multi_opt_frac=("multiple_optima", lambda s: s.astype(bool).mean()),
        )
        .reset_index()
    )
    # objective_mean only meaningful on optimal rows
    obj = ok.groupby(["m", "n"])["objective"].mean().rename("objective_mean_opt")
    agg = agg.drop(columns=["objective_mean"]).merge(obj, on=["m", "n"], how="left")
    agg.to_csv(TABLES / f"summary_{label}.csv", index=False)
    return agg


def plot_scaling(df_main: pd.DataFrame) -> None:
    ok = df_main[df_main["status"] == "optimal"]

    # --- 1. Pivots vs n for each m ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, sub in ok.groupby("m"):
        g = sub.groupby("n")["pivots"]
        mean = g.mean()
        std = g.std().fillna(0)
        ax.errorbar(mean.index, mean.values, yerr=std.values, marker="o", capsize=3, label=f"m = {m}")
    ax.set_xlabel("Number of variables  n")
    ax.set_ylabel("Simplex pivots (mean ± std)")
    ax.set_title("Pivot count vs problem size (number of variables)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "pivots_vs_n.png", dpi=150)
    plt.close(fig)

    # --- 2. Time vs n for each m (custom Simplex) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, sub in ok.groupby("m"):
        g = sub.groupby("n")["simplex_time"]
        mean = g.mean() * 1e3
        ax.plot(mean.index, mean.values, marker="s", label=f"m = {m}")
    ax.set_xlabel("Number of variables  n")
    ax.set_ylabel("Simplex solve time (ms, mean)")
    ax.set_title("Custom Simplex runtime vs number of variables")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(PLOTS / "time_vs_n.png", dpi=150)
    plt.close(fig)

    # --- 3. Custom Simplex vs SciPy time at m = 2 ---
    sub = ok[ok["m"] == 2]
    fig, ax = plt.subplots(figsize=(8, 5))
    g = sub.groupby("n")
    ax.plot(g["simplex_time"].mean().index, g["simplex_time"].mean().values * 1e3,
            marker="o", label="Custom two-phase Simplex")
    ax.plot(g["scipy_time"].mean().index, g["scipy_time"].mean().values * 1e3,
            marker="^", label="SciPy linprog (HiGHS)")
    ax.set_xlabel("Number of variables  n  (m = 2)")
    ax.set_ylabel("Mean solve time (ms)")
    ax.set_title("Custom Simplex vs SciPy/HiGHS runtime (m = 2)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(PLOTS / "custom_vs_scipy_time_m2.png", dpi=150)
    plt.close(fig)

    # --- 4. Pivots vs m for each n ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes = axes.ravel()
    for i, n in enumerate(sorted(ok["n"].unique())):
        ax = axes[i]
        sub = ok[ok["n"] == n]
        g = sub.groupby("m")["pivots"]
        mean, std = g.mean(), g.std().fillna(0)
        ax.errorbar(mean.index, mean.values, yerr=std.values, marker="o", capsize=3, color="C0")
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Constraints m")
        ax.set_ylabel("Pivots")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Pivot count vs number of constraints (one panel per n)")
    fig.tight_layout()
    fig.savefig(PLOTS / "pivots_vs_m_by_n.png", dpi=150)
    plt.close(fig)

    # --- 5. Time vs m for each n ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes = axes.ravel()
    for i, n in enumerate(sorted(ok["n"].unique())):
        ax = axes[i]
        sub = ok[ok["n"] == n]
        g = sub.groupby("m")["simplex_time"]
        ax.plot(g.mean().index, g.mean().values * 1e3, marker="s", color="C3")
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Constraints m")
        ax.set_ylabel("Time (ms)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle("Simplex runtime vs number of constraints (one panel per n)")
    fig.tight_layout()
    fig.savefig(PLOTS / "time_vs_m_by_n.png", dpi=150)
    plt.close(fig)

    # --- 6. Objective value vs n (m = 2) ---
    sub = ok[ok["m"] == 2]
    fig, ax = plt.subplots(figsize=(8, 5))
    g = sub.groupby("n")
    ax.errorbar(g["objective"].mean().index, g["objective"].mean().values,
                yerr=g["objective"].std().fillna(0).values, marker="o", capsize=3)
    ax.set_xlabel("n (m = 2)")
    ax.set_ylabel("Optimal objective z* (mean ± std)")
    ax.set_title("Optimal cost vs number of variables (m = 2)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "objective_vs_n_m2.png", dpi=150)
    plt.close(fig)

    # --- 7. Sparsity vs size ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, sub in ok.groupby("m"):
        g = sub.groupby("n")["sparsity"]
        ax.plot(g.mean().index, g.mean().values, marker="o", label=f"m = {m}")
    ax.set_xlabel("n")
    ax.set_ylabel("Sparsity of optimal x  (1 - nnz/n)")
    ax.set_title("Optimal-solution sparsity vs problem size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "sparsity_vs_n.png", dpi=150)
    plt.close(fig)

    # --- 8. Degeneracy / multiple-optima rate vs size ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, sub in ok.groupby("m"):
        g = sub.groupby("n")
        ax.plot(g["degenerate"].mean().index, g["degenerate"].mean().values,
                marker="o", label=f"degenerate (m={m})")
        ax.plot(g["multiple_optima"].mean().index, g["multiple_optima"].mean().values,
                marker="x", linestyle="--", label=f"multi-opt (m={m})")
    ax.set_xlabel("n")
    ax.set_ylabel("Fraction of optimal solves")
    ax.set_title("Degeneracy and alternative-optima frequency")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "degeneracy_vs_n.png", dpi=150)
    plt.close(fig)


def plot_mixed_status(df_mixed: pd.DataFrame, tag: str = "mixed") -> None:
    """Stacked status bars for a mixed-constraint grid."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, status, title in zip(
        axes,
        ["infeasible", "unbounded"],
        ["Infeasible fraction", "Unbounded fraction"],
    ):
        grid = (
            df_mixed.assign(bad=df_mixed["status"] == status)
            .groupby(["n", "m"])["bad"].mean()
            .unstack("m")
        )
        if grid.size == 0:
            continue
        im = ax.imshow(grid.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(grid.columns)), grid.columns)
        ax.set_yticks(range(len(grid.index)), grid.index)
        ax.set_xlabel("m (constraints)")
        ax.set_ylabel("n (variables)")
        ax.set_title(f"{title} ({tag})")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(PLOTS / f"mixed_status_heatmap_{tag}.png", dpi=150)
    plt.close(fig)

    # pivots for mixed instances that are optimal
    ok = df_mixed[df_mixed["status"] == "optimal"]
    if len(ok):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(ok["n"] + 0.1 * (ok["m"] - ok["m"].min()) / max(1, ok["m"].max() - ok["m"].min()),
                   ok["pivots"], c=ok["m"], cmap="viridis", alpha=0.7)
        ax.set_xlabel("n (jittered by m)")
        ax.set_ylabel("Pivots")
        ax.set_title(f"Pivots on mixed-constraint LPs that reached an optimum ({tag})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS / f"mixed_pivots_{tag}.png", dpi=150)
        plt.close(fig)


def plot_special_cases(df_sp: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    colors = {"optimal": "C2", "infeasible": "C3", "unbounded": "C4"}
    bar_colors = [colors.get(s, "C7") for s in df_sp["status"]]
    ax.barh(df_sp["instance"], df_sp["pivots"].fillna(0), color=bar_colors)
    ax.set_xlabel("Total pivots")
    ax.set_title("Pivots by special-case instance")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
    ax.barh(df_sp["instance"], df_sp["simplex_time"].fillna(0) * 1e3, color=bar_colors)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Solve time by special-case instance")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[2]
    opt = df_sp[df_sp["status"] == "optimal"]
    ax.barh(opt["instance"], opt["sparsity"], color="C0")
    ax.set_xlabel("Sparsity of x*")
    ax.set_title("Optimal-solution sparsity")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS / "special_cases.png", dpi=150)
    plt.close(fig)

    # pivot phase breakdown
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df_sp))
    ax.bar(x, df_sp["phase1_pivots"].fillna(0), label="Phase I pivots", color="C1")
    ax.bar(x, df_sp["phase2_pivots"].fillna(0), bottom=df_sp["phase1_pivots"].fillna(0),
           label="Phase II pivots", color="C0")
    ax.set_xticks(x, df_sp["instance"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Pivots")
    ax.set_title("Phase I / Phase II pivot breakdown (special cases)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "special_phase_breakdown.png", dpi=150)
    plt.close(fig)


def write_analysis_markdown(df_main, df_mixed_wp, df_mixed_st, df_sp, sum_main, sum_mixed_wp) -> None:
    """Auto-generated analysis with measured numbers + discussion prompts."""
    ok = df_main[df_main["status"] == "optimal"]
    m2 = ok[ok["m"] == 2]
    n_list = sorted(m2["n"].unique())

    lines: list[str] = []
    lines.append("# Experimental Analysis — Simplex on Increasing LP Sizes")
    lines.append("")
    lines.append("Generated by `analysis/analyze_results.py`.  Paste the measured tables")
    lines.append("into the report's Results / Experimental Analysis sections.")
    lines.append("")
    lines.append("## 1. Scaling with n (fixing m = 2)")
    lines.append("")
    lines.append("| n | pivots (mean ± std) | simplex time (ms) | scipy time (ms) | scipy nit | objective z* | sparsity | degenerate frac | multi-opt frac |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in n_list:
        sub = m2[m2["n"] == n]
        lines.append(
            f"| {n} | {sub['pivots'].mean():.1f} ± {sub['pivots'].std():.1f} "
            f"| {sub['simplex_time'].mean()*1e3:.3f} "
            f"| {sub['scipy_time'].mean()*1e3:.3f} "
            f"| {sub['scipy_nit'].mean():.1f} "
            f"| {sub['objective'].mean():.3f} "
            f"| {sub['sparsity'].mean():.3f} "
            f"| {sub['degenerate'].mean():.2f} "
            f"| {sub['multiple_optima'].mean():.2f} |"
        )
    lines.append("")
    lines.append("## 2. Scaling with m (for each n)")
    lines.append("")
    pivot_p = sum_main.pivot(index="n", columns="m", values="pivots_mean")
    time_p = sum_main.pivot(index="n", columns="m", values="time_mean_ms")
    obj_p = sum_main.pivot(index="n", columns="m", values="objective_mean_opt")
    lines.append("### Mean pivots")
    lines.append("")
    lines.append(pivot_p.round(2).to_markdown())
    lines.append("")
    lines.append("### Mean Simplex time (ms)")
    lines.append("")
    lines.append(time_p.round(4).to_markdown())
    lines.append("")
    lines.append("### Mean optimal objective")
    lines.append("")
    lines.append(obj_p.round(3).to_markdown())
    lines.append("")
    lines.append("## 3. Status counts")
    lines.append("")
    lines.append(f"- Main grid (random <= LPs): {df_main['status'].value_counts().to_dict()}")
    lines.append(f"- Mixed well-posed grid: {df_mixed_wp['status'].value_counts().to_dict()}")
    lines.append(f"- Mixed stress grid: {df_mixed_st['status'].value_counts().to_dict()}")
    lines.append(f"- SciPy agreement on main grid: {df_main['objective_match'].mean():.1%} of solves")
    lines.append("")
    lines.append("## 4. Special cases")
    lines.append("")
    cols = ["instance", "status", "objective", "pivots", "phase1_pivots", "phase2_pivots",
            "degenerate", "multiple_optima", "n_nonzero", "sparsity", "message"]
    lines.append(df_sp[cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 5. Discussion prompts (map each to evidence above)")
    lines.append("")
    lines.append("1. **Pivots vs n at fixed m=2**: the vertex set of {x >= 0 : Ax <= b} has")
    lines.append("   dimension related to m; increasing n mostly adds candidate directions.")
    lines.append("   Does the mean pivot count grow with n?  Point at the m=2 column of the")
    lines.append("   pivot table and the `pivots_vs_n.png` figure.")
    lines.append("2. **Pivots vs m at fixed n**: each extra constraint cuts the polyhedron and")
    lines.append("   can create new vertices / longer pivot paths.  Compare rows of the")
    lines.append("   pivot table for n=50 across m=2,6,10,14.")
    lines.append("3. **Time vs pivots**: runtime should track pivot count almost linearly for")
    lines.append("   dense tableau Simplex (each pivot costs O(m*n)).  Compare")
    lines.append("   `time_vs_n.png` with `pivots_vs_n.png`.")
    lines.append("4. **Infeasibility (mixed stress grid / special case)**: Phase I minimizes the")
    lines.append("   sum of artificials; a strictly positive optimum means the constraint set")
    lines.append("   is empty.  Point at `infeasible_parallel_constraints` and the")
    lines.append("   mixed-status heatmap — infeasibility appears when >= rows demand more of")
    lines.append("   a resource than <= rows allow.")
    lines.append("5. **Unboundedness**: if a nonbasic variable has a positive reduced cost but")
    lines.append("   no positive entry in its column, the LP has a feasible improving ray.")
    lines.append("   Removing a bounding constraint (special case) opens that ray.")
    lines.append("6. **Degeneracy / multiple optima**: several active constraints at a vertex")
    lines.append("   force a basic variable to 0; a zero reduced cost on a nonbasic variable")
    lines.append("   means alternative optima exist.  See degenerate_vertex_3active and")
    lines.append("   multiple_optima_face.")
    lines.append("7. **Sparsity**: a basic feasible solution has at most m nonzero variables in")
    lines.append("   standard form, so optimal vertices are structurally sparse as n grows with")
    lines.append("   fixed m.  In production planning / portfolio / logistics LPs this means")
    lines.append("   few active products/assets/links — the zeros are the business insight.")
    lines.append("")
    lines.append("## 6. Plots")
    lines.append("")
    for p in sorted(PLOTS.glob("*.png")):
        lines.append(f"- `results/plots/{p.name}`")
    lines.append("")

    out = RESULTS / "ANALYSIS.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def main() -> None:
    df_main = _load("grid_main.csv")
    df_mixed_wp = _load("grid_mixed_wellposed.csv")
    df_mixed_st = _load("grid_mixed_stress.csv")
    df_sp = _load("special_cases.csv")

    sum_main = summarize_grid(df_main, "main")
    sum_mixed_wp = summarize_grid(df_mixed_wp, "mixed_wellposed")
    summarize_grid(df_mixed_st, "mixed_stress")

    plot_scaling(df_main)
    plot_mixed_status(df_mixed_st, tag="stress")
    plot_mixed_status(df_mixed_wp, tag="wellposed")
    plot_special_cases(df_sp)
    write_analysis_markdown(df_main, df_mixed_wp, df_mixed_st, df_sp, sum_main, sum_mixed_wp)

    print("Tables:")
    for p in sorted(TABLES.glob("*.csv")):
        print(f"  {p.relative_to(ROOT)}")
    print("Plots:")
    for p in sorted(PLOTS.glob("*.png")):
        print(f"  {p.relative_to(ROOT)}")
    print()
    print(sum_main.to_string(index=False))


if __name__ == "__main__":
    main()
