"""Figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .analyze import load, q3_exactness  # noqa: E402

SOLVER_ORDER = ["greedy", "ilp", "dp", "dp_sliding_hirschberg"]


def fig_proxy_vs_true(results: list[dict], path: Path) -> None:
    labels = sorted({r["label"] for r in results})
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 3.8), squeeze=False)
    for ax, label in zip(axes[0], labels):
        for solver in SOLVER_ORDER:
            rs = sorted(
                [r for r in results if r["ok"] and r["label"] == label and r["solver"] == solver],
                key=lambda r: r["budget"],
            )
            if not rs:
                continue
            ax.plot([r["budget"] for r in rs], [r["proxy_peak_memory"] for r in rs],
                    linestyle="--", alpha=0.45, marker="o", markersize=3, label=f"{solver} (proxy)")
            ax.plot([r["budget"] for r in rs], [r["true_peak_memory"] for r in rs],
                    linestyle="-", marker="s", markersize=3, label=f"{solver} (true)")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("memory budget")
        ax.set_ylabel("peak memory (normalised)")
        ax.grid(alpha=0.25)
    axes[0][-1].legend(fontsize=6, ncol=2, loc="upper left")
    fig.suptitle("Objective the solver optimises (dashed) vs. simulated backward peak (solid)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_error_distribution(results: list[dict], path: Path) -> None:
    errs = [
        (r["true_peak_memory"] - r["proxy_peak_memory"]) / r["proxy_peak_memory"] * 100
        for r in results
        if r["ok"] and r["proxy_peak_memory"] > 0
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(errs, bins=40)
    ax.axvline(7.4, linestyle="--", label="greedy's reported proxy gap (7.4%)")
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("peak memory understated by (%)")
    ax.set_ylabel("count")
    ax.set_title("How far the knapsack objective is from the real backward peak")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_solver_cost(results: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    data, names = [], []
    for solver in SOLVER_ORDER:
        vals = [r["solver_seconds"] * 1000 for r in results if r["ok"] and r["solver"] == solver]
        if vals:
            data.append(vals)
            names.append(solver)
    ax.boxplot(data, labels=names, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("solver wall time (ms, log scale)")
    ax.set_title("Solver cost at realistic input scale (W <= 10,000)")
    ax.grid(alpha=0.25, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(outdir: str | Path) -> list[Path]:
    outdir = Path(outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    _, results = load(outdir)
    paths = [
        figdir / "proxy_vs_true.png",
        figdir / "error_distribution.png",
        figdir / "solver_cost.png",
    ]
    fig_proxy_vs_true(results, paths[0])
    fig_error_distribution(results, paths[1])
    fig_solver_cost(results, paths[2])
    return paths
