"""Summary statistics over the captured sweep."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


def load(outdir: str | Path) -> tuple[list[dict], list[dict]]:
    outdir = Path(outdir)
    graphs, results = [], []
    for gfile in sorted(outdir.rglob("graphs.json")):
        graphs.extend(json.loads(gfile.read_text()))
    for rfile in sorted(outdir.rglob("results.json")):
        results.extend(json.loads(rfile.read_text()))
    return graphs, results


def q1_input_scale(graphs: list[dict]) -> dict[str, Any]:
    if not graphs:
        return {}
    return {
        "n_graphs": len(graphs),
        "n_items_min": min(g["n_items"] for g in graphs),
        "n_items_max": max(g["n_items"] for g in graphs),
        "W_min": min(g["quantised_capacity"] for g in graphs),
        "W_max": max(g["quantised_capacity"] for g in graphs),
        "dp_table_bytes_max": max(g["dp_table_bytes"] for g in graphs),
        "dp_table_mb_max": max(g["dp_table_bytes"] for g in graphs) / 1e6,
        # structural, not a sample max: partitioner clamps budget <= 1
        "W_structural_ceiling": 10_000,
    }


def q2_proxy_error(results: list[dict]) -> dict[str, Any]:
    errs = []
    for r in results:
        if not r["ok"] or r["proxy_peak_memory"] <= 0:
            continue
        e = (r["true_peak_memory"] - r["proxy_peak_memory"]) / r["proxy_peak_memory"]
        if not math.isnan(e):
            errs.append(e)
    if not errs:
        return {}
    errs.sort()
    return {
        "n": len(errs),
        "median_pct": statistics.median(errs) * 100,
        "mean_pct": statistics.fmean(errs) * 100,
        "p90_pct": errs[int(0.9 * (len(errs) - 1))] * 100,
        "max_pct": errs[-1] * 100,
        "frac_over_10pct": sum(e > 0.10 for e in errs) / len(errs),
    }


def q3_exactness(results: list[dict]) -> dict[str, Any]:
    """Per (graph, budget): spread on the proxy objective vs. on the true one."""
    groups: dict[tuple[str, float], list[dict]] = {}
    for r in results:
        if r["ok"]:
            groups.setdefault((r["label"], r["budget"]), []).append(r)

    rows = []
    for (label, budget), rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        proxies = [r["proxy_peak_memory"] for r in rs]
        trues = [r["true_peak_memory"] for r in rs]
        best_proxy, best_true = min(proxies), min(trues)
        # spread among plans that tie at the proxy optimum
        tied = [r for r in rs if math.isclose(r["proxy_peak_memory"], best_proxy, rel_tol=1e-9)]
        tied_true = [r["true_peak_memory"] for r in tied]
        rows.append(
            {
                "label": label,
                "budget": budget,
                "n_solvers": len(rs),
                "proxy_spread_pct": (max(proxies) - best_proxy) / best_proxy * 100
                if best_proxy > 0 else float("nan"),
                "true_spread_pct": (max(trues) - best_true) / best_true * 100
                if best_true > 0 else float("nan"),
                "n_tied_on_proxy": len(tied),
                "tied_true_spread_pct": (max(tied_true) - min(tied_true)) / min(tied_true) * 100
                if len(tied) > 1 and min(tied_true) > 0 else 0.0,
                "best_true_solver": min(rs, key=lambda r: r["true_peak_memory"])["solver"],
                "best_proxy_solver": min(rs, key=lambda r: r["proxy_peak_memory"])["solver"],
            }
        )
    disagreements = sum(r["best_true_solver"] != r["best_proxy_solver"] for r in rows)
    return {
        "rows": rows,
        "n_cells": len(rows),
        "n_disagreements": disagreements,
        "disagreement_rate": disagreements / len(rows) if rows else float("nan"),
        "max_tied_true_spread_pct": max((r["tied_true_spread_pct"] for r in rows), default=0.0),
    }


def solver_cost(results: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for r in results:
        by.setdefault(r["solver"], []).append(r)
    out = []
    for name, rs in sorted(by.items()):
        ok = [r for r in rs if r["ok"]]
        out.append(
            {
                "solver": name,
                "n_runs": len(rs),
                "n_failed": len(rs) - len(ok),
                "median_seconds": statistics.median([r["solver_seconds"] for r in ok]) if ok else float("nan"),
                "max_seconds": max([r["solver_seconds"] for r in ok], default=float("nan")),
                "max_peak_mb": max([r["solver_peak_bytes"] for r in ok], default=0) / 1e6,
            }
        )
    return out


def report(outdir: str | Path) -> str:
    graphs, results = load(outdir)
    q1, q2, q3 = q1_input_scale(graphs), q2_proxy_error(results), q3_exactness(results)
    cost = solver_cost(results)

    lines: list[str] = []
    add = lines.append
    add("=" * 68)
    add("ACTIVATION-CHECKPOINTING KNAPSACK AUDIT")
    add("=" * 68)

    add("\nQ1  INPUT SCALE ON REAL CAPTURED GRAPHS")
    if q1:
        add(f"  graphs captured      : {q1['n_graphs']}")
        add(f"  n items              : {q1['n_items_min']} - {q1['n_items_max']}")
        add(f"  quantised capacity W : {q1['W_min']} - {q1['W_max']}")
        add(f"  W structural ceiling : {q1['W_structural_ceiling']}  (S=10000, budget<=1)")
        add(f"  largest dp table     : {q1['dp_table_mb_max']:.3f} MB")

    add("\nQ2  PROXY ERROR  (true peak vs. the objective the solver optimises)")
    if q2:
        add(f"  measurements         : {q2['n']}")
        add(f"  median understatement: {q2['median_pct']:+.1f}%")
        add(f"  mean                 : {q2['mean_pct']:+.1f}%")
        add(f"  p90                  : {q2['p90_pct']:+.1f}%")
        add(f"  max                  : {q2['max_pct']:+.1f}%")
        add(f"  fraction over 10%    : {q2['frac_over_10pct']*100:.0f}%")

    add("\nQ3  DOES EXACTNESS SURVIVE?")
    if q3 and q3["n_cells"]:
        add(f"  (graph, budget) cells: {q3['n_cells']}")
        add(f"  cells where the proxy-best solver is NOT the true-best: "
            f"{q3['n_disagreements']} ({q3['disagreement_rate']*100:.0f}%)")
        add(f"  max true-peak spread among plans TIED at the proxy optimum: "
            f"{q3['max_tied_true_spread_pct']:.1f}%")
        add("  reference: greedy's reported proxy-objective gap is ~7.4%")

    add("\nSOLVER COST")
    add(f"  {'solver':24s} {'median s':>10s} {'max s':>10s} {'peak MB':>10s} {'failed':>8s}")
    for c in cost:
        add(f"  {c['solver']:24s} {c['median_seconds']:10.4f} {c['max_seconds']:10.4f} "
            f"{c['max_peak_mb']:10.2f} {c['n_failed']:8d}")
    add("")
    return "\n".join(lines)
