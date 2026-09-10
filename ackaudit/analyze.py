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
    q1, q2 = q1_input_scale(graphs), q2_proxy_error(results)
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

    add("\nSOLVER COST")
    add(f"  {'solver':24s} {'median s':>10s} {'max s':>10s} {'peak MB':>10s} {'failed':>8s}")
    for c in cost:
        add(f"  {c['solver']:24s} {c['median_seconds']:10.4f} {c['max_seconds']:10.4f} "
            f"{c['max_peak_mb']:10.2f} {c['n_failed']:8d}")
    add("")
    return "\n".join(lines)


def heterogeneity(graphs: list[dict]) -> list[dict]:
    """Distinct item weights/values per graph. Uniform items force large tie sets."""
    out = []
    for g in graphs:
        n = g["n_items"]
        out.append(
            {
                "label": g["label"],
                "n_items": n,
                "distinct_memories": len(set(g["memories"])),
                "distinct_runtimes": len(set(g["runtimes"])),
                "distinct_pairs": len(set(zip(g["memories"], g["runtimes"]))),
                "uniformity": 1.0 - (len(set(zip(g["memories"], g["runtimes"]))) - 1) / max(n - 1, 1),
            }
        )
    return out


def tie_sets(results: list[dict]) -> dict[str, Any]:
    """How many solvers land on the proxy optimum, and how far apart their true peaks are."""
    cells: dict[tuple[str, float], list[dict]] = {}
    for r in results:
        if r["ok"]:
            cells.setdefault((r["label"], r["budget"]), []).append(r)

    rows = []
    for (label, budget), rs in sorted(cells.items()):
        opt = min(x["proxy_peak_memory"] for x in rs)
        tied = [x for x in rs if abs(x["proxy_peak_memory"] - opt) < 1e-12]
        trues = [x["true_peak_memory"] for x in tied]
        spread = (max(trues) - min(trues)) / min(trues) if len(tied) > 1 and min(trues) > 0 else 0.0
        rows.append(
            {
                "label": label,
                "budget": budget,
                "n_solvers": len(rs),
                "n_tied": len(tied),
                "all_tied": len(tied) == len(rs),
                "tied_true_spread_pct": spread * 100,
            }
        )
    spreads = [r["tied_true_spread_pct"] for r in rows if r["n_tied"] > 1]
    return {
        "rows": rows,
        "n_cells": len(rows),
        "n_all_tied": sum(r["all_tied"] for r in rows),
        "frac_all_tied": sum(r["all_tied"] for r in rows) / len(rows) if rows else float("nan"),
        "median_tied_spread_pct": statistics.median(spreads) if spreads else 0.0,
        "max_tied_spread_pct": max(spreads, default=0.0),
        "frac_tied_spread_over_10pct": sum(s > 10 for s in spreads) / len(spreads) if spreads else 0.0,
    }


def degeneracy_report(outdir: str | Path) -> str:
    graphs, results = load(outdir)
    het = heterogeneity(graphs)
    ties = tie_sets(results)
    lines = ["=" * 72, "DEGENERACY CHECK", "=" * 72, ""]
    lines.append(f"  {'graph':16s} {'n':>5s} {'distinct mem':>13s} {'distinct rt':>12s} {'distinct pairs':>15s}")
    for h in het:
        lines.append(f"  {h['label']:16s} {h['n_items']:5d} {h['distinct_memories']:13d} "
                     f"{h['distinct_runtimes']:12d} {h['distinct_pairs']:15d}")
    lines += ["", "TIE SETS AT THE PROXY OPTIMUM"]
    lines.append(f"  cells                        : {ties['n_cells']}")
    lines.append(f"  cells where ALL solvers tie  : {ties['n_all_tied']} ({ties['frac_all_tied']*100:.0f}%)")
    lines.append(f"  median true-peak spread      : {ties['median_tied_spread_pct']:.1f}%")
    lines.append(f"  max true-peak spread         : {ties['max_tied_spread_pct']:.1f}%")
    lines.append(f"  fraction of ties over 10%    : {ties['frac_tied_spread_over_10pct']*100:.0f}%")
    lines.append("")
    return "\n".join(lines)
