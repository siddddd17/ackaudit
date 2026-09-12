"""
Experiment: schedule sensitivity of PyTorch's backward-memory simulator.

The simulator linearises a DAG using a topological ordering. When multiple
valid topological orders exist, the resulting simulated backward peak can
depend on that ordering.

This experiment captures one fixed checkpoint plan per graph/budget and
evaluates the same plan under three valid schedules:

    A: default NetworkX topological ordering
    B: deterministic lexicographic topological ordering
    C: ordering induced by the FX graph

The goal is to quantify schedule sensitivity and determine whether it is
large enough to affect checkpointing decisions on realistic graphs.

This is an exploratory experiment, not evidence by itself of a PyTorch bug.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch._functorch.config as functorch_config
from torch._functorch._activation_checkpointing.graph_info_provider import (
    GraphInfoProvider,
)
from torch._functorch._activation_checkpointing.knapsack_evaluator import (
    KnapsackEvaluator,
)
from torch._functorch.partitioners import CustomKnapsackSolver, dp_knapsack

from ackaudit.audit import RuntimeRecorder
from ackaudit.schedule import SCHEDULES, Schedule


@dataclass
class ScheduleRow:
    graph: str
    budget: float
    n_items: int
    n_saved: int
    proxy_peak: float
    peaks: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def spread_pct(self) -> float:
        vals = [v for v in self.peaks.values()]
        if len(vals) < 2 or min(vals) <= 0:
            return 0.0
        return (max(vals) - min(vals)) / min(vals) * 100

    @property
    def n_distinct(self) -> int:
        return len({round(v, 9) for v in self.peaks.values()})


class _Comparer(CustomKnapsackSolver):
    """Solves once with dp, then re-scores that single plan under each schedule."""

    def __init__(self, graph: str, budgets: list[float], recorder: RuntimeRecorder):
        self.graph = graph
        self.budgets = budgets
        self.recorder = recorder
        self.rows: list[ScheduleRow] = []
        self._done = False

    def __call__(self, memory, joint_graph, max_memory, node_info, banned):
        runtimes = self.recorder.lookup(list(banned))
        if not self._done:
            self._done = True
            provider = GraphInfoProvider.inialize_from_graph(
                joint_graph=joint_graph,
                all_recomputable_banned_nodes=banned,
                recorded_knapsack_input_memories=list(memory),
                recorded_knapsack_input_runtimes=runtimes,
            )
            ev = KnapsackEvaluator(graph_info_provider=provider)
            order = provider.graph_nodes_in_order

            for b in self.budgets:
                _, saved, recomp = dp_knapsack(list(memory), runtimes, b)
                row = ScheduleRow(
                    graph=self.graph,
                    budget=b,
                    n_items=len(memory),
                    n_saved=len(saved),
                    proxy_peak=0.0,
                )
                for mode in SCHEDULES:
                    try:
                        with Schedule(mode, order):
                            out = ev.evaluate_knapsack_output(
                                saved_nodes_idxs=saved,
                                recomputable_node_idxs=recomp,
                                account_for_backward_pass=True,
                            )
                            if not row.proxy_peak:
                                row.proxy_peak = ev.evaluate_knapsack_output(
                                    saved_nodes_idxs=saved,
                                    recomputable_node_idxs=recomp,
                                    account_for_backward_pass=False,
                                )["peak_memory"]
                        row.peaks[mode] = out["peak_memory"]
                    except Exception as exc:
                        row.errors[mode] = f"{type(exc).__name__}: {exc}"
                self.rows.append(row)

        _, s, r = dp_knapsack(list(memory), runtimes, max_memory)
        return s, r

    def uuid(self) -> Any:
        return None


def compare_schedules(
    name: str,
    build,
    make_inputs,
    budgets: Optional[list[float]] = None,
) -> list[ScheduleRow]:
    budgets = budgets or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    recorder = RuntimeRecorder()
    comparer = _Comparer(name, budgets, recorder)

    prev_s = functorch_config.activation_memory_budget_solver
    prev_b = functorch_config.activation_memory_budget
    try:
        functorch_config.activation_memory_budget_solver = comparer
        functorch_config.activation_memory_budget = 0.5
        torch._dynamo.reset()
        with recorder:
            compiled = torch.compile(build(), backend="aot_eager", dynamic=False)
            compiled(*make_inputs()).backward()
    finally:
        functorch_config.activation_memory_budget_solver = prev_s
        functorch_config.activation_memory_budget = prev_b
    return comparer.rows


def run(
    outdir: str | Path,
    budgets: Optional[list[float]] = None,
    names: Optional[list[str]] = None,
) -> list[ScheduleRow]:
    from .graph_families import families

    zoo = families()
    rows: list[ScheduleRow] = []
    for name in names or list(zoo):
        build, mk = zoo[name]
        try:
            rows += compare_schedules(name, build, mk, budgets)
        except Exception as exc:
            print(f"  {name}: capture failed, {type(exc).__name__}: {exc}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "schedules.json").write_text(
        json.dumps(
            [
                {**asdict(r), "spread_pct": r.spread_pct, "n_distinct": r.n_distinct}
                for r in rows
            ],
            indent=2,
        )
    )
    return rows


def report(rows: list[ScheduleRow]) -> str:
    lines = [
        "=" * 78,
        "SCHEDULE COMPARISON  (same plan, three valid topological orders)",
        "=" * 78,
        "",
    ]
    lines.append(
        f"  {'graph':18s} {'budget':>7s} {'n':>4s} {'proxy':>8s} "
        f"{'A default':>10s} {'B lex':>10s} {'C fx':>10s} {'spread':>8s}"
    )
    for r in rows:
        a = r.peaks.get("default", float("nan"))
        b = r.peaks.get("lexicographic", float("nan"))
        c = r.peaks.get("fx_order", float("nan"))
        lines.append(
            f"  {r.graph:18s} {r.budget:7.2f} {r.n_items:4d} {r.proxy_peak:8.4f} "
            f"{a:10.4f} {b:10.4f} {c:10.4f} {r.spread_pct:7.1f}%"
        )

    by_graph: dict[str, list[ScheduleRow]] = {}
    for r in rows:
        by_graph.setdefault(r.graph, []).append(r)
    lines += ["", "PER GRAPH", ""]
    lines.append(
        f"  {'graph':18s} {'cells':>6s} {'schedules agree':>16s} {'median spread':>14s} {'max spread':>11s}"
    )
    for g, rs in by_graph.items():
        agree = sum(r.n_distinct == 1 for r in rs)
        spreads = sorted(r.spread_pct for r in rs)
        med = spreads[len(spreads) // 2] if spreads else 0.0
        lines.append(
            f"  {g:18s} {len(rs):6d} {f'{agree}/{len(rs)}':>16s} "
            f"{med:13.1f}% {max(spreads, default=0.0):10.1f}%"
        )
    lines.append("")
    return "\n".join(lines)
