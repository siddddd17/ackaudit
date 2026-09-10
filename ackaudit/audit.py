"""Capture and score activation-checkpointing knapsack instances.

Scores each plan two ways: the objective the solvers optimise (sum of saved
tensor sizes) and the simulated backward peak, which includes recomputation
chains. See README for why these differ.
"""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch._functorch._activation_checkpointing.graph_info_provider import (
    GraphInfoProvider,
)
from torch._functorch._activation_checkpointing.knapsack import (
    dp_knapsack,
    dp_knapsack_sliding_hirschberg,
    greedy_knapsack,
    ilp_knapsack,
)
from torch._functorch._activation_checkpointing.knapsack_evaluator import (
    KnapsackEvaluator,
)
from torch._functorch.partitioners import CustomKnapsackSolver

from .schedule import DEFAULT_SCHEDULE, Schedule

log = logging.getLogger(__name__)

# matches knapsack.dp_knapsack
QUANTISATION_SCALE = 10_000

SolverFn = Callable[[list[float], list[float], float], tuple[float, list[int], list[int]]]

SOLVERS: dict[str, SolverFn] = {
    "greedy": greedy_knapsack,
    "ilp": ilp_knapsack,
    "dp": dp_knapsack,
    "dp_sliding_hirschberg": dp_knapsack_sliding_hirschberg,
}


@dataclass
class GraphRecord:
    label: str
    n_items: int
    max_memory: float
    quantised_capacity: int
    dp_table_cells: int
    dp_table_bytes: int
    memory_min: float
    memory_max: float
    memory_sum: float
    runtime_min: float
    runtime_max: float
    runtime_sum: float
    memories: list[float] = field(default_factory=list)
    runtimes: list[float] = field(default_factory=list)


@dataclass
class SolverResult:
    label: str
    solver: str
    budget: float
    schedule: str
    ok: bool
    error: str = ""
    solver_seconds: float = 0.0
    solver_peak_bytes: int = 0
    n_saved: int = 0
    n_recomputed: int = 0
    # Objective as the knapsack sees it: sum of saved tensor sizes.
    proxy_peak_memory: float = 0.0
    # Objective as the backward pass actually experiences it.
    true_peak_memory: float = 0.0
    recomputation_runtime: float = 0.0
    non_ac_peak_memory: float = 0.0
    theoretical_max_runtime: float = 0.0

    @property
    def proxy_error(self) -> float:
        if self.proxy_peak_memory <= 0:
            return float("nan")
        return (self.true_peak_memory - self.proxy_peak_memory) / self.proxy_peak_memory


def _measure(fn: SolverFn, memories, runtimes, budget):
    tracemalloc.start()
    start = time.perf_counter()
    try:
        value, saved, recomputed = fn(memories, runtimes, budget)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        return True, "", elapsed, peak, saved, recomputed
    except Exception as exc:  # record OOM and solver bugs, don't raise
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        return False, f"{type(exc).__name__}: {exc}", elapsed, peak, [], []
    finally:
        tracemalloc.stop()


def audit_instance(
    label: str,
    memories: list[float],
    runtimes: list[float],
    max_memory: float,
    joint_graph: torch.fx.Graph,
    banned_nodes: list[torch.fx.Node],
    budgets: Optional[list[float]] = None,
    solvers: Optional[list[str]] = None,
    schedule: str = DEFAULT_SCHEDULE,
) -> tuple[GraphRecord, list[SolverResult]]:
    if budgets is None:
        budgets = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if solvers is None:
        solvers = list(SOLVERS)

    quantised = round(max_memory * QUANTISATION_SCALE)
    cells = (len(memories) + 1) * (quantised + 1)
    record = GraphRecord(
        label=label,
        n_items=len(memories),
        max_memory=max_memory,
        quantised_capacity=quantised,
        dp_table_cells=cells,
        dp_table_bytes=cells * 4,
        memory_min=min(memories) if memories else 0.0,
        memory_max=max(memories) if memories else 0.0,
        memory_sum=sum(memories),
        runtime_min=min(runtimes) if runtimes else 0.0,
        runtime_max=max(runtimes) if runtimes else 0.0,
        runtime_sum=sum(runtimes),
        memories=list(memories),
        runtimes=list(runtimes),
    )

    provider = GraphInfoProvider.inialize_from_graph(
        joint_graph=joint_graph,
        all_recomputable_banned_nodes=banned_nodes,
        recorded_knapsack_input_memories=memories,
        recorded_knapsack_input_runtimes=runtimes,
    )
    evaluator = KnapsackEvaluator(graph_info_provider=provider)

    results: list[SolverResult] = []
    for budget in budgets:
        for name in solvers:
            ok, err, secs, peak_bytes, saved, recomputed = _measure(
                SOLVERS[name], memories, runtimes, budget
            )
            res = SolverResult(
                label=label,
                solver=name,
                budget=budget,
                schedule=schedule,
                ok=ok,
                error=err,
                solver_seconds=secs,
                solver_peak_bytes=peak_bytes,
                n_saved=len(saved),
                n_recomputed=len(recomputed),
            )
            if ok:
                try:
                    with Schedule(schedule, provider.graph_nodes_in_order):
                        proxy = evaluator.evaluate_knapsack_output(
                            saved_nodes_idxs=saved,
                            recomputable_node_idxs=recomputed,
                            account_for_backward_pass=False,
                        )
                        true = evaluator.evaluate_knapsack_output(
                            saved_nodes_idxs=saved,
                            recomputable_node_idxs=recomputed,
                            account_for_backward_pass=True,
                        )
                    res.proxy_peak_memory = proxy["peak_memory"]
                    res.true_peak_memory = true["peak_memory"]
                    res.recomputation_runtime = proxy["recomputation_runtime"]
                    res.non_ac_peak_memory = proxy["non_ac_peak_memory"]
                    res.theoretical_max_runtime = proxy["theoretical_max_runtime"]
                except Exception as exc:
                    res.ok = False
                    res.error = f"evaluate: {type(exc).__name__}: {exc}"
            results.append(res)
    return record, results


class AuditingSolver(CustomKnapsackSolver):
    """Records the knapsack instance, audits it, then defers to `delegate`.

    Install with:
        torch._functorch.config.activation_memory_budget_solver = AuditingSolver(...)
    """

    def __init__(
        self,
        outdir: str | Path,
        label: str = "graph",
        budgets: Optional[list[float]] = None,
        solvers: Optional[list[str]] = None,
        delegate: SolverFn = dp_knapsack,
        recorder: Optional["RuntimeRecorder"] = None,
        schedule: str = DEFAULT_SCHEDULE,
    ) -> None:
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.label = label
        self.budgets = budgets
        self.solvers = solvers
        self.delegate = delegate
        self.recorder = recorder
        self.schedule = schedule
        self.records: list[GraphRecord] = []
        self.results: list[SolverResult] = []
        self._counter = 0

    def __call__(
        self,
        memory: list[float],
        joint_graph: torch.fx.Graph,
        max_memory: float,
        node_info: Any,
        all_recomputable_banned_nodes: list[torch.fx.Node],
    ) -> tuple[list[int], list[int]]:
        runtimes = self.recorder.lookup(all_recomputable_banned_nodes)
        tag = f"{self.label}#{self._counter}"
        self._counter += 1

        try:
            record, results = audit_instance(
                label=tag,
                memories=memory,
                runtimes=runtimes,
                max_memory=max_memory,
                joint_graph=joint_graph,
                banned_nodes=all_recomputable_banned_nodes,
                budgets=self.budgets,
                solvers=self.solvers,
                schedule=self.schedule,
            )
            self.records.append(record)
            self.results.extend(results)
            self.flush()
        except Exception:
            log.exception("audit failed for %s, continuing compile", tag)

        _, saved, recomputed = self.delegate(memory, runtimes, max_memory)
        return saved, recomputed

    def uuid(self) -> Any:
        # None so we're never cache-hit past; want a call per compile
        return None

    def flush(self) -> None:
        (self.outdir / "graphs.json").write_text(
            json.dumps([asdict(r) for r in self.records], indent=2)
        )
        (self.outdir / "results.json").write_text(
            json.dumps([asdict(r) for r in self.results], indent=2)
        )


class RuntimeRecorder:
    """Caches the runtimes the partitioner computes for each node.

    The solver runs inside `no_dispatch()` (partitioners.py, get_saved_values_
    knapsack), where estimate_runtime's flops mode executes node.target on real
    materialized tensors instead of fake ones. That crashes on ops like embedding
    whose arguments must be in range. The partitioner computes runtimes outside
    that block, so we record them there and look them up later.
    """

    def __init__(self) -> None:
        self.cache: dict[torch.fx.Node, float] = {}
        self._orig = None

    def __enter__(self) -> "RuntimeRecorder":
        import torch._functorch.partitioners as P

        self._orig = P.estimate_runtime

        def wrapped(node):
            value = self._orig(node)
            self.cache[node] = value
            return value

        P.estimate_runtime = wrapped
        return self

    def __exit__(self, *exc) -> None:
        import torch._functorch.partitioners as P

        if self._orig is not None:
            P.estimate_runtime = self._orig

    def lookup(self, nodes: list[torch.fx.Node]) -> list[float]:
        missing = [n for n in nodes if n not in self.cache]
        if missing:
            raise KeyError(
                f"{len(missing)}/{len(nodes)} nodes have no recorded runtime; "
                "the recorder was not active when the partitioner ran"
            )
        return [self.cache[n] for n in nodes]
