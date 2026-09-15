"""Standalone reproducer for PyTorch issue #196914.

Invokes the affected private helper on a tiny hand-built DAG. Prints the
whole trace and reports the torch version. Two cases:

    saved={node2}   affected: peak 0.7, residual 0.2; fixed: peak 0.4, 0.0
    saved={node1}   control unaffected by the bug: peak 0.4, residual 0.0
                    on both affected and fixed builds

The script prints RESULT lines and exits 0 whether the current build is
affected or fixed. It is a demonstration, not an assertion; use
experiments/backward_memory_accounting/saved_node_recomputation.py to fail a
build.

See https://github.com/pytorch/pytorch/issues/196914 and
https://github.com/pytorch/pytorch/pull/197117.
"""

from __future__ import annotations

import math
import sys

import torch
from torch._functorch._activation_checkpointing.graph_info_provider import (
    GraphInfoProvider,
)
from torch._functorch._activation_checkpointing.knapsack_evaluator import (
    KnapsackEvaluator,
)

NODE_MEMORIES = {"node1": 0.1, "node2": 0.2, "node5": 0.2}

CASES = [
    # (label, saved set, (peak, final) if affected, (peak, final) if fixed)
    # node2 is the failing case: a saved direct predecessor of a recomputed node.
    ("node2", {"node2"}, (0.7, 0.2), (0.4, 0.0)),
    # node1 is a control: node2 is not saved, so the affected filter never
    # sees a saved node in the initial predecessor queue and both builds agree.
    ("node1", {"node1"}, (0.5, 0.0), (0.5, 0.0)),
]


def build_provider() -> GraphInfoProvider:
    return GraphInfoProvider(
        graph_nodes_in_order=["node1", "node2", "node5"],
        graph_edges=[
            ("node1", "node2"),
            ("node2", "node5"),
        ],
        all_recomputable_banned_nodes=["node1", "node2", "node5"],
        recorded_knapsack_input_memories=[0.1, 0.2, 0.2],
        recorded_knapsack_input_runtimes=[1.0, 1.0, 1.0],
    )


def _matches(peak: float, final: float, expected: tuple[float, float]) -> bool:
    return math.isclose(peak, expected[0], abs_tol=1e-9) and math.isclose(
        final, expected[1], abs_tol=1e-9
    )


def run(saved: set[str]) -> list[tuple[float, str]]:
    provider = build_provider()
    evaluator = KnapsackEvaluator(provider)
    graph = provider.recomputable_node_only_graph_with_larger_graph_context
    return evaluator._get_backward_memory_from_topologically_sorted_graph(
        node_graph=graph,
        node_memories=provider.all_node_memories,
        saved_nodes_set=saved,
        peak_memory_after_forward_pass=sum(NODE_MEMORIES[n] for n in saved),
    )


def main() -> int:
    provider = build_provider()
    graph = provider.recomputable_node_only_graph_with_larger_graph_context

    print("PyTorch issue #196914: saved-node accounting")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Candidate graph edges: {sorted(graph.edges)}")
    print()

    for label, saved, affected, fixed in CASES:
        trace = run(saved)
        peak = max(memory for memory, _ in trace)
        final = trace[-1][0]
        print(f"saved={{{label}}}")
        for memory, event in trace:
            print(f"  {memory:+.1f}  {event}")
        print(f"  observed peak={peak:.1f} final={final:+.1f}")
        if affected == fixed and _matches(peak, final, fixed):
            verdict = "control (same on affected and fixed builds)"
        elif _matches(peak, final, affected):
            verdict = "affected (#196914 reproduced)"
        elif _matches(peak, final, fixed):
            verdict = "fixed (#196914 not reproduced)"
        else:
            raise AssertionError(
                f"saved={{{label}}}: unexpected ({peak}, {final}); "
                f"expected {affected} (affected) or {fixed} (fixed)"
            )
        print(f"  RESULT: {verdict}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
