"""Falsification harness for the saved-node accounting bug (#196914).

Distinct from docs/repro_knapsack_evaluator_saved_predecessor.py: that script
prints a trace and reports the build state. This one exits non-zero on any
affected build. Its oracle is not a pinned expected value but the general
invariant that the returned trace must end at 0.0 for any graph, memories,
and saved-node set. Three saved sets are exercised, including the {node1}
control which upstream's own tests cover and which is unaffected by the bug.

Uses the REAL KnapsackEvaluator / GraphInfoProvider API. No torch.compile, no
model, no FX graph -- the graph is the exact fixture from PyTorch's own
test/functorch/test_ac_knapsack.py::TestKnapsackEvaluator.setUp.

Upstream's test_recomputable_node_only_graph_with_larger_graph_context asserts
the candidate graph for this fixture is exactly:  node1 -> node2 -> node5.
Upstream's test_get_backward_memory_from_topologically_sorted_graph covers
saved={node1} only. It does not cover a saved node sitting at BFS depth 1 of a
recomputation walk, which is the case below.

    python -m experiments.backward_memory_accounting.saved_node_recomputation

Exits non-zero if any trace fails to return to zero.
"""

import sys

from torch._functorch._activation_checkpointing.graph_info_provider import (
    GraphInfoProvider,
)
from torch._functorch._activation_checkpointing.knapsack_evaluator import (
    KnapsackEvaluator,
)

provider = GraphInfoProvider(
    graph_nodes_in_order=["node1", "node2", "node3", "node4", "node5", "output"],
    graph_edges=[
        ("node1", "node2"),
        ("node2", "node3"),
        ("node3", "node4"),
        ("node4", "node5"),
        ("node5", "output"),
        ("node1", "output"),
    ],
    all_recomputable_banned_nodes=["node1", "node2", "node5"],
    recorded_knapsack_input_memories=[0.1, 0.2, 0.2],
    recorded_knapsack_input_runtimes=[100.0, 50.0, 51.0],
)
ev = KnapsackEvaluator(graph_info_provider=provider)
names = provider.all_recomputable_banned_nodes

print(
    "candidate graph edges:",
    sorted(provider.recomputable_node_only_graph_with_larger_graph_context.edges),
)
print("non_ac_peak_memory  :", provider.get_non_ac_peak_memory())
print()

failures = 0
for saved_idxs in ([0], [1], [0, 1]):
    recomp_idxs = [i for i in range(3) if i not in saved_idxs]
    saved_set = {names[i] for i in saved_idxs}

    trace = ev._get_backward_memory_from_topologically_sorted_graph(
        node_graph=provider.recomputable_node_only_graph_with_larger_graph_context,
        node_memories=provider.all_node_memories,
        saved_nodes_set=saved_set,
        peak_memory_after_forward_pass=sum(
            provider.all_node_memories[n] for n in saved_set
        ),
    )
    out = ev.evaluate_knapsack_output(
        saved_nodes_idxs=saved_idxs,
        recomputable_node_idxs=recomp_idxs,
        account_for_backward_pass=True,
    )
    final = trace[-1][0]
    bad = abs(final) > 1e-9
    failures += bad

    print(f"saved={sorted(saved_set)}")
    print(f"  peak_memory                          = {out['peak_memory']:.4f}")
    print(
        f"  percentage_of_theoretical_peak_memory= "
        f"{out['percentage_of_theoretical_peak_memory']:.4f}"
    )
    print(
        f"  trace ends at                        = {final:+.4f}"
        f"{'   <-- memory not returned to zero' if bad else ''}"
    )
    for value, label in trace:
        print(f"      {value:+.4f}  {label}")
    print()

print("Expected if the depth-1 predecessor filter is the defect:")
print("  saved={node1}          peak 0.5, trace ends at 0.0  (the covered case)")
print("  saved={node2}          peak 0.7, trace ends at 0.2")
print("  saved={node1,node2}    peak 0.7, trace ends at 0.2")
print("  and percentage_of_theoretical_peak_memory = 1.4 > 1, i.e. the simulator")
print("  reports checkpointing using more memory than saving every tensor.")
print()
print("Candidate fix, knapsack_evaluator.py, initial predecessor_queue only:")
print("      if dependency not in already_computed")
print("      and dependency not in saved_nodes_set          # <-- add")
print()
print("Proposed regression test: for any graph / memories / saved set, the final")
print("entry of the returned trace must be 0 within tolerance.")

sys.exit(1 if failures else 0)
