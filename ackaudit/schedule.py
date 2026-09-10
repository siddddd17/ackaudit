"""Explicit control over the schedule the backward-memory simulator uses.

PyTorch's evaluator linearises the graph with `nx.topological_sort`, which is
non-unique and whose tie-breaking depends on node insertion order. Insertion
order comes from a Python set, so the simulated peak varies across processes.
Reported upstream; see docs/UPSTREAM_BUG.md.

Until that is settled, every measurement here must name the schedule it used.
Three are available:

    A  "default"  what PyTorch currently does. Nondeterministic. Kept only so
                  the bug stays reproducible from this repo.
    B  "lexicographic"  ties broken by node name. Deterministic but arbitrary
                  ("addmm_10" sorts before "addmm_2").
    C  "fx_order"  ties broken by position in the FX graph. Deterministic, and
                  matches the convention torch.fx.passes.tools_common.
                  stable_topological_sort uses. This is the default here.

None of these is known to match the order autograd actually executes. That is
the open question, and answering it needs a measured comparison against
torch.cuda.max_memory_allocated().
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import torch._functorch._activation_checkpointing.knapsack_evaluator as _ke

SCHEDULES = ("default", "lexicographic", "fx_order")
DEFAULT_SCHEDULE = "fx_order"


class Schedule:
    """Context manager that pins the simulator's topological order.

    Patches `nx.topological_sort` inside the evaluator's module namespace for
    the duration of the block. Not thread-safe; evaluation here is synchronous.
    """

    def __init__(self, mode: str = DEFAULT_SCHEDULE, node_order: Optional[list[str]] = None):
        if mode not in SCHEDULES:
            raise ValueError(f"unknown schedule {mode!r}, expected one of {SCHEDULES}")
        if mode == "fx_order" and not node_order:
            raise ValueError("fx_order needs node_order (provider.graph_nodes_in_order)")
        self.mode = mode
        self.node_order = node_order
        self._orig = None

    def __enter__(self) -> "Schedule":
        self._orig = _ke.nx.topological_sort
        if self.mode == "default":
            return self
        if self.mode == "lexicographic":
            _ke.nx.topological_sort = lambda G, *a, **k: nx.lexicographical_topological_sort(G)
        else:
            rank = {name: i for i, name in enumerate(self.node_order)}
            _ke.nx.topological_sort = lambda G, *a, **k: nx.lexicographical_topological_sort(
                G, key=lambda n: rank[n]
            )
        return self

    def __exit__(self, *exc) -> None:
        if self._orig is not None:
            _ke.nx.topological_sort = self._orig
