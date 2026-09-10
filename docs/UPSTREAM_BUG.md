# Upstream: schedule-dependent and nondeterministic backward-memory simulation

Status: verified locally on torch 2.14.0. Not yet reported. Draft for one or two
PyTorch issues.

Two separable problems. The first is a straightforward reproducibility bug. The
second is a semantics question that is more interesting and harder to dismiss.

---

## Bug 1: `evaluate_knapsack_output` is nondeterministic across processes

`KnapsackEvaluator.evaluate_knapsack_output(..., account_for_backward_pass=True)`
returns different peak-memory values across Python processes for an identical
graph, an identical saved-node set, and an identical solver plan.

### Chain of causation

```
Python set iteration order
  -> networkx node insertion order
  -> Kahn tie resolution in topological_sort
  -> reverse topological order
  -> simulated backward schedule
  -> reported peak memory
```

Two sites, both in
`torch/_functorch/_activation_checkpointing/`:

`graph_info_provider.py`, lines 184 and 196:

```python
all_recomputable_banned_nodes_set = set(self.all_recomputable_banned_nodes)
...
candidate_graph.add_nodes_from(all_recomputable_banned_nodes_set)
```

`knapsack_evaluator.py`, in
`_get_backward_memory_from_topologically_sorted_graph`:

```python
sorted_nodes = list(reversed(list(nx.topological_sort(node_graph))))
```

`nx.topological_sort` returns one of many valid orders for a DAG; networkx
documents the non-uniqueness explicitly and provides
`lexicographical_topological_sort` for callers who need a stable choice. Which
order comes back depends on insertion order, which here comes from a set.

### Reproduction

A 6-branch network, 31 recomputable nodes, three budgets, four solvers, three
separate processes. Values are simulated peak memory:

```
run 1:  0.8387 0.8387 0.8387 0.8387 0.8387 0.8065 0.8387 0.8387 0.8710 ...
run 2:  0.8387 0.8387 0.8387 0.8387 0.8387 0.8710 0.8387 0.8387 0.8710 ...
run 3:  0.8387 0.8065 0.8387 0.8387 0.8710 0.7742 0.8710 0.8710 1.0000 ...
```

Across a full sweep of four synthetic graphs plus a 4-layer Llama, 21 of 240
results differed between two identical runs.

**The isolating observation:** every one of those 21 differed *only* in
`true_peak_memory`. Plan size and the proxy objective were identical, and the
solver inputs (node list, sizes, runtimes) hashed identically across four
processes. Since `greedy` and `dp` are deterministic algorithms on deterministic
inputs, their plans cannot have changed, so the variance is in the simulator.

Setting `PYTHONHASHSEED=0` removes the variance (0/40 differing versus 12/40
unset). This is supporting evidence rather than proof on its own, since the hash
seed touches many order-sensitive operations. The plan-identity result above is
the stronger argument.

### Proposed fix

```diff
- sorted_nodes = list(reversed(list(nx.topological_sort(node_graph))))
+ sorted_nodes = list(
+     reversed(list(nx.lexicographical_topological_sort(node_graph)))
+ )
```

Node names are strings, so the sort key is well defined. Verified: three
consecutive processes produced byte-identical peaks with this change and varying
peaks without it.

Note that this is sufficient on its own. Lexicographic order is determined by
node names regardless of insertion order, so it fixes determinism even with the
set-based construction left in place. Making `graph_info_provider.py` use an
ordered container is still worth doing as hygiene, but the fix does not depend
on it.

This is a determinism fix only. It makes no claim about which schedule is right.

---

## Bug 2: the simulator silently chooses a schedule

The function's docstring says it simulates the backward pass. What it actually
does is simulate **one arbitrarily chosen reverse-topological schedule**.

A topological order guarantees dependency correctness. It does not define a
unique execution schedule, and peak memory is schedule-dependent. On the same
branching graph at budget 0.3:

| schedule | simulated peak |
|---|---|
| `nx.topological_sort` (current) | 0.8387 |
| `nx.lexicographical_topological_sort` | 0.3548 |

Both are valid topological orders of the same DAG with the same saved-node set.

The consequence is that `true_peak_memory` is not a property of the checkpointing
strategy alone. It is a property of *(checkpointing strategy, backward schedule)*,
and the second component is currently unspecified and undocumented.

This does **not** mean the lower number is the correct one. It means the
evaluator is answering a question it has not defined.

### What would settle it

Three schedules:

- **A.** current `nx.topological_sort`
- **B.** `nx.lexicographical_topological_sort`
- **C.** `reversed(graph_info_provider.graph_nodes_in_order)`, the reverse FX
  graph order

Across graph families that isolate the structural variable: pure chain, single
fork, single join, fork-join, nested branches, the 6-branch net, and a real
transformer. Report schedule, peak, and the number of distinct peaks per graph.

C is available: `GraphInfoProvider` stores `graph_nodes_in_order`, built at line
80 as `[node.name for node in joint_graph.nodes]`.

**The decisive experiment is not "which schedule gives the smallest peak."** It
is whether any of them matches the memory behaviour of a real backward pass
measured with `torch.cuda.max_memory_allocated()`. Until that is done, reverse FX
order should not be described as the true backward order. The FX graph is stored
in forward topological order, and autograd's engine has its own scheduling
semantics; the two are not known to coincide.

---

## Caveats before filing

- Verified on torch 2.14.0. Confirm the code is unchanged on `main`.
- Both paths are off by default (`account_for_backward_pass=False`), so
  production partitioning is unaffected today. That lowers severity and may lower
  maintainer interest, but it also means a fix carries almost no regression risk.
- The schedule comparison currently rests on one graph family. Run the full A/B/C
  matrix before putting numbers in the issue.

## Consequence for this project

Every `true_peak_memory` result here was produced under an unpinned schedule, so
all of them are potentially schedule-dependent and need regenerating once a
schedule is fixed. Branching graphs vary across processes; the `deep_mlp` chain
does not, because a linear chain has essentially one topological order and no
ties to resolve. That is why the 84.6% `deep_mlp` result reproduced exactly across
two machines and two operating systems while the branching results did not.

Pin `PYTHONHASHSEED` in the harness and choose an explicit schedule before
regenerating anything.
