# Upstream: schedule-dependent and nondeterministic backward-memory simulation

| bug | what | status |
|---|---|---|
| 1 | `evaluate_knapsack_output` returns different peaks across processes for identical inputs | filed as pytorch/pytorch#196512, PR pytorch/pytorch#196872 open |
| 2 | the simulator silently chooses one of many valid schedules and calls the result "the" backward memory | not filed; blocked on the measured comparison described below |

Bug 1 is a symptom of Bug 2. Pinning the topological order fixes
reproducibility without establishing that the pinned order matches the schedule
autograd actually runs.

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

### Measured spread

Seven synthetic families, eight budgets each, 56 cells total. Full table in
`results/schedule_sensitivity/schedule_report.txt`.

| schedule | mean simulated peak | cells where uniquely lowest |
|---|---|---|
| A default `nx.topological_sort` | 0.5519 | 0 |
| B `nx.lexicographical_topological_sort` | 0.4599 | 20 |
| C reverse FX order | 0.4966 | 1 |

Per budget (mean peak across the seven families):

| budget | A | B | C |
|---|---|---|---|
| 0.20 | 0.4947 | 0.3151 | 0.3485 |
| 0.50 | 0.5773 | 0.4917 | 0.5813 |
| 0.80 | 0.6048 | 0.6048 | 0.6048 |

Every peak on every row is a valid answer to "simulate the backward pass"; the
DAGs and saved-node sets are identical. A is never the uniquely lowest and has
the highest mean, so if any schedule is worth removing on grounds of pessimism
alone it is the current default. This is not the same as saying B or C is
right. Lower is not more correct: only a measured comparison against real
backward memory can call one of these correct, and that comparison is not in
this repo.

**The decisive experiment is not "which schedule gives the smallest peak."** It
is whether any of them matches the memory behaviour of a real backward pass
measured with `torch.cuda.max_memory_allocated()`. Until that is done, reverse FX
order should not be described as the true backward order. The FX graph is stored
in forward topological order, and autograd's engine has its own scheduling
semantics; the two are not known to coincide.

### Regenerating

```
python -m experiments.schedule_sensitivity.run_schedules
```

Writes `results/schedule_sensitivity/schedule_report.txt` and
`schedules.json` under the same directory. The three schedules are defined in
`experiments/schedule_sensitivity/schedules.py`; graph families in
`experiments/schedule_sensitivity/graph_families.py`. Do not confuse either
with `ackaudit/schedule.py` (singular), the small context manager that pins
the evaluator's topological sort.

---

## Caveats before filing

- Verified on torch 2.14.0. Confirm the code is unchanged on `main`.
- Both paths are off by default (`account_for_backward_pass=False`), so
  production partitioning is unaffected today. That lowers severity and may lower
  maintainer interest, but it also means a fix carries almost no regression risk.
- The schedule comparison in the Measured spread table above covers seven
  synthetic families. It has not been run against a real transformer.

## Consequence for this project

Every `true_peak_memory` result here was produced under an unpinned schedule, so
all of them are potentially schedule-dependent and need regenerating once a
schedule is fixed. Branching graphs vary across processes; the `deep_mlp` chain
does not, because a linear chain has essentially one topological order and no
ties to resolve. That is why the 84.6% `deep_mlp` result reproduced exactly across
two machines and two operating systems while the branching results did not.

Pin `PYTHONHASHSEED` in the harness and choose an explicit schedule before
regenerating anything.
