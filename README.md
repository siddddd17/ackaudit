# ackaudit — auditing PyTorch's activation-checkpointing knapsack solvers

PyTorch's `min_cut_rematerialization_partition` decides which activations to keep
and which to recompute by solving a 0/1 knapsack: each recomputable node is an
item weighing its own tensor size, valued at the runtime it saves. Four solvers
ship in `torch/_functorch/_activation_checkpointing/knapsack.py` — `greedy`,
`ilp`, `dp`, and `dp_knapsack_sliding_hirschberg`.

This harness captures the knapsack instances that real compiled models actually
produce, runs all four solvers across a budget sweep, and scores every resulting
plan against **two** objectives:

| objective | how it's computed | who uses it |
|---|---|---|
| **proxy** | `sum(size of saved nodes)` | what the solvers optimise |
| **true**  | simulated backward pass incl. recomputation chains | `account_for_backward_pass=True`, off by default |

Both come from PyTorch's own `KnapsackEvaluator`. The second is never exercised
by PyTorch's evaluation path — `evaluate_distribution_of_results_for_knapsack_algo`
doesn't pass the flag, so it defaults to `False` everywhere.

Everything runs on CPU. `torch.compile` traces and partitions without executing a
kernel, so graph capture needs no accelerator.

## Install & run

```bash
pip install torch networkx scipy matplotlib
python scripts/run_audit.py --outdir out
python -c "from ackaudit.plots import make_all; make_all('out')"
```

## Preliminary results

Four model families (deep MLP, wide branching net, transformer blocks, conv
stack), 10 budgets, 4 solvers — 160 measurements, torch 2.14.0, CPU only.

**Q1 — input scale.** The partitioner rejects any `memory_budget` outside
`[0, 1]` (`partitioners.py`, line ~3502) and item weights arrive normalised via
`get_normalized_size`. With the solver's hardcoded `S = 10000`, the quantised
capacity `W` has a **structural ceiling of 10,000**. Observed: `W = 5000`,
`n = 11–31`, largest `dp_knapsack` table **0.64 MB**.

For comparison, the COLM 2026 paper introducing `dp_knapsack_sliding_hirschberg`
benchmarks at `W` between 1.4×10⁸ and 3.8×10⁸ and reports `dp_knapsack` OOMing at
`n = 100` on 64 GB. That is four to five orders of magnitude above what this
pipeline can produce. **The motivating OOM may not occur on real inputs.**

Relatedly, at realistic scale the new solver is roughly **2× slower** than
`dp_knapsack` (39 ms vs 21 ms median), not 25–28% faster: the extra `O(log n)`
divide-and-conquer passes cost more than the memory savings buy when `W` is
small.

**Q2 — proxy error.** Median understatement of peak memory **+31.6%**, mean
+211%, p90 +733%, max +2500%. 59% of plans understate by more than 10%. A user
who sets `memory_budget=0.3` can get a true backward peak of 0.52.

**Q3 — does exactness survive?** In **55%** of (graph, budget) cells, the solver
that wins on the proxy objective is *not* the one with the lowest true peak.
Among plans that **tie exactly at the proxy optimum**, true peaks spread by up to
**84.6%**. The published gap between greedy and the exact solvers is 7.4% — an
order of magnitude smaller than the noise the proxy introduces.

That is the whole result in one sentence: *the field is solving the wrong
objective exactly.*

## What this is not (yet)

Read these before building on any of it.

- **`n` is small.** 11–31 recomputable nodes. Real LLM graphs have hundreds to
  thousands. Everything here must be re-run on serious models (HF checkpoints,
  long sequence lengths) before it means anything. It is entirely possible the
  effect shrinks with scale.
- **The models are toys I wrote**, chosen to vary graph shape, not to represent
  production workloads.
- **`aot_eager` backend**, not `inductor`. Fusion changes which nodes are
  recomputable and may change the picture substantially.
- **"True peak" is still a simulation** — PyTorch's own model of backward memory,
  not a measured allocator high-water mark. Validating it against real
  `torch.cuda.max_memory_allocated()` on a handful of configs is the one step
  that genuinely needs a GPU, and it is the step that decides whether any of this
  is real.
- **The obvious objection**: a skeptic will say the two numbers measure different
  things, so of course they differ. The defensible framing is not "the evaluator
  disagrees with itself" — it's that **the budget the user sets does not bound
  the memory they get**, and the solver has no mechanism to notice.
- **torch 2.14.0 only.** Behaviour differs across versions; the COLM paper
  targets 2.10.
- Q1's normalisation claim was read off `main`. Verify `get_normalized_size` and
  check whether any call path supplies unnormalised
  `recorded_knapsack_input_memories` before relying on it.

## Layout

```
ackaudit/
├── audit.py      AuditingSolver (CustomKnapsackSolver hook), dual-objective sweep
├── capture.py    model zoo + torch.compile driver
├── analyze.py    Q1/Q2/Q3 statistics and text report
└── plots.py      three figures
run_audit.py      CLI
```

The hook is `CustomKnapsackSolver`, a documented extension point — no fork
required. `AuditingSolver` records, audits, then delegates to `dp_knapsack` so

