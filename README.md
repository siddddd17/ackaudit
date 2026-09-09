# ackaudit — auditing PyTorch's activation-checkpointing knapsack solvers

**Status: in progress.** The central effect reproduces but shrinks substantially
on real architectures. See [Findings](#findings) and [Open issues](#open-issues)
before relying on anything here.

PyTorch's `min_cut_rematerialization_partition` decides which activations to
keep and which to recompute by solving a 0/1 knapsack: each recomputable node is
an item weighing its own tensor size, valued at the runtime it saves. Four
solvers ship in `torch/_functorch/_activation_checkpointing/knapsack.py` —
`greedy`, `ilp`, `dp`, and `dp_knapsack_sliding_hirschberg`.

This harness captures the knapsack instances real compiled models produce, runs
all four solvers across a budget sweep, and scores every resulting plan against
**two** objectives:

| objective | how it's computed | who uses it |
|---|---|---|
| **proxy** | `sum(size of saved nodes)` | what the solvers optimise |
| **true**  | simulated backward pass incl. recomputation chains | `account_for_backward_pass=True`, off by default |

Both come from PyTorch's own `KnapsackEvaluator`. The second is never exercised
by PyTorch's evaluation path — `evaluate_distribution_of_results_for_knapsack_algo`
doesn't pass the flag, so it defaults to `False`.

Graph capture is CPU-only: `torch.compile` traces and partitions without running
a kernel, so everything except the allocator validation runs on a laptop.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # add ,hf for the transformers models
```

Record your torch version — solver availability and partitioner behaviour differ
across releases, and `dp_knapsack_sliding_hirschberg` only exists from 2.10.

```bash
python -c "import torch; print(torch.__version__)"
```

## Running

```bash
# toy models: deep chain, branching, transformer block, conv stack
python scripts/run_audit.py --outdir out

# real architectures from transformers configs (random weights, no hub download)
python scripts/run_hf.py --outdir out_hf --scale 1

# inspect the tied plans in the cell with the worst true-peak spread
python scripts/inspect_cell.py --outdir out

# figures
python -c "from ackaudit.plots import make_all; make_all('out')"
```

`--scale` multiplies depth and sequence length. Scale 4 is slow on CPU.

## Findings

Numbers below are from a 4-layer Llama and four toy models, torch 2.14.0, CPU.
**Regenerate before citing** — runtime estimates are hardware-sensitive.

### The proxy optimum is a large set

On every toy cell and 9 of 10 Llama cells, **all four solvers land on the same
proxy objective value**. Greedy included. The ~7.4% gap between greedy and the
exact solvers reported in the literature does not appear in this data at all.

Those tied plans are not equivalent. On `deep_mlp` at budget 0.3, all four hit
proxy = 0.281081; `ilp` achieves a true peak of 0.281 while the other three
reach 0.519 — an 84.6% spread among plans the objective calls identical.

### Why: topological asymmetry the objective cannot see

`scripts/inspect_cell.py` on that cell shows the mechanism. The 24 items have
*identical* memory and runtime, so any 13 of 24 is optimal — roughly 2.5 million
optima. `greedy`/`dp`/`hirschberg` save the head of the chain (indices 0–12);
`ilp` saves the tail (11–23).

Saving the tail is much better: recomputing an early node is a short walk from
the input, while recomputing a late node forces every unsaved predecessor to be
live at once. The proxy counts only bytes saved, so all 2.5M optima look the
same to it.

**`ilp` is not smarter here — it is lucky.** Nothing in the formulation prefers
the tail; HiGHS happened to enumerate in an order that landed there.

### The effect shrinks on real models

| | toy models | llama (4L) |
|---|---|---|
| median proxy error | +31.6% | **+6.6%** |
| fraction over 10% | 61% | 42% |
| cells where all solvers tie | 100% | **90%** |
| median tied true-peak spread | — | **6.1%** |
| max tied spread | 84.6% | 60.4% |

Llama's items are not uniform (6 distinct memory values vs. 1 for the toys), so
the degeneracy is not purely an artifact of uniform toy layers — but the median
proxy error falls *below* the 7.4% figure it was being compared against.

The defensible claim is therefore narrow: **the proxy optimum is a large set on
real graphs, and its members differ by a median 6% and a maximum 60% in true
peak memory.** Not "the proxy is badly wrong."

### Input scale

The partitioner clamps `memory_budget` to `[0, 1]` (`partitioners.py` ~line 3502)
and weights arrive normalised via `get_normalized_size`. With `S = 10000`
hardcoded in the solver, quantised capacity `W` has a **structural ceiling of
10,000**. Observed W = 5000, largest `dp_knapsack` table 0.82 MB.

For comparison, the COLM 2026 paper introducing `dp_knapsack_sliding_hirschberg`
benchmarks at `W` between 1.4e8 and 3.8e8 and reports `dp_knapsack` OOMing at
n = 100 on 64 GB — four to five orders of magnitude above what this pipeline
produces. At realistic scale the new solver is roughly **2x slower** than
`dp_knapsack`, not 25–28% faster.

## Open issues

- **ViT captures zero knapsack instances.** Compiles fine, partitioner never
  reaches the solver. Either nothing is recomputable or it short-circuits. If
  real vision models never hit this path, that is a scope limit on the whole
  line of work, not just a bug.
- **BERT crashes** with `IndexError` in embedding lookup — likely a config
  mismatch in `hf_models.BertWrap`.
- **One real model is not a result.** Need ViT and BERT working, plus scale 2
  and 4, before the tie-set numbers mean anything.
- **The `q3_exactness` statistic in `analyze.py` is unsound** and should be
  removed. It reports which solver "wins", but with every cell fully tied the
  winner is decided by iteration order — it returned 48%, 55%, and 60% across
  runs on byte-identical data. Use `tie_sets()` instead, which is
  order-independent. Kept for now only so the old reports stay reproducible.
- **"True peak" is a simulation** — PyTorch's model of backward memory, not a
  measured allocator high-water mark. Validating against
  `torch.cuda.max_memory_allocated()` is the one step needing a GPU, and it
  decides whether any of this is real.
- **`aot_eager` backend only.** Fusion under `inductor` changes which nodes are
  recomputable and may change the picture.
- The normalisation claim was read off `main`. Verify `get_normalized_size` and
  check whether any path supplies unnormalised
  `recorded_knapsack_input_memories`.

## Layout

```
ackaudit/
├── audit.py       AuditingSolver (CustomKnapsackSolver hook), dual-objective sweep
├── capture.py     toy model zoo + torch.compile driver
├── hf_models.py   transformers architectures built from config
├── analyze.py     summary statistics, heterogeneity and tie-set diagnostics
└── plots.py       figures
scripts/
├── run_audit.py   toy sweep
├── run_hf.py      real-architecture sweep + degeneracy check
└── inspect_cell.py  dump tied plans for one (graph, budget) cell
```

The hook is `CustomKnapsackSolver`, a documented extension point — no fork
required. `AuditingSolver` records, audits, then delegates to `dp_knapsack` so
compilation proceeds normally.
