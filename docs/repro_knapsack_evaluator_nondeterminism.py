"""
Repro: KnapsackEvaluator.evaluate_knapsack_output(account_for_backward_pass=True)
returns different peak-memory values across processes for identical inputs.

    python repro_knapsack_evaluator_nondeterminism.py

Spawns N subprocesses. Each compiles the same tiny model, intercepts the
partitioner at its knapsack call, builds a GraphInfoProvider from the captured
joint graph, and evaluates one FIXED, hardcoded saved-node set. No solver is
involved in the measurement, so any variance is in the evaluator.

Exits non-zero if the peaks disagree.
"""

import hashlib
import subprocess
import sys

N_PROCS = 8


def child():
    import torch
    import torch.nn as nn
    import torch._functorch.config as fc
    from torch._functorch.partitioners import CustomKnapsackSolver, dp_knapsack
    from torch._functorch._activation_checkpointing.graph_info_provider import (
        GraphInfoProvider,
    )
    from torch._functorch._activation_checkpointing.knapsack_evaluator import (
        KnapsackEvaluator,
    )

    # Parallel branches give the joint DAG many valid topological orders.
    # A pure chain has essentially one and will not reproduce this.
    class Branching(nn.Module):
        def __init__(self, width=192, branches=6, depth=5):
            super().__init__()
            self.stem = nn.Linear(width, width)
            self.branches = nn.ModuleList(
                nn.Sequential(*[nn.Sequential(nn.Linear(width, width), nn.Tanh())
                                for _ in range(depth)])
                for _ in range(branches)
            )
            self.head = nn.Linear(width, width)

        def forward(self, x):
            h = self.stem(x)
            return self.head(sum(b(h) for b in self.branches)).sum()

    captured = {}

    class Probe(CustomKnapsackSolver):
        def __call__(self, memory, joint_graph, max_memory, node_info, banned):
            if "peak" not in captured:
                # fingerprint the graph so the report can show it is byte-identical
                # across processes while the reported peak is not
                names = [n.name for n in joint_graph.nodes]
                edges = sorted((u.name, v.name) for u in joint_graph.nodes for v in u.users)
                captured["hash"] = hashlib.sha256(
                    repr((names, edges)).encode()
                ).hexdigest()[:16]
                runtimes = [1.0] * len(memory)
                provider = GraphInfoProvider.inialize_from_graph(
                    joint_graph=joint_graph,
                    all_recomputable_banned_nodes=banned,
                    recorded_knapsack_input_memories=list(memory),
                    recorded_knapsack_input_runtimes=runtimes,
                )
                ev = KnapsackEvaluator(graph_info_provider=provider)
                # Fixed evaluator input: nodes chosen directly by index.
                # No solver output is used in the measured result.
                saved = list(range(0, len(memory), 2))
                recomp = [i for i in range(len(memory)) if i % 2]
                captured["n"] = len(memory)
                captured["peak"] = ev.evaluate_knapsack_output(
                    saved_nodes_idxs=saved,
                    recomputable_node_idxs=recomp,
                    account_for_backward_pass=True,
                )["peak_memory"]
            _, s, r = dp_knapsack(memory, [1.0] * len(memory), max_memory)
            return s, r

        def uuid(self):
            return None

    fc.activation_memory_budget_solver = Probe()
    fc.activation_memory_budget = 0.5
    torch._dynamo.reset()
    compiled = torch.compile(Branching(), backend="aot_eager", dynamic=False)
    compiled(torch.randn(32, 192, requires_grad=True)).backward()

    print(f"{captured['n']} {captured['hash']} {captured['peak']:.10f}")


def parent():
    import torch

    print(f"torch {torch.__version__}, {N_PROCS} processes")
    print("same model, same graph, same hardcoded saved-node set\n")

    peaks = []
    hashes = []
    for i in range(N_PROCS):
        out = subprocess.run(
            [sys.executable, __file__, "--child"], capture_output=True, text=True
        )
        if out.returncode != 0:
            print(out.stderr[-2000:])
            raise SystemExit("child failed")
        n, ghash, peak = out.stdout.strip().split()
        peaks.append(peak)
        hashes.append(ghash)
        print(f"  process {i}: n={n}  graph_sha256={ghash}  peak_memory = {peak}")

    print(f"\ndistinct graph hashes: {len(set(hashes))}")
    distinct = sorted(set(peaks))
    print(f"\ndistinct values: {len(distinct)}  ->  {distinct}")
    if len(distinct) > 1:
        print("\nFAIL: identical inputs produced different peak memory")
        return 1
    print("\nno variance in this run; try rerunning or raising N_PROCS")
    return 0


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    else:
        raise SystemExit(parent())
