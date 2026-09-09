"""Inspect the tied plans in a single (graph, budget) cell.

results.json only records plan sizes, not the plans themselves, so this
re-captures the graph and re-solves to get the actual index lists.
"""

from __future__ import annotations

import argparse
import json
import glob
from collections import defaultdict

import torch
import torch._functorch.config as functorch_config
from torch._functorch._activation_checkpointing.graph_info_provider import (
    GraphInfoProvider,
)
from torch._functorch._activation_checkpointing.knapsack_evaluator import (
    KnapsackEvaluator,
)
from torch._functorch.partitioners import CustomKnapsackSolver

from ackaudit.audit import SOLVERS, _runtimes_for
from ackaudit.capture import MODELS


def worst_cell(outdir: str):
    """Find the (label, budget) cell with the largest spread among tied plans."""
    rows = []
    for f in sorted(glob.glob(f"{outdir}/*/results.json")):
        rows += json.load(open(f))
    cells = defaultdict(list)
    for r in rows:
        if r["ok"]:
            cells[(r["label"], r["budget"])].append(r)

    best = None
    for (label, budget), rs in cells.items():
        opt = min(x["proxy_peak_memory"] for x in rs)
        tied = [x for x in rs if abs(x["proxy_peak_memory"] - opt) < 1e-12]
        trues = [x["true_peak_memory"] for x in tied]
        if len(tied) < 2 or min(trues) <= 0:
            continue
        spread = (max(trues) - min(trues)) / min(trues)
        if best is None or spread > best[0]:
            best = (spread, label, budget, len(tied), opt)
    return best


class _Grabber(CustomKnapsackSolver):
    """Captures the live graph and solver inputs, then aborts the compile."""

    def __init__(self):
        self.payload = None

    def __call__(self, memory, joint_graph, max_memory, node_info, banned):
        self.payload = (list(memory), joint_graph, max_memory, list(banned))
        raise _Stop()

    def uuid(self):
        return None


class _Stop(Exception):
    pass


def grab(model_name: str):
    build, make_inputs = MODELS[model_name]
    g = _Grabber()
    prev_s = functorch_config.activation_memory_budget_solver
    prev_b = functorch_config.activation_memory_budget
    try:
        functorch_config.activation_memory_budget_solver = g
        functorch_config.activation_memory_budget = 0.5
        torch._dynamo.reset()
        try:
            compiled = torch.compile(build(), backend="aot_eager", dynamic=False)
            compiled(*make_inputs()).backward()
        except Exception:
            pass  # _Stop propagates wrapped; payload is what we want
    finally:
        functorch_config.activation_memory_budget_solver = prev_s
        functorch_config.activation_memory_budget = prev_b
    if g.payload is None:
        raise RuntimeError(f"failed to capture graph for {model_name}")
    return g.payload


def inspect(model_name: str, budget: float) -> None:
    memories, joint_graph, _, banned = grab(model_name)
    runtimes = _runtimes_for(banned)
    names = [n.name for n in banned]
    targets = [str(n.target) for n in banned]

    provider = GraphInfoProvider.inialize_from_graph(
        joint_graph=joint_graph,
        all_recomputable_banned_nodes=banned,
        recorded_knapsack_input_memories=memories,
        recorded_knapsack_input_runtimes=runtimes,
    )
    ev = KnapsackEvaluator(graph_info_provider=provider)

    plans = {}
    for name, fn in SOLVERS.items():
        _, saved, recomp = fn(memories, runtimes, budget)
        proxy = ev.evaluate_knapsack_output(saved, recomp, account_for_backward_pass=False)
        true = ev.evaluate_knapsack_output(saved, recomp, account_for_backward_pass=True)
        plans[name] = {
            "saved": sorted(saved),
            "recomp": sorted(recomp),
            "proxy": proxy["peak_memory"],
            "true": true["peak_memory"],
        }

    print(f"\n{model_name}  budget={budget}  n={len(memories)}")
    print("=" * 78)
    for name, p in plans.items():
        print(f"{name:24s} proxy={p['proxy']:.6f}  true={p['true']:.6f}  "
              f"|saved|={len(p['saved'])}")

    opt = min(p["proxy"] for p in plans.values())
    tied = {k: v for k, v in plans.items() if abs(v["proxy"] - opt) < 1e-12}
    print(f"\ntied at proxy optimum {opt:.6f}: {sorted(tied)}")
    if len(tied) < 2:
        return

    lo = min(tied, key=lambda k: tied[k]["true"])
    hi = max(tied, key=lambda k: tied[k]["true"])
    if lo == hi:
        print("all tied plans have identical true peak")
        return

    a, b = tied[lo], tied[hi]
    print(f"spread: {lo} true={a['true']:.6f}  vs  {hi} true={b['true']:.6f}  "
          f"({(b['true']-a['true'])/a['true']*100:+.1f}%)")

    sa, sb = set(a["saved"]), set(b["saved"])
    print(f"\nidentical saved sets: {sa == sb}")
    print(f"{'idx':>4} {'node':28s} {'target':26s} {'mem':>10s} {lo[:8]:>8s} {hi[:8]:>8s}")
    for i in sorted(sa ^ sb) or sorted(sa | sb):
        print(f"{i:4d} {names[i][:28]:28s} {targets[i][-26:]:26s} "
              f"{memories[i]:10.6f} {'save' if i in sa else '  --':>8s} "
              f"{'save' if i in sb else '  --':>8s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--model")
    ap.add_argument("--budget", type=float)
    args = ap.parse_args()

    if args.model is None:
        found = worst_cell(args.outdir)
        if found is None:
            raise SystemExit("no tied cells found")
        spread, label, budget, ntied, opt = found
        print(f"worst tied spread: {spread*100:.1f}%  in {label} at budget={budget} "
              f"({ntied} solvers tied at proxy={opt:.6f})")
        args.model, args.budget = label.split("#")[0], budget

    inspect(args.model, args.budget)
