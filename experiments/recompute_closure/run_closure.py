"""How much must be live to recompute one node?

`results/measured_backward/` records that llama's measured backward peak is
lowest at budget 0.15 and rises below it, while bert's is monotonic. That
section states the mechanism is not established. This measures the candidate
that fits: how far back a recomputation has to walk before it reaches saved
activations.

For a recomputed node n, materialising n during backward requires every
predecessor that is not itself saved; each of those requires its own unsaved
predecessors, and so on. The closure of that walk is the set of activations the
recomputation of n depends on. The knapsack weighs each item by its own tensor
size alone, so the size of this closure is invisible to the objective the solver
optimises.

The closure is not a live set. A schedule may free an intermediate once its
consumer has run, so the memory required lies between the largest single member
(a lower bound, since that member must be materialised at some point) and the
sum of all members (the upper bound, and what PyTorch's own backward simulator
assumes). Both are reported.

CPU only. Nothing is executed on an accelerator; the model is traced and
partitioned, the solver hook records the plan, and the closures are computed
from the candidate graph.

    python -m experiments.recompute_closure.run_closure --model llama
    python -m experiments.recompute_closure.run_closure --model bert --budgets 0.05 0.15 0.30
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import torch
from torch._functorch import config as functorch_config
from torch._functorch._activation_checkpointing.graph_info_provider import (
    GraphInfoProvider,
)
from torch._functorch._activation_checkpointing.knapsack import dp_knapsack
from torch._functorch.partitioners import CustomKnapsackSolver

from ackaudit.audit import RuntimeRecorder
from ackaudit.capture import _resolve

DEFAULT_BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]


def recompute_closure(graph: nx.DiGraph, node: str, saved: set[str]) -> set[str]:
    """Unsaved nodes that must be materialised to produce `node`.

    Walks predecessors, stopping at saved nodes because their activations are
    already resident. `node` itself is excluded. This is the traversal
    `_get_backward_memory_from_topologically_sorted_graph` performs, with the
    saved-node boundary that pytorch/pytorch#196914 restores.
    """
    if node not in graph:
        return set()
    seen: set[str] = set()
    queue = deque(p for p, _ in graph.in_edges(node) if p not in saved)
    while queue:
        dep = queue.popleft()
        if dep in seen:
            continue
        seen.add(dep)
        for pred, _ in graph.in_edges(dep):
            if pred not in saved and pred not in seen:
                queue.append(pred)
    return seen


@dataclass
class Cell:
    model: str
    budget: float
    scale: int
    torch_version: str
    device: str = "cpu"

    n_items: int = 0
    n_saved: int = 0
    n_recomputed: int = 0
    solver_called: bool = False
    error: str = ""

    # |closure| per recomputed node
    sizes: list[int] = field(default_factory=list)
    # summed node memory of each closure, in the solver's own units: the upper
    # bound on what the recomputation needs, and what PyTorch's simulator uses
    weights: list[float] = field(default_factory=list)
    # largest single member of each closure: the lower bound, since that member
    # has to be materialised at some point whatever the schedule
    maxima: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        if not self.sizes:
            return {}
        s = sorted(self.sizes)
        w = sorted(self.weights)
        m = sorted(self.maxima)
        return {
            "size_median": statistics.median(s),
            "size_p90": s[int(0.9 * (len(s) - 1))],
            "size_max": s[-1],
            "size_mean": statistics.fmean(s),
            "weight_median": statistics.median(w),
            "weight_p90": w[int(0.9 * (len(w) - 1))],
            "weight_max": w[-1],
            "lower_median": statistics.median(m),
            "lower_p90": m[int(0.9 * (len(m) - 1))],
            "lower_max": m[-1],
        }


class ClosureSolver(CustomKnapsackSolver):
    """Runs the production solver, then measures the plan it produced."""

    def __init__(self, recorder: RuntimeRecorder, cell: Cell) -> None:
        self.recorder = recorder
        self.cell = cell

    def __call__(
        self,
        memory: list[float],
        joint_graph: torch.fx.Graph,
        max_memory: float,
        node_info: Any,
        all_recomputable_banned_nodes: list[torch.fx.Node],
    ) -> tuple[list[int], list[int]]:
        runtimes = self.recorder.lookup(all_recomputable_banned_nodes)
        _value, saved, recomputed = dp_knapsack(memory, runtimes, max_memory)

        c = self.cell
        c.solver_called = True
        c.n_items = len(memory)
        c.n_saved = len(saved)
        c.n_recomputed = len(recomputed)

        try:
            provider = GraphInfoProvider.inialize_from_graph(
                joint_graph=joint_graph,
                all_recomputable_banned_nodes=all_recomputable_banned_nodes,
                recorded_knapsack_input_memories=memory,
                recorded_knapsack_input_runtimes=runtimes,
            )
            graph = provider.recomputable_node_only_graph_with_larger_graph_context
            memories = provider.all_node_memories
            names = provider.all_recomputable_banned_nodes
            saved_names = {names[i] for i in saved}

            for i in recomputed:
                closure = recompute_closure(graph, names[i], saved_names)
                c.sizes.append(len(closure))
                c.weights.append(sum(memories[n] for n in closure))
                c.maxima.append(max((memories[n] for n in closure), default=0.0))
        except Exception as exc:  # noqa: BLE001 -- record and keep sweeping
            c.error = f"{type(exc).__name__}: {exc}"

        return saved, recomputed

    def uuid(self) -> Any:
        return None


def run_one(model_name: str, budget: float, scale: int, device: str = "cpu") -> Cell:
    cell = Cell(
        model=model_name,
        budget=budget,
        scale=scale,
        torch_version=torch.__version__,
        device=device,
    )
    build, make_inputs = _resolve(model_name, scale=scale)
    model = build().to(device)
    args = tuple(a.to(device) for a in make_inputs())

    recorder = RuntimeRecorder()
    prev_solver = functorch_config.activation_memory_budget_solver
    prev_budget = functorch_config.activation_memory_budget
    try:
        functorch_config.activation_memory_budget_solver = ClosureSolver(recorder, cell)
        functorch_config.activation_memory_budget = budget
        torch._dynamo.reset()
        with recorder:
            torch.compile(model, backend="aot_eager", dynamic=False)(*args).backward()
    except Exception as exc:  # noqa: BLE001
        cell.error = cell.error or f"{type(exc).__name__}: {exc}"
    finally:
        functorch_config.activation_memory_budget_solver = prev_solver
        functorch_config.activation_memory_budget = prev_budget

    return cell


def report(cells: list[Cell]) -> str:
    ok = [c for c in cells if c.sizes]
    if not ok:
        return "no cells produced a plan; check errors in the json\n"

    out = [
        f"RECOMPUTE CLOSURE  {ok[0].model} scale={ok[0].scale}  torch {ok[0].torch_version}",
        "",
        "For each recomputed node, the number of unsaved activations that must be",
        "materialised to produce it. The knapsack weighs items by their own size",
        "only, so these counts do not enter the objective.",
        "",
        f"device={ok[0].device}  torch {ok[0].torch_version}",
        f"{'budget':>7} {'saved':>6} {'recomp':>7} {'median':>7} {'p90':>6} {'max':>6} "
        f"{'lo med':>8} {'lo max':>8} {'hi med':>8} {'hi max':>8}",
    ]
    for c in sorted(ok, key=lambda c: c.budget):
        s = c.summary()
        out.append(
            f"{c.budget:>7.2f} {c.n_saved:>6} {c.n_recomputed:>7} "
            f"{s['size_median']:>7.1f} {s['size_p90']:>6.0f} {s['size_max']:>6.0f} "
            f"{s['lower_median']:>8.4f} {s['lower_max']:>8.4f} "
            f"{s['weight_median']:>8.4f} {s['weight_max']:>8.4f}"
        )
    out.append("")
    out.append("size columns are node counts. lo is the largest single member of the")
    out.append(
        "closure, a lower bound on what the recomputation needs; hi is the sum of"
    )
    out.append(
        "all members, the upper bound and what PyTorch's simulator assumes. Both"
    )
    out.append("are in the solver's own memory units.")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama")
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--budgets", type=float, nargs="*", default=DEFAULT_BUDGETS)
    ap.add_argument("--outdir", default="results/recompute_closure")
    ap.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="the candidate node set differs between devices on some models; "
        "pass the device used for the measurement being explained",
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda requested but torch.cuda.is_available() is False"
        )

    cells = []
    for budget in args.budgets:
        c = run_one(args.model, budget, args.scale, args.device)
        cells.append(c)
        if c.sizes:
            s = c.summary()
            print(
                f"budget {budget:.2f}  n={c.n_items:<4} saved={c.n_saved:<4} "
                f"closure median={s['size_median']:.1f} max={s['size_max']:.0f}"
            )
        else:
            print(f"budget {budget:.2f}  {c.error or 'no solver call'}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_scale{args.scale}_{args.device}"
    (outdir / f"{stem}.json").write_text(
        json.dumps([asdict(c) for c in cells], indent=2)
    )
    text = report(cells)
    (outdir / f"{stem}_report.txt").write_text(text)
    print()
    print(text)


if __name__ == "__main__":
    main()
