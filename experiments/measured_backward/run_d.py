"""Measurement D: does any simulated backward schedule track the real allocator?

Every peak in this repo so far is simulated. `KnapsackEvaluator` linearises the
joint graph into one of many valid topological orders and reports the memory a
backward pass would use under that order. Which order it picks is unspecified
(docs/UPSTREAM_BUG.md), and the three we can pin disagree by up to 175% on
branching graphs.

This asks whether that number corresponds to anything. For each budget in a
sweep it lets the partitioner choose a plan as it normally would, records that
plan, executes a real forward and backward on CUDA, and reads
`torch.cuda.max_memory_allocated()`. The same plan is then scored under
schedules A, B and C.

The comparison is on curve shape and plan ranking across budgets, never on
absolute values: simulated peaks are normalised fractions of a graph-local
maximum, while the allocator reports raw bytes including parameters, gradients,
workspace and fragmentation. A systematic offset is expected.

Both outcomes are results. If one schedule tracks the measured curve, the
simulator is usable and the schedule spread matters. If none does, the
simulator is measuring something that does not happen, which is the more
fundamental finding.

    python -m experiments.measured_backward.run_d --model llama --scale 4
    python -m experiments.measured_backward.run_d --probe --model llama
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch._functorch import config as functorch_config
from torch._functorch._activation_checkpointing.graph_info_provider import (
    GraphInfoProvider,
)
from torch._functorch._activation_checkpointing.knapsack import dp_knapsack
from torch._functorch._activation_checkpointing.knapsack_evaluator import (
    KnapsackEvaluator,
)
from torch._functorch.partitioners import CustomKnapsackSolver

from ackaudit.audit import RuntimeRecorder
from ackaudit.capture import _resolve
from ackaudit.schedule import SCHEDULES, Schedule

# 0 and 1 short-circuit past the knapsack entirely, so stay strictly inside.
DEFAULT_BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class Cell:
    """One (model, budget) measurement: the plan, the simulation, the allocator."""

    model: str
    budget: float
    scale: int
    torch_version: str
    device_name: str

    eager: bool = False
    eager: bool = False
    n_items: int = 0
    n_saved: int = 0
    n_recomputed: int = 0
    solver_called: bool = False

    proxy_peak: float = 0.0
    simulated: dict[str, float] = field(default_factory=dict)
    simulate_error: str = ""

    # bytes, from the allocator
    resident_before: int = 0
    measured_peaks: list[int] = field(default_factory=list)
    measure_error: str = ""

    @property
    def measured_peak(self) -> int:
        return statistics.median(self.measured_peaks) if self.measured_peaks else 0

    @property
    def measured_activation(self) -> int:
        """Peak minus the parameters and gradients already resident."""
        return self.measured_peak - self.resident_before

    @property
    def noise(self) -> int:
        """Spread across identical repetitions. Any difference below this is unreadable."""
        if len(self.measured_peaks) < 2:
            return 0
        return max(self.measured_peaks) - min(self.measured_peaks)


class MeasuringSolver(CustomKnapsackSolver):
    """Delegates to dp_knapsack, then records and scores the plan it returned.

    Deliberately does not force a plan. The point is to measure what the
    partitioner actually does at each budget, then ask what the simulator says
    about that same plan.
    """

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
            evaluator = KnapsackEvaluator(graph_info_provider=provider)
            c.proxy_peak = evaluator.evaluate_knapsack_output(
                saved_nodes_idxs=saved,
                recomputable_node_idxs=recomputed,
                account_for_backward_pass=False,
            )["peak_memory"]
            for mode in SCHEDULES:
                with Schedule(mode, provider.graph_nodes_in_order):
                    c.simulated[mode] = evaluator.evaluate_knapsack_output(
                        saved_nodes_idxs=saved,
                        recomputable_node_idxs=recomputed,
                        account_for_backward_pass=True,
                    )["peak_memory"]
        except (
            Exception
        ) as exc:  # noqa: BLE001 -- a failed simulation must not stop the measurement
            c.simulate_error = f"{type(exc).__name__}: {exc}"

        return saved, recomputed

    def uuid(self) -> Any:
        return None  # never cache-hit past; we want a solver call per compile


def measure(
    model_name: str,
    budget: float,
    scale: int,
    reps: int,
    device: str,
    eager: bool = False,
) -> Cell:
    cell = Cell(
        model=model_name,
        budget=budget,
        scale=scale,
        torch_version=torch.__version__,
        device_name=torch.cuda.get_device_name(0),
        eager=eager,
    )

    build, make_inputs = _resolve(model_name, scale=scale)
    model = build().to(device)
    args = tuple(a.to(device) for a in make_inputs())

    recorder = RuntimeRecorder()
    solver = MeasuringSolver(recorder, cell)

    prev_solver = functorch_config.activation_memory_budget_solver
    prev_budget = functorch_config.activation_memory_budget
    try:
        functorch_config.activation_memory_budget_solver = solver
        functorch_config.activation_memory_budget = budget

        torch._dynamo.reset()
        torch.cuda.empty_cache()

        if eager:
            # No compilation, no partitioner, no knapsack. The config comment for
            # activation_memory_budget states the partitioner "should always use
            # less memory than eager", so this is the figure that claim refers to.
            compiled = model
            compiled(*args).backward()
            torch.cuda.synchronize()
        else:
            with recorder:
                compiled = torch.compile(model, backend="aot_eager", dynamic=False)
                # Warmup. Compiles forward, and the backward graph lazily on
                # .backward(). The solver runs here, so the plan is fixed before
                # anything is measured.
                compiled(*args).backward()
                torch.cuda.synchronize()

        # Leave gradient buffers allocated and zero them in place, so what varies
        # across budgets is activation memory rather than one-off grad allocation.
        model.zero_grad(set_to_none=False)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        cell.resident_before = torch.cuda.memory_allocated()

        for _ in range(reps):
            torch.cuda.reset_peak_memory_stats()
            compiled(*args).backward()
            torch.cuda.synchronize()
            cell.measured_peaks.append(torch.cuda.max_memory_allocated())
            model.zero_grad(set_to_none=False)

    except (
        Exception
    ) as exc:  # noqa: BLE001 -- record OOM and compile failures, keep sweeping
        cell.measure_error = f"{type(exc).__name__}: {exc}"
    finally:
        functorch_config.activation_memory_budget_solver = prev_solver
        functorch_config.activation_memory_budget = prev_budget
        del model, args
        torch.cuda.empty_cache()

    return cell


def probe(model_name: str, device: str, max_scale: int = 12) -> None:
    """Find the largest scale that fits, and whether the signal clears the noise."""
    print(f"probing {model_name} on {torch.cuda.get_device_name(0)}")
    print(
        f"{'scale':>6} {'n':>4} {'resident MB':>12} {'peak MB':>10} {'activation MB':>14} {'noise KB':>9}"
    )
    for scale in range(1, max_scale + 1):
        c = measure(model_name, budget=0.5, scale=scale, reps=3, device=device)
        if c.measure_error:
            print(f"{scale:>6} {'':>4} {c.measure_error[:60]}")
            break
        print(
            f"{scale:>6} {c.n_items:>4} {c.resident_before / 1e6:>12.1f} "
            f"{c.measured_peak / 1e6:>10.1f} {c.measured_activation / 1e6:>14.1f} "
            f"{c.noise / 1e3:>9.1f}"
        )


def report(cells: list[Cell]) -> str:
    out: list[str] = []
    ok = [
        c
        for c in cells
        if c.measured_peaks and c.solver_called and not c.simulate_error
    ]
    if not ok:
        return "no usable cells; check errors in results.json\n"

    head = ok[0]
    out.append(f"MEASUREMENT D  {head.model} scale={head.scale}")
    out.append(f"torch {head.torch_version}  {head.device_name}")
    out.append("")
    out.append("Simulated peaks are normalised fractions; measured is bytes. Compare")
    out.append("shape across budgets and plan ranking, not absolute values.")
    out.append("")

    noise = max(c.noise for c in ok)
    out.append(
        f"noise floor (max spread over {len(ok[0].measured_peaks)} identical reps): {noise / 1e3:.1f} KB"
    )
    span = max(c.measured_activation for c in ok) - min(
        c.measured_activation for c in ok
    )
    out.append(f"measured activation span across budgets: {span / 1e6:.1f} MB")
    if span <= noise * 3:
        out.append(
            "  WARNING: span is not clearly above the noise floor. Scale up before"
        )
        out.append("  drawing any conclusion from these numbers.")
    out.append("")

    out.append(
        f"{'budget':>7} {'saved':>6} {'recomp':>7} {'measured MB':>12} "
        f"{'proxy':>8} " + " ".join(f"{m:>13}" for m in SCHEDULES)
    )
    for c in sorted(ok, key=lambda c: c.budget):
        sims = " ".join(f"{c.simulated.get(m, float('nan')):>13.4f}" for m in SCHEDULES)
        out.append(
            f"{c.budget:>7.2f} {c.n_saved:>6} {c.n_recomputed:>7} "
            f"{c.measured_activation / 1e6:>12.1f} {c.proxy_peak:>8.4f} {sims}"
        )
    out.append("")

    # Rank correlation. Absolute values are not comparable; the ordering is.
    measured = [c.measured_activation for c in sorted(ok, key=lambda c: c.budget)]
    out.append("Spearman rank correlation with the measured curve:")
    for mode in ("proxy", *SCHEDULES):
        if mode == "proxy":
            sim = [c.proxy_peak for c in sorted(ok, key=lambda c: c.budget)]
        else:
            sim = [
                c.simulated.get(mode, float("nan"))
                for c in sorted(ok, key=lambda c: c.budget)
            ]
        try:
            rho = _spearman(measured, sim)
            out.append(f"  {mode:<16} {rho:+.3f}")
        except Exception:  # noqa: BLE001
            out.append(f"  {mode:<16}  n/a")
    out.append("")
    out.append(
        "rho near +1 means that schedule orders budgets the way the allocator does."
    )
    out.append(
        "rho near 0 across all of them means the simulator tracks nothing measurable."
    )
    return "\n".join(out) + "\n"


def _spearman(a: list[float], b: list[float]) -> float:
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    ra, rb = rank(a), rank(b)
    n = len(a)
    if n < 2:
        raise ValueError("need at least two points")
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    if den == 0:
        raise ValueError("no variance")
    return num / den


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument(
        "--reps", type=int, default=5, help="identical repetitions, for the noise floor"
    )
    ap.add_argument("--budgets", type=float, nargs="*", default=DEFAULT_BUDGETS)
    ap.add_argument("--outdir", default="results/measured_backward")
    ap.add_argument(
        "--probe",
        action="store_true",
        help="find the largest scale that fits, then exit",
    )
    ap.add_argument(
        "--eager",
        action="store_true",
        help="measure uncompiled eager instead; budgets are ignored, one cell is produced",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("measurement D needs CUDA; torch.cuda.is_available() is False")
    device = "cuda"

    if args.probe:
        probe(args.model, device)
        return

    cells: list[Cell] = []
    budgets = [0.0] if args.eager else args.budgets
    for budget in budgets:
        c = measure(args.model, budget, args.scale, args.reps, device, eager=args.eager)
        cells.append(c)
        status = (
            c.measure_error
            or c.simulate_error
            or f"{c.measured_activation / 1e6:.1f} MB"
        )
        print(f"budget {budget:.2f}  n={c.n_items:<3} saved={c.n_saved:<3} {status}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_scale{args.scale}" + ("_eager" if args.eager else "")
    (outdir / f"{stem}.json").write_text(
        json.dumps([asdict(c) for c in cells], indent=2)
    )
    text = report(cells)
    (outdir / f"{stem}_report.txt").write_text(text)
    print()
    print(text)


if __name__ == "__main__":
    main()
