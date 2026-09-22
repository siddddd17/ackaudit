"""Feasible 0.05-budget sweep over how many Llama MLP outputs are saved.

This is the next experiment for pytorch/pytorch#197838.

For each k in {0,4,8,12,16}, it forces k evenly-spaced Llama MLP outputs into
(or out of) the saved set, then fills the remaining activation-memory budget
with the production dp_knapsack optimum over the non-MLP candidates.

That construction matters: it keeps the total saved-weight <= 0.05 while
changing only the number/placement of MLP outputs that are forced saved. The
remaining non-MLP part is still runtime-optimal conditional on those forced
choices.

Measurement:
  - same model/config and backend as the issue: llama scale 8, aot_eager
  - one compile/warmup pass to fix the plan
  - post-warmup resident allocation is used as the baseline
  - median of N measured forward+backward steps (default 5)
  - peak = torch.cuda.max_memory_allocated() - resident
  - gradients stay allocated and are zeroed in place between steps

The script records the exact selected MLP layers/nodes and the full saved node
set, so the plan itself is reproducible rather than inferred from a peak.

Run from the ackaudit repo root:

    python -m experiments.memory_snapshot.sweep_mlp_saved \
        --budget 0.05 --ks 0 4 8 12 16 --repeats 5

It writes a compact JSON result under results/memory_snapshot/ unless --out is
specified.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from torch._functorch import config as functorch_config
from torch._functorch._activation_checkpointing.knapsack import dp_knapsack
from torch._functorch.partitioners import CustomKnapsackSolver

from ackaudit.audit import RuntimeRecorder
from ackaudit.capture import _resolve

_VIEW_OPS = {
    "view", "_unsafe_view", "reshape", "expand", "clone", "t", "transpose", "permute"
}


def _aten_op(node: Any) -> str | None:
    parts = str(getattr(node, "target", "")).split(".")
    return parts[1] if len(parts) >= 2 and parts[0] == "aten" else None


def _strip_views(node: Any) -> Any | None:
    while node is not None and _aten_op(node) in _VIEW_OPS:
        args = getattr(node, "args", ())
        node = args[0] if args else None
    return node


def _mlp_layer_from_output(node: Any) -> int | None:
    """Return the Llama layer for a down_proj output, if this is one."""
    if _aten_op(node) != "mm" or not getattr(node, "args", None):
        return None

    x = _strip_views(node.args[0])
    if x is None or _aten_op(x) != "mul":
        return None

    silu = next(
        (a for a in x.args if _aten_op(a) == "silu"),
        None,
    )
    if silu is None:
        return None

    name = str(getattr(silu, "name", ""))
    if name == "silu":
        return 0
    if name.startswith("silu_"):
        suffix = name.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _evenly_spaced(items: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    """Pick k candidates evenly across the ordered MLP layers."""
    n = len(items)
    if k < 0 or k > n:
        raise ValueError(f"k={k} but only {n} MLP outputs were found")
    if k == 0:
        return []
    if k == n:
        return list(items)
    positions = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    # Guard against accidental duplicate rounding if this helper is reused.
    if len(set(positions)) != k:
        positions = list(dict.fromkeys(positions))
        if len(positions) != k:
            raise AssertionError(f"could not choose {k} distinct evenly-spaced positions: {positions}")
    return [items[p] for p in positions]


class FeasibleMLPSweepSolver(CustomKnapsackSolver):
    """Production dp_knapsack + a fixed, budget-feasible MLP-output selection."""

    def __init__(self, recorder: RuntimeRecorder, k: int, record: dict[str, Any]) -> None:
        self.recorder = recorder
        self.k = k
        self.record = record

    def __call__(
        self,
        memory,
        joint_graph,
        max_memory,
        node_info,
        all_recomputable_banned_nodes,
    ) -> tuple[list[int], list[int]]:
        if self.record.get("calls", 0):
            raise RuntimeError("expected exactly one knapsack call for this Llama compile")

        runtimes = self.recorder.lookup(all_recomputable_banned_nodes)

        mlp: list[tuple[int, int]] = []  # (candidate index, layer)
        for i, node in enumerate(all_recomputable_banned_nodes):
            layer = _mlp_layer_from_output(node)
            if layer is not None:
                mlp.append((i, layer))

        mlp.sort(key=lambda x: x[1])
        if len(mlp) != 32:
            raise RuntimeError(f"expected 32 Llama MLP outputs, found {len(mlp)}")
        if [layer for _, layer in mlp] != list(range(32)):
            raise RuntimeError(f"MLP layer mapping is not 0..31: {[layer for _, layer in mlp]}")

        forced = _evenly_spaced(mlp, self.k)
        forced_idx = {i for i, _ in forced}
        forced_weight = sum(memory[i] for i in forced_idx)
        remaining_capacity = max_memory - forced_weight

        if remaining_capacity < -1e-9:
            raise RuntimeError(
                f"k={self.k} is infeasible: forced MLP weight {forced_weight:.6f} > budget {max_memory:.6f}"
            )

        other_idx = [i for i in range(len(memory)) if i not in forced_idx]
        other_memory = [memory[i] for i in other_idx]
        other_runtime = [runtimes[i] for i in other_idx]
        _value, other_saved_local, _other_recomputed_local = dp_knapsack(
            other_memory, other_runtime, remaining_capacity
        )
        other_saved = {other_idx[i] for i in other_saved_local}

        saved = sorted(forced_idx | other_saved)

        # k=0 is deliberately the unconstrained production DP plan. Keep an
        # explicit check so the control cannot silently drift from dp_knapsack.
        natural_value, natural_saved, _natural_recomputed = dp_knapsack(
            memory, runtimes, max_memory
        )
        if self.k == 0 and saved != sorted(natural_saved):
            raise AssertionError("k=0 plan does not match the production dp_knapsack plan")
        recomputed = sorted(set(range(len(memory))) - set(saved))

        saved_weight = sum(memory[i] for i in saved)
        runtime_saved = sum(runtimes[i] for i in saved)

        if saved_weight > max_memory + 5e-7:
            raise AssertionError(
                f"constructed plan exceeds budget: {saved_weight:.9f} > {max_memory:.9f}"
            )

        self.record.update(
            calls=1,
            n_items=len(memory),
            n_mlp_outputs=len(mlp),
            k=self.k,
            max_memory=max_memory,
            forced_mlp_indices=sorted(forced_idx),
            forced_mlp_layers=[layer for _, layer in forced],
            forced_mlp_weight=forced_weight,
            remaining_capacity=remaining_capacity,
            saved_weight=saved_weight,
            runtime_saved=runtime_saved,
            natural_runtime_saved=natural_value,
            conditional_runtime_loss=natural_value - runtime_saved,
            conditional_runtime_retained_pct=(100.0 * runtime_saved / natural_value if natural_value else None),
            n_saved=len(saved),
            n_recomputed=len(recomputed),
            saved_node_names=[all_recomputable_banned_nodes[i].name for i in saved],
            saved_mlp_layers=[
                layer for i, layer in mlp if i in set(saved)
            ],
        )
        return saved, recomputed

    def uuid(self) -> Any:
        # Re-run the hook every compile; the selected plan is a test intervention.
        return None


def measure_plan(
    *,
    model_name: str,
    scale: int,
    budget: float,
    backend: str,
    k: int,
    repeats: int,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")

    # Keep model weights/inputs identical across k so the only intended
    # difference is the saved-set constraint.
    torch.manual_seed(197838)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(197838)

    functorch_config.activation_memory_budget = budget
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    build, make_inputs = _resolve(model_name, scale=scale)
    model = build().cuda()
    args = tuple(a.cuda() for a in make_inputs())
    compiled = torch.compile(model, backend=backend, dynamic=False)

    solver_record: dict[str, Any] = {}
    recorder = RuntimeRecorder()
    prev_solver = functorch_config.activation_memory_budget_solver
    prev_budget = functorch_config.activation_memory_budget
    try:
        functorch_config.activation_memory_budget_solver = FeasibleMLPSweepSolver(
            recorder, k=k, record=solver_record
        )
        # Warmup/compile. This fixes the saved set for subsequent measurements.
        with recorder:
            compiled(*args).backward()
    finally:
        functorch_config.activation_memory_budget_solver = prev_solver
        functorch_config.activation_memory_budget = prev_budget

    torch.cuda.synchronize()
    model.zero_grad(set_to_none=False)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated()

    peaks: list[int] = []
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        loss = compiled(*args)
        torch.cuda.synchronize()
        loss.backward()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - resident)
        model.zero_grad(set_to_none=False)
        torch.cuda.synchronize()

    peak_median = statistics.median(peaks)
    peak_mean = statistics.mean(peaks)

    result = {
        "model": model_name,
        "scale": scale,
        "budget": budget,
        "backend": backend,
        "k": k,
        "repeats": repeats,
        "torch_version": torch.__version__,
        "resident_bytes": resident,
        "peak_bytes": peak_median,
        "peak_median_mb": peak_median / 1e6,
        "peak_mean_mb": peak_mean / 1e6,
        "peaks_mb": [p / 1e6 for p in peaks],
        "solver": solver_record,
    }

    del model, args, compiled
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama")
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--ks", type=int, nargs="+", default=[0, 4, 8, 12, 16])
    parser.add_argument("--backend", default="aot_eager")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--out",
        default="results/memory_snapshot/mlp_saved_sweep_b0.05.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    if not 0.0 < args.budget < 1.0:
        raise SystemExit("--budget must be strictly between 0 and 1")
    if any(k < 0 or k > 32 for k in args.ks):
        raise SystemExit("for this experiment, each --k must be between 0 and 32")

    results: list[dict[str, Any]] = []
    for k in args.ks:
        result = measure_plan(
            model_name=args.model,
            scale=args.scale,
            budget=args.budget,
            backend=args.backend,
            k=k,
            repeats=args.repeats,
        )
        results.append(result)
        s = result["solver"]
        print(
            f"k={k:2d}  peak={result['peak_median_mb']:8.1f} MB  "
            f"saved_weight={s['saved_weight']:.4f}/{args.budget:.4f}  "
            f"runtime_saved={s['runtime_saved']:.6g}  "
            f"MLP_layers={s['saved_mlp_layers']}"
        )
        print(f"      per-repeat peaks: {result['peaks_mb']}")

    natural = next((r for r in results if r["k"] == 0), None)
    if natural is not None:
        base_peak = natural["peak_median_mb"]
        base_runtime = natural["solver"]["runtime_saved"]
        for r in results:
            r["peak_vs_k0_pct"] = 100.0 * (r["peak_median_mb"] - base_peak) / base_peak
            r["runtime_saved_vs_k0_pct"] = (
                100.0 * (r["solver"]["runtime_saved"] - base_runtime) / base_runtime
                if base_runtime
                else None
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

