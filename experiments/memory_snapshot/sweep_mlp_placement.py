"""
Experiment 2: fixed-count MLP placement sweep for PyTorch #197838.

Question
--------
At activation_memory_budget=0.05, holding the number of forced-saved
Llama MLP outputs fixed, does *which layers* are saved materially change
physical peak memory?

Design
------
- Llama scale=8
- backend=aot_eager
- activation_memory_budget=0.05
- k=8 forced MLP outputs
- placement families:

    early  = [0,1,2,3,4,5,6,7]
    middle = [12,13,14,15,16,17,18,19]
    late   = [24,25,26,27,28,29,30,31]
    even   = [0,4,9,13,18,22,27,31]

For each placement:

1. Force the selected MLP outputs into the saved set.
2. Give the remaining capacity to the production dp_knapsack.
3. Use the same runtime estimates recorded by RuntimeRecorder.
4. Verify the final saved weight <= activation_memory_budget.
5. Warm up / compile once.
6. Measure repeated forward+backward runs.
7. Report median and mean physical peak memory above resident memory.

The purpose is to isolate layer placement at fixed k.

Important caveat
----------------
The conditional DP is still free to select other activations, so changing
the forced placement can also change the remaining selected set and the
runtime objective. Therefore this experiment should be interpreted together
with saved_weight and runtime_saved; it does not by itself prove that the
production solver is suboptimal.

Run from repo root
------------------

PYTHONUNBUFFERED=1 python -u -m experiments.memory_snapshot.sweep_mlp_placement \
  --budget 0.05 \
  --k 8 \
  --repeats 5

Run only one placement:

PYTHONUNBUFFERED=1 python -u -m experiments.memory_snapshot.sweep_mlp_placement \
  --budget 0.05 \
  --k 8 \
  --repeats 5 \
  --placements early:0,1,2,3,4,5,6,7

Custom placements:

--placements \
    early:0,1,2,3,4,5,6,7 \
    middle:12,13,14,15,16,17,18,19 \
    late:24,25,26,27,28,29,30,31 \
    even:0,4,9,13,18,22,27,31

Output
------
results/memory_snapshot/mlp_placement_sweep_b0.05_k8.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch._functorch import config as functorch_config
from torch._functorch._activation_checkpointing.knapsack import dp_knapsack
from torch._functorch.partitioners import CustomKnapsackSolver

from ackaudit.audit import RuntimeRecorder
from ackaudit.capture import _resolve


# View-like / shape-only operations that can sit between the MLP output
# computation and the node that appears in the recomputation candidate list.
_VIEW_OPS = {
    "view",
    "_unsafe_view",
    "reshape",
    "expand",
    "clone",
    "t",
    "transpose",
    "permute",
}


def _aten_op(node: Any) -> str | None:
    """
    Return the aten operation name from an FX node target.

    Example:
        aten.mm.default -> "mm"
        aten.silu.default -> "silu"
    """
    parts = str(getattr(node, "target", "")).split(".")

    if len(parts) >= 2 and parts[0] == "aten":
        return parts[1]

    return None


def _strip_views(node: Any) -> Any | None:
    """
    Walk backwards through view-like operations.
    """
    while node is not None and _aten_op(node) in _VIEW_OPS:
        args = getattr(node, "args", ())

        if not args:
            return None

        node = args[0]

    return node


def _mlp_layer_from_output(node: Any) -> int | None:
    """
    Identify a Llama down_proj MLP-output candidate and return its layer index.

    Expected structure is approximately:

        mm(
            mul(
                silu(gate_projection),
                up_projection,
            ),
            down_projection_weight,
        )

    In the captured graphs, the SiLU node names are:

        silu
        silu_1
        silu_2
        ...
        silu_31
    """

    if _aten_op(node) != "mm":
        return None

    args = getattr(node, "args", ())

    if not args:
        return None

    x = _strip_views(args[0])

    if x is None or _aten_op(x) != "mul":
        return None

    silu = next(
        (
            arg
            for arg in x.args
            if _aten_op(arg) == "silu"
        ),
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


def _parse_placements(
    raw: list[str],
    n_layers: int,
) -> dict[str, list[int]]:
    """
    Parse CLI placement specs:

        early:0,1,2,3,4,5,6,7
    """

    placements: dict[str, list[int]] = {}

    for spec in raw:

        if ":" not in spec:
            raise ValueError(
                f"invalid placement '{spec}', "
                f"expected name:0,1,2,..."
            )

        name, values = spec.split(":", 1)

        name = name.strip()

        if not name:
            raise ValueError(
                f"invalid placement '{spec}': empty name"
            )

        layers = [
            int(x)
            for x in values.split(",")
            if x.strip()
        ]

        if not layers:
            raise ValueError(
                f"invalid placement '{spec}': no layers"
            )

        if len(set(layers)) != len(layers):
            raise ValueError(
                f"placement '{name}' contains duplicate layers"
            )

        if any(
            layer < 0 or layer >= n_layers
            for layer in layers
        ):
            raise ValueError(
                f"placement '{name}' contains layer outside "
                f"[0,{n_layers - 1}]"
            )

        placements[name] = sorted(layers)

    k_values = {
        len(layers)
        for layers in placements.values()
    }

    if len(k_values) > 1:
        raise ValueError(
            "all placements must have the same k; "
            f"got {k_values}"
        )

    return placements


class FixedPlacementSolver(CustomKnapsackSolver):
    """
    Condition the production dp_knapsack on a fixed set of MLP outputs.

    Forced MLP outputs are always saved.

    All remaining candidate nodes are passed through the same production
    dp_knapsack implementation, using the runtime estimates captured from
    the partitioner's normal runtime estimation path.
    """

    def __init__(
        self,
        recorder: RuntimeRecorder,
        forced_layers: list[int],
        record: dict[str, Any],
        placement_name: str,
    ) -> None:

        self.recorder = recorder
        self.forced_layers = sorted(forced_layers)
        self.record = record
        self.placement_name = placement_name

    def __call__(
        self,
        memory,
        joint_graph,
        max_memory,
        node_info,
        all_recomputable_banned_nodes,
    ) -> tuple[list[int], list[int]]:

        if self.record.get("calls", 0):
            raise RuntimeError(
                "expected exactly one knapsack call"
            )

        print(
            f"[{self.placement_name}] "
            f"knapsack capture: {len(memory)} candidates",
            flush=True,
        )

        runtimes = self.recorder.lookup(
            all_recomputable_banned_nodes
        )

        # ------------------------------------------------------------
        # Identify the 32 Llama MLP-output candidates.
        # ------------------------------------------------------------

        mlp: list[tuple[int, int]] = []

        for i, node in enumerate(
            all_recomputable_banned_nodes
        ):
            layer = _mlp_layer_from_output(node)

            if layer is not None:
                mlp.append((i, layer))

        mlp.sort(key=lambda x: x[1])

        if len(mlp) != 32:
            raise RuntimeError(
                "expected 32 Llama MLP outputs, "
                f"found {len(mlp)}"
            )

        layer_to_idx = {
            layer: index
            for index, layer in mlp
        }

        if sorted(layer_to_idx) != list(range(32)):
            raise RuntimeError(
                "MLP layer mapping is not 0..31: "
                f"{sorted(layer_to_idx)}"
            )

        # ------------------------------------------------------------
        # Force requested MLP layers.
        # ------------------------------------------------------------

        forced_idx = {
            layer_to_idx[layer]
            for layer in self.forced_layers
        }

        forced_weight = sum(
            memory[i]
            for i in forced_idx
        )

        remaining_capacity = (
            max_memory - forced_weight
        )

        print(
            f"[{self.placement_name}] "
            f"forced_weight={forced_weight:.9f}, "
            f"remaining_capacity={remaining_capacity:.9f}",
            flush=True,
        )

        if remaining_capacity < -1e-9:
            raise RuntimeError(
                f"{self.placement_name} infeasible: "
                f"forced MLP weight "
                f"{forced_weight:.9f} > "
                f"budget {max_memory:.9f}"
            )

        # ------------------------------------------------------------
        # Remove forced nodes and run the actual production
        # dp_knapsack over everything else.
        # ------------------------------------------------------------

        other_idx = [
            i
            for i in range(len(memory))
            if i not in forced_idx
        ]

        other_memory = [
            memory[i]
            for i in other_idx
        ]

        other_runtime = [
            runtimes[i]
            for i in other_idx
        ]

        print(
            f"[{self.placement_name}] "
            f"running conditional dp_knapsack on "
            f"{len(other_idx)} remaining candidates...",
            flush=True,
        )

        other_value, other_saved_local, _ = dp_knapsack(
            other_memory,
            other_runtime,
            remaining_capacity,
        )

        other_saved = {
            other_idx[i]
            for i in other_saved_local
        }

        # ------------------------------------------------------------
        # Construct complete saved / recomputed sets.
        # ------------------------------------------------------------

        saved = sorted(
            forced_idx | other_saved
        )

        recomputed = sorted(
            set(range(len(memory))) - set(saved)
        )

        saved_weight = sum(
            memory[i]
            for i in saved
        )

        runtime_saved = sum(
            runtimes[i]
            for i in saved
        )

        if saved_weight > max_memory + 5e-7:
            raise AssertionError(
                f"{self.placement_name} exceeds budget: "
                f"{saved_weight:.9f} > "
                f"{max_memory:.9f}"
            )

        saved_index_set = set(saved)

        saved_mlp_layers = [
            layer
            for i, layer in mlp
            if i in saved_index_set
        ]

        # ------------------------------------------------------------
        # Record the full solver result.
        # ------------------------------------------------------------

        self.record.update(
            calls=1,
            n_items=len(memory),
            n_mlp_outputs=len(mlp),
            k=len(self.forced_layers),
            placement_name=self.placement_name,
            forced_mlp_layers=self.forced_layers,
            forced_mlp_indices=sorted(forced_idx),
            forced_mlp_weight=forced_weight,
            remaining_capacity=remaining_capacity,
            conditional_other_runtime_value=other_value,
            saved_weight=saved_weight,
            runtime_saved=runtime_saved,
            n_saved=len(saved),
            n_recomputed=len(recomputed),
            saved_node_names=[
                all_recomputable_banned_nodes[i].name
                for i in saved
            ],
            saved_mlp_layers=saved_mlp_layers,
        )

        print(
            f"[{self.placement_name}] "
            f"conditional DP complete: "
            f"saved={len(saved)}, "
            f"recomputed={len(recomputed)}, "
            f"saved_weight={saved_weight:.9f}, "
            f"runtime_saved={runtime_saved:.6g}",
            flush=True,
        )

        print(
            f"[{self.placement_name}] "
            f"saved MLP layers={saved_mlp_layers}",
            flush=True,
        )

        return saved, recomputed

    def uuid(self) -> Any:
        return None


def measure_plan(
    *,
    model_name: str,
    scale: int,
    budget: float,
    backend: str,
    placement_name: str,
    forced_layers: list[int],
    repeats: int,
) -> dict[str, Any]:

    if repeats < 1:
        raise ValueError(
            "--repeats must be >= 1"
        )

    start_time = time.time()

    print()
    print(
        "=" * 80,
        flush=True,
    )
    print(
        f"[{placement_name}] START",
        flush=True,
    )
    print(
        f"[{placement_name}] "
        f"forced_layers={forced_layers}",
        flush=True,
    )
    print(
        f"[{placement_name}] "
        f"budget={budget}, backend={backend}, "
        f"scale={scale}, repeats={repeats}",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    # ------------------------------------------------------------
    # Deterministic seed.
    # ------------------------------------------------------------

    torch.manual_seed(197838)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(197838)

    # ------------------------------------------------------------
    # Reset compiler / CUDA state.
    # ------------------------------------------------------------

    print(
        f"[{placement_name}] resetting Dynamo / CUDA...",
        flush=True,
    )

    functorch_config.activation_memory_budget = budget

    torch._dynamo.reset()

    torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Build model.
    # ------------------------------------------------------------

    print(
        f"[{placement_name}] building model...",
        flush=True,
    )

    build, make_inputs = _resolve(
        model_name,
        scale=scale,
    )

    model = build().cuda()

    args = tuple(
        a.cuda()
        for a in make_inputs()
    )

    print(
        f"[{placement_name}] "
        f"model built; compiling with "
        f"backend={backend}...",
        flush=True,
    )

    # ------------------------------------------------------------
    # Compile.
    # ------------------------------------------------------------

    compiled = torch.compile(
        model,
        backend=backend,
        dynamic=False,
    )

    print(
        f"[{placement_name}] "
        f"compile object created.",
        flush=True,
    )

    # ------------------------------------------------------------
    # Install custom conditioned solver.
    # ------------------------------------------------------------

    solver_record: dict[str, Any] = {}

    recorder = RuntimeRecorder()

    prev_solver = (
        functorch_config.activation_memory_budget_solver
    )

    prev_budget = (
        functorch_config.activation_memory_budget
    )

    try:

        functorch_config.activation_memory_budget_solver = (
            FixedPlacementSolver(
                recorder=recorder,
                forced_layers=forced_layers,
                record=solver_record,
                placement_name=placement_name,
            )
        )

        print(
            f"[{placement_name}] "
            f"running first forward+backward "
            f"to trigger compilation and capture solver...",
            flush=True,
        )

        with recorder:
            compiled(*args).backward()

    finally:

        functorch_config.activation_memory_budget_solver = (
            prev_solver
        )

        functorch_config.activation_memory_budget = (
            prev_budget
        )

    compile_elapsed = time.time() - start_time

    print(
        f"[{placement_name}] "
        f"warmup/compile complete "
        f"after {compile_elapsed:.1f}s",
        flush=True,
    )

    # ------------------------------------------------------------
    # Synchronize and establish resident baseline.
    # ------------------------------------------------------------

    torch.cuda.synchronize()

    model.zero_grad(
        set_to_none=False
    )

    torch.cuda.synchronize()

    torch.cuda.empty_cache()

    resident = torch.cuda.memory_allocated()

    print(
        f"[{placement_name}] "
        f"resident memory = "
        f"{resident / 1e6:.3f} MB",
        flush=True,
    )

    # ------------------------------------------------------------
    # Measurement loop.
    # ------------------------------------------------------------

    peaks: list[int] = []

    print(
        f"[{placement_name}] "
        f"starting {repeats} measurement repeats...",
        flush=True,
    )

    for i in range(repeats):

        torch.cuda.reset_peak_memory_stats()

        loss = compiled(*args)

        torch.cuda.synchronize()

        loss.backward()

        torch.cuda.synchronize()

        peak_allocated = (
            torch.cuda.max_memory_allocated()
        )

        peak_relative = (
            peak_allocated - resident
        )

        peaks.append(
            peak_relative
        )

        model.zero_grad(
            set_to_none=False
        )

        torch.cuda.synchronize()

        print(
            f"[{placement_name}] "
            f"repeat {i + 1}/{repeats}: "
            f"peak={peak_relative / 1e6:.3f} MB",
            flush=True,
        )

    # ------------------------------------------------------------
    # Aggregate.
    # ------------------------------------------------------------

    peak_median = statistics.median(peaks)

    peak_mean = statistics.mean(peaks)

    total_elapsed = time.time() - start_time

    result = {
        "model": model_name,
        "scale": scale,
        "budget": budget,
        "backend": backend,
        "placement": placement_name,
        "forced_mlp_layers": forced_layers,
        "repeats": repeats,
        "torch_version": torch.__version__,
        "resident_bytes": int(resident),
        "peak_bytes": int(peak_median),
        "peak_median_mb": peak_median / 1e6,
        "peak_mean_mb": peak_mean / 1e6,
        "peaks_mb": [
            p / 1e6
            for p in peaks
        ],
        "solver": solver_record,
        "elapsed_seconds": total_elapsed,
    }

    print(
        f"[{placement_name}] "
        f"MEDIAN PEAK="
        f"{peak_median / 1e6:.3f} MB",
        flush=True,
    )

    print(
        f"[{placement_name}] "
        f"MEAN PEAK="
        f"{peak_mean / 1e6:.3f} MB",
        flush=True,
    )

    print(
        f"[{placement_name}] "
        f"TOTAL ELAPSED="
        f"{total_elapsed:.1f}s",
        flush=True,
    )

    # Explicitly release this experiment's compiled model before the next
    # placement so separate placements don't accumulate GPU state.
    del model
    del args
    del compiled

    torch.cuda.empty_cache()

    print(
        f"[{placement_name}] DONE",
        flush=True,
    )

    return result


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--model",
        default="llama",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--backend",
        default="aot_eager",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--placements",
        nargs="+",
        default=[
            "early:0,1,2,3,4,5,6,7",
            "middle:12,13,14,15,16,17,18,19",
            "late:24,25,26,27,28,29,30,31",
            "even:0,4,9,13,18,22,27,31",
        ],
        help=(
            "Placement specs of the form "
            "name:layer0,layer1,..."
        ),
    )

    parser.add_argument(
        "--out",
        default=(
            "results/memory_snapshot/"
            "mlp_placement_sweep_b0.05_k8.json"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Basic validation.
    # ------------------------------------------------------------

    if not torch.cuda.is_available():
        raise SystemExit(
            "Experiment 2 requires CUDA."
        )

    if not 0.0 < args.budget < 1.0:
        raise SystemExit(
            "--budget must be strictly between 0 and 1"
        )

    if not 0 <= args.k <= 32:
        raise SystemExit(
            "--k must be between 0 and 32"
        )

    placements = _parse_placements(
        args.placements,
        32,
    )

    for name, layers in placements.items():

        if len(layers) != args.k:
            raise SystemExit(
                f"placement '{name}' has k={len(layers)} "
                f"but --k={args.k}"
            )

    # ------------------------------------------------------------
    # Environment summary.
    # ------------------------------------------------------------

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "EXPERIMENT 2: FIXED-K MLP PLACEMENT SWEEP",
        flush=True,
    )

    print(
        f"torch={torch.__version__}",
        flush=True,
    )

    print(
        f"device={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    print(
        f"model={args.model}",
        flush=True,
    )

    print(
        f"scale={args.scale}",
        flush=True,
    )

    print(
        f"backend={args.backend}",
        flush=True,
    )

    print(
        f"budget={args.budget}",
        flush=True,
    )

    print(
        f"k={args.k}",
        flush=True,
    )

    print(
        f"repeats={args.repeats}",
        flush=True,
    )

    print(
        "placements:",
        flush=True,
    )

    for name, layers in placements.items():
        print(
            f"  {name}: {layers}",
            flush=True,
        )

    print(
        "=" * 80,
        flush=True,
    )

    # ------------------------------------------------------------
    # Execute every placement.
    # ------------------------------------------------------------

    results: list[dict[str, Any]] = []

    experiment_start = time.time()

    for name, layers in placements.items():

        result = measure_plan(
            model_name=args.model,
            scale=args.scale,
            budget=args.budget,
            backend=args.backend,
            placement_name=name,
            forced_layers=layers,
            repeats=args.repeats,
        )

        results.append(result)

        solver = result["solver"]

        print()
        print(
            f"[SUMMARY:{name}] "
            f"peak={result['peak_median_mb']:.3f} MB | "
            f"saved_weight="
            f"{solver['saved_weight']:.9f}/"
            f"{args.budget:.9f} | "
            f"runtime_saved="
            f"{solver['runtime_saved']:.6g}",
            flush=True,
        )

        print(
            f"[SUMMARY:{name}] "
            f"forced={layers}",
            flush=True,
        )

        print(
            f"[SUMMARY:{name}] "
            f"peaks="
            f"{result['peaks_mb']}",
            flush=True,
        )

    # ------------------------------------------------------------
    # Compare against the first placement.
    #
    # By default "early" is first, so these are normally expressed
    # relative to early. We intentionally do NOT treat this as a
    # ranking; it is simply a reference point for effect size.
    # ------------------------------------------------------------

    if not results:
        raise RuntimeError(
            "no placement results produced"
        )

    baseline_peak = (
        results[0]["peak_median_mb"]
    )

    baseline_runtime = (
        results[0]["solver"]["runtime_saved"]
    )

    baseline_name = (
        results[0]["placement"]
    )

    for result in results:

        peak = result["peak_median_mb"]

        runtime_saved = (
            result["solver"]["runtime_saved"]
        )

        result["baseline_placement"] = (
            baseline_name
        )

        if baseline_peak:

            result["peak_vs_first_pct"] = (
                100.0
                * (peak - baseline_peak)
                / baseline_peak
            )

            result["peak_reduction_vs_first_pct"] = (
                100.0
                * (baseline_peak - peak)
                / baseline_peak
            )

        else:

            result["peak_vs_first_pct"] = None

            result["peak_reduction_vs_first_pct"] = None

        if baseline_runtime:

            result["runtime_saved_vs_first_pct"] = (
                100.0
                * (runtime_saved - baseline_runtime)
                / baseline_runtime
            )

        else:

            result["runtime_saved_vs_first_pct"] = None

    # ------------------------------------------------------------
    # Write JSON.
    # ------------------------------------------------------------

    out = Path(args.out)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            results,
            indent=2,
        )
        + "\n"
    )

    total_elapsed = (
        time.time() - experiment_start
    )

    # ------------------------------------------------------------
    # Human-readable final table.
    # ------------------------------------------------------------

    print()
    print(
        "=" * 100,
        flush=True,
    )

    print(
        "FINAL RESULTS",
        flush=True,
    )

    print(
        "=" * 100,
        flush=True,
    )

    print(
        f"{'placement':<12}"
        f"{'peak(MB)':>14}"
        f"{'saved_weight':>16}"
        f"{'runtime_saved':>20}"
        f"{'peak Δ vs first':>18}",
        flush=True,
    )

    print(
        "-" * 100,
        flush=True,
    )

    for result in results:

        solver = result["solver"]

        peak_delta = (
            result["peak_reduction_vs_first_pct"]
        )

        if peak_delta is None:
            peak_delta_text = "n/a"
        else:
            peak_delta_text = (
                f"{peak_delta:+.2f}%"
            )

        print(
            f"{result['placement']:<12}"
            f"{result['peak_median_mb']:>14.3f}"
            f"{solver['saved_weight']:>16.9f}"
            f"{solver['runtime_saved']:>20.6g}"
            f"{peak_delta_text:>18}",
            flush=True,
        )

    print(
        "-" * 100,
        flush=True,
    )

    print(
        f"Reference placement: {baseline_name}",
        flush=True,
    )

    print(
        f"Total experiment time: "
        f"{total_elapsed:.1f}s",
        flush=True,
    )

    print(
        f"JSON written to: {out}",
        flush=True,
    )

    print(
        "=" * 100,
        flush=True,
    )


if __name__ == "__main__":
    main()
