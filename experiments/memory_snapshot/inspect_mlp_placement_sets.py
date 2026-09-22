"""
Experiment 2.1: verify the exact logical saved sets for the fixed-k MLP
placement sweep from PyTorch #197838.

Goal
----
Experiment 2 showed four feasible plans at budget=0.05 and k=8 with:

    early   1017.997824 MB
    middle  1019.439616 MB
    late    1041.508352 MB
    even    1042.819072 MB

All four reported exactly the same:

    saved_weight                  0.04819231900170282
    runtime_saved                 44426100736
    n_saved                       27
    n_recomputed                  297
    forced_mlp_weight             0.014497261992673392

This script verifies the structural part:

    Are the four selected saved sets identical except for the eight
    intentionally forced MLP outputs?

It reruns ONLY the solver capture (no peak-memory measurement) and records
the selected candidate nodes with:

    - candidate index
    - FX node name
    - aten op
    - logical MLP layer, when identifiable
    - tensor shape / dtype, when available
    - memory cost
    - runtime estimate
    - role

It then checks:

    1. Each placement saves exactly its eight forced MLP layers.
    2. The selected non-MLP node set is identical across placements.
    3. Reported scalar knapsack values are identical.
    4. Per-layer MLP memory/runtime costs are consistent.

Run from repo root:

    PYTHONUNBUFFERED=1 python -u \
      -m experiments.memory_snapshot.inspect_mlp_placement_sets

Output:

    results/memory_snapshot/mlp_placement_sets_b0.05_k8.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch._functorch import config as functorch_config

from ackaudit.audit import RuntimeRecorder
from ackaudit.capture import _resolve

from experiments.memory_snapshot.sweep_mlp_placement import (
    FixedPlacementSolver,
    _aten_op,
    _mlp_layer_from_output,
    _parse_placements,
)


def _tensor_meta(node: Any) -> dict[str, Any]:
    """Extract JSON-safe tensor metadata from an FX node."""
    meta = getattr(node, "meta", {}) or {}
    tensor_meta = meta.get("tensor_meta")

    if tensor_meta is None:
        return {}

    result: dict[str, Any] = {}

    shape = getattr(tensor_meta, "shape", None)
    if shape is not None:
        result["shape"] = [int(x) for x in shape]

    dtype = getattr(tensor_meta, "dtype", None)
    if dtype is not None:
        result["dtype"] = str(dtype)

    requires_grad = getattr(tensor_meta, "requires_grad", None)
    if requires_grad is not None:
        result["requires_grad"] = bool(requires_grad)

    return result


def _as_text(value: Any) -> str | None:
    """Safely stringify optional FX metadata."""
    if value is None:
        return None

    try:
        return str(value)
    except Exception:
        return repr(value)


def _layer_from_metadata(node: Any) -> int | None:
    """
    Best-effort transformer-layer extraction from FX metadata.

    This is only supplemental. For MLP outputs, the structural
    _mlp_layer_from_output() mapping remains authoritative.
    """
    meta = getattr(node, "meta", {}) or {}

    values = [
        meta.get("nn_module_stack"),
        meta.get("source_fn_stack"),
        meta.get("stack_trace"),
        meta.get("source"),
    ]

    text = "\n".join(
        _as_text(value) or ""
        for value in values
    )

    patterns = [
        r"(?:layers|layer)[._\[](\d+)",
        r"(?:layers|layer)[^\d]{1,8}(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match is None:
            continue

        value = int(match.group(1))

        if 0 <= value < 32:
            return value

    return None


def _candidate_record(
    *,
    index: int,
    node: Any,
    memory: Any,
    runtime: Any,
) -> dict[str, Any]:
    """Serialize one recomputation candidate."""
    mlp_layer = _mlp_layer_from_output(node)

    return {
        "candidate_index": int(index),
        "name": str(getattr(node, "name", "")),
        "aten_op": _aten_op(node),
        "role": (
            "mlp_output"
            if mlp_layer is not None
            else "other"
        ),
        "mlp_layer": mlp_layer,
        "metadata_layer": _layer_from_metadata(node),
        "memory": float(memory[index]),
        "runtime": float(runtime[index]),
        "tensor": _tensor_meta(node),
        "nn_module_stack": _as_text(
            (getattr(node, "meta", {}) or {}).get(
                "nn_module_stack"
            )
        ),
        "source_fn_stack": _as_text(
            (getattr(node, "meta", {}) or {}).get(
                "source_fn_stack"
            )
        ),
    }


class InspectingPlacementSolver(FixedPlacementSolver):
    """
    Exact Experiment-2 solver semantics plus full selected-node capture.
    """

    def __init__(
        self,
        recorder: RuntimeRecorder,
        forced_layers: list[int],
        record: dict[str, Any],
        placement_name: str,
    ) -> None:
        super().__init__(
            recorder=recorder,
            forced_layers=forced_layers,
            record=record,
            placement_name=placement_name,
        )

        self.candidate_metadata: list[dict[str, Any]] = []
        self.selected_metadata: list[dict[str, Any]] = []
        self.recomputed_metadata: list[dict[str, Any]] = []

    def __call__(
        self,
        memory,
        joint_graph,
        max_memory,
        node_info,
        all_recomputable_banned_nodes,
    ) -> tuple[list[int], list[int]]:

        runtimes = self.recorder.lookup(
            all_recomputable_banned_nodes
        )

        self.candidate_metadata = [
            _candidate_record(
                index=i,
                node=node,
                memory=memory,
                runtime=runtimes,
            )
            for i, node in enumerate(
                all_recomputable_banned_nodes
            )
        ]

        saved, recomputed = super().__call__(
            memory,
            joint_graph,
            max_memory,
            node_info,
            all_recomputable_banned_nodes,
        )

        saved_set = set(saved)
        recomputed_set = set(recomputed)

        self.selected_metadata = [
            self.candidate_metadata[i]
            for i in sorted(saved_set)
        ]

        self.recomputed_metadata = [
            self.candidate_metadata[i]
            for i in sorted(recomputed_set)
        ]

        record = self.record

        record["selected_candidate_metadata"] = (
            self.selected_metadata
        )

        record["recomputed_candidate_metadata"] = (
            self.recomputed_metadata
        )

        record["selected_non_mlp_names"] = [
            item["name"]
            for item in self.selected_metadata
            if item["role"] != "mlp_output"
        ]

        record["selected_non_mlp_metadata"] = [
            item
            for item in self.selected_metadata
            if item["role"] != "mlp_output"
        ]

        record["selected_mlp"] = [
            {
                "layer": item["mlp_layer"],
                "name": item["name"],
                "candidate_index": item[
                    "candidate_index"
                ],
                "memory": item["memory"],
                "runtime": item["runtime"],
                "tensor": item["tensor"],
            }
            for item in self.selected_metadata
            if item["role"] == "mlp_output"
        ]

        return saved, recomputed


def run_capture(
    *,
    model_name: str,
    scale: int,
    budget: float,
    backend: str,
    placement_name: str,
    forced_layers: list[int],
) -> dict[str, Any]:

    print()
    print("=" * 100, flush=True)
    print(
        f"[{placement_name}] START",
        flush=True,
    )
    print(
        f"[{placement_name}] forced MLP layers={forced_layers}",
        flush=True,
    )

    torch.manual_seed(197838)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(197838)

    torch._dynamo.reset()
    torch.cuda.empty_cache()

    functorch_config.activation_memory_budget = budget

    build, make_inputs = _resolve(
        model_name,
        scale=scale,
    )

    model = build().cuda()

    args = tuple(
        arg.cuda()
        for arg in make_inputs()
    )

    print(
        f"[{placement_name}] compiling...",
        flush=True,
    )

    compiled = torch.compile(
        model,
        backend=backend,
        dynamic=False,
    )

    recorder = RuntimeRecorder()
    record: dict[str, Any] = {}

    solver = InspectingPlacementSolver(
        recorder=recorder,
        forced_layers=forced_layers,
        record=record,
        placement_name=placement_name,
    )

    previous_solver = (
        functorch_config.activation_memory_budget_solver
    )
    previous_budget = (
        functorch_config.activation_memory_budget
    )

    try:
        functorch_config.activation_memory_budget_solver = (
            solver
        )

        print(
            f"[{placement_name}] triggering capture...",
            flush=True,
        )

        with recorder:
            compiled(*args).backward()

    finally:
        functorch_config.activation_memory_budget_solver = (
            previous_solver
        )
        functorch_config.activation_memory_budget = (
            previous_budget
        )

    print(
        f"[{placement_name}] capture complete",
        flush=True,
    )

    selected = record[
        "selected_candidate_metadata"
    ]

    selected_mlp = [
        item
        for item in selected
        if item["role"] == "mlp_output"
    ]

    selected_other = [
        item
        for item in selected
        if item["role"] != "mlp_output"
    ]

    print()
    print(
        f"[{placement_name}] "
        f"saved={len(selected)} "
        f"MLP={len(selected_mlp)} "
        f"other={len(selected_other)}",
        flush=True,
    )

    print(
        f"[{placement_name}] "
        f"saved_weight={record['saved_weight']:.15f}",
        flush=True,
    )

    print(
        f"[{placement_name}] "
        f"runtime_saved={record['runtime_saved']:.15g}",
        flush=True,
    )

    print(
        f"[{placement_name}] selected MLP outputs:",
        flush=True,
    )

    for item in selected_mlp:
        print(
            f"    layer={item['mlp_layer']:2d} "
            f"name={item['name']:<10} "
            f"candidate={item['candidate_index']:3d} "
            f"memory={item['memory']:.12f} "
            f"runtime={item['runtime']:.6g} "
            f"shape={item['tensor'].get('shape')}",
            flush=True,
        )

    print(
        f"[{placement_name}] selected non-MLP nodes:",
        flush=True,
    )

    for item in selected_other:
        print(
            f"    {item['name']:<55} "
            f"op={str(item['aten_op']):<8} "
            f"candidate={item['candidate_index']:3d} "
            f"memory={item['memory']:.12f} "
            f"runtime={item['runtime']:.6g} "
            f"shape={item['tensor'].get('shape')}",
            flush=True,
        )

    result = {
        "model": model_name,
        "scale": scale,
        "budget": budget,
        "backend": backend,
        "torch_version": torch.__version__,
        "placement": placement_name,
        "forced_mlp_layers": forced_layers,
        "solver": record,
    }

    del model
    del args
    del compiled

    torch.cuda.empty_cache()

    print(
        f"[{placement_name}] DONE",
        flush=True,
    )

    return result


def compare_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:

    if not results:
        raise RuntimeError(
            "no results to compare"
        )

    reference = results[0]
    reference_name = reference["placement"]
    reference_solver = reference["solver"]

    # ------------------------------------------------------------
    # Scalar fields.
    # ------------------------------------------------------------

    scalar_fields = [
        "n_items",
        "n_mlp_outputs",
        "k",
        "forced_mlp_weight",
        "remaining_capacity",
        "conditional_other_runtime_value",
        "saved_weight",
        "runtime_saved",
        "n_saved",
        "n_recomputed",
    ]

    scalar_comparison: dict[str, Any] = {}

    for field in scalar_fields:

        values = [
            result["solver"].get(field)
            for result in results
        ]

        scalar_comparison[field] = {
            "values": values,
            "all_equal": all(
                value == values[0]
                for value in values
            ),
        }

    # ------------------------------------------------------------
    # Non-MLP set equality.
    # ------------------------------------------------------------

    reference_non_mlp = set(
        reference_solver[
            "selected_non_mlp_names"
        ]
    )

    non_mlp_comparison: dict[str, Any] = {}

    for result in results:

        name = result["placement"]

        current = set(
            result["solver"][
                "selected_non_mlp_names"
            ]
        )

        non_mlp_comparison[name] = {
            "count": len(current),
            "same_as_reference": (
                current == reference_non_mlp
            ),
            "only_in_this": sorted(
                current - reference_non_mlp
            ),
            "missing_vs_reference": sorted(
                reference_non_mlp - current
            ),
        }

    # ------------------------------------------------------------
    # MLP mapping check.
    # ------------------------------------------------------------

    mlp_comparison: dict[str, Any] = {}

    for result in results:

        name = result["placement"]

        forced = sorted(
            result["forced_mlp_layers"]
        )

        selected = sorted(
            item["layer"]
            for item in result["solver"][
                "selected_mlp"
            ]
        )

        mlp_comparison[name] = {
            "forced_layers": forced,
            "selected_layers": selected,
            "exact_match": (
                forced == selected
            ),
            "nodes": result["solver"][
                "selected_mlp"
            ],
        }

    # ------------------------------------------------------------
    # Check that each logical MLP layer has the same scalar
    # memory/runtime cost when it appears.
    # ------------------------------------------------------------

    per_layer: dict[int, list[dict[str, Any]]] = {}

    for result in results:

        for item in result["solver"][
            "selected_mlp"
        ]:

            layer = int(item["layer"])

            per_layer.setdefault(
                layer,
                [],
            ).append(
                {
                    "placement": result["placement"],
                    "name": item["name"],
                    "candidate_index": item[
                        "candidate_index"
                    ],
                    "memory": item["memory"],
                    "runtime": item["runtime"],
                    "tensor": item["tensor"],
                }
            )

    per_layer_comparison: dict[str, Any] = {}

    for layer, entries in sorted(
        per_layer.items()
    ):

        memories = [
            entry["memory"]
            for entry in entries
        ]

        runtimes = [
            entry["runtime"]
            for entry in entries
        ]

        per_layer_comparison[str(layer)] = {
            "entries": entries,
            "all_memory_equal": all(
                value == memories[0]
                for value in memories
            ),
            "all_runtime_equal": all(
                value == runtimes[0]
                for value in runtimes
            ),
        }

    # ------------------------------------------------------------
    # Final control verdict.
    # ------------------------------------------------------------

    all_scalars_equal = all(
        entry["all_equal"]
        for entry in scalar_comparison.values()
    )

    all_non_mlp_equal = all(
        entry["same_as_reference"]
        for entry in non_mlp_comparison.values()
    )

    all_mlp_match = all(
        entry["exact_match"]
        for entry in mlp_comparison.values()
    )

    controlled_difference = (
        all_scalars_equal
        and all_non_mlp_equal
        and all_mlp_match
    )

    return {
        "reference_placement": reference_name,
        "scalar_comparison": scalar_comparison,
        "non_mlp_comparison": non_mlp_comparison,
        "mlp_comparison": mlp_comparison,
        "per_layer_mlp_comparison": (
            per_layer_comparison
        ),
        "strong_control_check": {
            "all_reported_scalar_fields_equal": (
                all_scalars_equal
            ),
            "non_mlp_saved_set_identical": (
                all_non_mlp_equal
            ),
            "selected_mlp_layers_match_forced": (
                all_mlp_match
            ),
            "saved_set_differs_only_by_forced_mlp_placement": (
                controlled_difference
            ),
        },
    }


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
        "--backend",
        default="aot_eager",
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
    )

    parser.add_argument(
        "--out",
        default=(
            "results/memory_snapshot/"
            "mlp_placement_sets_b0.05_k8.json"
        ),
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "Experiment requires CUDA."
        )

    if not 0.0 < args.budget < 1.0:
        raise SystemExit(
            "--budget must be strictly between 0 and 1"
        )

    placements = _parse_placements(
        args.placements,
        32,
    )

    results: list[dict[str, Any]] = []

    for placement_name, forced_layers in (
        placements.items()
    ):

        results.append(
            run_capture(
                model_name=args.model,
                scale=args.scale,
                budget=args.budget,
                backend=args.backend,
                placement_name=placement_name,
                forced_layers=forced_layers,
            )
        )

    comparison = compare_results(
        results
    )

    output = {
        "experiment": (
            "Experiment 2.1: exact saved-set "
            "structural verification"
        ),
        "model": args.model,
        "scale": args.scale,
        "budget": args.budget,
        "backend": args.backend,
        "torch_version": torch.__version__,
        "placements": results,
        "comparison": comparison,
    }

    out = Path(args.out)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            output,
            indent=2,
        )
        + "\n"
    )

    checks = comparison[
        "strong_control_check"
    ]

    print()
    print("=" * 100)
    print(
        "EXPERIMENT 2.1 STRUCTURAL VERIFICATION",
        flush=True,
    )
    print("=" * 100)

    print(
        "All reported scalar fields equal:             "
        f"{checks['all_reported_scalar_fields_equal']}",
        flush=True,
    )

    print(
        "Non-MLP selected set identical:               "
        f"{checks['non_mlp_saved_set_identical']}",
        flush=True,
    )

    print(
        "Selected MLP layers == forced layers:         "
        f"{checks['selected_mlp_layers_match_forced']}",
        flush=True,
    )

    print(
        "Saved-set difference ONLY forced MLP placement:"
        f" {checks['saved_set_differs_only_by_forced_mlp_placement']}",
        flush=True,
    )

    print()

    if checks[
        "saved_set_differs_only_by_forced_mlp_placement"
    ]:
        print(
            "CONTROL PASSED: the four plans have the same "
            "non-MLP saved set and differ only in which "
            "8 MLP outputs are saved.",
            flush=True,
        )
        print(
            "Together with Experiment 2's different physical "
            "peaks, this is a clean placement/topology effect.",
            flush=True,
        )
    else:
        print(
            "CONTROL NOT PASSED: inspect the saved-set diff "
            "in the JSON before making a pure-placement claim.",
            flush=True,
        )

    print()
    print(
        f"JSON written to: {out}",
        flush=True,
    )
    print("=" * 100)


if __name__ == "__main__":
    main()

