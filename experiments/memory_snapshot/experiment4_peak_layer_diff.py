"""
Experiment 4: map peak-live backward allocations to logical Llama layers.

Purpose
-------
Experiment 3 established that the early and even plans have:

    - identical budget / saved_weight / runtime_saved
    - identical non-MLP saved-node sets
    - different physical backward peaks
    - different peak-live mm/mul/SiLU populations

This experiment goes one level deeper.

For each controlled plan (default: early vs even), it:

1. Recreates the exact Experiment-3 placement plan.
2. Captures the AOTAutograd joint FX graph during the solver call.
3. Records FX node metadata (module stack, source stack, op, args).
4. Records one allocator trace for a measured forward+backward.
5. Identifies the replayed requested-bytes peak.
6. Maps every peak-live backward allocation to:
      - FX node name
      - aten operation
      - logical Llama layer from nn_module_stack/source metadata when available
      - structural MLP-layer inference where possible
      - generated source line
7. Aggregates peak-live mm/mul/silu/add allocations by logical layer.
8. Produces a pairwise diff: early -> even.

The script deliberately reports "unknown" instead of inventing a layer when
the graph metadata does not support a precise mapping.

Run from repo root:

    PYTHONUNBUFFERED=1 python -u \
      -m experiments.memory_snapshot.experiment4_peak_layer_diff

Output:

    results/memory_snapshot/mlp_peak_layer_diff_b0.05.json

Snapshots:

    /tmp/ackaudit_snapshots/mlp_peak_layer_diff_<placement>.pickle
"""

from __future__ import annotations

import argparse
import collections
import json
import linecache
import pickle
import re
from pathlib import Path
from typing import Any

import torch
from torch._functorch import config as functorch_config

from ackaudit.audit import RuntimeRecorder
from ackaudit.capture import _resolve

from experiments.memory_snapshot.experiment3_peak_lifetime import (
    MAX_ENTRIES,
    analyze_snapshot,
    _start_recording,
    _stop_recording,
    fx_site,
    lhs_of,
    op_of,
)
from experiments.memory_snapshot.sweep_mlp_placement import (
    FixedPlacementSolver,
    _aten_op,
    _mlp_layer_from_output,
    _parse_placements,
)


_LAYER_PATTERNS = (
    re.compile(r"(?:layers|layer)[._\[](\d+)", re.IGNORECASE),
    re.compile(r"(?:layers|layer)[^\d]{1,12}(\d+)", re.IGNORECASE),
)

_MLP_MODULE_PATTERNS = (
    re.compile(r"\.mlp(?:\.|$)", re.IGNORECASE),
    re.compile(r"mlp", re.IGNORECASE),
)

_FORWARD_MLP_NAMES = re.compile(r"^(?:silu(?:_\d+)?|mm(?:_\d+)?)$")


def _stringify(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return repr(value)


def _collect_node_names(value: Any) -> list[str]:
    """Recursively collect FX node names from args/kwargs."""
    out: list[str] = []

    if isinstance(value, torch.fx.Node):
        out.append(value.name)
        return out

    if isinstance(value, (tuple, list)):
        for item in value:
            out.extend(_collect_node_names(item))
        return out

    if isinstance(value, dict):
        for item in value.values():
            out.extend(_collect_node_names(item))
        return out

    return out


def _metadata_text(node: Any) -> str:
    meta = getattr(node, "meta", {}) or {}

    values = [
        meta.get("nn_module_stack"),
        meta.get("source_fn_stack"),
        meta.get("stack_trace"),
        meta.get("source"),
    ]

    return "\n".join(
        _stringify(value)
        for value in values
        if value is not None
    )


def _parse_layer(text: str) -> int | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue

        layer = int(match.group(1))

        if 0 <= layer < 32:
            return layer

    return None


def _tensor_meta(node: Any) -> dict[str, Any]:
    meta = getattr(node, "meta", {}) or {}
    tm = meta.get("tensor_meta")

    if tm is None:
        return {}

    result: dict[str, Any] = {}

    shape = getattr(tm, "shape", None)
    if shape is not None:
        result["shape"] = [int(x) for x in shape]

    dtype = getattr(tm, "dtype", None)
    if dtype is not None:
        result["dtype"] = str(dtype)

    return result


def _node_record(node: Any) -> dict[str, Any]:
    meta = getattr(node, "meta", {}) or {}

    return {
        "name": node.name,
        "op": str(getattr(node, "op", "")),
        "target": _stringify(getattr(node, "target", "")),
        "aten_op": _aten_op(node),
        "args": _collect_node_names(
            getattr(node, "args", ())
        ),
        "kwargs": _collect_node_names(
            getattr(node, "kwargs", {})
        ),
        "metadata_layer": _parse_layer(
            _metadata_text(node)
        ),
        "metadata_text": _metadata_text(node),
        "tensor": _tensor_meta(node),
        "nn_module_stack": _stringify(
            meta.get("nn_module_stack")
        ) if meta.get("nn_module_stack") is not None else None,
        "source_fn_stack": _stringify(
            meta.get("source_fn_stack")
        ) if meta.get("source_fn_stack") is not None else None,
    }


def _build_structural_layer_map(
    graph_nodes: list[dict[str, Any]],
    mlp_output_layers: dict[str, int],
) -> dict[str, list[int]]:
    """
    Conservative structural inference.

    For each graph node, collect MLP-output anchors reachable through its
    transitive dataflow ancestors. This is intentionally reported as a SET,
    not forced into a single layer. A singleton set is a useful exact-ish
    provenance signal; multiple/empty sets stay explicit.
    """
    by_name = {
        item["name"]: item
        for item in graph_nodes
    }

    memo: dict[str, set[int]] = {}

    def ancestors(name: str, active: set[str] | None = None) -> set[int]:
        if name in memo:
            return memo[name]

        if active is None:
            active = set()

        if name in active:
            return set()

        active = set(active)
        active.add(name)

        if name in mlp_output_layers:
            memo[name] = {mlp_output_layers[name]}
            return memo[name]

        node = by_name.get(name)
        if node is None:
            memo[name] = set()
            return memo[name]

        layers: set[int] = set()

        for dep in node["args"] + node["kwargs"]:
            layers.update(
                ancestors(
                    dep,
                    active,
                )
            )

        memo[name] = layers
        return layers

    result: dict[str, list[int]] = {}

    for item in graph_nodes:
        layers = sorted(
            ancestors(item["name"])
        )
        result[item["name"]] = layers

    return result


class GraphCapturingPlacementSolver(FixedPlacementSolver):
    """Exact Experiment-3 solver plus joint-graph provenance capture."""

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
        self.graph_nodes: list[dict[str, Any]] = []

    def __call__(
        self,
        memory,
        joint_graph,
        max_memory,
        node_info,
        all_recomputable_banned_nodes,
    ):
        print(
            f"[{self.placement_name}] capturing joint FX graph metadata...",
            flush=True,
        )

        # AOTAutograd passes the raw torch.fx.Graph to the custom solver,
        # not a GraphModule. Therefore the node container is joint_graph.nodes.
        print(
            f"[{self.placement_name}] joint_graph type={type(joint_graph).__name__}",
            flush=True,
        )

        self.graph_nodes = [
            _node_record(node)
            for node in joint_graph.nodes
        ]

        # MLP output anchors are identified using the same structural helper
        # used by the placement solver.
        anchors: dict[str, int] = {}

        for node in all_recomputable_banned_nodes:
            layer = _mlp_layer_from_output(node)
            if layer is not None:
                anchors[node.name] = layer

        structural = _build_structural_layer_map(
            self.graph_nodes,
            anchors,
        )

        for item in self.graph_nodes:
            item["structural_mlp_layers"] = structural.get(
                item["name"],
                [],
            )

            if item["metadata_layer"] is not None:
                item["logical_layer"] = item["metadata_layer"]
                item["layer_source"] = "module/source metadata"
            elif len(item["structural_mlp_layers"]) == 1:
                item["logical_layer"] = item[
                    "structural_mlp_layers"
                ][0]
                item["layer_source"] = (
                    "transitive MLP-output ancestry"
                )
            else:
                item["logical_layer"] = None
                item["layer_source"] = "unresolved"

        self.record["joint_graph_node_count"] = len(
            self.graph_nodes
        )
        self.record["joint_graph_nodes"] = self.graph_nodes
        self.record["mlp_output_anchor_layers"] = anchors

        return super().__call__(
            memory,
            joint_graph,
            max_memory,
            node_info,
            all_recomputable_banned_nodes,
        )


def _map_live_allocation(
    allocation: dict[str, Any],
    graph_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Attach captured FX provenance to one allocator-live allocation."""
    node_name = allocation["node"]
    graph = graph_by_name.get(node_name)

    result = dict(allocation)

    if graph is None:
        result["graph_match"] = False
        result["logical_layer"] = None
        result["layer_source"] = "no exact FX-node match"
        result["structural_mlp_layers"] = []
        result["graph_target"] = None
        result["graph_args"] = []
        result["graph_kwargs"] = []
        result["module_metadata"] = None
        result["tensor"] = {}
        return result

    result["graph_match"] = True
    result["logical_layer"] = graph["logical_layer"]
    result["layer_source"] = graph["layer_source"]
    result["structural_mlp_layers"] = graph[
        "structural_mlp_layers"
    ]
    result["graph_target"] = graph["target"]
    result["graph_args"] = graph["args"]
    result["graph_kwargs"] = graph["kwargs"]
    result["module_metadata"] = graph[
        "nn_module_stack"
    ]
    result["source_metadata"] = graph[
        "source_fn_stack"
    ]
    result["tensor"] = graph["tensor"]

    return result


def _aggregate_by_layer(
    allocations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Aggregate peak-live backward allocations by logical layer and op.

    Unknown layer remains an explicit "unknown" bucket.
    """
    groups: dict[tuple[Any, str], list[int]] = (
        collections.defaultdict(
            lambda: [0, 0]
        )
    )

    for item in allocations:
        if item["phase"] != "backward":
            continue

        layer = item.get("logical_layer")
        if layer is None:
            layer_key: Any = "unknown"
        else:
            layer_key = int(layer)

        op = item.get("op") or item.get("aten_op") or "?"

        groups[
            (layer_key, op)
        ][0] += int(item["bytes"])

        groups[
            (layer_key, op)
        ][1] += 1

    rows = []

    for (layer, op), (bytes_, count) in groups.items():
        rows.append(
            {
                "layer": layer,
                "op": op,
                "bytes": bytes_,
                "count": count,
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple:
        layer = row["layer"]
        layer_value = 10_000 if layer == "unknown" else int(layer)
        return (
            layer_value,
            row["op"],
        )

    rows.sort(key=sort_key)
    return rows


def _canonical_aten_op(item: dict[str, Any]) -> str | None:
    """Return a canonical op name such as ``mm`` from an allocation record."""
    value = item.get("op")

    if value is None:
        value = item.get("aten_op")

    if value is None:
        return None

    value = str(value)

    if value.startswith("aten."):
        return value.split(".", 1)[1]

    return value


def _op_layer_details(
    allocations: list[dict[str, Any]],
    op: str,
) -> list[dict[str, Any]]:
    """
    Return individual peak-live allocations for one canonical op name,
    grouped by logical layer and sorted largest-first.

    ``analyze_snapshot()`` stores operation names as ``aten.mm``,
    ``aten.mul``, etc.  The caller uses canonical names (``mm``, ``mul``,
    ``silu``, ``add``), so normalize both representations here.
    """
    rows = [
        item
        for item in allocations
        if item["phase"] == "backward"
        and _canonical_aten_op(item) == op
    ]

    rows.sort(
        key=lambda item: (
            10_000
            if item.get("logical_layer") is None
            else int(item["logical_layer"]),
            -int(item["bytes"]),
            int(item["alloc_index"]),
        )
    )

    return rows


def run_one(
    *,
    model_name: str,
    scale: int,
    budget: float,
    backend: str,
    placement_name: str,
    forced_layers: list[int],
    snapshot_dir: Path,
) -> dict[str, Any]:

    print()
    print("=" * 110, flush=True)
    print(
        f"[{placement_name}] START PEAK-LAYER DIFF",
        flush=True,
    )
    print(
        f"[{placement_name}] forced={forced_layers}",
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
        item.cuda()
        for item in make_inputs()
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
    solver_record: dict[str, Any] = {}

    solver = GraphCapturingPlacementSolver(
        recorder=recorder,
        forced_layers=forced_layers,
        record=solver_record,
        placement_name=placement_name,
    )

    prev_solver = (
        functorch_config.activation_memory_budget_solver
    )
    prev_budget = (
        functorch_config.activation_memory_budget
    )

    try:
        functorch_config.activation_memory_budget_solver = solver

        print(
            f"[{placement_name}] warmup/capture...",
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

    torch.cuda.synchronize()

    model.zero_grad(
        set_to_none=False
    )

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    resident = torch.cuda.memory_allocated()

    requested_resident = torch.cuda.memory_stats().get(
        "requested_bytes.all.current"
    )

    print(
        f"[{placement_name}] resident="
        f"{resident / 1e6:.3f} MB",
        flush=True,
    )

    # ------------------------------------------------------------
    # Record exactly one measured forward+backward.
    # ------------------------------------------------------------

    print(
        f"[{placement_name}] starting allocator recording...",
        flush=True,
    )

    _start_recording()

    snap: dict[str, Any] | None = None

    try:
        torch.cuda.reset_peak_memory_stats()

        loss = compiled(*args)

        torch.cuda.synchronize()

        forward_snapshot = torch.cuda.memory._snapshot()

        trace = forward_snapshot[
            "device_traces"
        ][0]

        boundary = len(trace)

        print(
            f"[{placement_name}] "
            f"forward boundary={boundary}",
            flush=True,
        )

        loss.backward()

        torch.cuda.synchronize()

        measured_delta = (
            torch.cuda.max_memory_allocated()
            - resident
        )

        requested_peak = torch.cuda.memory_stats().get(
            "requested_bytes.all.peak"
        )

        snap = torch.cuda.memory._snapshot()

    finally:
        _stop_recording()

    if snap is None:
        raise RuntimeError(
            f"{placement_name}: no allocator snapshot"
        )

    trace = snap["device_traces"][0]

    if len(trace) >= MAX_ENTRIES:
        raise RuntimeError(
            f"{placement_name}: allocator trace reached "
            f"MAX_ENTRIES={MAX_ENTRIES}"
        )

    requested_delta = None

    if (
        requested_peak is not None
        and requested_resident is not None
    ):
        requested_delta = (
            int(requested_peak)
            - int(requested_resident)
        )

    print(
        f"[{placement_name}] "
        f"measured peak delta={measured_delta / 1e6:.3f} MB",
        flush=True,
    )

    if requested_delta is not None:
        print(
            f"[{placement_name}] "
            f"requested peak delta={requested_delta / 1e6:.3f} MB",
            flush=True,
        )

    analysis = analyze_snapshot(
        trace=trace,
        boundary=boundary,
        measured_delta=int(measured_delta),
        requested_delta=(
            int(requested_delta)
            if requested_delta is not None
            else None
        ),
    )

    graph_by_name = {
        item["name"]: item
        for item in solver_record[
            "joint_graph_nodes"
        ]
    }

    mapped_live = [
        _map_live_allocation(
            allocation,
            graph_by_name,
        )
        for allocation in analysis[
            "live_allocations"
        ]
        if allocation["phase"] == "backward"
    ]

    peak_mm_mul = [
        item
        for item in mapped_live
        if _canonical_aten_op(item) in {
            "mm",
            "mul",
            "silu",
            "add",
        }
    ]

    layer_aggregate = _aggregate_by_layer(
        mapped_live
    )

    op_details = {
        op: _op_layer_details(
            mapped_live,
            op,
        )
        for op in (
            "mm",
            "mul",
            "silu",
            "add",
        )
    }

    exact_graph_matches = sum(
        1
        for item in mapped_live
        if item["graph_match"]
    )

    unknown_layer = [
        item
        for item in mapped_live
        if item["logical_layer"] is None
    ]

    print()
    print(
        f"[{placement_name}] "
        f"peak-live backward allocations="
        f"{len(mapped_live)}",
        flush=True,
    )

    print(
        f"[{placement_name}] "
        f"exact FX-node matches="
        f"{exact_graph_matches}/{len(mapped_live)}",
        flush=True,
    )

    print(
        f"[{placement_name}] "
        f"logical layer unresolved="
        f"{len(unknown_layer)}/{len(mapped_live)}",
        flush=True,
    )

    print()
    print(
        f"[{placement_name}] peak-live mm/mul/silu/add by layer:",
        flush=True,
    )

    for row in layer_aggregate:
        print(
            f"    layer={str(row['layer']):>7} "
            f"{row['op']:<6} "
            f"{row['bytes'] / 1e6:>10.3f} MB "
            f"x{row['count']}",
            flush=True,
        )

    for op in (
        "mm",
        "mul",
        "silu",
    ):
        rows = op_details[op]

        print()
        print(
            f"[{placement_name}] individual peak-live "
            f"{op} allocations ({len(rows)} records):",
            flush=True,
        )

        for item in rows:
            print(
                f"    layer={str(item.get('logical_layer')):>7} "
                f"name={item['node']:<12} "
                f"bytes={item['bytes'] / 1e6:>9.3f} MB "
                f"alloc={item['alloc_index']:>6} "
                f"source={item['source']}",
                flush=True,
            )

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_path = (
        snapshot_dir
        / f"mlp_peak_layer_diff_{placement_name}.pickle"
    )

    with snapshot_path.open("wb") as handle:
        pickle.dump(
            snap,
            handle,
        )

    result = {
        "placement": placement_name,
        "forced_mlp_layers": forced_layers,
        "model": model_name,
        "scale": scale,
        "budget": budget,
        "backend": backend,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "resident_bytes": int(resident),
        "requested_resident_bytes": (
            int(requested_resident)
            if requested_resident is not None
            else None
        ),
        "measured_peak_delta_bytes": int(
            measured_delta
        ),
        "requested_peak_delta_bytes": (
            int(requested_delta)
            if requested_delta is not None
            else None
        ),
        "solver": solver_record,
        "analysis": analysis,
        "mapped_peak_live_backward": mapped_live,
        "layer_aggregate": layer_aggregate,
        "op_details": op_details,
        "snapshot_pickle": str(
            snapshot_path
        ),
    }

    del model
    del args
    del compiled
    del loss

    torch.cuda.empty_cache()

    print(
        f"[{placement_name}] "
        f"snapshot={snapshot_path}",
        flush=True,
    )

    print(
        f"[{placement_name}] DONE",
        flush=True,
    )

    return result


def compare(
    results: list[dict[str, Any]],
) -> dict[str, Any]:

    if len(results) < 2:
        return {
            "error": "comparison requires at least two placements"
        }

    ref = results[0]

    comparisons = []

    for other in results[1:]:

        ref_layers = collections.defaultdict(
            lambda: [0, 0]
        )
        other_layers = collections.defaultdict(
            lambda: [0, 0]
        )

        for row in ref["layer_aggregate"]:
            ref_layers[
                (
                    row["layer"],
                    row["op"],
                )
            ][0] += row["bytes"]
            ref_layers[
                (
                    row["layer"],
                    row["op"],
                )
            ][1] += row["count"]

        for row in other["layer_aggregate"]:
            other_layers[
                (
                    row["layer"],
                    row["op"],
                )
            ][0] += row["bytes"]
            other_layers[
                (
                    row["layer"],
                    row["op"],
                )
            ][1] += row["count"]

        keys = sorted(
            set(ref_layers)
            | set(other_layers),
            key=lambda key: (
                10_000
                if key[0] == "unknown"
                else int(key[0]),
                key[1],
            ),
        )

        layer_diffs = []

        for key in keys:

            ref_bytes, ref_count = ref_layers.get(
                key,
                [0, 0],
            )

            other_bytes, other_count = other_layers.get(
                key,
                [0, 0],
            )

            layer_diffs.append(
                {
                    "layer": key[0],
                    "op": key[1],
                    "reference_bytes": ref_bytes,
                    "other_bytes": other_bytes,
                    "delta_bytes": (
                        other_bytes - ref_bytes
                    ),
                    "reference_count": ref_count,
                    "other_count": other_count,
                    "delta_count": (
                        other_count - ref_count
                    ),
                }
            )

        ref_live = {
            (
                item["node"],
                item["bytes"],
                item["op"],
            )
            for item in ref["mapped_peak_live_backward"]
        }

        other_live = {
            (
                item["node"],
                item["bytes"],
                item["op"],
            )
            for item in other["mapped_peak_live_backward"]
        }

        only_ref = sorted(
            ref_live - other_live,
            key=str,
        )

        only_other = sorted(
            other_live - ref_live,
            key=str,
        )

        comparisons.append(
            {
                "reference": ref["placement"],
                "other": other["placement"],
                "measured_peak_delta_bytes": (
                    other[
                        "measured_peak_delta_bytes"
                    ]
                    - ref[
                        "measured_peak_delta_bytes"
                    ]
                ),
                "requested_peak_delta_bytes": (
                    None
                    if (
                        other[
                            "requested_peak_delta_bytes"
                        ]
                        is None
                        or ref[
                            "requested_peak_delta_bytes"
                        ]
                        is None
                    )
                    else (
                        other[
                            "requested_peak_delta_bytes"
                        ]
                        - ref[
                            "requested_peak_delta_bytes"
                        ]
                    )
                ),
                "layer_op_diffs": layer_diffs,
                "peak_live_allocations_only_in_reference": [
                    {
                        "node": item[0],
                        "bytes": item[1],
                        "op": item[2],
                    }
                    for item in only_ref
                ],
                "peak_live_allocations_only_in_other": [
                    {
                        "node": item[0],
                        "bytes": item[1],
                        "op": item[2],
                    }
                    for item in only_other
                ],
            }
        )

    return {
        "comparisons": comparisons
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
            "even:0,4,9,13,18,22,27,31",
        ],
    )

    parser.add_argument(
        "--out",
        default=(
            "results/memory_snapshot/"
            "mlp_peak_layer_diff_b0.05.json"
        ),
    )

    parser.add_argument(
        "--snapshot-dir",
        default="/tmp/ackaudit_snapshots",
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

    if len(placements) < 2:
        raise SystemExit(
            "provide at least two placements"
        )

    results = []

    for name, layers in placements.items():
        results.append(
            run_one(
                model_name=args.model,
                scale=args.scale,
                budget=args.budget,
                backend=args.backend,
                placement_name=name,
                forced_layers=layers,
                snapshot_dir=Path(
                    args.snapshot_dir
                ),
            )
        )

    comparison = compare(results)

    output = {
        "experiment": (
            "Experiment 4: peak-live allocation "
            "layer mapping and pairwise diff"
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

    print()
    print("=" * 115)
    print(
        "EXPERIMENT 4 FINAL SUMMARY",
        flush=True,
    )
    print("=" * 115)

    for result in results:
        analysis = result["analysis"]
        requested_peak = result[
            "requested_peak_delta_bytes"
        ]

        print(
            f"{result['placement']:<10} "
            f"measured_peak="
            f"{result['measured_peak_delta_bytes'] / 1e6:>9.3f} MB "
            f"requested_peak="
            f"{(requested_peak / 1e6 if requested_peak is not None else float('nan')):>9.3f} MB "
            f"live="
            f"{analysis['live_count']:>4}",
            flush=True,
        )

    for comparison in comparison.get(
        "comparisons",
        [],
    ):
        print()
        print(
            f"COMPARE "
            f"{comparison['reference']} -> "
            f"{comparison['other']}",
            flush=True,
        )

        print(
            f"    measured peak delta: "
            f"{comparison['measured_peak_delta_bytes'] / 1e6:+.3f} MB",
            flush=True,
        )

        if comparison[
            "requested_peak_delta_bytes"
        ] is not None:
            print(
                f"    requested peak delta: "
                f"{comparison['requested_peak_delta_bytes'] / 1e6:+.3f} MB",
                flush=True,
            )

        print(
            "    layer/op deltas with non-zero bytes:",
            flush=True,
        )

        for row in comparison["layer_op_diffs"]:
            if row["delta_bytes"] == 0:
                continue

            print(
                f"        layer={str(row['layer']):>7} "
                f"{row['op']:<6} "
                f"delta="
                f"{row['delta_bytes'] / 1e6:+.3f} MB "
                f"count_delta="
                f"{row['delta_count']:+d}",
                flush=True,
            )

        print()
        print(
            "    peak-live allocation signatures only in "
            "reference (first 30):",
            flush=True,
        )

        for item in comparison[
            "peak_live_allocations_only_in_reference"
        ][:30]:
            print(
                f"        {item['node']:<15} "
                f"{item['op']:<8} "
                f"{item['bytes'] / 1e6:.3f} MB",
                flush=True,
            )

        print(
            "    peak-live allocation signatures only in "
            "other (first 30):",
            flush=True,
        )

        for item in comparison[
            "peak_live_allocations_only_in_other"
        ][:30]:
            print(
                f"        {item['node']:<15} "
                f"{item['op']:<8} "
                f"{item['bytes'] / 1e6:.3f} MB",
                flush=True,
            )

    print()
    print(
        f"JSON written to: {out}",
        flush=True,
    )
    print(
        "=" * 115
    )


if __name__ == "__main__":
    main()

