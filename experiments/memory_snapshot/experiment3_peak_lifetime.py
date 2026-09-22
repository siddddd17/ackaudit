#!/usr/bin/env python3
"""
Experiment 3: allocator/lifetime mechanism comparison for PyTorch #197838.

Purpose
-------
Compare the allocator and generated-graph execution around the physical peak
for two Experiment-2 plans that are structurally controlled:

    early: MLP outputs 0..7 saved
    even:  MLP outputs [0,4,9,13,18,22,27,31] saved

Both plans use:
    budget = 0.05
    k = 8
    same saved_weight = 0.04819231900170282
    same runtime_saved = 44426100736
    same non-MLP saved set

Experiment 2 found:
    early = 1017.997824 MB
    even  = 1042.819072 MB

This experiment records CUDA allocator history for one measured
forward+backward per placement and answers:

1. Which allocations are live at the physical peak?
2. How much of peak is forward-live versus backward-live?
3. Which backward ops dominate the difference?
4. Which recomputed SiLU / MLP-inner tensors are live at the peak?
5. How long do the recomputed SiLU allocations remain live in the backward
   allocator trace (as a fraction of backward trace events, NOT wall time)?
6. Which peak-live allocation signatures are unique to early/even?

The script reuses the exact Experiment-2 FixedPlacementSolver so the logical
saved-set control remains unchanged.

Run from repo root:

    PYTHONUNBUFFERED=1 python -u \
      -m experiments.memory_snapshot.experiment3_peak_lifetime

Default placements are early and even. Custom placement syntax:

    --placements early:0,1,2,3,4,5,6,7 even:0,4,9,13,18,22,27,31

Outputs:

    results/memory_snapshot/mlp_peak_lifetime_b0.05.json
    /tmp/ackaudit_snapshots/mlp_peak_lifetime_<placement>.pickle

Notes
-----
- The allocator trace records requested allocation sizes. The script compares
  its replayed peak against requested-bytes peak and separately reports
  max_memory_allocated(), which counts rounded allocator blocks.
- The backward lifetime percentages are positions in allocator trace events,
  not elapsed execution time.
- CUDA + torch.compile are required on the user's machine.
"""

from __future__ import annotations

import argparse
import collections
import json
import linecache
import pickle
import re
import statistics
from pathlib import Path
from typing import Any

import torch
from torch._functorch import config as functorch_config

from ackaudit.audit import RuntimeRecorder
from ackaudit.capture import _resolve

from experiments.memory_snapshot.sweep_mlp_placement import (
    FixedPlacementSolver,
    _aten_op,
    _parse_placements,
)


MAX_ENTRIES = 2_000_000
FX_FILE = re.compile(r"eval_with_key")
OP_CALL = re.compile(
    r"torch\.ops\.([\w.]+?)\.(?:default|Tensor|Scalar|\w+)\("
)
LHS = re.compile(r"^\s*(\w+)\s*=")


# ---------------------------------------------------------------------------
# Pure trace-analysis helpers
# ---------------------------------------------------------------------------


def replay_peak(
    trace: list[dict],
    free_action: str,
) -> tuple[int, int]:
    """Replay requested allocations/frees and return (peak bytes, index)."""
    live: dict[int, int] = {}
    total = 0
    peak = 0
    peak_idx = -1

    for i, event in enumerate(trace):
        action = event.get("action")

        if action == "alloc":
            live[event["addr"]] = event["size"]
            total += event["size"]

            if total > peak:
                peak = total
                peak_idx = i

        elif action == free_action:
            total -= live.pop(event["addr"], 0)

    return peak, peak_idx


def live_at(
    trace: list[dict],
    upto: int,
    free_action: str,
) -> dict[int, tuple[int, dict]]:
    """Return allocations live immediately after trace[upto]."""
    live: dict[int, tuple[int, dict]] = {}

    for i, event in enumerate(trace[: upto + 1]):
        action = event.get("action")

        if action == "alloc":
            live[event["addr"]] = (i, event)

        elif action == free_action:
            live.pop(event["addr"], None)

    return live


def fx_site(frames: list[dict]) -> tuple[str, int, str]:
    """Return innermost generated-FX frame: (file, line, source)."""
    for frame in frames or []:
        filename = frame.get("filename", "")

        if FX_FILE.search(filename):
            line = int(frame.get("line", 0))
            source = linecache.getline(
                filename,
                line,
            ).strip()
            return filename, line, source

    return "", 0, ""


def op_of(source: str) -> str:
    match = OP_CALL.search(source)
    return match.group(1) if match else "?"


def lhs_of(source: str) -> str:
    match = LHS.match(source)
    return match.group(1) if match else "?"


def silu_layer_from_node(node: str) -> int | None:
    """Map generated FX SiLU node names to Llama layer indices."""
    if node == "silu":
        return 0

    if node.startswith("silu_"):
        suffix = node.split("_", 1)[1]

        if suffix.isdigit():
            layer = int(suffix)

            if 0 <= layer < 32:
                return layer

    return None


def classify_node(node: str, op: str) -> str:
    """Small structural classification for peak-live graph nodes."""
    if silu_layer_from_node(node) is not None:
        return "mlp_silu"

    if op == "mul" and node.startswith("mul"):
        return "mlp_mul_or_elementwise"

    if node.startswith("mm"):
        return "mm"

    if op == "add" and node.startswith("add"):
        return "add"

    return op


def select_free_action(
    trace: list[dict],
    requested_peak: int | None,
    measured_peak: int,
) -> tuple[str, dict[str, dict[str, int]]]:
    """Choose requested/completed free policy closest to allocator peaks."""
    candidates: dict[str, dict[str, int]] = {}

    targets = []
    if requested_peak is not None:
        targets.append(("requested_bytes", requested_peak))
    targets.append(("max_memory_allocated_delta", measured_peak))

    for action in ("free_requested", "free_completed"):
        peak, index = replay_peak(trace, action)

        error = min(
            abs(peak - target)
            for _, target in targets
        )

        candidates[action] = {
            "peak": peak,
            "index": index,
            "error_to_closest_target": error,
        }

    chosen = min(
        candidates,
        key=lambda action: candidates[action][
            "error_to_closest_target"
        ],
    )

    return chosen, candidates


def allocation_signature(
    allocation: dict[str, Any],
) -> tuple:
    """Hashable identity used when comparing live allocations between plans."""
    return (
        allocation["phase"],
        allocation["node"],
        allocation["op"],
        allocation["bytes"],
        allocation.get("line", 0),
    )


def source_lifetimes(
    trace: list[dict],
    boundary: int,
    free_action: str,
) -> list[dict[str, Any]]:
    """Pair allocations with frees and retain generated-FX provenance."""
    live: dict[int, dict[str, Any]] = {}
    lifetimes: list[dict[str, Any]] = []

    for i, event in enumerate(trace):
        action = event.get("action")

        if action == "alloc":
            filename, line, source = fx_site(
                event.get("frames", [])
            )
            node = lhs_of(source)
            op = op_of(source)

            live[event["addr"]] = {
                "addr": event["addr"],
                "bytes": event["size"],
                "alloc_index": i,
                "free_index": None,
                "phase": (
                    "forward"
                    if i < boundary
                    else "backward"
                ),
                "node": node,
                "op": op,
                "graph_file": filename,
                "line": line,
                "source": source,
            }

        elif action == free_action:
            item = live.pop(event["addr"], None)

            if item is not None:
                item["free_index"] = i
                lifetimes.append(item)

    # Allocations still live when tracing ended.
    for item in live.values():
        lifetimes.append(item)

    return lifetimes


def analyze_snapshot(
    *,
    trace: list[dict],
    boundary: int,
    measured_delta: int,
    requested_delta: int | None,
) -> dict[str, Any]:
    """Analyze peak, live allocations, and allocator lifetimes."""
    free_action, replay_candidates = select_free_action(
        trace,
        requested_delta,
        measured_delta,
    )

    replay_peak_bytes = replay_candidates[free_action]["peak"]
    peak_idx = replay_candidates[free_action]["index"]

    if peak_idx < 0:
        raise RuntimeError(
            "allocator trace contains no allocation peak"
        )

    live = live_at(
        trace,
        peak_idx,
        free_action,
    )

    live_allocations: list[dict[str, Any]] = []

    for alloc_idx, event in live.values():
        filename, line, source = fx_site(
            event.get("frames", [])
        )

        node = lhs_of(source)
        op = op_of(source)

        live_allocations.append(
            {
                "phase": (
                    "forward"
                    if alloc_idx < boundary
                    else "backward"
                ),
                "bytes": int(event["size"]),
                "alloc_index": int(alloc_idx),
                "node": node,
                "op": op,
                "class": classify_node(node, op),
                "graph_file": filename,
                "line": int(line),
                "source": source,
                "silu_layer": silu_layer_from_node(node),
                "addr": int(event["addr"]),
            }
        )

    live_allocations.sort(
        key=lambda item: item["alloc_index"]
    )

    # ---------------------------------------------------------------
    # Group live peak allocations.
    # ---------------------------------------------------------------

    phase_groups: dict[str, list[int]] = {
        "forward": [0, 0],
        "backward": [0, 0],
    }

    op_groups: collections.defaultdict[tuple, list[int]] = (
        collections.defaultdict(
            lambda: [0, 0]
        )
    )

    node_groups: collections.defaultdict[tuple, list[int]] = (
        collections.defaultdict(
            lambda: [0, 0]
        )
    )

    for item in live_allocations:
        phase = item["phase"]
        size = item["bytes"]
        op = item["op"]
        node = item["node"]

        phase_groups[phase][0] += size
        phase_groups[phase][1] += 1

        op_groups[(phase, op)][0] += size
        op_groups[(phase, op)][1] += 1

        node_groups[(phase, node, item["class"])][0] += size
        node_groups[(phase, node, item["class"])][1] += 1

    top_ops = sorted(
        (
            {
                "phase": phase,
                "op": op,
                "bytes": values[0],
                "count": values[1],
            }
            for (phase, op), values in op_groups.items()
        ),
        key=lambda item: -item["bytes"],
    )

    # ---------------------------------------------------------------
    # Peak-live SiLU details.
    # ---------------------------------------------------------------

    peak_live_silus = [
        item
        for item in live_allocations
        if item["silu_layer"] is not None
        and item["phase"] == "backward"
    ]

    peak_live_silu_layers = sorted(
        item["silu_layer"]
        for item in peak_live_silus
    )

    silu_bytes = sum(
        item["bytes"]
        for item in peak_live_silus
    )

    # ---------------------------------------------------------------
    # Backward allocator lifetimes for SiLU nodes.
    # ---------------------------------------------------------------

    lifetimes = source_lifetimes(
        trace,
        boundary,
        free_action,
    )

    backward_trace_len = max(
        len(trace) - boundary,
        1,
    )

    silu_lifetimes = []

    for item in lifetimes:
        layer = silu_layer_from_node(
            item["node"]
        )

        if layer is None or item["phase"] != "backward":
            continue

        start = item["alloc_index"] - boundary

        # If it remains live at trace end, use trace end as its release point.
        end_abs = item["free_index"]
        if end_abs is None:
            end_abs = len(trace)

        end = end_abs - boundary

        start = max(start, 0)
        end = max(end, start)

        span = end - start
        fraction = (
            100.0 * span / backward_trace_len
        )

        silu_lifetimes.append(
            {
                "layer": layer,
                "node": item["node"],
                "bytes": item["bytes"],
                "alloc_index": item["alloc_index"],
                "free_index": item["free_index"],
                "backward_start_event": start,
                "backward_end_event": end,
                "span_events": span,
                "span_percent_of_backward_trace": fraction,
                "live_at_peak": any(
                    alloc["node"] == item["node"]
                    and alloc["bytes"] == item["bytes"]
                    for alloc in peak_live_silus
                ),
            }
        )

    silu_lifetimes.sort(
        key=lambda item: item["layer"]
    )

    # Unique peak-live signatures for later plan comparison.
    signatures = collections.Counter(
        allocation_signature(item)
        for item in live_allocations
        if item["phase"] == "backward"
    )

    return {
        "measured_delta_bytes": measured_delta,
        "requested_delta_bytes": requested_delta,
        "replay_free_action": free_action,
        "replay_candidates": replay_candidates,
        "replay_peak_bytes": replay_peak_bytes,
        "replay_matches_requested": (
            requested_delta is not None
            and replay_peak_bytes == requested_delta
        ),
        "peak_index": peak_idx,
        "trace_len": len(trace),
        "boundary_index": boundary,
        "peak_in_backward": peak_idx >= boundary,
        "peak_position_in_backward_percent": (
            100.0 * (peak_idx - boundary)
            / backward_trace_len
            if peak_idx >= boundary
            else None
        ),
        "live_count": len(live_allocations),
        "phase_live": {
            phase: {
                "bytes": values[0],
                "count": values[1],
            }
            for phase, values in phase_groups.items()
        },
        "top_live_ops": top_ops[:30],
        "peak_live_silu": {
            "count": len(peak_live_silus),
            "bytes": silu_bytes,
            "layers": peak_live_silu_layers,
        },
        "silu_lifetimes": silu_lifetimes,
        "peak_live_backward_signatures": [
            {
                "signature": list(signature),
                "count": count,
            }
            for signature, count in signatures.items()
        ],
        "live_allocations": live_allocations,
    }


# ---------------------------------------------------------------------------
# GPU experiment
# ---------------------------------------------------------------------------


def _start_recording() -> None:
    """Start full CUDA allocator recording."""
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="python",
            max_entries=MAX_ENTRIES,
        )
    except TypeError:
        torch.cuda.memory._record_memory_history(
            True,
            trace_alloc_max_entries=MAX_ENTRIES,
            trace_alloc_record_context=True,
        )


def _stop_recording() -> None:
    """Stop CUDA allocator recording."""
    try:
        torch.cuda.memory._record_memory_history(
            enabled=None,
        )
    except TypeError:
        torch.cuda.memory._record_memory_history(False)


def measure_one(
    *,
    model_name: str,
    scale: int,
    budget: float,
    backend: str,
    placement_name: str,
    forced_layers: list[int],
    pickle_dir: Path,
) -> dict[str, Any]:
    """Compile the exact placement plan and record one measured step."""

    print()
    print("=" * 100, flush=True)
    print(
        f"[{placement_name}] START PEAK/LIFETIME SNAPSHOT",
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
    solver_record: dict[str, Any] = {}

    previous_solver = (
        functorch_config.activation_memory_budget_solver
    )
    previous_budget = (
        functorch_config.activation_memory_budget
    )

    solver = FixedPlacementSolver(
        recorder=recorder,
        forced_layers=forced_layers,
        record=solver_record,
        placement_name=placement_name,
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
            previous_solver
        )
        functorch_config.activation_memory_budget = (
            previous_budget
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

    print(
        f"[{placement_name}] starting allocator recording...",
        flush=True,
    )

    _start_recording()

    trace_overflowed = False
    snap: dict[str, Any] | None = None

    try:
        torch.cuda.reset_peak_memory_stats()

        loss = compiled(*args)

        torch.cuda.synchronize()

        # The trace length at this exact point is the forward/backward
        # boundary. This mirrors the existing snapshot methodology.
        forward_snapshot = torch.cuda.memory._snapshot()
        boundary = len(
            forward_snapshot["device_traces"][0]
        )

        print(
            f"[{placement_name}] forward complete; "
            f"trace boundary={boundary}",
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

    trace = snap["device_traces"][0]

    trace_overflowed = len(trace) >= MAX_ENTRIES

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
        f"[{placement_name}] measured max_memory_allocated delta="
        f"{measured_delta / 1e6:.3f} MB",
        flush=True,
    )

    if requested_delta is not None:
        print(
            f"[{placement_name}] requested-bytes delta="
            f"{requested_delta / 1e6:.3f} MB",
            flush=True,
        )

    print(
        f"[{placement_name}] trace length={len(trace):,}; "
        f"overflowed={trace_overflowed}",
        flush=True,
    )

    if trace_overflowed:
        raise RuntimeError(
            f"{placement_name}: allocator trace hit MAX_ENTRIES="
            f"{MAX_ENTRIES}; snapshot cannot be trusted"
        )

    analysis = analyze_snapshot(
        trace=trace,
        boundary=boundary,
        measured_delta=int(measured_delta),
        requested_delta=requested_delta,
    )

    # ---------------------------------------------------------------
    # Report concise mechanism summary.
    # ---------------------------------------------------------------

    phase = analysis["phase_live"]
    silu = analysis["peak_live_silu"]

    print(
        f"[{placement_name}] replay action="
        f"{analysis['replay_free_action']} "
        f"replay peak={analysis['replay_peak_bytes'] / 1e6:.3f} MB",
        flush=True,
    )

    print(
        f"[{placement_name}] peak position in backward="
        f"{analysis['peak_position_in_backward_percent']:.3f}%",
        flush=True,
    )

    print(
        f"[{placement_name}] live at peak="
        f"{analysis['live_count']} allocations; "
        f"forward={phase['forward']['bytes'] / 1e6:.3f} MB; "
        f"backward={phase['backward']['bytes'] / 1e6:.3f} MB",
        flush=True,
    )

    print(
        f"[{placement_name}] backward SiLU live="
        f"{silu['count']} tensors / "
        f"{silu['bytes'] / 1e6:.3f} MB; "
        f"layers={silu['layers']}",
        flush=True,
    )

    print(
        f"[{placement_name}] top peak-live ops:",
        flush=True,
    )

    for group in analysis["top_live_ops"][:10]:
        print(
            f"    {group['phase']:<9} "
            f"{group['op']:<32} "
            f"{group['bytes'] / 1e6:>9.3f} MB "
            f"x{group['count']}",
            flush=True,
        )

    # Save full allocator snapshot outside repo by default.
    pickle_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pickle_path = (
        pickle_dir
        / f"mlp_peak_lifetime_{placement_name}.pickle"
    )

    with pickle_path.open("wb") as handle:
        pickle.dump(snap, handle)

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
        "measured_max_memory_allocated_delta_bytes": int(
            measured_delta
        ),
        "requested_bytes_delta_bytes": requested_delta,
        "trace_overflowed": trace_overflowed,
        "solver": solver_record,
        "analysis": analysis,
        "snapshot_pickle": str(pickle_path),
    }

    del model
    del args
    del compiled
    del loss

    torch.cuda.empty_cache()

    print(
        f"[{placement_name}] snapshot saved to {pickle_path}",
        flush=True,
    )
    print(
        f"[{placement_name}] DONE",
        flush=True,
    )

    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare peak-live signatures and SiLU lifetimes between placements."""
    if len(results) < 2:
        return {
            "error": "comparison requires at least two placements"
        }

    reference = results[0]
    other_results = results[1:]

    ref_analysis = reference["analysis"]

    ref_signatures = collections.Counter()
    for entry in ref_analysis[
        "peak_live_backward_signatures"
    ]:
        ref_signatures[
            tuple(entry["signature"])
        ] = entry["count"]

    comparisons = []

    for result in other_results:
        current_analysis = result["analysis"]

        current_signatures = collections.Counter()
        for entry in current_analysis[
            "peak_live_backward_signatures"
        ]:
            current_signatures[
                tuple(entry["signature"])
            ] = entry["count"]

        only_ref = ref_signatures - current_signatures
        only_current = current_signatures - ref_signatures

        ref_silu_lifetimes = {
            item["layer"]: item
            for item in ref_analysis[
                "silu_lifetimes"
            ]
        }

        current_silu_lifetimes = {
            item["layer"]: item
            for item in current_analysis[
                "silu_lifetimes"
            ]
        }

        all_layers = sorted(
            set(ref_silu_lifetimes)
            | set(current_silu_lifetimes)
        )

        silu_layer_comparison = []

        for layer in all_layers:
            ref_item = ref_silu_lifetimes.get(layer)
            current_item = current_silu_lifetimes.get(layer)

            silu_layer_comparison.append(
                {
                    "layer": layer,
                    "reference": ref_item,
                    "other": current_item,
                    "span_percent_delta": (
                        None
                        if ref_item is None
                        or current_item is None
                        else (
                            current_item[
                                "span_percent_of_backward_trace"
                            ]
                            - ref_item[
                                "span_percent_of_backward_trace"
                            ]
                        )
                    ),
                }
            )

        comparisons.append(
            {
                "reference": reference["placement"],
                "other": result["placement"],
                "measured_peak_delta_bytes": (
                    result[
                        "measured_max_memory_allocated_delta_bytes"
                    ]
                    - reference[
                        "measured_max_memory_allocated_delta_bytes"
                    ]
                ),
                "requested_peak_delta_bytes": (
                    None
                    if result[
                        "requested_bytes_delta_bytes"
                    ]
                    is None
                    or reference[
                        "requested_bytes_delta_bytes"
                    ]
                    is None
                    else (
                        result[
                            "requested_bytes_delta_bytes"
                        ]
                        - reference[
                            "requested_bytes_delta_bytes"
                        ]
                    )
                ),
                "peak_live_byte_delta": (
                    current_analysis[
                        "replay_peak_bytes"
                    ]
                    - ref_analysis[
                        "replay_peak_bytes"
                    ]
                ),
                "backward_live_bytes_delta": (
                    current_analysis["phase_live"][
                        "backward"
                    ]["bytes"]
                    - ref_analysis["phase_live"][
                        "backward"
                    ]["bytes"]
                ),
                "silu_live_bytes_delta": (
                    current_analysis["peak_live_silu"][
                        "bytes"
                    ]
                    - ref_analysis["peak_live_silu"][
                        "bytes"
                    ]
                ),
                "silu_live_count_delta": (
                    current_analysis["peak_live_silu"][
                        "count"
                    ]
                    - ref_analysis["peak_live_silu"][
                        "count"
                    ]
                ),
                "silu_layer_difference": {
                    "only_in_reference": sorted(
                        set(
                            ref_analysis[
                                "peak_live_silu"
                            ]["layers"]
                        )
                        - set(
                            current_analysis[
                                "peak_live_silu"
                            ]["layers"]
                        )
                    ),
                    "only_in_other": sorted(
                        set(
                            current_analysis[
                                "peak_live_silu"
                            ]["layers"]
                        )
                        - set(
                            ref_analysis[
                                "peak_live_silu"
                            ]["layers"]
                        )
                    ),
                },
                "backward_peak_signatures_only_in_reference": [
                    {
                        "signature": list(signature),
                        "count": count,
                    }
                    for signature, count in only_ref.items()
                ],
                "backward_peak_signatures_only_in_other": [
                    {
                        "signature": list(signature),
                        "count": count,
                    }
                    for signature, count in only_current.items()
                ],
                "silu_layer_lifetime_comparison": silu_layer_comparison,
            }
        )

    return {
        "reference": reference["placement"],
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        help=(
            "placement specs name:l0,l1,...; default is early vs even"
        ),
    )

    parser.add_argument(
        "--out",
        default=(
            "results/memory_snapshot/"
            "mlp_peak_lifetime_b0.05.json"
        ),
    )

    parser.add_argument(
        "--pickle-dir",
        default="/tmp/ackaudit_snapshots",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "Experiment 3 requires CUDA."
        )

    placements = _parse_placements(
        args.placements,
        32,
    )

    results: list[dict[str, Any]] = []

    pickle_dir = Path(args.pickle_dir)

    for name, layers in placements.items():
        result = measure_one(
            model_name=args.model,
            scale=args.scale,
            budget=args.budget,
            backend=args.backend,
            placement_name=name,
            forced_layers=layers,
            pickle_dir=pickle_dir,
        )

        results.append(result)

    comparison = compare_results(
        results
    )

    output = {
        "experiment": (
            "Experiment 3: allocator peak/lifetime mechanism "
            "comparison"
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

    # ---------------------------------------------------------------
    # Human-readable final summary.
    # ---------------------------------------------------------------

    print()
    print("=" * 110, flush=True)
    print(
        "EXPERIMENT 3 FINAL SUMMARY",
        flush=True,
    )
    print("=" * 110, flush=True)

    for result in results:
        analysis = result["analysis"]
        solver = result["solver"]
        phase = analysis["phase_live"]
        silu = analysis["peak_live_silu"]

        print(
            f"{result['placement']:<10} "
            f"measured_peak={result['measured_max_memory_allocated_delta_bytes'] / 1e6:9.3f} MB  "
            f"replay_peak={analysis['replay_peak_bytes'] / 1e6:9.3f} MB  "
            f"saved_weight={solver['saved_weight']:.9f}  "
            f"runtime_saved={solver['runtime_saved']:.0f}",
            flush=True,
        )

        print(
            f"    live: forward={phase['forward']['bytes'] / 1e6:.3f} MB "
            f"backward={phase['backward']['bytes'] / 1e6:.3f} MB "
            f"total={analysis['replay_peak_bytes'] / 1e6:.3f} MB",
            flush=True,
        )

        print(
            f"    peak-live backward SiLU: "
            f"{silu['count']} tensors / {silu['bytes'] / 1e6:.3f} MB "
            f"layers={silu['layers']}",
            flush=True,
        )

    for comparison in comparison.get("comparisons", []):
        print()
        print(
            f"COMPARE {comparison['reference']} -> {comparison['other']}",
            flush=True,
        )

        print(
            f"    measured peak delta: "
            f"{comparison['measured_peak_delta_bytes'] / 1e6:+.3f} MB",
            flush=True,
        )

        if comparison["requested_peak_delta_bytes"] is not None:
            print(
                f"    requested peak delta: "
                f"{comparison['requested_peak_delta_bytes'] / 1e6:+.3f} MB",
                flush=True,
            )

        print(
            f"    backward-live delta: "
            f"{comparison['backward_live_bytes_delta'] / 1e6:+.3f} MB",
            flush=True,
        )

        print(
            f"    peak-live SiLU delta: "
            f"{comparison['silu_live_bytes_delta'] / 1e6:+.3f} MB "
            f"({comparison['silu_live_count_delta']:+d} tensors)",
            flush=True,
        )

        print(
            "    SiLU layers only in reference: "
            f"{comparison['silu_layer_difference']['only_in_reference']}",
            flush=True,
        )

        print(
            "    SiLU layers only in other:     "
            f"{comparison['silu_layer_difference']['only_in_other']}",
            flush=True,
        )

        print(
            "    unique backward peak-live signatures in reference: "
            f"{len(comparison['backward_peak_signatures_only_in_reference'])}",
            flush=True,
        )

        print(
            "    unique backward peak-live signatures in other:     "
            f"{len(comparison['backward_peak_signatures_only_in_other'])}",
            flush=True,
        )

    print()
    print(
        f"JSON written to: {out}",
        flush=True,
    )
    print(
        f"Full snapshots: {pickle_dir}",
        flush=True,
    )
    print("=" * 110, flush=True)


if __name__ == "__main__":
    main()

