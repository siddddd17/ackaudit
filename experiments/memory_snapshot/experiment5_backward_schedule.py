"""
Experiment 5 (corrected): backward allocator-event schedule reconstruction.

Purpose
-------
Reconstruct the exact allocator-event window around the requested-bytes
backward peak for the two controlled Experiment-2 placements:

    early = [0,1,2,3,4,5,6,7]
    even  = [0,4,9,13,18,22,27,31]

Unlike the previous version, this script DOES NOT depend on a particular
generated-FX filename such as "eval_with_key". The allocator snapshot is
recorded while the generated FX source file is still alive, and provenance
is extracted from any stack frame whose source line actually contains a
generated `torch.ops.aten.*` assignment.

It also uses the captured joint FX graph from Experiment 4 to map generated
node names to logical Llama layers.

The script aborts if provenance resolution is too poor to support the
requested mechanistic analysis instead of silently producing an all-unknown
report.

Run from repo root:

    PYTHONUNBUFFERED=1 python -u \
      -m experiments.memory_snapshot.experiment5_backward_schedule

Defaults:
    budget=0.05
    scale=8
    backend=aot_eager
    window=60 allocator events
    placements=early,even

Outputs:
    results/memory_snapshot/mlp_peak_schedule_diff_b0.05.json

Snapshots:
    /tmp/ackaudit_snapshots/
        mlp_peak_schedule_early.pickle
        mlp_peak_schedule_even.pickle
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
    _start_recording,
    _stop_recording,
)

from experiments.memory_snapshot.experiment4_peak_layer_diff import (
    GraphCapturingPlacementSolver,
    _parse_placements,
)


# A generated FX statement normally looks like:
#
#   silu_8 = torch.ops.aten.silu.default(_unsafe_view_60)
#
# or:
#
#   mm_60 = torch.ops.aten.mm.default(view_99, t_60)
#
_ASSIGN_RE = re.compile(
    r"^\s*(?P<lhs>[A-Za-z_]\w*)\s*="
)

_ATEN_CALL_RE = re.compile(
    r"torch\.ops\.(?P<target>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\("
)

_NODE_TOKEN_RE = re.compile(
    r"\b[A-Za-z_]\w*\b"
)


def _canonical_op(source: str) -> str | None:
    match = _ATEN_CALL_RE.search(source)

    if match is None:
        return None

    target = match.group("target")
    # The regex captures the qualified target after `torch.ops.`:
    #   aten.silu.default
    #   aten.mul.Tensor
    #   aten.mm.default
    #
    # Split off the `aten` namespace, then take the operator name before
    # its overload.
    parts = target.split(".")
    if len(parts) < 2 or parts[0] != "aten":
        return None

    return parts[1]


def _lhs(source: str) -> str | None:
    match = _ASSIGN_RE.match(source)

    if match is None:
        return None

    return match.group("lhs")


def _source_from_frames(
    frames: list[dict[str, Any]] | None,
) -> tuple[str, int, str]:
    """
    Find a generated FX source line without assuming a generated filename.

    This is the critical fix over the previous Experiment-5 version.
    """
    candidates: list[tuple[int, str, int, str]] = []

    for depth, frame in enumerate(frames or []):
        filename = str(frame.get("filename", "") or "")
        line = int(frame.get("line", 0) or 0)

        if not filename or line <= 0:
            continue

        linecache.checkcache(filename)
        source = linecache.getline(
            filename,
            line,
        ).strip()

        if "torch.ops.aten." not in source:
            continue

        lhs = _lhs(source)
        op = _canonical_op(source)

        if lhs is None or op is None:
            continue

        # Prefer the first innermost-looking generated frame. Frame order
        # in CUDA snapshots is stable for this setup; source content is the
        # actual selection criterion, not the filename.
        candidates.append(
            (
                depth,
                filename,
                line,
                source,
            )
        )

    if not candidates:
        return "", 0, ""

    _, filename, line, source = candidates[0]
    return filename, line, source


def _node_dependencies(source: str) -> list[str]:
    """
    Extract likely FX node dependencies from a generated aten assignment.

    Everything inside the call after the operation name is considered an
    identifier candidate. Constants are harmless because only names that
    appear in the captured graph map are retained later.
    """
    match = _ATEN_CALL_RE.search(source)

    if match is None:
        return []

    rhs = source[match.end():]

    return _NODE_TOKEN_RE.findall(rhs)


def _mlp_layer_from_name(node: str) -> int | None:
    if node == "silu":
        return 0

    if node.startswith("silu_"):
        suffix = node.split("_", 1)[1]

        if suffix.isdigit():
            layer = int(suffix)

            if 0 <= layer < 32:
                return layer

    return None


def _build_graph_provenance(
    joint_graph_nodes: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Build node -> provenance from Experiment-4 joint FX metadata.

    Prefer Experiment-4's logical_layer field. Fall back to direct SiLU
    naming when available.
    """
    result: dict[str, dict[str, Any]] = {}

    for node in joint_graph_nodes:
        name = str(node.get("name", ""))

        if not name:
            continue

        logical_layer = node.get("logical_layer")

        if logical_layer is None:
            logical_layer = _mlp_layer_from_name(name)

        result[name] = {
            "name": name,
            "logical_layer": logical_layer,
            "layer_source": node.get(
                "layer_source",
                "Experiment-4 joint-graph metadata",
            ),
            "aten_op": node.get("aten_op"),
            "target": node.get("target"),
            "tensor": node.get("tensor", {}),
        }

    return result


def _collect_source_records(
    trace: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Collect generated-node source records from the live allocator trace.

    Unlike the broken implementation, this does not require a specific
    filename. It only trusts a source line if it contains an actual aten
    operator assignment.
    """
    records: dict[str, dict[str, Any]] = {}

    for event in trace:
        if event.get("action") != "alloc":
            continue

        filename, line, source = _source_from_frames(
            event.get("frames", [])
        )

        if not source:
            continue

        node = _lhs(source)
        op = _canonical_op(source)

        if node is None or op is None:
            continue

        records.setdefault(
            node,
            {
                "node": node,
                "op": op,
                "source": source,
                "filename": filename,
                "line": line,
                "dependencies": _node_dependencies(source),
            },
        )

    return records


def _attach_provenance(
    source_records: dict[str, dict[str, Any]],
    graph_provenance: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Attach direct logical-layer provenance only.

    Do not recursively union dependency labels: residual/normalization
    tensors can span many transformer blocks, producing meaningless labels
    such as ``0,1,2,...,31``. Trust exact joint-FX metadata, then use only
    direct ``silu_N`` naming as a narrow fallback; otherwise remain unknown.
    """
    result = {}
    for name, record in source_records.items():
        item = dict(record)
        graph_item = graph_provenance.get(name)
        if graph_item is not None:
            layer = graph_item.get("logical_layer")
            if isinstance(layer, int) and 0 <= layer < 32:
                item["logical_layer"] = layer
                item["layer_source"] = "joint FX graph"
            else:
                item["logical_layer"] = None
                item["layer_source"] = "joint FX graph: no layer"
        else:
            direct = _mlp_layer_from_name(name)
            if direct is not None:
                item["logical_layer"] = direct
                item["layer_source"] = "direct silu node naming"
            else:
                item["logical_layer"] = None
                item["layer_source"] = "unresolved"
        result[name] = item
    return result


def _event_node(
    event: dict[str, Any],
    provenance: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    filename, line, source = _source_from_frames(
        event.get("frames", [])
    )

    node = _lhs(source) or "?"
    op = _canonical_op(source) or "?"

    graph_item = provenance.get(node, {})

    layer = graph_item.get(
        "logical_layer"
    )

    if layer is None:
        direct = _mlp_layer_from_name(node)
        layer = direct

    return {
        "node": node,
        "op": op,
        "logical_layer": layer,
        "layer_source": graph_item.get(
            "layer_source",
            "unresolved",
        ),
        "graph_match": node in provenance,
        "filename": filename,
        "line": int(line),
        "source": source,
    }


def _replay(
    *,
    trace: list[dict[str, Any]],
    boundary: int,
    free_action: str,
    provenance: dict[str, dict[str, Any]],
) -> tuple[
    int,
    int,
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    """
    Replay allocator trace.

    Returns:
        peak_bytes,
        peak_idx,
        peak_live,
        lifetime_rows,
        event_rows
    """
    active: dict[int, dict[str, Any]] = {}
    lifetime_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    total = 0
    peak = 0
    peak_idx = -1
    peak_live: dict[int, dict[str, Any]] = {}

    for i, event in enumerate(trace):

        action = event.get("action")
        addr = int(event.get("addr", -1))

        if action == "alloc":

            size = int(event.get("size", 0))
            fx = _event_node(
                event,
                provenance,
            )

            item = {
                "addr": addr,
                "alloc_index": i,
                "bytes": size,
                "phase": (
                    "forward"
                    if i < boundary
                    else "backward"
                ),
                "node": fx["node"],
                "op": fx["op"],
                "logical_layer": fx[
                    "logical_layer"
                ],
                "layer_source": fx[
                    "layer_source"
                ],
                "graph_match": fx[
                    "graph_match"
                ],
                "filename": fx[
                    "filename"
                ],
                "line": fx["line"],
                "source": fx[
                    "source"
                ],
                "free_index": None,
            }

            active[addr] = item
            total += size

            event_rows.append(
                {
                    "event_index": i,
                    "backward_event_index": (
                        i - boundary
                        if i >= boundary
                        else None
                    ),
                    "relative_to_peak": None,
                    "phase": item["phase"],
                    "action": "alloc",
                    "addr": addr,
                    "bytes": size,
                    "delta_bytes": size,
                    "live_bytes_after": total,
                    "live_count_after": len(active),
                    **fx,
                }
            )

            if total > peak:
                peak = total
                peak_idx = i
                peak_live = {
                    live_addr: dict(live_item)
                    for live_addr, live_item
                    in active.items()
                }

        elif action == free_action:

            item = active.pop(addr, None)

            if item is not None:
                size = int(item["bytes"])
                item["free_index"] = i
                lifetime_rows.append(
                    dict(item)
                )

                fx = {
                    "node": item["node"],
                    "op": item["op"],
                    "logical_layer": item[
                        "logical_layer"
                    ],
                    "layer_source": item[
                        "layer_source"
                    ],
                    "graph_match": item[
                        "graph_match"
                    ],
                    "filename": item[
                        "filename"
                    ],
                    "line": item["line"],
                    "source": item[
                        "source"
                    ],
                }

            else:
                size = int(event.get("size", 0))
                fx = _event_node(
                    event,
                    provenance,
                )

            total -= size

            event_rows.append(
                {
                    "event_index": i,
                    "backward_event_index": (
                        i - boundary
                        if i >= boundary
                        else None
                    ),
                    "relative_to_peak": None,
                    "phase": (
                        "forward"
                        if i < boundary
                        else "backward"
                    ),
                    "action": free_action,
                    "addr": addr,
                    "bytes": size,
                    "delta_bytes": -size,
                    "live_bytes_after": total,
                    "live_count_after": len(active),
                    **fx,
                }
            )

    # Anything still live at trace end gets a null free index.
    for item in active.values():
        lifetime_rows.append(
            dict(item)
        )

    for row in event_rows:
        row["relative_to_peak"] = (
            row["event_index"] - peak_idx
        )

    return (
        peak,
        peak_idx,
        peak_live,
        lifetime_rows,
        event_rows,
    )


def _window_rows(
    event_rows: list[dict[str, Any]],
    *,
    peak_idx: int,
    window: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in event_rows
        if abs(
            int(row["event_index"]) - peak_idx
        ) <= window
    ]


def _peak_live_rows(
    peak_live: dict[int, dict[str, Any]],
    lifetime_rows: list[dict[str, Any]],
    peak_idx: int,
) -> list[dict[str, Any]]:
    free_by_alloc = {
        int(item["alloc_index"]): item.get(
            "free_index"
        )
        for item in lifetime_rows
    }

    rows = []

    for item in peak_live.values():
        row = dict(item)

        free_index = free_by_alloc.get(
            int(item["alloc_index"])
        )

        row["age_at_peak_events"] = (
            peak_idx
            - int(item["alloc_index"])
        )

        row["free_distance_after_peak_events"] = (
            None
            if free_index is None
            else int(free_index) - peak_idx
        )

        rows.append(row)

    rows.sort(
        key=lambda item: (
            -int(item["bytes"]),
            int(item["alloc_index"]),
        )
    )

    return rows


def _summarize_layers(
    peak_live_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str],
        list[int],
    ] = collections.defaultdict(
        lambda: [0, 0]
    )

    for item in peak_live_rows:

        if item["phase"] != "backward":
            continue

        layer = item.get(
            "logical_layer"
        )

        layer_text = (
            "unknown"
            if layer is None
            else str(layer)
        )

        op = str(
            item.get("op") or "?"
        )

        groups[
            (layer_text, op)
        ][0] += int(item["bytes"])

        groups[
            (layer_text, op)
        ][1] += 1

    rows = []

    for (layer, op), (
        bytes_,
        count,
    ) in groups.items():
        rows.append(
            {
                "layer": layer,
                "op": op,
                "bytes": bytes_,
                "count": count,
            }
        )

    def key(row: dict[str, Any]):
        try:
            layer = int(row["layer"])
        except Exception:
            layer = 10000

        return (
            layer,
            row["op"],
        )

    rows.sort(key=key)

    return rows


def _provenance_quality(
    event_rows: list[dict[str, Any]],
    peak_live_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    backward = [
        row
        for row in event_rows
        if row["phase"] == "backward"
        and row["action"] == "alloc"
    ]

    peak_backward = [
        row
        for row in peak_live_rows
        if row["phase"] == "backward"
    ]

    source_resolved = sum(
        bool(row["source"])
        for row in backward
    )

    graph_matched = sum(
        bool(row["graph_match"])
        for row in peak_backward
    )

    layer_resolved = sum(
        row.get("logical_layer") is not None
        for row in peak_backward
    )

    return {
        "backward_allocations": len(backward),
        "backward_source_resolved": source_resolved,
        "backward_source_resolution_pct": (
            100.0
            * source_resolved
            / max(len(backward), 1)
        ),
        "peak_live_backward": len(
            peak_backward
        ),
        "peak_live_graph_matched": graph_matched,
        "peak_live_graph_match_pct": (
            100.0
            * graph_matched
            / max(len(peak_backward), 1)
        ),
        "peak_live_layer_resolved": layer_resolved,
        "peak_live_layer_resolution_pct": (
            100.0
            * layer_resolved
            / max(len(peak_backward), 1)
        ),
    }


def _analyze_one(
    *,
    model_name: str,
    scale: int,
    budget: float,
    backend: str,
    placement_name: str,
    forced_layers: list[int],
    window: int,
    snapshot_dir: Path,
    min_source_resolution_pct: float,
    min_peak_graph_match_pct: float,
) -> dict[str, Any]:

    print()
    print("=" * 110, flush=True)
    print(
        f"[{placement_name}] START EXPERIMENT 5",
        flush=True,
    )
    print(
        f"[{placement_name}] forced={forced_layers}",
        flush=True,
    )

    torch.manual_seed(197838)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            197838
        )

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

    solver = GraphCapturingPlacementSolver(
        recorder=recorder,
        forced_layers=forced_layers,
        record=solver_record,
        placement_name=placement_name,
    )

    previous_solver = (
        functorch_config.activation_memory_budget_solver
    )
    previous_budget = (
        functorch_config.activation_memory_budget
    )

    try:
        functorch_config.activation_memory_budget_solver = solver

        print(
            f"[{placement_name}] "
            f"warmup/capture...",
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
        f"[{placement_name}] "
        f"resident={resident / 1e6:.3f} MB",
        flush=True,
    )

    # ------------------------------------------------------------
    # One measured run, with allocator recording active.
    # Crucially, the generated FX source files are still present
    # while _snapshot() is captured and parsed.
    # ------------------------------------------------------------

    print(
        f"[{placement_name}] "
        f"starting allocator recording...",
        flush=True,
    )

    _start_recording()

    try:
        torch.cuda.reset_peak_memory_stats()

        loss = compiled(*args)

        torch.cuda.synchronize()

        forward_trace = (
            torch.cuda.memory._snapshot()
            ["device_traces"][0]
        )

        boundary = len(forward_trace)

        print(
            f"[{placement_name}] "
            f"forward boundary={boundary}",
            flush=True,
        )

        loss.backward()

        torch.cuda.synchronize()

        measured_peak = (
            torch.cuda.max_memory_allocated()
            - resident
        )

        requested_peak = (
            torch.cuda.memory_stats().get(
                "requested_bytes.all.peak"
            )
        )

        snapshot = torch.cuda.memory._snapshot()

    finally:
        _stop_recording()

    trace = snapshot["device_traces"][0]

    if len(trace) >= MAX_ENTRIES:
        raise RuntimeError(
            f"{placement_name}: "
            f"trace reached MAX_ENTRIES={MAX_ENTRIES}"
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

    graph_provenance = (
        _build_graph_provenance(
            solver_record[
                "joint_graph_nodes"
            ]
        )
    )

    source_records = _collect_source_records(
        trace
    )

    provenance = _attach_provenance(
        source_records,
        graph_provenance,
    )

    peak_bytes, peak_idx, peak_live, lifetime_rows, event_rows = _replay(
        trace=trace,
        boundary=boundary,
        free_action="free_requested",
        provenance=provenance,
    )

    peak_live_rows = _peak_live_rows(
        peak_live,
        lifetime_rows,
        peak_idx,
    )

    quality = _provenance_quality(
        event_rows,
        peak_live_rows,
    )

    print(
        f"[{placement_name}] "
        f"measured_peak={measured_peak / 1e6:.3f} MB",
        flush=True,
    )

    print(
        f"[{placement_name}] "
        f"requested_peak={requested_delta / 1e6:.3f} MB"
        if requested_delta is not None
        else (
            f"[{placement_name}] requested_peak unavailable"
        ),
        flush=True,
    )

    print(
        f"[{placement_name}] "
        f"replayed requested peak="
        f"{peak_bytes / 1e6:.3f} MB "
        f"at event={peak_idx}",
        flush=True,
    )

    print(
        f"[{placement_name}] "
        f"backward source resolution="
        f"{quality['backward_source_resolution_pct']:.2f}%",
        flush=True,
    )

    print(
        f"[{placement_name}] "
        f"peak-live graph match="
        f"{quality['peak_live_graph_match_pct']:.2f}%",
        flush=True,
    )

    if (
        quality[
            "backward_source_resolution_pct"
        ]
        < min_source_resolution_pct
    ):
        raise RuntimeError(
            f"{placement_name}: generated-FX source resolution "
            f"only "
            f"{quality['backward_source_resolution_pct']:.2f}% "
            f"(required >= {min_source_resolution_pct:.2f}%). "
            f"Do not interpret this run."
        )

    if (
        quality[
            "peak_live_graph_match_pct"
        ]
        < min_peak_graph_match_pct
    ):
        raise RuntimeError(
            f"{placement_name}: peak-live graph matching only "
            f"{quality['peak_live_graph_match_pct']:.2f}% "
            f"(required >= {min_peak_graph_match_pct:.2f}%). "
            f"Do not interpret this run."
        )

    backward_len = (
        len(trace) - boundary
    )

    backward_peak_event = (
        peak_idx - boundary
    )

    peak_position_pct = (
        100.0
        * backward_peak_event
        / max(backward_len, 1)
        if peak_idx >= boundary
        else None
    )

    window_rows = _window_rows(
        event_rows,
        peak_idx=peak_idx,
        window=window,
    )

    # Mark each event's distance from the peak.
    for row in window_rows:
        row["relative_to_peak"] = (
            int(row["event_index"]) - peak_idx
        )

    peak_mlp = [
        row
        for row in peak_live_rows
        if row["phase"] == "backward"
        and row.get("op") in {
            "mm",
            "mul",
            "silu",
            "rsqrt",
            "add",
        }
    ]

    layer_summary = _summarize_layers(
        peak_live_rows
    )

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_path = (
        snapshot_dir
        / f"mlp_peak_schedule_{placement_name}.pickle"
    )

    with snapshot_path.open("wb") as handle:
        pickle.dump(
            snapshot,
            handle,
        )

    print()
    print(
        f"[{placement_name}] peak-live MLP allocations:",
        flush=True,
    )

    for item in peak_mlp:
        free_distance = item.get(
            "free_distance_after_peak_events"
        )

        free_text = (
            "still-live"
            if free_distance is None
            else f"free=+{free_distance}"
        )

        print(
            f"    layer={str(item.get('logical_layer')):>6} "
            f"op={str(item.get('op')):<5} "
            f"node={str(item.get('node')):<12} "
            f"size={item['bytes'] / 1e6:>8.3f} MB "
            f"age={item['age_at_peak_events']:>5} "
            f"{free_text}",
            flush=True,
        )

    print()
    print(
        f"[{placement_name}] critical MLP schedule "
        f"(±{window} allocator events):",
        flush=True,
    )

    for row in window_rows:
        if (
            row["phase"] != "backward"
            or row["op"] not in {
                "mm",
                "mul",
                "silu",
                "rsqrt",
                "add",
            }
        ):
            continue

        print(
            f"    rel={row['relative_to_peak']:>4} "
            f"back={str(row['backward_event_index']):>5} "
            f"{row['action']:<14} "
            f"Δ={row['delta_bytes'] / 1e6:+8.3f} MB "
            f"live={row['live_bytes_after'] / 1e6:>9.3f} MB "
            f"layer={str(row['logical_layer']):>6} "
            f"op={str(row['op']):<6} "
            f"node={str(row['node']):<12}",
            flush=True,
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
            None
            if requested_resident is None
            else int(requested_resident)
        ),
        "measured_peak_delta_bytes": int(
            measured_peak
        ),
        "requested_peak_delta_bytes": (
            None
            if requested_delta is None
            else int(requested_delta)
        ),
        "replayed_requested_peak_bytes": int(
            peak_bytes
        ),
        "peak_event_index": int(
            peak_idx
        ),
        "boundary": int(boundary),
        "backward_peak_event": int(
            backward_peak_event
        ),
        "backward_trace_len": int(
            backward_len
        ),
        "peak_position_in_backward_percent": (
            peak_position_pct
        ),
        "provenance_quality": quality,
        "solver": solver_record,
        "peak_live_backward": peak_live_rows,
        "peak_layer_op_summary": layer_summary,
        "critical_window": window_rows,
        "peak_mlp_allocations": peak_mlp,
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


def _compare(
    early: dict[str, Any],
    even: dict[str, Any],
) -> dict[str, Any]:

    def summary_map(
        result: dict[str, Any]
    ) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            (
                str(row["layer"]),
                str(row["op"]),
            ): row
            for row in result[
                "peak_layer_op_summary"
            ]
        }

    e = summary_map(early)
    v = summary_map(even)

    keys = sorted(
        set(e) | set(v),
        key=lambda key: (
            10_000
            if key[0] == "unknown"
            else int(key[0].split(",")[0]),
            key[1],
        ),
    )

    layer_op_diffs = []

    for layer, op in keys:
        a = e.get(
            (layer, op),
            {
                "bytes": 0,
                "count": 0,
            },
        )

        b = v.get(
            (layer, op),
            {
                "bytes": 0,
                "count": 0,
            },
        )

        delta_bytes = (
            int(b["bytes"])
            - int(a["bytes"])
        )

        delta_count = (
            int(b["count"])
            - int(a["count"])
        )

        if (
            delta_bytes == 0
            and delta_count == 0
        ):
            continue

        layer_op_diffs.append(
            {
                "layer": layer,
                "op": op,
                "early_bytes": int(
                    a["bytes"]
                ),
                "even_bytes": int(
                    b["bytes"]
                ),
                "delta_bytes": delta_bytes,
                "early_count": int(
                    a["count"]
                ),
                "even_count": int(
                    b["count"]
                ),
                "delta_count": delta_count,
            }
        )

    # Compare exact peak-live generated node signatures rather than FX
    # allocation indices. Node names are stable in this graph and are much
    # more meaningful for the mechanism investigation.
    def node_sigs(
        result: dict[str, Any]
    ) -> set[tuple]:
        return {
            (
                str(item.get("node")),
                str(item.get("op")),
                str(item.get("logical_layer")),
                int(item["bytes"]),
            )
            for item in result[
                "peak_live_backward"
            ]
        }

    early_sigs = node_sigs(early)
    even_sigs = node_sigs(even)

    only_early = sorted(
        early_sigs - even_sigs,
        key=str,
    )

    only_even = sorted(
        even_sigs - early_sigs,
        key=str,
    )

    return {
        "requested_peak_delta_bytes": (
            int(
                even[
                    "replayed_requested_peak_bytes"
                ]
            )
            - int(
                early[
                    "replayed_requested_peak_bytes"
                ]
            )
        ),
        "requested_peak_delta_mb": (
            int(
                even[
                    "replayed_requested_peak_bytes"
                ]
            )
            - int(
                early[
                    "replayed_requested_peak_bytes"
                ]
            )
        )
        / 1e6,
        "measured_peak_delta_bytes": (
            int(
                even[
                    "measured_peak_delta_bytes"
                ]
            )
            - int(
                early[
                    "measured_peak_delta_bytes"
                ]
            )
        ),
        "measured_peak_delta_mb": (
            int(
                even[
                    "measured_peak_delta_bytes"
                ]
            )
            - int(
                early[
                    "measured_peak_delta_bytes"
                ]
            )
        )
        / 1e6,
        "backward_peak_event_delta": (
            int(
                even[
                    "backward_peak_event"
                ]
            )
            - int(
                early[
                    "backward_peak_event"
                ]
            )
        ),
        "peak_position_delta_percentage_points": (
            float(
                even[
                    "peak_position_in_backward_percent"
                ]
            )
            - float(
                early[
                    "peak_position_in_backward_percent"
                ]
            )
        ),
        "layer_op_diffs": layer_op_diffs,
        "peak_live_only_early": [
            {
                "node": item[0],
                "op": item[1],
                "logical_layer": item[2],
                "bytes": item[3],
            }
            for item in only_early
        ],
        "peak_live_only_even": [
            {
                "node": item[0],
                "op": item[1],
                "logical_layer": item[2],
                "bytes": item[3],
            }
            for item in only_even
        ],
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
        "--window",
        type=int,
        default=60,
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
        "--snapshot-dir",
        type=Path,
        default=Path(
            "/tmp/ackaudit_snapshots"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/memory_snapshot/"
            "mlp_peak_schedule_diff_b0.05.json"
        ),
    )

    parser.add_argument(
        "--min-source-resolution-pct",
        type=float,
        default=90.0,
    )

    parser.add_argument(
        "--min-peak-graph-match-pct",
        type=float,
        default=90.0,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "Experiment requires CUDA."
        )

    if args.window < 0:
        raise SystemExit(
            "--window must be >= 0"
        )

    placements = _parse_placements(
        args.placements,
        32,
    )

    if len(placements) < 2:
        raise SystemExit(
            "provide at least two placements"
        )

    # Regression guard: never silently classify ATen overloads as operators.
    parser_examples = {
        "torch.ops.aten.silu.default(x)": "silu",
        "torch.ops.aten.mul.Tensor(x, y)": "mul",
        "torch.ops.aten.mm.default(x, y)": "mm",
    }

    for source, expected in parser_examples.items():
        actual = _canonical_op(source)
        if actual != expected:
            raise RuntimeError(
                f"ATen parser self-check failed: "
                f"{source!r} -> {actual!r}, expected {expected!r}"
            )

    print("=" * 110)
    print(
        "EXPERIMENT 5: BACKWARD ALLOCATOR-EVENT "
        "SCHEDULE RECONSTRUCTION",
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
        f"budget={args.budget}",
        flush=True,
    )
    print(
        f"window=±{args.window} allocator events",
        flush=True,
    )
    print(
        "provenance source selection: generated aten source line "
        "(filename-independent)",
        flush=True,
    )
    print("=" * 110)

    results: list[dict[str, Any]] = []

    for placement_name, forced_layers in (
        placements.items()
    ):
        results.append(
            _analyze_one(
                model_name=args.model,
                scale=args.scale,
                budget=args.budget,
                backend=args.backend,
                placement_name=placement_name,
                forced_layers=forced_layers,
                window=args.window,
                snapshot_dir=args.snapshot_dir,
                min_source_resolution_pct=(
                    args.min_source_resolution_pct
                ),
                min_peak_graph_match_pct=(
                    args.min_peak_graph_match_pct
                ),
            )
        )

    comparison = _compare(
        results[0],
        results[1],
    )

    output = {
        "experiment": (
            "Experiment 5: backward allocator-event "
            "schedule reconstruction"
        ),
        "model": args.model,
        "scale": args.scale,
        "budget": args.budget,
        "backend": args.backend,
        "window": args.window,
        "placements": results,
        "comparison": comparison,
    }

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.out.write_text(
        json.dumps(
            output,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 115)
    print(
        "EXPERIMENT 5 FINAL SUMMARY",
        flush=True,
    )
    print("=" * 115)

    for result in results:

        print(
            f"{result['placement']:<10} "
            f"requested_peak="
            f"{result['replayed_requested_peak_bytes'] / 1e6:>9.3f} MB "
            f"measured_peak="
            f"{result['measured_peak_delta_bytes'] / 1e6:>9.3f} MB "
            f"backward_event="
            f"{result['backward_peak_event']:>5}",
            flush=True,
        )

        quality = result[
            "provenance_quality"
        ]

        print(
            f"    source_resolution="
            f"{quality['backward_source_resolution_pct']:.2f}% "
            f"peak_graph_match="
            f"{quality['peak_live_graph_match_pct']:.2f}% "
            f"peak_layer_resolution="
            f"{quality['peak_live_layer_resolution_pct']:.2f}%",
            flush=True,
        )

    print()
    print(
        f"COMPARE "
        f"{results[0]['placement']} -> "
        f"{results[1]['placement']}",
        flush=True,
    )

    print(
        f"    requested peak delta: "
        f"{comparison['requested_peak_delta_mb']:+.3f} MB",
        flush=True,
    )

    print(
        f"    measured peak delta: "
        f"{comparison['measured_peak_delta_mb']:+.3f} MB",
        flush=True,
    )

    print(
        f"    backward peak-event delta: "
        f"{comparison['backward_peak_event_delta']:+d}",
        flush=True,
    )

    print(
        f"    peak-position delta: "
        f"{comparison['peak_position_delta_percentage_points']:+.3f} "
        f"percentage points",
        flush=True,
    )

    print()
    print(
        "NON-ZERO LAYER/OP DELTAS",
        flush=True,
    )

    for row in comparison[
        "layer_op_diffs"
    ]:
        print(
            f"    layer={str(row['layer']):>7} "
            f"op={row['op']:<8} "
            f"delta="
            f"{row['delta_bytes'] / 1e6:+.3f} MB "
            f"count_delta="
            f"{row['delta_count']:+d}",
            flush=True,
        )

    print()
    print(
        "PEAK-LIVE NODES ONLY IN REFERENCE",
        flush=True,
    )

    for item in comparison[
        "peak_live_only_early"
    ][:60]:
        print(
            f"    layer={str(item['logical_layer']):>6} "
            f"{item['op']:<8} "
            f"{item['node']:<15} "
            f"{item['bytes'] / 1e6:.3f} MB",
            flush=True,
        )

    print()
    print(
        "PEAK-LIVE NODES ONLY IN OTHER",
        flush=True,
    )

    for item in comparison[
        "peak_live_only_even"
    ][:60]:
        print(
            f"    layer={str(item['logical_layer']):>6} "
            f"{item['op']:<8} "
            f"{item['node']:<15} "
            f"{item['bytes'] / 1e6:.3f} MB",
            flush=True,
        )

    print()
    print(
        f"JSON written to: {args.out}",
        flush=True,
    )
    print("=" * 115)


if __name__ == "__main__":
    main()

