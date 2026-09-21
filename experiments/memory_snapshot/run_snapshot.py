"""What is resident at the backward peak?

pytorch/pytorch#197838 reports that on llama the incremental peak allocated
memory is 4.17x higher at activation_memory_budget 0.05 than at 0.15, and stops
short of saying what the extra memory is. This records CUDA allocator history
around one measured forward and backward and replays it to find the allocations
live at the peak.

Each live allocation is attributed two ways:

- phase: allocated during the forward, or during the backward. The forward/
  backward boundary is taken from the trace length immediately after the
  compiled forward returns.
- site: the line of FX-generated graph code that allocated it, recovered from
  the allocation's Python stack.

The replayed peak is checked against torch.cuda.max_memory_allocated() for the
same step. If they disagree, nothing below the check should be trusted.

    python -m experiments.memory_snapshot.run_snapshot --budgets 0.05 0.10 0.15
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

from ackaudit.capture import _resolve

MAX_ENTRIES = 2_000_000
FX_FILE = re.compile(r"eval_with_key")
OP_CALL = re.compile(r"torch\.ops\.([\w.]+?)\.(?:default|Tensor|Scalar|\w+)\(")
LHS = re.compile(r"^\s*(\w+)\s*=")


# --------------------------------------------------------------------------
# analysis: pure functions over a device trace, tested without a GPU
# --------------------------------------------------------------------------

def replay_peak(trace: list[dict], free_action: str) -> tuple[int, int]:
    """Replay allocs and frees; return (peak bytes, index of the alloc that set it).

    Frees of addresses never allocated inside the trace (tensors that existed
    before recording started) are ignored, so the result is incremental.
    """
    live: dict[int, int] = {}
    total = peak = 0
    peak_idx = -1
    for i, e in enumerate(trace):
        act = e.get("action")
        if act == "alloc":
            live[e["addr"]] = e["size"]
            total += e["size"]
            if total > peak:
                peak, peak_idx = total, i
        elif act == free_action:
            total -= live.pop(e["addr"], 0)
    return peak, peak_idx


def live_at(trace: list[dict], upto: int, free_action: str) -> dict[int, tuple[int, dict]]:
    """Allocations live immediately after trace[upto]: addr -> (alloc index, entry)."""
    live: dict[int, tuple[int, dict]] = {}
    for i, e in enumerate(trace[: upto + 1]):
        act = e.get("action")
        if act == "alloc":
            live[e["addr"]] = (i, e)
        elif act == free_action:
            live.pop(e["addr"], None)
    return live


def fx_site(frames: list[dict]) -> tuple[str, int, str]:
    """Innermost FX-generated frame: (file, line, source text). Empty if none."""
    for f in frames or []:
        fn = f.get("filename", "")
        if FX_FILE.search(fn):
            line = int(f.get("line", 0))
            src = linecache.getline(fn, line).strip()
            return fn, line, src
    return "", 0, ""


def op_of(src: str) -> str:
    m = OP_CALL.search(src)
    return m.group(1) if m else "?"


def lhs_of(src: str) -> str:
    m = LHS.match(src)
    return m.group(1) if m else "?"


def analyze(trace: list[dict], boundary: int, measured: int) -> dict[str, Any]:
    """Find the peak, check it against the allocator, and attribute what is live."""
    candidates = {}
    for act in ("free_requested", "free_completed"):
        peak, idx = replay_peak(trace, act)
        candidates[act] = {"peak": peak, "idx": idx, "err": abs(peak - measured)}
    chosen = min(candidates, key=lambda a: candidates[a]["err"])
    peak, idx = candidates[chosen]["peak"], candidates[chosen]["idx"]

    live = live_at(trace, idx, chosen)

    by_phase: dict[str, list[int]] = {"forward": [0, 0], "backward": [0, 0]}
    groups: dict[tuple, list[int]] = collections.defaultdict(lambda: [0, 0])
    items = []
    live_nodes: list[dict] = []
    for alloc_idx, e in live.values():
        phase = "forward" if alloc_idx < boundary else "backward"
        fn, line, src = fx_site(e.get("frames"))
        by_phase[phase][0] += e["size"]
        by_phase[phase][1] += 1
        key = (phase, fn, op_of(src))
        groups[key][0] += e["size"]
        groups[key][1] += 1
        items.append((e["size"], phase, alloc_idx, fn, line, src))
        live_nodes.append({
            "phase": phase, "bytes": e["size"], "alloc_index": alloc_idx,
            "node": lhs_of(src), "op": op_of(src),
        })

    items.sort(reverse=True)
    top_groups = sorted(groups.items(), key=lambda kv: -kv[1][0])
    live_nodes.sort(key=lambda n: n["alloc_index"])

    span = max(len(trace) - boundary, 1)
    bw_pos = sorted((n["alloc_index"] - boundary) / span
                    for n in live_nodes if n["phase"] == "backward")
    timing = {
        "peak_position_in_backward": (idx - boundary) / span,
        "live_backward_alloc_position": (
            {"min": bw_pos[0], "median": bw_pos[len(bw_pos) // 2], "max": bw_pos[-1]}
            if bw_pos else None
        ),
    }
    names_by_op: dict[str, list[str]] = collections.defaultdict(list)
    for n in live_nodes:
        if n["phase"] == "backward":
            names_by_op[n["op"]].append(n["node"])

    return {
        "measured_delta": measured,
        "replay": candidates,
        "free_action_used": chosen,
        "peak_bytes": peak,
        "peak_matches_measured": candidates[chosen]["err"] == 0,
        "peak_index": idx,
        "trace_len": len(trace),
        "boundary_index": boundary,
        "peak_in_backward": idx >= boundary,
        "live_count": len(live),
        "by_phase": {k: {"bytes": v[0], "count": v[1]} for k, v in by_phase.items()},
        "top_groups": [
            {"phase": k[0], "graph_file": k[1], "op": k[2], "bytes": v[0], "count": v[1]}
            for k, v in top_groups[:20]
        ],
        "timing": timing,
        "backward_names_by_op": {k: sorted(set(v)) for k, v in names_by_op.items()},
        "live_nodes": live_nodes,
        "top_allocations": [
            {"bytes": s, "phase": p, "alloc_index": ai, "graph_file": f, "line": ln, "source": src}
            for s, p, ai, f, ln, src in items[:25]
        ],
    }


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def _start_recording() -> None:
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="python", max_entries=MAX_ENTRIES
        )
    except TypeError:  # older signature
        torch.cuda.memory._record_memory_history(
            True, trace_alloc_max_entries=MAX_ENTRIES, trace_alloc_record_context=True
        )


def _stop_recording() -> None:
    try:
        torch.cuda.memory._record_memory_history(enabled=None)
    except TypeError:
        torch.cuda.memory._record_memory_history(False)


def measure(
    model_name: str, scale: int, budget: float, backend: str, reorder: bool = True
) -> tuple[dict, dict]:
    # min_cut_rematerialization_partition looks this up by module-level name at
    # call time, so replacing the attribute disables the pass for this compile.
    import torch._functorch.partitioners as partitioners

    if not hasattr(partitioners, "_ackaudit_original_reorder"):
        partitioners._ackaudit_original_reorder = partitioners.reordering_to_mimic_autograd_engine
    partitioners.reordering_to_mimic_autograd_engine = (
        partitioners._ackaudit_original_reorder if reorder else (lambda gm: gm)
    )
    functorch_config.activation_memory_budget = budget
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    build, make_inputs = _resolve(model_name, scale=scale)
    model = build().cuda()
    args = tuple(a.cuda() for a in make_inputs())
    compiled = torch.compile(model, backend=backend, dynamic=False)

    # warmup: compiles forward, and backward lazily; fixes the plan
    compiled(*args).backward()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=False)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated()
    req_resident = torch.cuda.memory_stats().get("requested_bytes.all.current")

    _start_recording()
    torch.cuda.reset_peak_memory_stats()
    loss = compiled(*args)
    torch.cuda.synchronize()
    boundary = len(torch.cuda.memory._snapshot()["device_traces"][0])
    loss.backward()
    torch.cuda.synchronize()
    measured = torch.cuda.max_memory_allocated() - resident
    req_peak = torch.cuda.memory_stats().get("requested_bytes.all.peak")
    snap = torch.cuda.memory._snapshot()
    _stop_recording()

    trace = snap["device_traces"][0]
    result = analyze(trace, boundary, measured)
    result.update(
        budget=budget,
        model=model_name,
        scale=scale,
        backend=backend,
        resident_before=resident,
        torch_version=torch.__version__,
        trace_overflowed=len(trace) >= MAX_ENTRIES,
        reorder_pass=reorder,
        measured_requested=(
            req_peak - req_resident if req_peak is not None and req_resident is not None else None
        ),
    )
    if result["measured_requested"] is not None:
        result["replay_vs_requested_err"] = abs(result["peak_bytes"] - result["measured_requested"])

    del model, args, compiled, loss
    torch.cuda.empty_cache()
    return result, snap


def dump_graph_sources(outdir: Path) -> list[str]:
    """Write every FX-generated source still in linecache, so sites can be read in context."""
    written = []
    for i, fn in enumerate(sorted(k for k in linecache.cache if FX_FILE.search(str(k)))):
        lines = linecache.getlines(fn)  # resolves lazy entries, which are 1-tuples
        if not lines:
            continue
        path = outdir / f"graph_{i:03d}.py"
        path.write_text(f"# {fn}\n" + "".join(lines))
        written.append(f"{path.name}: {fn}")
    return written


def report(r: dict) -> str:
    mb = lambda b: b / 1e6
    out = [
        f"budget {r['budget']:.2f}  {r['model']} scale {r['scale']}  {r['backend']}  torch {r['torch_version']}",
        f"  measured delta          {mb(r['measured_delta']):9.1f} MB",
        f"  replayed peak           {mb(r['peak_bytes']):9.1f} MB   ({r['free_action_used']})",
        f"  reorder pass            {'on' if r.get('reorder_pass', True) else 'OFF (ablation)'}",
    ]
    for act, v in r["replay"].items():
        out.append(f"    via {act:<16} {mb(v['peak']):9.1f} MB   off by {v['err']} bytes")
    if r.get("measured_requested") is not None:
        out.append(
            f"  requested-bytes delta   {mb(r['measured_requested']):9.1f} MB   "
            f"replay off by {r['replay_vs_requested_err']} bytes   <- the exact check"
        )
        out.append(
            "  (max_memory_allocated counts rounded blocks, the trace records requested sizes,"
        )
        out.append("   so the replay is compared against requested bytes)")
    if r["trace_overflowed"]:
        out.append("  WARNING: trace hit MAX_ENTRIES; early events are missing and the phase split is wrong")
    out.append(
        f"  peak at trace index {r['peak_index']} of {r['trace_len']}; "
        f"backward starts at {r['boundary_index']}; peak in backward: {r['peak_in_backward']}"
    )
    t = r["timing"]
    out.append(f"  peak is {100 * t['peak_position_in_backward']:.1f}% of the way through the backward")
    lp = t["live_backward_alloc_position"]
    if lp:
        out.append(
            f"  live backward tensors were allocated between {100 * lp['min']:.1f}% and "
            f"{100 * lp['max']:.1f}% of the backward (median {100 * lp['median']:.1f}%)"
        )
    out.append(f"  live at peak: {r['live_count']} allocations")
    for ph, v in r["by_phase"].items():
        out.append(f"    allocated in {ph:<9} {mb(v['bytes']):9.1f} MB  ({v['count']} allocations)")
    out.append("  largest groups (phase, op):")
    for g in r["top_groups"][:12]:
        out.append(f"    {g['phase']:<9} {g['op']:<32} {mb(g['bytes']):9.1f} MB  x{g['count']}")
    out.append("  distinct live backward nodes, by op:")
    for op, names in sorted(r["backward_names_by_op"].items(), key=lambda kv: -len(kv[1]))[:6]:
        shown = ", ".join(names[:8]) + (f", ... ({len(names)} total)" if len(names) > 8 else "")
        out.append(f"    {op:<20} {len(names):>3}  {shown}")
    out.append("  largest single allocations:")
    for a in r["top_allocations"][:10]:
        out.append(f"    {mb(a['bytes']):8.1f} MB  {a['phase']:<9} {a['source'][:90]}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama")
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--budgets", type=float, nargs="*", default=[0.05, 0.10, 0.15])
    ap.add_argument("--backend", default="aot_eager")
    ap.add_argument(
        "--no-reorder", action="store_true",
        help="ablation: replace reordering_to_mimic_autograd_engine with the identity",
    )
    ap.add_argument("--outdir", default="results/memory_snapshot")
    ap.add_argument(
        "--pickle-dir", default="/tmp/ackaudit_snapshots",
        help="full snapshots for pytorch.org/memory_viz; large, kept out of the repo",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdir = Path(args.pickle_dir)
    pdir.mkdir(parents=True, exist_ok=True)

    results = []
    for b in args.budgets:
        r, snap = measure(args.model, args.scale, b, args.backend, reorder=not args.no_reorder)
        results.append(r)
        stem = f"{args.model}_scale{args.scale}_{args.backend}" + ("_noreorder" if args.no_reorder else "") + f"_b{b:.2f}"
        with open(pdir / f"{stem}.pickle", "wb") as fh:
            pickle.dump(snap, fh)
        print(report(r))
        print()

    stem = f"{args.model}_scale{args.scale}_{args.backend}" + ("_noreorder" if args.no_reorder else "")
    (outdir / f"{stem}.json").write_text(json.dumps(results, indent=2))
    (outdir / f"{stem}_report.txt").write_text("\n\n".join(report(r) for r in results) + "\n")
    srcdir = outdir / f"{stem}_graphs"
    srcdir.mkdir(exist_ok=True)
    listed = dump_graph_sources(srcdir)
    print(f"graph sources written: {len(listed)}  ({srcdir})")
    print(f"full snapshots: {pdir}  (open at https://pytorch.org/memory_viz)")


if __name__ == "__main__":
    main()
