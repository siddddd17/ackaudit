"""When is each recomputed activation created, and when is it last used?

Reads the backward-graph source dumped by run_snapshot.py. FX-generated code
runs top to bottom and frees each value on the line of its last use (it appends
`; name = None`), so line position in the backward graph is execution order.

For each node matching --pattern (default: the recomputed `silu` activations,
one per llama layer), reports the line it is defined on and the line it is last
used on, both as a fraction of the backward graph. A value defined near the top
and last used near the bottom is held for almost the whole backward.

    python -m experiments.memory_snapshot.lifetimes --budget 0.05
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def backward_graph_file(results: list[dict], budget: float) -> str:
    """The FX file that allocated this budget's live backward tensors."""
    for r in results:
        if abs(r["budget"] - budget) < 1e-9:
            files = collections.Counter(
                a["graph_file"] for a in r["top_allocations"] if a["phase"] == "backward" and a["graph_file"]
            )
            if not files:
                raise SystemExit(f"no backward graph file recorded for budget {budget}")
            return files.most_common(1)[0][0]
    raise SystemExit(f"budget {budget} not in results")


def find_source(graph_dir: Path, fx_name: str) -> Path:
    for p in sorted(graph_dir.glob("graph_*.py")):
        with open(p) as fh:
            if fh.readline().strip() == f"# {fx_name}":
                return p
    raise SystemExit(f"{fx_name} not among dumped sources in {graph_dir}")


def lifetimes(lines: list[str], pattern: str) -> list[dict]:
    """For each node whose name fully matches `pattern`: definition and last-use line."""
    name_re = re.compile(rf"^\s*({pattern})\s*=")
    defs = {}
    for i, line in enumerate(lines):
        m = name_re.match(line)
        if m:
            defs[m.group(1)] = i
    out = []
    for name, d in defs.items():
        tok = re.compile(rf"\b{re.escape(name)}\b")
        last = max(i for i, line in enumerate(lines) if tok.search(line))
        out.append({"node": name, "defined": d, "last_used": last})
    body = len(lines)
    for o in out:
        o["defined_frac"] = o["defined"] / body
        o["last_used_frac"] = o["last_used"] / body
        o["held_frac"] = o["last_used_frac"] - o["defined_frac"]
    return sorted(out, key=lambda o: o["defined"])


def natural(name: str) -> int:
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--results", default="results/memory_snapshot/llama_scale8_aot_eager.json")
    ap.add_argument("--graphs", default="results/memory_snapshot/llama_scale8_aot_eager_graphs")
    ap.add_argument("--pattern", default=r"silu(?:_\d+)?")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    fx_name = backward_graph_file(results, args.budget)
    src = find_source(Path(args.graphs), fx_name)
    lines = src.read_text().splitlines()[1:]  # drop the header comment
    rows = lifetimes(lines, args.pattern)

    print(f"budget {args.budget:.2f}  backward graph {fx_name}  ({src.name}, {len(lines)} lines)")
    print(f"{'node':<10} {'defined':>8} {'last used':>10} {'defined %':>10} {'last used %':>12} {'held %':>8}")
    for r in sorted(rows, key=lambda r: natural(r["node"])):
        print(
            f"{r['node']:<10} {r['defined']:>8} {r['last_used']:>10} "
            f"{100 * r['defined_frac']:>9.1f}% {100 * r['last_used_frac']:>11.1f}% "
            f"{100 * r['held_frac']:>7.1f}%"
        )
    if rows:
        held = sorted(r["held_frac"] for r in rows)
        print(
            f"\n{len(rows)} nodes; held for a median {100 * held[len(held) // 2]:.1f}% "
            f"of the backward graph (min {100 * held[0]:.1f}%, max {100 * held[-1]:.1f}%)"
        )


if __name__ == "__main__":
    main()
