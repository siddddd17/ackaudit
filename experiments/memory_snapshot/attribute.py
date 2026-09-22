"""
Reads the snapshot JSON and the dumped FX graphs under results/memory_snapshot/
and the harness sweep under results/measured_backward/. Needs no GPU and does
not import torch: graphs are read with Python's ast module.

    python -m experiments.memory_snapshot.attribute

Definitions used throughout:

- "recomputed": allocated during the backward by a graph statement with no
  dataflow from the tangent input. The backward graph's only inputs are saved
  forward values and the tangent, so such a statement computes a
  forward-derived value.
- "MLP inner tensor": an allocation of batch x seq x intermediate x 4 bytes. In
  this model only the MLP's gate projection, up projection, SiLU and SiLU * up
  have that shape.
- "MLP output": a saved mm whose input, through view-like ops, is a mul with a
  SiLU operand, i.e. down_proj(silu(gate) * up).
"""

from __future__ import annotations

import ast
import collections
import json
import statistics
from pathlib import Path

RES = Path("results/memory_snapshot")
HARNESS = Path("results/measured_backward/fine/llama_scale8.json")
BATCH, SEQ, INTERMEDIATE, LAYERS = 2, 1024, 688, 32
INNER = BATCH * SEQ * INTERMEDIATE * 4
VIEW_OPS = {"view", "_unsafe_view", "reshape", "expand", "clone", "t", "transpose", "permute"}
SILU = ["silu"] + [f"silu_{k}" for k in range(1, LAYERS)]


# ---------------------------------------------------------------------------
# reading FX-generated source
# ---------------------------------------------------------------------------

def _op(call: ast.AST) -> str | None:
    """'mm' for torch.ops.aten.mm.default(...), else None."""
    if not isinstance(call, ast.Call):
        return None
    parts, f = [], call.func
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    parts.reverse()
    return parts[3] if parts[:3] == ["torch", "ops", "aten"] and len(parts) >= 4 else None


class Graph:
    def __init__(self, path: Path) -> None:
        src = path.read_text()
        self.header = src.splitlines()[0].lstrip("# ").strip()
        self.nlines = len(src.splitlines()) - 1
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef))
        self.defs: dict[str, ast.Assign] = {}
        self.order: list[str] = []
        self.uses: dict[str, list[int]] = collections.defaultdict(list)
        self.returned: list[str] = []
        for st in fn.body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                if not (isinstance(st.value, ast.Constant) and st.value.value is None):
                    self.defs[st.targets[0].id] = st
                    self.order.append(st.targets[0].id)
            elif isinstance(st, ast.Return):
                self.returned = [x.id for x in ast.walk(st.value) if isinstance(x, ast.Name)]
            for node in ast.walk(st):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    self.uses[node.id].append(st.lineno)
        self.text = src

    def op(self, name: str) -> str | None:
        st = self.defs.get(name)
        return _op(st.value) if st else None

    def args(self, name: str) -> list[str]:
        st = self.defs.get(name)
        if not st or not isinstance(st.value, ast.Call):
            return []
        return [a.id for a in st.value.args if isinstance(a, ast.Name)]

    def pos(self, name: str) -> tuple[float, float]:
        """(created, last used) as % of the graph's statements, by line."""
        return (100 * (self.defs[name].lineno - 2) / self.nlines,
                100 * (max(self.uses[name]) - 2) / self.nlines)

    def tangent_derived(self) -> set[str]:
        # boxed calling convention: tangents arrive as `tangents_1 = next(args_iter)`
        t = {n for n in self.order if n.startswith("tangents_")}
        for n in self.order:
            names = {x.id for x in ast.walk(self.defs[n].value) if isinstance(x, ast.Name)}
            if names & t:
                t.add(n)
        return t

    def strip_views(self, n: str | None) -> str | None:
        while n and self.op(n) in VIEW_OPS:
            a = self.args(n)
            n = a[0] if a else None
        return n

    def is_forward(self) -> bool:
        return "primals_" in self.text and "tangents_" not in self.text


def layer_of(silu_name: str) -> int:
    return 0 if silu_name == "silu" else int(silu_name.split("_")[1])


def key(g: Graph) -> int:
    return int(g.header.rsplit(".", 1)[1])


# ---------------------------------------------------------------------------
# per-run analysis
# ---------------------------------------------------------------------------

def load(tag: str) -> tuple[dict[float, dict], list[Graph]]:
    runs = {round(r["budget"], 2): r for r in json.loads((RES / f"llama_scale8_aot_eager{tag}.json").read_text())}
    graphs = [Graph(p) for p in sorted((RES / f"llama_scale8_aot_eager{tag}_graphs").glob("graph_*.py"))]
    return runs, graphs


def graphs_for(run: dict, graphs: list[Graph]) -> tuple[Graph, Graph]:
    """The backward graph that allocated this run's backward tensors, and the
    forward graph compiled immediately before it."""
    bw_name = collections.Counter(
        a["graph_file"] for a in run["top_allocations"] if a["phase"] == "backward" and a["graph_file"]
    ).most_common(1)[0][0]
    bw = next(g for g in graphs if g.header == bw_name)
    fw = max((g for g in graphs if g.is_forward() and key(g) < key(bw)), key=key)
    return fw, bw


def saved(fw: Graph) -> dict:
    out = collections.Counter()
    mlp_layers: set[int] = set()
    for n in (x for x in fw.returned if x in fw.defs):
        st = fw.defs[n]
        if isinstance(st.value, ast.Subscript) and isinstance(st.value.value, ast.Name) \
                and st.value.value.id.startswith("_scaled_dot_product"):
            out["attention"] += 1
        elif fw.op(n) == "mm":
            m = fw.strip_views(fw.args(n)[0] if fw.args(n) else None)
            silus = [a for a in fw.args(m) if fw.op(a) == "silu"] if m and fw.op(m) == "mul" else []
            if silus:
                mlp_layers.add(layer_of(silus[0]))
            else:
                out["other_mm"] += 1
        elif fw.op(n) == "add" and all(fw.op(a) == "mul" for a in fw.args(n)):
            out["rotary_add"] += 1
    return {**out, "mlp_layers": mlp_layers}


def backward_live(run: dict, bw: Graph) -> dict:
    t = bw.tangent_derived()
    live = [n for n in run["live_nodes"] if n["phase"] == "backward"]
    rec = [n for n in live if n["node"] in bw.defs and n["node"] not in t]
    inner_ops = collections.Counter(n["op"].split(".")[-1] for n in rec if n["bytes"] == INNER)
    return {
        "total": sum(n["bytes"] for n in live),
        "recomputed": sum(n["bytes"] for n in rec),
        "mlp_inner": sum(n["bytes"] for n in rec if n["bytes"] == INNER),
        "mlp_inner_ops": dict(sorted(inner_ops.items())),
        "silu_layers": sorted(layer_of(n["node"]) for n in live if n["op"] == "aten.silu"),
    }


def released_by_gradient(bw: Graph) -> int:
    """How many recomputed SiLUs are last used by a mul whose other operand
    depends on the tangent, i.e. by their layer's MLP gradient."""
    t, line_of = bw.tangent_derived(), {bw.defs[n].lineno: n for n in bw.order}
    count = 0
    for n in SILU:
        rel = line_of.get(max(bw.uses[n]))
        other = [a for a in bw.args(rel) if a != n] if rel else []
        count += bool(rel) and bw.op(rel) == "mul" and bool(other) and all(a in t for a in other)
    return count


def spans(bw: Graph) -> list[tuple[float, float]]:
    return [bw.pos(n) for n in SILU]


# ---------------------------------------------------------------------------

def mb(b: float) -> str:
    return f"{b / 1e6:.1f} MB"


def main() -> None:
    harness = {round(c["budget"], 2): c for c in json.loads(HARNESS.read_text())}
    nat, nat_g = load("")
    off, off_g = load("_noreorder")
    ctl, _ = load("_int-none")
    fin, fin_g = load("_int-save-mlp-out")
    fout, fout_g = load("_int-recompute-mlp-out")

    print(f"MLP inner tensor size: {INNER} bytes = {INNER / 1e6:.1f} MB\n")
    print("== measurement ==")
    for b, r in sorted(nat.items()):
        h = statistics.median(harness[b]["measured_peaks"]) - harness[b]["resident_before"]
        print(f"  {b:.2f}  peak {mb(r['measured_delta'])}; identical to harness median: "
              f"{r['measured_delta'] == h}; replay == requested_bytes: {r['replay_vs_requested_err'] == 0}")
    for tag, d in [("pass off", off), ("hook, unedited", ctl), ("MLP outputs forced in", fin),
                   ("MLP outputs forced out", fout)]:
        print(f"  {tag:<22} replay == requested_bytes: {all(r['replay_vs_requested_err'] == 0 for r in d.values())}")
    for b, r in sorted(ctl.items()):
        print(f"  hook unedited at {b:.2f} identical to unhooked: {r['measured_delta'] == nat[b]['measured_delta']}")

    print("\n== what is live at the peak, and what the forward saved ==")
    table = {}
    for b, r in sorted(nat.items()):
        fw, bw = graphs_for(r, nat_g)
        s, live = saved(fw), backward_live(r, bw)
        unsaved = sorted(set(range(LAYERS)) - s["mlp_layers"])
        table[b] = (fw, bw, unsaved)
        print(f"  {b:.2f}  peak {mb(r['measured_delta'])}; allocated in backward {mb(live['total'])}, "
              f"recomputed {mb(live['recomputed'])}, MLP inner {mb(live['mlp_inner'])}")
        print(f"        saved: {s.get('attention', 0)} attention outputs, {len(s['mlp_layers'])} MLP outputs, "
              f"{s.get('other_mm', 0)} other mm, {s.get('rotary_add', 0)} rotary add")
        print(f"        MLP inner tensors by op: {live['mlp_inner_ops']}")
        print(f"        MLP output not saved: {unsaved or 'none'}; SiLU live at peak: {live['silu_layers']}")
        print(f"        recomputed SiLUs last used by a tangent-dependent mul: {released_by_gradient(bw)} of {LAYERS}")
        print(f"        forward graph {fw.header}, backward graph {bw.header}")

    print("\n== held span of each recomputed SiLU, grouped by whether its MLP output was saved ==")
    cells = collections.defaultdict(list)
    for b, (fw, bw, unsaved) in table.items():
        for k, (c, u) in enumerate(spans(bw)):
            if k == LAYERS - 1:
                continue  # processed first: recomputed just before its own gradient either way
            group = "layers 0-20" if k <= 20 else "layers 21-30"
            cells[(group, k in unsaved)].append((u - c, b))
    for (group, unsv), v in sorted(cells.items()):
        spans_ = [x for x, _ in v]
        print(f"  {group:<13} {'not saved' if unsv else 'saved':<9} {min(spans_):5.1f}% to {max(spans_):5.1f}%  "
              f"budgets {sorted({b for _, b in v})}")
    bw05 = table[0.05][1]
    c05 = spans(bw05)
    print(f"  0.05: every SiLU created before any released: {max(c for c, _ in c05) < min(u for _, u in c05)}; "
          f"all created within {max(c for c, _ in c05):.1f}% of the graph; layer 0 held {c05[0][1] - c05[0][0]:.1f}%")

    print("\n== intervention ==")
    rows = []
    for label, d, g, b in [("natural 0.05", nat, nat_g, 0.05), ("0.15, MLP outputs forced out", fout, fout_g, 0.15),
                           ("natural 0.10", nat, nat_g, 0.10), ("0.05, MLP outputs forced in", fin, fin_g, 0.05),
                           ("natural 0.15", nat, nat_g, 0.15)]:
        r = d[b]
        fw, bw = graphs_for(r, g)
        weight = r["solver"]["final_saved_weight"] if r.get("solver") else harness[b]["proxy_peak"]
        live = backward_live(r, bw)
        c = spans(bw)
        early = sum(ci < min(u for _, u in c) for ci, _ in c)
        rows.append((weight, label))
        print(f"  {label:<30} saved weight {weight:.4f}; MLP outputs saved {len(saved(fw)['mlp_layers']):>2} "
              f"(from the compiled forward graph); peak {mb(r['measured_delta'])}; MLP inner live "
              f"{mb(live['mlp_inner'])}; SiLU created before the first release: {early}")
    s_in, s_out = fin[0.05]["solver"], fout[0.15]["solver"]
    print(f"  forced out stays within budget: {s_out['final_saved_weight'] <= s_out['max_memory']} "
          f"({s_out['final_saved_weight']:.4f} <= {s_out['max_memory']:.4f})")
    print(f"  forced in exceeds budget: {s_in['final_saved_weight'] > s_in['max_memory']}; "
          f"the 32 MLP outputs weigh {s_in['final_saved_weight'] - s_in['solver_saved_weight']:.4f}")
    print(f"  forced out raises the 0.15 peak {fout[0.15]['measured_delta'] / nat[0.15]['measured_delta']:.2f}x; "
          f"forced in lowers the 0.05 peak by "
          f"{100 * (1 - fin[0.05]['measured_delta'] / nat[0.05]['measured_delta']):.1f}% "
          f"({mb(nat[0.05]['measured_delta'] - fin[0.05]['measured_delta'])})")

    print("\n== reordering pass disabled ==")
    for b, r in sorted(off.items()):
        cut = 100 * (1 - nat[b]["measured_delta"] / r["measured_delta"])
        print(f"  {b:.2f}  pass on {mb(nat[b]['measured_delta'])}, pass off {mb(r['measured_delta'])}; "
              f"the pass lowers the peak by {cut:.1f}%")
    _, bw_off = graphs_for(off[0.05], off_g)
    diff = max(abs((u1 - c1) - (u2 - c2)) for (c1, u1), (c2, u2) in zip(spans(bw05), spans(bw_off)))
    print(f"  0.05: largest change in any SiLU's held span with the pass off: {diff:.2f} percentage points")

    print("\n== within the budget: forcing k MLP outputs, dp_knapsack fills the rest ==")
    sweep = sorted(json.loads((RES / "mlp_saved_sweep_b0.05.json").read_text()), key=lambda r: r["k"])
    # In the sweep files peak_bytes is already the median of max_memory_allocated()
    # minus the resident baseline; per-repeat values are in peaks_mb.
    base_peak = next(r for r in sweep if r["k"] == 0)["peak_bytes"]
    for r in sweep:
        sv = r["solver"]
        peak, peaks = r["peak_bytes"], r["peaks_mb"]
        print(f"  k={r['k']:<3} saved weight {sv['saved_weight']:.4f} (<= {sv['max_memory']:.2f}: "
              f"{sv['saved_weight'] <= sv['max_memory'] + 5e-7}); peak {mb(peak)}, "
              f"{100 * (1 - peak / base_peak):.1f}% below k=0; estimated runtime saved "
              f"{100 * sv['runtime_saved'] / sv['natural_runtime_saved']:.1f}% of the natural plan; "
              f"repeats identical: {len(set(peaks)) == 1}")
    print(f"  natural plan (k=0) matches the unhooked 0.05 run: "
          f"{base_peak == nat[0.05]['measured_delta']}")

    print("\n== which layers, at a fixed count: k = 8 ==")
    place = json.loads((RES / "mlp_placement_sweep_b0.05_k8.json").read_text())
    pk = {}
    for r in place:
        pk[r["placement"]] = r["peak_bytes"]
        print(f"  {r['placement']:<7} layers {r['forced_mlp_layers']}; saved weight "
              f"{r['solver']['saved_weight']:.6f}; estimated runtime saved {r['solver']['runtime_saved']}; "
              f"peak {mb(pk[r['placement']])}; repeats identical: {len(set(r['peaks_mb'])) == 1}")
    print(f"  identical saved weight and runtime across placements: "
          f"{len({(r['solver']['saved_weight'], r['solver']['runtime_saved']) for r in place}) == 1}")
    print(f"  peak spread: {100 * (max(pk.values()) / min(pk.values()) - 1):.1f}%")

    print("\n== measured step time for the within-budget plans ==")
    sweep_peaks = {r["k"]: r["peak_bytes"] for r in sweep}
    for backend in ("aot_eager", "inductor"):
        path = RES / f"step_time_sweep_b0.05_{backend}.json"
        if not path.exists():
            print(f"  {backend}: {path.name} not committed")
            continue
        t = json.loads(path.read_text())
        rows = {r["k"]: r for r in t["summary"]}
        base = rows[0]
        print(f"  {backend} ({t['device']}, torch {t['torch_version']}, {t['rounds']} rounds x {t['iters']} steps):")
        for k, r in sorted(rows.items()):
            print(f"    k={k:<3} peak {mb(r['peak_bytes'])}, {100 * (1 - r['peak_bytes'] / base['peak_bytes']):.1f}% below k=0; "
                  f"saved weight {r['saved_weight']:.4f} within budget: {r['within_budget']}; "
                  f"step {r['step_ms_median']:.1f} ms ({r['step_time_vs_first_pct']:+.1f}%), "
                  f"per round {[round(x, 1) for x in r['step_ms_per_round']]}; "
                  f"peak identical across rounds: {r['peak_identical_across_rounds']}")
        k24 = rows.get(24)
        if k24:
            print(f"    k=24 removes {mb(base['peak_bytes'] - k24['peak_bytes'])} of peak")
        if backend == "aot_eager":
            print(f"    every peak identical to the committed saved-count sweep: "
                  f"{all(rows[k]['peak_bytes'] == sweep_peaks.get(k) for k in rows)}")
        else:
            h = json.loads(Path("results/measured_backward/inductor/llama_scale8_inductor.json").read_text())
            c = next(c for c in h if abs(c["budget"] - 0.05) < 1e-9)
            hp = statistics.median(c["measured_peaks"]) - c["resident_before"]
            print(f"    natural plan (k=0) identical to the inductor harness run in the issue: {base['peak_bytes'] == hp}")


if __name__ == "__main__":
    main()
