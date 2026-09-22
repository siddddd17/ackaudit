"""Which MLP outputs each within-budget plan actually saves.

time_saved_sweep.py records the layers it *forces* but not the final saved set.
The solver refills the remaining budget from every non-forced item, including
the unforced MLP outputs, so a plan could in principle save more than k. This
compiles each plan once, records the solver's final saved MLP layers, and checks
that it is the plan that was timed: its peak must equal the committed timing
result byte for byte, or the run stops.

Writes a new file and leaves the committed timing results untouched.

    python -m experiments.memory_snapshot.check_saved_mlp --backend inductor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch._functorch import config as functorch_config

from ackaudit.audit import RuntimeRecorder
from ackaudit.capture import _resolve
from experiments.memory_snapshot.sweep_mlp_saved import FeasibleMLPSweepSolver
from experiments.memory_snapshot.time_saved_sweep import GraphOrderSolver

RES = Path("results/memory_snapshot")


def compile_plan(*, model_name: str, scale: int, budget: float, backend: str, k: int) -> dict[str, Any]:
    # identical setup to time_saved_sweep.time_plan, up to and including the peak measurement
    torch.manual_seed(197838)
    torch.cuda.manual_seed_all(197838)
    functorch_config.activation_memory_budget = budget
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    build, make_inputs = _resolve(model_name, scale=scale)
    model = build().cuda()
    args = tuple(a.cuda() for a in make_inputs())
    compiled = torch.compile(model, backend=backend, dynamic=False)

    record: dict[str, Any] = {}
    recorder = RuntimeRecorder()
    prev_solver = functorch_config.activation_memory_budget_solver
    try:
        solver_cls = FeasibleMLPSweepSolver if backend == "aot_eager" else GraphOrderSolver
        functorch_config.activation_memory_budget_solver = solver_cls(recorder, k=k, record=record)
        with recorder:
            compiled(*args).backward()
    finally:
        functorch_config.activation_memory_budget_solver = prev_solver
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=False)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    loss = compiled(*args)
    torch.cuda.synchronize()
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - resident

    del model, args, compiled, loss
    torch.cuda.empty_cache()
    return {
        "k": k,
        "peak_bytes": peak,
        "forced_mlp_layers": record.get("forced_mlp_layers"),
        "saved_mlp_layers": record.get("saved_mlp_layers"),
        "saved_weight": record.get("saved_weight"),
        "max_memory": record.get("max_memory"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama")
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 8, 16, 24])
    ap.add_argument("--backend", default="inductor")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")

    timed_path = RES / f"step_time_sweep_b{args.budget:.2f}_{args.backend}.json"
    if not timed_path.exists():
        raise SystemExit(f"{timed_path} not found; this check compares against the committed timing run")
    timed = {r["k"]: r["peak_bytes"] for r in json.loads(timed_path.read_text())["summary"]}

    rows = []
    for k in args.ks:
        r = compile_plan(model_name=args.model, scale=args.scale, budget=args.budget, backend=args.backend, k=k)
        if k not in timed:
            raise SystemExit(f"k={k} was not timed in {timed_path.name}")
        if r["peak_bytes"] != timed[k]:
            raise SystemExit(
                f"k={k}: peak {r['peak_bytes']} != timed {timed[k]}; not the plan that was timed, stopping"
            )
        forced, saved = set(r["forced_mlp_layers"] or []), set(r["saved_mlp_layers"] or [])
        r["extra_saved_by_solver"] = sorted(saved - forced)
        r["forced_all_saved"] = forced <= saved
        r["within_budget"] = r["saved_weight"] <= r["max_memory"] + 5e-7
        rows.append(r)
        print(f"k={k:<3} peak {r['peak_bytes'] / 1e6:7.1f} MB (matches timed run)  forced {len(forced):>2}  "
              f"saved {len(saved):>2}  extra chosen by the solver: {r['extra_saved_by_solver'] or 'none'}  "
              f"within budget: {r['within_budget']}", flush=True)

    out = {
        "model": args.model, "scale": args.scale, "budget": args.budget, "backend": args.backend,
        "torch_version": torch.__version__, "device": torch.cuda.get_device_name(),
        "compared_against": timed_path.name, "plans": rows,
    }
    path = RES / f"saved_mlp_check_b{args.budget:.2f}_{args.backend}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
