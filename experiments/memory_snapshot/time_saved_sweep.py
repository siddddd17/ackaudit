"""Measured step time for the within-budget plans of the saved-count sweep.

The saved-count sweep (sweep_mlp_saved.py) prices lower-peak plans in the
knapsack's *estimated* runtime saved. This measures what they cost in wall-clock
forward+backward time, for exactly the same plans: it reuses the sweep's
FeasibleMLPSweepSolver, seed and model construction.

Validity check: each plan's peak is re-measured and compared with the committed
sweep result for the same budget and k. A mismatch means a different plan was
timed, and the run stops.

Laptop GPUs drift with temperature, so plans are timed in several rounds with
the order reversed on alternate rounds, and every round is reported.

    python -m experiments.memory_snapshot.time_saved_sweep --ks 0 8 16 24
    python -m experiments.memory_snapshot.time_saved_sweep --ks 0 8 16 24 --backend inductor
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

from ackaudit.audit import RuntimeRecorder
from ackaudit.capture import _resolve
from experiments.memory_snapshot.sweep_mlp_saved import FeasibleMLPSweepSolver

RES = Path("results/memory_snapshot")


def committed_peaks(budget: float) -> dict[int, int]:
    """Peak bytes per k from the committed aot_eager saved-count sweep, if present."""
    path = RES / f"mlp_saved_sweep_b{budget:.2f}.json"
    if not path.exists():
        return {}
    return {r["k"]: r["peak_bytes"] for r in json.loads(path.read_text()) if r.get("backend") == "aot_eager"}


def time_plan(
    *, model_name: str, scale: int, budget: float, backend: str, k: int, warmup: int, iters: int
) -> dict[str, Any]:
    torch.manual_seed(197838)  # same weights and inputs as the sweep
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
        functorch_config.activation_memory_budget_solver = FeasibleMLPSweepSolver(recorder, k=k, record=record)
        with recorder:
            compiled(*args).backward()  # compiles and fixes the plan
    finally:
        functorch_config.activation_memory_budget_solver = prev_solver
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=False)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated()

    # one measured step for the peak, exactly as the sweep measures it
    torch.cuda.reset_peak_memory_stats()
    loss = compiled(*args)
    torch.cuda.synchronize()
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - resident
    model.zero_grad(set_to_none=False)

    for _ in range(warmup):
        compiled(*args).backward()
        model.zero_grad(set_to_none=False)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        compiled(*args).backward()
        torch.cuda.synchronize()
        times_ms.append(1000 * (time.perf_counter() - t0))
        model.zero_grad(set_to_none=False)

    del model, args, compiled, loss
    torch.cuda.empty_cache()
    return {
        "k": k,
        "peak_bytes": peak,
        "times_ms": times_ms,
        "median_ms": statistics.median(times_ms),
        "saved_weight": record.get("saved_weight"),
        "max_memory": record.get("max_memory"),
        "runtime_saved": record.get("runtime_saved"),
        "natural_runtime_saved": record.get("natural_runtime_saved"),
        "forced_mlp_layers": record.get("forced_mlp_layers"),
    }


def quartiles(xs: list[float]) -> tuple[float, float]:
    q = statistics.quantiles(xs, n=4)
    return q[0], q[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama")
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 8, 16, 24])
    ap.add_argument("--backend", default="aot_eager")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")

    expected = committed_peaks(args.budget) if args.backend == "aot_eager" else {}
    rounds: list[list[dict[str, Any]]] = []
    for rnd in range(args.rounds):
        order = args.ks if rnd % 2 == 0 else list(reversed(args.ks))
        results = []
        for k in order:
            r = time_plan(model_name=args.model, scale=args.scale, budget=args.budget, backend=args.backend,
                          k=k, warmup=args.warmup, iters=args.iters)
            if k in expected and r["peak_bytes"] != expected[k]:
                raise SystemExit(
                    f"k={k}: peak {r['peak_bytes']} != committed sweep {expected[k]}; a different plan was timed"
                )
            results.append(r)
            print(f"round {rnd + 1}  k={k:<3} peak {r['peak_bytes'] / 1e6:8.1f} MB  "
                  f"median step {r['median_ms']:8.1f} ms"
                  f"{'  (peak matches committed sweep)' if k in expected else ''}", flush=True)
        rounds.append(results)

    print()
    by_k = {k: [r for rr in rounds for r in rr if r["k"] == k] for k in args.ks}
    base = statistics.median(t for r in by_k[args.ks[0]] for t in r["times_ms"])
    summary = []
    for k in args.ks:
        rs = by_k[k]
        all_t = [t for r in rs for t in r["times_ms"]]
        med = statistics.median(all_t)
        q1, q3 = quartiles(all_t)
        r0 = rs[0]
        est = 100 * r0["runtime_saved"] / r0["natural_runtime_saved"] if r0["natural_runtime_saved"] else None
        row = {
            "k": k,
            "forced_mlp_layers": r0["forced_mlp_layers"],
            "saved_weight": r0["saved_weight"],
            "within_budget": r0["saved_weight"] is not None and r0["saved_weight"] <= args.budget + 5e-7,
            "peak_bytes": r0["peak_bytes"],
            "peak_identical_across_rounds": len({r["peak_bytes"] for r in rs}) == 1,
            "estimated_runtime_saved_pct_of_natural": est,
            "step_ms_median": med,
            "step_ms_q1": q1,
            "step_ms_q3": q3,
            "step_ms_per_round": [r["median_ms"] for r in rs],
            "step_time_vs_first_pct": 100 * (med / base - 1),
        }
        summary.append(row)
        print(f"k={k:<3} peak {row['peak_bytes'] / 1e6:7.1f} MB  saved weight {row['saved_weight']:.4f}  "
              f"est. runtime saved {est:5.1f}%  step {med:7.1f} ms (IQR {q1:.1f}-{q3:.1f}), "
              f"{row['step_time_vs_first_pct']:+.1f}% vs k={args.ks[0]}  per round "
              f"{[round(x, 1) for x in row['step_ms_per_round']]}")

    out = {
        "model": args.model, "scale": args.scale, "budget": args.budget, "backend": args.backend,
        "torch_version": torch.__version__, "device": torch.cuda.get_device_name(),
        "rounds": args.rounds, "warmup": args.warmup, "iters": args.iters,
        "summary": summary, "raw": rounds,
    }
    path = RES / f"step_time_sweep_b{args.budget:.2f}_{args.backend}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
