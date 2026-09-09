"""Compile models under the auditing solver to collect knapsack instances."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import torch
import torch._functorch.config as functorch_config
import torch.nn as nn

from .audit import AuditingSolver

log = logging.getLogger(__name__)


# Varied graph shapes: chain, branching, transformer, conv.
# TODO: swap for real HF checkpoints once the toy numbers are understood.


class DeepMLP(nn.Module):
    def __init__(self, width: int = 256, depth: int = 24) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(width, width), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).sum()


class BranchingNet(nn.Module):
    def __init__(self, width: int = 192, branches: int = 6, depth: int = 5) -> None:
        super().__init__()
        self.stem = nn.Linear(width, width)
        self.branches = nn.ModuleList(
            nn.Sequential(*[nn.Sequential(nn.Linear(width, width), nn.Tanh())
                            for _ in range(depth)])
            for _ in range(branches)
        )
        self.head = nn.Linear(width, width)

    def forward(self, x):
        h = self.stem(x)
        outs = [b(h) for b in self.branches]
        return self.head(sum(outs)).sum()


class TransformerBlock(nn.Module):
    def __init__(self, d: int = 256, heads: int = 8, blocks: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "attn": nn.MultiheadAttention(d, heads, batch_first=True),
                    "n1": nn.LayerNorm(d),
                    "n2": nn.LayerNorm(d),
                    "ff1": nn.Linear(d, 4 * d),
                    "ff2": nn.Linear(4 * d, d),
                }
            )
            for _ in range(blocks)
        )

    def forward(self, x):
        for b in self.blocks:
            h = b["n1"](x)
            a, _ = b["attn"](h, h, h, need_weights=False)
            x = x + a
            h = b["n2"](x)
            x = x + b["ff2"](torch.nn.functional.gelu(b["ff1"](h)))
        return x.sum()


class ConvStack(nn.Module):
    def __init__(self, ch: int = 48, depth: int = 10) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, ch, 3, padding=1)]
        for _ in range(depth):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).sum()


ModelSpec = tuple[Callable[[], nn.Module], Callable[[], tuple]]

MODELS: dict[str, ModelSpec] = {
    "deep_mlp": (DeepMLP, lambda: (torch.randn(32, 256, requires_grad=True),)),
    "branching": (BranchingNet, lambda: (torch.randn(32, 192, requires_grad=True),)),
    "transformer": (TransformerBlock, lambda: (torch.randn(8, 128, 256, requires_grad=True),)),
    "convstack": (ConvStack, lambda: (torch.randn(8, 3, 32, 32, requires_grad=True),)),
}


def capture(
    name: str,
    outdir: str | Path,
    budget: float = 0.5,
    budgets: Optional[list[float]] = None,
    solvers: Optional[list[str]] = None,
) -> AuditingSolver:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; have {sorted(MODELS)}")
    build, make_inputs = MODELS[name]

    auditor = AuditingSolver(
        outdir=Path(outdir) / name, label=name, budgets=budgets, solvers=solvers
    )

    model = build()
    args = make_inputs()

    prev_solver = functorch_config.activation_memory_budget_solver
    prev_budget = functorch_config.activation_memory_budget
    try:
        functorch_config.activation_memory_budget_solver = auditor
        # 0 and 1 short-circuit past the knapsack, so keep this strictly inside
        functorch_config.activation_memory_budget = budget

        torch._dynamo.reset()
        compiled = torch.compile(model, backend="aot_eager", dynamic=False)
        out = compiled(*args)
        out.backward()
    finally:
        functorch_config.activation_memory_budget_solver = prev_solver
        functorch_config.activation_memory_budget = prev_budget

    auditor.flush()
    return auditor


def capture_all(
    outdir: str | Path,
    names: Optional[list[str]] = None,
    budget: float = 0.5,
    budgets: Optional[list[float]] = None,
) -> dict[str, AuditingSolver]:
    out: dict[str, AuditingSolver] = {}
    for name in names or list(MODELS):
        try:
            out[name] = capture(name, outdir, budget=budget, budgets=budgets)
            log.info("captured %s", name)
        except Exception:
            log.exception("capture failed for %s", name)
    return out
