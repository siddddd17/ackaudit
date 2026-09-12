"""Graph families for the schedule experiment.

Each isolates one structural property, so that a difference between schedules can
be attributed to something specific rather than to "it's a big graph".

    chain        no branching at all: exactly one topological order, so schedules
                 cannot differ. The control.
    fork         one source feeding parallel independent tails.
    join         parallel independent heads feeding one sink.
    fork_join    both, symmetric branches.
    fork_join_uneven  both, branches of different lengths.
    nested       branches that themselves branch.
    wide         many short branches.
    deep_narrow  few long branches.

The point is not that any of these resembles a real model. It is that if the
schedules diverge, we can say which structure causes it.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class Chain(nn.Module):
    def __init__(self, width: int = 128, depth: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            *[nn.Sequential(nn.Linear(width, width), nn.GELU()) for _ in range(depth)]
        )

    def forward(self, x):
        return self.net(x).sum()


class Fork(nn.Module):
    """One stem, then independent tails that are never merged."""

    def __init__(self, width: int = 128, branches: int = 4, depth: int = 4):
        super().__init__()
        self.stem = nn.Linear(width, width)
        self.tails = nn.ModuleList(
            nn.Sequential(*[nn.Sequential(nn.Linear(width, width), nn.GELU())
                            for _ in range(depth)])
            for _ in range(branches)
        )

    def forward(self, x):
        h = self.stem(x)
        return sum(t(h).sum() for t in self.tails)


class Join(nn.Module):
    """Independent heads on separate slices, merged once at the end."""

    def __init__(self, width: int = 128, branches: int = 4, depth: int = 4):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(*[nn.Sequential(nn.Linear(width, width), nn.GELU())
                            for _ in range(depth)])
            for _ in range(branches)
        )
        self.merge = nn.Linear(width, width)

    def forward(self, xs):
        return self.merge(sum(h(x) for h, x in zip(self.heads, xs))).sum()


class ForkJoin(nn.Module):
    def __init__(self, width: int = 128, branches: int = 4, depths: list[int] | None = None):
        super().__init__()
        depths = depths or [4] * branches
        self.stem = nn.Linear(width, width)
        self.branches = nn.ModuleList(
            nn.Sequential(*[nn.Sequential(nn.Linear(width, width), nn.GELU())
                            for _ in range(d)])
            for d in depths
        )
        self.head = nn.Linear(width, width)

    def forward(self, x):
        h = self.stem(x)
        return self.head(sum(b(h) for b in self.branches)).sum()


class Nested(nn.Module):
    """Branches that branch again, so recomputation chains are trees not paths."""

    def __init__(self, width: int = 128, outer: int = 2, inner: int = 2, depth: int = 3):
        super().__init__()
        self.stem = nn.Linear(width, width)
        self.outer = nn.ModuleList(
            nn.ModuleDict({
                "pre": nn.Linear(width, width),
                "inner": nn.ModuleList(
                    nn.Sequential(*[nn.Sequential(nn.Linear(width, width), nn.GELU())
                                    for _ in range(depth)])
                    for _ in range(inner)
                ),
            })
            for _ in range(outer)
        )
        self.head = nn.Linear(width, width)

    def forward(self, x):
        h = self.stem(x)
        outs = []
        for blk in self.outer:
            hh = blk["pre"](h)
            outs.append(sum(m(hh) for m in blk["inner"]))
        return self.head(sum(outs)).sum()


ModelSpec = tuple[Callable[[], nn.Module], Callable[[], tuple]]


def families(width: int = 128, batch: int = 16) -> dict[str, ModelSpec]:
    def x():
        return (torch.randn(batch, width, requires_grad=True),)

    def xs(n):
        return ([torch.randn(batch, width, requires_grad=True) for _ in range(n)],)

    return {
        # control: unique topological order, schedules cannot differ
        "chain": (lambda: Chain(width, 16), x),
        "fork": (lambda: Fork(width, 4, 4), x),
        "join": (lambda: Join(width, 4, 4), lambda: xs(4)),
        "fork_join": (lambda: ForkJoin(width, 4), x),
        "fork_join_uneven": (lambda: ForkJoin(width, 4, [1, 3, 5, 8]), x),
        "nested": (lambda: Nested(width, 2, 2, 3), x),
        "wide": (lambda: ForkJoin(width, 8, [2] * 8), x),
        "deep_narrow": (lambda: ForkJoin(width, 2, [10, 10]), x),
    }
