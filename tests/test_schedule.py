import pytest

torch = pytest.importorskip("torch")

import torch._functorch._activation_checkpointing.knapsack_evaluator as _ke

from ackaudit.schedule import Schedule


def test_guard_fires_when_evaluator_no_longer_calls_topological_sort(monkeypatch):
    class FakeEvaluator:
        def _get_backward_memory_from_topologically_sorted_graph(self):
            # simulates the upstream change to nx.lexicographical_topological_sort
            return nx.lexicographical_topological_sort  # noqa: F821

    monkeypatch.setattr(_ke, "KnapsackEvaluator", FakeEvaluator)

    with pytest.raises(RuntimeError, match="no longer calls nx.topological_sort"), Schedule(
        "lexicographic"
    ):
        pass


def test_guard_passes_on_current_evaluator():
    with Schedule("lexicographic"):
        pass
