import pytest

torch = pytest.importorskip("torch")

from ackaudit.audit import QUANTISATION_SCALE, SOLVERS, audit_instance
from ackaudit.capture import capture


def test_all_solvers_registered():
    assert set(SOLVERS) == {"greedy", "ilp", "dp", "dp_sliding_hirschberg"}


def test_capture_records_one_instance(tmp_path):
    auditor = capture("deep_mlp", tmp_path, budget=0.5, budgets=[0.3, 0.6])
    assert len(auditor.records) == 1
    assert auditor.records[0].n_items > 0


def test_quantised_capacity_stays_under_ceiling(tmp_path):
    # budget is clamped to [0,1] upstream, so W can't exceed the scale factor
    auditor = capture("deep_mlp", tmp_path, budget=0.9, budgets=[0.5])
    for rec in auditor.records:
        assert rec.quantised_capacity <= QUANTISATION_SCALE


def test_every_solver_produces_a_partition(tmp_path):
    auditor = capture("convstack", tmp_path, budget=0.5, budgets=[0.4])
    for res in auditor.results:
        if res.ok:
            assert res.n_saved + res.n_recomputed > 0
