from ackaudit.analyze import heterogeneity, q1_input_scale, tie_sets


def test_tie_sets_all_solvers_tie_at_optimum():
    results = [
        {"ok": True, "label": "g", "budget": 0.5, "solver": "a",
         "proxy_peak_memory": 1.0, "true_peak_memory": 2.0},
        {"ok": True, "label": "g", "budget": 0.5, "solver": "b",
         "proxy_peak_memory": 1.0, "true_peak_memory": 3.0},
    ]
    out = tie_sets(results)
    assert out["n_cells"] == 1
    assert out["n_all_tied"] == 1
    assert out["frac_all_tied"] == 1.0
    assert out["max_tied_spread_pct"] == 50.0


def test_tie_sets_partial_tie_reports_only_the_tied_spread():
    results = [
        {"ok": True, "label": "g", "budget": 0.5, "solver": "a",
         "proxy_peak_memory": 1.0, "true_peak_memory": 2.0},
        {"ok": True, "label": "g", "budget": 0.5, "solver": "b",
         "proxy_peak_memory": 1.0, "true_peak_memory": 4.0},
        {"ok": True, "label": "g", "budget": 0.5, "solver": "c",
         "proxy_peak_memory": 2.0, "true_peak_memory": 9.0},
    ]
    out = tie_sets(results)
    assert out["n_all_tied"] == 0
    assert out["max_tied_spread_pct"] == 100.0


def test_tie_sets_empty_input():
    out = tie_sets([])
    assert out["n_cells"] == 0
    assert out["n_all_tied"] == 0
    assert out["max_tied_spread_pct"] == 0.0


def test_heterogeneity_counts_distinct_pairs():
    graphs = [
        {"label": "uniform", "n_items": 4, "memories": [1, 1, 1, 1], "runtimes": [1, 1, 1, 1]},
        {"label": "mixed", "n_items": 3, "memories": [1, 2, 2], "runtimes": [5, 5, 7]},
    ]
    rows = {r["label"]: r for r in heterogeneity(graphs)}
    assert rows["uniform"]["distinct_pairs"] == 1
    assert rows["uniform"]["distinct_memories"] == 1
    assert rows["mixed"]["distinct_pairs"] == 3


def test_heterogeneity_empty_input():
    assert heterogeneity([]) == []


def test_q1_input_scale_reports_min_max_and_dp_mb():
    graphs = [
        {"n_items": 10, "quantised_capacity": 100, "dp_table_bytes": 4_000_000},
        {"n_items": 40, "quantised_capacity": 500, "dp_table_bytes": 20_000_000},
    ]
    out = q1_input_scale(graphs)
    assert out["n_graphs"] == 2
    assert out["n_items_min"] == 10
    assert out["n_items_max"] == 40
    assert out["W_max"] == 500
    assert out["dp_table_mb_max"] == 20.0
    assert out["W_structural_ceiling"] == 10_000


def test_q1_input_scale_empty_input():
    assert q1_input_scale([]) == {}
