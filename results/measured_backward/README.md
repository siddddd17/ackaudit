# measured_backward

Output of `experiments/measured_backward/run_d.py`. Each run writes a JSON file
and a text report named `<model>_scale<N>`.

All runs: torch 2.14.0+cu130, NVIDIA GeForce GTX 1650 (4 GB), batch 2, five
identical repetitions per cell. Memory values in the JSON are bytes;
`measured_peaks` is the list of five `torch.cuda.max_memory_allocated()` readings
and `resident_before` is `torch.cuda.memory_allocated()` after warmup, before
the measured repetitions. The reports show `measured_peak - resident_before` in
MB.

Candidate-node counts: llama 164 at scale 4, 244 at scale 6, 324 at scale 8;
bert 354 at scale 8.

| directory | model | scale | budgets |
|---|---|---|---|
| `coarse/` | llama | 8 | 0.10 to 0.90 in steps of 0.10 |
| `coarse_rerun/` | llama | 8 | same, separate process |
| `fine/` | llama | 8 | 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35 |
| `fine/` | bert | 8 | 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60 |
| `baseline/` | bert | 8 | 0.70, 0.80, 0.90 |
| `scale4/` | llama | 4 | 0.05, 0.10, 0.15, 0.20, 0.30 |
| `scale6/` | llama | 6 | 0.05, 0.10, 0.15, 0.20, 0.30 |

`coarse` and `coarse_rerun` are the same configuration run in two separate
processes. Their `measured_peaks` are identical at every budget.

`coarse` and `fine` overlap at budgets 0.10, 0.20 and 0.30. The readings are
identical at 0.10 and differ by 524288 bytes at 0.20 and 0.30. So the zero
spread reported inside a single run is a within-run figure; across runs of the
same budget the observed difference is up to 0.5 MB.

Cells with `n_items` 0 and `solver_called` false are budgets where the
partitioner returned before calling the solver: llama scale 8 at 0.70, 0.80 and
0.90, and bert scale 8 at 0.90. The measured peak for those cells is the peak
with no knapsack-selected rematerialization.

The bert runs require the `max_position` change to `BertWrap` in
`ackaudit/hf_models.py`; without it, scale 8 raises during tracing because the
sequence length exceeds `max_position_embeddings`.
