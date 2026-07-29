# zarr cellxgene — creation & query benchmarks

How storage layout affects loading a cellxgene (cells × genes) expression matrix.

**Variables under test** (change one per run, hold the rest byte-identical):
- **Format** — dense vs sparse CSR vs CSC
- **Query type** — gene-wise (column) vs cell/cell-type (row) reads
- **Row order** — sorted (contiguous cell-type spans) vs unsorted (scattered runs)
- **Zarr read threading** — `async.concurrency` (dispatch) × `threading.max_workers` (decode)
- **Chunk / shard / codec** — the underlying layout knobs

**Tools** (the engines live outside this folder; results/figures collect here):
- Creation — [`../convert_bench.py`](../convert_bench.py) and `tools/convert-to-zarr`
- Query — `tools/zarr-query-bench` (`pixi run zarr-bench …`); capture layout with `zarr-bench-inspect`

`results/` holds raw per-run JSON/CSV; `figures/` holds plots.

<!-- TODO: drop in the sparse-vs-dense / gene-vs-cell / sorted-vs-unsorted result sets + figures. -->
