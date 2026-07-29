# Benchmarks

Quantitative comparisons across the tools and storage layouts. Every run records full
provenance (library versions, git commit, host/CPU/GPU, dataset shape + nnz + dtype,
chunk/shard/codec) and saves **raw** per-run numbers (JSON/CSV), not just summaries.

| Path | What |
|------|------|
| [`convert_bench.py`](convert_bench.py) | h5ad → zarr **conversion** benchmark (time + peak RAM + output size) across anndata / naive h5py / icechunk / convert-to-zarr. |
| [`zarr_query/`](zarr_query/) | cell by gene matrix **creation + query** benchmarks: dense vs sparse (CSR/CSC), gene- vs cell-wise queries, sorted vs unsorted, zarr read threading. Query engine is the `zarr-query-bench` tool; results/figures collect here. |
| [`rapids_singlecell/`](rapids_singlecell/) | RAPIDS **GPU pipeline** timings: per-step wall / host RSS / GPU VRAM across dataset sizes and cluster configs. GPU-only. |

Each subfolder has `results/` (raw JSON/CSV) and `figures/`.
