# Benchmarking results

**Findings, not code.** Each folder holds what a benchmark *found* — a findings README,
`figures/`, and `results/` (raw per-run JSON/CSV with full provenance: library versions, git
commit, host/CPU/GPU, dataset shape + nnz + dtype, chunk/shard/codec). The instruments that
produced them live in [`tools/`](../tools/), each with its own README.

| Findings | What | Instrument |
|----------|------|------------|
| [`zarr_layouts/`](zarr_layouts/) | cell×gene **storage-layout query** benchmarks: dense vs sparse (CSR/CSC), gene- vs cell-axis queries, sorted vs unsorted, zarr read threading. | [`tools/zarr-query-bench`](../tools/zarr-query-bench/); stores built with [`tools/convert-to-zarr`](../tools/convert-to-zarr/) |
| [`rapids_vs_scanpy/`](rapids_vs_scanpy/) | **GPU pipeline** timings: RAPIDS vs Scanpy (in-memory and Zarr/Dask-streamed) per-step wall / host RSS / GPU VRAM across dataset sizes and cluster configs. | [`tools/rapids-benchmark`](../tools/rapids-benchmark/) |
