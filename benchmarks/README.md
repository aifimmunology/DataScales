# Benchmarks

Quantitative comparisons across the tools and storage layouts — query latency,
parallel-write scaling, dense vs. sparse (CSR/CSC), chunk/codec tradeoffs, Icechunk
vs. plain Zarr, and RAPIDS GPU-pipeline timings.

Every run records full provenance (library versions, git commit, host/CPU/GPU, dataset
shape + nnz + dtype, chunk/shard/codec) and saves **raw** per-run numbers, not just summaries.

Scripts and raw results (JSON/CSV) live here.
