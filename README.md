# DataScales — Project

Tools and experiments for fast, memory-bounded storage, access, and analysis of large
single-cell (and future multimodal) genomic data on Zarr — with Icechunk versioning,
Dask streaming, and RAPIDS GPU analysis.

This is a monorepo: each tool under `tools/` is its own installable package with its own
pixi environment; each project folder holds a README and its scripts.

## Tools (`tools/`)

- **[annizarr](https://github.com/A-Jolly-Holly/annizarr)** — now its own repository (`pip install annizarr` once published):
  AnnData zarr stores: convert (`.h5ad`/10x → streaming dense/sparse, mixed-input concat), edit in
  place (add-expr, rechunk, sort, append), and version with Icechunk, through one CLI
  (`annizarr`/`anz`) and a Python API. It grew out of `tools/convert-to-zarr`, `tools/zarrsmith`
  and `tools/scizarr_IC`.
- **Legacy:** [convert-to-zarr](tools/convert-to-zarr/README.md) and [zarrsmith](tools/zarrsmith/README.md)
  stay frozen at their last standalone versions for tracking and for reproducing the findings under
  `benchmarking_results/`; their READMEs map the old commands onto annizarr. `tools/scizarr_IC` is a
  pointer only.
- **[zarr-query-bench](tools/zarr-query-bench/README.md)** — query-time benchmark for a store's
  `X` (row/column, sequential/random/cell-type; dense vs CSR/CSC).
- **[rapids-benchmark](tools/rapids-benchmark/README.md)** — per-step GPU single-cell pipeline
  benchmark (wall / host RSS / VRAM) on Dask-CUDA; GPU-node only.

## Projects

- **[Benchmarking results](benchmarking_results/)** — rapids/zarr findings + figures + raw results.
- **[Icechunk_multiuser_workflow](Icechunk_multiuser_workflow/)** — concurrent, versioned store access.
- **[rapids_user_notebook](rapids_user_notebook/)** — standard GPU single-cell usage notebook.
- **[Megazarr_build](Megazarr_build/)** — building one large Zarr from multiple cohorts.
- **[datavis_realtime_analysis](datavis_realtime_analysis/)** — gene-wise viz + real-time analysis.

## License

Released under the [MIT License](LICENSE).
