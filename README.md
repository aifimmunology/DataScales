# DataScales — Project

Tools and experiments for fast, memory-bounded storage, access, and analysis of large
single-cell (and future multimodal) genomic data on Zarr — with Icechunk versioning,
Dask streaming, and RAPIDS GPU analysis.

This is a monorepo: each tool under `tools/` is its own installable package with its own
pixi environment; each project folder holds a README and its scripts. A single agent
(`.claude/` + `CLAUDE.md`) at the root spans every tool and project.

## Tools (`tools/`)

- **[convert-to-zarr](tools/convert-to-zarr/README.md)** — AnnData / `.h5ad` → anndata-readable
  Zarr v3 (streaming dense/sparse, multi-h5ad concat, sort/partition, optional Icechunk);
  includes a conversion benchmark vs anndata / h5py / icechunk (`benchmarking/`).
- **[zarrsmith](tools/zarrsmith/README.md)** — edit existing AnnData zarr stores in place:
  add-expr (lognorm gene layer), rechunk, physical sort, append; builds on convert-to-zarr's core.
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
