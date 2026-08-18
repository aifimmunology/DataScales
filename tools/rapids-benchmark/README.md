# rapids-benchmark

Configurable per-step benchmark for the GPU single-cell pipeline.
[`rapids_benchmark.py`](rapids_benchmark.py) times **wall-clock + peak host RSS (process
tree) + peak GPU VRAM (NVML, device-wide)** for each stage of an out-of-core, multi-GPU
pipeline that reads a Zarr store lazily through Dask-CUDA:

```
load_zarr → h2d_transfer → preprocessing (qc+normalize+log1p) → hvg → scaling
          → pca → harmony (if --batch-key) → neighbors → umap → leiden → write_results
```

> Findings produced with this tool live in
> [`benchmarking_results/rapids_vs_scanpy/`](../../benchmarking_results/rapids_vs_scanpy/README.md)
> (RAPIDS vs Scanpy per-step speedups, memory, multi-GPU scaling, raw sweeps).

Every field of the `Config` dataclass has a matching `--flag`, so a sweep is a shell loop —
no source edits. GPU/CUDA-only; run it on the GPU node through this directory's pixi env
(linux-64; `--help` works anywhere):

```bash
# 4-GPU run, capacity preset (tcp + managed memory)
pixi run python rapids_benchmark.py \
    --data-path /path/to/5M.zarr --gpus 0,1,2,3 --label 5M_4gpu
```

Key knobs: `--gpus` (physical ids, single-sourced to cluster + NVML + client RMM),
`--preset capacity|speed`, `--rmm-mode managed|pool`, `--zarr-concurrency`/
`--zarr-max-workers` (applied on *every* worker), `--chunk-rows`, and the pipeline params
(`--n-top-genes`, `--n-comps`, `--n-neighbors`, `--leiden-resolution`, `--batch-key ""` to
skip harmony). By default the final step writes the UMAP embedding (`obsm/X_umap`) and
leiden labels (`obs/leiden`) back **into the input store** as an anndata-readable layer — no
h5ad, no X rematerialization; `--results-store` retargets it, `--no-write-results` skips it.
Each run appends a per-step summary to `results/Run_results.txt` headed by the date, store,
GPUs, and any cfg options left off their defaults. See the module docstring and `--help` for
the rest.

[`sweep_single_gpu.sh`](sweep_single_gpu.sh) loops the benchmark over datasets × RMM modes ×
chunk-rows × thread splits (edit the CONFIG block; `OUTDIR` env overrides the results dir).
