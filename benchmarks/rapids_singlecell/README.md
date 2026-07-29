# rapids-singlecell — GPU pipeline benchmark

Per-step **wall-clock + peak host RSS + peak GPU VRAM** for an out-of-core, multi-GPU
single-cell pipeline that reads a Zarr store lazily through Dask-CUDA. GPU/CUDA-only —
run on the GPU node's pixi env, not the local CPU env.

## Steps

`load_zarr` → `h2d_transfer` → `preprocessing` (qc + normalize + log1p) → `hvg` →
`scaling` → `pca` → `harmony` (if `--batch-key`) → `neighbors` → `umap` → `leiden` →
`write_h5ad` (if `--h5ad-out`).

## Run

Defaults live in the `Config` dataclass in [`rapids_benchmark.py`](rapids_benchmark.py);
**every field has a matching `--flag`**, so a sweep is a shell loop — no source edits.

```bash
# 4-GPU baseline (capacity preset: tcp + managed memory)
pixi run python rapids_benchmark.py \
    --data-path /path/to/2M_50M.zarr --gpus 0,1,2,3 \
    --results-json results/2M_4gpu.json --label 2M_4gpu

# scaling sweep: same store, vary GPU count
for g in "0" "0,1" "0,1,2,3"; do
  pixi run python rapids_benchmark.py --data-path /path/to/2M_50M.zarr \
      --gpus "$g" --results-json "results/2M_${g//,/-}.json" --label "2M_${g//,/-}gpu"
done
```

**Canned single-vs-multi-GPU × zarr-config grid:** [`sweep_gpu_zarr.sh`](sweep_gpu_zarr.sh)
runs `{gpus} × {zarr concurrency 4,16} × {max-workers 4,8}` (neighbors algo derived:
multi→`mg_ivfflat`, single→`ivfflat`), one isolated subprocess per config, and aggregates
every per-run JSON into `results/sweep_summary.{csv}` + a printed comparison table. Sets
`threads_per_worker = 64 / n_gpus` to fill the 64-core box while keeping total host threads
equal across the GPU-count variable. Edit the grid/DATA at the top; env-var overrides:

```bash
DRY_RUN=1 bash sweep_gpu_zarr.sh                       # preview the planned commands
bash sweep_gpu_zarr.sh                                 # run the full grid
DATA=/home/workspace/zarrs/other.zarr bash sweep_gpu_zarr.sh
```

## Key knobs

| Knob | Flag | Note |
|------|------|------|
| GPUs | `--gpus 0,1,2,3` | physical ids; single-sourced to cluster + NVML + client RMM |
| Cluster preset | `--preset capacity\|speed` | `capacity`=tcp+managed, `speed`=ucx+rmm-pool |
| Hard overrides | `--protocol`, `--rmm-mode`, `--rmm-pool-size`, `--threads-per-worker`, `--enable-cudf-spill/--no-enable-cudf-spill` | beat the preset |
| Zarr read | `--zarr-concurrency`, `--zarr-max-workers` | **applied on every worker** (see below) |
| Layout | `--chunk-rows` | row block; multiple of the store's row chunk |
| Pipeline | `--n-top-genes`, `--n-comps`, `--n-neighbors`, `--neighbors-algorithm`, `--leiden-resolution`, `--umap-min-dist`, `--pca-float64/--no-pca-float64`, `--batch-key` | `--batch-key ""` skips harmony |


## Output

- `results/*.json` — raw provenance (versions, CUDA/driver, GPU model, dataset shape/nnz)
  + per-step numbers. `--results-txt` appends a human-readable table.
- `figures/` — plots embedded below.

## Results

_Pending: per-step timings across dataset sizes + figures._

| dataset | GPUs | total wall | peak host | peak GPU | notes |
|---------|------|-----------|-----------|----------|-------|
| _TODO_  |      |           |           |          |       |
