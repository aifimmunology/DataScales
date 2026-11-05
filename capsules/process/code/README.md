# RAPIDS single-cell on the GPU

This capsule is a ready-to-use GPU environment for single-cell analysis with
[rapids-singlecell](https://rapids-singlecell.readthedocs.io), plus two notebooks that show how to use it:

- **The environment:** RAPIDS, rapids-singlecell, Dask-CUDA, scanpy, scvi-tools and torch, squidpy and
  spatialdata, all pinned to versions that work together on CUDA 13. Use it here, or copy it to your own
  capsule or machine (see below).
- **Two notebooks:** an out-of-core walkthrough for stores too big for memory, and a spatial
  transcriptomics example. Each can save its results to a new `.zarr`.
- **Optional:** the out-of-core notebook's steps as a pipeline you can start from the App Panel or as a
  Reproducible Run (see the last section). You don't need it to use the environment or the notebooks.

Tested on one Tesla T4 (15 GB), where a million cells take about four minutes.

## Contents

```
code/
├── README.md       this file
├── environment/    requirements.txt and pixi.toml: the environment's package list
├── notebooks/      the two notebooks, and test_notebooks.sh
├── run             optional: the App Panel / Reproducible Run pipeline (./run --help)
├── process/        that pipeline (config, cluster, steps, store, report, main) and its tests
└── test_gpu.sh     runs the notebook tests and the pipeline tests on a GPU
```

## The environment

The main packages:

| Purpose | Packages |
|---|---|
| GPU single-cell | rapids-singlecell 0.18.1, cuDF / cuML / cuGraph 26.10, CuPy, RMM |
| Out-of-core | Dask 2026.7.1, Dask-CUDA 26.10 |
| Single-cell on CPU | scanpy 1.12.4, anndata 0.13.4 |
| Deep learning | torch 2.14.1 (CUDA 13), scvi-tools 1.5.1 |
| Spatial | squidpy 1.8.3, spatialdata 0.8.0, omnipath 1.0.12 |
| Storage | zarr 3.4.1, icechunk 2.3.0, annizarr 0.2.3 |

The full list is [`environment/requirements.txt`](environment/requirements.txt). It needs an NVIDIA
GPU whose driver supports CUDA 13; the packages bring their own CUDA libraries.

### Copy it into another Code Ocean capsule

Pick one:

- **Duplicate this capsule.** You get the environment and notebooks as they are.
- **Set up a new capsule:**
  1. On its environment page, choose the base image `codeocean/cuda12-runtime-uv:main-5692e50`.
  2. Open the **pip3** installer's gear menu, choose **Edit Bulk**, and paste in
     `environment/requirements.txt`.
  3. Pick a GPU machine and build.

### Use it outside Code Ocean

On Linux (x86-64) with an NVIDIA driver for CUDA 13, use either:

```bash
# pixi
CONDA_OVERRIDE_CUDA=13 pixi install --manifest-path environment/pixi.toml
pixi shell --manifest-path environment/pixi.toml

# or plain pip, with uv
uv venv --python 3.13 rapids-env
source rapids-env/bin/activate
uv pip install -r environment/requirements.txt
```

It's about 5 GB to download. With pixi, set `PIXI_CACHE_DIR` to a disk with room.

## Notebooks

Each notebook reads its data from a **data asset you attach**:
1. Attach the data asset to the capsule. It appears under `/data/<asset name>/`.
2. Open the notebook in JupyterLab or VS Code.
3. Set `STORE` in the settings cell at the top to your data, and run the cells in order.

The settings cell also has `TABLE` (a SpatialData store's table), `SAVE` and `OUTPUT` (see below), and
`RESULTS`, the folder for the default `OUTPUT` and, in the out-of-core notebook, `umap.png` and
`cluster_summary.csv` (default `/results`).

| Notebook | What it shows | Data it needs |
|---|---|---|
| `rapids_scrna_out_of_core_dask.ipynb` | The analysis of large stores step by step: normalize, highly variable genes, PCA, optional harmony, UMAP and leiden, reading the counts in blocks so they never need to fit in memory | An AnnData `.zarr`, or a SpatialData `.zarr` (its table `tables/<TABLE>` is used), with raw counts in a sparse (CSR) `X`. `OBS_COLUMNS` (default `sample_id`) are optional: those the store has are read for plots. `BATCH_KEYS` are for harmony |
| `rapids_spatial_atera_breast_cancer.ipynb` | A spatial analysis: QC, a cell-neighbor graph, cell-type labels from markers, tissue niches, ligand-receptor signaling, spatially variable genes | A SpatialData `.zarr` whose table has raw counts and `obsm["spatial"]`; details in the notebook. Needs internet for OmniPath |

The marker genes and settings are starting points to adjust for your own tissue.

### Saving the results

Nothing is written unless you set `SAVE = True`. The input is never changed, so a read-only data asset is
fine as input.

- **What's saved:**
  - out-of-core: `obsm/X_pca` (and `X_pca_harmony`), `obsm/X_umap`, `obs/leiden`, `var/highly_variable`,
    `layers/lognorm` (normalized, log-transformed counts, with a `target_sum` attr) and `uns/process` (the
    settings, the input path, and its snapshot id or `"zarr"`). It also writes `umap.png` and
    `cluster_summary.csv` (cells per cluster, by `sample_id` when there is one) to `RESULTS`.
  - spatial: `obs/spatial_cell_type`, `spatial_leiden`, `spatial_niche`, `obsm/X_pca_spatial`,
    `X_umap_spatial` and `uns/spatial_process`, under their own names so they don't replace the out-of-core
    results. Cells removed by QC get empty values.
- **Where:** a new `.zarr` at `OUTPUT` (`.zarr` is added if it's missing), by default
  `<RESULTS>/<input name>.processed.zarr`. It's a full
  copy of the input with the results added, so it stands alone (it can become its own data asset):
  - an AnnData input is copied with `annizarr.rechunk`;
  - a SpatialData input is copied whole (images, labels, shapes, points, tables) and the results go in its
    table, so the copy opens with `sd.read_zarr`.
- The copy is built at `<OUTPUT>.partial` and renamed when it's complete. An existing output is never
  overwritten: the next free `<name>-1.zarr`, `-2`, … is used instead. A save that dies midway leaves the
  `.partial` folder, which you can delete.

## Icechunk instead of zarr

Both notebooks can read an icechunk repo, an AnnData one or one laid out like a SpatialData store (the
table at `tables/<TABLE>`), and save the results as **one commit on `BRANCH`**. Uncomment the icechunk
block at the end of the settings cell; it sets `repo`, and the open and save cells use it when it's set.

- A data asset is a read-only copy of a repo; set `ORIGIN` to its writable location (e.g.
  `s3://bucket/prefix`) to commit there. The data is then read through `ORIGIN` too, so the results are
  computed on the same snapshot they're committed onto.
- If the branch moved since the store was opened, nothing is written; reopen it and rerun.
- Any branch can take the results, `main` included. Earlier results stay in the history.

As of spatialdata 0.8.0, `sdata.write()` and `sd.read_zarr()` can't use icechunk, so on a SpatialData
layout only the table is read and written (with `anndata.io.read_elem` / `write_elem`). spatialdata's own
element writers work in an icechunk session for images, labels, tables and zarr-encoded shapes, but not
for points or GeoParquet shapes. Keep a plain SpatialData `.zarr` for tools that need the whole object
(napari, `sd.read_zarr`).

## Optional: run from the App Panel or as a Reproducible Run

The capsule is first an environment with notebooks; this pipeline is an extra. It runs the out-of-core
notebook's steps without opening a notebook, on every GPU of the machine:

```
load -> normalize/log1p -> highly variable genes -> PCA (+ harmony)  -> neighbors -> UMAP -> leiden
                                                 \-> scVI (train)  -/
```

**A Reproducible Run with no input store does nothing but print how to use the capsule, and succeeds.**
To process data, attach a data asset, open the **App Panel**, set *Input store*, and press **Run**.

### App Panel fields

The panel passes named parameters (`--<name> <value>`); an empty field keeps the default.

| Category | Field | `--name` | Default | What it does |
|---|---|---|---|---|
| Input data | Input store | `input` | empty: print usage | An AnnData `.zarr`, a SpatialData `.zarr`, or an icechunk repo; raw counts in a CSR `X` |
| | SpatialData table | `table` | `table` | SpatialData: which `tables/<name>` |
| | Icechunk branch | `branch` | `main` | Read from, and the results committed on |
| | Icechunk origin | `origin` | — | `s3://…` behind a read-only data asset; read and committed through it |
| | Subset column / values | `subset_column`, `subset_values` | — | Use only cells whose column is one of the values (`SP1, SP2`) |
| Embedding | Method | `method` | `PCA` | `PCA` or `scVI (train)`, see below |
| | Batch key | `batch_key` | — | PCA: harmony on it; scVI: the batch (more keys, comma-separated: harmony uses all, scVI uses the rest as covariates). Every cell needs a value |
| | Highly variable genes | `n_top_genes` | 2000 | |
| | PCA components | `n_comps` | 50 | Neighbors use the first `n_pcs` (30) |
| Neighbors & UMAP | Neighbors | `n_neighbors` | 20 | |
| | UMAP min_dist | `umap_min_dist` | 0.45 | |
| Clustering | Leiden resolutions | `leiden_resolutions` | 1.1 | Comma-separated; one neighbor graph for all; empty means 1.1 |
| Output | Output path | `output` | `/results/<name>.processed.zarr` | Zarr input only |

### Methods

- **PCA:** normalize to 10,000 counts and log1p, seurat HVGs, scale (max 10, no centering), PCA
  (`obsm/X_pca`). With a batch key, harmony (`obsm/X_pca_harmony`), which neighbors then use.
- **scVI (train):** trains a new scVI model on the **raw counts of the HVGs** of every (subset) cell, with
  the batch key as its batch. The latent space is `obsm/X_scVI`; the model is saved to `/results/scvi_model`
  (`scvi_model-1`, … if that exists). The HVG counts are held in host memory (about 2–3 GB for a million
  cells × 2,000 genes), and the Dask-CUDA cluster is shut down before training, so torch has the GPU to itself.
  By default scvi-tools picks the number of epochs from the cell count (`min(400, 20000 / n_cells × 400)`):
  8 epochs at a million cells, but 400 at 20,000 cells; at batch size 128 a million cells is about 63,000
  steps, which can take a while on a T4. Set `SCVI_MAX_EPOCHS` (and maybe a larger `SCVI_BATCH_SIZE`) to bound it.

### Outputs

The same rules as the notebooks' save section:

- **Zarr input:** a new full copy at `output` (default `/results/<name>.processed.zarr`), built at
  `.partial` and renamed; an existing output is never overwritten (`-1`, `-2`, …). A SpatialData store is
  copied whole and the results go in its table.
- **Icechunk input:** one commit on `branch` (through `origin` if given); nothing is written if the branch
  moved during the run.
- **Checked before any GPU work**, each a one-line error (exit 1): an input that can't be opened, a SpatialData
  store without the table, a batch key or subset column the store doesn't have, a subset with no cells, and
  cells with no batch value (missing, or `""` as ingest writes it). Pick another key, or a subset without them.

Written: the embedding (`X_pca`, `X_pca_harmony` or `X_scVI`), `obsm/X_umap`, `obs/leiden` (or one
`obs/leiden_<res>` per resolution when several are given), `var/highly_variable`, `layers/lognorm` (all cells,
`target_sum` attr) and `uns/process` (method, basis, resolutions, subset, every setting, the input and its
snapshot). With a subset, cells outside it get empty values. Earlier `X_pca`/`X_pca_harmony`/`X_scVI`/`X_umap`
and `leiden*` results in the store are replaced or removed, so all of them come from one run. Beside it, in
`/results`: `umap.png`, `cluster_summary.csv` (`cluster_summary_<key>.csv` per resolution) and
`process_report.txt` (wall time and peak host/GPU memory per step).

### Every other setting

Each field of `Config` in [`process/config.py`](process/config.py) is a setting: change its default there,
pass `--<name> <value>` (`--n-pcs` works too), or set the upper-case environment variable (`N_PCS=40`).
Flags beat environment variables, and an empty value (or a bare `--flag`) keeps the default. Relative paths
are read from the folder you start `./run` in.

`gpus` (e.g. `0,1`) sets `CUDA_VISIBLE_DEVICES` for the whole run: the Dask workers, `mg_ivfflat`/`mg_ivfpq`
neighbors (which use every GPU the process sees), and scVI, which trains on the first one listed. The workers
talk over tcp; UCX would need `distributed-ucxx-cu13`, which the environment doesn't have.

| Group | Settings (default) |
|---|---|
| Cluster | `gpus` (all visible), `rmm_mode` (`managed`: GPU memory spills to host RAM; `pool`: `rmm_pool_size` (0.8) of each GPU; `none`), `threads_per_worker` (1), `enable_cudf_spill` (true), `zarr_concurrency` (12), `zarr_max_workers` (12), `sample_interval_s` (0.1: how often peak memory is sampled for the report) |
| Data | `chunk_rows` (24000), `target_sum` (1e4), `sample_key` (`sample_id`), `results_dir` (`/results`), `plots` (true) |
| Steps | `hvg_flavor` (`seurat`), `hvg_batch_key`, `scale_max_value` (10), `n_pcs` (30), `neighbors_algorithm` (`ivfflat`; `mg_ivfflat` for several GPUs), `umap_n_components` (2), `leiden_iterations` (100), `harmony_max_iter` (30), `harmony_theta` (2), `harmony_alpha` (0.2), `random_seed` (0) |
| scVI | `scvi_n_latent` (30), `scvi_n_layers` (2), `scvi_n_hidden` (128), `scvi_dropout_rate` (0.1), `scvi_gene_likelihood` (`nb`), `scvi_max_epochs` (0: scvi-tools picks from the cell count), `scvi_batch_size` (128), `scvi_lr` (1e-3), `scvi_train_size` (0.9), `scvi_early_stopping` (false), `scvi_early_stopping_patience` (45), `scvi_accelerator` (`auto`), `scvi_model_dir` (`<results_dir>/scvi_model`) |

The scVI training settings are deliberately not on the panel. For a panel run or a Reproducible Run, change
their defaults in `process/config.py`, or add an `export SCVI_MAX_EPOCHS=50`-style line to `run`.

### Examples

```bash
./run                                                     # prints usage, exits 0
./run --input /data/pbmc/pbmc.zarr                        # PCA, leiden 1.1 -> /results/pbmc.processed.zarr
./run --input /data/pbmc/pbmc.zarr --batch_key chip_id --leiden_resolutions 0.5,1.0,2.0
./run --input /data/pbmc/pbmc.zarr --subset_column sample_id --subset_values SP1,SP2
SCVI_N_LATENT=20 SCVI_MAX_EPOCHS=50 ./run --input /data/pbmc/pbmc.zarr --method scvi --batch_key sample_id
./run --input /data/tissue/tissue.zarr --table table      # SpatialData: the results go in the copy's table
./run --input /data/repo --origin s3://bucket/repo --branch main   # icechunk: one commit on the origin
GPUS=0,1 NEIGHBORS_ALGORITHM=mg_ivfflat ./run --input ... # two GPUs
```

## Tests

```bash
./test_gpu.sh                          # the notebooks and the pipeline; `./test_gpu.sh big` adds a 1M-cell timing
notebooks/test_notebooks.sh            # the notebooks only: every case, or name some, e.g. scrna_zarr spatial_sdata
python -m pytest process/tests         # the pipeline only
```

It builds tiny synthetic stores (an AnnData `.zarr`, two ingest-style `.zarr`s with no `layers`, one also
without `sample_id`, a SpatialData `.zarr`, icechunk repos, and a read-only icechunk copy with a writable
origin), runs each notebook headless with `jupyter nbconvert --execute` and saving on, and checks that the new
object or commit holds the results and reads back. It needs a GPU and internet (OmniPath). `WORK=<dir>`
keeps the stores and executed notebooks. The pipeline tests check the settings and the save rules, then run
`./run` end to end (PCA, harmony, scVI, several resolutions, a subset, an ingest-style store, SpatialData,
icechunk through an origin, no input); without a GPU, only the parts that need none run.
