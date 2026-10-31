# RAPIDS single-cell on the GPU

This capsule is a ready-to-use GPU environment for single-cell analysis with
[rapids-singlecell](https://rapids-singlecell.readthedocs.io), plus guides for using it:

- **The environment:** RAPIDS, rapids-singlecell, Dask-CUDA, scanpy, scvi-tools and torch, squidpy and
  spatialdata, all pinned to versions that work together on CUDA 13. Use it here, or copy it to your own
  capsule or machine (see below).
- **Two notebooks:** an out-of-core walkthrough for stores too big for memory, and a spatial
  transcriptomics example.
- **A basic `run` script** that adds PCA, UMAP and leiden clusters to a store, from the terminal.

Tested on one Tesla T4 (15 GB), where a million cells take about four minutes.

## Contents

```
code/
├── README.md       this file
├── environment/    requirements.txt and pixi.toml: the environment's package list
├── notebooks/      the guides
├── run             the pipeline: ./run --input <store>
└── process/        the pipeline's code and tests
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

- **Duplicate this capsule.** You get the environment, notebooks and scripts as they are.
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
3. Set the path variable at the top to your data, and run the cells in order.

| Notebook | What it shows | Data it needs |
|---|---|---|
| `rapids_scrna_out_of_core_dask.ipynb` | The analysis of large stores step by step: normalize, highly variable genes, PCA, UMAP and leiden, reading the counts in blocks so they never need to fit in memory | An AnnData or SpatialData `.zarr`, or an icechunk repo holding either, with raw counts in a sparse (CSR) `X`. Set `STORE` (and `BRANCH` for icechunk, `TABLE` for SpatialData) |
| `rapids_spatial_atera_breast_cancer.ipynb` | A spatial analysis: QC, a cell-neighbor graph, cell-type labels from markers, tissue niches, ligand-receptor signaling, spatially variable genes. Can save its labels back into the table | A SpatialData `.zarr`, or an icechunk repo laid out like one, whose table has raw counts and `obsm["spatial"]`; details in the notebook. Set `DATA_PATH`. Needs internet for OmniPath |

The out-of-core notebook only reads. The spatial one writes only if you set `SAVE = True` in its last
section, and then only into the table. The marker genes and settings are starting points to adjust for
your own tissue.

## The `run` script

`run` does the same analysis as the out-of-core notebook and saves the results into the store. Run it
from a terminal in the capsule:

```bash
./run --input /data/<asset>/<store>                                   # PCA, UMAP, leiden
./run --input /data/<asset>/<store> --integration harmony --batch-keys chip_id
./run --help                                                          # every setting and its default
```

- **Input:** an icechunk repo (it processes the `qc` branch; change it with `--branch`), or a plain
  `.zarr`. Either can hold an AnnData or a SpatialData; for SpatialData only the table is used
  (`--table`, default `table`). The counts must be a sparse (CSR) matrix.
- **Results in the store:** `obsm/X_pca`, `obsm/X_umap` (and `obsm/X_pca_harmony`), `obs/leiden`,
  `var/highly_variable`, `layers/lognorm` (normalized, log-transformed counts) and `uns/process`
  (the settings used).
- **Where they're saved:**
  - icechunk: a new commit on the branch;
  - a `.zarr` you can write to: the same `.zarr`;
  - a read-only `.zarr` (a data asset), or with `--output <path>`: a copy, by default
    `/results/<name>.processed.zarr`. `--output-mode results-only` copies everything but the counts
    (`X`, `layers`, `raw`); for SpatialData that's just the table, as an AnnData `.zarr`.
  - SpatialData: the results go in the table only; images, shapes and points are untouched.
- **A read-only icechunk data asset** needs `--origin s3://bucket/prefix`, where its writable copy lives.
- **`/results`** also gets `run.json` (timings, memory, where results went) and `cluster_summary.csv`
  (cells per cluster, by sample).

Running again replaces the earlier results. On icechunk, the earlier ones stay in the branch's history.

## SpatialData and icechunk

To edit a SpatialData table (add clusters, embeddings, a layer) without rewriting the images and
shapes, and keep a history of those edits, put it in an icechunk repo laid out like a SpatialData
store (the table at `tables/table`). Each `run`, or a save from the spatial notebook, is one commit
that touches only the table.

As of spatialdata 0.8.0, `sdata.write()` and `sd.read_zarr()` can't use icechunk directly, so:
- write the repo with `anndata.io.write_elem` for the table (and spatialdata's element writers for images
  and labels; points and GeoParquet shapes don't work there yet);
- read the table with `anndata.io.read_elem(root["tables/table"])`, as the notebooks do;
- keep a plain SpatialData `.zarr` for tools that need the whole object (napari, `sd.read_zarr`).

A plain SpatialData `.zarr` also works everywhere above. Edits go in place without history.

### Tests

```bash
cd process && python -m pytest tests -q
```

They use small synthetic stores; the pipeline tests need a GPU.
