"""Re-run the RAPIDS single-cell UMAP pipeline on cells selected in the datavis app.

Script form of rapids_user_notebook/rapids_GPU_sc_analysis.ipynb, with one change:
the AnnData is built from only the barcodes listed in the datavis selection file.
Run on the HISE/GPU node:  pixi run python rerun_umap_on_selection.py
"""

# ── Variables ────────────────────────────────────────────────────────────────
data_pth       = "/home/workspace/temp/expression.zarr"  # CSR zarr store (raw counts)
SELECTION_FILE = "selection.json"                         # barcodes from the datavis app
output_h5ad    = "./selection_analysis.h5ad"
ROW_CHUNK_SIZE = 24_000
RANDOM_SEED    = 4242
BATCH_KEY      = []   # obs columns to harmony-integrate on; [] = skip harmony

# ── CUDA dask cluster ────────────────────────────────────────────────────────
import dask
dask.config.set({"distributed.scheduler.worker-ttl": None})
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

cluster = LocalCUDACluster(
    CUDA_VISIBLE_DEVICES="0,1,2,3",
    protocol="tcp",
    threads_per_worker=16,
    rmm_managed_memory=True,
    rmm_allocator_external_lib_list="cupy",
    enable_cudf_spill=True,
)
client = Client(cluster)

# ── Imports + managed memory ─────────────────────────────────────────────────
import json
import numpy as np
import zarr
import anndata as ad
from anndata.experimental import read_elem_lazy as read_dask
import rapids_singlecell as rsc

# zarr read-tuning is per worker process — set it ON the workers, not the client.
client.run(lambda: __import__("zarr").config.set(
    {"async.concurrency": 4, "threading.max_workers": 4}))

import rmm
import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator
rmm.reinitialize(managed_memory=True, pool_allocator=False)
cp.cuda.set_allocator(rmm_cupy_allocator)

# ── Load ONLY the selected cells ─────────────────────────────────────────────
f = zarr.open(data_pth, mode="r")
shape = f["X"].attrs["shape"]                       # [n_obs, n_vars]
obs = ad.io.read_elem(f["obs"])

selection = json.load(open(SELECTION_FILE))         # {"barcodes": [...], ...}
rows = obs.index.get_indexer(selection["barcodes"])
assert (rows >= 0).all(), "some selected barcodes are not in this store"
rows = np.unique(rows)                              # ascending, deduped → chunk-friendly

X_dask = read_dask(f["X"], (ROW_CHUNK_SIZE, shape[1]))
raw_counts = np.issubdtype(X_dask.dtype, np.integer)
X_dask = X_dask[rows]                               # keep only the selected cells (lazy)
if raw_counts:
    X_dask = X_dask.astype(np.float32)

adata = ad.AnnData(X=X_dask, obs=obs.iloc[rows].copy(), var=ad.io.read_elem(f["var"]))
print("Selected cells:", adata.shape)
rsc.get.anndata_to_GPU(adata)

# ── Preprocess (only if raw counts) ──────────────────────────────────────────
if raw_counts:
    rsc.pp.normalize_total(adata)
    rsc.pp.log1p(adata)

# ── Highly variable genes ────────────────────────────────────────────────────
rsc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
adata_full = adata                                  # full genes → written to output
adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()

n_rows, n_cols = adata.shape
adata.X = adata.X.rechunk(((n_rows + 3) // 4, n_cols)).persist()  # one band per GPU
adata.X.compute_chunk_sizes()

# ── Scale + PCA ──────────────────────────────────────────────────────────────
adata.X = adata.X.astype("float64")                 # rounding accuracy for scale only
rsc.pp.scale(adata, zero_center=False, max_value=10)
rsc.pp.pca(adata, n_comps=50, random_state=RANDOM_SEED)
adata.obsm["X_pca"] = adata.obsm["X_pca"].persist().compute()

# ── Harmony (optional) ───────────────────────────────────────────────────────
rep = "X_pca"
if BATCH_KEY:
    adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype("category")
    rsc.pp.harmony_integrate(adata, key=BATCH_KEY,
                             basis="X_pca", adjusted_basis="X_pca_harmony")
    rep = "X_pca_harmony"

# ── Neighbors → UMAP → Leiden ────────────────────────────────────────────────
rsc.pp.neighbors(adata, n_neighbors=20, n_pcs=30, use_rep=rep,
                 algorithm="brute", random_state=RANDOM_SEED)
rsc.tl.umap(adata, min_dist=0.45, init_pos="spectral", n_components=2,
            random_state=RANDOM_SEED)
rsc.tl.leiden(adata, resolution=1.1, n_iterations=100, random_state=RANDOM_SEED)
print("clusters:", len(adata.obs["leiden"].cat.categories))

# ── Write (full genes + new embeddings; small subset → safe to gather) ───────
adata_full.obs  = adata.obs
adata_full.obsm = adata.obsm
adata_full.obsp = adata.obsp
adata_full.uns  = adata.uns
adata_full.X = adata_full.X.compute()
rsc.get.anndata_to_CPU(adata_full)
adata_full.write_h5ad(output_h5ad)
print("wrote", output_h5ad)
