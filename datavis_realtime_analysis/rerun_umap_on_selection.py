"""Re-run the RAPIDS single-cell UMAP pipeline on cells selected in the datavis app.

Script form of rapids_user_notebook/rapids_GPU_sc_analysis.ipynb, with two changes:
the AnnData is built from only the barcodes in the datavis selection file, and the
result is written as a small self-contained "view" store (VIEW_STORE) rather than an
.h5ad. The view holds just obsm/X_umap (the new coords) + obs (AIFI_L* labels and
barcodes) for the selected cells and NO X — exactly what the datavis viewer reads, by
row position. Open it in the app with DATA_DIR=<VIEW_STORE>.
Run on the HISE/GPU node:  pixi run python rerun_umap_on_selection.py

NOTE: dask-cuda spawns its worker processes (CUDA can't survive fork, so spawn is
forced), and spawn re-imports THIS module in every child to bootstrap it. So all
cluster/client/pipeline code MUST stay under the `if __name__ == "__main__"` guard
below — otherwise each spawned worker rebuilds the cluster during import and
multiprocessing aborts with "an attempt has been made to start a new process before
the current process has finished its bootstrapping phase".
"""

# ── Variables ────────────────────────────────────────────────────────────────
data_pth       = "/home/workspace/temp/expression.zarr"  # CSR zarr store (raw counts)
SELECTION_FILE = "./datascales-umap-poc-main/data/3M_subset_bcell_selection.json"  # barcodes from the datavis app
VIEW_STORE     = data_pth + "/umap_views/bcell_selection"  # mini view store to write (obsm/X_umap + obs, no X); point the datavis app's DATA_DIR here
ROW_CHUNK_SIZE = 24_000
RANDOM_SEED    = 4242
BATCH_KEY      = []   # obs columns to harmony-integrate on; [] = skip harmony


def main():
    # Heavy/CUDA imports live inside main() so the worker-bootstrap re-import of this
    # module (see module docstring) stays trivial and doesn't init CUDA before the
    # worker has claimed its device.
    import dask
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client

    import json
    import numpy as np
    import zarr
    import anndata as ad
    from anndata.experimental import read_elem_lazy as read_dask
    import rapids_singlecell as rsc

    import rmm
    import cupy as cp
    from rmm.allocators.cupy import rmm_cupy_allocator

    # ── CUDA dask cluster ────────────────────────────────────────────────────────
    dask.config.set({"distributed.scheduler.worker-ttl": None})
    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        protocol="tcp",
        threads_per_worker=16,
        rmm_managed_memory=True,
        rmm_allocator_external_lib_list="cupy",
        enable_cudf_spill=True,
    )
    client = Client(cluster)

    # zarr read-tuning is per worker process — set it ON the workers, not the client.
    client.run(lambda: __import__("zarr").config.set(
        {"async.concurrency": 4, "threading.max_workers": 4}))

    # ── Managed memory (client process) ──────────────────────────────────────────
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

    umap = adata.obsm["X_umap"]
    if hasattr(umap, "get"):                            # cupy → host
        umap = umap.get()
    ad.settings.zarr_write_format = 3                   # zarrita (the viewer) reads v3 only
    view = ad.AnnData(obs=adata.obs.copy(), obsm={"X_umap": np.asarray(umap, dtype=np.float32)})
    view.write_zarr(VIEW_STORE)
    print(f"wrote view store: {VIEW_STORE}  ({view.n_obs} cells)")


if __name__ == "__main__":
    main()
