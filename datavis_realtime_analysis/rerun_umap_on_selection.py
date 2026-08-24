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

# ── Variables (env-overridable so the datavis backend can drive this per job) ──
import os

data_pth       = os.environ.get("RERUN_DATA", "/home/workspace/temp/expression.zarr")  
SELECTION_FILE = os.environ.get("RERUN_SELECTION", "./datascales-umap-poc-main/data/3M_subset_bcell_selection.json")
VIEW_STORE     = os.environ.get("RERUN_OUT", data_pth + "/umap_views/bcell_selection")  # view store to write (obsm/X_umap + obs, no X)
GPUS           = os.environ.get("RERUN_GPUS", "0") 
ROW_CHUNK_SIZE = 24_000
RANDOM_SEED    = 4242
BATCH_KEY      = []   # obs columns to harmony-integrate on; [] = skip harmony


def main():
    # Heavy/CUDA imports live inside main() so the worker-bootstrap re-import of this
    # module (see module docstring) stays trivial and doesn't init CUDA before the
    # worker has claimed its device.
    import json
    import time
    import numpy as np
    import zarr
    import anndata as ad
    import rapids_singlecell as rsc

    import rmm
    import cupy as cp
    from rmm.allocators.cupy import rmm_cupy_allocator

    # RERUN_DATA may be gs:// (zarr resolves it via gcsfs + ADC on the box);
    # crank read concurrency so the cold obs/X loads don't serialize on GCS latency
    zarr.config.set({"async.concurrency": 32, "threading.max_workers": 8})

    selection = json.load(open(SELECTION_FILE))         # {"barcodes": [...], ...}
    # if small selections fit one GPU eagerly — skip the dask-cuda cluster entirely
    eager = len(selection["barcodes"]) <= int(os.environ.get("RERUN_EAGER_MAX", "150000"))

    if not eager:
        print("stage: starting CUDA cluster", flush=True)
        import dask
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client

        dask.config.set({"distributed.scheduler.worker-ttl": None})
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=GPUS,
            protocol="tcp",
            threads_per_worker=16,
            rmm_managed_memory=True,
            rmm_allocator_external_lib_list="cupy",
            enable_cudf_spill=True,
        )
        client = Client(cluster)
        
        client.run(lambda: __import__("zarr").config.set(
            {"async.concurrency": 4, "threading.max_workers": 4}))

    # ── Managed memory (client process) ──────────────────────────────────────────
    rmm.reinitialize(managed_memory=True, pool_allocator=False)
    cp.cuda.set_allocator(rmm_cupy_allocator)

    # Wall clock: cluster/CUDA setup is done; time load → pipeline → write only.
    t0 = time.perf_counter()

    # ── Load ONLY the selected cells ─────────────────────────────────────────────
    print("stage: loading selected cells", flush=True)
    f = zarr.open(data_pth, mode="r")
    shape = f["X"].attrs["shape"]      # [n_obs, n_vars]
    obs = ad.io.read_elem(f["obs"])

    rows = obs.index.get_indexer(selection["barcodes"])
    assert (rows >= 0).all(), "some selected barcodes are not in this store"
    rows = np.unique(rows)                              # ascending, deduped → chunk-friendly

    if eager:
        X = ad.io.sparse_dataset(f["X"])[rows]      

    else: #dask
        from anndata.experimental import read_elem_lazy as read_dask
        X = read_dask(f["X"], (ROW_CHUNK_SIZE, shape[1]))
        X = X[rows]                                     
    
    raw_counts = np.issubdtype(X.dtype, np.integer)
    if raw_counts:
            X = X.astype(np.float32)

    adata = ad.AnnData(X=X, obs=obs.iloc[rows].copy(), var=ad.io.read_elem(f["var"]))
    print("Selected cells:", adata.shape, "(eager)" if eager else "(dask)")
    rsc.get.anndata_to_GPU(adata)

    # ── Preprocess (only if raw counts) ──────────────────────────────────────────
    if raw_counts:
        rsc.pp.normalize_total(adata)
        rsc.pp.log1p(adata)

    # ── Highly variable genes ────────────────────────────────────────────────────
    print("stage: highly variable genes", flush=True)
    rsc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()

    if not eager:
        n_rows, n_cols = adata.shape
        n_gpus = len(GPUS.split(","))
        adata.X = adata.X.rechunk(((n_rows + n_gpus - 1) // n_gpus, n_cols)).persist()  # one band per GPU
        adata.X.compute_chunk_sizes()

    # ── Scale + PCA ──────────────────────────────────────────────────────────────
    print("stage: scale + PCA", flush=True)
    adata.X = adata.X.astype("float64")                 # rounding accuracy for scale only
    rsc.pp.scale(adata, zero_center=False, max_value=10)
    rsc.pp.pca(adata, n_comps=50, random_state=RANDOM_SEED)
    if not eager:
        adata.obsm["X_pca"] = adata.obsm["X_pca"].persist().compute()

    # ── Harmony (optional) ───────────────────────────────────────────────────────
    rep = "X_pca"
    if BATCH_KEY:
        adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype("category")
        rsc.pp.harmony_integrate(adata, key=BATCH_KEY,
                                 basis="X_pca", adjusted_basis="X_pca_harmony")
        rep = "X_pca_harmony"

    # ── Neighbors → UMAP → Leiden ────────────────────────────────────────────────
    print("stage: neighbors + UMAP + leiden", flush=True)
    rsc.pp.neighbors(adata, n_neighbors=20, n_pcs=30, use_rep=rep,
                     algorithm="brute", random_state=RANDOM_SEED)
    rsc.tl.umap(adata, min_dist=0.45, init_pos="spectral", n_components=2,
                random_state=RANDOM_SEED)
    rsc.tl.leiden(adata, resolution=1.1, n_iterations=100, random_state=RANDOM_SEED)
    print("clusters:", len(adata.obs["leiden"].cat.categories))

    print("stage: writing view store", flush=True)
    umap = adata.obsm["X_umap"]
    if hasattr(umap, "get"):                            # cupy → host
        umap = umap.get()
    ad.settings.zarr_write_format = 3                   # zarrita (the viewer) reads v3 only
    view = ad.AnnData(obs=adata.obs.copy(), obsm={"X_umap": np.asarray(umap, dtype=np.float32)})
    view.write_zarr(VIEW_STORE)
    print(f"wrote view store: {VIEW_STORE}  ({view.n_obs} cells)")
    print(f"wall (load → pipeline → write): {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
