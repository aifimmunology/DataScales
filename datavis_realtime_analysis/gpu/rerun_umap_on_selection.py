"""Re-run the RAPIDS single-cell UMAP pipeline on cells selected in the datavis app.

Importable: gpu/worker.py (the backend's warm pipeline process) calls warm_up()
once, then run() per job. Standalone: `pixi run python rerun_umap_on_selection.py`
driven by the RERUN_* env vars below. The result is a small self-contained "view"
store — obsm/X_umap + obs for the selected cells, NO X — exactly what the datavis
viewer reads, by row position. Selections ≤ RERUN_EAGER_MAX cells run eagerly on
one GPU; larger ones stream through a per-run dask-cuda cluster.

NOTE: dask-cuda spawns its worker processes (CUDA can't survive fork, so spawn is
forced), and spawn re-imports the __main__ module in every child to bootstrap it.
Module level here must stay trivial and all cluster/pipeline code lives inside
functions.
"""

import os

# Pin host BLAS/OpenMP threads (env vars inherit into dask-cuda worker children) so
# worker threads don't each spawn a full BLAS pool — same as rapids-benchmark.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

data_pth       = os.environ.get("RERUN_DATA", "/home/workspace/temp/expression.zarr")
SELECTION_FILE = os.environ.get("RERUN_SELECTION", "./data/3M_subset_bcell_selection.json")
VIEW_STORE     = os.environ.get("RERUN_OUT", data_pth + "/umap_views/bcell_selection")
GPUS           = os.environ.get("RERUN_GPUS", "0")
THREADS_PER_WORKER = int(os.environ.get("RERUN_THREADS_PER_WORKER", "12"))  # dask-cuda worker threadpool
ZARR_CONCURRENCY   = int(os.environ.get("RERUN_ZARR_CONCURRENCY", "64"))    # zarr fetch-dispatch semaphore
ZARR_MAX_WORKERS   = int(os.environ.get("RERUN_ZARR_MAX_WORKERS", "12"))    # zarr decode threadpool
EAGER_MAX      = int(os.environ.get("RERUN_EAGER_MAX", "500000"))
ROW_CHUNK_SIZE = 24_000
RANDOM_SEED    = 4242
BATCH_KEY      = []   # obs columns to harmony-integrate on; [] = skip harmony

_gpu_ready = False


def _set_zarr_config(concurrency: int, max_workers: int) -> None:
    """Module-level so it's picklable for client.run: zarr.config is runtime state,
    per process — setting it on the client does NOT reach the dask-cuda workers,
    and the lazy chunk reads happen ON the workers (rapids-benchmark pattern)."""
    import zarr
    zarr.config.set({"async.concurrency": concurrency, "threading.max_workers": max_workers})


def warm_up() -> None:
    """Heavy imports + RMM/cupy allocator, once per process."""
    global _gpu_ready
    import anndata, pandas, zarr  # noqa: F401
    import cupy as cp
    import rapids_singlecell  # noqa: F401
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    if not _gpu_ready:
        rmm.reinitialize(managed_memory=True, pool_allocator=False)
        cp.cuda.set_allocator(rmm_cupy_allocator)
        _gpu_ready = True


def run(store: str, selection_file: str, out: str) -> str:
    import json
    import time

    import anndata as ad
    import numpy as np
    import pandas as pd
    import rapids_singlecell as rsc
    import zarr

    warm_up()
    _set_zarr_config(ZARR_CONCURRENCY, ZARR_MAX_WORKERS)

    selection = json.load(open(selection_file))         # {"barcodes": [...], ...}
    eager = len(selection["barcodes"]) <= EAGER_MAX

    client = cluster = None
    if not eager:
        print("stage: starting CUDA cluster", flush=True)
        import dask
        from dask.distributed import Client
        from dask_cuda import LocalCUDACluster

        dask.config.set({"distributed.scheduler.worker-ttl": None})
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=GPUS,
            protocol="tcp",
            threads_per_worker=THREADS_PER_WORKER,
            rmm_managed_memory=True,
            rmm_allocator_external_lib_list="cupy",
            enable_cudf_spill=True,
        )
        client = Client(cluster)
        client.run(_set_zarr_config, ZARR_CONCURRENCY, ZARR_MAX_WORKERS)
        n_gpus = len(GPUS.split(","))
        print(f"host decode budget ≈ {n_gpus} gpu × {THREADS_PER_WORKER} threads × "
              f"{ZARR_MAX_WORKERS} zarr workers", flush=True)

    try:
        # Wall clock: cluster/CUDA setup is done; time load → pipeline → write only.
        t0 = time.perf_counter()

        print("stage: loading selected cells", flush=True)
        f = zarr.open(store, mode="r")
        shape = f["X"].attrs["shape"]      # [n_obs, n_vars]
        # index + categorical columns only: the string/numeric extras are the bulk of
        # a full obs read (~19s vs ~2s on the 3M store) and the viewer never shows them
        obs_grp = f["obs"]
        idx_name = obs_grp.attrs["_index"]
        obs = pd.DataFrame(index=pd.Index(ad.io.read_elem(obs_grp[idx_name]), name=idx_name))
        for c in obs_grp.attrs.get("column-order", []):
            if obs_grp[c].attrs.get("encoding-type") == "categorical":
                obs[c] = ad.io.read_elem(obs_grp[c])

        rows = obs.index.get_indexer(selection["barcodes"])
        assert (rows >= 0).all(), "some selected barcodes are not in this store"
        rows = np.unique(rows)                          # ascending, deduped → chunk-friendly

        if eager:
            X = ad.io.sparse_dataset(f["X"])[rows]
        else:
            from anndata.experimental import read_elem_lazy as read_dask
            X = read_dask(f["X"], (ROW_CHUNK_SIZE, shape[1]))
            X = X[rows]

        raw_counts = np.issubdtype(X.dtype, np.integer)
        if raw_counts:
            X = X.astype(np.float32)

        obs_sel = obs.iloc[rows].copy()
        obs_sel["root_row"] = rows.astype(np.int32)  # view row -> source-store row (gene highlighting in views)
        adata = ad.AnnData(X=X, obs=obs_sel, var=ad.io.read_elem(f["var"]))
        print("Selected cells:", adata.shape, "(eager)" if eager else "(dask)")
        rsc.get.anndata_to_GPU(adata)

        if not eager:
            # one full pass over X now; without it normalize (median), HVG, and the
            # post-HVG rechunk each re-read X from GCS (measured ~2.5x total wall)
            from dask.distributed import futures_of, wait
            adata.X = adata.X.persist()
            wait(futures_of(adata.X))

        if raw_counts:
            rsc.pp.normalize_total(adata)
            rsc.pp.log1p(adata)

        print("stage: highly variable genes", flush=True)
        rsc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
        adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()

        if not eager:
            n_rows, n_cols = adata.shape
            n_gpus = len(GPUS.split(","))
            adata.X = adata.X.rechunk(((n_rows + n_gpus - 1) // n_gpus, n_cols)).persist()  # one band per GPU
            adata.X.compute_chunk_sizes()

        print("stage: scale + PCA", flush=True)
        adata.X = adata.X.astype("float64")             # rounding accuracy for scale only
        rsc.pp.scale(adata, zero_center=False, max_value=10)
        rsc.pp.pca(adata, n_comps=50, random_state=RANDOM_SEED)
        if not eager:
            adata.obsm["X_pca"] = adata.obsm["X_pca"].persist().compute()

        rep = "X_pca"
        if BATCH_KEY:
            adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype("category")
            rsc.pp.harmony_integrate(adata, key=BATCH_KEY,
                                     basis="X_pca", adjusted_basis="X_pca_harmony")
            rep = "X_pca_harmony"

        print("stage: neighbors + UMAP + leiden", flush=True)
        rsc.pp.neighbors(adata, n_neighbors=20, n_pcs=30, use_rep=rep,
                         algorithm="brute", random_state=RANDOM_SEED)
        rsc.tl.umap(adata, min_dist=0.45, init_pos="spectral", n_components=2,
                    random_state=RANDOM_SEED)
        rsc.tl.leiden(adata, resolution=1.1, n_iterations=100, random_state=RANDOM_SEED)
        print("clusters:", len(adata.obs["leiden"].cat.categories))

        print("stage: writing view store", flush=True)
        umap = adata.obsm["X_umap"]
        if hasattr(umap, "get"):                        # cupy → host
            umap = umap.get()
        ad.settings.zarr_write_format = 3               # zarrita (the viewer) reads v3 only
        view = ad.AnnData(obs=adata.obs.copy(),
                          obsm={"X_umap": np.asarray(umap, dtype=np.float32)})
        view.write_zarr(out)                            # gs:// or local, via zarr's store resolution
        print(f"wrote view store: {out}  ({view.n_obs} cells)")
        print(f"wall (load → pipeline → write): {time.perf_counter() - t0:.2f}s")
        return out
    finally:
        if client is not None:
            client.close()
            cluster.close()


def main():
    run(data_pth, SELECTION_FILE, VIEW_STORE)


if __name__ == "__main__":
    main()
