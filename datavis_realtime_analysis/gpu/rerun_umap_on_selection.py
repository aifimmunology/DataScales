"""Re-embed cells selected in the datavis app with the RAPIDS single-cell pipeline.

Importable: gpu/worker.py (the backend's warm pipeline process) calls warm_up() once,
then run() per job. Standalone: `pixi run python rerun_umap_on_selection.py` driven by the
RERUN_* env vars below. The result is a small self-contained "view" store — obsm/X_umap +
obs for the selected cells, NO X — exactly what the datavis viewer reads, by row position.

Two routes into neighbors → UMAP → leiden:
  * a store whose neighbor graph was built from a stored latent space (ATAC: PeakVI in
    obsm, named by uns/neighbors/params/use_rep) re-embeds from that representation and
    never touches X;
  * otherwise X (raw counts) goes through normalize → HVG → scale → PCA first. Selections
    ≤ RERUN_EAGER_MAX cells run eagerly on one GPU; larger ones stream through a per-run
    dask-cuda cluster.

NOTE: dask-cuda spawns its worker processes (CUDA can't survive fork, so spawn is
forced), and spawn re-imports the __main__ module in every child to bootstrap it.
Module level here must stay trivial and all cluster/pipeline code lives inside functions.
"""

import os
from contextlib import contextmanager, nullcontext

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
LABEL_ATTR     = "datavis-label"  # view root attribute the backend lists views by

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


@contextmanager
def cuda_cluster():
    """A per-run dask-cuda cluster for selections too large to embed eagerly."""
    import dask
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster

    print("stage: starting CUDA cluster", flush=True)
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
    print(f"host decode budget ≈ {n_gpus()} gpu × {THREADS_PER_WORKER} threads × "
          f"{ZARR_MAX_WORKERS} zarr workers", flush=True)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


def n_gpus() -> int:
    return len(GPUS.split(","))


def read_obs(root):
    """Index + categorical columns only: the string/numeric extras are the bulk of a full
    obs read (~19s vs ~2s on the 3M store) and the viewer never shows them."""
    import anndata as ad
    import pandas as pd

    grp = root["obs"]
    idx_name = grp.attrs["_index"]
    obs = pd.DataFrame(index=pd.Index(ad.io.read_elem(grp[idx_name]), name=idx_name))
    for c in grp.attrs.get("column-order", []):
        if grp[c].attrs.get("encoding-type") == "categorical":
            obs[c] = ad.io.read_elem(grp[c])
    return obs


def selected_rows(obs, barcodes):
    import numpy as np

    rows = obs.index.get_indexer(barcodes)
    assert (rows >= 0).all(), "some selected barcodes are not in this store"
    return np.unique(rows)                               # ascending, deduped → chunk-friendly


def latent_rep(root):
    """The obsm the store's own neighbor graph was built from, when the store records one."""
    import anndata as ad

    params = root.get("uns/neighbors/params")
    if params is None or "use_rep" not in params:
        return None
    rep = str(ad.io.read_elem(params["use_rep"]))
    return rep if rep in root["obsm"] else None


def load_cells(root, obs, rows, rep, eager):
    """The selected cells as an AnnData: the latent representation alone when there is one,
    else X (eager CSR slice, or a lazy dask array for the cluster path) + var."""
    import anndata as ad
    import numpy as np

    obs_sel = obs.iloc[rows].copy()
    obs_sel["root_row"] = rows.astype(np.int32)  # view row -> source-store row (gene highlighting in views)
    if rep is not None:
        return ad.AnnData(obs=obs_sel, obsm={rep: root["obsm"][rep].oindex[rows]})

    shape = root["X"].attrs["shape"]      # [n_obs, n_vars]
    if eager:
        X = ad.io.sparse_dataset(root["X"])[rows]
    else:
        from anndata.experimental import read_elem_lazy as read_dask
        X = read_dask(root["X"], (ROW_CHUNK_SIZE, shape[1]))[rows]
    return ad.AnnData(X=X, obs=obs_sel, var=ad.io.read_elem(root["var"]))


def reduce_counts(adata, eager):
    """normalize → HVG → scale → PCA (→ harmony) on X; returns the obsm key to embed from."""
    import numpy as np
    import rapids_singlecell as rsc

    raw_counts = np.issubdtype(adata.X.dtype, np.integer)
    if raw_counts:
        adata.X = adata.X.astype(np.float32)
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
        adata.X = adata.X.rechunk(((n_rows + n_gpus() - 1) // n_gpus(), n_cols)).persist()  # one band per GPU
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
    return adata, rep


def embed(adata, rep, eager):
    """neighbors → UMAP → leiden from `rep` (a stored latent space, else PCA computed here)."""
    import cupy as cp
    import rapids_singlecell as rsc

    if rep is None:
        adata, rep = reduce_counts(adata, eager)
        n_pcs = 30
    else:
        adata.obsm[rep] = cp.asarray(adata.obsm[rep], dtype=cp.float32)
        n_pcs = None

    print(f"stage: neighbors + UMAP + leiden ({rep})", flush=True)
    rsc.pp.neighbors(adata, n_neighbors=20, n_pcs=n_pcs, use_rep=rep,
                     algorithm="brute", random_state=RANDOM_SEED)
    rsc.tl.umap(adata, min_dist=0.45, init_pos="spectral", n_components=2,
                random_state=RANDOM_SEED)
    rsc.tl.leiden(adata, resolution=1.1, n_iterations=100, random_state=RANDOM_SEED)
    print("clusters:", len(adata.obs["leiden"].cat.categories))
    return adata


def umap_coords(adata):
    import numpy as np

    umap = adata.obsm["X_umap"]
    if hasattr(umap, "get"):                        # cupy → host
        umap = umap.get()
    return np.asarray(umap, dtype=np.float32)


def write_view(adata, out, label=None):
    import anndata as ad
    import zarr

    ad.settings.zarr_write_format = 3               # zarrita (the viewer) reads v3 only
    view = ad.AnnData(obs=adata.obs.copy(), obsm={"X_umap": umap_coords(adata)})
    view.write_zarr(out)                            # gs:// or local, via zarr's store resolution
    zarr.open_group(out, mode="r+").update_attributes({LABEL_ATTR: label or os.path.basename(out)})
    print(f"wrote view store: {out}  ({view.n_obs} cells)")


def run(store: str, selection_file: str, out: str, label: str | None = None) -> str:
    import json
    import time

    import zarr

    warm_up()
    _set_zarr_config(ZARR_CONCURRENCY, ZARR_MAX_WORKERS)

    selection = json.load(open(selection_file))         # {"barcodes": [...], ...}
    root = zarr.open(store, mode="r")
    rep = latent_rep(root)
    eager = rep is not None or len(selection["barcodes"]) <= EAGER_MAX

    with (nullcontext() if eager else cuda_cluster()):
        # Wall clock: cluster/CUDA setup is done; time load → pipeline → write only.
        t0 = time.perf_counter()
        print("stage: loading selected cells", flush=True)
        obs = read_obs(root)
        rows = selected_rows(obs, selection["barcodes"])
        adata = load_cells(root, obs, rows, rep, eager)
        print("Selected cells:", adata.shape, f"rep={rep}" if rep else "(eager)" if eager else "(dask)")
        adata = embed(adata, rep, eager)
        print("stage: writing view store", flush=True)
        write_view(adata, out, label)
        print(f"wall (load → pipeline → write): {time.perf_counter() - t0:.2f}s")
    return out


def main():
    run(data_pth, SELECTION_FILE, VIEW_STORE)


if __name__ == "__main__":
    main()
