"""Re-embed cells selected in the datavis app with the RAPIDS single-cell pipeline.

gpu/worker.py calls warm_up() once, then run() per job. The result is a small "view"
store — obsm/X_umap + obs for the selected cells, no X — read by the viewer by row position.

Two routes into neighbors → UMAP → leiden, chosen by RunParams.method (see resolve_rep):
  * a stored latent space (recorded in uns/neighbors/params/use_rep) is embedded directly;
  * otherwise X (raw counts) goes through normalize → HVG → scale → PCA first, eagerly on
    one GPU up to RERUN_EAGER_MAX cells, through a per-run dask-cuda cluster beyond.

dask-cuda spawns its workers, and spawn re-imports __main__ in every child: module level
must stay trivial, with all cluster/pipeline code inside functions.
"""

import os
from contextlib import contextmanager, nullcontext

from run_params import RunParams
from store_meta import LABEL_ATTR, PARAMS_ATTR, latent_rep

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

GPUS = os.environ.get("RERUN_GPUS", "0")
THREADS_PER_WORKER = int(os.environ.get("RERUN_THREADS_PER_WORKER", "12"))
ZARR_CONCURRENCY = int(os.environ.get("RERUN_ZARR_CONCURRENCY", "64"))
ZARR_MAX_WORKERS = int(os.environ.get("RERUN_ZARR_MAX_WORKERS", "12"))
EAGER_MAX = int(os.environ.get("RERUN_EAGER_MAX", "500000"))
ROW_CHUNK_SIZE = 24_000
RANDOM_SEED = 4242
N_PCS = 30

_gpu_ready = False


def _set_zarr_config(concurrency: int, max_workers: int) -> None:
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
def _cuda_cluster():
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
    try:
        yield client
    finally:
        client.close()
        cluster.close()


def _n_gpus() -> int:
    return len(GPUS.split(","))


def _read_obs(root):
    import anndata as ad
    import pandas as pd

    grp = root["obs"]
    idx_name = grp.attrs["_index"]
    obs = pd.DataFrame(index=pd.Index(ad.io.read_elem(grp[idx_name]), name=idx_name))
    for c in grp.attrs.get("column-order", []):
        if grp[c].attrs.get("encoding-type") == "categorical":
            obs[c] = ad.io.read_elem(grp[c])
    return obs


def _selected_rows(obs, barcodes):
    import numpy as np

    rows = obs.index.get_indexer(barcodes)
    if (rows < 0).any():
        raise ValueError(f"{int((rows < 0).sum())} selected barcodes are not in this store")
    return np.unique(rows)


def resolve_rep(root, method: str = "auto"):
    """The obsm key to embed from, or None for normalize → PCA on X."""
    if method == "pca":
        return None
    rep = latent_rep(root)
    if method == "latent" and rep is None:
        raise ValueError("store records no latent space (uns/neighbors/params/use_rep)")
    return rep


def _load_cells(root, obs, rows, rep, eager):
    import anndata as ad
    import numpy as np

    obs_sel = obs.iloc[rows].copy()
    obs_sel["root_row"] = rows.astype(np.int32)
    if rep is not None:
        return ad.AnnData(obs=obs_sel, obsm={rep: root["obsm"][rep].oindex[rows]})
    if eager:
        X = ad.io.sparse_dataset(root["X"])[rows]
    else:
        from anndata.experimental import read_elem_lazy

        X = read_elem_lazy(root["X"], (ROW_CHUNK_SIZE, root["X"].attrs["shape"][1]))[rows]
    return ad.AnnData(X=X, obs=obs_sel, var=ad.io.read_elem(root["var"]))


def _reduce_counts(adata, eager, batch_key=None):
    import numpy as np
    import rapids_singlecell as rsc

    raw_counts = np.issubdtype(adata.X.dtype, np.integer)
    if raw_counts:
        adata.X = adata.X.astype(np.float32)
    rsc.get.anndata_to_GPU(adata)
    if not eager:
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
        adata.X = adata.X.rechunk(((n_rows + _n_gpus() - 1) // _n_gpus(), n_cols)).persist()
        adata.X.compute_chunk_sizes()

    print("stage: scale + PCA", flush=True)
    adata.X = adata.X.astype("float64")
    rsc.pp.scale(adata, zero_center=False, max_value=10)
    rsc.pp.pca(adata, n_comps=50, random_state=RANDOM_SEED)
    if not eager:
        adata.obsm["X_pca"] = adata.obsm["X_pca"].persist().compute()

    if not batch_key:
        return adata, "X_pca"
    print(f"stage: harmony ({batch_key})", flush=True)
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    rsc.pp.harmony_integrate(adata, key=batch_key, basis="X_pca", adjusted_basis="X_pca_harmony")
    return adata, "X_pca_harmony"


def _embed(adata, rep, eager, p: RunParams):
    import cupy as cp
    import rapids_singlecell as rsc

    if rep is None:
        adata, rep = _reduce_counts(adata, eager, p.batch_key)
        n_pcs = N_PCS
    else:
        adata.obsm[rep] = cp.asarray(adata.obsm[rep], dtype=cp.float32)
        n_pcs = None

    print(f"stage: neighbors + UMAP + leiden ({rep})", flush=True)
    rsc.pp.neighbors(adata, n_neighbors=p.n_neighbors, n_pcs=n_pcs, use_rep=rep,
                     algorithm="brute", random_state=RANDOM_SEED)
    rsc.tl.umap(adata, min_dist=p.min_dist, init_pos="spectral", n_components=2,
                random_state=RANDOM_SEED)
    rsc.tl.leiden(adata, resolution=p.resolution, n_iterations=100, random_state=RANDOM_SEED)
    adata.uns["rep"] = rep
    return adata


def _umap_coords(adata):
    import numpy as np

    umap = adata.obsm["X_umap"]
    if hasattr(umap, "get"):
        umap = umap.get()
    return np.asarray(umap, dtype=np.float32)


def _write_view(adata, out, label, params: RunParams):
    import anndata as ad
    import zarr

    ad.settings.zarr_write_format = 3
    ad.AnnData(obs=adata.obs.copy(), obsm={"X_umap": _umap_coords(adata)}).write_zarr(out)
    zarr.open_group(out, mode="r+").update_attributes({
        LABEL_ATTR: label or os.path.basename(out),
        PARAMS_ATTR: {**params.to_dict(), "rep": adata.uns.get("rep")},
    })


def run(store: str, selection_file: str, out: str, label: str | None = None, params: dict | None = None) -> str:
    """Embed the barcodes in selection_file ({"barcodes": [...]}) from `store` into the view at `out`."""
    import json

    import zarr

    warm_up()
    _set_zarr_config(ZARR_CONCURRENCY, ZARR_MAX_WORKERS)
    with open(selection_file) as f:
        barcodes = json.load(f)["barcodes"]
    root = zarr.open(store, mode="r")
    p = RunParams.from_dict(params)
    rep = resolve_rep(root, p.method)
    eager = rep is not None or len(barcodes) <= EAGER_MAX

    with nullcontext() if eager else _cuda_cluster():
        print("stage: loading selected cells", flush=True)
        obs = _read_obs(root)
        adata = _load_cells(root, obs, _selected_rows(obs, barcodes), rep, eager)
        adata = _embed(adata, rep, eager, p)
        print("stage: writing view store", flush=True)
        _write_view(adata, out, label, p)
    return out
