"""Icechunk + Zarr + RAPIDS demo: versioned, branch-based GPU single-cell work
on one large dataset, where unchanged data is shared across snapshots.

  Step 0  create repo, ingest an existing anndata CSR zarr (raw counts) on main
  Step 1  User 1 branches, normalizes + log1p's all cells, stashes counts to
          raw/, overwrites X, commits, pushes to main
  Step 2  User 2 sees the changes, checks out one cell type, runs the clustering
          pipeline on that subset, commits the subset results into a scoped
          group (X is never rewritten), pushes to main
  Step 3  open main and show the final repo + history

GPU/CUDA only — runs on the server, not the laptop. Mirrors rapids_zarr_example.py.
"""
from __future__ import annotations

import dask
dask.config.set({"distributed.scheduler.worker-ttl": None})

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import cupy as cp

import numpy as np
import scipy.sparse as sp
import zarr
import anndata as ad
from anndata.io import read_elem, write_elem
from anndata.experimental import read_elem_lazy

from icechunk import Repository, local_filesystem_storage

import rapids_singlecell as rsc

# ── config ────────────────────────────────────────────────────────────────────
SOURCE_ZARR = "zarrs/13M_50M_pbmc_soundlife.zarr"   # existing anndata CSR zarr
REPO_PATH = "zarrs/demo_repo"                       # icechunk repo (created fresh)
CELL_TYPE_COL = "predicted_AIFI_L1"
CELL_TYPE_VAL = "b_cells"                           # User 2's subset
SPARSE_CHUNK_SIZE = 24_000
N_TOP_GENES = 2000
RANDOM_SEED = 5671


# ── helpers ─────────────────────────────────────────────────────────────────--
def _copy_csr(src_grp, dst_parent, name):
    """Stream-copy a csr group (data/indices/indptr 1D arrays) into dst_parent[name]."""
    dst = dst_parent.require_group(name)
    dst.attrs["encoding-type"] = src_grp.attrs["encoding-type"]
    dst.attrs["encoding-version"] = "0.1.0"
    dst.attrs["shape"] = list(src_grp.attrs["shape"])
    for child in ("data", "indices", "indptr"):
        s = src_grp[child]
        d = dst.require_array(child, shape=s.shape, dtype=s.dtype,
                              chunks=s.chunks, overwrite=True)
        d.attrs["encoding-type"] = "array"
        d.attrs["encoding-version"] = "0.2.0"
        step = s.chunks[0] * 16
        for i in range(0, s.shape[0], step):
            d[i:i + step] = s[i:i + step]


def _load_adata(store):
    """Lazy AnnData (dask X) backed by a zarr/icechunk store."""
    f = zarr.open_group(store=store, mode="r")
    X = f["X"]
    shape = tuple(X.attrs["shape"]) if "shape" in X.attrs else tuple(X.shape)
    X_dask = read_elem_lazy(X, (SPARSE_CHUNK_SIZE, shape[1]))
    if np.issubdtype(X_dask.dtype, np.integer):
        X_dask = X_dask.astype(np.float32)
    return ad.AnnData(X=X_dask, obs=read_elem(f["obs"]), var=read_elem(f["var"]))


def _host_csr(X):
    """Materialize a dask CSR block-by-block onto the host (peak RAM ≈ one CSR)."""
    parts = []
    for bi in range(X.numblocks[0]):
        ck = X.blocks[bi, 0].compute()
        if hasattr(ck, "get"):
            ck = ck.get()
        parts.append(ck)
    return sp.vstack(parts, format="csr")


# ── steps ───────────────────────────────────────────────────────────────────--
def ingest(repo):
    src = zarr.open_group(SOURCE_ZARR, mode="r")
    s = repo.writable_session("main")
    root = zarr.open_group(store=s.store, mode="a")
    root.attrs["encoding-type"] = "anndata"
    root.attrs["encoding-version"] = "0.1.0"
    _copy_csr(src["X"], root, "X")
    write_elem(root, "obs", read_elem(src["obs"]))
    write_elem(root, "var", read_elem(src["var"]))
    print("step0 ingest:", s.commit("ingest: raw counts"))


def user1_normalize(repo):
    repo.create_branch("user1-normalize", repo.lookup_branch("main"))
    s = repo.writable_session("user1-normalize")
    root = zarr.open_group(store=s.store, mode="a")

    raw = root.require_group("raw")
    raw.attrs["encoding-type"] = "raw"
    raw.attrs["encoding-version"] = "0.1.0"
    _copy_csr(root["X"], raw, "X")
    write_elem(raw, "var", read_elem(root["var"]))

    adata = _load_adata(s.store)
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.calculate_qc_metrics(adata)
    rsc.pp.normalize_total(adata)
    rsc.pp.log1p(adata)

    write_elem(root, "X", _host_csr(adata.X))
    snap = s.commit("user1: normalized + log1p; raw counts in raw/")
    repo.reset_branch("main", snap)
    print("step1 user1:", snap)


def user2_subset(repo):
    print("step2 history seen by user2:")
    for info in repo.ancestry(branch="main"):
        print("   ", info.id, info.message)

    repo.create_branch("user2-subset", repo.lookup_branch("main"))
    s = repo.writable_session("user2-subset")
    root = zarr.open_group(store=s.store, mode="a")

    adata = _load_adata(s.store)
    mask = (adata.obs[CELL_TYPE_COL] == CELL_TYPE_VAL).to_numpy()
    idx = np.where(mask)[0]
    adata = adata[mask].copy()

    rsc.get.anndata_to_GPU(adata)
    rsc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=N_TOP_GENES)
    hvg_mask = np.asarray(adata.var["highly_variable"])
    adata = adata[:, hvg_mask].copy()

    n_cols = adata.shape[1]
    rows_per_worker = (adata.shape[0] + 4 - 1) // 4
    adata.X = adata.X.rechunk((rows_per_worker, n_cols)).persist()
    adata.X.compute_chunk_sizes()

    adata.X = adata.X.astype("float64")
    rsc.pp.scale(adata, max_value=10, zero_center=False)
    rsc.pp.pca(adata, n_comps=30, svd_solver="covariance_eigh", random_state=RANDOM_SEED)
    adata.obsm["X_pca"] = adata.obsm["X_pca"].persist()
    adata.obsm["X_pca"].compute_chunk_sizes()
    adata.obsm["X_pca"] = adata.obsm["X_pca"].compute()
    rsc.pp.neighbors(adata, n_neighbors=20, n_pcs=30,
                     algorithm="mg_ivfflat", random_state=RANDOM_SEED)
    rsc.tl.umap(adata, min_dist=0.45, init_pos="spectral",
                n_components=2, random_state=RANDOM_SEED)
    rsc.tl.leiden(adata, resolution=1.1, n_iterations=100, random_state=RANDOM_SEED)
    rsc.get.anndata_to_CPU(adata)

    g = root.require_group("analyses").require_group(CELL_TYPE_VAL)
    write_elem(g, "cell_index", idx.astype("int64"))
    write_elem(g, "leiden", np.asarray(adata.obs["leiden"].astype(str)))
    write_elem(g, "X_umap", np.asarray(adata.obsm["X_umap"]))
    write_elem(g, "highly_variable", hvg_mask)

    snap = s.commit(f"user2: leiden/umap/hvg for {CELL_TYPE_VAL} subset ({idx.size} cells)")
    repo.reset_branch("main", snap)
    print("step2 user2:", snap)


def show(repo):
    root = zarr.open_group(store=repo.readonly_session(branch="main").store, mode="r")
    print("step3 final main root keys:", list(root.keys()))
    print("   X shape:", tuple(root["X"].attrs["shape"]))
    a = root["analyses"][CELL_TYPE_VAL]
    print("   analyses/%s:" % CELL_TYPE_VAL, {k: a[k].shape for k in a.keys()})
    print("   history:")
    for info in repo.ancestry(branch="main"):
        print("     ", info.id, info.message)


def main():
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
    rmm.reinitialize(managed_memory=True, pool_allocator=False, devices=[0, 1, 2, 3])
    cp.cuda.set_allocator(rmm_cupy_allocator)

    repo = Repository.create(local_filesystem_storage(REPO_PATH))
    ingest(repo)
    user1_normalize(repo)
    user2_subset(repo)
    show(repo)


if __name__ == "__main__":
    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        protocol="tcp",
        threads_per_worker=16,
        rmm_managed_memory=True,
        rmm_allocator_external_lib_list="cupy",
    )
    client = Client(cluster)
    try:
        main()
    finally:
        try:
            client.shutdown()
        except Exception:
            pass
