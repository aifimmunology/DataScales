"""Icechunk + Zarr + RAPIDS demo: versioned, branch-based GPU single-cell work
on one large dataset, where unchanged data is shared across snapshots.

  Step 0  create repo, ingest an existing anndata CSR zarr (raw counts) on main
  Step 1  User 1 branches, normalizes + log1p's all cells (multi-GPU dask), and
          writes the result to a NEW layer 'layers/norm' — raw counts stay in X
          untouched (zero copy). The normalized matrix is streamed to disk
          block-by-block (peak host RAM ~= one block). Commits, pushes to main.
  Step 2  User 2 checks out one cell type from layers/norm, runs the clustering
          pipeline on that (bounded) subset in-memory on a single GPU, and commits
          the results into a scoped group (X/layers are never rewritten).
  Step 3  open main and show the final repo + history.

Reruns are incremental: an existing REPO_PATH is reused (not wiped), and any step
whose output is already committed on main is skipped — layers/norm present => skip
Step 1, analyses/<cell_type> present => skip Step 2. Delete REPO_PATH to force a
full fresh run.

GPU/CUDA only — runs on the server, not the laptop.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import warnings

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

# ── config ──────────────────────────────────────────────────────────────────--
SOURCE_ZARR = "/home/workspace/zarrs/13M_50M_pbmc_soundlife.zarr"   # anndata CSR zarr
REPO_PATH = "/home/workspace/zarrs/demo_repo"                       # icechunk repo
CELL_TYPE_COL = "predicted_AIFI_L1"
CELL_TYPE_VAL = "b_cells"
SPARSE_CHUNK_SIZE = 5_000
N_TOP_GENES = 2000
RANDOM_SEED = 5671

# zarr v3 chunk-level parallelism for the CSR byte-copy (no host dask needed):
#   async.concurrency     -> max chunk get/decode/encode/set ops in flight
#   threading.max_workers -> Blosc/Zstd codec thread-pool size
# Blosc has its own internal threads; pinning it to 1 keeps all parallelism in the
# zarr layer instead of multiplying to ~workers*N threads (oversubscription).
ZARR_ASYNC_CONCURRENCY = 60
ZARR_CODEC_WORKERS = 60
BLOSC_NTHREADS = 1


def _log(msg):
    """Timestamped, flushed progress line (block-buffered stdout looks 'hung')."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True, file=sys.stderr)


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
        n = s.shape[0]
        _log(f"    _copy_csr/{name}/{child}: {n:,} elems ({s.dtype})")
        for i in range(0, n, step):
            d[i:i + step] = s[i:i + step]


def _load_adata(store, x_key="X"):
    """Lazy AnnData (dask X) backed by a zarr/icechunk store; x_key may be nested."""
    f = zarr.open_group(store=store, mode="r")
    node = f
    for part in x_key.split("/"):
        node = node[part]
    shape = tuple(node.attrs["shape"]) if "shape" in node.attrs else tuple(node.shape)
    X_dask = read_elem_lazy(node, (SPARSE_CHUNK_SIZE, shape[1]))
    if np.issubdtype(X_dask.dtype, np.integer):
        X_dask = X_dask.astype(np.float32)
    return ad.AnnData(X=X_dask, obs=read_elem(f["obs"]), var=read_elem(f["var"]))


def _canon_csr(m):
    """Clean, C-contiguous, int32-indexed, float32 CSR.

    This store has total nnz > 2**31, so anndata wrote int64 indptr/indices, and a
    dask boolean-mask gather can leave blocks non-contiguous. Both make cupyx CSR
    blocks miss every `mean_var_minor` nanobind overload (it binds only contiguous
    {int32,int64}+{float32,float64}). For a single cell-type subset every value fits
    int32, so we rebuild one clean in-memory CSR before moving to the GPU.
    """
    m = m if sp.isspmatrix_csr(m) else sp.csr_matrix(m)
    return sp.csr_matrix(
        (np.ascontiguousarray(m.data, dtype=np.float32),
         np.ascontiguousarray(m.indices, dtype=np.int32),
         np.ascontiguousarray(m.indptr, dtype=np.int32)),
        shape=m.shape,
    )


def _main_has(repo, *path):
    """True iff the nested group/array `path` exists on the main branch."""
    try:
        node = zarr.open_group(store=repo.readonly_session(branch="main").store, mode="r")
    except Exception:
        return False
    for part in path:
        if part not in node:
            return False
        node = node[part]
    return True


def _fresh_branch(repo, name, base="main"):
    """Create `name` off `base`, resetting it if a prior (crashed) run left it behind."""
    base_snap = repo.lookup_branch(base)
    if name in repo.list_branches():
        repo.reset_branch(name, base_snap)
    else:
        repo.create_branch(name, base_snap)


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
    """Normalize + log1p ALL cells (multi-GPU dask, streamed) into layers/norm."""
    _log("step1 user1_normalize: START")
    _fresh_branch(repo, "user1-normalize")
    s = repo.writable_session("user1-normalize")
    root = zarr.open_group(store=s.store, mode="a")

    # X (raw counts on main) is left untouched; the normalized result goes to a NEW
    # layer, so the raw counts are never copied. Read through a read-only session:
    # its store is picklable, so dask-distributed can ship the lazy-X graph to the
    # GPU workers (a writable session refuses to pickle uncommitted changeset state).
    adata = _load_adata(repo.readonly_session(branch="user1-normalize").store)
    _log(f"  X.shape={adata.X.shape} chunks={adata.X.chunksize}; -> GPU + normalize")
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.calculate_qc_metrics(adata)
    rsc.pp.normalize_total(adata)
    rsc.pp.log1p(adata)

    # anndata's dask-sparse writer streams block-by-block (device->host one block at
    # a time), so peak host RAM ~= one block, never the full CSR.
    layers = root.require_group("layers")
    layers.attrs["encoding-type"] = "dict"
    layers.attrs["encoding-version"] = "0.1.0"
    _log("  streaming normalized X -> layers/norm")
    write_elem(layers, "norm", adata.X)
    snap = s.commit("user1: normalized+log1p in layers/norm; raw counts kept in X")
    repo.reset_branch("main", snap)
    print("step1 user1:", snap, flush=True)


def user2_subset(repo):
    """Cluster one cell type from layers/norm in-memory on a single GPU."""
    print("step2 history seen by user2:")
    for info in repo.ancestry(branch="main"):
        print("   ", info.id, info.message)

    _fresh_branch(repo, "user2-subset")
    s = repo.writable_session("user2-subset")
    root = zarr.open_group(store=s.store, mode="a")

    # Source X from layers/norm (X is still raw counts). Materialize ONLY this cell
    # type to host as one clean CSR — a bounded working set — then run the whole
    # subset pipeline in-memory on the GPU. This sidesteps the dask-sparse HVG path
    # whose per-block arrays miss the mean_var_minor kernel overloads.
    adata = _load_adata(repo.readonly_session(branch="user2-subset").store,
                        x_key="layers/norm")
    mask = (adata.obs[CELL_TYPE_COL] == CELL_TYPE_VAL).to_numpy()
    idx = np.where(mask)[0]
    _log(f"  {CELL_TYPE_VAL}: {idx.size:,} cells; materializing subset to host")
    sub = adata[mask]
    adata = ad.AnnData(X=_canon_csr(sub.X.compute()),
                       obs=sub.obs.copy(), var=sub.var.copy())

    rsc.get.anndata_to_GPU(adata)
    rsc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=N_TOP_GENES)
    hvg_mask = np.asarray(adata.var["highly_variable"])
    adata = adata[:, hvg_mask].copy()

    adata.X = adata.X.astype("float64")  # covariance_eigh PCA is happier in float64
    rsc.pp.scale(adata, max_value=10, zero_center=False)
    rsc.pp.pca(adata, n_comps=30, svd_solver="covariance_eigh", random_state=RANDOM_SEED)
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
    print("   X shape (raw counts):", tuple(root["X"].attrs["shape"]))
    if "layers" in root and "norm" in root["layers"]:
        print("   layers/norm shape (normalized):",
              tuple(root["layers"]["norm"].attrs["shape"]))
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

    # zarr chunk-level parallelism + codec threads (provenance for the byte-copy path).
    zarr.config.set({"async.concurrency": ZARR_ASYNC_CONCURRENCY,
                     "threading.max_workers": ZARR_CODEC_WORKERS})
    from numcodecs import blosc
    blosc.set_nthreads(BLOSC_NTHREADS)
    # anndata's writer floods stderr with one autosharding notice per obs/var column;
    # silence only that exact message (layout is unchanged — we don't autoshard).
    warnings.filterwarnings("ignore", message=".*autosharding will be the default.*",
                            category=UserWarning)
    _log(f"parallelism: zarr async.concurrency={ZARR_ASYNC_CONCURRENCY} "
         f"threading.max_workers={ZARR_CODEC_WORKERS} blosc.nthreads={BLOSC_NTHREADS} "
         f"| zarr {zarr.__version__}")

    # Reuse an existing repo (lets a rerun skip already-committed steps), else create
    # fresh + ingest. icechunk only creates into a clean prefix, so wipe in-process.
    abs_repo = os.path.abspath(REPO_PATH)
    storage = local_filesystem_storage(abs_repo)
    if Repository.exists(storage):
        print("repo exists, reusing:", abs_repo)
        repo = Repository.open(storage)
    else:
        print("creating fresh repo:", abs_repo)
        shutil.rmtree(abs_repo, ignore_errors=True)
        repo = Repository.create(storage)
        ingest(repo)

    if _main_has(repo, "layers", "norm"):
        print("layers/norm already on main; skipping user1_normalize")
    else:
        user1_normalize(repo)

    if _main_has(repo, "analyses", CELL_TYPE_VAL):
        print(f"analyses/{CELL_TYPE_VAL} already on main; skipping user2_subset")
    else:
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
