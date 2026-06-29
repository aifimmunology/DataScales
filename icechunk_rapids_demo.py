"""Icechunk + Zarr + RAPIDS demo: versioned, branch-based GPU single-cell work
on one large dataset, where unchanged data is shared across snapshots.

  Step 0  create repo, ingest an existing anndata CSR zarr (raw counts) on main
  Step 1  User 1 branches, normalizes + log1p's all cells, and writes the result
          to a NEW layer 'layers/norm' — raw counts stay in X untouched (zero
          copy). The normalized matrix is streamed to disk block-by-block
          (peak host RAM ~= one block, never the whole CSR). Commits, pushes to main
  Step 2  User 2 sees the changes, checks out one cell type, runs the clustering
          pipeline on that subset (reading layers/norm), commits the subset
          results into a scoped group (X/layers are never rewritten), pushes to main
  Step 3  open main and show the final repo + history

GPU/CUDA only — runs on the server, not the laptop. Mirrors rapids_zarr_example.py.
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

# ── parallelism config (CPU copy/codec path) ───────────────────────────────────
# These tune the zarr v3 chunk-level parallelism used by _copy_csr's slice
# read/writes (no host dask needed — it's a pure chunk-aligned byte copy):
#   async.concurrency   -> max chunk get/decode/encode/set ops in flight per op
#   threading.max_workers -> size of the codec (Blosc/Zstd) thread pool
ZARR_ASYNC_CONCURRENCY = 60
ZARR_CODEC_WORKERS = 60
# Blosc has its OWN internal threads; with a 60-wide codec pool above, leaving
# Blosc multi-threaded would multiply to ~60*N threads (silent killer #1:
# oversubscription). Pin it to 1 so all parallelism lives in the zarr layer.
BLOSC_NTHREADS = 1


# ── helpers ─────────────────────────────────────────────────────────────────--
def _log(msg):
    """Timestamped, flushed progress line (block-buffered stdout looks 'hung')."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True, file=sys.stderr)


def _configure_parallelism():
    """Set zarr chunk-level concurrency + codec threads, and record them (provenance)."""
    zarr.config.set({
        "async.concurrency": ZARR_ASYNC_CONCURRENCY,
        "threading.max_workers": ZARR_CODEC_WORKERS,
    })
    from numcodecs import blosc
    blosc.set_nthreads(BLOSC_NTHREADS)
    # anndata's writer warns once per element that zarr v3 autosharding will become
    # the default. It's informational (we don't autoshard here, so layout is
    # unchanged) and floods stderr with one line per obs/var column written. Silence
    # only that exact message — never blanket-mute warnings.
    warnings.filterwarnings("ignore", message=".*autosharding will be the default.*",
                            category=UserWarning)
    _log(f"parallelism: zarr async.concurrency={ZARR_ASYNC_CONCURRENCY} "
         f"threading.max_workers={ZARR_CODEC_WORKERS} blosc.nthreads={BLOSC_NTHREADS} "
         f"| zarr {zarr.__version__}")


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
        t0 = time.perf_counter()
        _log(f"    _copy_csr/{name}/{child}: {n:,} elems ({s.dtype}), "
             f"step={step:,}, src_chunks={s.chunks}")
        for i in range(0, n, step):
            d[i:i + step] = s[i:i + step]
            _log(f"      {name}/{child}: {min(i + step, n):,}/{n:,} "
                 f"({100 * min(i + step, n) / n:.0f}%)")
        _log(f"    _copy_csr/{name}/{child}: done in {time.perf_counter() - t0:.1f}s")


def _load_adata(store, x_key="X"):
    """Lazy AnnData (dask X) backed by a zarr/icechunk store.

    x_key may be a nested path (e.g. "layers/norm") — we walk it so X can be
    sourced from a layer instead of the root X.
    """
    f = zarr.open_group(store=store, mode="r")
    node = f
    for part in x_key.split("/"):
        node = node[part]
    X = node
    shape = tuple(X.attrs["shape"]) if "shape" in X.attrs else tuple(X.shape)
    X_dask = read_elem_lazy(X, (SPARSE_CHUNK_SIZE, shape[1]))
    if np.issubdtype(X_dask.dtype, np.integer):
        X_dask = X_dask.astype(np.float32)
    return ad.AnnData(X=X_dask, obs=read_elem(f["obs"]), var=read_elem(f["var"]))


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
    _log("step1 user1_normalize: START")
    repo.create_branch("user1-normalize", repo.lookup_branch("main"))
    s = repo.writable_session("user1-normalize")
    root = zarr.open_group(store=s.store, mode="a")
    _log("  branch 'user1-normalize' created from main; writable session open")

    # NO physical stash of raw counts: X is left untouched as the raw-count
    # matrix. The normalized result is written to a NEW layer 'layers/norm', so
    # nothing in X is copied or rewritten (zero data movement for the raw counts).
    #
    # Read through a read-only session: its store is picklable, so dask-distributed
    # can ship the lazy-X graph to the GPU workers. A writable session refuses to
    # pickle (uncommitted changeset state) and would raise on .compute()/.persist().
    # X (raw counts) is already committed on main, which this branch was forked from.
    _log("  loading lazy AnnData (read-only session, dask-backed X = raw counts)")
    adata = _load_adata(repo.readonly_session(branch="user1-normalize").store)
    _log(f"  AnnData: X.shape={adata.X.shape} dtype={adata.X.dtype} "
         f"chunks={adata.X.chunksize}; moving to GPU")
    rsc.get.anndata_to_GPU(adata)
    _log("  on GPU; calculate_qc_metrics")
    rsc.pp.calculate_qc_metrics(adata)
    _log("  normalize_total")
    rsc.pp.normalize_total(adata)
    _log("  log1p")
    rsc.pp.log1p(adata)
    _log("  normalization done; streaming normalized X -> layers/norm")

    # anndata's dask-sparse writer streams: it writes the first row-block, then
    # appends each subsequent block. write_cupy_dask maps .get() per block so the
    # device->host transfer is also one block at a time. Peak host RAM ~= one
    # block, never the full CSR (this replaces the old _host_csr full-materialize).
    layers = root.require_group("layers")
    layers.attrs["encoding-type"] = "dict"
    layers.attrs["encoding-version"] = "0.1.0"
    write_elem(layers, "norm", adata.X)
    _log("  layers/norm written; committing user1 snapshot")
    snap = s.commit("user1: normalized+log1p in layers/norm; raw counts kept in X")
    repo.reset_branch("main", snap)
    print("step1 user1:", snap, flush=True)
    _log("step1 user1_normalize: DONE")


def user2_subset(repo):
    print("step2 history seen by user2:")
    for info in repo.ancestry(branch="main"):
        print("   ", info.id, info.message)

    repo.create_branch("user2-subset", repo.lookup_branch("main"))
    s = repo.writable_session("user2-subset")
    root = zarr.open_group(store=s.store, mode="a")

    # Read via a picklable read-only session (see user1_normalize); user1 wrote the
    # normalized matrix to layers/norm (X is still raw counts), so source X from there.
    adata = _load_adata(repo.readonly_session(branch="user2-subset").store,
                        x_key="layers/norm")
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

    _configure_parallelism()

    # Icechunk only creates into a *clean* prefix (root_is_clean: rejects if any
    # object exists under REPO_PATH). The demo is meant to start fresh each run, so
    # wipe the prefix in-process — this uses the script's cwd, immune to a shell
    # `rm` that ran against a different working directory than the script sees.
    abs_repo = os.path.abspath(REPO_PATH)
    print("repo prefix:", abs_repo)
    shutil.rmtree(abs_repo, ignore_errors=True)
    repo = Repository.create(local_filesystem_storage(abs_repo))
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
