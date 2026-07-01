"""PoC: virtually ingest an anndata CSR zarr into icechunk (no chunk copy).

Compares against the notebook's `_copy_csr` on the one thing that matters:
does the result still round-trip through anndata's `read_elem` / `read_elem_lazy`?

Run:  pixi run python dev/virtual_ingest_poc.py
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
import anndata as ad
from anndata.io import read_elem, write_elem
from anndata.experimental import read_elem_lazy

import icechunk as ic
from icechunk.virtual import VirtualChunkContainer, VirtualChunkSpec

ad.settings.zarr_write_format = 3


# ---------------------------------------------------------------------------
# The ingest under test — build the CSR scaffolding natively, reference the
# source chunk *files* instead of copying their bytes.
# ---------------------------------------------------------------------------
def virtual_ingest_csr(src_grp, dst_parent, name, store):
    """Recreate an anndata CSR group in `store` with its data/indices/indptr
    chunks pointing (virtually) at the source zarr's chunk files."""
    dst = dst_parent.require_group(name)
    dst.attrs["encoding-type"] = src_grp.attrs["encoding-type"]
    dst.attrs["encoding-version"] = "0.1.0"
    dst.attrs["shape"] = list(src_grp.attrs["shape"])
    for child in ("data", "indices", "indptr"):
        s = src_grp[child]
        if s.shards is not None:
            raise NotImplementedError(f"{s.name}: sharded source needs inner-chunk offsets")
        d = dst.create_array(child, shape=s.shape, dtype=s.dtype, chunks=s.chunks,
                             filters=s.filters, compressors=s.compressors,
                             serializer=s.serializer, fill_value=s.fill_value,
                             overwrite=True)
        d.attrs["encoding-type"] = "array"
        d.attrs["encoding-version"] = "0.2.0"
        src_dir = os.path.join(str(s.store.root), s.name.lstrip("/"))
        specs = []
        for i in range(math.ceil(s.shape[0] / s.chunks[0])):
            key = s.metadata.chunk_key_encoding.encode_chunk_key((i,))  # 'c/0'
            path = os.path.join(src_dir, key)
            if os.path.exists(path):  # missing => all-fill chunk, leave unreferenced
                specs.append(VirtualChunkSpec(index=[i], location=f"file://{path}",
                                              offset=0, length=os.path.getsize(path)))
        failed = store.set_virtual_refs(f"/{dst.path}/{child}", specs, validate_containers=True)
        assert failed is None, f"{child}: unmatched refs {failed}"
        print(f"    {child}: {len(specs)} virtual refs ({s.dtype}, {s.shape[0]:,} elems, 0 bytes copied)")
    return dst


# ---------------------------------------------------------------------------
# 1. Build a tiny, non-sharded, multi-chunk anndata CSR zarr (the "source").
# ---------------------------------------------------------------------------
def build_fixture(path, n_obs=200, n_var=50, density=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.random(n_obs, n_var, density=density, format="csr",
                  random_state=seed, dtype=np.float32)
    X.sort_indices()

    src = zarr.open_group(path, mode="w")
    src.attrs["encoding-type"] = "anndata"
    src.attrs["encoding-version"] = "0.1.0"

    g = src.require_group("X")
    g.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0",
                    "shape": [n_obs, n_var]})
    for child, arr, ch in [("data", X.data, 128),
                           ("indices", X.indices.astype(np.int32), 128),
                           ("indptr", X.indptr.astype(np.int64), 64)]:
        a = g.create_array(child, shape=arr.shape, dtype=arr.dtype, chunks=(ch,))
        a.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
        a[:] = arr

    obs = pd.DataFrame({"cell_type": pd.Categorical(rng.choice(["B cell", "T cell"], n_obs))},
                       index=[f"cell{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene{j}" for j in range(n_var)])
    write_elem(src, "obs", obs)
    write_elem(src, "var", var)
    return X


def main():
    tmp = tempfile.mkdtemp(prefix="viz_poc_")
    src_path = os.path.join(tmp, "source.zarr")
    repo_path = os.path.join(tmp, "repo")
    try:
        X_true = build_fixture(src_path)
        src = zarr.open_group(src_path, mode="r")
        print(f"source: X {tuple(src['X'].attrs['shape'])}, nnz={X_true.nnz}, "
              f"data.chunks={src['X']['data'].chunks}, shards={src['X']['data'].shards}")

        # icechunk repo with a file:// virtual container for the source tree
        src_abs = os.path.abspath(src_path)
        prefix = f"file://{src_abs}/"
        config = ic.config.RepositoryConfig.default()
        config.set_virtual_chunk_container(
            VirtualChunkContainer(prefix, ic.local_filesystem_store(src_abs)))
        storage = ic.local_filesystem_storage(repo_path)
        repo = ic.Repository.create(storage, config=config,
                                    authorize_virtual_chunk_access={prefix: None})
        repo.save_config()

        session = repo.writable_session("main")
        root = zarr.open_group(store=session.store, mode="a")
        root.attrs["encoding-type"] = "anndata"
        root.attrs["encoding-version"] = "0.1.0"
        print("  ingest X (virtual):")
        virtual_ingest_csr(src["X"], root, "X", session.store)
        write_elem(root, "obs", read_elem(src["obs"]))   # small, copied normally
        write_elem(root, "var", read_elem(src["var"]))
        snap = session.commit("virtual ingest of raw CSR counts")
        print(f"  committed {snap}")

        # ---- read back through anndata (reopen fresh) ----
        repo2 = ic.Repository.open(storage, authorize_virtual_chunk_access={prefix: None})
        rroot = zarr.open_group(store=repo2.readonly_session(branch="main").store, mode="r")

        eager = read_elem(rroot["X"])
        lazy = read_elem_lazy(rroot["X"], (50, X_true.shape[1])).compute()
        obs_back = read_elem(rroot["obs"])

        ok_eager = sp.issparse(eager) and (eager != X_true).nnz == 0
        ok_lazy = (sp.csr_matrix(lazy) != X_true).nnz == 0
        ok_obs = list(obs_back["cell_type"]) == list(read_elem(src["obs"])["cell_type"])
        print(f"\n  read_elem  round-trip: {'PASS' if ok_eager else 'FAIL'} "
              f"(type={type(eager).__name__})")
        print(f"  read_elem_lazy         : {'PASS' if ok_lazy else 'FAIL'}")
        print(f"  obs cell_type          : {'PASS' if ok_obs else 'FAIL'}")
        print(f"\nRESULT: {'ALL PASS' if (ok_eager and ok_lazy and ok_obs) else 'FAILURE'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
