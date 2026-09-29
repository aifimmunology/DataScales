from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from annizarr._runtime import stage
from annizarr._sources._matrix import is_backed, matrix_format
from annizarr._writers._dense import _write_dense_streaming, _write_sparse_as_dense
from annizarr._writers._encoding import autoshard_setting, set_anndata_root_attrs, set_raw_group_attrs, write_elem
from annizarr._writers._sparse import _local_tmp_dir, _write_sparse_streaming, write_transposed_sparse
from annizarr._zarr import get_group

if TYPE_CHECKING:
    import anndata as ad
    import zarr

    from annizarr._config import AppConfig


def write_matrix(group: zarr.Group, matrix: Any, key: str, cfg: AppConfig) -> None:
    # input may be any format (CSR, CSC, dense ndarray, backed SparseDataset) — adata.X is
    # guaranteed CSR by the caller, but other layers and raw.X may not be. Dispatches on
    # cfg.io.x_storage: dense -> row-chunked streaming write (an in-memory CSC is converted
    # to CSR first for row slicing; a backed CSC has no in-memory .tocsr(), so it goes
    # through a temporary streamed CSC->CSR transpose instead — never materialised whole);
    # csr/csc -> ensure that format, stream row/col-batches over a thread pool (parallel for
    # in-memory input; a process pool for h5py-backed input).
    mode = cfg.io.x_storage
    fmt = matrix_format(matrix)
    backed = is_backed(matrix)

    if mode == "dense":
        if fmt == "csr":
            _write_sparse_as_dense(group, matrix, key, cfg)
        elif fmt == "csc":
            if backed:
                _write_backed_csc_as_dense(group, matrix, key, cfg)
            else:
                _write_sparse_as_dense(group, matrix.tocsr(), key, cfg)
        else:  # already dense
            _write_dense_streaming(group, matrix, key, cfg)
        return

    # Sparse output (csr, csc).
    # A backed input whose format doesn't match the target streams through the bucket
    # transpose engine (no full-matrix materialisation). For matching format (CSR→CSR or
    # CSC→CSC), streaming works directly from backed storage.
    if backed and fmt in ("csr", "csc"):
        if fmt != mode:
            write_transposed_sparse(group, key, matrix, cfg, target=mode)
            return
    elif fmt == "dense":
        if not backed:
            # eager dense layer / raw.X under sparse output: sparsify in memory (issue #4)
            matrix = sp.csr_matrix(np.asarray(matrix)) if mode == "csr" else sp.csc_matrix(np.asarray(matrix))
        # backed dense + sparse output has no incremental conversion path; left as-is
        # (rejected earlier, at the convert-op level, for X — this is a latent gap for layers)
    elif fmt != mode:
        matrix = matrix.tocsr() if mode == "csr" else matrix.tocsc()

    _write_sparse_streaming(group, matrix, key, cfg, csr=(mode == "csr"))


def _write_backed_csc_as_dense(group: zarr.Group, matrix: Any, key: str, cfg: AppConfig) -> None:
    # anndata's backed _CSCDataset has no .tocsr(); stream-transpose it into a temporary
    # zarr CSR store instead (never materialising the whole matrix), then densify from that
    # — zarr-backed, so thread-safe — CSR the same way as a native CSR source.
    import shutil
    import tempfile
    from pathlib import Path

    import zarr
    from anndata.io import sparse_dataset

    tmp_dir = Path(tempfile.mkdtemp(prefix="annizarr_csc2csr_", dir=_local_tmp_dir(group)))
    try:
        tmp_root = zarr.open_group(str(tmp_dir), mode="w")
        tmp_group = write_transposed_sparse(tmp_root, "X", matrix, cfg, target="csr")
        _write_sparse_as_dense(group, sparse_dataset(tmp_group), key, cfg)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def write_adata(adata: ad.AnnData, store: zarr.Group, cfg: AppConfig, x_override: Any | None = None) -> None:
    # writes an AnnData with CSR X (in-memory or backed SparseDataset) into store; matrices
    # are written with the target format and chunking in one step. The caller owns opening
    # the store and finalising it (consolidate / icechunk commit).
    set_anndata_root_attrs(store)

    with autoshard_setting(cfg.chunks.auto_shard), stage("Writing metadata (obs/var/uns/obsm/varm/obsp/varp)"):
        write_elem(store, "obs", adata.obs)
        write_elem(store, "var", adata.var)
        write_elem(store, "uns", dict(adata.uns))
        write_elem(store, "obsm", dict(adata.obsm))
        write_elem(store, "varm", dict(adata.varm))
        write_elem(store, "obsp", dict(adata.obsp))
        write_elem(store, "varp", dict(adata.varp))

    # X: the heavy step — sub-progress comes from _runtime.progress() inside the writers.
    x_matrix = x_override if x_override is not None else adata.X
    x_nnz = getattr(x_matrix, "nnz", None)
    x_info = f"shape={x_matrix.shape}, {cfg.io.x_storage}"
    if x_nnz is not None:
        x_info += f", nnz={x_nnz}"
    with stage(f"Writing X ({x_info})"):
        write_matrix(store, x_matrix, "X", cfg)

    if adata.layers:
        with autoshard_setting(cfg.chunks.auto_shard):
            write_elem(store, "layers", {})
        layers_group = get_group(store, "layers")
        for name, data in adata.layers.items():
            with stage(f"Writing layers/{name} (shape={data.shape})"):
                write_matrix(layers_group, data, name, cfg)

    if adata.raw is not None:
        raw_group = store.require_group("raw")
        set_raw_group_attrs(raw_group)
        with autoshard_setting(cfg.chunks.auto_shard):
            write_elem(raw_group, "var", adata.raw.var)
            write_elem(raw_group, "varm", dict(adata.raw.varm))
        with stage(f"Writing raw/X (shape={adata.raw.X.shape})"):
            write_matrix(raw_group, adata.raw.X, "X", cfg)
