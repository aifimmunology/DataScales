from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from annizarr._runtime import stage
from annizarr._sources._matrix import is_backed, matrix_format
from annizarr._writers._dense import _write_dense_streaming, _write_sparse_as_dense
from annizarr._writers._encoding import set_anndata_root_attrs, set_raw_group_attrs, write_elem
from annizarr._writers._sparse import _write_sparse_streaming
from annizarr._zarr import get_group

if TYPE_CHECKING:
    import anndata as ad
    import zarr

    from annizarr._config import AppConfig


def write_matrix(group: zarr.Group, matrix: Any, key: str, cfg: AppConfig) -> None:
    # input may be any format (CSR, CSC, dense ndarray, backed SparseDataset) — adata.X is
    # guaranteed CSR by the caller, but other layers and raw.X may not be. Dispatches on
    # cfg.io.x_storage: dense -> row-chunked streaming write (CSC converted to CSR first for
    # row slicing); csr/csc -> ensure that format, stream row/col-batches over a thread pool
    # (parallel for in-memory input; a process pool for backed input).
    mode = cfg.io.x_storage
    fmt = matrix_format(matrix)
    backed = is_backed(matrix)

    if mode == "dense":
        if fmt in ("csr", "csc"):
            if fmt != "csr":
                matrix = matrix.tocsr()  # backed CSC: load into memory and convert; in-memory: direct
            _write_sparse_as_dense(group, matrix, key, cfg)
        else:  # already dense
            _write_dense_streaming(group, matrix, key, cfg)
        return

    # Sparse output (csr, csc).
    # For backed input whose format doesn't match the target, we have no choice but to
    # load into memory and convert (no incremental transpose). For matching format
    # (CSR→CSR or CSC→CSC), streaming works directly from backed storage.
    if backed and fmt in ("csr", "csc"):
        if fmt != mode:
            matrix = matrix[:].tocsc() if mode == "csc" else matrix[:].tocsr()
    elif fmt == "dense":
        if not backed:
            # eager dense layer / raw.X under sparse output: sparsify in memory (issue #4)
            matrix = sp.csr_matrix(np.asarray(matrix)) if mode == "csr" else sp.csc_matrix(np.asarray(matrix))
        # backed dense + sparse output has no incremental conversion path; left as-is
        # (rejected earlier, at the convert-op level, for X — this is a latent gap for layers)
    elif fmt != mode:
        matrix = matrix.tocsr() if mode == "csr" else matrix.tocsc()

    _write_sparse_streaming(group, matrix, key, cfg, csr=(mode == "csr"))


def write_adata(adata: ad.AnnData, store: zarr.Group, cfg: AppConfig, x_override: Any | None = None) -> None:
    # writes an AnnData with CSR X (in-memory or backed SparseDataset) into store; matrices
    # are written with the target format and chunking in one step. The caller owns opening
    # the store and finalising it (consolidate / icechunk commit).
    set_anndata_root_attrs(store)

    with stage("Writing metadata (obs/var/uns/obsm/varm/obsp/varp)"):
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
        write_elem(store, "layers", {})
        layers_group = get_group(store, "layers")
        for name, data in adata.layers.items():
            with stage(f"Writing layers/{name} (shape={data.shape})"):
                write_matrix(layers_group, data, name, cfg)

    if adata.raw is not None:
        raw_group = store.require_group("raw")
        set_raw_group_attrs(raw_group)
        write_elem(raw_group, "var", adata.raw.var)
        write_elem(raw_group, "varm", dict(adata.raw.varm))
        with stage(f"Writing raw/X (shape={adata.raw.X.shape})"):
            write_matrix(raw_group, adata.raw.X, "X", cfg)
