from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse as sp
import zarr

from .._config import AppConfig
from .._runtime import stage
from ._dense import _write_dense_streaming, _write_sparse_as_dense_dask
from ._encoding import set_anndata_root_attrs, set_raw_group_attrs, write_elem
from ._sparse import _write_sparse_streaming


def write_matrix(
    group: zarr.Group, matrix: Any, key: str, cfg: AppConfig
) -> None:
    """Write a single matrix to zarr in the target format, directly.

    Input may be any format (CSR, CSC, dense ndarray, backed SparseDataset) -> because vvv
    NOTE adata.X is guaranteed CSR by the caller, but other layers and raw.X may not be.

    Dispatches based on cfg.io.x_storage:
      dense      → dask row-chunked write; CSC converted to CSR first for row slicing
      sparse-csr → ensure CSR, stream row-batches via dask (parallel for in-memory input)
      sparse-csc → ensure CSC, stream col-batches via dask (parallel for in-memory input)
    """
    mode = cfg.io.x_storage

    # Backed SparseDataset (anndata HDF5-backed): has .format but is not a scipy sparse matrix.
    is_backed_sparse = not sp.issparse(matrix) and hasattr(matrix, "format")

    if mode == "dense":
        if sp.issparse(matrix) or is_backed_sparse:
            if is_backed_sparse and getattr(matrix, "format", None) != "csr":
                matrix = matrix.tocsr()  # backed CSC: load into memory and convert
            elif sp.issparse(matrix) and not sp.isspmatrix_csr(matrix):
                matrix = matrix.tocsr()
            _write_sparse_as_dense_dask(group, matrix, key, cfg)

        else:  # already dense
            _write_dense_streaming(group, matrix, key, cfg)
        return

    # Sparse output (sparse-csr, sparse-csc).
    # For backed input whose format doesn't match the target, we have no choice but to
    # load into memory and convert (no incremental transpose). For matching format
    # (CSR→CSR or CSC→CSC), streaming works directly from backed storage.
    if is_backed_sparse:
        dataset_format = getattr(matrix, "format", None)
        if mode == "sparse-csc" and dataset_format != "csc":
            # _CSRDataset has no tocsc(); load into memory first then convert.
            matrix = matrix[:].tocsc()
        elif mode == "sparse-csr" and dataset_format != "csr":
            matrix = matrix[:].tocsr()
    elif isinstance(matrix, np.ndarray):
        # eager dense layer / raw.X under sparse output: sparsify in memory (issue #4)
        matrix = sp.csr_matrix(matrix) if mode == "sparse-csr" else sp.csc_matrix(matrix)
    elif mode == "sparse-csr" and sp.issparse(matrix) and not sp.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()
    elif mode == "sparse-csc" and sp.issparse(matrix) and not sp.isspmatrix_csc(matrix):
        matrix = matrix.tocsc()

    _write_sparse_streaming(group, matrix, key, cfg, csr=(mode == "sparse-csr"))


def write_adata(
    adata: ad.AnnData,
    store: zarr.Group,
    cfg: AppConfig,
    x_override: Any | None = None,
) -> None:
    """Write an AnnData with CSR X (in-memory or backed SparseDataset) into ``store``.

    Matrices are written with the target format and chunking in one step. The caller
    owns opening the store and finalising it (consolidate / icechunk commit).
    """
    set_anndata_root_attrs(store)

    with stage("Writing metadata (obs/var/uns/obsm/varm/obsp/varp)"):
        write_elem(store, "obs", adata.obs)
        write_elem(store, "var", adata.var)
        write_elem(store, "uns", dict(adata.uns))
        write_elem(store, "obsm", dict(adata.obsm))
        write_elem(store, "varm", dict(adata.varm))
        write_elem(store, "obsp", dict(adata.obsp))
        write_elem(store, "varp", dict(adata.varp))

    # X: the heavy step — sub-progress comes from dask's ProgressBar inside.
    x_matrix = x_override if x_override is not None else adata.X
    x_nnz = getattr(x_matrix, "nnz", None)
    x_info = f"shape={x_matrix.shape}, {cfg.io.x_storage}"
    if x_nnz is not None:
        x_info += f", nnz={x_nnz}"
    with stage(f"Writing X ({x_info})"):
        write_matrix(store, x_matrix, "X", cfg)

    if adata.layers:
        write_elem(store, "layers", {})
        layers_group = store["layers"]
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
