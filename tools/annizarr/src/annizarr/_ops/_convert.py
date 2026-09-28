from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse as sp

from .._config import AppConfig, resolve_backend_cfg
from .._runtime import configure_runtime
from .._sorting import _write_sorted_backed, maybe_sort_adata
from .._sources import close_backed_if_needed, load_10x_h5, load_h5ad
from .._storage import open_output_store, store_name
from .._validation import validate_single_cell_anndata
from .._writers import write_adata
from ..errors import ConversionError


def write_adata_to_store(
    adata: ad.AnnData,
    output_path: str | Path,
    cfg: AppConfig,
    load_warnings: list[str],
    allow_grouping: bool = False,
) -> list[str]:
    """Write AnnData to zarr (or icechunk).

    adata.X is expected to be CSR; CSC is converted to CSR in memory with a warning, and
    dense X is accepted (streamed for dense output, sparsified in memory for sparse output
    on eager loads). When ``allow_grouping`` and grouping is enabled, rows are sorted first.
    """
    cfg = resolve_backend_cfg(cfg)
    configure_runtime(cfg.chunks.cpus)
    warnings = list(load_warnings)

    if allow_grouping:
        if cfg.grouping.enabled and cfg.io.backed:
            # Backed input: sort X without ever materialising it (streamed bucket + concat).
            return _write_sorted_backed(adata, Path(output_path), cfg, warnings)
        adata = maybe_sort_adata(adata, cfg, warnings)
    elif cfg.grouping.enabled:
        raise ConversionError(
            "grouping (sort_by) is only supported by convert for now."
        )

    x_for_write: Any | None = None
    x = adata.X
    is_sparse_mem = sp.issparse(x)
    backed_format = getattr(x, "format", None)  # anndata backed sparse datasets
    is_csr = sp.isspmatrix_csr(x) or (not is_sparse_mem and backed_format == "csr")
    is_csc = sp.isspmatrix_csc(x) or (not is_sparse_mem and backed_format == "csc")
    # dense: in-memory ndarray, or a backed h5py dataset (2-D, no sparse format);
    # masked arrays are excluded — silently dropping a mask would corrupt values
    is_dense = (
        not is_sparse_mem
        and backed_format is None
        and getattr(x, "ndim", 0) == 2
        and not np.ma.isMaskedArray(x)
    )

    if not is_csr:
        if is_csc:
            if is_sparse_mem:
                x_for_write = x.tocsr()
            else:
                x_for_write = x[:].tocsr()
            warnings.append(
                "adata.X was CSC and has been converted to CSR in memory before zarr conversion."
            )
        elif is_dense and cfg.io.x_storage == "dense":
            pass  # the writer streams dense input to dense output directly
        elif is_dense and not cfg.io.backed:
            # eager dense (issue #4): X is already in memory — sparsify for sparse output,
            # directly in the target format (no CSR->CSC double conversion)
            to_sparse = sp.csr_matrix if cfg.io.x_storage == "sparse-csr" else sp.csc_matrix
            x_for_write = to_sparse(np.asarray(x))
            warnings.append(
                "adata.X was dense and has been converted to sparse in memory for sparse output."
            )
        elif is_dense:
            raise ConversionError(
                "backed dense X with sparse output is not supported: use --x-storage dense "
                "(streamed) or omit --backed to convert in memory."
            )
        else:
            raise ConversionError(
                f"adata.X must be CSR, CSC, or dense. Got: {type(x).__name__}"
            )

    validation_result = validate_single_cell_anndata(adata, cfg.validation)
    ad.settings.zarr_write_format = 3

    print(
        f"Converting → {output_path} "
        f"(n_obs={adata.n_obs}, n_vars={adata.n_vars}, {cfg.io.x_storage}, backend={cfg.io.backend})",
        flush=True, file=sys.stderr,
    )
    t0 = time.perf_counter()
    store, finalize = open_output_store(
        output_path, cfg, commit_message=f"annizarr convert → {store_name(output_path)}",
    )
    write_adata(adata, store, cfg, x_override=x_for_write)
    finalize()
    print(
        f"Done in {time.perf_counter() - t0:.1f}s",
        flush=True, file=sys.stderr,
    )
    return [*warnings, *validation_result.warnings]


def convert_adata(adata: ad.AnnData, output_zarr: str, cfg: AppConfig) -> list[str]:
    """Write an in-memory AnnData to a zarr (or icechunk) store; returns warnings.

    Public library entry for data that doesn't start as .h5ad/10x: X may be CSR, CSC
    (converted), or dense; honors x_storage/backend/sort_by exactly like convert_h5ad.
    """
    if cfg.io.backed:
        raise ConversionError(
            "convert_adata takes an in-memory AnnData; io.backed does not apply."
        )
    return write_adata_to_store(adata, output_zarr, cfg, [], allow_grouping=True)


def convert_h5ad(input_h5ad: str, output_zarr: str, cfg: AppConfig) -> list[str]:
    """Convert a .h5ad file to zarr; returns the warnings encountered."""
    adata = None
    load_warnings: list[str] = []
    try:
        adata, load_warnings = load_h5ad(Path(input_h5ad), cfg)
        return write_adata_to_store(adata, output_zarr, cfg, load_warnings, allow_grouping=True)
    except Exception as e:
        load_warnings = f"Warnings: {load_warnings}" if load_warnings else ""
        raise ConversionError(f"Failed to convert .h5ad file: {e}.{load_warnings}") from e
    finally:
        if adata is not None:
            close_backed_if_needed(adata)


def convert_10x_h5(input_h5: str, output_zarr: str, cfg: AppConfig) -> list[str]:
    """Convert a 10x Cell Ranger .h5 to zarr; expects CSR from the 10x load."""
    try:
        adata = load_10x_h5(input_h5)
    except Exception as e:
        raise ConversionError(f"Failed to read 10x H5 file: {e}") from e

    return write_adata_to_store(adata, output_zarr, cfg, [])
