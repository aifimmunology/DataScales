from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import scipy.sparse as sp

from ..config import AppConfig, _resolve_backend_cfg
from ..engine import configure_runtime
from ..errors import ConversionError
from ..sources import _close_backed_if_needed, _load_h5ad_for_conversion
from ..storage import open_output_store
from ..validation import validate_single_cell_anndata
from ..writers import _write_csr_adata_direct
from .sort import _maybe_sort_adata, _write_sorted_backed


def _write_adata_to_zarr(
    adata: ad.AnnData,
    output_path: Path,
    cfg: AppConfig,
    load_warnings: list[str],
    allow_grouping: bool = False,
) -> list[str]:
    """Write AnnData to zarr (or icechunk).

    adata.X is expected to be CSR, but CSC is accepted and converted to CSR in memory with
    a warning. When ``allow_grouping`` and grouping is enabled, rows are sorted first.
    """
    cfg = _resolve_backend_cfg(cfg)
    configure_runtime(cfg.chunks.cpus)
    warnings = list(load_warnings)

    if allow_grouping:
        if cfg.grouping.enabled and cfg.io.backed:
            # Backed input: sort X without ever materialising it (streamed bucket + concat).
            return _write_sorted_backed(adata, output_path, cfg, warnings)
        adata = _maybe_sort_adata(adata, cfg, warnings)
    elif cfg.grouping.enabled:
        raise ConversionError(
            "grouping (sort_by) is only supported by convert for now."
        )

    x_for_write: Any | None = None
    is_csr = sp.isspmatrix_csr(adata.X) or (
        not sp.issparse(adata.X) and getattr(adata.X, "format", None) == "csr"  # backed mode
    )
    is_csc = sp.isspmatrix_csc(adata.X) or (
        not sp.issparse(adata.X) and getattr(adata.X, "format", None) == "csc"  # backed mode
    )

    if not is_csr:
        if is_csc:
            if sp.issparse(adata.X):
                x_for_write = adata.X.tocsr()
            else:
                x_for_write = adata.X[:].tocsr()
            warnings.append(
                "adata.X was CSC and has been converted to CSR in memory before zarr conversion."
            )
        else:
            raise ConversionError(
                f"adata.X must be CSR or CSC format. Got: {type(adata.X).__name__}"
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
        output_path, cfg, commit_message=f"zarrsmith convert → {output_path.name}",
    )
    _write_csr_adata_direct(adata, store, cfg, x_override=x_for_write)
    finalize()
    print(
        f"Done in {time.perf_counter() - t0:.1f}s",
        flush=True, file=sys.stderr,
    )
    return [*warnings, *validation_result.warnings]


def convert_h5ad_to_zarr(input_h5ad: str, output_zarr: str, cfg: AppConfig) -> list[str]:
    """Convert a .h5ad file to zarr; returns the warnings encountered."""
    adata = None
    load_warnings: list[str] = []
    try:
        adata, load_warnings = _load_h5ad_for_conversion(Path(input_h5ad), cfg)
        return _write_adata_to_zarr(adata, Path(output_zarr), cfg, load_warnings, allow_grouping=True)
    except Exception as e:
        load_warnings = f"Warnings: {load_warnings}" if load_warnings else ""
        raise ConversionError(f"Failed to convert .h5ad file: {e}.{load_warnings}") from e
    finally:
        if adata is not None:
            _close_backed_if_needed(adata)


def convert_10x_h5_to_zarr(input_h5: str, output_zarr: str, cfg: AppConfig) -> list[str]:
    """Convert a 10x Cell Ranger .h5 to zarr; expects CSR from the 10x load."""
    try:
        import scanpy as sc
        adata = sc.read_10x_h5(str(input_h5))
    except Exception as e:
        raise ConversionError(f"Failed to read 10x H5 file: {e}") from e

    return _write_adata_to_zarr(adata, Path(output_zarr), cfg, [])
