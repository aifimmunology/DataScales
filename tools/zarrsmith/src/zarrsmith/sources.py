from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import scipy.sparse as sp

from .config import AppConfig
from .engine import _stage
from .errors import ConversionError


def _load_h5ad_for_conversion(input_path: Path, cfg: AppConfig) -> tuple[ad.AnnData, list[str]]:
    mode = "backed (streaming)" if cfg.io.backed else "eager (full load)"
    with _stage(f"Reading {input_path} [{mode}]"):
        if cfg.io.backed:
            try:
                return ad.read_h5ad(input_path, backed="r"), []
            except Exception as exc:
                raise ConversionError(
                    f"Backed load failed for {input_path} ({type(exc).__name__}: {exc}). "
                    "Remove --backed / set backed=false to load eagerly."
                ) from exc
        return ad.read_h5ad(input_path), []


def _close_backed_if_needed(adata: ad.AnnData) -> None:
    if getattr(adata, "isbacked", False):
        file_manager = getattr(adata, "file", None)
        if file_manager is not None:
            file_manager.close()


def _get_indptr(matrix: Any):
    """Return indptr as a numpy array for in-memory or backed sparse input."""
    import numpy as np
    if sp.issparse(matrix):
        return np.asarray(matrix.indptr)
    if hasattr(matrix, "indptr"):
        return np.asarray(matrix.indptr[:])
    if hasattr(matrix, "group"):
        return np.asarray(matrix.group["indptr"][:])
    raise ConversionError(
        f"Cannot locate indptr on sparse input of type {type(matrix).__name__}"
    )


def _ensure_csr(matrix: Any, label: str) -> tuple[Any, str | None]:
    """Return (csr_matrix, optional warning). Accepts CSR or CSC (in-memory or backed)."""
    is_csr = sp.isspmatrix_csr(matrix) or (
        not sp.issparse(matrix) and getattr(matrix, "format", None) == "csr"
    )
    is_csc = sp.isspmatrix_csc(matrix) or (
        not sp.issparse(matrix) and getattr(matrix, "format", None) == "csc"
    )
    if is_csr:
        return matrix, None
    if is_csc:
        if sp.issparse(matrix):
            converted = matrix.tocsr()
        else:
            converted = matrix[:].tocsr()
        return converted, f"[{label}] adata.X was CSC and converted to CSR in memory."
    raise ConversionError(
        f"[{label}] adata.X must be CSR or CSC; got {type(matrix).__name__}"
    )
