from __future__ import annotations

from typing import Any

import scipy.sparse as sp

from ..errors import ConversionError


def get_indptr(matrix: Any):
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


def ensure_csr(matrix: Any, label: str) -> tuple[Any, str | None]:
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
