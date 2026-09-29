from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy.sparse as sp

from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["ensure_csr", "get_indptr", "is_backed", "matrix_format"]


def matrix_format(matrix: Any) -> Literal["csr", "csc", "dense"]:
    # recognises in-memory scipy sparse and anndata backed datasets (_CSRDataset/_CSCDataset,
    # via their .format attribute) as csr/csc; a plain 2-D array-like (ndarray, h5py-backed
    # dense) is "dense"; anything else (COO/LIL/DOK/BSR sparse, 1-D arrays, unknown objects)
    # is rejected outright rather than silently treated as dense
    if sp.isspmatrix_csr(matrix):
        return "csr"
    if sp.isspmatrix_csc(matrix):
        return "csc"
    if sp.issparse(matrix):
        raise ConversionError(f"adata.X must be CSR, CSC, or a 2-D dense array. Got: {matrix.format}")
    backed_fmt = getattr(matrix, "format", None)
    if backed_fmt == "csr":
        return "csr"
    if backed_fmt == "csc":
        return "csc"
    if backed_fmt is None and getattr(matrix, "ndim", 0) == 2:
        return "dense"
    got = backed_fmt or type(matrix).__name__
    raise ConversionError(f"adata.X must be CSR, CSC, or a 2-D dense array. Got: {got}")


def is_backed(matrix: Any) -> bool:
    return not sp.issparse(matrix) and not isinstance(matrix, np.ndarray)


def get_indptr(matrix: Any) -> NDArray[np.int64]:
    if sp.issparse(matrix):
        return np.asarray(matrix.indptr)
    if hasattr(matrix, "indptr"):
        return np.asarray(matrix.indptr[:])
    if hasattr(matrix, "group"):
        return np.asarray(matrix.group["indptr"][:])
    raise ConversionError(f"Cannot locate indptr on sparse input of type {type(matrix).__name__}")


def ensure_csr(matrix: Any, label: str, *, eager_max_bytes: int | None = None) -> tuple[Any, str | None]:
    """(csr_matrix, optional warning); accepts CSR or CSC (in-memory or backed).

    Parameters
    ----------
    eager_max_bytes
        For a backed CSC input, refuse (instead of silently loading it) when its on-disk
        size exceeds this. ``None`` skips the check (matches the pre-existing, unbounded
        behavior) — concat is the only caller that has a budget to enforce.
    """
    fmt = matrix_format(matrix)
    if fmt == "csr":
        return matrix, None
    if fmt == "csc":
        if is_backed(matrix):
            if eager_max_bytes is not None:
                nbytes = _backed_csc_nbytes(matrix)
                if nbytes > eager_max_bytes:
                    raise ConversionError(
                        f"[{label}] backed CSC X is {nbytes} bytes on disk, over eager_max_bytes "
                        f"({eager_max_bytes}); loading it whole for concat would blow the memory "
                        f"bound. Run `annizarr convert --x-storage csr` on that input first."
                    )
            converted = matrix[:].tocsr()
        else:
            converted = matrix.tocsr()
        return converted, f"[{label}] adata.X was CSC and converted to CSR in memory."
    raise ConversionError(f"[{label}] adata.X must be CSR or CSC; got {type(matrix).__name__}")


def _backed_csc_nbytes(matrix: Any) -> int:
    g = matrix.group
    return int(g["data"].nbytes) + int(g["indices"].nbytes) + int(g["indptr"].nbytes)
