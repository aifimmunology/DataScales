from __future__ import annotations

from ._h5ad import close_backed_if_needed, load_h5ad
from ._matrix import ensure_csr, get_indptr
from ._tenx import load_10x_h5

__all__ = [
    "close_backed_if_needed",
    "ensure_csr",
    "get_indptr",
    "load_10x_h5",
    "load_h5ad",
]
