from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

    import anndata as ad


def load_10x_h5(path: str | os.PathLike[str]) -> ad.AnnData:
    """Load a 10x Genomics Cell Ranger HDF5 (.h5) file as an in-memory CSR AnnData.

    Lazily imports scanpy — this is the only scanpy call site (dropped in Phase 2).
    """
    import scanpy as sc
    return sc.read_10x_h5(str(path))
