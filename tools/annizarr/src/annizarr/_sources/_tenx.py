from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

    import anndata as ad


def load_10x_h5(path: str | os.PathLike[str]) -> ad.AnnData:
    # the only scanpy call site (dropped in Phase 2)
    import scanpy as sc

    return sc.read_10x_h5(str(path))
