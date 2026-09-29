from __future__ import annotations

from typing import TYPE_CHECKING

from annizarr._sources._base import Source

if TYPE_CHECKING:
    from anndata import AnnData


def load_anndata(adata: AnnData) -> Source:
    return Source(adata=adata, kind="anndata", backed=False, warnings=())
