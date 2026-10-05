from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from annizarr._core._config import AppConfig
from annizarr._core._runtime import stage
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


def load_h5ad(input_path: Path, cfg: AppConfig) -> tuple[ad.AnnData, list[str]]:
    import anndata as ad

    mode = "lazy (streaming)" if cfg.io.lazy else "eager (full load)"
    with stage(f"Reading {input_path} [{mode}]"):
        if cfg.io.lazy:
            try:
                return ad.read_h5ad(input_path, backed="r"), []
            except Exception as exc:
                raise ConversionError(
                    f"Lazy load failed for {input_path} ({type(exc).__name__}: {exc}). "
                    "Remove --lazy / set lazy=false to load eagerly."
                ) from exc
        return ad.read_h5ad(input_path), []


def close_lazy_if_needed(adata: ad.AnnData) -> None:
    if getattr(adata, "isbacked", False):
        file_manager = getattr(adata, "file", None)
        if file_manager is not None:
            file_manager.close()
