from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from annizarr._config import AppConfig
from annizarr._runtime import stage
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    import anndata as ad


def load_h5ad(input_path: Path, cfg: AppConfig) -> tuple[ad.AnnData, list[str]]:
    import anndata as ad

    mode = "backed (streaming)" if cfg.io.backed else "eager (full load)"
    with stage(f"Reading {input_path} [{mode}]"):
        if cfg.io.backed:
            try:
                return ad.read_h5ad(input_path, backed="r"), []
            except Exception as exc:
                raise ConversionError(
                    f"Backed load failed for {input_path} ({type(exc).__name__}: {exc}). "
                    "Remove --backed / set backed=false to load eagerly."
                ) from exc
        return ad.read_h5ad(input_path), []


def close_backed_if_needed(adata: ad.AnnData) -> None:
    if getattr(adata, "isbacked", False):
        file_manager = getattr(adata, "file", None)
        if file_manager is not None:
            file_manager.close()
