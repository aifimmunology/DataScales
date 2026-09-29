from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from annizarr._config import AppConfig
from annizarr._runtime import stage
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


def _peek_x_nbytes(path: Path) -> int:
    """On-disk byte size of ``X``, from HDF5 dataset metadata only — no data read."""
    import h5py

    with h5py.File(path, "r") as f:
        if "X" not in f:
            raise ConversionError(f"{path} has no X.")
        x = f["X"]
        if isinstance(x, h5py.Dataset):
            return int(x.nbytes)
        return int(x["data"].nbytes + x["indices"].nbytes + x["indptr"].nbytes)


def load_h5ad(input_path: Path, cfg: AppConfig) -> tuple[ad.AnnData, list[str]]:
    import anndata as ad

    backed = cfg.io.backed
    if backed is None:
        x_bytes = _peek_x_nbytes(input_path)
        backed = x_bytes > cfg.io.eager_max_bytes
        logger.info(
            f"Auto-selected {'backed' if backed else 'eager'} load for {input_path}: "
            f"X on-disk size={x_bytes} bytes, eager_max_bytes={cfg.io.eager_max_bytes} bytes."
        )

    mode = "backed (streaming)" if backed else "eager (full load)"
    with stage(f"Reading {input_path} [{mode}]"):
        if backed:
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
