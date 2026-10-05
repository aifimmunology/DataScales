from __future__ import annotations

from typing import TYPE_CHECKING

from annizarr._writers._dense import write_dense
from annizarr._writers._sparse import write_csc, write_csr

if TYPE_CHECKING:
    import zarr

    from annizarr._core._config import AppConfig
    from annizarr._sources._readers import Reader

__all__ = ["write_matrix"]


def write_matrix(group: zarr.Group, key: str, reader: Reader, cfg: AppConfig) -> None:
    # dispatches purely on cfg.io.x_storage; the reader's backing kind never matters here
    # (only Reader.thread_safe, inside each writer, for the parallel-mode choice).
    if cfg.io.x_storage == "dense":
        write_dense(group, key, reader, cfg)
    elif cfg.io.x_storage == "csr":
        write_csr(group, key, reader, cfg)
    else:
        write_csc(group, key, reader, cfg)
