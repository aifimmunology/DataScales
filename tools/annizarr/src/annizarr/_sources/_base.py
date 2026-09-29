from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from annizarr._config import AppConfig
from annizarr._sources._h5ad import close_backed_if_needed
from annizarr.typing import PathLike

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["Loader", "Sniffer", "Source"]


@dataclass(frozen=True, slots=True)
class Source:
    """An opened conversion input, ready to write.

    Parameters
    ----------
    adata
        The loaded AnnData — eager or backed, per ``kind``/``backed``.
    kind
        Registered source kind that produced this (e.g. ``"h5ad"``, ``"10x"``, ``"anndata"``).
    backed
        Whether ``adata`` is backed (HDF5-streamed) rather than fully in memory.
    warnings
        Non-fatal notes collected while loading (e.g. a format conversion).
    """

    adata: AnnData
    kind: str
    backed: bool
    warnings: tuple[str, ...]

    def close(self) -> None:
        """Close a backed h5 handle; a no-op for an eager or in-memory source."""
        close_backed_if_needed(self.adata)


# Loader/Sniffer are plain Callable aliases; a custom kind is not restricted to the
# builtin kinds below, so the return type here is documentation, not an enforced bound.
Loader: TypeAlias = Callable[[PathLike, AppConfig], Source]
Sniffer: TypeAlias = Callable[[Path], str | None]
