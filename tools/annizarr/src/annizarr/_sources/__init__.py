from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from annizarr._sources._base import Loader, Sniffer, Source
from annizarr._sources._detect import detect_format
from annizarr._sources._detect import register_sniffer as _register_sniffer
from annizarr._sources._h5ad import close_backed_if_needed, load_h5ad
from annizarr._sources._matrix import ensure_csr, get_indptr
from annizarr._sources._memory import load_anndata
from annizarr._sources._tenx import load_10x_h5
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from anndata import AnnData

    from annizarr._config import AppConfig
    from annizarr.typing import PathLike

__all__ = [
    "Loader",
    "Sniffer",
    "Source",
    "close_backed_if_needed",
    "detect_format",
    "ensure_csr",
    "get_indptr",
    "load_10x_h5",
    "load_h5ad",
    "open_source",
    "register_source",
]


def _load_h5ad_source(path: PathLike, cfg: AppConfig) -> Source:
    adata, warnings = load_h5ad(Path(path), cfg)
    return Source(adata=adata, kind="h5ad", backed=cfg.io.backed, warnings=tuple(warnings))


def _load_10x_source(path: PathLike, cfg: AppConfig) -> Source:
    warnings: list[str] = []
    if cfg.io.backed:
        warnings.append("10x input is always loaded eagerly; --backed has no effect for it.")
    try:
        adata = load_10x_h5(path)
    except Exception as exc:
        raise ConversionError(f"Failed to read 10x H5 file: {exc}") from exc
    return Source(adata=adata, kind="10x", backed=False, warnings=tuple(warnings))


_LOADERS: dict[str, Loader] = {
    "h5ad": _load_h5ad_source,
    "10x": _load_10x_source,
}


def register_source(kind: str, loader: Loader, *, sniffer: Sniffer | None = None) -> None:
    """Register a loader (and optional content sniffer) for a new input kind.

    Parameters
    ----------
    kind
        Name under which ``loader`` is registered; pass it as ``fmt=`` to
        :func:`open_source`, or let a registered ``sniffer`` select it automatically.
    loader
        Called as ``loader(path, cfg)``; must return a :class:`Source`.
    sniffer
        Optional content sniffer consulted by :func:`detect_format` before the builtin
        rules; it returns ``kind`` when it recognises ``path``, else ``None``.
    """
    _LOADERS[kind] = loader
    if sniffer is not None:
        _register_sniffer(sniffer)


def open_source(source: PathLike | AnnData, cfg: AppConfig, *, fmt: str | None = None) -> Source:
    """Open any registered input — a path/URI or an in-memory AnnData — as a Source.

    Parameters
    ----------
    source
        A path/URI to detect and load, or an in-memory :class:`~anndata.AnnData`.
    cfg
        Resolved configuration, passed through to the selected loader.
    fmt
        Explicit kind, overriding content detection (ignored for an AnnData ``source``).

    Returns
    -------
    Source
        The opened input, ready to write; call :meth:`Source.close` when done with it.

    Raises
    ------
    ConversionError
        ``fmt`` (or the detected kind) is not a registered source kind.
    """
    import anndata as ad

    if isinstance(source, ad.AnnData):
        return load_anndata(source)

    kind = fmt or detect_format(source)
    try:
        loader = _LOADERS[kind]
    except KeyError:
        known = ", ".join(sorted(_LOADERS))
        raise ConversionError(f"Unknown source kind {kind!r}; registered kinds: {known}.") from None
    return loader(source, cfg)
