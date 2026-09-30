from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from annizarr._sources._base import Sniffer
    from annizarr.typing import PathLike

__all__ = ["detect_format"]

_REMOTE_SCHEMES = frozenset({"s3", "gs", "gcs"})
_CUSTOM_SNIFFERS: list[Sniffer] = []


def register_sniffer(sniffer: Sniffer) -> None:
    # consulted in registration order, before the builtin rules below
    _CUSTOM_SNIFFERS.append(sniffer)


def detect_format(path: PathLike) -> Literal["h5ad", "10x", "zarr", "icechunk"]:
    """Sniff the input format of ``path`` from its content, never its extension."""
    if urlparse(str(path)).scheme in _REMOTE_SCHEMES:
        return "icechunk"

    p = Path(path)
    if not p.exists():
        raise ConversionError(f"Input path does not exist: {p}. Pass --from h5ad|10x to override detection.")

    for sniffer in _CUSTOM_SNIFFERS:
        kind = sniffer(p)
        if kind is not None:
            return kind  # type: ignore[return-value]  # custom kinds are not in the Literal

    if p.is_dir():
        if (p / "zarr.json").exists():
            return "zarr"
        if (p / "repo").exists() and (p / "snapshots").is_dir():
            return "icechunk"
        raise ConversionError(
            f"{p} is a directory but is neither a zarr store (zarr.json) nor an icechunk "
            "repo (repo + snapshots/). Pass --from h5ad|10x to override detection."
        )

    return _sniff_hdf5(p)


def _sniff_hdf5(p: Path) -> Literal["h5ad", "10x"]:
    import h5py

    try:
        f = h5py.File(p, "r")
    except OSError as exc:
        raise ConversionError(
            f"{p} is not a readable HDF5 file (h5ad or 10x). Pass --from h5ad|10x to override detection."
        ) from exc

    with f:
        matrix = f.get("matrix")
        if isinstance(matrix, h5py.Group) and "barcodes" in matrix:
            return "10x"  # 10x Cell Ranger v3
        for child in f.values():
            if isinstance(child, h5py.Group) and "barcodes" in child and "genes" in child:
                return "10x"  # 10x Cell Ranger v2 (one <genome> group per top-level key)
        if f.attrs.get("encoding-type") == "anndata" or ("obs" in f and "var" in f):
            return "h5ad"

    raise ConversionError(
        f"{p} is HDF5 but its layout is not recognised as h5ad or 10x. Pass --from h5ad|10x to override detection."
    )
