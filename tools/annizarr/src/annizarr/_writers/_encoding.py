from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from anndata._io.specs import write_elem  # private API; verified against .claude/vendor/anndata/src (0.12.19)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import zarr

__all__ = ["autoshard_setting", "sparse_shards", "suppress_autoshard_warning", "write_elem"]

ARRAY_ENCODING_TYPE = "array"
ARRAY_ENCODING_VERSION = "0.2.0"
ANNDATA_ENCODING_TYPE = "anndata"
ANNDATA_ENCODING_VERSION = "0.1.0"
RAW_ENCODING_TYPE = "raw"
RAW_ENCODING_VERSION = "0.1.0"
CSR_ENCODING_TYPE = "csr_matrix"
CSC_ENCODING_TYPE = "csc_matrix"
SPARSE_ENCODING_VERSION = "0.1.0"


def set_array_attrs(arr: zarr.Array[Any]) -> None:
    arr.attrs["encoding-type"] = ARRAY_ENCODING_TYPE
    arr.attrs["encoding-version"] = ARRAY_ENCODING_VERSION


def set_anndata_root_attrs(group: zarr.Group) -> None:
    group.attrs["encoding-type"] = ANNDATA_ENCODING_TYPE
    group.attrs["encoding-version"] = ANNDATA_ENCODING_VERSION


def set_raw_group_attrs(group: zarr.Group) -> None:
    group.attrs["encoding-type"] = RAW_ENCODING_TYPE
    group.attrs["encoding-version"] = RAW_ENCODING_VERSION


@contextmanager
def autoshard_setting(auto_shard: bool) -> Iterator[None]:
    """Temporarily set ``ad.settings.auto_shard_zarr_v3`` around our own ``write_elem`` calls.

    Restored on exit (even on error), so the setting never leaks to the caller's process.
    With this set, anndata's own ``write_elem`` passes ``shards="auto"`` to every array it
    writes (obs/var columns, obsm, uns, …) and suppresses zarr's "automatic shard shape
    inference is experimental" warning itself (``zarr_v3_sharding``/``suppress_autoshard_warning``
    in ``anndata._io.specs.methods``) — nothing further to do here for those calls.

    Parameters
    ----------
    auto_shard
        Value to set. ``False`` is an explicit value, distinct from anndata's unset default
        (``None``): either explicit value silences anndata's "autosharding will be the default"
        warning, and ``False`` reproduces the pre-autoshard on-disk layout exactly.
    """
    import anndata as ad

    previous = ad.settings.auto_shard_zarr_v3
    ad.settings.auto_shard_zarr_v3 = auto_shard
    try:
        yield
    finally:
        ad.settings.auto_shard_zarr_v3 = previous


@contextmanager
def suppress_autoshard_warning(enabled: bool) -> Iterator[None]:
    """Suppress zarr's "shard shape inference is experimental" warning, mirroring anndata's
    own ``suppress_autoshard_warning`` decorator, around OUR OWN ``shards="auto"`` array
    creations (the 1-D sparse ``data``/``indices`` arrays we create directly via
    ``zarr.Group.require_array`` — anndata's ``write_elem`` is not involved, so anndata's
    decorator never sees these calls).

    Parameters
    ----------
    enabled
        No-op when ``False`` (an explicit ``shards=`` — or none — should still warn if it
        legitimately would).
    """
    if not enabled:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Automatic shard shape inference is experimental", UserWarning)
        yield


def sparse_shards(auto_shard: bool) -> Any:
    """The ``shards=`` kwarg for a 1-D sparse ``data``/``indices`` array: ``"auto"`` when
    ``auto_shard``, else ``None`` (unsharded). Dense X/layers are never auto-sharded — they
    keep the explicit ``x_shard_factor`` (see ``_layout.dense_shards``)."""
    return "auto" if auto_shard else None


def make_sparse_group(parent: zarr.Group, key: str, *, csr: bool, shape: tuple[int, int]) -> zarr.Group:
    # callers still create the data/indices/indptr child arrays themselves
    group = parent.require_group(key)
    group.attrs["encoding-type"] = CSR_ENCODING_TYPE if csr else CSC_ENCODING_TYPE
    group.attrs["encoding-version"] = SPARSE_ENCODING_VERSION
    group.attrs["shape"] = list(shape)
    return group
