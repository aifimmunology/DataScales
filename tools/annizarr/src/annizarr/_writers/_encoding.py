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
    import anndata as ad  # restored on exit; makes anndata's own write_elem pass shards="auto" too

    previous = ad.settings.auto_shard_zarr_v3
    ad.settings.auto_shard_zarr_v3 = auto_shard
    try:
        yield
    finally:
        ad.settings.auto_shard_zarr_v3 = previous


@contextmanager
def suppress_autoshard_warning(enabled: bool) -> Iterator[None]:
    # mirrors anndata's own decorator, for raw require_array calls it never sees
    if not enabled:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Automatic shard shape inference is experimental", UserWarning)
        yield


def sparse_shards(auto_shard: bool) -> Any:
    return "auto" if auto_shard else None


def make_sparse_group(parent: zarr.Group, key: str, *, csr: bool, shape: tuple[int, int]) -> zarr.Group:
    group = parent.require_group(key)
    group.attrs["encoding-type"] = CSR_ENCODING_TYPE if csr else CSC_ENCODING_TYPE
    group.attrs["encoding-version"] = SPARSE_ENCODING_VERSION
    group.attrs["shape"] = list(shape)
    return group
