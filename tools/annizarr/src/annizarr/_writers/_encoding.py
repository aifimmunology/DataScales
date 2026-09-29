from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anndata._io.specs import write_elem  # private API; verified against .claude/vendor/anndata/src (0.12.19)

if TYPE_CHECKING:
    import zarr

__all__ = ["write_elem"]

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


def make_sparse_group(parent: zarr.Group, key: str, *, csr: bool, shape: tuple[int, int]) -> zarr.Group:
    # callers still create the data/indices/indptr child arrays themselves
    group = parent.require_group(key)
    group.attrs["encoding-type"] = CSR_ENCODING_TYPE if csr else CSC_ENCODING_TYPE
    group.attrs["encoding-version"] = SPARSE_ENCODING_VERSION
    group.attrs["shape"] = list(shape)
    return group
