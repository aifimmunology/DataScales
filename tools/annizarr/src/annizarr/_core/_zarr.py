from __future__ import annotations

from typing import Any, cast

import zarr
import zarr.errors

from annizarr.errors import ConversionError


def as_array(node: zarr.Array[Any] | zarr.Group) -> zarr.Array[Any]:
    if isinstance(node, zarr.Array):
        return node
    raise ConversionError(f"expected an array at '{node.path}', got a group")


def as_group(node: zarr.Array[Any] | zarr.Group) -> zarr.Group:
    if isinstance(node, zarr.Group):
        return node
    raise ConversionError(f"expected a group at '{node.path}', got an array")


def get_array(group: zarr.Group, key: str) -> zarr.Array[Any]:
    try:
        return group.get_array(key)
    except zarr.errors.BaseZarrError as exc:
        raise ConversionError(f"'{key}' is not an array in '{group.path or '/'}': {exc}") from exc


def get_group(group: zarr.Group, key: str) -> zarr.Group:
    try:
        return group.get_group(key)
    except zarr.errors.BaseZarrError as exc:
        raise ConversionError(f"'{key}' is not a group in '{group.path or '/'}': {exc}") from exc


# `.attrs` is a MutableMapping[str, JSON] (zarr.core.attributes.Attributes), so reading a
# specific, encoding-contract-known shape (a 2-tuple, a string, a string list) out of it is a
# cast, not a real union — the anndata-zarr encoding guarantees the shape, mypy cannot.
def shape_attr(node: zarr.Array[Any] | zarr.Group) -> tuple[int, int]:
    n_obs, n_vars = cast("list[int]", node.attrs["shape"])
    return int(n_obs), int(n_vars)


def str_attr(node: zarr.Array[Any] | zarr.Group, key: str) -> str:
    return cast(str, node.attrs[key])


def str_list_attr(node: zarr.Array[Any] | zarr.Group, key: str) -> list[str]:
    return cast("list[str]", node.attrs[key])
