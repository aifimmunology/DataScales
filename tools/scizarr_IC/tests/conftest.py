"""Shared fixtures: tiny anndata-style zarr stores built in tmp_path (hermetic)."""
from __future__ import annotations

import numpy as np
import pytest
import zarr


@pytest.fixture(autouse=True)
def _zarr_v3():
    zarr.config.set({"default_zarr_format": 3})
    yield


def _write_anndata_like(root: zarr.Group, x: np.ndarray) -> None:
    """Minimal anndata-readable dense store: root attrs + X + obs/var index groups."""
    n_obs, n_var = x.shape
    root.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
    xa = root.create_array("X", shape=x.shape, dtype=x.dtype, chunks=(max(1, n_obs // 2), n_var))
    xa[...] = x
    xa.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
    for name, n in (("obs", n_obs), ("var", n_var)):
        g = root.create_group(name)
        g.attrs.update(
            {"encoding-type": "dataframe", "encoding-version": "0.2.0",
             "_index": "_index", "column-order": []}
        )
        idx = g.create_array("_index", shape=(n,), dtype="int64", chunks=(n,))
        idx[...] = np.arange(n)
        idx.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})


@pytest.fixture
def src_zarr(tmp_path):
    """A tiny dense anndata-like zarr store; returns (path, X array)."""
    path = tmp_path / "src.zarr"
    x = np.arange(60, dtype="float32").reshape(12, 5)
    root = zarr.open_group(str(path), mode="w")
    _write_anndata_like(root, x)
    return path, x


@pytest.fixture
def repo_path(tmp_path):
    return tmp_path / "store.icechunk"
