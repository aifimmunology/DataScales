"""Shared fixtures: tiny anndata-style zarr stores and a simulated read-only mount.

Everything is hermetic (tmp_path). The "mount" imitates a Code Ocean data asset: a
plain copy of an origin repo that SCIZARR_IC_READONLY_PREFIXES marks read-only (tests
run as root, so chmod can't). The copy never sees the origin's later writes, so it also
doubles as a *stale* mount. ``test_live_codeocean.py`` exercises the real thing.
"""
from __future__ import annotations

import shutil
from pathlib import Path

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


def snapshot_tree(root: Path) -> dict[str, int]:
    """{relative file: mtime_ns} — to prove a read-only location was not touched."""
    return {
        str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()
    }


@pytest.fixture
def mounted(src_zarr, tmp_path, monkeypatch):
    """(mount_path, origin_path, mount_tree_before) with HEAD sidecars in tmp_path/home."""
    from scizarr_ic import Repo

    path, _ = src_zarr
    origin = tmp_path / "origin.icechunk"
    Repo.init(path, origin, message="import")
    (origin / "scizarr_head").unlink()  # the copy must not carry the origin's HEAD file

    mount_root = tmp_path / "mount"
    mount = mount_root / "store"
    shutil.copytree(origin, mount)
    monkeypatch.setenv("SCIZARR_IC_READONLY_PREFIXES", str(mount_root))
    monkeypatch.setenv("SCIZARR_IC_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SCIZARR_IC_ORIGIN", raising=False)
    return mount, origin, snapshot_tree(mount)
