"""The store a run appends to: an icechunk repo (one commit per run) or a plain AnnData zarr (in place).

Both kinds share one interface; they differ only in how they open, bootstrap, and finish a run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import zarr
import zarr.errors
from anndata.io import read_elem
from annizarr import AppConfig, Repo, append, convert

_PROBE = "_write_probe"


class StoreError(RuntimeError):
    """The store's location or state doesn't allow this run."""


class IcechunkStore:
    kind = "icechunk"

    def __init__(self, path: str, origin: str = ""):
        self.path = path
        self.origin = origin if origin and origin != path else None
        self.repo: Repo | None = None

    @property
    def location(self) -> str:
        return self.origin or self.path

    def exists(self) -> bool:
        return Repo.exists(self.location)

    def open(self, create: bool) -> None:
        """Opens the repo; with `create`, makes it first unless an (empty) repo is already there."""
        fresh = create and not self.exists()
        self.repo = Repo.create(self.location) if fresh else Repo(self.path, origin=self.origin)
        if self.repo.origin is None:
            raise StoreError(f"{self.path} is read-only; give its writable origin (e.g. s3://bucket/prefix).")

    def root(self) -> zarr.Group | None:
        try:
            return self.repo.open_zarr("r")
        except zarr.errors.GroupNotFoundError:
            return None

    def working_root(self) -> zarr.Group:
        """The root as this run sees it, staged appends included."""
        return self.repo.open_zarr("w")

    def bootstrap(self, adata: ad.AnnData, cfg: AppConfig, message: str) -> str:
        return convert(adata, output=self.repo, cfg=cfg, message=message).snapshot_id

    def append(self, cells: str, cfg: AppConfig) -> None:
        append(self.repo, cells=cells, cfg=cfg, commit=False)

    def commit(self, message: str, metadata: dict[str, Any]) -> str:
        return self.repo.commit(message, metadata=metadata)

    def discard(self) -> None:
        self.repo.discard()


class ZarrStore:
    kind = "zarr"

    def __init__(self, path: str):
        self.path = path

    @property
    def location(self) -> str:
        return self.path

    def exists(self) -> bool:
        return self.root() is not None

    def open(self, create: bool) -> None:
        if not create:
            _probe_writable(self.path)
        elif not self.exists() and _is_nonempty_dir(self.path):
            raise StoreError(f"{self.path} holds files but is not a zarr store; refusing to overwrite it.")

    def root(self) -> zarr.Group | None:
        try:
            return zarr.open_group(self.path, mode="r")
        except (zarr.errors.GroupNotFoundError, FileNotFoundError):
            return None

    def working_root(self) -> zarr.Group:
        return self.root()

    def bootstrap(self, adata: ad.AnnData, cfg: AppConfig, message: str) -> None:
        convert(adata, output=self.path, cfg=cfg)

    def append(self, cells: str, cfg: AppConfig) -> None:
        append(self.path, cells=cells, cfg=cfg)

    def commit(self, message: str, metadata: dict[str, Any]) -> None:
        return None

    def discard(self) -> None:
        return None


def open_store(path: str, origin: str = "") -> IcechunkStore | ZarrStore:
    """`.zarr` paths are plain AnnData zarr stores; anything else is an icechunk repo."""
    if not path:
        raise StoreError("No store given; set the store path (a data asset, local path, or s3:// URL).")
    if path.rstrip("/").endswith(".zarr"):
        if origin:
            raise StoreError("origin applies to icechunk stores only; a plain zarr is written where it is.")
        return ZarrStore(path)
    return IcechunkStore(path, origin)


def _probe_writable(path: str) -> None:
    try:
        root = zarr.open_group(path, mode="r+")
        root.create_array(_PROBE, shape=(1,), dtype="u1", overwrite=True)
        del root[_PROBE]
    except Exception as exc:
        raise StoreError(f"{path} is not writable ({exc}); a plain zarr is appended in place.") from exc


def _is_nonempty_dir(path: str) -> bool:
    p = Path(path)
    return "://" not in path and p.is_dir() and any(p.iterdir())


def has_data(root: zarr.Group | None) -> bool:
    return root is not None and "X" in root


def sample_ids(root: zarr.Group) -> set[str]:
    return set(pd.Series(read_elem(root["obs/sample_id"])).astype(str).unique())


def var_names(root: zarr.Group) -> np.ndarray:
    var = root["var"]
    return np.asarray(read_elem(var[var.attrs["_index"]]), dtype=object)


def obs_columns(root: zarr.Group) -> list[str]:
    return list(root["obs"].attrs["column-order"])


def n_obs(root: zarr.Group) -> int:
    return int(root["X"].attrs["shape"][0])
