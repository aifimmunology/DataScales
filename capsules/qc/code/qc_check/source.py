"""The input store, opened read-only: an AnnData icechunk repo read at `main`'s tip, or a plain zarr.

A plain zarr holds an AnnData at its root, or is a SpatialData store whose cell table (`tables/<name>`) is the
AnnData reviewed; its images, labels, points and shapes are never read."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import zarr
import zarr.errors
from anndata.io import read_elem
from annizarr import Repo
from annizarr.errors import AnzError

MAIN = "main"
PROBE = ".write_probe"
FINGERPRINT_ROWS = 1_000_000
TABLES = "tables"
SAMPLE = "sample_id"


class SourceError(RuntimeError):
    """The input can't be opened or read as this run needs."""


@dataclass
class Source:
    kind: str
    location: str
    store: zarr.Group
    table: str
    fingerprint: str
    repo: Repo | None = None

    @property
    def root(self) -> zarr.Group:
        """The AnnData group: the store root, or the SpatialData cell table."""
        return self.store[self.table] if self.table else self.store

    @property
    def spatial(self) -> bool:
        return bool(self.table)

    @cached_property
    def sample_column(self) -> str:
        """`sample_id`, or for a SpatialData table without one, the column naming each cell's region."""
        uns = self.root["uns"] if "uns" in self.root else {}
        if SAMPLE in self.obs_columns() or not self.spatial or "spatialdata_attrs" not in uns:
            return SAMPLE
        return str(read_elem(uns["spatialdata_attrs"]).get("region_key", SAMPLE))

    def coords(self) -> np.ndarray | None:
        """Cell centroids from `obsm/spatial`, when the store has them."""
        if "obsm" not in self.root or "spatial" not in self.root["obsm"]:
            return None
        return np.asarray(read_elem(self.root["obsm/spatial"]), dtype=np.float64)[:, :2]

    @property
    def stem(self) -> str:
        return PurePosixPath(self.location.rstrip("/")).name.removesuffix(".zarr")

    def obs_columns(self) -> list[str]:
        return list(self.root["obs"].attrs["column-order"])

    def var_names(self) -> np.ndarray:
        var = self.root["var"]
        return np.asarray(read_elem(var[var.attrs["_index"]]), dtype=object)

    def read_obs(self, columns: list[str], categorical: tuple[str, ...] = ()) -> pd.DataFrame:
        """Reads only `columns` of obs, one element at a time; `categorical` ones come back as category."""
        obs = self.root["obs"]
        data = {}
        for col in columns:
            values = pd.Series(read_elem(obs[col]))
            if col in categorical:
                values = values.astype("category")
                values = values.cat.rename_categories([str(c) for c in values.cat.categories])
            data[col] = values
        return pd.DataFrame(data)

    def curation(self, branch: str) -> dict | None:
        """The `uns/qc` entry at the tip of a curation branch, if the branch and the entry exist."""
        if self.repo is None or branch not in self.repo.branches():
            return None
        root = Repo(self.repo.path, branch=branch, anonymous=self.repo.anonymous).open_zarr("r")
        if "uns" not in root or "qc" not in root["uns"]:
            return None
        return read_elem(root["uns/qc"])


def open_source(path: str, origin: str = "", table: str = "") -> Source:
    if not path:
        raise SourceError("No input given; set the input (an icechunk repo or a plain AnnData .zarr).")
    try:
        is_repo = Repo.exists(path)
    except AnzError as exc:
        raise SourceError(f"Can't open {path}: {exc}") from exc
    return _open_icechunk(path, origin, table) if is_repo else _open_zarr(path, origin, table)


def _open_icechunk(path: str, origin: str, table: str) -> Source:
    try:
        repo = Repo(path, branch=MAIN, origin=origin or None)
    except AnzError as exc:
        raise SourceError(f"{path}: {exc}") from exc
    snapshot = repo.log()[0].id
    store = repo.open_zarr("r", snapshot_id=snapshot)
    if table_path(store, path, table):
        raise SourceError(f"{path}: SpatialData is supported as a plain .zarr only, not as an icechunk repo.")
    source = Source("icechunk", path, store, "", snapshot, repo)
    check_layout(source)
    return source


def _open_zarr(path: str, origin: str, table: str) -> Source:
    if "://" in path:
        raise SourceError(f"{path}: a plain .zarr must be local or a data asset; only icechunk repos can be on S3.")
    if origin:
        raise SourceError("origin applies to icechunk inputs only; a plain zarr is read where it is.")
    try:
        store = zarr.open_group(path, mode="r")
    except (zarr.errors.GroupNotFoundError, FileNotFoundError):
        raise SourceError(f"No icechunk repo or AnnData zarr at {path}.") from None
    source = Source("zarr", path, store, table_path(store, path, table), "")
    check_layout(source)
    source.fingerprint = zarr_fingerprint(source.root)
    return source


def table_path(store: zarr.Group, location: str, name: str) -> str:
    """'' for an AnnData store; `tables/<name>` for SpatialData (`table`, or its only table, unless named)."""
    if "X" in store or TABLES not in store:
        if name:
            raise SourceError(f"{location} is not a SpatialData store, so it has no table {name!r}.")
        return ""
    names = sorted(store[TABLES].group_keys())
    chosen = name or ("table" if "table" in names else names[0] if len(names) == 1 else "")
    if chosen not in names:
        raise SourceError(f"{location}: pick one of its tables with TABLE ({', '.join(names) or 'none found'}).")
    return f"{TABLES}/{chosen}"


def check_layout(source: Source) -> None:
    root, where = source.root, source.location + (f" ({source.table})" if source.table else "")
    missing = [k for k in ("X", "obs", "var") if k not in root]
    if missing:
        raise SourceError(f"{where} is not an AnnData store (missing {', '.join(missing)}).")
    encoding = root["X"].attrs.get("encoding-type")
    if encoding != "csr_matrix":
        raise SourceError(f"{where}: X must be CSR, found {encoding or 'a dense array'}.")
    if source.sample_column not in source.obs_columns():
        raise SourceError(f"{where}: obs has no {source.sample_column} column to group cells by sample.")


def zarr_fingerprint(root: zarr.Group) -> str:
    """SHA-256 over the root, obs and X attrs, the X shape and nonzero count, and the obs index."""
    digest = hashlib.sha256()
    for attrs in (root.attrs, root["obs"].attrs, root["X"].attrs):
        digest.update(json.dumps(dict(attrs), sort_keys=True, default=str).encode())
    digest.update(json.dumps(root["X"].attrs["shape"]).encode())
    digest.update(str(int(root["X/indptr"][-1])).encode())
    index = root["obs"][root["obs"].attrs["_index"]]
    if isinstance(index, zarr.Group):
        index = index["values"]
    step = index.chunks[0] * max(1, FINGERPRINT_ROWS // index.chunks[0])
    for start in range(0, index.shape[0], step):
        digest.update("\0".join(map(str, index[start:start + step])).encode())
        digest.update(b"\1")
    return digest.hexdigest()


def probe_writable(path: str) -> None:
    """Writes and deletes a small key at `path`'s root; raises SourceError when that fails."""
    try:
        root = zarr.open_group(path, mode="a")
        root.create_array(PROBE, shape=(1,), dtype="u1", overwrite=True)
        del root[PROBE]
    except Exception as exc:
        raise SourceError(f"{path} is not writable ({exc}); give a writable output path (local or s3://).") from exc

