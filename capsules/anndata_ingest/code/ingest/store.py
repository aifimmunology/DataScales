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
from annizarr import AppConfig, Repo, append, rechunk

_PROBE = "_write_probe"
RESULT_UNS = ("process",)


class StoreError(RuntimeError):
    """The store's location or state doesn't allow this run."""


class _Repo(Repo):
    """A Repo whose next commit, made inside an annizarr op, carries the run's message and metadata."""
    next_commit: tuple[str, dict[str, Any]] | None = None

    def commit(self, message: str, *, metadata: dict[str, Any] | None = None, allow_empty: bool = False) -> str:
        if self.next_commit:
            (message, metadata), self.next_commit = self.next_commit, None
        return super().commit(message, metadata=metadata, allow_empty=allow_empty)


class IcechunkStore:
    kind = "icechunk"

    def __init__(self, path: str, origin: str = ""):
        self.path = path
        self.origin = origin if origin and origin != path else None
        self.repo: _Repo | None = None
        self.state: Repo | None = None

    @property
    def location(self) -> str:
        return self.origin or self.path

    def exists(self) -> bool:
        return Repo.exists(self.location)

    def open(self, create: bool) -> None:
        """Opens the repo; with `create`, makes it first unless an (empty) repo is already there."""
        fresh = create and not self.exists()
        self.repo = _Repo.create(self.location) if fresh else _Repo(self.path, origin=self.origin)
        if self.repo.origin is None:
            raise StoreError(f"{self.path} is read-only; give its writable origin (e.g. s3://bucket/prefix).")
        # A data-asset mount can lag its origin by several commits, so the store's state (which
        # samples it holds, its schema and genes) is read from the origin.
        self.state = Repo(self.repo.origin) if self.repo.origin != self.path else self.repo

    def root(self) -> zarr.Group | None:
        try:
            return self.state.open_zarr("r")
        except zarr.errors.GroupNotFoundError:
            return None

    def working_root(self) -> zarr.Group:
        """The root as this run sees it, staged appends included."""
        return self.repo.open_zarr("w")

    def bootstrap(self, staged: Path, cfg: AppConfig, message: str, metadata: dict[str, Any]) -> str:
        self.repo.next_commit = (message, metadata)
        try:
            return rechunk(str(staged), output=self.repo, cfg=cfg).snapshot_id
        finally:
            self.repo.next_commit = None

    def append(self, cells: str, cfg: AppConfig) -> None:
        """Derived elements are dropped (the run has warned and records them). The plan reads the open
        writable session, so the origin rather than a mount that may lag it."""
        self.working_root()
        append(self.repo, cells=cells, cfg=cfg, commit=False, drop_derived=True)

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
        return zarr.open_group(self.path, mode="r+")

    def bootstrap(self, staged: Path, cfg: AppConfig, message: str, metadata: dict[str, Any]) -> None:
        rechunk(str(staged), output=self.path, cfg=cfg)

    def append(self, cells: str, cfg: AppConfig) -> None:
        append(self.path, cells=cells, cfg=cfg, drop_derived=True)

    def commit(self, message: str, metadata: dict[str, Any]) -> None:
        return None

    def discard(self) -> None:
        return None


def open_store(path: str, origin: str = "") -> IcechunkStore | ZarrStore:
    """`.zarr` paths and local folders holding a plain zarr group are AnnData zarr stores; anything
    else is an icechunk repo."""
    if not path:
        raise StoreError("No store given; set the store path (a data asset, local path, or s3:// URL).")
    if Path(path).suffix.lower() in (".h5", ".h5ad", ".h5mu", ".loom"):
        raise StoreError(f"{path} is a file; the store is an icechunk repo or a .zarr folder, which a file "
                         "can't be appended to. Give a new path inside /results to start a store.")
    if path.rstrip("/").endswith(".zarr") or _is_plain_zarr(path):
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


def _is_plain_zarr(path: str) -> bool:
    p = Path(path)
    return "://" not in path and p.is_dir() and not (p / "repo").exists() and (
        (p / "zarr.json").is_file() or (p / ".zgroup").is_file())


def derived_elements(root: zarr.Group | None) -> list[str]:
    """Elements built for the current cells (embeddings, graphs, layers, raw, process results); appending
    invalidates them."""
    if root is None:
        return []
    found = [f"{group}/{key}" for group in ("obsm", "obsp", "layers") if group in root for key in root[group]]
    found += ["raw"] if "raw" in root else []
    return found + [f"uns/{key}" for key in RESULT_UNS if "uns" in root and key in root["uns"]]


def drop_unappendable(root: zarr.Group) -> None:
    """Deletes what annizarr's append neither extends nor drops itself: raw and the process results."""
    if "raw" in root:
        del root["raw"]
    for key in RESULT_UNS:
        if "uns" in root and key in root["uns"]:
            del root["uns"][key]


def _x_values(root: zarr.Group) -> zarr.Array:
    x = root["X"]
    return x["data"] if isinstance(x, zarr.Group) else x


def x_dtype(root: zarr.Group) -> np.dtype:
    return np.dtype(_x_values(root).dtype)


def x_holds_counts(root: zarr.Group, n: int = 200_000) -> bool:
    """Whether X looks like counts: its first `n` stored values are non-negative whole numbers."""
    data = _x_values(root)
    values = np.asarray(data[:n] if data.ndim == 1 else data[:100])
    return bool(np.all(values >= 0) and np.all(np.mod(values, 1) == 0))


_ENCODED = {"categorical": "category", "nullable-integer": "Int64", "nullable-boolean": "boolean",
            "string-array": "str", "nullable-string-array": "str"}


def obs_kinds(root: zarr.Group) -> dict[str, str]:
    """Each obs column's kind (category, Int64, boolean, str, or a numpy dtype), from metadata alone."""
    obs = root["obs"]
    return {c: _ENCODED.get(obs[c].attrs.get("encoding-type"), str(getattr(obs[c], "dtype", "str")))
            for c in obs.attrs["column-order"]}


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
