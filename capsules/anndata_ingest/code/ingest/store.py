"""The store a run appends to: an icechunk repo (one commit per run) or a plain AnnData zarr (in place).

Both kinds share one interface; they differ only in how they open, bootstrap, and finish a run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
import zarr.errors
from anndata.io import read_elem
from annizarr import AppConfig, Repo, append, rechunk

_PROBE = "_write_probe"
INGEST_ATTR = "ingest"  # a plain zarr's root attrs: {"sources": {sample_id: source file}}
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
        if fresh and _is_nonempty_dir(self.location):
            raise StoreError(f"{self.location} holds files but is not an icechunk repo; refusing to overwrite it. "
                             "Give an empty or new path (a plain AnnData zarr needs a .zarr path).")
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

    def recorded_sources(self) -> dict[str, str]:
        """Each ingested sample's source file, from the commits' metadata (the latest wins)."""
        sources: dict[str, str] = {}
        for snapshot in reversed(list(self.state.log())):
            sources.update((snapshot.metadata or {}).get("sources") or {})
        return sources


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
        self._record(metadata)

    def append(self, cells: str, cfg: AppConfig) -> None:
        append(self.path, cells=cells, cfg=cfg, drop_derived=True)

    def commit(self, message: str, metadata: dict[str, Any]) -> None:
        """A plain zarr has no commits; the samples' sources are kept in its root attrs instead."""
        self._record(metadata)

    def discard(self) -> None:
        return None

    def recorded_sources(self) -> dict[str, str]:
        root = self.root()
        return dict((root.attrs.get(INGEST_ATTR) or {}).get("sources") or {}) if root is not None else {}

    def _record(self, metadata: dict[str, Any]) -> None:
        """Adds the run's sample sources to the root attrs (annizarr's append keeps root attrs)."""
        root = zarr.open_group(self.path, mode="r+", use_consolidated=False)
        info = dict(root.attrs.get(INGEST_ATTR) or {})
        info["sources"] = {**(info.get("sources") or {}), **(metadata.get("sources") or {})}
        root.attrs[INGEST_ATTR] = info
        if zarr.open_group(self.path, mode="r").metadata.consolidated_metadata is not None:
            zarr.consolidate_metadata(self.path)


def open_store(path: str, origin: str = "", new_format: str = "auto") -> IcechunkStore | ZarrStore:
    """An existing store keeps its kind: a local icechunk repo, or a local folder holding a plain zarr
    group. Otherwise `new_format` picks the kind of the store to create: "icechunk", "zarr", or "auto"
    (a `.zarr` path is a plain AnnData zarr, anything else an icechunk repo)."""
    if not path:
        raise StoreError("No store given; set the store path (a data asset, local path, or s3:// URL).")
    if Path(path).suffix.lower() in (".h5", ".h5ad", ".h5mu", ".loom"):
        raise StoreError(f"{path} is a file; the store is an icechunk repo or a .zarr folder, which a file "
                         "can't be appended to. Give a new path inside /results to start a store.")
    if _is_local_repo(path):
        return IcechunkStore(path, origin)
    if _is_plain_zarr(path):
        return _zarr_store(path, origin)
    if new_format == "zarr" and "://" in path and _remote_repo_exists(path):
        return IcechunkStore(path, origin)  # an existing remote repo keeps its kind
    if new_format == "auto":
        new_format = "zarr" if path.rstrip("/").endswith(".zarr") else "icechunk"
    return _zarr_store(path, origin) if new_format == "zarr" else IcechunkStore(path, origin)


def _zarr_store(path: str, origin: str) -> ZarrStore:
    if "://" in path:
        raise StoreError(f"{path}: a plain zarr store must be local (annizarr writes remote stores as icechunk "
                         "only); on S3 choose the Icechunk format (and drop any .zarr suffix, e.g. "
                         "s3://bucket/prefix/atlas).")
    if origin:
        raise StoreError("origin applies to icechunk stores only; a plain zarr is written where it is.")
    return ZarrStore(path)


def _remote_repo_exists(path: str) -> bool:
    """Whether an icechunk repo is already there; unreachable counts as no (the zarr refusal follows)."""
    try:
        return Repo.exists(path)
    except Exception:
        return False


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


def _is_local_repo(path: str) -> bool:
    return "://" not in path and (Path(path) / "repo").is_file()


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
    if "sample_id" not in root["obs"]:
        raise StoreError("The store's obs has no sample_id column, which this capsule uses to skip samples it "
                         "already holds; add one (e.g. with annizarr) or start a new store.")
    return set(pd.Series(read_elem(root["obs/sample_id"])).astype(str).unique())


def var_names(root: zarr.Group) -> np.ndarray:
    var = root["var"]
    return np.asarray(read_elem(var[var.attrs["_index"]]), dtype=object)


def obs_columns(root: zarr.Group) -> list[str]:
    return list(root["obs"].attrs["column-order"])


def n_obs(root: zarr.Group) -> int:
    """Rows, from the obs index, which every AnnData has whether X is sparse or dense."""
    obs = root["obs"]
    index = obs[obs.attrs["_index"]]
    if isinstance(index, zarr.Group):  # nullable-string-array (values, mask) or categorical (codes, categories)
        index = index["codes"] if "codes" in index else index["values"]
    return int(index.shape[0])
