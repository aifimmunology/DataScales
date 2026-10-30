"""Where a run reads and writes its results.

- icechunk: one commit on the curation branch.
- writable plain zarr: in place.
- read-only plain zarr, or one given --output: a new zarr, either a full copy or everything but the counts.

A SpatialData store works the same way; only its table (`tables/<table>`) is read and written.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import annizarr
import numpy as np
import scipy.sparse as sp
import zarr
from anndata.io import read_elem, write_elem
from annizarr import Repo

from config import Settings
from steps import Embedding

_PROBE = ".write_probe"
_TARGET_SUM_ATTR = "target_sum"
DERIVED_OBSM = ("X_pca", "X_pca_harmony", "X_umap")


class StoreError(RuntimeError):
    """The input's location or state doesn't allow this run."""


class IcechunkTarget:
    kind = "icechunk"

    def __init__(self, settings: Settings):
        self.path = settings.input
        self.branch = settings.branch
        self.repo = Repo(self.path, branch=self.branch, origin=settings.origin or None)
        if self.repo.origin is None:
            raise StoreError(f"{self.path} is read-only; give its writable origin (e.g. s3://bucket/prefix).")
        self.repo.open_zarr("w")
        self.snapshot = self.repo.log()[0].id
        self.table = table_path(self.repo.open_zarr("r", snapshot_id=self.snapshot), settings.table, self.path)

    @property
    def location(self) -> str:
        return self.repo.origin or self.path

    def source(self) -> dict:
        return {"kind": self.kind, "location": self.location, "table": self.table, "branch": self.branch,
                "snapshot": self.snapshot}

    def read_root(self) -> zarr.Group:
        return subgroup(self.repo.open_zarr("r", snapshot_id=self.snapshot), self.table)

    def write(self, emb: Embedding, uns: dict, lognorm, settings: Settings) -> dict:
        n_cells = len(emb.leiden)
        tip = self.repo.log()[0].id
        if tip != self.snapshot:
            raise StoreError(f"Branch '{self.branch}' moved from {self.snapshot[:8]} to {tip[:8]} during the run; "
                             "nothing was written. Rerun on the new tip.")
        try:
            group = subgroup(self.repo.open_zarr("w"), self.table)
            write_lognorm(group, lognorm, settings.target_sum)
            write_embedding(group, emb, uns)
            snapshot = self.repo.commit(
                f"Process: {settings.integration}, {emb.n_clusters} leiden clusters over {n_cells:,} cells "
                f"(source {self.snapshot[:8]})",
                metadata={
                    "pipeline": "process",
                    "curation_branch": self.branch,
                    "source_snapshot": self.snapshot,
                    "table": self.table,
                    "integration": settings.integration,
                    "batch_keys": settings.integration_keys,
                    "n_cells": n_cells,
                    "n_clusters": emb.n_clusters,
                },
            )
        except BaseException:
            self.repo.discard()
            raise
        return {"branch": self.branch, "snapshot": snapshot}


class ZarrTarget:
    kind = "zarr"

    def __init__(self, settings: Settings):
        self.path = settings.input
        self.table = table_path(zarr.open_group(self.path, mode="r"), settings.table, self.path)
        copy = bool(settings.output) or settings.output_mode == "results-only"
        self.in_place = not copy and writable(self.path)
        self.dest = self.path if self.in_place else settings.output_path()
        self.mode = "in-place" if self.in_place else settings.output_mode
        if "://" in self.dest:
            raise StoreError(f"Output {self.dest} must be a local path or a /results folder.")
        if not self.in_place and exists(self.dest):
            raise StoreError(f"Output {self.dest} already exists; remove it or set --output to a new path.")
        self.fingerprint = fingerprint(self.read_root())

    @property
    def location(self) -> str:
        return self.path

    def source(self) -> dict:
        return {"kind": self.kind, "location": self.path, "table": self.table, "fingerprint": self.fingerprint}

    def read_root(self) -> zarr.Group:
        return subgroup(zarr.open_group(self.path, mode="r"), self.table)

    def write(self, emb: Embedding, uns: dict, lognorm, settings: Settings) -> dict:
        if fingerprint(self.read_root()) != self.fingerprint:
            raise StoreError(f"{self.path} changed during the run; nothing was written. Rerun on the new data.")
        if self.in_place:
            self._write_results(self.dest, self.table, emb, uns, lognorm, settings)
            return {"output": self.dest, "mode": self.mode}
        if exists(self.dest):
            raise StoreError(f"Output {self.dest} appeared during the run; nothing was written.")
        partial = f"{self.dest.rstrip('/')}.partial-{os.getpid()}"
        try:
            if self.mode == "results-only":
                copy_without_counts(self.read_root(), partial)
                table = ""
            elif self.table:
                copy_tree(self.path, partial)
                table = self.table
            else:
                annizarr.rechunk(self.path, output=partial)
                table = ""
            self._write_results(partial, table, emb, uns, lognorm, settings)
            os.rename(partial, self.dest)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        return {"output": self.dest, "mode": self.mode}

    def _write_results(self, path: str, table: str, emb: Embedding, uns: dict, lognorm, settings: Settings) -> None:
        unconsolidate(path, table)
        group = subgroup(zarr.open_group(path, mode="r+", use_consolidated=False), table)
        if "uns" in group and "process" in group["uns"]:
            del group["uns"]["process"]
        write_lognorm(group, lognorm, settings.target_sum)
        write_embedding(group, emb, uns)
        zarr.consolidate_metadata(path)


def open_target(settings: Settings) -> IcechunkTarget | ZarrTarget:
    """An icechunk repo if annizarr finds one at the input, otherwise a plain zarr."""
    if Repo.exists(settings.input):
        return IcechunkTarget(settings)
    if settings.origin:
        raise StoreError("origin applies to icechunk repos only; a plain zarr is written in place or copied.")
    if "://" in settings.input:
        raise StoreError("A remote plain zarr isn't supported; put it in an icechunk repo, or attach it as a data asset.")
    if not exists(settings.input):
        raise StoreError(f"No icechunk repo or zarr store at {settings.input}.")
    return ZarrTarget(settings)


def table_path(root: zarr.Group, table: str, where: str) -> str:
    """Where the AnnData is: the root of an AnnData store, or `tables/<table>` of a SpatialData store."""
    if "X" in root:
        return ""
    if "tables" in root and table in root["tables"]:
        return f"tables/{table}"
    raise StoreError(f"{where} has no X and no tables/{table}; it isn't an AnnData or SpatialData store.")


def subgroup(root: zarr.Group, path: str) -> zarr.Group:
    return root[path] if path else root


def exists(path: str) -> bool:
    try:
        zarr.open_group(path, mode="r")
    except (zarr.errors.GroupNotFoundError, FileNotFoundError):
        return Path(path).exists()
    return True


def writable(path: str) -> bool:
    try:
        root = zarr.open_group(path, mode="r+")
        root.create_array(_PROBE, shape=(1,), dtype="u1", overwrite=True)
        del root[_PROBE]
    except Exception:
        return False
    return True


def fingerprint(root: zarr.Group) -> str:
    """SHA-256 over the X and obs attrs, X's nonzero count, and the obs index (streamed by chunk).

    Leaves out `column-order`, which this pipeline rewrites.
    """
    h = hashlib.sha256()
    for node in (root["obs"], root["X"]):
        attrs = {k: v for k, v in node.attrs.items() if k != "column-order"}
        h.update(json.dumps(attrs, sort_keys=True, default=str).encode())
    h.update(str(int(root["X/indptr"][-1])).encode())
    index = root["obs"][root["obs"].attrs["_index"]]
    if isinstance(index, zarr.Group):
        index = index["values"]
    step = index.chunks[0]
    for start in range(0, index.shape[0], step):
        h.update("\x00".join(map(str, index[start:start + step])).encode())
    return h.hexdigest()


def copy_without_counts(src: zarr.Group, dest: str) -> None:
    """A new AnnData zarr holding everything in `src` but X, layers and raw."""
    out = zarr.open_group(dest, mode="w-", zarr_format=src.metadata.zarr_format, attributes=dict(src.attrs))
    for key in src:
        if key not in ("X", "layers", "raw"):
            write_elem(out, key, read_elem(src[key]))


def copy_tree(src: str, dest: str) -> None:
    """A writable copy of a local store, whatever the source's permissions."""
    shutil.copytree(src, dest, copy_function=shutil.copyfile)
    for folder, _, _ in os.walk(dest):
        os.chmod(folder, 0o755)


def unconsolidate(path: str, table: str = "") -> None:
    """Drops consolidated metadata from the root and every group down to `table`, so a write that dies
    midway can't leave a listing of deleted elements. The write consolidates the root again at the end."""
    folder = Path(path)
    for part in ["", *Path(table).parts]:
        folder = folder / part
        (folder / ".zmetadata").unlink(missing_ok=True)
        meta = folder / "zarr.json"
        doc = json.loads(meta.read_text()) if meta.exists() else {}
        if doc.pop("consolidated_metadata", None) is not None:
            tmp = meta.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, indent=2))
            os.replace(tmp, meta)


def write_lognorm(root: zarr.Group, lognorm, target_sum: float) -> None:
    """Streams the pipeline's GPU log-normalized matrix to `layers/lognorm`, one block at a time."""
    host = lognorm.map_blocks(lambda b: b.get(), dtype=lognorm.dtype,
                              meta=sp.csr_matrix((0, 0), dtype=lognorm.dtype))
    ensure_dict(root, "layers")
    write_elem(root["layers"], "lognorm", host)
    root["layers"]["lognorm"].attrs[_TARGET_SUM_ATTR] = float(target_sum)


def ensure_dict(root: zarr.Group, name: str) -> None:
    if name not in root:
        write_elem(root, name, {})


def _replace(group: zarr.Group, key: str, value) -> None:
    if key in group:
        del group[key]
    write_elem(group, key, value)


def write_embedding(root: zarr.Group, emb: Embedding, uns: dict[str, Any]) -> None:
    """Replaces every earlier `process` result with this run's; `uns/process` goes last, marking a complete write."""
    for name in ("obsm", "uns"):
        ensure_dict(root, name)
    for key, value in emb.obsm.items():
        _replace(root["obsm"], key, np.ascontiguousarray(value))
    _replace(root["obs"], "leiden", emb.leiden)
    _replace(root["var"], "highly_variable", np.asarray(emb.highly_variable, dtype=bool))
    for frame, column in (("obs", "leiden"), ("var", "highly_variable")):
        order = [c for c in root[frame].attrs.get("column-order", []) if c != column]
        root[frame].attrs["column-order"] = [*order, column]
    for key in set(DERIVED_OBSM) - set(emb.obsm):
        if key in root["obsm"]:
            del root["obsm"][key]
    _replace(root["uns"], "process", uns)
