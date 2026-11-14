"""Opening the input and saving results, by the notebooks' rules.

`child` to `save_commit` are the notebooks' helper cell, word for word (tests/test_store.py checks it).
- zarr input (AnnData, or a SpatialData store's tables/<table>): a new full copy built at `<out>.partial` and
  renamed; an existing output is never overwritten, the next free `<stem>-<n>.zarr` is used.
- icechunk input: one commit on the branch, through the origin when given; refused only if the branch moved.
outputs.py decides which of these a run uses, and where.
"""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import annizarr
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata.io import write_elem
from annizarr import Repo

from config import EMBEDDING_KEYS, LEIDEN_DEFAULT

EMBEDDINGS = EMBEDDING_KEYS  # obsm keys a run replaces or removes, besides its UMAP
ACCESS_WORDS = ("403", "forbidden", "access denied", "accessdenied", "credentials", "not authorized", "unauthorized")


def leiden_pattern(base: str = LEIDEN_DEFAULT) -> re.Pattern:
    """The columns a run with leiden_key `base` owns: <base> and <base>_<resolution>."""
    return re.compile(rf"^{re.escape(base)}(_[0-9.eE+-]+)?$")


LEIDEN_COLUMN = leiden_pattern()


class InputError(RuntimeError):
    pass


class BranchMoved(InputError):
    """Someone committed on the icechunk branch while the run was computing, so the results no longer match it.

    save_commit is the notebooks' helper word for word and raises a plain RuntimeError; outputs.save turns it into this.
    """

    TEXT = "moved since the store was opened"


def is_access_error(exc: BaseException) -> bool:
    """A 403 / missing credentials from S3 (s3fs, botocore and icechunk word it differently)."""
    seen, todo = set(), [exc]
    while todo:
        e = todo.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        code = str(getattr(e, "response", {}).get("Error", {}).get("Code", "")) if isinstance(getattr(e, "response", None), dict) else ""
        if (isinstance(e, PermissionError) or code in ("403", "AccessDenied", "InvalidAccessKeyId")
                or any(w in f"{type(e).__name__} {e}".lower() for w in ACCESS_WORDS)):
            return True
        todo += [e.__cause__, e.__context__]
    return False


def access_hint(location: str, writing: bool) -> str:
    """One line for a refused S3 access, saying what to attach."""
    need = "write" if writing else "read"
    return (f"no {need} access to {location}: attach an AWS secret with {need} permission for it "
            "to the capsule (Environment > Secrets > AWS), then run again")


@dataclass
class Source:
    path: str
    table: str
    snapshot: str
    root: zarr.Group
    repo: Repo | None = None

    @property
    def group(self) -> zarr.Group:
        return self.root[self.table] if self.table else self.root


def open_source(path: str, table: str, branch: str, origin: str) -> Source:
    try:
        if Repo.exists(path):
            repo = Repo(path, branch=branch, origin=origin or None)
            view = Repo(origin, branch=branch) if origin else repo
            snapshot = view.log()[0].id
            root = view.open_zarr("r", snapshot_id=snapshot)
        else:
            repo, root, snapshot = None, zarr.open_group(path, mode="r"), "zarr"
    except Exception as exc:
        if "://" in path and is_access_error(exc):
            raise InputError(access_hint(path, writing=False)) from None
        raise InputError(f"can't open {path} as an AnnData or SpatialData .zarr or an icechunk repo "
                         f"({type(exc).__name__}: {exc})") from None
    if origin and not repo:
        print(f"origin {origin} is for icechunk repos; {path} is a zarr, so it's ignored")
    table_path = "" if "X" in root else f"tables/{table}"
    if table_path and table_path not in root:
        raise InputError(f"{path} has no X and no {table_path}; it isn't an AnnData or SpatialData store")
    return Source(path, table_path, snapshot, root, repo)


def child(group, key):
    if key not in group:
        write_elem(group, key, {})
    return group[key]


def put(group, key, value):
    if key in group:
        del group[key]
    write_elem(group, key, value)


def put_column(frame, name, values):
    put(frame, name, values)
    frame.attrs["column-order"] = [*(c for c in frame.attrs["column-order"] if c != name), name]


def free_path(stem, suffix=""):
    out, n = f"{stem}{suffix}", 0
    while Path(out).exists() or Path(f"{out}.partial").exists():
        n += 1
        out = f"{stem}-{n}{suffix}"
    return out


def new_output(store, output, results):
    out = output or f"{results}/{Path(store.rstrip('/')).name.removesuffix('.zarr')}.processed"
    return free_path(out.rstrip("/").removesuffix(".zarr"), ".zarr")


def save_copy(store, table, write_results, output, results):
    out = new_output(store, output, results)
    partial = f"{out}.partial"
    Path(partial).parent.mkdir(parents=True, exist_ok=True)
    if table:
        shutil.copytree(store, partial, copy_function=shutil.copyfile)
        for folder, _, _ in os.walk(partial):
            os.chmod(folder, 0o755)  # a read-only data asset's folders would be copied read-only
    else:
        annizarr.rechunk(store, output=partial)
    copy = zarr.open_group(partial, mode="r+", use_consolidated=False)
    write_results(copy[table] if table else copy)
    zarr.consolidate_metadata(partial)
    os.rename(partial, out)
    return out


def save_commit(repo, branch, snapshot, table, write_results, message, metadata=None):
    store = repo.open_zarr("w")
    if repo.log()[0].id != snapshot:
        repo.discard()
        raise RuntimeError(f"Branch {branch} moved since the store was opened; nothing was written. Reopen it and rerun.")
    try:
        write_results(store[table] if table else store)
        return repo.commit(message, metadata={"source_snapshot": snapshot, "table": table, **(metadata or {})})
    except BaseException:
        repo.discard()
        raise


def drop_column(frame, name):
    del frame[name]
    frame.attrs["column-order"] = [c for c in frame.attrs["column-order"] if c != name]


@dataclass
class Results:
    lognorm: object
    target_sum: float
    highly_variable: np.ndarray
    obsm: dict[str, np.ndarray]
    obs: dict[str, pd.Categorical]
    uns: dict = field(default_factory=dict)
    leiden_key: str = LEIDEN_DEFAULT  # obs columns named <leiden_key>[_<res>] from an earlier run are dropped

    def write(self, group) -> None:
        host = self.lognorm.map_blocks(lambda b: b.get(), dtype=self.lognorm.dtype,
                                       meta=sp.csr_matrix((0, 0), dtype=self.lognorm.dtype))
        put(child(group, "layers"), "lognorm", host)
        group["layers/lognorm"].attrs["target_sum"] = self.target_sum
        obsm = child(group, "obsm")
        for key in dict.fromkeys((*EMBEDDINGS, *self.obsm)):
            if key in self.obsm:
                put(obsm, key, np.ascontiguousarray(self.obsm[key], dtype=np.float32))
            elif key in obsm:
                del obsm[key]
        obs, owned = group["obs"], leiden_pattern(self.leiden_key)
        for name in [c for c in obs.attrs["column-order"] if owned.match(c) and c not in self.obs]:
            drop_column(obs, name)
        for name, values in self.obs.items():
            put_column(obs, name, values)
        put_column(group["var"], "highly_variable", self.highly_variable)
        put(child(group, "uns"), "process", {**self.uns, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
