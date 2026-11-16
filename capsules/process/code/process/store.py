"""Opening the input, deciding where results go, and saving them.

Input: open_source; an AnnData zarr, a SpatialData zarr (tables/<table>) or an icechunk repo.
`child` to `save_commit` are the notebooks' helper cell, word for word (tests/test_store.py checks it).
Output (`plan`, `check`, `save`): decided and checked before any GPU work, checked again and written at the end.

One destination per run, from `output` and `output_format` ("auto": the input's format for a copy, what is there
for anything else):
- a copy in results_dir: no output, a bare name, or a path under results_dir. An existing name is never reused:
  <name>-1, -2, ...
- an output location (s3:// or any other path), e.g. an icechunk origin:
  - icechunk input: the location must be the input's own repo (it holds the input's latest snapshot on `branch`).
    The data is read from there and one commit is made on `branch`. Anything else is refused as a likely typo.
  - zarr input, nothing at the location: a new store, zarr unless `output_format` says icechunk.
  - zarr input, an existing zarr with the same layout, cells and genes: the results are written into it in place.
  `output_format`, when not "auto", must match what's already at the location.

A location being written over S3 (no atomic rename there) carries a `.process-writing` marker until done:
"new" means its data is incomplete and it's refused; "results" means only the results may be partial and a rerun
in place fixes them. Local copies are built at `<out>.partial` and renamed instead.
"""
from __future__ import annotations

import os
import re
import shutil
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import annizarr
import fsspec
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata.io import read_elem, write_elem
from annizarr import Repo

try:
    from annizarr.ic._copy import copy_group
except ImportError as exc:
    raise ImportError("annizarr no longer has ic._copy.copy_group (the one private API used); pin annizarr==0.2.4") from exc

from config import EMBEDDING_KEYS, FORMATS, LEIDEN_DEFAULT

ACCESS_WORDS = ("(403)", "forbidden", "access denied", "accessdenied", "credentials", "not authorized", "unauthorized")


def leiden_pattern(base: str = LEIDEN_DEFAULT) -> re.Pattern:
    """The columns a run with leiden_key `base` owns: <base> and <base>_<resolution>."""
    return re.compile(rf"^{re.escape(base)}(_[0-9.eE+-]+)?$")


LEIDEN_COLUMN = leiden_pattern()


class InputError(RuntimeError):
    pass


class BranchMoved(InputError):
    """Someone committed on the icechunk branch while the run was computing, so the results no longer match it.

    save_commit is the notebooks' helper word for word and raises a plain RuntimeError; save turns it into this.
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
        for key in dict.fromkeys((*EMBEDDING_KEYS, *self.obsm)):
            if key in self.obsm:
                put(obsm, key, np.ascontiguousarray(self.obsm[key], dtype=np.float32))
            elif key in obsm:
                del obsm[key]
        obs, owned = group["obs"], leiden_pattern(self.leiden_key)
        mine = set(read_elem(group["uns/process/leiden"])) if "uns/process/leiden" in group else set()
        for name in [c for c in obs.attrs["column-order"]
                     if owned.match(c) and c not in self.obs and (c in mine or self.leiden_key == LEIDEN_DEFAULT)]:
            drop_column(obs, name)
        for name, values in self.obs.items():
            put_column(obs, name, values)
        put_column(group["var"], "highly_variable", self.highly_variable)
        put(child(group, "uns"), "process", {**self.uns, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})



MARKER = ".process-writing"
FIRST_COMMIT = "1CECHNKREP0F1RSTCMT0"  # every icechunk repo starts with this snapshot: it proves nothing


class OutputError(InputError):
    pass


@dataclass(frozen=True)
class Destination:
    mode: str       # "copy" (into results_dir), "new", "in_place" or "commit" (external)
    kind: str       # "zarr" or "icechunk"
    location: str   # for "copy", the name before -<n> and .zarr are added
    pending: str = ""

    def describe(self) -> str:
        what = {"copy": "a new copy in", "new": "a new store at", "in_place": "written into", "commit": "a commit on"}
        return f"{what[self.mode]} {self.kind} {self.location}"


def is_remote(location: str) -> bool:
    return "://" in location


def detect(location: str) -> tuple[str, str]:
    """(kind, marker): kind is "missing", "zarr", "icechunk" or "other"; marker is the .process-writing content."""
    try:
        return _detect(location)
    except OutputError:
        raise
    except Exception as exc:
        if is_remote(location) and is_access_error(exc):
            raise OutputError(access_hint(location, writing=True)) from None
        raise


def _detect(location: str) -> tuple[str, str]:
    fs, path = fsspec.core.url_to_fs(location)
    fs.invalidate_cache()  # s3fs answers exists() from listings cached hours ago, at plan time
    try:
        marker = fs.cat_file(f"{path}/{MARKER}").decode().strip()
    except (FileNotFoundError, NotADirectoryError):
        marker = ""
    try:
        if Repo.exists(location):
            return "icechunk", marker
    except Exception as exc:
        if is_remote(location):
            if is_access_error(exc):
                raise OutputError(access_hint(location, writing=True)) from None
            raise OutputError(f"can't check {location} ({type(exc).__name__}: {exc})") from None
    if not fs.exists(path) or not fs.ls(path, detail=False):
        return "missing", marker
    if fs.isfile(path):
        return "other", marker
    return ("zarr" if fs.exists(f"{path}/zarr.json") or fs.exists(f"{path}/.zgroup") else "other"), marker


def probe_write(location: str) -> None:
    """Remote locations only: put and delete one empty object, so missing write or delete permission stops the run
    now, before the GPU work, instead of after it. The store's own marker needs both."""
    if not is_remote(location):
        return
    fs, path = fsspec.core.url_to_fs(location)
    probe = f"{path}/.process-write-check"
    try:
        fs.pipe_file(probe, b"")
        fs.rm_file(probe)
    except Exception as exc:
        if is_access_error(exc):
            raise OutputError(access_hint(location, writing=True)) from None
        raise OutputError(f"can't write to {location} ({type(exc).__name__}: {exc})") from None


def input_kind(path: str) -> str:
    try:
        return "icechunk" if Repo.exists(path) else "zarr"
    except Exception as exc:
        if is_remote(path) and is_access_error(exc):
            raise InputError(access_hint(path, writing=False)) from None
        return "zarr"


def _stem(path: str) -> str:
    return re.sub(r"\.(zarr|icechunk)$", "", Path(path.rstrip("/")).name)


def copy_location(cfg) -> str | None:
    """Where a copy in results_dir goes: no output, a bare name, or a path under results_dir. None: an external output."""
    out, results = cfg.output.strip().rstrip("/"), cfg.results_dir.rstrip("/")
    if not out:
        return f"{results}/{_stem(cfg.input)}.processed"
    if is_remote(out):
        return None
    if not os.path.isabs(out):
        return f"{results}/{out}"
    root, here = os.path.realpath(results), os.path.realpath(out)
    return out if here != root and os.path.commonpath([here, root]) == root else None


def plan(cfg) -> Destination:
    """The destination of this run, refused here with a one-line OutputError when it can't work."""
    source_kind = input_kind(cfg.input)
    copy = copy_location(cfg)
    if copy is not None:
        kind = source_kind if cfg.output_format == "auto" else cfg.output_format
        return Destination("copy", kind, copy.removesuffix(".zarr"))

    location, wanted = cfg.output.strip().rstrip("/"), cfg.output_format
    same = (location == cfg.input.rstrip("/")) if is_remote(location) else os.path.realpath(location) == os.path.realpath(cfg.input)
    if same:
        raise OutputError("the output is the same as the input. The input is never changed, so choose another "
                          "Output location, or leave it as is to save a copy in /results")
    kind, marker = detect(location)
    if marker == "new":
        raise OutputError(f"{location} was left unfinished by an earlier run that stopped. Delete it (it holds a file "
                          f"named {MARKER}), or choose another Output location")
    if kind == "other":
        raise OutputError(f"output location {location} holds something that isn't a zarr or an icechunk repo")
    if kind != "missing" and wanted != "auto" and wanted != kind:
        raise OutputError(f"output location {location} already holds a {kind} store, but Output format is {wanted}: "
                          f"set it to auto or {kind}, or choose another Output location")
    if not is_remote(location):
        if os.path.exists(f"{location}.partial"):
            raise OutputError(f"{location}.partial was left by an earlier run that stopped; delete it and rerun")
        parent = next(p for p in Path(location).absolute().parents if p.exists()) if kind == "missing" else Path(location)
        if not os.access(parent, os.W_OK):
            raise OutputError(f"output location {location} can't be written ({parent} is read-only)")
    probe_write(location)
    if source_kind == "icechunk":
        if kind != "icechunk":
            found = "nothing there" if kind == "missing" else "a zarr there"
            raise OutputError(f"the input is an icechunk repo, so the output must be that same repo's writable address, "
                              f"but there's {found} at {location}: check it for typos")
        _check_same_repo(cfg.input, location, cfg.branch)
        return Destination("commit", "icechunk", location)
    if kind == "icechunk":
        raise OutputError(f"{location} is an icechunk repo but the input is a zarr: choose another Output location")
    if kind == "zarr":
        return Destination("in_place", "zarr", location, marker)
    return Destination("new", "zarr" if wanted == "auto" else wanted, location)


def _check_same_repo(input_path: str, location: str, branch: str) -> None:
    try:
        external = {s.id for s in Repo(location, branch=branch).log()}
    except Exception as exc:
        if is_remote(location) and is_access_error(exc):
            raise OutputError(access_hint(location, writing=True)) from None
        raise OutputError(f"can't use branch {branch} of the output repo {location} ({type(exc).__name__}: {exc})") from None
    try:
        tip = Repo(input_path, branch=branch).log()[0].id
    except Exception as exc:
        raise OutputError(f"the input has no branch {branch} ({type(exc).__name__}: {exc})") from None
    if tip == FIRST_COMMIT:
        raise OutputError(f"the input's branch {branch} has no commits with data, so it can't be matched to {location}")
    if tip not in external:
        raise OutputError(f"{location} is not the repo the input was read from (its latest version on branch {branch} "
                          f"differs): check the address and the Icechunk branch")


def check(dest: Destination, src: Source) -> None:
    """The checks that need the opened input."""
    if dest.kind == "icechunk" and dest.mode in ("copy", "new") and not src.repo and src.table:
        raise OutputError("a SpatialData zarr can't be stored as icechunk (its shapes and points are parquet); "
                          "pick zarr")
    if dest.mode in ("copy", "new") and not src.repo and is_remote(src.path):
        raise OutputError(f"a zarr input on {src.path.split('://')[0]}:// can't be copied; attach it as a data asset")
    if dest.mode == "in_place":
        _check_same_cells(dest.location, src)


def _check_same_cells(location: str, src: Source) -> None:
    try:
        root = zarr.open_group(location, mode="r")
        group = root[src.table] if src.table else root
        shape, index = group["X"].attrs["shape"], _index(group, "obs")
        genes = _index(group, "var")
    except Exception as exc:
        layout = f"a SpatialData store with {src.table}" if src.table else "an AnnData store"
        raise OutputError(f"output location {location} isn't {layout} like the input "
                          f"({type(exc).__name__}: {exc})") from None
    ours = src.group
    if (list(shape) != list(ours["X"].attrs["shape"]) or list(index) != list(_index(ours, "obs"))
            or list(genes) != list(_index(ours, "var"))):
        raise OutputError(f"{location} has {shape[0]:,} cells x {shape[1]:,} genes, not the input's, so the results "
                          "would not line up: choose a new location (it is created) or a zarr made from this input")


def _index(group, frame: str):
    return read_elem(group[frame][group[frame].attrs["_index"]])


def save(dest: Destination, src: Source, write_results, message: str, branch: str, metadata: dict) -> str:
    """Write the results; a failure is one line (OutputError), the destination left as the error says."""
    try:
        return _save(dest, src, write_results, message, branch, metadata)
    except (OutputError, BranchMoved):
        raise
    except Exception as exc:
        if is_remote(dest.location) and is_access_error(exc):
            raise OutputError(access_hint(dest.location, writing=True)) from None
        if isinstance(exc, RuntimeError) and BranchMoved.TEXT in str(exc):
            raise BranchMoved(f"someone else saved to branch {branch} while this ran, so the results no longer match "
                              "its data and were not saved. Run again to start from the latest version.") from None
        left = {"commit": "nothing was saved", "in_place": "its results may be partial; a rerun replaces them",
                "new": "it's marked half-written" if is_remote(dest.location) else "nothing was left there",
                "copy": "nothing was left there"}[dest.mode]
        raise OutputError(f"writing to {dest.location} failed, {left} ({type(exc).__name__}: {exc})") from None


def _save(dest: Destination, src: Source, write_results, message: str, branch: str, metadata: dict) -> str:
    """Write the results; the destination is checked again first, since a run can take hours."""
    if dest.mode == "commit":
        snapshot = save_commit(src.repo, branch, src.snapshot, src.table, write_results, message, metadata)
        return f"saved as a new version ({snapshot}) on branch {branch} at {dest.location}"
    if dest.mode in ("new", "in_place"):
        kind, marker = detect(dest.location)
        if kind != ("missing" if dest.mode == "new" else "zarr") or marker == "new":
            raise OutputError(f"output location {dest.location} changed during the run (it's now {kind}); "
                              f"nothing was written")
    if dest.mode == "in_place":
        if dest.pending:
            print(f"{dest.location}: an earlier run stopped while writing its results; they're rewritten now")
        write = lambda: _write_into(dest.location, src.table, write_results)  # noqa: E731
        _with_marker(dest.location, "results", write) if is_remote(dest.location) else write()
        return f"written into {dest.location}"

    if dest.mode == "copy" and dest.kind == "zarr" and not src.repo:
        out = new_output(src.path, dest.location, "")
        try:
            return f"saved to {save_copy(src.path, src.table, write_results, out, '')}"
        except BaseException:
            shutil.rmtree(f"{out}.partial", ignore_errors=True)
            raise
    meta = {"source": src.path, **metadata}
    if dest.mode == "copy" and src.repo and dest.kind == "icechunk":
        build = lambda p: _copy_repo(src, p, write_results, message, branch, metadata)  # noqa: E731
    else:
        writer = _new_zarr if dest.kind == "zarr" else _new_repo
        build = lambda p: writer(src, p, write_results, message, branch, meta)  # noqa: E731
    if dest.mode == "copy":
        out = free_path(dest.location, ".zarr" if dest.kind == "zarr" else "")
    else:
        out = dest.location
    snapshot = _with_marker(out, "new", lambda: build(out)) if is_remote(out) else _local(out, build)
    return f"saved to {out}" if dest.kind == "zarr" else f"saved as a new version ({snapshot}) on branch {branch} of a new repo at {out}"


def _local(out: str, build):
    partial = f"{out}.partial"
    Path(partial).parent.mkdir(parents=True, exist_ok=True)
    try:
        result = build(partial)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    os.rename(partial, out)
    return result


def _with_marker(location: str, state: str, write):
    fs, path = fsspec.core.url_to_fs(location)
    fs.pipe_file(f"{path}/{MARKER}", state.encode())
    result = write()
    fs.rm_file(f"{path}/{MARKER}")
    return result


def _consolidate(location: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zarr.consolidate_metadata(location)


def _new_zarr(src: Source, location: str, write_results, message, branch, metadata) -> None:
    """A full zarr copy of the input at a new location: its files as they are, or the snapshot's arrays."""
    if src.repo:
        copy_group(src.root, zarr.open_group(location, mode="a"))
    elif not is_remote(location):
        shutil.copytree(src.path, location, copy_function=shutil.copyfile, dirs_exist_ok=True)
        for folder, _, _ in os.walk(location):
            os.chmod(folder, 0o755)
    else:
        fs, path = fsspec.core.url_to_fs(location)
        root = Path(src.path)
        files = [f for f in root.rglob("*") if f.is_file()]
        fs.put([str(f) for f in files], [f"{path}/{f.relative_to(root).as_posix()}" for f in files])
    copy = zarr.open_group(location, mode="r+", use_consolidated=False)
    write_results(copy[src.table] if src.table else copy)
    _consolidate(location)


def _new_repo(src: Source, location: str, write_results, message, branch, metadata) -> str:
    """A new icechunk repo holding the input and the results as one commit on `branch`."""
    repo = Repo.create(location)
    if branch != "main":  # main holds the input, so the repo opens with data; the results go on the branch
        copy_group(src.root, repo.open_zarr("w"))
        repo.commit(f"copy of {src.path}", metadata={"source": src.path})
        repo.checkout(branch, create=True)
    start = repo.log()[0].id
    if branch == "main":
        copy_group(src.root, repo.open_zarr("w"))
    return save_commit(repo, branch, start, src.table, write_results, message, metadata)


def _copy_repo(src: Source, location: str, write_results, message, branch, metadata) -> str:
    """The input repo, every branch and snapshot, with the results as one commit on `branch`."""
    src.repo.copy(location)
    return save_commit(Repo(location, branch=branch), branch, src.snapshot, src.table, write_results, message, metadata)


def _write_into(location: str, table: str, write_results) -> None:
    root = zarr.open_group(location, mode="r+", use_consolidated=False)
    write_results(root[table] if table else root)
    _consolidate(location)
