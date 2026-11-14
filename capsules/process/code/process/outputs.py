"""Where a run's results go: decided and checked before any GPU work, checked again and written at the end.

One destination per run, never both:
- a copy in results_dir (the default): `results_copy` names it, `results_format` picks zarr or icechunk
  ("auto": the input's own format). An existing name is never reused: <name>-1, -2, ...
- an external location (s3:// or a path), `external`:
  - icechunk input: the location must be the input's own repo (it holds the input's latest snapshot on `branch`).
    The data is read from there and one commit is made on `branch`. Anything else is refused as a likely typo.
  - zarr input, nothing at the location: a new store, zarr unless `external_format` says icechunk.
  - zarr input, an existing zarr with the same layout, cells and genes: the results are written into it in place.
  `external_format`, when not "auto", must match what's already at the location.

A location being written over S3 (no atomic rename there) carries a `.process-writing` marker until done:
"new" means its data is incomplete and it's refused; "results" means only the results may be partial and a rerun
in place fixes them. Local copies are built at `<out>.partial` and renamed instead.
"""
from __future__ import annotations

import os
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import fsspec
import zarr
from annizarr import AnzError, Repo

from store import InputError, Source, free_path, save_commit, save_copy

FORMATS = ("auto", "zarr", "icechunk")
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
    fs, path = fsspec.core.url_to_fs(location)
    fs.invalidate_cache()  # s3fs answers exists() from listings cached hours ago, at plan time
    marker = fs.cat_file(f"{path}/{MARKER}").decode().strip() if fs.exists(f"{path}/{MARKER}") else ""
    try:
        if Repo.exists(location):
            return "icechunk", marker
    except Exception as exc:
        if is_remote(location):
            raise OutputError(f"can't check {location} ({type(exc).__name__}: {exc})") from None
    if not fs.exists(path) or not fs.ls(path, detail=False):
        return "missing", marker
    if fs.isfile(path):
        return "other", marker
    return ("zarr" if fs.exists(f"{path}/zarr.json") or fs.exists(f"{path}/.zgroup") else "other"), marker


def input_kind(path: str) -> str:
    try:
        return "icechunk" if Repo.exists(path) else "zarr"
    except Exception:
        return "zarr"


def _stem(path: str) -> str:
    return re.sub(r"\.(zarr|icechunk)$", "", Path(path.rstrip("/")).name)


def plan(cfg) -> Destination:
    """The destination of this run, refused here with a one-line OutputError when it can't work."""
    source_kind = input_kind(cfg.input)
    if not cfg.external:
        name = cfg.results_copy or f"{_stem(cfg.input)}.processed"
        location = name if os.path.isabs(name) else f"{cfg.results_dir.rstrip('/')}/{name}"
        kind = source_kind if cfg.results_format == "auto" else cfg.results_format
        return Destination("copy", kind, location.rstrip("/").removesuffix(".zarr"))

    location, wanted = cfg.external.rstrip("/"), cfg.external_format
    if not is_remote(location) and os.path.realpath(location) == os.path.realpath(cfg.input):
        raise OutputError(f"the external location is the input itself, which is never changed: pick another location "
                          f"(for a read-only data asset, its writable S3 location)")
    kind, marker = detect(location)
    if marker == "new":
        raise OutputError(f"external location {location} was left half-written by an earlier run "
                          f"({MARKER} is still there); delete it, or pick another location")
    if kind == "other":
        raise OutputError(f"external location {location} holds something that isn't a zarr or an icechunk repo")
    if kind != "missing" and wanted != "auto" and wanted != kind:
        raise OutputError(f"external location {location} is an existing {kind}, but external_format is {wanted}")
    if not is_remote(location):
        if os.path.exists(f"{location}.partial"):
            raise OutputError(f"{location}.partial was left by an earlier run that stopped; delete it and rerun")
        parent = next(p for p in Path(location).absolute().parents if p.exists()) if kind == "missing" else Path(location)
        if not os.access(parent, os.W_OK):
            raise OutputError(f"external location {location} can't be written ({parent} is read-only)")
    if source_kind == "icechunk":
        if kind != "icechunk":
            found = "nothing there" if kind == "missing" else "a zarr there"
            raise OutputError(f"an icechunk input commits onto its own repo at the external location, but there's "
                              f"{found}: check {location} for a typo")
        _check_same_repo(cfg.input, location, cfg.branch)
        return Destination("commit", "icechunk", location)
    if kind == "icechunk":
        raise OutputError(f"external location {location} is an icechunk repo; only an icechunk input of that same "
                          f"repo can commit onto it")
    if kind == "zarr":
        return Destination("in_place", "zarr", location, marker)
    return Destination("new", "zarr" if wanted == "auto" else wanted, location)


def _check_same_repo(input_path: str, location: str, branch: str) -> None:
    try:
        external = {s.id for s in Repo(location, branch=branch).log()}
    except Exception as exc:
        raise OutputError(f"can't use branch {branch} of the external repo {location} ({type(exc).__name__}: {exc})") from None
    try:
        tip = Repo(input_path, branch=branch).log()[0].id
    except Exception as exc:
        raise OutputError(f"the input has no branch {branch} ({type(exc).__name__}: {exc})") from None
    if tip == FIRST_COMMIT:
        raise OutputError(f"the input's branch {branch} has no commits with data, so it can't be matched to {location}")
    if tip not in external:
        raise OutputError(f"the external repo {location} doesn't hold the input's snapshot {tip[:12]} on {branch}: "
                          f"it's another repo, check the location for a typo")


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
        raise OutputError(f"external location {location} isn't {layout} like the input "
                          f"({type(exc).__name__}: {exc})") from None
    ours = src.group
    if (list(shape) != list(ours["X"].attrs["shape"]) or list(index) != list(_index(ours, "obs"))
            or list(genes) != list(_index(ours, "var"))):
        raise OutputError(f"external location {location} doesn't have the input's cells and genes "
                          f"({shape[0]:,} x {shape[1]:,}); it would get results for other cells")


def _index(group, frame: str):
    from anndata.io import read_elem

    return read_elem(group[frame][group[frame].attrs["_index"]])


def save(dest: Destination, src: Source, write_results, message: str, branch: str, metadata: dict) -> str:
    """Write the results; a failure is one line (OutputError), the destination left as the error says."""
    try:
        return _save(dest, src, write_results, message, branch, metadata)
    except (RuntimeError, AnzError, OSError) as exc:
        left = {"commit": "nothing was committed", "in_place": "its results may be partial; a rerun replaces them",
                "new": "it's marked half-written" if is_remote(dest.location) else "nothing was left there",
                "copy": "nothing was left there"}[dest.mode]
        raise OutputError(f"writing to {dest.location} failed, {left} ({type(exc).__name__}: {exc})") from None


def _save(dest: Destination, src: Source, write_results, message: str, branch: str, metadata: dict) -> str:
    """Write the results; the destination is checked again first, since a run can take hours."""
    if dest.mode == "commit":
        snapshot = save_commit(src.repo, branch, src.snapshot, src.table, write_results, message, metadata)
        return f"committed {snapshot} on {branch} at {dest.location}"
    if dest.mode in ("new", "in_place"):
        kind, marker = detect(dest.location)
        if kind != ("missing" if dest.mode == "new" else "zarr") or marker == "new":
            raise OutputError(f"external location {dest.location} changed during the run (it's now {kind}); "
                              f"nothing was written")
    if dest.mode == "in_place":
        if dest.pending:
            print(f"{dest.location}: an earlier run stopped while writing its results; they're rewritten now")
        _with_marker(dest.location, "results", lambda: _write_into(dest.location, src.table, write_results))
        return f"written into {dest.location}"

    meta = {"source": src.path, **metadata}
    if dest.mode == "copy" and dest.kind == "zarr" and not src.repo:
        return f"saved to {save_copy(src.path, src.table, write_results, dest.location, '')}"
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
    return f"saved to {out}" if dest.kind == "zarr" else f"committed {snapshot} on {branch} of a new repo at {out}"


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
    from annizarr.ic._copy import copy_group

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
    from annizarr.ic._copy import copy_group

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
