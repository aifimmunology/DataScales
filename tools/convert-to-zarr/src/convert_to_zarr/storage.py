"""Output-store abstraction shared by all converters.

A converter never opens a store directly; it calls :func:`open_output_store`, which returns
``(root_group, finalize)``. The ``root_group`` is a plain zarr v3 group regardless of backend,
so every existing writer (``write_elem``, ``da.store``, ``require_array`` …) works unchanged.
``finalize()`` is called once after all writes to make them durable:

* ``backend="zarr"``  — optionally consolidates metadata.
* ``backend="icechunk"`` — commits the writable session (one commit per conversion, per the
  Icechunk "few, large commits" guidance).

Icechunk's API is verified against the vendored source (v2.1.2); it is imported lazily so the
default zarr path never touches it.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Callable

import zarr

from .config import AppConfig


class StorageError(RuntimeError):
    """Raised when an output/input store cannot be opened for the configured backend."""


def _prepare_output_path(output_path: Path, overwrite: bool) -> None:
    """Remove an existing output path when overwrite is enabled, else raise."""
    if output_path.exists():
        if not overwrite:
            raise StorageError(
                f"Output path already exists: {output_path}. "
                "Use overwrite=true in config or --overwrite flag."
            )
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()


def _is_s3_url(path: str | Path) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def _store_name(path: str | Path) -> str:
    """Tail component of a local path or s3:// URL, for commit messages."""
    return str(path).rstrip("/").rsplit("/", 1)[-1]


def _icechunk_storage(path: str | Path):
    """Icechunk Storage for a local path or an ``s3://bucket/prefix`` URL.

    S3 credentials come from the environment (AWS_* vars / default chain)."""
    import icechunk

    if _is_s3_url(path):
        bucket, _, prefix = str(path)[len("s3://"):].partition("/")
        if not bucket or not prefix.strip("/"):
            raise StorageError(f"Invalid S3 URL '{path}': expected s3://bucket/prefix.")
        return icechunk.s3_storage(bucket=bucket, prefix=prefix.strip("/"))
    return icechunk.local_filesystem_storage(str(path))


def open_output_store(
    output_path: str | Path, cfg: AppConfig, *, commit_message: str | None = None
) -> tuple[zarr.Group, Callable[[], None]]:
    """Open the output root group for the configured backend.

    Returns ``(root_group, finalize)``. Handles overwrite/exists checks for both backends.
    ``finalize()`` must be called exactly once after all writes complete.
    """
    if _is_s3_url(output_path) and cfg.io.backend != "icechunk":
        raise StorageError("s3:// outputs require backend='icechunk' (--icechunk).")
    if cfg.io.backend == "icechunk":
        import icechunk  # noqa: F401  (verify install early, before any writes)

        if not _is_s3_url(output_path):
            output_path = Path(output_path)
            _prepare_output_path(output_path, cfg.io.overwrite)
            # Repos live in a directory; open_or_create needs the parent to exist.
            output_path.parent.mkdir(parents=True, exist_ok=True)
        storage = _icechunk_storage(output_path)
        repo = icechunk.Repository.open_or_create(storage)
        session = repo.writable_session("main")
        root = zarr.open_group(store=session.store, mode="w")

        def finalize() -> None:
            msg = commit_message or f"convert-to-zarr write → {_store_name(output_path)}"
            snapshot_id = session.commit(msg)
            print(
                f"  icechunk commit {snapshot_id} on branch 'main'",
                flush=True, file=sys.stderr,
            )

        return root, finalize

    # Default: plain on-disk zarr.
    output_path = Path(output_path)
    _prepare_output_path(output_path, cfg.io.overwrite)
    root = zarr.open_group(str(output_path), mode="w")

    def finalize() -> None:
        if cfg.io.consolidate_metadata:
            zarr.consolidate_metadata(str(output_path))

    return root, finalize


def open_store_rw(
    store_path: str | Path, cfg: AppConfig, *, commit_message: str | None = None
) -> tuple[zarr.Group, Callable[[], None]]:
    """Open an existing store for in-place update; finalize() commits/re-consolidates.

    Icechunk targets are auto-detected (s3:// URL or local repo layout) so in-place ops
    work on a repo without --icechunk."""
    if (
        cfg.io.backend == "icechunk"
        or _is_s3_url(store_path)
        or _is_icechunk_repo(Path(store_path))
    ):
        import icechunk

        repo = icechunk.Repository.open(_icechunk_storage(store_path))
        session = repo.writable_session("main")
        root = zarr.open_group(store=session.store, mode="r+")

        def finalize() -> None:
            msg = commit_message or f"convert-to-zarr update → {_store_name(store_path)}"
            snapshot_id = session.commit(msg)
            print(
                f"  icechunk commit {snapshot_id} on branch 'main'",
                flush=True, file=sys.stderr,
            )

        return root, finalize

    store_path = Path(store_path)
    if not store_path.exists():
        raise StorageError(f"Store does not exist: {store_path}")
    # use_consolidated=False: anndata's write_elem refuses to edit a group opened
    # through consolidated metadata; finalize() re-consolidates below.
    root = zarr.open_group(str(store_path), mode="r+", use_consolidated=False)

    import json

    meta_file = store_path / "zarr.json"
    had_consolidated = (
        meta_file.exists()
        and json.loads(meta_file.read_text()).get("consolidated_metadata") is not None
    )

    def finalize() -> None:
        if had_consolidated or cfg.io.consolidate_metadata:
            zarr.consolidate_metadata(str(store_path))

    return root, finalize


def _is_icechunk_repo(path: Path) -> bool:
    """Local icechunk repos carry repo/ + snapshots/ and no zarr.json at the root."""
    return (
        path.is_dir()
        and not (path / "zarr.json").exists()
        and (path / "snapshots").is_dir()
        and (path / "repo").exists()
    )


def open_input_group(
    path: str, *, icechunk: bool = False, branch: str = "main"
) -> zarr.Group:
    """Open an existing store read-only as a zarr group (for the reader).

    Icechunk repos are auto-detected (s3:// URL or local repo layout) and opened as a
    read-only session at ``branch``; anything else opens as a plain zarr directory.
    """
    if icechunk or _is_s3_url(path) or _is_icechunk_repo(Path(path)):
        import icechunk as ic

        repo = ic.Repository.open(_icechunk_storage(path))
        session = repo.readonly_session(branch=branch)
        return zarr.open_group(store=session.store, mode="r")
    return zarr.open_group(path, mode="r")
