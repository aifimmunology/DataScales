"""Output-store abstraction shared by all converters.

A converter never opens a store directly; it calls :func:`open_output_store`, which returns
``(root_group, finalize)``. The ``root_group`` is a plain zarr v3 group regardless of backend,
so every existing writer (``write_elem``, ``da.store``, ``require_array`` …) works unchanged.
``finalize()`` is called once after all writes to make them durable:

* ``backend="zarr"``  — optionally consolidates metadata.
* ``backend="icechunk"`` — commits the writable session (one commit per conversion, per the
  Icechunk "few, large commits" guidance).

Icechunk's API is verified against the vendored source (v2.0.6); it is imported lazily so the
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


def _icechunk_storage(output_path: Path, cfg: AppConfig):
    """Build an Icechunk Storage for the configured target (local now; GCS scaffolded)."""
    import icechunk

    target = cfg.io.icechunk_storage
    if target == "local":
        return icechunk.local_filesystem_storage(str(output_path))
    if target == "gcs":
        # Scaffolding only — credentials/wiring are intentionally not finished yet.
        if not cfg.io.gcs_bucket:
            raise StorageError("icechunk_storage='gcs' requires io.gcs_bucket to be set.")
        raise StorageError(
            "icechunk GCS backend is scaffolded but not wired up yet. "
            "Use icechunk_storage='local' for now. "
            "(icechunk.gcs_storage(bucket=..., prefix=...) is the intended entry point.)"
        )
    raise StorageError(f"Unknown icechunk_storage '{target}'. Expected 'local' or 'gcs'.")


def open_output_store(
    output_path: Path, cfg: AppConfig, *, commit_message: str | None = None
) -> tuple[zarr.Group, Callable[[], None]]:
    """Open the output root group for the configured backend.

    Returns ``(root_group, finalize)``. Handles overwrite/exists checks for both backends.
    ``finalize()`` must be called exactly once after all writes complete.
    """
    if cfg.io.backend == "icechunk":
        import icechunk  # noqa: F401  (verify install early, before any writes)

        _prepare_output_path(output_path, cfg.io.overwrite)
        # Repos live in a directory; open_or_create needs the parent to exist.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        storage = _icechunk_storage(output_path, cfg)
        repo = icechunk.Repository.open_or_create(storage)
        session = repo.writable_session("main")
        root = zarr.open_group(store=session.store, mode="w")

        def finalize() -> None:
            msg = commit_message or f"convert-to-zarr convert → {output_path.name}"
            snapshot_id = session.commit(msg)
            print(
                f"  icechunk commit {snapshot_id} on branch 'main'",
                flush=True, file=sys.stderr,
            )

        return root, finalize

    # Default: plain on-disk zarr.
    _prepare_output_path(output_path, cfg.io.overwrite)
    root = zarr.open_group(str(output_path), mode="w")

    def finalize() -> None:
        if cfg.io.consolidate_metadata:
            zarr.consolidate_metadata(str(output_path))

    return root, finalize


def open_store_rw(
    store_path: Path, cfg: AppConfig, *, commit_message: str | None = None
) -> tuple[zarr.Group, Callable[[], None]]:
    """Open an existing store for in-place update; finalize() commits/re-consolidates."""
    if cfg.io.backend == "icechunk":
        import icechunk

        repo = icechunk.Repository.open(_icechunk_storage(store_path, cfg))
        session = repo.writable_session("main")
        root = zarr.open_group(store=session.store, mode="r+")

        def finalize() -> None:
            msg = commit_message or f"zarrsmith update → {store_path.name}"
            snapshot_id = session.commit(msg)
            print(
                f"  icechunk commit {snapshot_id} on branch 'main'",
                flush=True, file=sys.stderr,
            )

        return root, finalize

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


def open_input_group(
    path: str, *, icechunk: bool = False, branch: str = "main"
) -> zarr.Group:
    """Open an existing store read-only as a zarr group (for the reader).

    Plain zarr opens the directory directly; icechunk opens a read-only session at ``branch``.
    """
    if icechunk:
        import icechunk as ic

        repo = ic.Repository.open(ic.local_filesystem_storage(path))
        session = repo.readonly_session(branch=branch)
        return zarr.open_group(store=session.store, mode="r")
    return zarr.open_group(path, mode="r")
