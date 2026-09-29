from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from annizarr._config import AppConfig
from annizarr._storage._backends import is_icechunk_repo, require_icechunk, storage_for
from annizarr._storage._uri import is_remote, prepare_output_path, store_name
from annizarr.errors import StorageError
from annizarr.typing import PathLike

if TYPE_CHECKING:
    import zarr

logger = logging.getLogger(__name__)


def open_output_store(
    output_path: PathLike, cfg: AppConfig, *, commit_message: str | None = None
) -> tuple[zarr.Group, Callable[[], str | None]]:
    import zarr

    # (root_group, finalize); finalize() returns the icechunk snapshot id, or None for plain zarr
    if cfg.io.backend == "icechunk":
        icechunk = require_icechunk()

        if not is_remote(output_path):
            local_path = Path(output_path)
            prepare_output_path(local_path, cfg.io.overwrite)
            # Repos live in a directory; open_or_create needs the parent to exist.
            local_path.parent.mkdir(parents=True, exist_ok=True)
        repo = icechunk.Repository.open_or_create(storage_for(output_path))
        session = repo.writable_session("main")
        root = zarr.open_group(store=session.store, mode="w")
        committed: str | None = None

        def finalize_icechunk() -> str | None:
            nonlocal committed
            if committed is None:
                msg = commit_message or f"annizarr write → {store_name(output_path)}"
                committed = session.commit(msg)
                logger.info(f"icechunk commit {committed} on branch 'main'")
            return committed

        return root, finalize_icechunk

    if is_remote(output_path):
        raise StorageError(
            f"Remote output '{output_path}' requires the icechunk backend — pass --ic "
            "(or set io.backend='icechunk' in config)."
        )

    # Default: plain on-disk zarr.
    output_path = Path(output_path)
    prepare_output_path(output_path, cfg.io.overwrite)
    root = zarr.open_group(str(output_path), mode="w")
    finalized = False

    def finalize() -> str | None:
        nonlocal finalized
        if not finalized:
            if cfg.io.consolidate_metadata:
                zarr.consolidate_metadata(str(output_path))
            finalized = True
        return None

    return root, finalize


def open_store_rw(
    store_path: PathLike, cfg: AppConfig, *, commit_message: str | None = None
) -> tuple[zarr.Group, Callable[[], str | None]]:
    import zarr

    # icechunk targets are auto-detected (remote URI or local repo layout), so in-place
    # ops work on a repo without cfg.io.backend == "icechunk"
    if cfg.io.backend == "icechunk" or is_remote(store_path) or is_icechunk_repo(store_path):
        icechunk = require_icechunk()

        repo = icechunk.Repository.open(storage_for(store_path))
        session = repo.writable_session("main")
        root = zarr.open_group(store=session.store, mode="r+")
        committed: str | None = None

        def finalize_icechunk() -> str | None:
            nonlocal committed
            if committed is None:
                msg = commit_message or f"annizarr update → {store_name(store_path)}"
                committed = session.commit(msg)
                logger.info(f"icechunk commit {committed} on branch 'main'")
            return committed

        return root, finalize_icechunk

    store_path = Path(store_path)
    if not store_path.exists():
        raise StorageError(f"Store does not exist: {store_path}")
    # use_consolidated=False: anndata's write_elem refuses to edit a group opened
    # through consolidated metadata; finalize() re-consolidates below.
    root = zarr.open_group(str(store_path), mode="r+", use_consolidated=False)

    meta_file = store_path / "zarr.json"
    had_consolidated = meta_file.exists() and json.loads(meta_file.read_text()).get("consolidated_metadata") is not None
    finalized = False

    def finalize() -> str | None:
        nonlocal finalized
        if not finalized:
            if had_consolidated or cfg.io.consolidate_metadata:
                zarr.consolidate_metadata(str(store_path))
            finalized = True
        return None

    return root, finalize


def open_input_group(path: PathLike, *, icechunk: bool = False, branch: str = "main") -> zarr.Group:
    import zarr

    # icechunk repos are auto-detected (remote URI or local repo layout) and opened
    # read-only at `branch`; anything else opens as a plain zarr directory
    if icechunk or is_remote(path) or is_icechunk_repo(path):
        ic = require_icechunk()

        repo = ic.Repository.open(storage_for(path))
        session = repo.readonly_session(branch=branch)
        return zarr.open_group(store=session.store, mode="r")
    return zarr.open_group(str(path), mode="r")
