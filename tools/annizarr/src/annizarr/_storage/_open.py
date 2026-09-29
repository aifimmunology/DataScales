from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from annizarr._config import AppConfig
from annizarr._storage._backends import is_icechunk_repo
from annizarr._storage._uri import is_remote, prepare_output_path, store_name
from annizarr.errors import StorageError
from annizarr.typing import PathLike

if TYPE_CHECKING:
    import zarr

logger = logging.getLogger(__name__)


@dataclass
class OutputStore:
    """Handle returned by :func:`open_output_store` for a brand-new store.

    Parameters
    ----------
    root
        Writable root group to write the new store into.
    finalize
        Make the store durable and return the Icechunk snapshot id (``None`` for
        plain zarr). Plain zarr: optionally consolidates, verifies the written temp
        store, then atomically swaps it onto ``output_path``. Icechunk: commits the
        session. Idempotent — a second call is a no-op and returns the same result.
    abort
        Discard a failed write attempt: removes the plain-zarr temp directory, or
        discards the Icechunk session's uncommitted changes. Leaves any pre-existing
        target untouched either way. Safe to call after ``finalize`` (no-op).
    """

    root: zarr.Group
    finalize: Callable[[], str | None]
    abort: Callable[[], None]


def open_output_store(
    output_path: PathLike,
    cfg: AppConfig,
    *,
    commit_message: str | None = None,
    branch: str | None = None,
) -> OutputStore:
    """Open a brand-new store to write: plain zarr, or an Icechunk repo with ``--ic``.

    Parameters
    ----------
    output_path
        Destination store path or URI.
    cfg
        Resolved configuration (consults ``cfg.io.backend``, ``cfg.io.overwrite``,
        ``cfg.io.consolidate_metadata``).
    commit_message
        Icechunk commit message; ignored for plain zarr. ``None`` uses a generic
        message naming the destination — callers pass one naming the actual op.
    branch
        Icechunk branch to write to; created off the current tip if it doesn't exist
        yet, left alone (current HEAD, or ``main``) if ``None``. Ignored for plain zarr.

    Returns
    -------
    OutputStore

    Raises
    ------
    StorageError
        ``output_path`` is remote and ``cfg.io.backend`` isn't ``"icechunk"``; the
        plain-zarr target already exists and ``cfg.io.overwrite`` is not set; or an
        Icechunk repo already exists at ``output_path`` and ``cfg.io.overwrite`` is
        not set.
    """
    import zarr

    if cfg.io.backend == "icechunk":
        from annizarr._ic import Repo

        exists = Repo.exists(str(output_path))
        if exists and not cfg.io.overwrite:
            raise StorageError(
                f"Icechunk repo already exists at '{output_path}'. Use overwrite=true in "
                "config or --overwrite to replace its contents."
            )
        if not exists and not is_remote(output_path):
            # clears any stale non-repo directory (e.g. a leftover plain-zarr store) so
            # Repo.create's empty-destination check doesn't trip on it
            prepare_output_path(Path(output_path), cfg.io.overwrite)
        repo = Repo(str(output_path)) if exists else Repo.create(str(output_path))
        if branch is not None and branch != repo.branch:
            repo.checkout(branch, create=True)
        # a NEW output must start from an empty root even when the branch already holds
        # data from a previous write (e.g. a re-run with --overwrite)
        root = repo.open_zarr("w", truncate=True)
        committed: str | None = None

        def finalize_icechunk() -> str | None:
            nonlocal committed
            if committed is None:
                msg = commit_message or f"annizarr write → {store_name(output_path)}"
                committed = repo.commit(msg)
                logger.info(f"icechunk commit {committed} on branch '{repo.branch}'")
            return committed

        def abort_icechunk() -> None:
            repo.discard()

        return OutputStore(root, finalize_icechunk, abort_icechunk)

    if is_remote(output_path):
        raise StorageError(
            f"Remote output '{output_path}' requires the icechunk backend — pass --ic "
            "(or set io.backend='icechunk' in config)."
        )

    # Plain on-disk zarr: fail fast on a pre-existing target (nothing removed yet), write
    # into a sibling temp directory (same filesystem, so the final swap is a plain rename),
    # and only replace the target once finalize() has verified the temp store.
    target = Path(output_path)
    if target.exists() and not cfg.io.overwrite:
        raise StorageError(f"Output path already exists: {target}. Use overwrite=true in config or --overwrite flag.")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f"{target.name}.tmp-{uuid4().hex[:8]}"
    root = zarr.open_group(str(tmp_path), mode="w")
    finalized = False

    def finalize() -> str | None:
        nonlocal finalized
        if finalized:
            return None
        _verify_new_store(tmp_path)
        if cfg.io.consolidate_metadata:
            zarr.consolidate_metadata(str(tmp_path))
        if target.exists():
            if not cfg.io.overwrite:
                raise StorageError(f"Output path already exists: {target}.")
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        os.replace(tmp_path, target)
        finalized = True
        return None

    def abort() -> None:
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)

    return OutputStore(root, finalize, abort)


def _verify_new_store(path: Path) -> None:
    # light verification before the atomic swap: opens read-only (never `ad.read_zarr`,
    # which would materialise X) and checks the anndata root encoding attr + that X exists
    import zarr

    root = zarr.open_group(str(path), mode="r")
    if root.attrs.get("encoding-type") != "anndata":
        raise StorageError(f"Refusing to finalize '{path}': missing anndata root encoding-type.")
    if "X" not in root:
        raise StorageError(f"Refusing to finalize '{path}': no X in the written store.")


def open_store_rw(
    store_path: PathLike, cfg: AppConfig, *, commit_message: str | None = None, branch: str | None = None
) -> tuple[zarr.Group, Callable[[], str | None]]:
    """Open an existing store to edit in place (``add-expr``, ``append``).

    Parameters
    ----------
    store_path
        Existing store path or URI: a plain zarr directory or an Icechunk repo
        (auto-detected — a remote URI or local repo layout doesn't need
        ``cfg.io.backend`` set to ``"icechunk"``).
    cfg
        Resolved configuration.
    commit_message
        Icechunk commit message; ignored for plain zarr.
    branch
        Icechunk branch to edit; created off the current tip if it doesn't exist yet,
        left alone (current HEAD, or ``main``) if ``None``. Ignored for plain zarr.

    Returns
    -------
    tuple[zarr.Group, Callable[[], str | None]]
        ``(root_group, finalize)``; ``finalize()`` returns the Icechunk snapshot id,
        or ``None`` for plain zarr, and is idempotent.

    Raises
    ------
    StorageError
        ``store_path`` does not exist (plain zarr only).
    """
    import zarr

    if cfg.io.backend == "icechunk" or is_remote(store_path) or is_icechunk_repo(store_path):
        from annizarr._ic import Repo

        repo = Repo(str(store_path))
        if branch is not None and branch != repo.branch:
            repo.checkout(branch, create=True)
        root = repo.open_zarr("w")
        committed: str | None = None

        def finalize_icechunk() -> str | None:
            nonlocal committed
            if committed is None:
                msg = commit_message or f"annizarr update → {store_name(store_path)}"
                committed = repo.commit(msg)
                logger.info(f"icechunk commit {committed} on branch '{repo.branch}'")
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


def open_input_group(path: PathLike, *, branch: str | None = None, snapshot_id: str | None = None) -> zarr.Group:
    """Open an existing store read-only: plain zarr, or an Icechunk repo (auto-detected).

    Parameters
    ----------
    path
        Store path or URI.
    branch
        Icechunk branch to read; defaults to the persisted HEAD, falling back to the
        repository's default branch. Ignored for plain zarr.
    snapshot_id
        Icechunk snapshot id to time-travel to, instead of ``branch``'s tip. Ignored
        for plain zarr.

    Returns
    -------
    zarr.Group
    """
    import zarr

    if is_remote(path) or is_icechunk_repo(path):
        from annizarr._ic import Repo

        repo = Repo(str(path), branch=branch)
        return repo.open_zarr("r", snapshot_id=snapshot_id)
    return zarr.open_group(str(path), mode="r")
