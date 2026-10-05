from __future__ import annotations

import enum
import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from annizarr._core._config import AppConfig
from annizarr._storage._backends import is_icechunk_repo
from annizarr._storage._uri import is_remote, prepare_output_path, store_name
from annizarr.errors import StorageError
from annizarr.typing import PathLike

if TYPE_CHECKING:
    import zarr

logger = logging.getLogger(__name__)


class _SwapState(enum.Enum):
    OPEN = "open"  # tmp store being written; target (if any) untouched
    SWAPPING = "swapping"  # mid-swap; target may momentarily be missing
    DONE = "done"  # swap complete; tmp no longer exists at its tmp path
    ABORTED = "aborted"  # writer gave up; tmp cleaned up (or never existed)


def check_output_target(output_path: PathLike, cfg: AppConfig) -> None:
    # called before any expensive load; open_output_store repeats this right before writing.
    if cfg.io.backend == "icechunk":
        from annizarr.ic import Repo

        if Repo.exists(str(output_path)) and not cfg.io.overwrite:
            raise StorageError(
                f"Icechunk repo already exists at '{output_path}'. Use overwrite=true in "
                "config or --overwrite to replace its contents."
            )
        return
    if is_remote(output_path):
        return  # rejected later by open_output_store (remote requires the icechunk backend)
    target = Path(output_path)
    if target.exists() and not cfg.io.overwrite:
        raise StorageError(f"Output path already exists: {target}. Use overwrite=true in config or --overwrite flag.")


@dataclass
class OutputStore:
    root: zarr.Group
    finalize: Callable[[], str | None]
    abort: Callable[[], None]


def open_output_store(
    output_path: PathLike,
    cfg: AppConfig,
    *,
    commit_message: str | None = None,
    branch: str | None = None,
    expected_shape: tuple[int, int] | None = None,
) -> OutputStore:
    import zarr

    check_output_target(output_path, cfg)

    if cfg.io.backend == "icechunk":
        from annizarr.ic import Repo

        exists = Repo.exists(str(output_path))
        if not exists and not is_remote(output_path):
            prepare_output_path(Path(output_path), cfg.io.overwrite)
        repo = Repo(str(output_path)) if exists else Repo.create(str(output_path))
        if branch is not None and branch != repo.branch:
            repo.checkout(branch, create=True)
        root: zarr.Group = repo.open_zarr("w", truncate=True)  # truncate: a re-run must not append to old data
        committed: str | None = None

        def finalize_icechunk() -> str | None:
            nonlocal committed
            if committed is None:
                _verify_new_store(root, expected_shape=expected_shape)
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

    # atomic swap: same-filesystem temp dir (plain rename); old store moved aside, not
    # deleted, until the new one is in place, so a crash mid-swap can always restore it.
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f"{target.name}.tmp-{uuid4().hex[:8]}"
    root = zarr.open_group(str(tmp_path), mode="w")
    state = _SwapState.OPEN

    def finalize() -> str | None:
        nonlocal state
        if state is _SwapState.DONE:
            return None
        if state is not _SwapState.OPEN:
            return None  # a previous finalize() already failed and cleaned up; nothing to redo
        verify_root = zarr.open_group(str(tmp_path), mode="r")
        _verify_new_store(verify_root, expected_shape=expected_shape)
        if cfg.io.consolidate_metadata:
            zarr.consolidate_metadata(str(tmp_path))

        aside: Path | None = None
        if target.exists():
            if not cfg.io.overwrite:
                raise StorageError(f"Output path already exists: {target}.")
            aside = target.parent / f".{target.name}.old-{uuid4().hex[:8]}"

        state = _SwapState.SWAPPING
        try:
            if aside is not None:
                os.replace(target, aside)
            os.replace(tmp_path, target)
        except BaseException:
            # restore `aside` so the target is never left empty or gone.
            if aside is not None and not target.exists() and aside.exists():
                os.replace(aside, target)
            shutil.rmtree(tmp_path, ignore_errors=True)
            state = _SwapState.ABORTED
            raise

        if aside is not None:
            shutil.rmtree(aside, ignore_errors=True)
        state = _SwapState.DONE
        return None

    def abort() -> None:
        nonlocal state
        if state is _SwapState.DONE or state is _SwapState.ABORTED:
            return  # safe to call after finalize(), or after finalize() already cleaned up
        if state is _SwapState.OPEN:
            shutil.rmtree(tmp_path, ignore_errors=True)
        state = _SwapState.ABORTED

    return OutputStore(root, finalize, abort)


def _verify_new_store(root: zarr.Group, *, expected_shape: tuple[int, int] | None = None) -> None:
    # never ad.read_zarr here, which would materialise X
    import zarr

    from annizarr._core._zarr import shape_attr

    if root.attrs.get("encoding-type") != "anndata":
        raise StorageError("Refusing to finalize: missing anndata root encoding-type.")
    if "X" not in root:
        raise StorageError("Refusing to finalize: no X in the written store.")
    if expected_shape is None:
        return
    x = root["X"]
    actual = x.shape if isinstance(x, zarr.Array) else shape_attr(x)
    actual_shape = (int(actual[0]), int(actual[1]))
    if actual_shape != expected_shape:
        raise StorageError(f"X shape mismatch: expected {expected_shape}, got {actual_shape}.")


def open_store_rw(
    store_path: PathLike, cfg: AppConfig, *, commit_message: str | None = None, branch: str | None = None
) -> tuple[zarr.Group, Callable[[], str | None]]:
    import zarr

    if cfg.io.backend == "icechunk" or is_remote(store_path) or is_icechunk_repo(store_path):
        from annizarr.ic import Repo

        repo = Repo(str(store_path))
        if branch is not None and branch != repo.branch:
            repo.checkout(branch, create=True)
        root: zarr.Group = repo.open_zarr("w")
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
    # use_consolidated=False: anndata's write_elem refuses to edit a consolidated group.
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
    import zarr

    if is_remote(path) or is_icechunk_repo(path):
        from annizarr.ic import Repo

        repo = Repo(str(path), branch=branch)
        return repo.open_zarr("r", snapshot_id=snapshot_id)
    return zarr.open_group(str(path), mode="r")
