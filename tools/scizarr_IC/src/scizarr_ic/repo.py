"""Git-like wrapper around an icechunk repository.

One ``Repo`` == one icechunk repository (local dir or s3://—gs:// URI). The current
branch ("HEAD") persists across CLI invocations in a ``scizarr_head`` file at the root
of local repos; remote repos keep it on the instance only (default ``main``).

Icechunk sessions stage changes in memory: ``writable()`` opens a session on the
current branch, and ``commit()`` makes the staged changes durable as one snapshot.
Batch writes into few, large commits.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import ScizarrError
from .storage import is_remote, storage_for

if TYPE_CHECKING:
    import zarr
    from icechunk import SnapshotInfo

DEFAULT_BRANCH = "main"
_HEAD_FILE = "scizarr_head"


class Repo:
    """Open an existing repo; use :meth:`init` to create one from a zarr store."""

    def __init__(self, path: str | Path, *, branch: str | None = None) -> None:
        import icechunk

        self.path = str(path)
        try:
            self._repo = icechunk.Repository.open(storage_for(self.path))
        except Exception as exc:
            raise ScizarrError(f"No icechunk repository at '{self.path}': {exc}") from exc
        self._session: Any = None

        requested = branch or self._load_head()
        branches = self._repo.list_branches()
        if requested in branches:
            self._branch = requested
        elif branch is not None:
            raise ScizarrError(f"Branch '{branch}' does not exist in '{self.path}'")
        elif DEFAULT_BRANCH in branches:
            if requested is not None:
                print(
                    f"warning: HEAD branch '{requested}' is gone, falling back to '{DEFAULT_BRANCH}'",
                    file=sys.stderr,
                )
            self._branch = DEFAULT_BRANCH
        else:
            raise ScizarrError(f"No '{DEFAULT_BRANCH}' branch in '{self.path}'")

    @classmethod
    def init(
        cls, zarr_path: str | Path, out_path: str | Path, *, message: str | None = None
    ) -> "Repo":
        """Create a repo at ``out_path`` seeded from the zarr store at ``zarr_path``.

        The whole store is imported as a single commit on ``main``.
        """
        import icechunk
        import zarr

        from .copy import copy_group

        out = str(out_path)
        if not is_remote(out):
            p = Path(out)
            if p.exists() and any(p.iterdir()):
                raise ScizarrError(f"Output path already exists and is not empty: {out}")
            p.parent.mkdir(parents=True, exist_ok=True)
        try:
            src = zarr.open_group(str(zarr_path), mode="r")
        except Exception as exc:
            raise ScizarrError(f"Could not open source zarr '{zarr_path}': {exc}") from exc
        try:
            ic_repo = icechunk.Repository.create(storage_for(out))
        except Exception as exc:
            raise ScizarrError(f"Could not create icechunk repository at '{out}': {exc}") from exc

        session = ic_repo.writable_session(DEFAULT_BRANCH)
        dst = zarr.open_group(store=session.store, mode="w")
        copy_group(src, dst)
        session.commit(message or f"init from {zarr_path}")

        repo = cls(out)
        repo._save_head()
        return repo

    # -- branch state --------------------------------------------------------

    @property
    def branch(self) -> str:
        """Name of the current branch."""
        return self._branch

    def branches(self) -> list[str]:
        return sorted(self._repo.list_branches())

    def checkout(self, branch: str, *, create: bool = False) -> str:
        """Switch the current branch (persisted for local repos); return its tip snapshot.

        ``create=True`` branches off the current tip when ``branch`` doesn't exist.
        """
        self._reject_uncommitted()
        if branch not in self._repo.list_branches():
            if not create:
                raise ScizarrError(f"No branch '{branch}' (pass create=True / -b to create it)")
            self._repo.create_branch(branch, self._repo.lookup_branch(self._branch))
        self._branch = branch
        self._session = None
        self._save_head()
        return self._repo.lookup_branch(branch)

    # -- reading & writing ---------------------------------------------------

    def root(self, *, snapshot_id: str | None = None) -> "zarr.Group":
        """Read-only zarr group at the current branch tip (or a specific snapshot)."""
        import zarr

        if snapshot_id is not None:
            session = self._repo.readonly_session(snapshot_id=snapshot_id)
        else:
            session = self._repo.readonly_session(branch=self._branch)
        return zarr.open_group(store=session.store, mode="r")

    def writable(self) -> "zarr.Group":
        """Zarr group on a writable session at the current branch tip.

        Changes stage in memory until :meth:`commit`; repeated calls reuse the open session.
        """
        import zarr

        if self._session is None or self._session.read_only:
            self._session = self._repo.writable_session(self._branch)
        return zarr.open_group(store=self._session.store, mode="a")

    def commit(
        self,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
        allow_empty: bool = False,
    ) -> str:
        """Commit the open writable session to the current branch; return the snapshot id."""
        if self._session is None or self._session.read_only:
            raise ScizarrError("No writable session — call writable() and make changes first")
        try:
            snapshot_id = self._session.commit(message, metadata, allow_empty=allow_empty)
        except Exception as exc:
            raise ScizarrError(f"Commit on '{self._branch}' failed: {exc}") from exc
        self._session = None
        return snapshot_id

    def discard(self) -> None:
        """Drop any uncommitted changes and close the writable session."""
        if self._session is not None and not self._session.read_only:
            self._session.discard_changes()
        self._session = None

    # -- history -------------------------------------------------------------

    def log(self, *, branch: str | None = None) -> list["SnapshotInfo"]:
        """Snapshots on the current (or given) branch, newest first."""
        target = branch or self._branch
        if target not in self._repo.list_branches():
            raise ScizarrError(f"No branch '{target}' in '{self.path}'")
        return list(self._repo.ancestry(branch=target))

    def tree(self) -> dict[str, list["SnapshotInfo"]]:
        """All branches mapped to their snapshots, newest first."""
        return {b: list(self._repo.ancestry(branch=b)) for b in self.branches()}

    def cherrypick(self, snapshot_id: str) -> str:
        """Point the current branch at ``snapshot_id`` (history reset, like git reset --hard)."""
        self._reject_uncommitted()
        try:
            info = self._repo.lookup_snapshot(snapshot_id)
        except Exception as exc:
            raise ScizarrError(f"Unknown snapshot '{snapshot_id}': {exc}") from exc
        self._repo.reset_branch(self._branch, info.id)
        self._session = None
        return info.id

    # -- internals -----------------------------------------------------------

    def _reject_uncommitted(self) -> None:
        if (
            self._session is not None
            and not self._session.read_only
            and self._session.has_uncommitted_changes
        ):
            raise ScizarrError(
                f"Uncommitted changes on '{self._branch}' — commit() or discard() first"
            )

    def _head_path(self) -> Path | None:
        return None if is_remote(self.path) else Path(self.path) / _HEAD_FILE

    def _load_head(self) -> str | None:
        head = self._head_path()
        if head is not None and head.is_file():
            return head.read_text().strip() or None
        return None

    def _save_head(self) -> None:
        head = self._head_path()
        if head is not None:
            head.write_text(self._branch + "\n")
