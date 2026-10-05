"""Git-like wrapper around an icechunk repository.

One ``Repo`` == one icechunk repository (local dir or s3://, gs:// URI), opened on a
branch: ``main`` unless ``branch=`` says otherwise. ``checkout`` switches branches for
this object only; icechunk has no notion of a current branch, so nothing is persisted.

**Read/write split.** Reads (``log``, ``tree``, ``open_zarr("r")``...) go through ``path``
as given. Writes go to ``origin`` when one is given (``Repo(path, origin=...)``, e.g. the
``s3://`` prefix behind a read-only local mirror) and to ``path`` otherwise. The origin is
opened lazily on the first write, with credentials from the environment; once opened it
also serves reads in this process, since a mirror can lag behind fresh writes. A
read-only ``path`` with no origin is reads-only.

Icechunk sessions stage changes in memory: ``open_zarr("w")`` opens a session on the
current branch, and ``commit()`` makes the staged changes durable as one snapshot.
Batch writes into few, large commits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from annizarr._storage import (
    canonical_location,
    explain_open_failure,
    is_readonly_path,
    is_remote,
    open_repository,
    require_icechunk,
    storage_for,
)
from annizarr.errors import RepoError

if TYPE_CHECKING:
    from annizarr.ic._display import Branches, Group, Log, Tree

DEFAULT_BRANCH = "main"


def _fresh_destination(location: str | Path) -> str:
    """Validate a location a new repo will be written to; return it as ``str``.

    Rejects read-only paths, non-empty local dirs and remote prefixes that already
    hold a repo; creates the local parent directory.
    """
    icechunk = require_icechunk()

    out = str(location)
    if is_readonly_path(out):
        raise RepoError(f"Destination '{out}' is read-only: use a writable path or s3://bucket/prefix.")
    if is_remote(out):
        if icechunk.Repository.exists(storage_for(out)):
            raise RepoError(f"Destination already holds an icechunk repo: {out}")
        return out
    p = Path(out)
    if p.exists() and any(p.iterdir()):
        raise RepoError(f"Destination already exists and is not empty: {out}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return out


class Repo:
    """Open an existing repo on a branch; use :meth:`init` / :meth:`create` to make one.

    Parameters
    ----------
    path
        Repository location to open: a local directory or an ``s3://``/``gs://`` URI.
    branch
        Branch to check out; defaults to ``"main"``.
    origin
        Writable location for a read-only ``path``; writes go here instead of ``path``.
    anonymous
        Read a public bucket with unsigned requests; otherwise credentials come from
        the environment. Writes always use the environment.

    Raises
    ------
    RepoError
        No repository exists at ``path``, or ``branch`` (or the default ``"main"``)
        doesn't exist there.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        branch: str | None = None,
        origin: str | None = None,
        anonymous: bool = False,
    ) -> None:
        self.path = str(path)
        self.anonymous = anonymous
        self._reader = open_repository(self.path, anonymous=anonymous)
        self._writer: Any = None
        self._session: Any = None

        self.readonly_path = is_readonly_path(self.path)
        self.origin = str(origin) if origin else (None if self.readonly_path else self.path)
        self._branch = branch or DEFAULT_BRANCH
        if not self._branch_exists(self._branch):
            raise RepoError(f"No branch '{self._branch}' in '{self.path}'")

    def __repr__(self) -> str:
        from annizarr.ic._display import fmt_snapshot

        lines = [f"Repo({self.path!r})", f"  branch: {self._branch}"]
        try:
            tip = next(self._repo.ancestry(branch=self._branch), None)
        except Exception:
            tip = None
        if tip is not None:
            lines.append(f"  tip:    {fmt_snapshot(tip)}")
        if self.resolved:
            lines.append(f"  writes: {self.origin}")
        if self._dirty:
            lines.append("  staged: uncommitted changes")
        return "\n".join(lines)

    @classmethod
    def create(cls, out_path: str | Path) -> Repo:
        """Create an EMPTY repo at a fresh, writable ``out_path`` (local dir or URI).

        It starts with icechunk's "Repository initialized" snapshot on ``main``.
        """
        icechunk = require_icechunk()

        out = _fresh_destination(out_path)
        try:
            icechunk.Repository.create(storage_for(out))
        except Exception as exc:
            raise RepoError(f"Could not create icechunk repository at '{out}': {exc}") from exc
        return cls(out)

    @classmethod
    def init(cls, zarr_path: str | Path, out_path: str | Path, *, message: str | None = None) -> Repo:
        """Create a repo at ``out_path`` seeded from the zarr store at ``zarr_path``.

        The whole store is imported as a single commit on ``main``.
        """
        import zarr

        from annizarr.ic._copy import copy_group

        try:
            src = zarr.open_group(str(zarr_path), mode="r")
        except Exception as exc:
            raise RepoError(f"Could not open source zarr '{zarr_path}': {exc}") from exc
        repo = cls.create(out_path)
        copy_group(src, repo.open_zarr("w"))
        repo.commit(message or f"init from {zarr_path}")
        return repo

    @classmethod
    def exists(cls, path: str | Path, *, anonymous: bool = False) -> bool:
        """Does an icechunk repo exist at ``path``? Never creates anything."""
        icechunk = require_icechunk()

        p = str(path)
        if not is_remote(p) and not os.path.exists(p):
            return False
        try:
            return bool(icechunk.Repository.exists(storage_for(p, anonymous=anonymous)))
        except Exception as exc:
            raise explain_open_failure(p, exc) from exc

    def copy(self, dest: str | Path) -> Repo:
        """Copy this repo, every branch and snapshot with ids intact, to a fresh ``dest``.

        Local dir or ``s3://`` prefix (the latter needs ``boto3``). Reads come from
        ``path`` as given; nothing is written into the source. The copy opens on the
        current branch when it has it.
        """
        from annizarr.ic._copy import check_copyable, copy_repo

        dest = str(dest)
        check_copyable(self.path, dest)
        dest = _fresh_destination(dest)
        copy_repo(self.path, dest)
        repo = type(self)(dest)
        if self._branch in repo.branches():
            repo.checkout(self._branch)
        return repo

    @property
    def resolved(self) -> bool:
        """True when writes go somewhere other than ``path``."""
        return self.origin is not None and canonical_location(self.origin) != canonical_location(self.path)

    @property
    def branch(self) -> str:
        """Name of the current branch."""
        return self._branch

    def branches(self) -> Branches:
        """Sorted branch names (a list); shown one per line, ``*`` on the current branch."""
        from annizarr.ic._display import Branches

        return Branches(sorted(self._repo.list_branches()), current=self._branch)

    def checkout(self, branch: str, *, create: bool = False) -> str:
        """Switch this object to ``branch`` and return its tip snapshot id.

        ``create=True`` branches off the current tip when ``branch`` doesn't exist.
        Switching needs no write access; creating does.
        """
        self._reject_uncommitted()
        if not self._branch_exists(branch):
            if not create:
                raise RepoError(f"No branch '{branch}' (pass create=True to create it)")
            writer = self._writer_repo()
            writer.create_branch(branch, writer.lookup_branch(self._branch))
        self._branch = branch
        self._session = None
        tip: str = self._repo.lookup_branch(branch)
        return tip

    def open_zarr(self, mode: str, *, snapshot_id: str | None = None, truncate: bool = False) -> Group:
        """Open the store's root zarr group: ``mode="r"`` (read) or ``mode="w"`` (write).

        ``"r"`` returns a read-only group at the current branch tip, or at
        ``snapshot_id`` when given (time-travel to a past commit). ``"w"`` returns a
        writable group on a session at the branch tip: edits stage in memory until
        :meth:`commit`, and repeated ``"w"`` calls reuse the one open session. Writes
        only ever go to the branch tip, so ``snapshot_id`` is rejected with ``"w"`` —
        read a past snapshot with ``"r"``, or :meth:`cherrypick` to reset the branch
        there first. (``"w"`` opens an *editable* session, like zarr ``mode="a"``, and
        never truncates the store, unless ``truncate=True``.)

        The returned root is a ``zarr.Group`` whose repr shows location, branch,
        snapshot and top-level members instead of icechunk's session dump.

        Parameters
        ----------
        truncate
            With ``mode="w"``, replace the branch's current root contents with an
            empty one (like zarr's own ``mode="w"``) instead of opening it editable —
            used when writing a brand-new output onto a branch that may already hold
            data from a previous write.
        """
        import zarr

        from annizarr.ic._display import Group

        if mode == "r":
            session = (
                self._repo.readonly_session(snapshot_id=snapshot_id)
                if snapshot_id is not None
                else self._repo.readonly_session(branch=self._branch)
            )
            group = zarr.open_group(store=session.store, mode="r")
        elif mode == "w":
            if snapshot_id is not None:
                raise RepoError(
                    "snapshot_id is read-only: writes go to the branch tip, not a past "
                    'snapshot. Read it with open_zarr("r", snapshot_id=...), or cherrypick() '
                    "to reset the branch there first."
                )
            if self._session is None or self._session.read_only:
                self._session = self._writer_repo().writable_session(self._branch)
            group = zarr.open_group(store=self._session.store, mode="w" if truncate else "a")
        else:
            raise RepoError(f"open_zarr mode must be 'r' or 'w', got {mode!r}")
        location = self.path
        if self._writer is not None and self.resolved and self.origin is not None:
            location = self.origin
        return Group(
            group._async_group,
            _location=location,
            _branch="" if snapshot_id is not None else self._branch,
        )

    def commit(
        self,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
        allow_empty: bool = False,
    ) -> str:
        """Commit the open writable session to the current branch; return the snapshot id."""
        if self._session is None or self._session.read_only:
            raise RepoError('No writable session: call open_zarr("w") and make changes first')
        try:
            snapshot_id: str = self._session.commit(message, metadata, allow_empty=allow_empty)
        except Exception as exc:
            raise RepoError(f"Commit on '{self._branch}' failed: {exc}") from exc
        self._session = None
        return snapshot_id

    def discard(self) -> None:
        """Drop any uncommitted changes and close the writable session."""
        if self._session is not None and not self._session.read_only:
            self._session.discard_changes()
        self._session = None

    def log(self, *, branch: str | None = None) -> Log:
        """Snapshots on the current (or given) branch, newest first (a list of
        ``SnapshotInfo``); shown one commit per line."""
        from annizarr.ic._display import Log

        target = branch or self._branch
        if not self._branch_exists(target):
            raise RepoError(f"No branch '{target}' in '{self.path}'")
        return Log(self._repo.ancestry(branch=target))

    def tree(self) -> Tree:
        """Every branch's history as one plain-text commit graph (display only)."""
        from annizarr.ic._display import Tree

        return Tree(self._branch, str(self._repo.ancestry_graph(plain=True)))

    def cherrypick(self, snapshot_id: str) -> str:
        """Point the current branch at ``snapshot_id`` (history reset, like git reset --hard)."""
        self._reject_uncommitted()
        writer = self._writer_repo()
        try:
            info = writer.lookup_snapshot(snapshot_id)
        except Exception as exc:
            raise RepoError(f"Unknown snapshot '{snapshot_id}': {exc}") from exc
        writer.reset_branch(self._branch, info.id)
        self._session = None
        resolved: str = info.id
        return resolved

    @property
    def _repo(self) -> Any:
        # repository used for reads: the origin once opened (authoritative), else path
        return self._writer if self._writer is not None else self._reader

    @property
    def _dirty(self) -> bool:
        return self._session is not None and not self._session.read_only and self._session.has_uncommitted_changes

    def _writer_repo(self) -> Any:
        # repository at the writable origin, opened on first use
        if self._writer is not None:
            return self._writer
        if self.origin is None:
            raise RepoError(
                f"'{self.path}' is read-only and no writable origin was given. Open it as "
                f"Repo(path, origin='s3://bucket/prefix') to write to the location behind "
                f"it, or take a writable copy with repo.copy(dest)."
            )
        if not self.resolved:
            self._writer = self._reader
            return self._writer
        try:
            self._writer = open_repository(self.origin)
        except RepoError as exc:
            raise RepoError(f"Cannot open the writable origin '{self.origin}' (for '{self.path}'): {exc}") from exc
        return self._writer

    def _branch_exists(self, name: str) -> bool:
        # a read-only mirror may be stale, so fall back to the origin when consulting it
        if name in self._repo.list_branches():
            return True
        if self.resolved and self._writer is None:
            try:
                writer = self._writer_repo()
            except RepoError:
                return False
            return name in writer.list_branches()
        return False

    def _reject_uncommitted(self) -> None:
        if self._dirty:
            raise RepoError(f"Uncommitted changes on '{self._branch}': commit() or discard() first")
