"""Git-like wrapper around an icechunk repository.

One ``Repo`` == one icechunk repository (local dir or s3://—gs:// URI).

**Read/write split.** Reads (``log``, ``tree``, ``root``...) go through ``path`` as
given. Writes go to ``origin`` when one is given (``Repo(path, origin=...)``, CLI
``--origin``, or ``SCIZARR_IC_ORIGIN``) — e.g. the ``s3://`` prefix behind a read-only
local mirror — and to ``path`` otherwise. The origin is opened lazily on the first
write, with credentials from the environment; once opened it also serves reads in this
process, since a mirror can lag behind fresh writes. A read-only ``path`` with no
origin is reads-only.

**HEAD.** The current branch persists across CLI invocations locally: a
``scizarr_head`` file for writable local repos, a per-user sidecar under
``$SCIZARR_IC_HOME`` otherwise (see ``head.py``).

Icechunk sessions stage changes in memory: ``writable()`` opens a session on the
current branch, and ``commit()`` makes the staged changes durable as one snapshot.
Batch writes into few, large commits.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import ScizarrError
from .head import HeadStore
from .storage import ENV_ORIGIN, canonical_location, is_readonly_path, is_remote, storage_for

if TYPE_CHECKING:
    import zarr
    from icechunk import SnapshotInfo

DEFAULT_BRANCH = "main"


def _fresh_destination(location: str | Path) -> str:
    """Validate a location a new repo will be written to; return it as ``str``.

    Rejects read-only paths, non-empty local dirs and remote prefixes that already
    hold a repo; creates the local parent directory.
    """
    import icechunk

    out = str(location)
    if is_readonly_path(out):
        raise ScizarrError(f"Destination '{out}' is read-only — use a writable path or s3://bucket/prefix.")
    if is_remote(out):
        if icechunk.Repository.exists(storage_for(out)):
            raise ScizarrError(f"Destination already holds an icechunk repo: {out}")
        return out
    p = Path(out)
    if p.exists() and any(p.iterdir()):
        raise ScizarrError(f"Destination already exists and is not empty: {out}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return out


class Repo:
    """Open an existing repo; use :meth:`init` / :meth:`create` to make one."""

    def __init__(
        self,
        path: str | Path,
        *,
        branch: str | None = None,
        origin: str | None = None,
    ) -> None:
        import icechunk

        self.path = str(path)
        try:
            self._reader = icechunk.Repository.open(storage_for(self.path))
        except Exception as exc:
            raise ScizarrError(f"No icechunk repository at '{self.path}': {exc}") from exc
        self._writer: Any = None
        self._session: Any = None

        self.readonly_path = is_readonly_path(self.path)
        self.origin = str(origin or os.environ.get(ENV_ORIGIN) or "") or (
            None if self.readonly_path else self.path
        )
        self._head = HeadStore(
            key=canonical_location(self.origin or self.path),
            in_repo_dir=self.path if self.origin == self.path and not is_remote(self.path) else None,
        )

        requested = branch or self._head.load()
        if requested is not None and self._branch_exists(requested):
            self._branch = requested
        elif branch is not None:
            raise ScizarrError(f"Branch '{branch}' does not exist in '{self.path}'")
        elif self._branch_exists(DEFAULT_BRANCH, consult_origin=False):
            if requested is not None:
                print(
                    f"warning: HEAD branch '{requested}' is gone, falling back to '{DEFAULT_BRANCH}'",
                    file=sys.stderr,
                )
            self._branch = DEFAULT_BRANCH
        else:
            raise ScizarrError(f"No '{DEFAULT_BRANCH}' branch in '{self.path}'")

    # -- creating ------------------------------------------------------------

    @classmethod
    def create(cls, out_path: str | Path) -> "Repo":
        """Create an EMPTY repo at a fresh, writable ``out_path`` (local dir or URI).

        It starts with icechunk's "Repository initialized" snapshot on ``main``.
        """
        import icechunk

        out = _fresh_destination(out_path)
        try:
            icechunk.Repository.create(storage_for(out))
        except Exception as exc:
            raise ScizarrError(f"Could not create icechunk repository at '{out}': {exc}") from exc
        repo = cls(out)
        repo._head.save(repo._branch)
        return repo

    @classmethod
    def init(
        cls, zarr_path: str | Path, out_path: str | Path, *, message: str | None = None
    ) -> "Repo":
        """Create a repo at ``out_path`` seeded from the zarr store at ``zarr_path``.

        The whole store is imported as a single commit on ``main``.
        """
        import zarr

        from .copy import copy_group

        try:
            src = zarr.open_group(str(zarr_path), mode="r")
        except Exception as exc:
            raise ScizarrError(f"Could not open source zarr '{zarr_path}': {exc}") from exc
        repo = cls.create(out_path)
        copy_group(src, repo.writable())
        repo.commit(message or f"init from {zarr_path}")
        return repo

    @classmethod
    def exists(cls, path: str | Path) -> bool:
        """Does an icechunk repo exist at ``path``? Never creates anything."""
        import icechunk

        p = str(path)
        if not is_remote(p) and not os.path.exists(p):
            return False
        try:
            return bool(icechunk.Repository.exists(storage_for(p)))
        except Exception as exc:
            raise ScizarrError(f"Could not check for a repository at '{p}': {exc}") from exc

    def copy(self, dest: str | Path) -> "Repo":
        """Copy this repo — every branch and snapshot, ids intact — to a fresh ``dest``.

        Local dir or ``s3://`` prefix (the latter needs ``boto3``). Reads come from
        ``path`` as given; nothing is written into the source. The current branch
        carries over when the copy has it.
        """
        from .copy import check_copyable, copy_repo

        dest = str(dest)
        check_copyable(self.path, dest)
        dest = _fresh_destination(dest)
        copy_repo(self.path, dest)
        repo = type(self)(dest)
        if self._branch in repo.branches():
            repo.checkout(self._branch)
        else:
            repo._head.save(repo._branch)
        return repo

    # -- location ------------------------------------------------------------

    @property
    def resolved(self) -> bool:
        """True when writes go somewhere other than ``path``."""
        return self.origin is not None and canonical_location(self.origin) != canonical_location(
            self.path
        )

    # -- branch state --------------------------------------------------------

    @property
    def branch(self) -> str:
        """Name of the current branch."""
        return self._branch

    def branches(self) -> list[str]:
        return sorted(self._repo.list_branches())

    def checkout(self, branch: str, *, create: bool = False) -> str:
        """Switch the current branch (persisted locally); return its tip snapshot.

        ``create=True`` branches off the current tip when ``branch`` doesn't exist.
        """
        self._reject_uncommitted()
        if not self._branch_exists(branch):
            if not create:
                raise ScizarrError(f"No branch '{branch}' (pass create=True / -b to create it)")
            writer = self._writer_repo()
            writer.create_branch(branch, writer.lookup_branch(self._branch))
        self._branch = branch
        self._session = None
        self._head.save(branch)
        return self._repo.lookup_branch(branch)

    # -- reading & writing ---------------------------------------------------

    def root(self, *, snapshot_id: str | None = None) -> "zarr.Group":
        """Read-only zarr group at the current branch tip (or a specific snapshot)."""
        import zarr

        if snapshot_id is not None:
            session = self._repo.readonly_session(snapshot_id=snapshot_id)
        else:
            session = self.session()
        return zarr.open_group(store=session.store, mode="r")

    def writable(self) -> "zarr.Group":
        """Zarr group on a writable session at the current branch tip.

        Changes stage in memory until :meth:`commit`; repeated calls reuse the open session.
        """
        import zarr

        return zarr.open_group(store=self.session(writable=True).store, mode="a")

    def session(self, *, writable: bool = False):
        """The underlying icechunk session, for callers that need more than a zarr group.

        ``writable=True`` returns the one open writable session (shared with
        :meth:`writable`, committed by :meth:`commit`); otherwise a fresh read-only
        session at the current branch tip.
        """
        if not writable:
            return self._repo.readonly_session(branch=self._branch)
        if self._session is None or self._session.read_only:
            self._session = self._writer_repo().writable_session(self._branch)
        return self._session

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
        if not self._branch_exists(target):
            raise ScizarrError(f"No branch '{target}' in '{self.path}'")
        return list(self._repo.ancestry(branch=target))

    def tree(self) -> dict[str, list["SnapshotInfo"]]:
        """All branches mapped to their snapshots, newest first."""
        return {b: list(self._repo.ancestry(branch=b)) for b in self.branches()}

    def cherrypick(self, snapshot_id: str) -> str:
        """Point the current branch at ``snapshot_id`` (history reset, like git reset --hard)."""
        self._reject_uncommitted()
        writer = self._writer_repo()
        try:
            info = writer.lookup_snapshot(snapshot_id)
        except Exception as exc:
            raise ScizarrError(f"Unknown snapshot '{snapshot_id}': {exc}") from exc
        writer.reset_branch(self._branch, info.id)
        self._session = None
        return info.id

    # -- internals -----------------------------------------------------------

    @property
    def _repo(self):
        """Repository used for reads: the origin once opened (authoritative), else ``path``."""
        return self._writer if self._writer is not None else self._reader

    def _writer_repo(self):
        """Repository at the writable origin, opened on first use."""
        if self._writer is not None:
            return self._writer
        if self.origin is None:
            raise ScizarrError(
                f"'{self.path}' is read-only and no writable origin was given. Pass "
                f"origin=... (CLI: --origin s3://bucket/prefix, or {ENV_ORIGIN}), or take a "
                f"writable copy: scizarr-ic copy -C {self.path} DEST"
            )
        if not self.resolved:
            self._writer = self._reader
            return self._writer

        import icechunk

        try:
            self._writer = icechunk.Repository.open(storage_for(self.origin))
        except Exception as exc:
            hint = ""
            if is_remote(self.origin):
                hint = (
                    " Object-store credentials come from the environment (AWS_* variables, "
                    "a profile, or an instance/container role) — check this process has "
                    "write access to the bucket."
                )
            raise ScizarrError(
                f"Cannot open the writable origin '{self.origin}' (for '{self.path}'): {exc}.{hint}"
            ) from exc
        return self._writer

    def _branch_exists(self, name: str, *, consult_origin: bool = True) -> bool:
        """Is ``name`` a branch? Falls back to the origin when the read view may be stale.

        A read-only mirror is a cached view of the origin: a branch created moments ago
        (possibly by another process) may not show up yet. If the origin is a different
        location and hasn't been opened, open it and re-check there.
        """
        if name in self._repo.list_branches():
            return True
        if consult_origin and self.resolved and self._writer is None:
            try:
                writer = self._writer_repo()
            except ScizarrError:
                return False
            return name in writer.list_branches()
        return False

    def _reject_uncommitted(self) -> None:
        if (
            self._session is not None
            and not self._session.read_only
            and self._session.has_uncommitted_changes
        ):
            raise ScizarrError(
                f"Uncommitted changes on '{self._branch}' — commit() or discard() first"
            )
