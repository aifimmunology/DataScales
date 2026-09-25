"""Git-like wrapper around an icechunk repository.

One ``Repo`` == one icechunk repository (local dir or s3://—gs:// URI).

**Read/write split.** Reads (``log``, ``tree``, ``root``...) go through ``path`` as
given. Writes go to the repo's *origin*: the same location for a writable local dir or
a remote URI, but for a READ-ONLY local mount — e.g. a Code Ocean *linked* data asset
under ``/data`` that mirrors an S3 prefix — the origin is the ``s3://`` URL stamped in
the repo's metadata (``origin_url``; see ``Repo.init``), opened lazily on the first
write with credentials from the environment. Once the origin has been opened it also
serves reads in this process, since a mount can lag behind fresh writes.

A read-only mount that is a *frozen copy* of its source (Code Ocean's internal EFS
assets; see ``storage.is_frozen_mount``) has no usable origin: writes are refused and
``copy`` is the way forward. ``--origin`` / ``SCIZARR_IC_ORIGIN`` still force one.

**HEAD.** The current branch persists across CLI invocations locally (never inside a
read-only repo): a ``scizarr_head`` file for writable local repos, a per-user sidecar
under ``$SCIZARR_IC_HOME`` otherwise (see ``head.py``).

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
from .storage import (
    ENV_ORIGIN,
    ORIGIN_KEY,
    canonical_location,
    is_frozen_mount,
    is_readonly_path,
    is_remote,
    mount_fstype,
    origin_of,
    storage_for,
)

if TYPE_CHECKING:
    import zarr
    from icechunk import SnapshotInfo

DEFAULT_BRANCH = "main"


def _origin_for(location: str) -> str:
    """What to stamp as ``origin_url`` for a repo created at ``location``."""
    return location if is_remote(location) else os.path.abspath(location)


def _fresh_destination(location: str | Path) -> str:
    """Validate a location a new repo will be written to; return it as ``str``.

    Rejects read-only paths (mounted data assets), non-empty local dirs and remote
    prefixes that already hold a repo; creates the local parent directory.
    """
    import icechunk

    out = str(location)
    if is_readonly_path(out):
        raise ScizarrError(
            f"Destination '{out}' is read-only (a mounted data asset?) — "
            "use a writable location, e.g. /results/..., /scratch/... or s3://bucket/prefix."
        )
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
    """Open an existing repo; use :meth:`init` to create one from a zarr store."""

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
        self.frozen = is_frozen_mount(self.path)
        self.origin = self._resolve_origin(origin)
        self._head = HeadStore(
            key=canonical_location(self.origin or self.path),
            in_repo_dir=(
                self.path
                if not is_remote(self.path) and not self.readonly_path and not self.resolved
                else None
            ),
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

    @classmethod
    def create(cls, out_path: str | Path) -> "Repo":
        """Create an EMPTY repo at a fresh, writable ``out_path`` (local dir or URI).

        The repo starts with icechunk's "Repository initialized" snapshot on ``main``
        and its writable location stamped into the metadata (``origin_url``) so a
        read-only mount of it later (a data asset) can still resolve where writes go.
        """
        import icechunk

        out = _fresh_destination(out_path)
        try:
            ic_repo = icechunk.Repository.create(storage_for(out))
        except Exception as exc:
            raise ScizarrError(f"Could not create icechunk repository at '{out}': {exc}") from exc
        ic_repo.set_metadata({ORIGIN_KEY: _origin_for(out)})
        repo = cls(out)
        repo._head.save(repo._branch)
        return repo

    @classmethod
    def init(
        cls, zarr_path: str | Path, out_path: str | Path, *, message: str | None = None
    ) -> "Repo":
        """Create a repo at ``out_path`` seeded from the zarr store at ``zarr_path``.

        The whole store is imported as a single commit on ``main`` (see :meth:`create`).
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

        This is the escape hatch for a read-only asset you cannot (or should not)
        write back to: the copy is stamped as its own origin, so it is fully writable
        wherever it lands (local dir or ``s3://`` prefix). Reads come from ``path`` as
        given; nothing is written into the source. The current branch carries over.
        """
        import icechunk

        from .copy import check_copyable, copy_repo

        dest = str(dest)
        check_copyable(self.path, dest)
        dest = _fresh_destination(dest)
        copy_repo(self.path, dest)
        icechunk.Repository.open(storage_for(dest)).update_metadata({ORIGIN_KEY: _origin_for(dest)})
        repo = type(self)(dest)
        if self._branch in repo.branches():  # a stale mount may lack a branch HEAD names
            repo.checkout(self._branch)
        else:
            repo._head.save(repo._branch)
        return repo

    # -- location ------------------------------------------------------------

    @property
    def resolved(self) -> bool:
        """True when writes go somewhere other than ``path`` (a mount resolved to its origin)."""
        return self.origin is not None and canonical_location(self.origin) != canonical_location(
            self.path
        )

    @property
    def read_only(self) -> bool:
        """True when ``path`` can't be written and no origin is known (or usable) — reads only."""
        return self.readonly_path and self.origin is None

    def origin_url(self) -> str | None:
        """The ``origin_url`` stamped in the repo metadata (None if never stamped)."""
        return origin_of(self._repo)

    def set_origin(self, url: str) -> None:
        """Stamp/replace the repo's writable location (needs write access to the repo)."""
        self._writer_repo().update_metadata({ORIGIN_KEY: str(url)})

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
            session = self._repo.readonly_session(branch=self._branch)
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

    def _resolve_origin(self, explicit: str | None) -> str | None:
        if explicit:
            return str(explicit)
        env = os.environ.get(ENV_ORIGIN)
        if env:
            return env
        if self.frozen:
            return None  # a copy, not a view: the stamped origin is disconnected
        if self.readonly_path:
            return origin_of(self._reader)
        return self.path

    def _writer_repo(self):
        """Repository at the writable origin, opened on first use."""
        if self._writer is not None:
            return self._writer
        if self.origin is None and self.frozen:
            raise ScizarrError(
                f"'{self.path}' is a frozen copy (read-only {mount_fstype(self.path)} mount — "
                f"an internal data asset): its stamped origin "
                f"'{self.origin_url() or '-'}' is not connected to this mount, so writes "
                "there would never show up here. Take a writable copy instead: "
                f"scizarr-ic copy -C {self.path} DEST (Repo.copy). To force an origin "
                f"anyway pass origin=... (CLI: --origin URL, or {ENV_ORIGIN})."
            )
        if self.origin is None:
            raise ScizarrError(
                f"'{self.path}' is read-only (a mounted data asset?) and has no "
                f"'{ORIGIN_KEY}' stamped in its metadata, so there is nowhere to write. "
                f"Pass origin=... (CLI: --origin s3://bucket/prefix, or {ENV_ORIGIN}), or "
                "stamp it once against the writable copy: "
                "scizarr-ic origin -C s3://bucket/prefix s3://bucket/prefix"
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
                    "a profile, or the container role / Code Ocean AWS secret) — check this "
                    "process has write access to the bucket."
                )
            raise ScizarrError(
                f"Cannot open the writable origin '{self.origin}' "
                f"(resolved from '{self.path}'): {exc}.{hint}"
            ) from exc
        return self._writer

    def _branch_exists(self, name: str, *, consult_origin: bool = True) -> bool:
        """Is ``name`` a branch? Falls back to the origin when the read view may be stale.

        A read-only mount is a cached view of the origin: a branch created moments ago
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
