from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import zarr

if TYPE_CHECKING:
    from icechunk import SnapshotInfo

SHORT_ID = 12


def short(snapshot_id: str) -> str:
    return snapshot_id[:SHORT_ID]


def fmt_snapshot(info: SnapshotInfo) -> str:
    return f"{short(info.id)}  {info.written_at.astimezone():%Y-%m-%d %H:%M}  {info.message}"


class Branches(list[str]):
    """Sorted branch names; shown one per line with ``*`` on the current branch."""

    def __init__(self, names: list[str], *, current: str) -> None:
        super().__init__(names)
        self.current = current

    def __repr__(self) -> str:
        return "\n".join(("* " if b == self.current else "  ") + b for b in self)


class Log(list["SnapshotInfo"]):
    """``SnapshotInfo`` list, newest first; shown one commit per line."""

    def __repr__(self) -> str:
        return "\n".join(fmt_snapshot(s) for s in self) or "(no commits)"


class Tree:
    """Every branch's history as icechunk's plain-text commit graph (display only)."""

    def __init__(self, branch: str, graph: str) -> None:
        self.branch = branch
        self.graph = graph

    def __repr__(self) -> str:
        return f"On branch {self.branch}\n{self.graph}"


@dataclass(frozen=True, repr=False)
class Group(zarr.Group):
    """The root ``zarr.Group`` of a repo whose repr describes the group, not the session.

    Only the root returned by :meth:`~annizarr.ic.Repo.open_zarr` is this type; subgroups
    reached through it are plain ``zarr.Group`` with zarr's default repr.
    """

    _location: str = ""
    _branch: str = ""

    def __repr__(self) -> str:
        session = self.store.session  # type: ignore[attr-defined]  # IcechunkStore, untyped by zarr's Store ABC
        at = f"branch {self._branch}" if self._branch else "detached"
        mode = "read-only" if self.read_only else "writable"
        if not self.read_only and session.has_uncommitted_changes:
            mode += ", uncommitted changes"
        head = f"<Group {self.name!r} at {self._location}  {at}  snapshot {short(session.snapshot_id)}  {mode}>"
        try:
            arrays, groups = sorted(self.array_keys()), sorted(self.group_keys())
        except Exception:
            return head
        lines = [head]
        if arrays:
            lines.append(f"  arrays: {', '.join(arrays)}")
        if groups:
            lines.append(f"  groups: {', '.join(groups)}")
        return "\n".join(lines)


def _alias_in_anndata_registry() -> None:
    # anndata dispatches read/write on the exact store type, so its zarr.Group entries are
    # repeated for Group (private registry, verified against .claude/vendor/anndata 0.12.19)
    from anndata._io.specs import _LAZY_REGISTRY, _REGISTRY

    for registry in (_REGISTRY, _LAZY_REGISTRY):
        for key, func in list(registry.read.items()):
            if key[0] is zarr.Group:
                registry.read.setdefault((Group, *key[1:]), func)
        for key, func in list(registry.write.items()):
            if key[0] is zarr.Group:
                registry.write.setdefault((Group, *key[1:]), func)


_alias_in_anndata_registry()
