from __future__ import annotations

import argparse
import logging
import os
from typing import TYPE_CHECKING

from annizarr.errors import StorageError

if TYPE_CHECKING:
    import datetime

    from annizarr.ic import Repo

_LOG = logging.getLogger(__name__)


def _quiet_icechunk_logs() -> None:
    # silences icechunk's benign Rust-core WARN spam unless ICECHUNK_LOG is set
    if os.environ.get("ICECHUNK_LOG"):
        return
    from annizarr._storage import require_icechunk

    try:
        icechunk = require_icechunk()
        icechunk.set_logs_filter("error")
    except (StorageError, AttributeError):
        pass


def _short(snapshot_id: str) -> str:
    return snapshot_id[:12]


def _fmt_when(dt: datetime.datetime) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _open(args: argparse.Namespace) -> Repo:
    from annizarr.ic import Repo

    return Repo(args.repo, origin=getattr(args, "origin", None))


def _note_resolved(repo: Repo) -> None:
    # tells the user when a write is going somewhere other than the path they passed
    if repo.resolved:
        _LOG.info(f"note: writing to {repo.origin} (resolved from {repo.path})")


def _cmd_init(args: argparse.Namespace) -> int:
    from annizarr.ic import Repo

    _quiet_icechunk_logs()
    repo = Repo.init(args.zarr, args.repo, message=args.message)
    _LOG.info(f"Initialized icechunk repo at {repo.path} (branch '{repo.branch}')")
    return 0


def _cmd_log(args: argparse.Namespace) -> int:
    _quiet_icechunk_logs()
    repo = _open(args)
    for snap in repo.log(branch=args.branch):
        if args.oneline:
            print(f"{_short(snap.id)} {snap.message}")
        else:
            print(f"commit {snap.id}")
            print(f"Date:   {_fmt_when(snap.written_at)}")
            print(f"\n    {snap.message}\n")
    return 0


def _cmd_tree(args: argparse.Namespace) -> int:
    _quiet_icechunk_logs()
    repo = _open(args)
    current = repo.branch
    tree = repo.tree()
    for i, (branch, snaps) in enumerate(sorted(tree.items())):
        marker = "*" if branch == current else " "
        print(f"{marker} {branch}")
        for j, snap in enumerate(snaps):
            connector = "└─" if j == len(snaps) - 1 else "├─"
            print(f"    {connector} {_short(snap.id)}  {snap.message}")
        if i != len(tree) - 1:
            print()
    return 0


def _cmd_checkout(args: argparse.Namespace) -> int:
    _quiet_icechunk_logs()
    repo = _open(args)
    creating = args.create and args.branch not in repo.branches()
    if creating:
        _note_resolved(repo)
    tip = repo.checkout(args.branch, create=args.create)
    verb = "Created and switched to" if creating else "Switched to"
    _LOG.info(f"{verb} branch '{args.branch}' (tip {_short(tip)})")
    return 0


def _cmd_cherrypick(args: argparse.Namespace) -> int:
    _quiet_icechunk_logs()
    repo = _open(args)
    _note_resolved(repo)
    resolved = repo.cherrypick(args.snapshot)
    _LOG.info(f"Branch '{repo.branch}' now at {_short(resolved)}")
    return 0


def _cmd_copy(args: argparse.Namespace) -> int:
    _quiet_icechunk_logs()
    repo = _open(args).copy(args.dest)
    _LOG.info(f"Copied {args.repo} -> {repo.path} (branch '{repo.branch}')")
    return 0


def _add_origin_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--origin",
        default=None,
        metavar="URL",
        help="Where writes go when the repo path is read-only (e.g. the s3://bucket/prefix behind a read-only mirror)",
    )


def add_ic_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_ic = subparsers.add_parser("ic", help="Git-like version control for Zarr stores (Icechunk)")
    ic_sub = p_ic.add_subparsers(dest="ic_command", required=True)

    p_init = ic_sub.add_parser("init", help="Create a repo from an existing zarr store")
    p_init.add_argument("zarr", metavar="SRC.zarr", help="Source zarr store path")
    p_init.add_argument("repo", metavar="REPO", help="Output icechunk repo path/URI")
    p_init.add_argument("-m", "--message", help="Commit message for the import")
    p_init.set_defaults(func=_cmd_init)

    p_log = ic_sub.add_parser("log", help="Show commit history for a branch")
    p_log.add_argument("repo", metavar="REPO")
    p_log.add_argument("-b", "--branch", help="Branch to log (default: current)")
    p_log.add_argument("--oneline", action="store_true", help="One line per commit")
    _add_origin_arg(p_log)
    p_log.set_defaults(func=_cmd_log)

    p_tree = ic_sub.add_parser("tree", help="Show all branches and their commits")
    p_tree.add_argument("repo", metavar="REPO")
    _add_origin_arg(p_tree)
    p_tree.set_defaults(func=_cmd_tree)

    p_co = ic_sub.add_parser("checkout", help="Switch the current branch")
    p_co.add_argument("repo", metavar="REPO")
    p_co.add_argument("branch", help="Branch to switch to")
    p_co.add_argument("-b", "--create", action="store_true", help="Create the branch off the current tip")
    _add_origin_arg(p_co)
    p_co.set_defaults(func=_cmd_checkout)

    p_cp = ic_sub.add_parser("cherrypick", help="Reset the current branch to a snapshot (change the store's state)")
    p_cp.add_argument("repo", metavar="REPO")
    p_cp.add_argument("snapshot", help="Snapshot id to point the branch at")
    _add_origin_arg(p_cp)
    p_cp.set_defaults(func=_cmd_cherrypick)

    p_copy = ic_sub.add_parser("copy", help="Copy the repo (all branches and snapshots) to a fresh writable location")
    p_copy.add_argument("repo", metavar="REPO")
    p_copy.add_argument("dest", help="Destination path/URI (local dir or s3://bucket/prefix)")
    _add_origin_arg(p_copy)
    p_copy.set_defaults(func=_cmd_copy)
