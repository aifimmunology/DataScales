"""argparse CLI for scizarr-ic (``scizarr-ic`` / ``scz``).

Subcommands mirror git: ``init``, ``log``, ``tree``, ``checkout``, ``cherrypick``.
Every non-init command takes the repo path (``-C``/``--repo``, default ``.``).
Errors print to stderr and exit 1; normal output stays on stdout for piping.

``commit`` is intentionally Python-API only (``Repo.commit``): icechunk stages
edits in a session's memory, so there is nothing for a fresh CLI process to commit.
"""
from __future__ import annotations

import argparse
import os
import sys

from .errors import ScizarrError
from .repo import Repo


def _quiet_icechunk_logs() -> None:
    """Silence icechunk's benign Rust-core WARN spam unless ICECHUNK_LOG is set."""
    if os.environ.get("ICECHUNK_LOG"):
        return
    try:
        import icechunk

        icechunk.set_logs_filter("error")
    except Exception:
        pass


def _short(snapshot_id: str) -> str:
    return snapshot_id[:12]


def _fmt_when(dt) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _cmd_init(args) -> int:
    repo = Repo.init(args.zarr, args.out, message=args.message)
    print(f"Initialized icechunk repo at {repo.path} (branch '{repo.branch}')")
    return 0


def _cmd_log(args) -> int:
    repo = Repo(args.repo)
    for snap in repo.log(branch=args.branch):
        if args.oneline:
            print(f"{_short(snap.id)} {snap.message}")
        else:
            print(f"commit {snap.id}")
            print(f"Date:   {_fmt_when(snap.written_at)}")
            print(f"\n    {snap.message}\n")
    return 0


def _cmd_tree(args) -> int:
    repo = Repo(args.repo)
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


def _cmd_checkout(args) -> int:
    repo = Repo(args.repo)
    tip = repo.checkout(args.branch, create=args.create)
    verb = "Created and switched to" if args.create else "Switched to"
    print(f"{verb} branch '{args.branch}' (tip {_short(tip)})")
    return 0


def _cmd_cherrypick(args) -> int:
    repo = Repo(args.repo)
    resolved = repo.cherrypick(args.snapshot)
    print(f"Branch '{repo.branch}' now at {_short(resolved)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scizarr-ic", description="Git-like version control for Zarr stores (Icechunk)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_repo_arg(p):
        p.add_argument(
            "-C", "--repo", default=".", help="Path/URI of the icechunk repo (default: .)"
        )

    p_init = sub.add_parser("init", help="Create a repo from an existing zarr store")
    p_init.add_argument("zarr", help="Source zarr store path")
    p_init.add_argument("out", help="Output icechunk repo path/URI")
    p_init.add_argument("-m", "--message", help="Commit message for the import")
    p_init.set_defaults(func=_cmd_init)

    p_log = sub.add_parser("log", help="Show commit history for a branch")
    add_repo_arg(p_log)
    p_log.add_argument("-b", "--branch", help="Branch to log (default: current)")
    p_log.add_argument("--oneline", action="store_true", help="One line per commit")
    p_log.set_defaults(func=_cmd_log)

    p_tree = sub.add_parser("tree", help="Show all branches and their commits")
    add_repo_arg(p_tree)
    p_tree.set_defaults(func=_cmd_tree)

    p_co = sub.add_parser("checkout", help="Switch the current branch")
    add_repo_arg(p_co)
    p_co.add_argument("branch", help="Branch to switch to")
    p_co.add_argument(
        "-b", "--create", action="store_true", help="Create the branch off the current tip"
    )
    p_co.set_defaults(func=_cmd_checkout)

    p_cp = sub.add_parser(
        "cherrypick", help="Reset the current branch to a snapshot (change the store's state)"
    )
    add_repo_arg(p_cp)
    p_cp.add_argument("snapshot", help="Snapshot id to point the branch at")
    p_cp.set_defaults(func=_cmd_cherrypick)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _quiet_icechunk_logs()
    try:
        return args.func(args)
    except ScizarrError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
