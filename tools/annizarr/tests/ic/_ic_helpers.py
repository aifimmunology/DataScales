from __future__ import annotations

from pathlib import Path


def snapshot_tree(root: Path) -> dict[str, int]:
    """{relative file: mtime_ns} — to prove a location was not touched."""
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
