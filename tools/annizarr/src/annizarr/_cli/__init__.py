from __future__ import annotations

import sys

from .._version import __version__


def main(argv: list[str] | None = None) -> int:
    """Stub entry point: the full command surface is rebuilt in unit 1b.2."""
    args = sys.argv[1:] if argv is None else argv
    if "--version" in args:
        print(f"annizarr {__version__}")
        return 0
    print("annizarr: CLI under construction", file=sys.stderr)
    return 2
