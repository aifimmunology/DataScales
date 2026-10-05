from __future__ import annotations

import argparse
import logging
import sys

from annizarr._version import __version__
from annizarr.errors import AnzError

_LOGGER_NAME = "annizarr"


def _build_parser() -> argparse.ArgumentParser:
    from annizarr._cli._convert import add_parser as add_convert_parser
    from annizarr._cli._edit import add_add_expr_parser, add_append_parser, add_rechunk_parser, add_sort_parser

    parser = argparse.ArgumentParser(
        prog="annizarr", description="AnnData zarr stores: convert, edit, and version with Icechunk."
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="warnings only")

    sub = parser.add_subparsers(dest="command")
    add_convert_parser(sub)
    add_add_expr_parser(sub)
    add_rechunk_parser(sub)
    add_sort_parser(sub)
    add_append_parser(sub)
    return parser


def _configure_logging(level: int) -> None:
    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"annizarr {__version__}")
        return 0

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    _configure_logging(level)

    if getattr(args, "command", None) is None:
        parser.print_usage(sys.stderr)
        return 2

    try:
        exit_code: int = args.func(args)
        return exit_code
    except AnzError as exc:
        logging.getLogger(f"{_LOGGER_NAME}._cli").error(f"error: {exc}")
        return 1
