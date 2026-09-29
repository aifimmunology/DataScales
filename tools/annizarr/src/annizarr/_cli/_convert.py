from __future__ import annotations

import argparse
import logging

from annizarr._cli._args import (
    add_chunk_args,
    add_config_arg,
    add_consolidate_arg,
    add_cpus_arg,
    add_ic_arg,
    add_overwrite_arg,
    build_config,
)

_LOG = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("convert", help="Convert h5ad/10x input(s) into an AnnData zarr (or Icechunk) store")
    p.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="input file(s); two or more h5ad inputs are concatenated",
    )
    p.add_argument("-o", "--output", required=True, help="output store path or URI")
    p.add_argument(
        "--from",
        dest="from_",
        choices=("h5ad", "10x"),
        help="override content-based format detection",
    )
    p.add_argument(
        "--backed",
        action="store_true",
        default=None,
        help="stream h5ad input instead of loading it eagerly",
    )
    p.add_argument("--x-storage", choices=("csr", "csc", "dense"), help="output X layout")
    add_cpus_arg(p)
    add_chunk_args(p)
    p.add_argument(
        "--sort-by",
        dest="sort_by",
        nargs="+",
        metavar="COL",
        help="physically sort rows by these obs column(s), primary key first",
    )
    p.add_argument(
        "--obs-columns",
        dest="obs_columns",
        nargs="+",
        metavar="COL",
        help="project obs to this subset before concatenating (two or more inputs only)",
    )
    add_consolidate_arg(p)
    add_overwrite_arg(p)
    add_ic_arg(p)
    add_config_arg(p)
    p.set_defaults(func=_run, _parser=p)


def _run(args: argparse.Namespace) -> int:
    from annizarr._ops import convert

    if args.obs_columns and len(args.inputs) < 2:
        args._parser.error("--obs-columns requires at least two inputs")

    cfg = build_config(args)
    result = convert(args.inputs, output=args.output, cfg=cfg, fmt=args.from_)
    _LOG.info(f"wrote {result.path} ({result.n_obs} x {result.n_vars})")
    if result.snapshot_id is not None:
        _LOG.info(f"icechunk snapshot {result.snapshot_id}")
    return 0
