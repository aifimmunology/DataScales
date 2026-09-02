from __future__ import annotations

import argparse
import sys

from convert_to_zarr.config import apply_cli_overrides, load_config

from .append import append_cells
from .expr import add_expr_layer
from .rechunk import rechunk_store
from .sort import sort_store


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zarrsmith")
    subparsers = parser.add_subparsers(dest="command", required=True)

    addexpr = subparsers.add_parser(
        "add-expr",
        help="Add a log-normalized expression layer (layers/gexp) derived from CSR X, in place",
    )
    addexpr.add_argument("--store", required=True, help="Existing AnnData zarr store (CSR X)")
    addexpr.add_argument("--format", choices=["csc", "dense", "csr"], default="csc",
                         help="Layer layout; csc (default) is the gene-query layout")
    addexpr.add_argument("--layer", default="gexp", help="Layer name (default: gexp)")
    addexpr.add_argument("--chunk-elems", type=int, default=1_000_000,
                         help="Elements per chunk (1-D sparse arrays; dense derives cols from it)")
    addexpr.add_argument("--target-sum", type=float, default=1e4,
                         help="normalize_total target before log1p (default: 1e4)")
    addexpr.add_argument("--overwrite", action="store_true", help="Replace the layer if it exists")
    addexpr.add_argument("--icechunk", action="store_true", help="Store is an Icechunk repository")
    addexpr.add_argument("--config", help="Path to YAML/TOML config file")

    rechunk = subparsers.add_parser(
        "rechunk", help="Rewrite a matrix with new chunking into a new store"
    )
    rechunk.add_argument("--store", required=True, help="Input AnnData zarr store")
    rechunk.add_argument("--output", required=True, help="Path to output .zarr directory")
    rechunk.add_argument("--array", default="X", help="Matrix to rechunk: X, layers/<name>, raw/X")
    for flag, kw in (
        ("--x-row-chunk", dict(type=int, help="New row chunk for a dense target")),
        ("--x-col-chunk", dict(type=int, help="New column chunk for a dense target")),
        ("--x-shard-factor", dict(type=int, help="Shard factor for a dense target")),
        ("--sparse-flat-chunk", dict(type=int, help="New flat chunk for a sparse target")),
        ("--cpus", dict(type=int, help="Threads for the tile copy")),
    ):
        rechunk.add_argument(flag, **kw)
    rechunk.add_argument("--overwrite", action="store_true")
    rechunk.add_argument("--consolidate-metadata", action="store_true")
    rechunk.add_argument("--icechunk", action="store_true", help="Write output through Icechunk")
    rechunk.add_argument("--config", help="Path to YAML/TOML config file")

    sort = subparsers.add_parser(
        "sort", help="Physically sort an existing store by obs column(s) into a new store"
    )
    sort.add_argument("--store", required=True, help="Input AnnData zarr store (CSR X)")
    sort.add_argument("--output", required=True, help="Path to output .zarr directory")
    sort.add_argument("--by", nargs="+", required=True, metavar="OBS_COLUMN",
                      help="Sort keys, primary first; each key tuple becomes a contiguous block")
    sort.add_argument("--cpus", type=int)
    sort.add_argument("--overwrite", action="store_true")
    sort.add_argument("--consolidate-metadata", action="store_true")
    sort.add_argument("--icechunk", action="store_true", help="Write output through Icechunk")
    sort.add_argument("--config", help="Path to YAML/TOML config file")

    append = subparsers.add_parser(
        "append", help="Append the cells of another zarr store onto this one, in place"
    )
    append.add_argument("--store", required=True, help="Store to extend (CSR X)")
    append.add_argument("--cells", required=True, help="Zarr store with the cells to append")
    append.add_argument("--drop-obsp", action="store_true",
                        help="Pre-approve dropping obsp graphs (invalidated by new cells)")
    append.add_argument("--refresh-expr", action="store_true",
                        help="Pre-approve re-deriving layers/gexp after the append")
    append.add_argument("--yes", action="store_true",
                        help="Approve the printed loss plan non-interactively")
    append.add_argument("--icechunk", action="store_true", help="Store is an Icechunk repository")
    append.add_argument("--config", help="Path to YAML/TOML config file")

    return parser


def _run_store_op(args) -> tuple[list[str], str]:
    config = load_config(getattr(args, "config", None))
    config = apply_cli_overrides(
        config,
        overwrite=True if getattr(args, "overwrite", False) else None,
        consolidate_metadata=True if getattr(args, "consolidate_metadata", False) else None,
        x_row_chunk=getattr(args, "x_row_chunk", None),
        x_col_chunk=getattr(args, "x_col_chunk", None),
        sparse_flat_chunk=getattr(args, "sparse_flat_chunk", None),
        x_shard_factor=getattr(args, "x_shard_factor", None),
        cpus=getattr(args, "cpus", None),
        backend="icechunk" if getattr(args, "icechunk", False) else None,
        sort_by=getattr(args, "by", None),
    )
    if args.command == "add-expr":
        return add_expr_layer(
            args.store, config, fmt=args.format, layer=args.layer,
            chunk_elems=args.chunk_elems, target_sum=args.target_sum,
        ), args.store
    if args.command == "rechunk":
        return rechunk_store(args.store, args.output, config, array=args.array), args.output
    if args.command == "sort":
        return sort_store(args.store, args.output, config), args.output
    return append_cells(
        args.store, args.cells, config,
        drop_obsp=args.drop_obsp, refresh_expr=args.refresh_expr, assume_yes=args.yes,
    ), args.store


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        warnings, target = _run_store_op(args)
    except KeyError as exc:
        print(f"ERROR: store is missing element {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{args.command} complete.")
    print(f"Store: {target}")
    if warnings:
        print("Warnings:")
        for msg in warnings:
            print(f"- {msg}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
