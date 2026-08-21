from __future__ import annotations

import argparse
import sys

from .config import apply_cli_overrides, load_config
from .ops import (
    add_expr_layer,
    append_cells,
    convert_10x_h5_to_zarr,
    convert_h5ad_to_zarr,
    convert_h5ads_to_zarr,
    rechunk_store,
    sort_store,
)

_CONVERTERS = {
    "convert": convert_h5ad_to_zarr,
    "convert-10x": convert_10x_h5_to_zarr,
    "concat": convert_h5ads_to_zarr,
}


def _add_common_args(
    required: argparse._ArgumentGroup,
    optional: argparse._ArgumentGroup,
) -> None:
    required.add_argument("--output", required=True, help="Path to output .zarr directory")

    optional.add_argument("--config", help="Path to YAML/TOML config file")
    optional.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists",
    )
    optional.add_argument(
        "--consolidate-metadata",
        action="store_true",
        help="Write consolidated zarr metadata after write; useful for remote stores (S3, GCS)",
    )
    optional.add_argument(
        "--x-storage",
        choices=["sparse-csr", "sparse-csc", "dense"],
        help="Output format for X/layers/raw.X (default: sparse-csr)",
    )
    optional.add_argument(
        "--x-row-chunk",
        type=int,
        help="Row chunk size for dense X; honored exactly via 2D tiling (default: 2048)",
    )
    optional.add_argument(
        "--x-col-chunk",
        type=int,
        help="Column chunk size for dense X (default: 2048)",
    )
    optional.add_argument(
        "--sparse-flat-chunk",
        type=int,
        help="Flat chunk size for sparse arrays (data/indices/indptr); tune to median nnz per row (default: 1000000)",
    )
    optional.add_argument(
        "--x-shard-factor",
        type=int,
        help="Pack dense X inner chunks into shards of (x-row-chunk, x-col-chunk) * factor. "
             "1 (default) = no sharding. Use >1 with small chunks to keep read granularity "
             "fine while cutting file/object count (dense X only; sparse output ignores it).",
    )
    optional.add_argument(
        "--cpus",
        type=int,
        help="Workers for parallel matrix chunk writes (default: 1). Raise on HPC. In-memory uses threads; backed uses processes (most effective for dense output).",
    )
    optional.add_argument(
        "--icechunk",
        action="store_true",
        help="Write the output through an Icechunk repository (transactional, versioned) "
             "instead of a plain zarr directory. Commits to the 'main' branch. "
             "Local storage; eager input only (not --backed).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zarrsmith")
    #dest is what field to grab froms args object. EG args.command will be the below subparser name
    subparsers = parser.add_subparsers(dest="command", required=True)

    h5ad = subparsers.add_parser(
        "convert", help="Convert non-spatial single-cell .h5ad to zarr"
    )
    h5ad_required = h5ad.add_argument_group("required arguments")
    h5ad_required.add_argument("--input", required=True, help="Path to input .h5ad file")
    h5ad_optional = h5ad.add_argument_group("optional arguments")
    h5ad_optional.add_argument(
        "--backed",
        action="store_true",
        help="Stream X from disk without loading into RAM. Recommended for large files. Errors if backed load fails.",
    )
    h5ad_optional.add_argument(
        "--sort-by",
        nargs="+",
        metavar="OBS_COLUMN",
        help="Sort + partition rows by these obs column(s), primary key first "
             "(e.g. --sort-by AIFI_L1 batch_id). Physically sorts rows so each distinct key "
             "tuple is a contiguous block; the output is a plain sorted AnnData (no tool "
             "index) — derive ranges from the sorted obs column(s) and slice X[start:end] with "
             "stock anndata/zarr. Eager load handles sparse-csr or dense; with --backed the "
             "sort is streamed (memory-bounded) for sparse-csr only, and layers/raw/obsp must "
             "be absent.",
    )
    _add_common_args(h5ad_required, h5ad_optional)

    h5 = subparsers.add_parser(
        "convert-10x", help="Convert 10x Genomics Cell Ranger HDF5 (.h5) to zarr"
    )
    h5_required = h5.add_argument_group("required arguments")
    h5_required.add_argument("--input", required=True, help="Path to 10x Cell Ranger .h5 file")
    h5_optional = h5.add_argument_group("optional arguments")
    _add_common_args(h5_required, h5_optional)

    concat = subparsers.add_parser(
        "concat",
        help="Concatenate multiple .h5ad files along obs (rows) into one zarr. "
             "Requires identical var (genes) and obs schema across files.",
    )
    concat_required = concat.add_argument_group("required arguments")
    concat_required.add_argument(
        "--inputs", nargs="+", required=True,
        help="Two or more input .h5ad file paths",
    )
    concat_optional = concat.add_argument_group("optional arguments")
    concat_optional.add_argument(
        "--backed", action="store_true",
        help="Stream X from disk without loading into RAM. Recommended for large files.",
    )
    concat_optional.add_argument(
        "--obs-columns", nargs="+", metavar="COL",
        help="obs columns to keep and join on (e.g. --obs-columns cell_type donor). "
             "Default (omitted): require an identical obs schema across all inputs. "
             "When given: each input must contain these columns; obs is projected to "
             "exactly these (in this order) and all other columns are dropped.",
    )
    _add_common_args(concat_required, concat_optional)

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
                        help="Drop obsp graphs (invalidated by new cells) instead of refusing")
    append.add_argument("--refresh-expr", action="store_true",
                        help="Re-derive layers/gexp after the append instead of refusing")
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
        drop_obsp=args.drop_obsp, refresh_expr=args.refresh_expr,
    ), args.store


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in ("add-expr", "rechunk", "sort", "append"):
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

    converter = _CONVERTERS.get(args.command)
    if converter is None:
        parser.error(f"Unsupported command: {args.command}")

    try:
        config = load_config(args.config)
        config = apply_cli_overrides(
            config,
            overwrite=True if args.overwrite else None,
            consolidate_metadata=True if args.consolidate_metadata else None,
            x_storage=args.x_storage,
            x_row_chunk=args.x_row_chunk,
            x_col_chunk=args.x_col_chunk,
            sparse_flat_chunk=args.sparse_flat_chunk,
            x_shard_factor=getattr(args, "x_shard_factor", None),
            cpus=getattr(args, "cpus", None),
            backed=True if getattr(args, "backed", False) else None,
            backend="icechunk" if getattr(args, "icechunk", False) else None,
            sort_by=getattr(args, "sort_by", None),
            obs_columns=getattr(args, "obs_columns", None),
        )

        if args.command == "concat":
            warnings = converter(args.inputs, args.output, config)
            input_label = ", ".join(args.inputs)
        else:
            warnings = converter(args.input, args.output, config)
            input_label = args.input
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Conversion complete.")
    print(f"Input: {input_label}")
    print(f"Output: {args.output}")
    print(f"Chunks: ({config.chunks.x_row_chunk}, {config.chunks.x_col_chunk})")
    print(f"X storage mode: {config.io.x_storage}")
    print(f"Storage backend: {config.io.backend}")
    if config.grouping.enabled:
        print(f"Sorted by: {list(config.grouping.sort_by)}")
    if args.command == "concat" and config.concat.obs_columns:
        print(f"obs columns kept: {list(config.concat.obs_columns)}")
    print(f"Backed load: {config.io.backed}")
    print(f"CPUs: {config.chunks.cpus}")

    if warnings:
        print("Warnings:")
        for msg in warnings:
            print(f"- {msg}")

    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

