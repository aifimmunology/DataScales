from __future__ import annotations

import argparse
import sys

from .config import apply_cli_overrides, load_config
from .converter import convert_10x_h5_to_zarr, convert_h5ad_to_zarr
from .validation import ValidationError

_CONVERTERS = {
    "convert-h5ad": convert_h5ad_to_zarr,
    "convert-10x-h5": convert_10x_h5_to_zarr,
}


def _add_common_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--output", required=True, help="Path to output .zarr directory")
    sub.add_argument("--config", required=False, help="Path to YAML/TOML config file")
    sub.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists",
    )
    sub.add_argument(
        "--consolidate-metadata",
        action="store_true",
        help="Enable zarr metadata consolidation after write",
    )
    sub.add_argument(
        "--x-storage",
        choices=["sparse-csr", "sparse-csc", "dense"],
        help="How to store X/layers/raw.X in output zarr (default: sparse-csr)",
    )
    sub.add_argument("--x-row-chunk", type=int, help="Row chunk size for X")
    sub.add_argument("--x-col-chunk", type=int, help="Column chunk size for X")
    sub.add_argument(
        "--sparse-flat-chunk",
        type=int,
        help="Chunk size for sparse flat arrays (data/indices/indptr); tune to median nnz per cell",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datascale")
    #dest is what field to grab froms args object. EG args.command will be the below subparser name
    subparsers = parser.add_subparsers(dest="command", required=True)

    h5ad = subparsers.add_parser(
        "convert-h5ad", help="Convert non-spatial single-cell .h5ad to zarr"
    )
    h5ad.add_argument("--input", required=True, help="Path to input .h5ad file")
    _add_common_args(h5ad)

    h5 = subparsers.add_parser(
        "convert-10x-h5", help="Convert 10x Genomics Cell Ranger HDF5 (.h5) to zarr"
    )
    h5.add_argument("--input", required=True, help="Path to 10x Cell Ranger .h5 file")
    _add_common_args(h5)

    return parser


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

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
        )

        warnings = converter(args.input, args.output, config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Conversion complete.")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Chunks: ({config.chunks.x_row_chunk}, {config.chunks.x_col_chunk})")
    print(f"X storage mode: {config.io.x_storage}")

    if warnings:
        print("Warnings:")
        for msg in warnings:
            print(f"- {msg}")

    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

