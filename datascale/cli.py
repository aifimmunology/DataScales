from __future__ import annotations

import argparse
import sys

from .config import apply_cli_overrides, load_config
from .converter import convert_h5ad_to_zarr
from .validation import ValidationError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datascale")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert non-spatial single-cell h5ad to zarr")
    convert.add_argument("--input", required=True, help="Path to input .h5ad")
    convert.add_argument("--output", required=True, help="Path to output .zarr directory")
    convert.add_argument("--config", required=False, help="Path to YAML/TOML config file")

    convert.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists",
    )
    convert.add_argument(
        "--no-consolidate-metadata",
        action="store_true",
        help="Disable zarr metadata consolidation after write",
    )
    convert.add_argument(
        "--x-storage",
        choices=["auto", "sparse", "dense"],
        help="How to store X/layers/raw.X in output zarr (auto keeps original representation)",
    )

    convert.add_argument("--x-row-chunk", type=int, help="Row chunk size for X")
    convert.add_argument("--x-col-chunk", type=int, help="Column chunk size for X")

    return parser


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "convert":
        parser.error("Unsupported command")

    try:
        config = load_config(args.config)
        config = apply_cli_overrides(
            config,
            overwrite=True if args.overwrite else None,
            consolidate_metadata=False if args.no_consolidate_metadata else None,
            x_storage=args.x_storage,
            x_row_chunk=args.x_row_chunk,
            x_col_chunk=args.x_col_chunk,
        )

        warnings = convert_h5ad_to_zarr(args.input, args.output, config)
    except (FileNotFoundError, ValueError, ValidationError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Conversion complete.")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(
        "Chunks: "
        f"({config.chunks.x_row_chunk}, {config.chunks.x_col_chunk})"
    )
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
