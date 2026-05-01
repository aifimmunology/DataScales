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
        help="Row chunk size for dense X; auto-capped at 64 MB per chunk (default: 2048)",
    )
    optional.add_argument(
        "--x-col-chunk",
        type=int,
        help="Column chunk size for dense X (default: 2048)",
    )
    optional.add_argument(
        "--sparse-flat-chunk",
        type=int,
        help="Flat chunk size for sparse arrays (data/indices/indptr); tune to median nnz per row (default: 4096)",
    )
    optional.add_argument(
        "--n-dense-workers",
        type=int,
        help="Threads for parallel dense chunk writes (default: 1). Raise on HPC. No effect with --backed.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datascale")
    #dest is what field to grab froms args object. EG args.command will be the below subparser name
    subparsers = parser.add_subparsers(dest="command", required=True)

    h5ad = subparsers.add_parser(
        "convert-h5ad", help="Convert non-spatial single-cell .h5ad to zarr"
    )
    h5ad_required = h5ad.add_argument_group("required arguments")
    h5ad_required.add_argument("--input", required=True, help="Path to input .h5ad file")
    h5ad_optional = h5ad.add_argument_group("optional arguments")
    h5ad_optional.add_argument(
        "--backed",
        action="store_true",
        help="Stream X from disk without loading into RAM. Recommended for large files. Errors if backed load fails.",
    )
    _add_common_args(h5ad_required, h5ad_optional)

    h5 = subparsers.add_parser(
        "convert-10x-h5", help="Convert 10x Genomics Cell Ranger HDF5 (.h5) to zarr"
    )
    h5_required = h5.add_argument_group("required arguments")
    h5_required.add_argument("--input", required=True, help="Path to 10x Cell Ranger .h5 file")
    h5_optional = h5.add_argument_group("optional arguments")
    _add_common_args(h5_required, h5_optional)

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
            n_dense_workers=getattr(args, "n_dense_workers", None),
            backed=True if getattr(args, "backed", False) else None,
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
    print(f"Backed load: {config.io.backed}")
    if config.io.x_storage == "dense" and not config.io.backed:
        print(f"Dense workers: {config.chunks.n_dense_workers}")

    if warnings:
        print("Warnings:")
        for msg in warnings:
            print(f"- {msg}")

    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

