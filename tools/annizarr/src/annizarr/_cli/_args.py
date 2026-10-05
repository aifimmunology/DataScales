from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from annizarr._core._config import ChunkConfig, apply_cli_overrides, load_config

if TYPE_CHECKING:
    from annizarr._core._config import AppConfig


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        help="TOML/YAML config file whose keys mirror these flags (see example_config.toml); "
        "precedence: defaults < file < flags",
    )


def add_overwrite_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--overwrite", action="store_true", default=None, help="replace an existing output")


def add_consolidate_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--consolidate-metadata",
        action="store_true",
        default=None,
        help="consolidate zarr metadata after writing",
    )


def add_ic_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ic", action="store_true", default=None, help="write through an Icechunk repository")


def add_branch_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--branch",
        default=None,
        metavar="B",
        help="Icechunk branch to read/write, created off main if missing (default: main); ignored for plain zarr",
    )


def add_message_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m",
        "--message",
        default=None,
        metavar="MSG",
        help="Icechunk commit message (default: an auto-generated one naming the op); ignored for plain zarr",
    )


def add_cpus_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cpus", type=int, help="parallel band workers (default: all cores)")


def add_chunk_args(parser: argparse.ArgumentParser) -> None:
    defaults = ChunkConfig()
    parser.add_argument("--x-row-chunk", type=int, help=f"row chunk size for X (default: {defaults.x_row_chunk})")
    parser.add_argument(
        "--x-col-chunk", type=int, help=f"column chunk size for dense X (default: {defaults.x_col_chunk})"
    )
    parser.add_argument(
        "--sparse-flat-chunk",
        type=int,
        help=f"flat chunk size for sparse X data/indices (default: {defaults.sparse_flat_chunk})",
    )
    parser.add_argument(
        "--x-shard-factor",
        type=int,
        help=f"pack this many chunks per shard (dense X only) (default: {defaults.x_shard_factor})",
    )


def add_autoshard_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--auto-shard",
        dest="auto_shard",
        action="store_true",
        default=None,
        help="shard the anndata-written elements and the 1-D sparse arrays with zarr's automatic shard shape "
        "(default: off)",
    )


def build_config(args: argparse.Namespace) -> AppConfig:
    # resolves from --config plus whichever override flags the calling subcommand's
    # parser defined; flags absent from args are simply skipped
    cfg = load_config(getattr(args, "config", None))
    return apply_cli_overrides(
        cfg,
        overwrite=getattr(args, "overwrite", None),
        consolidate_metadata=getattr(args, "consolidate_metadata", None),
        x_storage=getattr(args, "x_storage", None),
        x_row_chunk=getattr(args, "x_row_chunk", None),
        x_col_chunk=getattr(args, "x_col_chunk", None),
        sparse_flat_chunk=getattr(args, "sparse_flat_chunk", None),
        x_shard_factor=getattr(args, "x_shard_factor", None),
        auto_shard=getattr(args, "auto_shard", None),
        cpus=getattr(args, "cpus", None),
        lazy=getattr(args, "lazy", None),
        backend=("icechunk" if getattr(args, "ic", False) else None),
        sort_by=getattr(args, "sort_by", None),
        obs_columns=getattr(args, "obs_columns", None),
    )
