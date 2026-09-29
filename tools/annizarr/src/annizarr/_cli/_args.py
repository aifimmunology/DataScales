from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from annizarr._config import apply_cli_overrides, load_config

if TYPE_CHECKING:
    from annizarr._config import AppConfig


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="TOML/YAML config file")


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
        help="Icechunk branch to read/write (default: current HEAD, falling back to 'main'); ignored for plain zarr",
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
    parser.add_argument("--cpus", type=int, help="parallel workers for matrix chunk writes")


def add_chunk_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--x-row-chunk", type=int, help="row chunk size for X")
    parser.add_argument("--x-col-chunk", type=int, help="column chunk size for dense X")
    parser.add_argument("--sparse-flat-chunk", type=int, help="flat chunk size for sparse X data/indices")
    parser.add_argument("--x-shard-factor", type=int, help="pack this many chunks per shard (dense X only)")


def add_autoshard_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--auto-shard",
        dest="auto_shard",
        action="store_true",
        default=False,
        help="shard the anndata-written elements and the 1-D sparse arrays with zarr's "
        "automatic shard shape (default: unsharded)",
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
        backed=getattr(args, "backed", None),
        backend=("icechunk" if getattr(args, "ic", False) else None),
        sort_by=getattr(args, "sort_by", None),
        obs_columns=getattr(args, "obs_columns", None),
    )
