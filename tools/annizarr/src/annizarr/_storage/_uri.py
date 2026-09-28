from __future__ import annotations

import shutil
from pathlib import Path

from ..errors import StorageError


def prepare_output_path(output_path: Path, overwrite: bool) -> None:
    """Remove an existing output path when overwrite is enabled, else raise."""
    if output_path.exists():
        if not overwrite:
            raise StorageError(
                f"Output path already exists: {output_path}. "
                "Use overwrite=true in config or --overwrite flag."
            )
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()


def is_s3_url(path: str | Path) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def store_name(path: str | Path) -> str:
    """Tail component of a local path or s3:// URL, for commit messages."""
    return str(path).rstrip("/").rsplit("/", 1)[-1]
