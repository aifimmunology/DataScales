from __future__ import annotations

from pathlib import Path

from ..errors import StorageError
from ._uri import is_s3_url


def icechunk_storage(path: str | Path):
    """Icechunk Storage for a local path or an ``s3://bucket/prefix`` URL.

    S3 credentials come from the environment (AWS_* vars / default chain)."""
    import icechunk

    if is_s3_url(path):
        bucket, _, prefix = str(path)[len("s3://"):].partition("/")
        if not bucket or not prefix.strip("/"):
            raise StorageError(f"Invalid S3 URL '{path}': expected s3://bucket/prefix.")
        return icechunk.s3_storage(bucket=bucket, prefix=prefix.strip("/"))
    return icechunk.local_filesystem_storage(str(path))


def is_icechunk_repo(path: Path) -> bool:
    """Local icechunk repos carry repo/ + snapshots/ and no zarr.json at the root."""
    return (
        path.is_dir()
        and not (path / "zarr.json").exists()
        and (path / "snapshots").is_dir()
        and (path / "repo").exists()
    )
