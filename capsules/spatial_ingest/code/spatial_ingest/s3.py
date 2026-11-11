"""A store at an `s3://` prefix is copied to local disk, grown there, and synced back (spec §3S.6):
new and changed files are uploaded (compared by S3 ETag), remote keys no longer present are deleted,
and the root `zarr.json` goes last, so a reader never sees new consolidated metadata before its files."""
from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

PART_SIZE = 8 * 1024 * 1024
ROOT_META = "zarr.json"


class S3Error(RuntimeError):
    """An S3 location that can't be read or written."""


def split(url: str) -> tuple[str, str]:
    bucket, _, prefix = url.removeprefix("s3://").partition("/")
    if not bucket:
        raise S3Error(f"{url} names no bucket; give s3://bucket/prefix.")
    return bucket, prefix.strip("/")


def client():
    import boto3

    return boto3.client("s3")


def transfer_config():
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(multipart_threshold=PART_SIZE, multipart_chunksize=PART_SIZE)


def _key(prefix: str, rel: str) -> str:
    return f"{prefix}/{rel}" if prefix else rel


def remote_files(s3, bucket: str, prefix: str) -> dict[str, str]:
    """Relative path -> ETag of every object under the prefix."""
    out = {}
    base = f"{prefix}/" if prefix else ""
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=base):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):
                out[obj["Key"][len(base):]] = obj["ETag"].strip('"')
    return out


def local_files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def etag(path: Path) -> str:
    """The ETag S3 gives this file when uploaded with `transfer_config`: MD5, or the multipart form."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size < PART_SIZE:
            return hashlib.md5(f.read()).hexdigest()
        digests = [hashlib.md5(chunk).digest() for chunk in iter(lambda: f.read(PART_SIZE), b"")]
    return f"{hashlib.md5(b''.join(digests)).hexdigest()}-{len(digests)}"


def root_meta(url: str) -> dict | None:
    """The store's root zarr.json, read alone (so a store can be judged before it is copied), or
    None when the prefix holds no objects; {} when it holds objects but no root zarr.json."""
    import json

    s3 = client()
    bucket, prefix = split(url)
    try:
        if not s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/" if prefix else "", MaxKeys=2).get("KeyCount"):
            return None
        body = s3.get_object(Bucket=bucket, Key=_key(prefix, ROOT_META))["Body"].read()
    except s3.exceptions.NoSuchKey:
        return {}
    except Exception as exc:
        raise S3Error(f"Can't read {url}: {exc}") from None
    return json.loads(body)


def stat(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def download(url: str, local: Path, workers: int) -> dict[str, tuple[int, int]]:
    """Copies every object under the prefix to `local`; returns each file's size and mtime, so the
    sync can skip files the run didn't touch."""
    s3 = client()
    bucket, prefix = split(url)
    files = remote_files(s3, bucket, prefix)
    cfg = transfer_config()

    def get(rel: str) -> None:
        target = local / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, _key(prefix, rel), str(target), Config=cfg)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(get, files))
    logger.info("Copied %d files from %s to %s", len(files), url, local)
    return {rel: stat(local / rel) for rel in files}


def sync(local: Path, url: str, workers: int, downloaded: dict[str, tuple[int, int]] | None = None) -> dict[str, int]:
    """Makes the prefix match `local`: uploads new and changed files, deletes keys gone locally, and
    uploads the root zarr.json last. Files unchanged since `download` (same size and mtime) are
    trusted without hashing; the rest are compared by ETag."""
    s3 = client()
    bucket, prefix = split(url)
    remote = remote_files(s3, bucket, prefix)
    files = local_files(local)
    downloaded = downloaded or {}

    def unchanged(rel: str) -> bool:
        if rel not in remote:
            return False
        if downloaded.get(rel) == stat(local / rel):
            return True
        return remote[rel] == etag(local / rel)

    changed = [rel for rel in files if not unchanged(rel)]
    gone = sorted(set(remote) - set(files))
    cfg = transfer_config()

    def put(rel: str) -> None:
        s3.upload_file(str(local / rel), bucket, _key(prefix, rel), Config=cfg)

    first = [rel for rel in changed if rel != ROOT_META]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(put, first))
    for start in range(0, len(gone), 1000):
        batch = [{"Key": _key(prefix, rel)} for rel in gone[start:start + 1000] if rel != ROOT_META]
        if batch:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
    if ROOT_META in changed:
        put(ROOT_META)
    result = {"uploaded": len(changed), "deleted": len(gone), "unchanged": len(files) - len(changed)}
    logger.info("Synced %s to %s: %s", local, url, result)
    return result
