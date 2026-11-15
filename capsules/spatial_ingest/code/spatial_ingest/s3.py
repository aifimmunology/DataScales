"""A store at an `s3://` prefix is copied to local disk, grown there, and synced back (spec §3S.6).

One run per store: `acquire_lock` creates `<prefix>/.spatial_ingest_lock` with a conditional put (it
fails when the object exists), refreshes its heartbeat while the run lasts and takes over a lock
whose heartbeat is older than the stale limit; `release_lock` deletes it. As a second guard, `sync`
refuses when the remote root `zarr.json` is no longer the one `download` copied (another writer).

What a reader sees during a sync: readers open the store through the root `zarr.json` (its
consolidated metadata), which is uploaded after every other file, so until it flips they read the
store as it was. `sync` uploads in this order: new data files, changed data files, other metadata
files (deepest first), the root, then deletes. Data files a run changes are only ever extended (the
joint table keeps one chunk length per array for life, see joint.INDPTR_CHUNK), so the old metadata
still reads them correctly. A sync killed at any point leaves the old store readable; the next run
downloads it and its repair removes the uncommitted rest.

The first sync of a new store writes `.spatial_ingest_creating` first and removes it after the root:
a prefix holding only that marker's partial upload is cleared and created again (`root_meta`)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PART_SIZE = 8 * 1024 * 1024
PART_THREADS = 4                    # threads per multipart transfer (boto3 default 10)
ROOT_META = "zarr.json"
POOL_MIN = 32
CONTROL = ".spatial_ingest_"                  # control objects at the prefix root, never part of the store
LOCK = ".spatial_ingest_lock"
CREATING = ".spatial_ingest_creating"
ROOT_ETAG = ".root_etag"                      # key in download()'s result: the root zarr.json's ETag
LOCK_STALE_MINUTES = 60
METADATA = ("zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata", "zmetadata")

_HELD: dict[str, "S3Lock"] = {}               # url -> this process's lock on it


class S3Error(RuntimeError):
    """An S3 location that can't be read or written."""


def split(url: str) -> tuple[str, str]:
    bucket, _, prefix = url.removeprefix("s3://").partition("/")
    if not bucket:
        raise S3Error(f"{url} names no bucket; give s3://bucket/prefix.")
    return bucket, prefix.strip("/")


def client():
    """An S3 client whose connection pool fits the parallel transfers (S3_WORKERS files at once, each split
    into up to PART_THREADS parts in flight), so urllib3 never drops connections with "Connection pool is
    full" warnings."""
    import boto3
    from botocore.config import Config

    try:
        workers = int(os.environ.get("S3_WORKERS", "16"))
    except ValueError:
        workers = 16
    return boto3.client("s3", config=Config(max_pool_connections=max(POOL_MIN, PART_THREADS * workers)))


def transfer_config():
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(multipart_threshold=PART_SIZE, multipart_chunksize=PART_SIZE,
                          max_concurrency=PART_THREADS)


def _key(prefix: str, rel: str) -> str:
    return f"{prefix}/{rel}" if prefix else rel


def _code(exc: Exception) -> str:
    return getattr(exc, "response", {}).get("Error", {}).get("Code", "")


def remote_files(s3, bucket: str, prefix: str, control: bool = False) -> dict[str, str]:
    """Relative path -> ETag of every object under the prefix (control objects only with `control`)."""
    out = {}
    base = f"{prefix}/" if prefix else ""
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=base):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(base):]
            if not obj["Key"].endswith("/") and (control or not rel.startswith(CONTROL)):
                out[rel] = obj["ETag"].strip('"')
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


# --------------------------------------------------------------------------- lock

@dataclass
class S3Lock:
    """This run's lock on a store prefix; its heartbeat thread keeps the lock object fresh."""
    url: str
    etag: str
    body: dict
    stale_minutes: float
    lost: bool = False
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    @property
    def location(self) -> tuple[str, str]:
        bucket, prefix = split(self.url)
        return bucket, _key(prefix, LOCK)


def _stale_minutes(stale_minutes: float | None) -> float:
    if stale_minutes is not None:
        return float(stale_minutes)
    try:
        return float(os.environ.get("S3_LOCK_STALE_MINUTES", LOCK_STALE_MINUTES))
    except ValueError:
        return float(LOCK_STALE_MINUTES)


def _put_lock(s3, bucket: str, key: str, body: dict, **condition) -> str:
    return s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(body).encode(), **condition)["ETag"]


def acquire_lock(url: str, stale_minutes: float | None = None) -> S3Lock:
    """Takes the store's lock for this run, or raises S3Error naming the run that holds it. A lock whose
    heartbeat is older than `stale_minutes` (default S3_LOCK_STALE_MINUTES or 60) is taken over with a
    WARNING (its run is gone). The heartbeat is refreshed in the background until `release_lock`."""
    s3 = client()
    bucket, prefix = split(url)
    key = _key(prefix, LOCK)
    stale = _stale_minutes(stale_minutes)
    now = time.time()
    body = {"host": socket.gethostname(), "pid": os.getpid(), "started": now, "heartbeat": now}
    try:
        tag = _put_lock(s3, bucket, key, body, IfNoneMatch="*")
    except Exception as exc:
        if _code(exc) not in ("PreconditionFailed", "ConditionalRequestConflict"):
            raise S3Error(f"Can't lock {url}: {exc}") from None
        tag = _take_over(s3, bucket, key, url, body, stale)
    lock = S3Lock(url, tag, body, stale)
    lock.thread = threading.Thread(target=_heartbeat, args=(lock,), daemon=True, name="s3-lock-heartbeat")
    lock.thread.start()
    _HELD[url] = lock
    return lock


def _dead_here(info: dict) -> bool:
    """The lock's run was on this host and its process is gone (killed), so the lock can be taken at once."""
    if info.get("host") != socket.gethostname():
        return False
    try:
        os.kill(int(info.get("pid", -1)), 0)
    except ProcessLookupError:
        return True
    except (PermissionError, ValueError, OverflowError):
        return False
    return False


def _take_over(s3, bucket: str, key: str, url: str, body: dict, stale: float) -> str:
    try:
        held = s3.get_object(Bucket=bucket, Key=key)
        info, old = json.loads(held["Body"].read() or b"{}"), held["ETag"]
    except Exception:
        info, old = {}, None
    age = (time.time() - float(info.get("heartbeat", 0))) / 60
    if old is not None and age < stale and not _dead_here(info):
        started = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(float(info.get("started", 0))))
        raise S3Error(f"Another run is adding to {url} (host {info.get('host', '?')}, pid {info.get('pid', '?')}, "
                      f"started {started}, last heartbeat {age:.0f} min ago); wait for it to finish. A lock without "
                      f"heartbeat for {stale:.0f} min is taken over.")
    why = "is no longer running on this host" if _dead_here(info) else f"sent no heartbeat for {age:.0f} min"
    logger.warning("Taking over the lock of %s: its run (host %s, pid %s) %s", url, info.get("host", "?"),
                   info.get("pid", "?"), why)
    try:
        return _put_lock(s3, bucket, key, body, **({"IfMatch": old} if old else {"IfNoneMatch": "*"}))
    except Exception as exc:
        raise S3Error(f"Another run took the lock of {url} at the same time; run again later ({_code(exc) or exc}).") \
            from None


def _heartbeat(lock: S3Lock) -> None:
    """Refreshes the lock's heartbeat (conditional on it still being ours) until stopped."""
    interval = max(5.0, lock.stale_minutes * 60 / 4)
    s3 = client()
    bucket, key = lock.location
    while not lock.stop.wait(interval):
        lock.body["heartbeat"] = time.time()
        try:
            lock.etag = _put_lock(s3, bucket, key, lock.body, IfMatch=lock.etag)
        except Exception as exc:
            lock.lost = True
            logger.warning("Lost the lock of %s (%s); this run won't sync to it", lock.url, _code(exc) or exc)
            return


def release_lock(lock: S3Lock | None) -> None:
    """Stops the heartbeat and deletes the lock object if it is still this run's."""
    if lock is None:
        return
    lock.stop.set()
    if lock.thread is not None:
        lock.thread.join(timeout=30)
    _HELD.pop(lock.url, None)
    if lock.lost:
        return
    s3 = client()
    bucket, key = lock.location
    try:
        s3.delete_object(Bucket=bucket, Key=key, IfMatch=lock.etag)
    except Exception as exc:
        if _code(exc) in ("PreconditionFailed", "NoSuchKey", "404"):
            return
        try:
            s3.delete_object(Bucket=bucket, Key=key)       # endpoints without conditional delete
        except Exception:
            logger.warning("Couldn't remove the lock of %s; it expires after %.0f min", lock.url, lock.stale_minutes)


def _check_lock(url: str) -> None:
    lock = _HELD.get(url)
    if lock is not None and lock.lost:
        raise S3Error(f"This run lost its lock on {url} (another run took it over); nothing was synced. Run again.")


# --------------------------------------------------------------------------- store at the prefix

def _root_etag(s3, bucket: str, prefix: str) -> str | None:
    try:
        return s3.head_object(Bucket=bucket, Key=_key(prefix, ROOT_META))["ETag"].strip('"')
    except Exception as exc:
        if _code(exc) in ("404", "NoSuchKey", "NotFound"):
            return None
        raise S3Error(f"Can't read {_key(prefix, ROOT_META)}: {exc}") from None


def _delete(s3, bucket: str, prefix: str, rels: list[str]) -> None:
    for start in range(0, len(rels), 1000):
        batch = [{"Key": _key(prefix, rel)} for rel in rels[start:start + 1000]]
        if batch:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})


def root_meta(url: str) -> dict | None:
    """The store's root zarr.json, read alone (so a store can be judged before it is copied); None when
    the prefix holds no store objects; {} when it holds objects but no root zarr.json. A prefix holding
    only the partial upload of a killed first sync (its `.spatial_ingest_creating` marker present) is
    cleared (only the keys under this prefix) and reported as None, unless another live run holds the lock."""
    s3 = client()
    bucket, prefix = split(url)
    try:
        files = remote_files(s3, bucket, prefix, control=True)
        store = [rel for rel in files if not rel.startswith(CONTROL)]
        if not store:
            return None
        if ROOT_META in files:
            return json.loads(s3.get_object(Bucket=bucket, Key=_key(prefix, ROOT_META))["Body"].read())
    except Exception as exc:
        raise S3Error(f"Can't read {url}: {exc}") from None
    if CREATING not in files:
        return {}
    if LOCK in files and url not in _HELD:
        try:
            info = json.loads(s3.get_object(Bucket=bucket, Key=_key(prefix, LOCK))["Body"].read() or b"{}")
        except Exception:
            info = {}
        if (time.time() - float(info.get("heartbeat", 0))) / 60 < _stale_minutes(None):
            raise S3Error(f"Another run is creating the store at {url}; wait for it to finish.")
    logger.warning("%s holds a store whose creation was killed (%d files, no root zarr.json); cleared to create "
                   "it again", url, len(store))
    _delete(s3, bucket, prefix, store + [CREATING])
    return None


def stat(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def download(url: str, local: Path, workers: int) -> dict:
    """Copies every store object under the prefix to `local`; returns each file's size and mtime (so the
    sync can skip files the run didn't touch) and, under ROOT_ETAG, the root zarr.json's ETag (so the
    sync can tell whether another writer changed the store meanwhile)."""
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
    out: dict = {rel: stat(local / rel) for rel in files}
    out[ROOT_ETAG] = files.get(ROOT_META)
    return out


def _phase(name: str) -> None:
    """Called after each sync phase (a hook for fault-injection tests)."""
    logger.debug("sync phase done: %s", name)


def _is_meta(rel: str) -> bool:
    return rel.rsplit("/", 1)[-1] in METADATA


def sync(local: Path, url: str, workers: int, downloaded: dict | None = None) -> dict[str, int]:
    """Makes the prefix match `local` in an order that keeps it readable at every step (module doc):
    new data files, changed data files, metadata files deepest first, the root zarr.json, deletes.
    Refuses when the remote root zarr.json isn't the one downloaded (another run wrote the store) or
    this run lost its lock. Files unchanged since `download` (same size and mtime) are trusted without
    hashing; the rest are compared by ETag."""
    s3 = client()
    bucket, prefix = split(url)
    _check_lock(url)
    downloaded = downloaded or {}
    expected = downloaded.get(ROOT_ETAG)
    current = _root_etag(s3, bucket, prefix)
    if current != expected:
        raise S3Error(f"The store at {url} changed during this run (another run wrote to it); nothing was synced. "
                      "Run again: the samples will be added to the current store.")
    remote = remote_files(s3, bucket, prefix)
    files = local_files(local)
    creating = current is None

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

    def upload(rels: list[str]) -> None:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(put, rels))

    if creating:
        s3.put_object(Bucket=bucket, Key=_key(prefix, CREATING), Body=b"")
    data = [rel for rel in changed if not _is_meta(rel)]
    upload([rel for rel in data if rel not in remote])
    _phase("new data")
    upload([rel for rel in data if rel in remote])
    _phase("changed data")
    meta = [rel for rel in changed if _is_meta(rel) and rel != ROOT_META]
    for depth in sorted({rel.count("/") for rel in meta}, reverse=True):
        upload([rel for rel in meta if rel.count("/") == depth])
    _phase("metadata")
    _check_lock(url)
    if ROOT_META in changed:
        put(ROOT_META)
    _phase("root")
    _delete(s3, bucket, prefix, [rel for rel in gone if rel != ROOT_META])
    if creating:
        s3.delete_object(Bucket=bucket, Key=_key(prefix, CREATING))
    _phase("deletes")
    result = {"uploaded": len(changed), "deleted": len(gone), "unchanged": len(files) - len(changed)}
    logger.info("Synced %s to %s: %s", local, url, result)
    return result
